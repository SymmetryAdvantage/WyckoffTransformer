"""Tests for the de novo ranking protocol."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pandas as pd
import pytest

from wyckoff_transformer.cli.protocol import (
    PYXTAL_COLUMNS,
    RELAXATION_COLUMNS,
    RowLog,
    Timeout,
    LOGM_ROUNDOFF,
    _init_relax_worker,
    _pin_visible_device,
    _quiet_logm_roundoff,
    aggregate_structures,
    claim_device,
    build_parser,
    resolve_devices,
    time_limit,
)
from wyckoff_transformer.cryspr.relaxer import (
    RATTLE_ACCEPT_EV_PER_ATOM,
    perturb,
)
from wyckoff_transformer.evaluation.hull_energy import HullEnergyCalculator
from wyckoff_transformer.evaluation.hull_mlips import (
    DEFAULT_HULL_MLIP,
    HULL_MLIPS,
    PUBLISHED_HULL_ENTRIES,
    UnsupportedHullMlip,
    _use_cpu_orb_neighbors_when_needed,
    resolve_hull_mlip,
)
from wyckoff_transformer.evaluation.protocol import (
    DEFAULT_TRIAL_SCHEDULE,
    GeneFingerprinter,
    GeneScreen,
    funnel,
    parse_trial_schedule,
    positional_dof,
    read_screen,
    screen_genes,
    trials_for_dof,
    write_screen,
)

NACL = {"group": 225, "species": ["Na", "Cl"], "numIons": [4, 4], "sites": [["4a"], ["4b"]]}
# Same gene, sites listed in the other order: a duplicate the fingerprint must catch.
NACL_REORDERED = {
    "group": 225, "species": ["Cl", "Na"], "numIons": [4, 4], "sites": [["4b"], ["4a"]],
}
OTHER = {"group": 221, "species": ["Sr", "Ti", "O"], "numIons": [1, 1, 3],
         "sites": [["1a"], ["1b"], ["3c"]]}
# 16z does not exist in space group 225.
ILLEGAL = {"group": 225, "species": ["Na"], "numIons": [16], "sites": [["16z"]]}


class TestHullMlips(unittest.TestCase):
    def test_default_is_orb(self):
        self.assertEqual(DEFAULT_HULL_MLIP, "orb_conserv_inf")
        self.assertTrue(resolve_hull_mlip(DEFAULT_HULL_MLIP).is_runnable)

    def test_every_published_hull_is_registered(self):
        # The splits of LeMaterial/LeMat-Bulk-MLIP-Hull.
        self.assertEqual(
            set(HULL_MLIPS),
            {"dft", "mace_mp", "mace_omat", "orb_conserv_inf", "orb_direct_20", "uma"},
        )

    def test_raises_for_mlip_without_a_published_hull(self):
        for name in ("chgnet", "pet", "grace", "mace", "orb"):
            with self.assertRaises(UnsupportedHullMlip):
                resolve_hull_mlip(name)

    def test_raises_for_dft_which_is_a_hull_but_not_a_potential(self):
        with self.assertRaises(UnsupportedHullMlip):
            resolve_hull_mlip("dft")

    def test_orb_checkpoint_is_the_pinned_url(self):
        spec = resolve_hull_mlip("orb_conserv_inf")
        self.assertIn("orb-v3-conservative-inf-omat-20250404.ckpt", spec.checkpoint)

    def test_orb_checkpoints_match_the_installed_orb_models(self):
        """The recorded URLs must be what orb-models actually loads.

        These are what ties our energies to LeMat-Bulk's orb hulls. If a future
        orb-models release repoints the same function at different weights, the
        pairing breaks silently -- so fail here instead.
        """
        pretrained = pytest.importorskip("orb_models.forcefield.pretrained")
        import inspect

        from wyckoff_transformer.evaluation.hull_mlips import (
            ORB_CONSERV_INF_CHECKPOINT,
            ORB_DIRECT_20_CHECKPOINT,
        )

        for function_name, expected in (
            ("orb_v3_conservative_inf_omat", ORB_CONSERV_INF_CHECKPOINT),
            ("orb_v3_direct_20_omat", ORB_DIRECT_20_CHECKPOINT),
        ):
            signature = inspect.signature(getattr(pretrained, function_name))
            self.assertEqual(signature.parameters["weights_path"].default, expected)

    def test_ambiguous_mace_checkpoint_is_flagged_and_named_explicitly(self):
        self.assertIn("UNIDENTIFIED CHECKPOINT", HULL_MLIPS["mace_mp"].note)
        # Named outright rather than left to mace-torch's version-dependent
        # mace_mp(model=None) alias, which is what made it ambiguous.
        self.assertEqual(HULL_MLIPS["mace_mp"].checkpoint, "MACE-MP-0a-medium")

    def test_cpu_only_warp_uses_cpu_neighbors_for_cuda_orb(self):
        calculator = MagicMock()
        calculator.adapter.from_ase_atoms.return_value = SimpleNamespace(
            to=MagicMock(return_value="cuda-batch")
        )
        calculator.device = "cuda:0"
        calculator.model.predict.return_value = "prediction"
        atoms = object()

        with patch("warp.get_devices", return_value=["cpu"]):
            with patch("ase.calculators.calculator.Calculator.calculate") as calculate:
                result = _use_cpu_orb_neighbors_when_needed(calculator, "cuda:0")
                result.calculate(atoms)

        self.assertIs(result, calculator)
        calculate.assert_called_once_with(calculator, atoms)
        calculator.adapter.from_ase_atoms.assert_called_once_with(
            atoms=atoms,
            max_num_neighbors=calculator.max_num_neighbors,
            edge_method=calculator.edge_method,
            half_supercell=calculator.half_supercell,
            device="cpu",
        )
        calculator.adapter.from_ase_atoms.return_value.to.assert_called_once_with("cuda:0")
        calculator.model.predict.assert_called_once_with("cuda-batch")
        calculator._update_results.assert_called_once_with("prediction")

    def test_cuda_warp_keeps_orb_calculator_unmodified(self):
        calculator = MagicMock()
        original_calculate = calculator.calculate
        with patch("warp.get_devices", return_value=["cuda:0"]):
            result = _use_cpu_orb_neighbors_when_needed(calculator, "cuda:0")
        self.assertIs(result, calculator)
        self.assertIs(calculator.calculate, original_calculate)


def _cached_hull_parquet(hull_type: str = "orb_conserv_inf"):
    """The published hull parquet if it is already in the HuggingFace cache."""
    from huggingface_hub import try_to_load_from_cache

    from wyckoff_transformer.evaluation.hull_mlips import HULL_REPO_ID

    path = try_to_load_from_cache(
        repo_id=HULL_REPO_ID,
        filename=f"data/{hull_type}-00000-of-00001.parquet",
        repo_type="dataset",
    )
    return path if isinstance(path, str) else None


class TestHullProvenance(unittest.TestCase):
    """The hull must be the whole published one, and say so in the manifest."""

    def test_a_row_count_is_pinned_for_every_published_split(self):
        self.assertEqual(set(PUBLISHED_HULL_ENTRIES), set(HULL_MLIPS))

    def test_the_loaded_hull_is_the_full_published_split(self):
        parquet = _cached_hull_parquet()
        if parquet is None:
            self.skipTest("the published hull is not in the HuggingFace cache")
        hull = HullEnergyCalculator("orb_conserv_inf", parquet=parquet)
        self.assertEqual(
            hull.provenance["entries"], PUBLISHED_HULL_ENTRIES["orb_conserv_inf"]
        )
        self.assertEqual(hull.provenance["hull_type"], "orb_conserv_inf")

    def test_the_full_hull_covers_the_elements_the_training_labels_drop(self):
        """The training labels used to exclude Yb and everything past Po.

        That exclusion never belonged here -- a generated structure containing
        Yb has to be scored, not silently dropped -- and as of the 2026-09-07
        relabelling it is gone from the training side too. This pins the
        property that mattered on this side all along.
        """
        parquet = _cached_hull_parquet()
        if parquet is None:
            self.skipTest("the published hull is not in the HuggingFace cache")
        from pymatgen.core import Composition

        hull = HullEnergyCalculator("orb_conserv_inf", parquet=parquet)
        for formula in ("Yb2O3", "U2O", "ThO2"):
            with self.subTest(formula=formula):
                # A real number, not a raise: the subspace is populated.
                self.assertIsInstance(
                    hull.energy_above_hull(-100.0, Composition(formula)), float
                )

    def test_a_subset_of_the_hull_is_reported_as_one(self):
        parquet = _cached_hull_parquet()
        if parquet is None:
            self.skipTest("the published hull is not in the HuggingFace cache")
        frame = pd.read_parquet(parquet).head(1000)
        with tempfile.TemporaryDirectory() as tmp:
            subset = Path(tmp) / "subset.parquet"
            frame.to_parquet(subset)
            with self.assertLogs(
                "wyckoff_transformer.evaluation.hull_energy", level="WARNING"
            ) as logs:
                hull = HullEnergyCalculator("orb_conserv_inf", parquet=subset)
        self.assertEqual(hull.provenance["entries"], 1000)
        self.assertTrue(any("not the full LeMat-Bulk" in line for line in logs.output))


class TestResolveDevices(unittest.TestCase):
    def test_cores_give_one_cpu_slot_each(self):
        self.assertEqual(resolve_devices(4, None, 1), ["cpu"] * 4)

    def test_devices_are_repeated_per_worker(self):
        self.assertEqual(
            resolve_devices(None, "cuda:0,cuda:1", 2),
            ["cuda:0", "cuda:0", "cuda:1", "cuda:1"],
        )

    def test_default_is_a_single_cpu_worker(self):
        self.assertEqual(resolve_devices(None, None, 1), ["cpu"])

    def test_cores_and_devices_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            resolve_devices(4, "cuda:0", 1)

    def test_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            resolve_devices(0, None, 1)
        with self.assertRaises(ValueError):
            resolve_devices(None, " , ", 1)
        with self.assertRaises(ValueError):
            resolve_devices(None, "cuda:0", 0)


class TestClaimDevice(unittest.TestCase):
    """Each worker must get the slot it was allotted, never a fallback.

    A queue of devices looked equivalent and was not: ``Queue.put`` defers to a
    feeder thread, so a worker calling ``get_nowait`` at spawn time could find
    it empty and silently fall back to CPU -- one idle GPU and a few trials
    running ten times too slowly, with nothing in the log to say so.
    """

    @staticmethod
    def _counter(start=0):
        import multiprocessing

        return multiprocessing.get_context("spawn").Value("i", start)

    def test_every_slot_is_handed_out_once(self):
        slots = ["cuda:0", "cuda:0", "cuda:1", "cuda:1", "cuda:2"]
        counter = self._counter()
        claimed = [claim_device(counter, slots) for _ in slots]
        self.assertEqual(claimed, slots)

    def test_a_replacement_worker_takes_a_real_device_not_the_cpu(self):
        # The pool respawns a worker after a crash; it must land on a card
        # rather than quietly halving the run's throughput.
        slots = ["cuda:0", "cuda:1"]
        counter = self._counter()
        claimed = [claim_device(counter, slots) for _ in range(5)]
        self.assertEqual(claimed, ["cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0"])
        self.assertNotIn("cpu", claimed)

    def test_the_initialiser_takes_the_counter_and_the_slots(self):
        # Guards the initargs tuple against drifting from the signature, which
        # a pool reports only as a worker that dies at startup.
        import inspect

        self.assertEqual(
            list(inspect.signature(_init_relax_worker).parameters),
            ["counter", "slots", "mlip", "debug"],
        )


class TestLogmRoundoffFilter(unittest.TestCase):
    """SciPy's logm chatter must go without taking a real warning with it.

    ASE takes a matrix logarithm once per optimiser step, so these arrive in
    the thousands. They cannot be filtered by message: SciPy interpolates the
    residual into the text, so every one is a different string and both the
    `default` and `once` actions -- which key their registries by text -- print
    every one. That is why the relaxation log drowns in them.
    """

    def setUp(self):
        import warnings

        self._saved = warnings.showwarning
        self.addCleanup(setattr, warnings, "showwarning", self._saved)
        self.seen = []
        warnings.showwarning = lambda message, *a, **k: self.seen.append(str(message))
        _quiet_logm_roundoff()

    @staticmethod
    def _warn(text):
        import warnings

        warnings.warn(text, RuntimeWarning)

    def test_roundoff_residuals_are_dropped(self):
        for residual in (6.89e-13, 7.12e-13, 4.55e-13, 9.01e-13):
            self._warn(f"logm result may be inaccurate, approximate err = {residual}")
        self.assertEqual(self.seen, [])

    def test_a_residual_worth_seeing_survives(self):
        # 1e-6 is a near-singular deformation gradient, i.e. a collapsing cell.
        self._warn("logm result may be inaccurate, approximate err = 1e-06")
        self.assertEqual(len(self.seen), 1)

    def test_other_warnings_are_untouched(self):
        self._warn("something else entirely")
        self.assertEqual(self.seen, ["something else entirely"])

    def test_an_unparseable_residual_is_kept_rather_than_guessed(self):
        self._warn("logm result may be inaccurate, approximate err = nonsense")
        self.assertEqual(len(self.seen), 1)

    def test_installing_twice_does_not_nest_the_wrapper(self):
        import warnings

        wrapped = warnings.showwarning
        _quiet_logm_roundoff()
        self.assertIs(warnings.showwarning, wrapped)

    def test_the_threshold_sits_well_above_scipys(self):
        # SciPy warns above 1000*eps = 2.2e-13; the drop threshold must cover
        # that with room, and stay far below anything that moves a relaxation.
        self.assertGreater(LOGM_ROUNDOFF, 1000 * 2.220446049250313e-16)
        self.assertLess(LOGM_ROUNDOFF, 1e-6)


class TestPinVisibleDevice(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        self.addCleanup(self._restore)

    def _restore(self):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        if self._saved is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._saved

    def test_a_named_card_becomes_the_only_visible_one(self):
        # The worker keeps its own card and loses the others, so nothing it does
        # can leave a CUDA context on a GPU somebody else is using.
        self.assertEqual(_pin_visible_device("cuda:1"), "cuda:0")
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")

    def test_cpu_is_left_alone(self):
        self.assertEqual(_pin_visible_device("cpu"), "cpu")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)

    def test_bare_cuda_pins_nothing(self):
        # No card was named, so there is nothing to hide and no renumbering.
        self.assertEqual(_pin_visible_device("cuda"), "cuda")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)


class TestTwoStageNovelty(unittest.TestCase):
    """Novelty is the fingerprint *and* the matcher, not the fingerprint alone."""

    @staticmethod
    def _structures():
        from pymatgen.core import Lattice, Structure

        silicon = Structure(
            Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]]
        )
        salt = Structure(
            Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        return silicon, salt

    def _filter(self, reference_structure):
        from wyckoff_transformer.evaluation.novelty import NoveltyFilter

        return NoveltyFilter(
            pd.DataFrame(
                {"fingerprint": ["F"], "structure": [reference_structure]},
                index=["agm000000001"],
            )
        )

    def test_a_shared_fingerprint_with_a_different_structure_is_still_novel(self):
        # The case the gene screen alone gets wrong: same space group, same
        # elements on the same Wyckoff orbits, different structure.
        silicon, salt = self._structures()
        novelty = self._filter(silicon)
        record = pd.Series({"fingerprint": "F", "structure": salt})
        self.assertTrue(novelty.is_novel(record))

    def test_a_shared_fingerprint_with_the_same_structure_is_known(self):
        silicon, _ = self._structures()
        novelty = self._filter(silicon)
        record = pd.Series({"fingerprint": "F", "structure": silicon.copy()})
        self.assertFalse(novelty.is_novel(record))

    def test_an_unseen_fingerprint_needs_no_matching(self):
        silicon, salt = self._structures()
        novelty = self._filter(silicon)
        record = pd.Series({"fingerprint": "G", "structure": silicon.copy()})
        self.assertTrue(novelty.is_novel(record))

    def test_uniqueness_keeps_different_structures_sharing_a_fingerprint(self):
        from wyckoff_transformer.evaluation.novelty import filter_by_unique_structure

        silicon, salt = self._structures()
        frame = pd.DataFrame(
            {
                "fingerprint": ["F", "F", "F"],
                "structure": [silicon, salt, silicon.copy()],
            },
            index=[0, 1, 2],
        )
        kept = filter_by_unique_structure(frame)
        # 2 is silicon again, so it goes; 1 is a different structure, so it stays.
        self.assertEqual(list(kept.index), [0, 1])


class TestNoveltyReference(unittest.TestCase):
    def test_nothing_to_match_needs_no_reference_data(self):
        # Every generated fingerprint absent from LeMat-Bulk means no candidate
        # can exist, so neither the 4M-row cache nor the 1 GB CIF export is read.
        from wyckoff_transformer.evaluation.structure_novelty import (
            build_novelty_reference,
        )

        reference = build_novelty_reference(
            [], cache=Path("/nonexistent.pkl.gz"),
            lemat_cif_csv=Path("/nonexistent.csv.gz"),
        )
        self.assertTrue(reference.empty)
        self.assertEqual(list(reference.columns), ["fingerprint", "structure"])


class TestRelaxedFingerprint(unittest.TestCase):
    """The relaxed structure is re-fingerprinted, not trusted to match its gene."""

    def test_a_structure_fingerprints_to_its_gene_when_symmetry_is_unchanged(self):
        from pymatgen.core import Lattice, Structure

        fingerprinter = GeneFingerprinter()
        # Rocksalt NaCl: Na on 4a, Cl on 4b of Fm-3m -- exactly the NACL gene.
        rocksalt = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        self.assertEqual(
            fingerprinter.fingerprint_structure(rocksalt),
            fingerprinter.fingerprint(NACL),
        )


class TestCliDefaults(unittest.TestCase):
    def test_protocol_defaults_match_the_specification(self):
        args = build_parser().parse_args(["genes.json", "--output-dir", "out"])
        self.assertEqual(args.mlip, "orb_conserv_inf")
        self.assertEqual(args.n_trials, DEFAULT_TRIAL_SCHEDULE)
        # All four stages: the unconstrained one is the rattle's baseline.
        self.assertTrue(args.release_symmetry)
        self.assertTrue(args.rattle)

    def test_the_unconstrained_stage_can_be_dropped(self):
        args = build_parser().parse_args(
            ["genes.json", "--output-dir", "out", "--no-release-symmetry"]
        )
        self.assertFalse(args.release_symmetry)

    def test_rattle_can_be_turned_off(self):
        args = build_parser().parse_args(
            ["genes.json", "--output-dir", "out", "--no-rattle"]
        )
        self.assertFalse(args.rattle)

    def test_release_symmetry_can_be_turned_back_on(self):
        args = build_parser().parse_args(
            ["genes.json", "--output-dir", "out", "--release-symmetry"]
        )
        self.assertTrue(args.release_symmetry)

    def test_the_parser_can_print_its_own_help(self):
        """argparse runs `help % params`, so a literal % in a help string raises.

        `parse_args` never formats the help, so nothing else in this class
        catches it -- and a `78%` in a help string turned `--help` into a
        TypeError once already.
        """
        build_parser().format_help()

    def test_mlip_choices_are_restricted_to_published_hulls(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["genes.json", "--output-dir", "out", "--mlip", "chgnet"]
            )

    def test_the_four_stages_are_separately_runnable(self):
        for stage in ("screen", "generate", "relax", "score", "all"):
            args = build_parser().parse_args(
                ["genes.json", "--output-dir", "out", "--stage", stage]
            )
            self.assertEqual(args.stage, stage)

    def test_generation_and_relaxation_have_their_own_timeouts(self):
        """Neither stage may be held hostage by one gene it cannot finish."""
        args = build_parser().parse_args(["genes.json", "--output-dir", "out"])
        self.assertEqual(args.pyxtal_timeout, 300.0)
        self.assertEqual(args.relax_timeout, 1800.0)
        self.assertIsNone(args.pyxtal_cores)  # every core

    def test_resume_is_on_by_default(self):
        # Both per-trial logs are written row by row, so repeating finished
        # trials is pure waste; --no-resume is the deliberate fresh start.
        args = build_parser().parse_args(["genes.json", "--output-dir", "out"])
        self.assertTrue(args.resume)
        args = build_parser().parse_args(
            ["genes.json", "--output-dir", "out", "--no-resume"]
        )
        self.assertFalse(args.resume)


class TestTimeLimit(unittest.TestCase):
    def test_a_slow_block_is_interrupted(self):
        with self.assertRaises(Timeout):
            with time_limit(0.05):
                while True:
                    pass

    def test_a_timeout_survives_a_broad_except_clause(self):
        """PyXtal's callers catch Exception; a timeout must not look like one.

        ``single_pyxtal`` reports every failure as ``None``, so a Timeout
        derived from Exception would be recorded as an ordinary generation
        failure rather than as the gene that hung.
        """
        self.assertFalse(issubclass(Timeout, Exception))

    def test_no_limit_leaves_the_block_alone(self):
        with time_limit(0):
            pass
        with time_limit(None):
            pass


class TestRowLog(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "rows.csv"
        self.addCleanup(self._tmp.cleanup)

    def test_rows_are_on_disk_before_the_stage_ends(self):
        log = RowLog(self.path, PYXTAL_COLUMNS, resume=False)
        log.write({"index": 3, "trial": 0, "status": "ok"})
        # Deliberately not closed: an interrupted stage never closes it either.
        self.assertEqual(len(pd.read_csv(self.path)), 1)
        log.close()

    def test_resuming_skips_what_is_already_recorded(self):
        log = RowLog(self.path, PYXTAL_COLUMNS, resume=False)
        log.write({"index": 3, "trial": 0, "status": "ok"})
        log.write({"index": 3, "trial": 1, "status": "failed"})
        log.close()

        resumed = RowLog(self.path, PYXTAL_COLUMNS, resume=True)
        self.assertEqual(resumed.done, {(3, 0), (3, 1)})
        resumed.write({"index": 4, "trial": 0, "status": "ok"})
        self.assertEqual(len(resumed.frame()), 3)
        resumed.close()

    def test_no_resume_starts_from_scratch(self):
        first = RowLog(self.path, PYXTAL_COLUMNS, resume=False)
        first.write({"index": 3, "trial": 0, "status": "ok"})
        first.close()
        fresh = RowLog(self.path, PYXTAL_COLUMNS, resume=False)
        self.assertEqual(fresh.done, set())
        fresh.close()
        self.assertEqual(len(pd.read_csv(self.path)), 0)

    def test_a_line_cut_in_half_by_a_kill_is_dropped_not_fatal(self):
        log = RowLog(self.path, PYXTAL_COLUMNS, resume=False)
        log.write({"index": 3, "trial": 0, "status": "ok"})
        log.close()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("4,0,ok,NaCl")  # killed mid-row: no newline, no rest
        resumed = RowLog(self.path, PYXTAL_COLUMNS, resume=True)
        self.assertEqual(resumed.done, {(3, 0)})
        resumed.close()


class TestAggregateStructures(unittest.TestCase):
    """The reduction from per-trial rows to the one structure a gene kept."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, draws, relaxations):
        pd.DataFrame(draws, columns=list(PYXTAL_COLUMNS)).to_csv(
            self.out / "pyxtal.csv", index=False
        )
        pd.DataFrame(relaxations, columns=list(RELAXATION_COLUMNS)).to_csv(
            self.out / "relaxations.csv", index=False
        )

    def _cif(self, name, text):
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_the_lowest_energy_trial_wins_and_its_cif_is_kept(self):
        self._write(
            draws=[
                {"index": 0, "trial": 0, "status": "ok", "dof_positional": 3,
                 "n_trials": 2, "seconds": 1.0},
                {"index": 0, "trial": 1, "status": "ok", "dof_positional": 3,
                 "n_trials": 2, "seconds": 1.0},
            ],
            relaxations=[
                {"index": 0, "trial": 0, "status": "ok", "formula": "NaCl",
                 "energy": -5.0, "energy_per_atom": -2.5, "n_atoms": 2,
                 "device": "cuda:0", "seconds": 2.0,
                 "cif": self._cif("cryspr/0/trial-0/NaCl_kept.cif", "data_high")},
                {"index": 0, "trial": 1, "status": "ok", "formula": "NaCl",
                 "energy": -9.0, "energy_per_atom": -4.5, "n_atoms": 2,
                 "device": "cuda:1", "seconds": 2.0,
                 "cif": self._cif("cryspr/0/trial-1/NaCl_kept.cif", "data_low")},
            ],
        )
        frame = aggregate_structures(self.out)
        self.assertTrue(bool(frame.at[0, "has_structure"]))
        self.assertEqual(frame.at[0, "energy"], -9.0)
        self.assertEqual(frame.at[0, "best_trial"], 1)
        self.assertEqual(frame.at[0, "n_relaxed"], 2)
        self.assertEqual(
            (self.out / "cifs" / "0.cif").read_text(encoding="utf-8"), "data_low"
        )

    def test_a_failed_relaxation_is_reported_with_its_reason(self):
        """The funnel says a gene has no structure; only this says why.

        A whole cohort once came back with has_structure False on every gene
        and nothing anywhere recording that the potential had failed to load.
        """
        self._write(
            draws=[{"index": 1, "trial": 0, "status": "ok", "dof_positional": 0,
                    "n_trials": 1, "seconds": 1.0}],
            relaxations=[{"index": 1, "trial": 0, "status": "failed",
                          "error": "RuntimeError: CUDA out of memory",
                          "seconds": 0.5}],
        )
        frame = aggregate_structures(self.out)
        self.assertFalse(bool(frame.at[1, "has_structure"]))
        self.assertIn("CUDA out of memory", frame.at[1, "error"])

    def test_a_gene_pyxtal_could_not_draw_is_told_apart_from_a_relaxation_failure(self):
        self._write(
            draws=[{"index": 2, "trial": 0, "status": "timeout",
                    "dof_positional": 12, "n_trials": 3, "seconds": 300.0}],
            relaxations=[],
        )
        frame = aggregate_structures(self.out)
        self.assertFalse(bool(frame.at[2, "has_structure"]))
        self.assertIn("timeout", frame.at[2, "error"])
        self.assertEqual(frame.at[2, "n_drawn"], 0)

    def test_a_draw_that_was_never_relaxed_keeps_a_row(self):
        # The relax stage has not reached this gene: that is not a failure of
        # the gene, and the reason must not read like one.
        self._write(
            draws=[{"index": 5, "trial": 0, "status": "ok", "dof_positional": 1,
                    "n_trials": 2, "seconds": 1.0}],
            relaxations=[],
        )
        frame = aggregate_structures(self.out)
        self.assertEqual(frame.at[5, "error"], "generated but never relaxed")


class TestTrialSchedule(unittest.TestCase):
    def test_the_default_spends_trials_where_the_free_coordinates_are(self):
        schedule = parse_trial_schedule(DEFAULT_TRIAL_SCHEDULE)
        # A rigid gene has nothing to redraw, so a second trial cannot help.
        self.assertEqual(trials_for_dof(0, schedule), 1)
        self.assertEqual(trials_for_dof(1, schedule), 2)
        self.assertEqual(trials_for_dof(2, schedule), 2)
        self.assertEqual(trials_for_dof(3, schedule), 3)
        self.assertEqual(trials_for_dof(97, schedule), 3)

    def test_a_bare_integer_is_a_constant_schedule(self):
        schedule = parse_trial_schedule("3")
        self.assertEqual(trials_for_dof(0, schedule), 3)
        self.assertEqual(trials_for_dof(42, schedule), 3)

    def test_bins_are_inclusive_upper_bounds(self):
        schedule = parse_trial_schedule("0:1,2:2,5:3,*:4")
        self.assertEqual([trials_for_dof(d, schedule) for d in range(8)],
                         [1, 2, 2, 3, 3, 3, 4, 4])

    def test_a_schedule_that_leaves_high_dof_genes_out_is_refused(self):
        with self.assertRaises(ValueError):
            parse_trial_schedule("0:1,5:2")

    def test_unordered_or_malformed_schedules_are_refused(self):
        for spec in ("5:2,0:1,*:3", "0:1,0:2,*:3", "nonsense", "*:0", "", "2"):
            with self.subTest(spec=spec):
                if spec == "2":  # a valid constant schedule, for contrast
                    self.assertEqual(len(parse_trial_schedule(spec)), 1)
                    continue
                with self.assertRaises(ValueError):
                    parse_trial_schedule(spec)


class TestPositionalDof(unittest.TestCase):
    def test_a_fully_determined_gene_has_no_free_coordinate(self):
        # 4a and 4b of Fm-3m are both fixed points: rock salt has none.
        self.assertEqual(positional_dof(NACL), 0)
        self.assertEqual(positional_dof(OTHER), 0)

    def test_a_free_orbit_contributes_its_own_degrees_of_freedom(self):
        # 4i of P4/mmm sits at (0, 1/2, z).
        gene = {"group": 123, "species": ["Cu"], "numIons": [4], "sites": [["4i"]]}
        self.assertEqual(positional_dof(gene), 1)


def _mock_atoms(energy: float = -1.0) -> MagicMock:
    atoms = MagicMock()
    atoms.copy.return_value = atoms
    atoms.get_chemical_formula.return_value = "NaCl"
    atoms.get_potential_energy.return_value = energy
    atoms.__len__.return_value = 8
    return atoms


class TestStepwiseRelaxStages(unittest.TestCase):
    """Each schedule must run exactly the stages it claims, and not silently no-op."""

    def _labels_for(self, **kwargs) -> list[str]:
        from wyckoff_transformer.cryspr import relaxer

        atoms = _mock_atoms()
        seen = []

        def fake_relaxer(*, label, **_):
            seen.append(label)
            return atoms

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(relaxer, "run_ase_relaxer", side_effect=fake_relaxer), \
                patch.object(relaxer, "write"):
            relaxer.stepwise_relax(
                atoms_in=atoms, calculator=MagicMock(), wdir=Path(tmp), **kwargs
            )
        return seen

    def test_every_stage_by_default(self):
        self.assertEqual(
            self._labels_for(),
            ["1_fix-cell", "2_sym_cell+pos", "3_no-sym_cell+pos",
             "4_rattle_no-sym"],
        )

    def test_dropping_the_unconstrained_stage_still_rattles(self):
        self.assertEqual(
            self._labels_for(release_symmetry=False),
            ["1_fix-cell", "2_sym_cell+pos", "4_rattle_no-sym"],
        )

    def test_no_stage_relaxes_the_cell_is_refused(self):
        from wyckoff_transformer.cryspr import relaxer

        with self.assertRaises(ValueError):
            relaxer.stepwise_relax(
                atoms_in=MagicMock(), calculator=MagicMock(),
                fix_symmetry=False, release_symmetry=False, rattle=False,
            )


class TestRattleStage(unittest.TestCase):
    """The rattle must actually perturb, and must only be kept when it wins."""

    def _structure(self):
        from ase import Atoms
        from ase.constraints import FixSymmetry

        atoms = Atoms("Cu4", cell=[3.6, 3.6, 3.6], pbc=True, scaled_positions=[
            (0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0),
        ])
        atoms.set_constraint([FixSymmetry(atoms, symprec=1e-3)])
        return atoms

    def test_a_fixsymmetry_constraint_would_undo_the_rattle(self):
        # Why perturb() clears the constraints first: set_positions enforces
        # them, so FixSymmetry symmetrises the displacement away and the stage
        # would relax from exactly where it started.
        atoms = self._structure()
        before = atoms.get_positions()
        atoms.rattle(stdev=0.05, seed=1)
        self.assertTrue((atoms.get_positions() == before).all())

    def test_perturb_moves_the_atoms_and_the_cell(self):
        atoms = self._structure()
        perturbed = perturb(atoms, seed=1)
        self.assertFalse((perturbed.get_positions() == atoms.get_positions()).all())
        self.assertFalse((perturbed.cell.array == atoms.cell.array).all())
        self.assertEqual(perturbed.constraints, [])

    def test_the_same_seed_gives_the_same_perturbation(self):
        atoms = self._structure()
        self.assertTrue(
            (perturb(atoms, seed=7).get_positions()
             == perturb(atoms, seed=7).get_positions()).all()
        )
        self.assertFalse(
            (perturb(atoms, seed=7).get_positions()
             == perturb(atoms, seed=8).get_positions()).all()
        )

    def _kept(self, energy_after: float):
        """Run the rattle stage against a fake relaxer with a known outcome."""
        from wyckoff_transformer.cryspr import relaxer

        before = _mock_atoms(energy=-1.0)
        after = _mock_atoms(energy=energy_after)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(relaxer, "run_ase_relaxer", return_value=after):
            kept = relaxer._rattle_stage(
                before,
                rattle_stdev=0.05,
                strain_stdev=0.01,
                rattle_accept=RATTLE_ACCEPT_EV_PER_ATOM,
                seed=1,
                logfile=Path(tmp) / "rattle.log",
                calculator=MagicMock(),
                optimizer=MagicMock(),
                hydrostatic_strain=False,
                symprec=1e-3,
                fmax=0.05,
                steps_limit=500,
                wdir=Path(tmp),
            )
        return kept is after

    def test_a_win_larger_than_the_margin_is_kept(self):
        # 8 atoms, so -0.02 eV total is 2.5 meV/atom below the margin.
        self.assertTrue(self._kept(energy_after=-1.02))

    def test_noise_below_the_margin_is_rejected(self):
        self.assertFalse(self._kept(energy_after=-1.000001))

    def test_a_worse_structure_is_rejected(self):
        self.assertFalse(self._kept(energy_after=-0.5))


class TestScreenGenes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fingerprinter = GeneFingerprinter()
        cls.nacl_fp = cls.fingerprinter.fingerprint(NACL)

    def test_illegal_wyckoff_letter_is_invalid(self):
        screen = screen_genes([ILLEGAL], set(), self.fingerprinter)
        self.assertEqual(screen.valid, [])
        self.assertEqual(screen.invalid, [0])
        self.assertIn(0, screen.invalid_reason)

    def test_duplicates_are_counted_not_dropped(self):
        screen = screen_genes([NACL, NACL, OTHER], set(), self.fingerprinter)
        self.assertEqual(screen.n_sampled, 3)
        self.assertEqual(len(screen.valid), 3)
        self.assertEqual(screen.n_unique, 2)
        self.assertEqual(screen.counts[0], 2)  # both NaCl samples land on index 0
        self.assertEqual(screen.counts[2], 1)

    def test_reordered_sites_are_the_same_gene(self):
        screen = screen_genes([NACL, NACL_REORDERED], set(), self.fingerprinter)
        self.assertEqual(screen.n_unique, 1)
        self.assertEqual(screen.counts[0], 2)

    def test_known_genes_are_split_off_and_never_relaxed(self):
        screen = screen_genes([NACL, OTHER], {self.nacl_fp}, self.fingerprinter)
        self.assertEqual(screen.known, [0])
        self.assertEqual(screen.novel, [1])

    def test_sampled_counts_track_representatives(self):
        screen = screen_genes([NACL, NACL, OTHER], {self.nacl_fp}, self.fingerprinter)
        self.assertEqual(screen.n_sampled_known, 2)
        self.assertEqual(screen.n_sampled_novel, 1)

    def test_summary_rates_use_the_sampled_denominator(self):
        screen = screen_genes([NACL, NACL, ILLEGAL], set(), self.fingerprinter)
        summary = screen.summary()
        self.assertEqual(summary["sampled"], 3)
        self.assertEqual(summary["valid_gene"], 2)
        self.assertAlmostEqual(summary["valid_gene_rate"], 2 / 3)
        self.assertAlmostEqual(summary["unique_gene_rate"], 1 / 3)

    def test_round_trip_through_disk(self):
        screen = screen_genes([NACL, NACL, OTHER], {self.nacl_fp}, self.fingerprinter)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screen.json"
            write_screen(screen, path)
            restored = read_screen(path)
        self.assertEqual(restored.counts, screen.counts)
        self.assertEqual(restored.novel, screen.novel)
        self.assertEqual(restored.known, screen.known)
        self.assertEqual(restored.summary(), screen.summary())


class TestFunnel(unittest.TestCase):
    def _screen(self) -> GeneScreen:
        # 10 sampled genes: 8 valid, of which 4 unique. Representative 0 stands
        # for 3 samples, 1 for 2, 2 for 2, 3 for 1. Genes 0-2 are novel.
        return GeneScreen(
            n_sampled=10,
            valid=[0, 1, 2, 3],
            invalid=[8, 9],
            counts={0: 3, 1: 2, 2: 2, 3: 1},
            novel=[0, 1, 2],
            known=[3],
        )

    def test_rates_are_per_sampled_gene_and_weighted_by_duplicates(self):
        structures = pd.DataFrame(
            {
                "has_structure": [True, True, True],
                "valid_structure": [True, True, False],
                "unique_structure": [True, True, True],
                "novel_structure": [True, True, True],
                "e_above_hull": [0.05, 0.3, 0.0],
            },
            index=[0, 1, 2],
        )
        report = funnel(self._screen(), structures)
        self.assertEqual(report["sampled"], 10)
        self.assertEqual(report["valid_structure"], 2)
        # Genes 0 and 1 stand for 3 + 2 = 5 of the 10 sampled genes.
        self.assertAlmostEqual(report["valid_structure_per_sampled_gene"], 0.5)
        # Only gene 0 is a unique structure at or below 0.1 eV/atom.
        self.assertEqual(report["metastable"], 1)
        self.assertEqual(report["metastable_among_novel"], 1)
        self.assertAlmostEqual(report["metastable_per_sampled_gene"], 0.3)
        self.assertAlmostEqual(report["metasun_per_sampled_gene"], 0.3)
        self.assertEqual(report["stable"], 0)
        self.assertEqual(report["stable_among_novel"], 0)
        self.assertAlmostEqual(report["sun_per_sampled_gene"], 0.0)

    def test_metastable_ignores_novelty_but_metasun_applies_it(self):
        # Genes 0-3 are all unique structures below 0.1 eV/atom; 1 and 3 are not
        # novel. counts are {0: 3, 1: 2, 2: 2, 3: 1} over 10 sampled genes.
        structures = pd.DataFrame(
            {
                "has_structure": [True, True, True, True],
                "valid_structure": [True, True, True, True],
                "unique_structure": [True, True, True, True],
                "novel_structure": [True, False, True, False],
                "e_above_hull": [0.05, 0.02, -0.01, -0.2],
            },
            index=[0, 1, 2, 3],
        )
        report = funnel(self._screen(), structures)
        self.assertEqual(report["metastable"], 4)
        self.assertEqual(report["metastable_among_novel"], 2)
        self.assertAlmostEqual(report["metastable_per_sampled_gene"], 0.8)
        self.assertAlmostEqual(report["metasun_per_sampled_gene"], 0.5)
        # Only genes 2 and 3 are at or below 0 eV/atom; only gene 2 is novel.
        self.assertEqual(report["stable"], 2)
        self.assertEqual(report["stable_among_novel"], 1)
        self.assertAlmostEqual(report["stable_per_sampled_gene"], 0.3)
        self.assertAlmostEqual(report["sun_per_sampled_gene"], 0.2)

    def test_novelty_transitions_are_counted_against_the_sampled_gene(self):
        structures = pd.DataFrame(
            {
                "has_structure": [True, True, True, True],
                "valid_structure": [True, True, True, True],
                "unique_structure": [True, True, True, True],
                # gene 3 is known but relaxed to a structure the matcher rejects;
                # gene 1 is a novel gene that relaxed onto a known structure.
                "novel_structure": [True, False, True, True],
                "relaxed_fingerprint_resolved": [True, True, False, True],
                "relaxed_fingerprint_changed": [False, True, False, True],
            },
            index=[0, 1, 2, 3],
        )
        report = funnel(self._screen(), structures)
        self.assertEqual(report["gene_known_became_novel"], 1)
        self.assertEqual(report["gene_novel_became_known"], 1)
        # weighted by counts {3: 1, 1: 2} over 10 sampled genes.
        self.assertAlmostEqual(
            report["gene_known_became_novel_per_sampled_gene"], 0.1
        )
        self.assertAlmostEqual(
            report["gene_novel_became_known_per_sampled_gene"], 0.2
        )
        self.assertEqual(report["relaxed_fingerprint_resolved"], 3)
        self.assertEqual(report["relaxed_fingerprint_changed"], 2)

    def test_novelty_transitions_are_none_without_the_columns(self):
        structures = pd.DataFrame({"has_structure": [True]}, index=[0])
        report = funnel(self._screen(), structures)
        self.assertIsNone(report["gene_known_became_novel"])
        self.assertIsNone(report["gene_novel_became_known"])
        self.assertIsNone(report["relaxed_fingerprint_changed"])

    def test_a_stage_cannot_resurrect_a_gene_an_earlier_one_dropped(self):
        structures = pd.DataFrame(
            {
                "has_structure": [True],
                "valid_structure": [False],
                "unique_structure": [True],
                "novel_structure": [True],
                "e_above_hull": [-0.5],
            },
            index=[0],
        )
        report = funnel(self._screen(), structures)
        self.assertEqual(report["novel_structure"], 0)
        self.assertEqual(report["stable"], 0)

    def test_missing_columns_report_none_rather_than_assuming_success(self):
        structures = pd.DataFrame({"has_structure": [True]}, index=[0])
        report = funnel(self._screen(), structures)
        self.assertEqual(report["structure"], 1)
        self.assertIsNone(report["valid_structure"])
        self.assertIsNone(report["metasun_per_sampled_gene"])


if __name__ == "__main__":
    unittest.main()
