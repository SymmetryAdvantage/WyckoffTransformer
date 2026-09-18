"""Tests for the de novo ranking protocol."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pandas as pd
import pytest

from wyckoff_transformer.cli.protocol import (
    _budget,
    _generate_one,
    PYXTAL_COLUMNS,
    RELAXATION_COLUMNS,
    RowLog,
    TEMPLATE_TRIAL,
    Timeout,
    LOGM_ROUNDOFF,
    DeviceMemoryBudget,
    _init_relax_worker,
    _pin_visible_device,
    _quiet_logm_roundoff,
    aggregate_structures,
    claim_device,
    build_parser,
    estimate_relaxation_memory,
    resolve_device_budgets,
    resolve_devices,
    time_limit,
)
from wyckoff_transformer.cryspr.generator import DEFAULT_PYXTAL_TOL_FACTOR
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

    def test_mace_mp_hull_is_paired_with_mace_mp_0b3(self):
        # Named outright rather than left to mace-torch's version-dependent
        # mace_mp(model=None) alias, which is not what built the hull.
        self.assertEqual(HULL_MLIPS["mace_mp"].checkpoint, "MACE-MP-0b3")

    def test_mace_checkpoint_urls_match_mace_torch_registry(self):
        """The hull pairings were identified against mace-torch's own URLs."""
        foundations = pytest.importorskip("mace.calculators.foundations_models")
        from wyckoff_transformer.cryspr.mace_urls import MODEL_URLS

        for name, alias in (
            (HULL_MLIPS["mace_mp"].checkpoint, "medium-0b3"),
            (HULL_MLIPS["mace_omat"].checkpoint, "medium-omat-0"),
        ):
            self.assertEqual(MODEL_URLS[name], foundations.mace_mp_urls[alias])

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
        # a pool reports only as a worker that dies at startup.  Both the relax
        # and the prescreen stage build their tuple positionally, and the
        # prescreen one passes None for the scoring potential.
        import inspect

        self.assertEqual(
            list(inspect.signature(_init_relax_worker).parameters),
            ["counter", "slots", "mlip", "prerelax_mlip", "debug", "budgets"],
        )


class TestEstimateRelaxationMemory(unittest.TestCase):
    def test_minimum_floor(self):
        # Very small structures (N <= 13) get the minimum floor of 600 MB
        atoms_small = MagicMock()
        atoms_small.__len__.return_value = 4
        self.assertEqual(estimate_relaxation_memory(atoms_small), 600)

        atoms_13 = MagicMock()
        atoms_13.__len__.return_value = 13
        self.assertEqual(estimate_relaxation_memory(atoms_13), 600)

    def test_linear_scaling(self):
        # M(N) = max(600, 400 + 15 * N)
        atoms_50 = MagicMock()
        atoms_50.__len__.return_value = 50
        self.assertEqual(estimate_relaxation_memory(atoms_50), 400 + 15 * 50)  # 1150

        atoms_100 = MagicMock()
        atoms_100.__len__.return_value = 100
        self.assertEqual(estimate_relaxation_memory(atoms_100), 400 + 15 * 100)  # 1900

        atoms_200 = MagicMock()
        atoms_200.__len__.return_value = 200
        self.assertEqual(estimate_relaxation_memory(atoms_200), 400 + 15 * 200)  # 3400


class TestDeviceMemoryBudget(unittest.TestCase):
    def test_budget_properties(self):
        budget = DeviceMemoryBudget("cuda:0", 4000)
        self.assertEqual(budget.device, "cuda:0")
        self.assertEqual(budget.total_mb, 4000)
        self.assertEqual(budget.available_mb, 4000)

    def test_reserve_deducts_and_restores(self):
        budget = DeviceMemoryBudget("cuda:0", 4000)
        with budget.reserve(1200, label="test"):
            self.assertEqual(budget.available_mb, 2800)
        self.assertEqual(budget.available_mb, 4000)

    def test_reserve_restores_on_exception(self):
        budget = DeviceMemoryBudget("cuda:0", 4000)
        with self.assertRaises(RuntimeError):
            with budget.reserve(1500, label="error_test"):
                self.assertEqual(budget.available_mb, 2500)
                raise RuntimeError("something went wrong")
        self.assertEqual(budget.available_mb, 4000)

    def test_oversized_job_allowed_when_idle(self):
        # A structure requiring 5000 MB on a 4000 MB device should be allowed to run
        # when the card is idle, rather than deadlocking forever.
        budget = DeviceMemoryBudget("cuda:0", 4000)
        with budget.reserve(5000, label="oversized"):
            self.assertEqual(budget.available_mb, 0)
        self.assertEqual(budget.available_mb, 4000)

    def test_concurrent_reservations_serialize_when_over_budget(self):
        import time
        import threading

        budget = DeviceMemoryBudget("cuda:0", 4000)
        order = []

        def heavy_job_1():
            with budget.reserve(2500, label="heavy1"):
                order.append("heavy1_start")
                time.sleep(0.05)
                order.append("heavy1_end")

        def heavy_job_2():
            # Wait a tiny moment to ensure heavy1 grabs the lock first
            time.sleep(0.01)
            with budget.reserve(2500, label="heavy2"):
                order.append("heavy2_start")
                order.append("heavy2_end")

        t1 = threading.Thread(target=heavy_job_1)
        t2 = threading.Thread(target=heavy_job_2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Because 2500 + 2500 = 5000 > 4000, heavy2 must wait until heavy1 ends
        self.assertEqual(order, ["heavy1_start", "heavy1_end", "heavy2_start", "heavy2_end"])

    def test_concurrent_light_jobs_run_in_parallel(self):
        import time
        import threading

        budget = DeviceMemoryBudget("cuda:0", 4000)
        active_counts = []
        lock = threading.Lock()
        active = 0

        def light_job():
            nonlocal active
            with budget.reserve(1000, label="light"):
                with lock:
                    active += 1
                    active_counts.append(active)
                time.sleep(0.02)
                with lock:
                    active -= 1

        threads = [threading.Thread(target=light_job) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At least one point in time had > 1 active concurrent worker
        self.assertGreater(max(active_counts), 1)
        self.assertEqual(budget.available_mb, 4000)

    def test_resolve_device_budgets(self):
        slots = ["cuda:0", "cuda:0", "cuda:1"]
        budgets = resolve_device_budgets(slots, device_budget_overrides={"cuda:0": 3500})
        self.assertIn("cuda:0", budgets)
        self.assertIn("cuda:1", budgets)
        self.assertEqual(budgets["cuda:0"].total_mb, 3500)
        self.assertGreater(budgets["cuda:1"].total_mb, 0)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    def test_reserve_calls_empty_cache_on_cuda(self, mock_empty_cache, mock_cuda_avail):
        budget = DeviceMemoryBudget("cuda:0", 4000)
        with budget.reserve(1000):
            pass
        mock_empty_cache.assert_called_once()

    @patch("torch.cuda.empty_cache")
    def test_reserve_does_not_call_empty_cache_on_cpu(self, mock_empty_cache):
        budget = DeviceMemoryBudget("cpu", 4000)
        with budget.reserve(1000):
            pass
        mock_empty_cache.assert_not_called()


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

    def test_a_colliding_entry_without_geometry_is_refused_not_scored_novel(self):
        # Dropped silently, the fingerprint would lose its only candidate and
        # every structure on it would read as novel.
        from pymatgen.core import Lattice, Structure

        from wyckoff_transformer.evaluation import structure_novelty
        from wyckoff_transformer.evaluation.structure_novelty import (
            UnresolvedReferenceError,
            build_novelty_reference,
        )

        silicon = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
        hits = {"F": ["mp-1"], "G": ["agm000000002"]}
        with patch.object(structure_novelty, "collect_reference_ids", return_value=hits), \
                patch.object(structure_novelty, "load_reference_structures",
                             return_value={"mp-1": silicon}):
            with self.assertRaisesRegex(UnresolvedReferenceError, "1 of 2 .*agm000000002"):
                build_novelty_reference(["F", "G"])

    def test_every_colliding_entry_with_geometry_builds_the_reference(self):
        from pymatgen.core import Lattice, Structure

        from wyckoff_transformer.evaluation import structure_novelty
        from wyckoff_transformer.evaluation.structure_novelty import build_novelty_reference

        silicon = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
        with patch.object(structure_novelty, "collect_reference_ids",
                          return_value={"F": ["mp-1", "mp-2"]}), \
                patch.object(structure_novelty, "load_reference_structures",
                             return_value={"mp-1": silicon, "mp-2": silicon.copy()}):
            reference = build_novelty_reference(["F"])
        self.assertEqual(sorted(reference.index), ["mp-1", "mp-2"])


class TestReferenceChoice(unittest.TestCase):
    """Novelty is judged against the current LeMat-Bulk variant, and only one of them."""

    CU = {"group": 225, "species": ["Cu"], "numIons": [4], "sites": [["4a"]]}

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)

    def test_the_default_reference_is_the_current_lemat_variant(self):
        # CLAUDE.md: lemat_bulk_fmax1_stress is the only current variant; the
        # others must not be used for new evaluation.
        from wyckoff_transformer.cli import protocol_wandb
        from wyckoff_transformer.evaluation.protocol import DEFAULT_REFERENCE_CACHE

        self.assertEqual(DEFAULT_REFERENCE_CACHE,
                         Path("cache/lemat_bulk_fmax1_stress/data.pkl.gz"))
        for parser, argv in (
            (build_parser(), ["genes.json", "--output-dir", "out"]),
            (protocol_wandb.build_parser(), ["run", "--output-dir", "out"]),
        ):
            args = parser.parse_args(argv)
            self.assertEqual(args.reference_cache, DEFAULT_REFERENCE_CACHE)
            self.assertIsNone(args.reference_fingerprint_cache)

    def test_the_template_index_lives_beside_the_default_reference(self):
        from wyckoff_transformer.cryspr.template import DEFAULT_INDEX_PATH
        from wyckoff_transformer.evaluation.protocol import DEFAULT_REFERENCE_CACHE

        self.assertEqual(DEFAULT_INDEX_PATH.parent, DEFAULT_REFERENCE_CACHE.parent)

    def test_the_fingerprint_cache_follows_the_reference_and_its_splits(self):
        from wyckoff_transformer.evaluation.protocol import default_fingerprint_cache

        cache = Path("cache/some_variant/data.pkl.gz")
        self.assertEqual(default_fingerprint_cache(cache, ("train", "val", "test")),
                         Path("cache/some_variant/gene_fingerprints.pkl.gz"))
        self.assertEqual(default_fingerprint_cache(cache, ("train",)),
                         Path("cache/some_variant/gene_fingerprints_train.pkl.gz"))

    def _screen(self, reference_cache, fingerprint_cache=None):
        from wyckoff_transformer.cli.protocol import stage_screen

        genes = self.out / "genes.json"
        genes.write_text(json.dumps([self.CU]), encoding="utf-8")
        args = SimpleNamespace(
            input=genes, output_dir=self.out, reference_cache=reference_cache,
            reference_splits="train,val,test",
            reference_fingerprint_cache=fingerprint_cache,
        )
        with patch("wyckoff_transformer.cli.protocol.load_reference_fingerprints",
                   return_value=set()) as load:
            stage_screen(args)
        return load

    def test_the_screen_loads_the_fingerprints_of_the_reference_it_was_given(self):
        load = self._screen(Path("cache/variant_b/data.pkl.gz"))
        self.assertEqual(load.call_args.kwargs["fingerprint_cache"],
                         Path("cache/variant_b/gene_fingerprints.pkl.gz"))
        explicit = self._screen(Path("cache/variant_b/data.pkl.gz"), Path("/elsewhere.pkl.gz"))
        self.assertEqual(explicit.call_args.kwargs["fingerprint_cache"],
                         Path("/elsewhere.pkl.gz"))

    def test_the_screen_records_its_reference(self):
        from wyckoff_transformer.cli.protocol import LINEAGE_KEY, MANIFEST_FILE, SCREEN_FILE

        self._screen(Path("cache/variant_b/data.pkl.gz"))
        record = json.loads((self.out / MANIFEST_FILE).read_text())[LINEAGE_KEY][SCREEN_FILE]
        self.assertEqual(record["reference"],
                         {"cache": "variant_b/data.pkl.gz", "splits": ["train", "val", "test"]})
        self.assertEqual(record["reference_fingerprints"], 0)

    def test_score_accepts_the_screens_reference_however_its_path_is_spelled(self):
        from wyckoff_transformer.cli.protocol import require_screen_reference

        self._screen(Path("cache/variant_b/data.pkl.gz"))
        require_screen_reference(
            self.out, Path("/mnt/store/cache/variant_b/data.pkl.gz"), ("train", "val", "test"))

    def test_score_refuses_a_screen_of_another_reference(self):
        from wyckoff_transformer.cli.protocol import StaleOutputError, require_screen_reference

        self._screen(Path("cache/variant_a/data.pkl.gz"))
        with self.assertRaisesRegex(StaleOutputError, "variant_a.*variant_b"):
            require_screen_reference(
                self.out, Path("cache/variant_b/data.pkl.gz"), ("train", "val", "test"))
        with self.assertRaisesRegex(StaleOutputError, r"\(train\)"):
            require_screen_reference(self.out, Path("cache/variant_a/data.pkl.gz"), ("train",))

    def test_score_refuses_a_screen_that_predates_the_record(self):
        from wyckoff_transformer.cli.protocol import (
            SCREEN_FILE, StaleOutputError, _write_lineage, require_screen_reference,
        )

        with self.assertRaisesRegex(StaleOutputError, "records no reference"):
            require_screen_reference(
                self.out, Path("cache/variant_b/data.pkl.gz"), ("train", "val", "test"))
        _write_lineage(self.out, SCREEN_FILE, {"id": "s", "parent": "genes:sha256:x"})
        with self.assertRaisesRegex(StaleOutputError, "lemat_bulk_ehull"):
            require_screen_reference(
                self.out, Path("cache/variant_b/data.pkl.gz"), ("train", "val", "test"))


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

    def test_the_template_stage_is_runnable_but_not_part_of_all(self):
        from wyckoff_transformer.cli.protocol import STAGES

        args = build_parser().parse_args(
            ["genes.json", "--output-dir", "out", "--stage", "template"]
        )
        self.assertEqual(args.stage, "template")
        # `--stage all` runs STAGES, so leaving `template` out of it is what
        # keeps the default protocol a random-start protocol.
        self.assertNotIn("template", STAGES)

    def test_the_template_trial_cannot_collide_with_a_scheduled_one(self):
        """The template start shares pyxtal.csv with the random draws.

        A schedule that ever allotted TEMPLATE_TRIAL+1 trials to a gene would
        make (index, trial) ambiguous, and `--resume` would then treat a random
        trial as already done because the template one is.
        """
        schedule = parse_trial_schedule(DEFAULT_TRIAL_SCHEDULE)
        self.assertLess(max(trials for _, trials in schedule), TEMPLATE_TRIAL)

    def test_template_candidate_default_leaves_room_for_a_retry(self):
        args = build_parser().parse_args(["genes.json", "--output-dir", "out"])
        self.assertGreater(args.template_candidates, 1)
        self.assertIsNone(args.template_index)  # built from the reference cache

    def test_the_nep89_variants_are_off_by_default(self):
        """Every published number was measured with the single-stage arm.

        A variant that changed a default would invalidate the trial schedule,
        the stage design and the funnel rates all at once, and silently.
        """
        args = build_parser().parse_args(["genes.json", "--output-dir", "out"])
        self.assertIsNone(args.prerelax_mlip)
        self.assertEqual(args.trial_multiplier, 1)
        self.assertEqual(args.relax_from, "pyxtal")

    def test_the_prescreen_stage_is_runnable_but_not_part_of_all(self):
        from wyckoff_transformer.cli.protocol import OPTIONAL_STAGES, STAGES

        args = build_parser().parse_args(
            ["genes.json", "--output-dir", "out", "--stage", "prescreen"]
        )
        self.assertEqual(args.stage, "prescreen")
        self.assertNotIn("prescreen", STAGES)
        self.assertIn("prescreen", OPTIONAL_STAGES)

    def test_the_pre_relaxation_potential_need_not_have_a_published_hull(self):
        """Which is exactly why it is resolved through a separate registry.

        NEP89 has no LeMat-Bulk hull, so --mlip must refuse it; the
        pre-relaxation computes no reported energy, so --prerelax-mlip must not.
        """
        args = build_parser().parse_args(
            ["genes.json", "--output-dir", "out", "--prerelax-mlip", "nep89"]
        )
        self.assertEqual(args.prerelax_mlip, "nep89")
        self.assertNotIn("nep89", HULL_MLIPS)
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["genes.json", "--output-dir", "out", "--mlip", "nep89"]
            )

    def test_the_pre_relaxation_converges_more_loosely_than_the_scoring_one(self):
        args = build_parser().parse_args(["genes.json", "--output-dir", "out"])
        self.assertGreater(args.prerelax_fmax, args.fmax)
        self.assertGreater(args.prescreen_fmax, args.fmax)

    def test_generation_and_relaxation_have_their_own_timeouts(self):
        """Neither stage may be held hostage by one gene it cannot finish."""
        args = build_parser().parse_args(["genes.json", "--output-dir", "out"])
        self.assertEqual(args.pyxtal_timeout, 300.0)
        self.assertEqual(args.relax_timeout, 300.0)
        self.assertIsNone(args.pyxtal_cores)  # every core

    def test_the_pyxtal_distance_floor_is_unchanged_by_default(self):
        """Every published number was drawn under factor 1.3.

        The option exists so the floor can be *measured*
        (docs/pyxtal_tolerance_sweep.md); a changed default would silently
        redefine what the protocol's draws are.
        """
        args = build_parser().parse_args(["genes.json", "--output-dir", "out"])
        self.assertEqual(args.pyxtal_tol_factor, 1.3)
        self.assertEqual(args.pyxtal_tol_factor, DEFAULT_PYXTAL_TOL_FACTOR)

    def test_a_more_permissive_floor_can_be_asked_for(self):
        args = build_parser().parse_args(
            ["genes.json", "--output-dir", "out", "--pyxtal-tol-factor", "0.4"]
        )
        self.assertEqual(args.pyxtal_tol_factor, 0.4)

    def test_resume_is_on_by_default(self):
        # Both per-trial logs are written row by row, so repeating finished
        # trials is pure waste; --no-resume is the deliberate fresh start.
        args = build_parser().parse_args(["genes.json", "--output-dir", "out"])
        self.assertTrue(args.resume)
        args = build_parser().parse_args(
            ["genes.json", "--output-dir", "out", "--no-resume"]
        )
        self.assertFalse(args.resume)


class TestPyxtalTolFactorReachesTheDraw(unittest.TestCase):
    """The flag is worthless if it stops at the parser.

    `--pyxtal-tol-factor` crosses a process boundary -- `stage_generate` hands
    it to a spawned pool worker, which builds the tolerance matrix itself
    because a `Tol_matrix` would have to be pickled otherwise -- so the value
    arriving at `single_pyxtal` is worth asserting on directly.
    """

    _GENE = {"group": 225, "species": ["Na", "Cl"], "numIons": [4, 4],
             "sites": [["4a"], ["4b"]]}

    def _iadm_seen_by_pyxtal(self, *args_to_generate_one):
        with patch("wyckoff_transformer.cryspr.generator.single_pyxtal") as mock:
            mock.return_value = None
            _generate_one(*args_to_generate_one)
        return mock.call_args.kwargs["iadm"]

    def test_the_default_draw_uses_the_shipped_floor(self):
        # Identity rather than a factor attribute: PyXtal stores the factor
        # halved (`Tol_matrix.f`), so comparing against the cached matrix for a
        # known factor says what is meant without depending on that.
        from wyckoff_transformer.cryspr.generator import pyxtal_tol_matrix

        with tempfile.TemporaryDirectory() as tmp:
            iadm = self._iadm_seen_by_pyxtal(0, 0, self._GENE, tmp, None)
        self.assertIs(iadm, pyxtal_tol_matrix(DEFAULT_PYXTAL_TOL_FACTOR))

    def test_a_permissive_factor_reaches_pyxtal(self):
        from wyckoff_transformer.cryspr.generator import pyxtal_tol_matrix

        with tempfile.TemporaryDirectory() as tmp:
            iadm = self._iadm_seen_by_pyxtal(0, 0, self._GENE, tmp, None, 0.4)
        self.assertIs(iadm, pyxtal_tol_matrix(0.4))
        self.assertIsNot(iadm, pyxtal_tol_matrix(DEFAULT_PYXTAL_TOL_FACTOR))

    def test_the_generate_stage_records_the_factor_it_drew_under(self):
        """Nothing else in a run's output says which floor produced its CIFs."""
        from wyckoff_transformer.cli.protocol import MANIFEST_FILE, stage_generate

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "genes.json").write_text(json.dumps([self._GENE]), encoding="utf-8")
            (out / "screen.json").write_text(
                json.dumps({
                    "n_sampled": 1, "valid": [0], "invalid": [],
                    "invalid_reason": {}, "counts": {"0": 1},
                    "novel": [0], "known": [],
                }),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                input=out / "genes.json", output_dir=out, limit=None,
                n_trials="1", trial_multiplier=1, resume=False, retry_failed=False,
                pyxtal_cores=1, pyxtal_timeout=60.0, pyxtal_tol_factor=0.4,
                debug=False,
            )
            stage_generate(args)
            manifest = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
        self.assertEqual(manifest["pyxtal_tol_factor"], 0.4)

    def test_the_wandb_wrapper_mirrors_the_flag(self):
        """It builds the stage arguments itself, so a missing key is an
        AttributeError inside `stage_generate` rather than a parser error."""
        from wyckoff_transformer.cli.protocol_wandb import (
            build_parser as build_wandb_parser,
            build_stage_args,
        )

        parser = build_wandb_parser()
        default = parser.parse_args(["run-id", "--output-dir", "out"])
        self.assertEqual(default.pyxtal_tol_factor, DEFAULT_PYXTAL_TOL_FACTOR)
        asked = parser.parse_args(
            ["run-id", "--output-dir", "out", "--pyxtal-tol-factor", "0.7"]
        )
        self.assertEqual(
            build_stage_args(asked, Path("genes.json")).pyxtal_tol_factor, 0.7
        )


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

    def test_resuming_retries_tasks_that_failed_due_to_worker_crash(self):
        log = RowLog(self.path, RELAXATION_COLUMNS, resume=False)
        log.write({"index": 1, "trial": 0, "status": "ok"})
        log.write({"index": 2, "trial": 0, "status": "failed", "error": "BrokenProcessPool: worker crashed"})
        log.write({"index": 3, "trial": 0, "status": "failed", "error": "ValueError: clash"})
        log.close()

        resumed = RowLog(self.path, RELAXATION_COLUMNS, resume=True)
        # index 1 succeeded, index 3 failed chemically (so done), index 2 was a worker crash (so retried)
        self.assertEqual(resumed.done, {(1, 0), (3, 0)})
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


class TestTrialMultiplier(unittest.TestCase):
    """The wide arm widens the *draw* budget and nothing else."""

    GENE = {"group": 225, "species": ["Na", "Cl"], "numIons": [4, 4],
            "sites": [["4a"], ["4b"]]}

    def _schedule(self):
        return parse_trial_schedule(DEFAULT_TRIAL_SCHEDULE)

    def test_the_default_multiplier_changes_nothing(self):
        dof, trials = _budget(self.GENE, self._schedule())
        self.assertEqual((dof, trials), _budget(self.GENE, self._schedule(), 1))

    def test_the_multiplier_scales_the_schedule_rather_than_replacing_it(self):
        """So the extra draws stay proportional to the free coordinates.

        A flat "draw 30 of everything" would spend the same budget on the fifth
        of genes with no free coordinate at all, where the schedule already
        establishes that a second draw provably changes nothing.
        """
        schedule = self._schedule()
        _, one = _budget(self.GENE, schedule, 1)
        _, ten = _budget(self.GENE, schedule, 10)
        self.assertEqual(ten, one * 10)

    def test_a_zero_multiplier_is_refused(self):
        with self.assertRaises(ValueError):
            _budget(self.GENE, self._schedule(), 0)

    def test_the_prescreen_selects_the_unmultiplied_budget_back_down(self):
        """The arm's whole claim: the expensive relaxation count is unchanged."""
        schedule = self._schedule()
        _, drawn = _budget(self.GENE, schedule, 10)
        _, selected = _budget(self.GENE, schedule)  # what _prescreen_select uses
        self.assertEqual(drawn, 10 * selected)


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
        # The rattle stage fills the fixed-symmetry slot with what it was handed;
        # stepwise_relax_stages overwrites it with the symmetric stages' output.
        assert kept.fixed_symmetry is before
        assert kept.rattled is after
        assert kept.rattle_accepted is (kept.kept is after)
        return kept.kept is after

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


class TestStageWorkerFailure(unittest.TestCase):
    def setUp(self):
        import json
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    @staticmethod
    def _pool_that_loses_every_trial(error="BrokenProcessPool: Process died"):
        """A stand-in for run_supervised whose every trial was lost to its worker."""
        from wyckoff_transformer.cli.worker_pool import PoolReport

        def fake(tasks, fn, *, on_failure, **kwargs):
            tasks = list(tasks)
            for key, _ in tasks:
                on_failure(key, "failed", error)
            return PoolReport(rounds=1, pool_breaks=1, unanswered=len(tasks))

        return fake

    def test_stage_relax_raises_when_all_workers_fail(self):
        from ase.build import bulk
        from ase.io import write as ase_write
        from wyckoff_transformer.cli.protocol import (
            IncompleteStageError, stage_relax, PYXTAL_FILE, PYXTAL_TRIALS_FILE,
            RELAXATIONS_FILE, RELAXATION_COLUMNS, read_rows,
        )

        atoms = bulk("Cu", "fcc", a=3.6)
        atoms.info = {"gene": 0, "trial": 0}
        ase_write(str(self.out / PYXTAL_FILE), atoms, format="extxyz")
        pd.DataFrame([{"index": 0, "trial": 0, "status": "ok", "dof_positional": 0, "n_trials": 1, "seconds": 0.1}], columns=list(PYXTAL_COLUMNS)).to_csv(self.out / PYXTAL_TRIALS_FILE, index=False)

        args = SimpleNamespace(
            output_dir=self.out,
            mlip="orb_conserv_inf",
            cores=1,
            devices=None,
            workers_per_device=1,
            limit=None,
            resume=False,
            fmax=0.05,
            release_symmetry=True,
            rattle=True,
            relax_timeout=10.0,
            debug=False,
        )

        with patch("wyckoff_transformer.cli.protocol.run_supervised",
                   side_effect=self._pool_that_loses_every_trial()):
            with self.assertRaises(IncompleteStageError) as ctx:
                stage_relax(args)
        self.assertIn("1 in relaxations.csv", str(ctx.exception))
        # Written down, and in a form --resume re-runs.
        relax_df = read_rows(self.out / RELAXATIONS_FILE, RELAXATION_COLUMNS)
        self.assertEqual(len(relax_df), 1)
        self.assertEqual(RowLog(self.out / RELAXATIONS_FILE, RELAXATION_COLUMNS,
                                resume=True).done, set())

    def test_stage_relax_hands_the_supervisor_its_timeout(self):
        """The hang deadline is derived from --relax-timeout; see test_worker_pool."""
        from ase.build import bulk
        from ase.io import write as ase_write
        from wyckoff_transformer.cli.protocol import stage_relax, PYXTAL_FILE, PYXTAL_TRIALS_FILE

        atoms = bulk("Cu", "fcc", a=3.6)
        atoms.info = {"gene": 0, "trial": 0}
        ase_write(str(self.out / PYXTAL_FILE), atoms, format="extxyz")
        pd.DataFrame([{"index": 0, "trial": 0, "status": "ok", "dof_positional": 0, "n_trials": 1, "seconds": 0.1}], columns=list(PYXTAL_COLUMNS)).to_csv(self.out / PYXTAL_TRIALS_FILE, index=False)
        args = SimpleNamespace(
            output_dir=self.out, mlip="orb_conserv_inf", cores=1, devices=None,
            workers_per_device=1, limit=None, resume=False, fmax=0.05,
            release_symmetry=True, rattle=True, relax_timeout=1234.0, debug=False,
            allow_incomplete=True,
        )
        with patch("wyckoff_transformer.cli.protocol.run_supervised",
                   side_effect=self._pool_that_loses_every_trial()) as supervised:
            stage_relax(args)
        self.assertEqual(supervised.call_args.kwargs["task_timeout"], 1234.0)

    def test_shutdown_pool_terminates_workers_before_shutdown_clears_processes(self):
        from wyckoff_transformer.cli.worker_pool import shutdown_pool

        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.is_alive.return_value = True

        mock_pool = MagicMock()
        mock_pool._processes = {99999: mock_proc}

        def fake_shutdown(*args, **kwargs):
            mock_pool._processes = None

        mock_pool.shutdown.side_effect = fake_shutdown

        shutdown_pool(mock_pool, grace=0.01)

        mock_pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()

    def test_stage_generate_raises_when_all_workers_fail(self):
        import json
        from wyckoff_transformer.cli.protocol import IncompleteStageError, stage_generate, SCREEN_FILE

        genes_file = self.out / "genes.json"
        genes_file.write_text(
            json.dumps([{"group": 225, "species": ["Cu"], "numIons": [4], "sites": [["4a"]]}]),
            encoding="utf-8",
        )
        screen = GeneScreen(n_sampled=1, valid=[0], invalid=[], counts={0: 1}, novel=[0], known=[])
        write_screen(screen, self.out / SCREEN_FILE)

        args = SimpleNamespace(
            input=genes_file,
            output_dir=self.out,
            pyxtal_cores=1,
            pyxtal_timeout=10.0,
            pyxtal_tol_factor=1.3,
            n_trials="1",
            limit=None,
            resume=False,
            debug=False,
        )

        with patch("wyckoff_transformer.cli.protocol.run_supervised",
                   side_effect=self._pool_that_loses_every_trial()):
            with self.assertRaises(IncompleteStageError) as ctx:
                stage_generate(args)
        self.assertIn("1 in pyxtal.csv", str(ctx.exception))

    def test_stage_score_refuses_when_all_relaxations_failed_with_worker_errors(self):
        from wyckoff_transformer.cli.protocol import (
            IncompleteStageError,
            stage_score,
            SCREEN_FILE,
            STRUCTURES_FILE,
            RELAXATIONS_FILE,
            RELAXATION_COLUMNS,
        )

        screen = GeneScreen(n_sampled=1, valid=[0], invalid=[], counts={0: 1}, novel=[0], known=[])
        write_screen(screen, self.out / SCREEN_FILE)
        pd.DataFrame([{"index": 0, "has_structure": False, "error": "failed"}]).to_csv(
            self.out / STRUCTURES_FILE
        )
        pd.DataFrame(
            [{"index": 0, "trial": 0, "status": "failed", "error": "BrokenProcessPool: worker died"}],
            columns=list(RELAXATION_COLUMNS),
        ).to_csv(self.out / RELAXATIONS_FILE, index=False)

        args = SimpleNamespace(
            input=self.out / "genes.json",
            output_dir=self.out,
            mlip="orb_conserv_inf",
        )
        with self.assertRaises(IncompleteStageError) as ctx:
            stage_score(args)
        self.assertIn("1 in relaxations.csv", str(ctx.exception))

    def test_quiet_cif_parser_warnings_prints_at_most_once(self):
        import warnings
        from wyckoff_transformer.cli.protocol import _quiet_cif_parser_warnings

        calls = []
        original = lambda msg, cat, fname, lineno, file=None, line=None: calls.append(str(msg))
        with patch("warnings.showwarning", original):
            _quiet_cif_parser_warnings()
            warnings.warn("Issues encountered while parsing CIF: 4 coords rounded", UserWarning)
            warnings.warn("Issues encountered while parsing CIF: 2 coords rounded", UserWarning)
            warnings.warn("Issues encountered while parsing CIF: 8 coords rounded", UserWarning)

        self.assertEqual(len(calls), 1)
        self.assertIn("4 coords rounded", calls[0])


if __name__ == "__main__":
    unittest.main()


class TestFixedSymmetryReadout(unittest.TestCase):
    """The funnel reports the gene screen and both structure readouts, nested.

    The symmetry release and the rattle lower energy, which is why they are on
    by default. They can also discard the Wyckoff orbits WyFormer predicted and
    relax a novel structure onto a known one, and neither shows up in the kept
    structure's own numbers.
    """

    @staticmethod
    def _screen(n=4):
        return GeneScreen(
            n_sampled=n, valid=list(range(n)), counts={i: 1 for i in range(n)},
            novel=list(range(n)), known=[],
        )

    @staticmethod
    def _frame(rows):
        return pd.DataFrame(rows).set_index("index")

    def test_the_two_readouts_are_reported_in_separate_sections(self):
        free = self._frame([
            {"index": 0, "has_structure": True, "valid_structure": True,
             "unique_structure": True, "novel_structure": True, "e_above_hull": 0.02},
            # gene 1: relaxing past the fixed symmetry made it a known structure
            {"index": 1, "has_structure": True, "valid_structure": True,
             "unique_structure": True, "novel_structure": False, "e_above_hull": 0.01},
        ])
        fixed = self._frame([
            {"index": 0, "has_structure": True, "valid_structure": True,
             "unique_structure": True, "novel_structure": True, "e_above_hull": 0.05},
            {"index": 1, "has_structure": True, "valid_structure": True,
             "unique_structure": True, "novel_structure": True, "e_above_hull": 0.04},
        ])
        report = funnel(self._screen(2), free, fixed)

        self.assertEqual(set(report), {"gene", "fixed_symmetry", "free"})
        self.assertEqual(report["free"]["metasun_per_sampled_gene"], 0.5)
        self.assertEqual(report["fixed_symmetry"]["metasun_per_sampled_gene"], 1.0)
        # The gene screen belongs to neither readout and is reported once.
        self.assertIn("valid_gene_rate", report["gene"])
        self.assertNotIn("valid_gene_rate", report["free"])
        self.assertNotIn("valid_gene_rate", report["fixed_symmetry"])

    def test_the_report_is_plain_json(self):
        free = self._frame([
            {"index": 0, "has_structure": True, "valid_structure": True,
             "unique_structure": True, "novel_structure": True, "e_above_hull": 0.02},
        ])
        report = funnel(self._screen(1), free, free)
        self.assertEqual(json.loads(json.dumps(report)), dict(report))

    def test_a_run_without_the_fixed_symmetry_structures_reports_nulls(self):
        """A cohort relaxed before the fixed-symmetry CIFs existed must stay readable."""
        free = self._frame([
            {"index": 0, "has_structure": True, "valid_structure": True,
             "unique_structure": True, "novel_structure": True, "e_above_hull": 0.02},
        ])
        report = funnel(self._screen(1), free)
        self.assertIsNone(report["fixed_symmetry"]["structure"])
        self.assertIsNone(report["fixed_symmetry"]["metasun_per_sampled_gene"])
        self.assertEqual(report["free"]["structure"], 1)

    def test_flat_lookups_read_the_free_readout(self):
        """Scripts written against the flat funnel still read the kept structure."""
        free = self._frame([
            {"index": 0, "has_structure": True, "valid_structure": True,
             "unique_structure": True, "novel_structure": True, "e_above_hull": 0.02},
        ])
        fixed = self._frame([
            {"index": 0, "has_structure": True, "valid_structure": True,
             "unique_structure": True, "novel_structure": False, "e_above_hull": 0.02},
        ])
        report = funnel(self._screen(1), free, fixed)
        for key in ("metasun_per_sampled_gene", "sun_per_sampled_gene",
                    "novel_structure_per_sampled_gene", "metastable", "sampled"):
            self.assertIn(key, report)
        self.assertEqual(report["metasun_per_sampled_gene"], 1.0)
        self.assertEqual(report.get("sampled"), 1)


class TestAggregateBothReadouts(unittest.TestCase):
    """Each readout picks its own lowest-energy trial and keeps its own CIF."""

    def test_the_fixed_symmetry_winner_can_be_a_different_trial(self):
        import csv

        from wyckoff_transformer.cli.protocol import (
            CIF_DIR, CIF_FIXED_DIR, PYXTAL_TRIALS_FILE, RELAXATIONS_FILE,
            RELAXATION_COLUMNS, STRUCTURES_FILE, STRUCTURES_FIXED_FILE,
            aggregate_structures,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pd.DataFrame([
                {"index": 0, "trial": t, "status": "ok", "formula": "Na2",
                 "n_atoms": 2, "dof_positional": 3, "n_trials": 2, "seconds": 1.0,
                 "error": ""}
                for t in (0, 1)
            ], columns=list(PYXTAL_COLUMNS)).to_csv(out / PYXTAL_TRIALS_FILE, index=False)
            rows = []
            for trial, (free, fixed) in enumerate(((-10.0, -8.0), (-9.0, -8.5))):
                kept, sym = out / f"t{trial}_kept.cif", out / f"t{trial}_fixed_symmetry.cif"
                kept.write_text(f"kept {trial}")
                sym.write_text(f"fixed {trial}")
                rows.append({
                    "index": 0, "trial": trial, "status": "ok", "formula": "Na2",
                    "energy": free, "energy_per_atom": free / 2, "n_atoms": 2,
                    "device": "cpu", "seconds": 1.0, "cif": str(kept), "error": "",
                    "status_fixed": "ok", "energy_fixed": fixed,
                    "energy_per_atom_fixed": fixed / 2, "cif_fixed": str(sym),
                })
            with (out / RELAXATIONS_FILE).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(RELAXATION_COLUMNS))
                writer.writeheader()
                writer.writerows(rows)

            aggregate_structures(out)

            free = pd.read_csv(out / STRUCTURES_FILE, index_col="index")
            fixed = pd.read_csv(out / STRUCTURES_FIXED_FILE, index_col="index")
            self.assertEqual(int(free.loc[0, "best_trial"]), 0)
            self.assertEqual(int(fixed.loc[0, "best_trial"]), 1)
            self.assertEqual(float(fixed.loc[0, "energy"]), -8.5)
            self.assertEqual((out / CIF_DIR / "0.cif").read_text(), "kept 0")
            self.assertEqual((out / CIF_FIXED_DIR / "0.cif").read_text(), "fixed 1")


class TestBasinHopTrialKeys(unittest.TestCase):
    """A walk visits many minima, so its keys must not collide or drift."""

    def test_keys_are_unique_across_walks_and_hops(self):
        from wyckoff_transformer.cli.protocol import (
            BASINHOP_TRIAL_BASE,
            BASINHOP_TRIAL_STRIDE,
        )

        keys = {
            BASINHOP_TRIAL_BASE + trial * BASINHOP_TRIAL_STRIDE + hop
            for trial in range(30)
            for hop in range(200)
        }
        self.assertEqual(len(keys), 30 * 200)

    def test_keys_cannot_collide_with_a_scheduled_or_template_trial(self):
        from wyckoff_transformer.cli.protocol import BASINHOP_TRIAL_BASE

        schedule = parse_trial_schedule(DEFAULT_TRIAL_SCHEDULE)
        largest_draw = max(trials for _, trials in schedule) * 10  # x10 multiplier
        self.assertLess(largest_draw, TEMPLATE_TRIAL)
        self.assertLess(TEMPLATE_TRIAL, BASINHOP_TRIAL_BASE)


class TestRowLogMigration(unittest.TestCase):
    """A column added between runs must not silently misalign a resumed log.

    Appending today's fieldnames to a file written with yesterday's header makes
    ``read_csv`` label values by position, so a resumed run's energies would
    land in whatever column happens to sit where they were written. Nothing
    raises; the numbers are just wrong.
    """

    def test_added_columns_are_migrated_and_rows_stay_aligned(self):
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.csv"
            old = ["index", "trial", "status", "energy"]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=old)
                writer.writeheader()
                writer.writerow({"index": 0, "trial": 0, "status": "ok", "energy": -1.5})
            new = old + ["energy_fixed"]
            log = RowLog(path, new, resume=True)
            self.assertEqual(log.done, {(0, 0)})
            log.write({"index": 1, "trial": 0, "status": "ok", "energy": -2.0,
                       "energy_fixed": -1.9})
            log.close()

            frame = log.frame()
            self.assertEqual(list(frame.columns), new)
            self.assertEqual(float(frame.loc[0, "energy"]), -1.5)
            self.assertTrue(pd.isna(frame.loc[0, "energy_fixed"]))
            self.assertEqual(float(frame.loc[1, "energy_fixed"]), -1.9)

    def test_an_unchanged_header_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.csv"
            columns = ["index", "trial", "status"]
            first = RowLog(path, columns, resume=False)
            first.write({"index": 0, "trial": 0, "status": "ok"})
            first.close()
            before = path.read_text()
            second = RowLog(path, columns, resume=True)
            second.close()
            self.assertEqual(path.read_text(), before)


class TestResumeAfterAKilledWorker(unittest.TestCase):
    """A killed worker is not an answered trial.

    A relaxation that diverged has been answered and --resume is right to skip
    it. A trial whose worker was killed under it has not been answered at all,
    and skipping it turns an infrastructure failure into a permanent hole in the
    cohort: it happened here to 1034 of 1800 trials, and read as a collapsed
    reconstruction rate rather than as a crash.
    """

    COLUMNS = ["index", "trial", "status", "error"]

    def _log_with(self, rows, **kwargs):
        import csv

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "log.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        log = RowLog(path, self.COLUMNS, resume=True, **kwargs)
        self.addCleanup(log.close)
        return log

    def test_a_broken_pool_row_is_retried(self):
        log = self._log_with([
            {"index": 0, "trial": 0, "status": "ok", "error": ""},
            {"index": 1, "trial": 0, "status": "failed",
             "error": "BrokenProcessPool: A process in the process pool was "
                      "terminated abruptly"},
        ])
        self.assertEqual(log.done, {(0, 0)})

    def test_a_failed_gpu_row_is_retried(self):
        """What a broken card on iapetus wrote for 750 trials of one run."""
        log = self._log_with([
            {"index": 1, "trial": 0, "status": "failed",
             "error": "AcceleratorError: CUDA error: unspecified launch failure\n"
                      "Search for `cudaErrorLaunchFailure' in https://docs.nvidia.com"},
        ])
        self.assertEqual(log.done, set())

    def test_a_genuine_failure_stays_done_by_default(self):
        log = self._log_with([
            {"index": 0, "trial": 0, "status": "failed",
             "error": "ValueError: the cell collapsed"},
        ])
        self.assertEqual(log.done, {(0, 0)})

    def test_retry_failed_reruns_a_genuine_failure_too(self):
        log = self._log_with(
            [{"index": 0, "trial": 0, "status": "failed",
              "error": "ValueError: the cell collapsed"}],
            retry_failed=True,
        )
        self.assertEqual(log.done, set())

    def test_a_successful_row_is_never_retried(self):
        log = self._log_with(
            [{"index": 0, "trial": 0, "status": "ok", "error": ""}],
            retry_failed=True,
        )
        self.assertEqual(log.done, {(0, 0)})


class TestResumeLineage(unittest.TestCase):
    """A resumed log must have been built from the inputs it is resumed with.

    Stage logs are keyed by (gene index, trial) alone. protocol_ehull5x-20260904-213346
    v1 resumed the draws and relaxations of one gene file under a newly sampled
    one, and 789 of its 998 scored structures belonged to another gene.
    """

    CU = {"group": 225, "species": ["Cu"], "numIons": [4], "sites": [["4a"]]}
    NACL = {"group": 225, "species": ["Na", "Cl"], "numIons": [4, 4], "sites": [["4a"], ["4b"]]}

    def setUp(self):
        from wyckoff_transformer.cli.protocol import SCREEN_FILE

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        write_screen(
            GeneScreen(n_sampled=1, valid=[0], invalid=[], counts={0: 1}, novel=[0], known=[]),
            self.out / SCREEN_FILE,
        )

    def _genes(self, *genes) -> Path:
        path = self.out / "genes.json"
        path.write_text(json.dumps(list(genes)), encoding="utf-8")
        return path

    def _draws_log(self, formula="Cu4"):
        from wyckoff_transformer.cli.protocol import PYXTAL_COLUMNS, PYXTAL_TRIALS_FILE

        pd.DataFrame(
            [{"index": 0, "trial": 0, "status": "ok", "formula": formula, "n_atoms": 4,
              "dof_positional": 0, "n_trials": 1, "seconds": 0.1}],
            columns=list(PYXTAL_COLUMNS),
        ).to_csv(self.out / PYXTAL_TRIALS_FILE, index=False)

    def _generate(self, genes_file, resume):
        from wyckoff_transformer.cli.protocol import stage_generate
        from wyckoff_transformer.cli.worker_pool import PoolReport

        args = SimpleNamespace(
            input=genes_file, output_dir=self.out, limit=None, n_trials="1",
            trial_multiplier=1, resume=resume, retry_failed=False, pyxtal_cores=1,
            pyxtal_timeout=10.0, pyxtal_tol_factor=1.3, debug=False,
        )
        with patch("wyckoff_transformer.cli.protocol.run_supervised",
                   return_value=PoolReport()) as supervised:
            stage_generate(args)
        return supervised

    def _lineage(self):
        from wyckoff_transformer.cli.protocol import LINEAGE_KEY, MANIFEST_FILE

        return json.loads((self.out / MANIFEST_FILE).read_text())[LINEAGE_KEY]

    def test_the_digest_ignores_the_file_it_was_read_from(self):
        from wyckoff_transformer.cli.protocol import genes_digest

        self.assertEqual(genes_digest([dict(self.CU)]),
                         genes_digest([dict(reversed(list(self.CU.items())))]))
        self.assertNotEqual(genes_digest([self.CU]), genes_digest([self.NACL]))

    def test_generate_records_the_gene_file_it_drew_from(self):
        from wyckoff_transformer.cli.protocol import PYXTAL_TRIALS_FILE, genes_digest

        self._generate(self._genes(self.CU), resume=False)
        self.assertEqual(self._lineage()[PYXTAL_TRIALS_FILE]["parent"], genes_digest([self.CU]))

    def test_generate_refuses_to_resume_draws_of_another_gene_file(self):
        from wyckoff_transformer.cli.protocol import StaleOutputError

        self._generate(self._genes(self.CU), resume=False)
        self._draws_log("Cu4")
        with self.assertRaises(StaleOutputError):
            self._generate(self._genes(self.NACL), resume=True)

    def test_no_resume_starts_over_on_another_gene_file(self):
        from wyckoff_transformer.cli.protocol import PYXTAL_TRIALS_FILE, genes_digest

        self._generate(self._genes(self.CU), resume=False)
        first = self._lineage()[PYXTAL_TRIALS_FILE]["id"]
        self._draws_log("Cu4")
        self._generate(self._genes(self.NACL), resume=False)
        record = self._lineage()[PYXTAL_TRIALS_FILE]
        self.assertEqual(record["parent"], genes_digest([self.NACL]))
        self.assertNotEqual(record["id"], first)

    def test_resuming_the_same_gene_file_keeps_the_log_id(self):
        from wyckoff_transformer.cli.protocol import PYXTAL_TRIALS_FILE

        genes = self._genes(self.CU)
        self._generate(genes, resume=False)
        self._draws_log("Cu4")
        first = self._lineage()[PYXTAL_TRIALS_FILE]["id"]
        self._generate(genes, resume=True)
        self.assertEqual(self._lineage()[PYXTAL_TRIALS_FILE]["id"], first)

    def test_an_unrecorded_log_of_other_compositions_is_refused(self):
        """What the ehull5x run's own directory would have been checked with."""
        from wyckoff_transformer.cli.protocol import StaleOutputError

        self._draws_log("Na4Cl4")
        with self.assertRaisesRegex(StaleOutputError, "1 of 1 rows"):
            self._generate(self._genes(self.CU), resume=True)

    def test_an_unrecorded_log_of_matching_compositions_is_adopted(self):
        from wyckoff_transformer.cli.protocol import PYXTAL_TRIALS_FILE, genes_digest

        self._draws_log("Cu2")  # a primitive cell of the same composition
        supervised = self._generate(self._genes(self.CU), resume=True)
        self.assertEqual(list(supervised.call_args.args[0]), [])
        self.assertEqual(self._lineage()[PYXTAL_TRIALS_FILE]["parent"], genes_digest([self.CU]))

    def test_generate_refuses_a_screen_of_another_gene_file(self):
        from wyckoff_transformer.cli.protocol import (
            SCREEN_FILE, StaleOutputError, _write_lineage, genes_digest,
        )

        _write_lineage(self.out, SCREEN_FILE, {"id": "s", "parent": genes_digest([self.NACL])})
        with self.assertRaisesRegex(StaleOutputError, "screen"):
            self._generate(self._genes(self.CU), resume=False)

    def _relax(self, resume):
        from ase.build import bulk
        from ase.io import write as ase_write
        from wyckoff_transformer.cli.protocol import PYXTAL_FILE, stage_relax
        from wyckoff_transformer.cli.worker_pool import PoolReport

        atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
        atoms.info = {"gene": 0, "trial": 0}
        ase_write(str(self.out / PYXTAL_FILE), atoms, format="extxyz")
        args = SimpleNamespace(
            output_dir=self.out, mlip="orb_conserv_inf", cores=1, devices=None,
            workers_per_device=1, limit=None, resume=resume, fmax=0.05,
            release_symmetry=True, rattle=True, relax_timeout=10.0, debug=False,
        )
        with patch("wyckoff_transformer.cli.protocol.run_supervised",
                   return_value=PoolReport()), \
                patch("wyckoff_transformer.cli.protocol.aggregate_structures",
                      return_value=pd.DataFrame({"has_structure": [True]})):
            stage_relax(args)

    def _relaxations_log(self, formula="Cu4"):
        from wyckoff_transformer.cli.protocol import RELAXATION_COLUMNS, RELAXATIONS_FILE

        pd.DataFrame(
            [{"index": 0, "trial": 0, "status": "ok", "formula": formula, "energy": -1.0,
              "n_atoms": 4, "device": "cpu"}],
            columns=list(RELAXATION_COLUMNS),
        ).to_csv(self.out / RELAXATIONS_FILE, index=False)

    def test_relax_refuses_to_resume_relaxations_of_replaced_draws(self):
        """PyXtal is not seeded: `generate --no-resume` re-draws under the same keys."""
        from wyckoff_transformer.cli.protocol import StaleOutputError

        self._draws_log()
        self._relax(resume=False)
        self._relaxations_log()
        self._generate(self._genes(self.CU), resume=False)
        self._draws_log()
        with self.assertRaisesRegex(StaleOutputError, "relaxations.csv"):
            self._relax(resume=True)

    def test_relax_resumes_relaxations_of_the_same_draws(self):
        from wyckoff_transformer.cli.protocol import RELAXATIONS_FILE

        self._generate(self._genes(self.CU), resume=False)
        self._draws_log()
        self._relax(resume=False)
        self._relaxations_log()
        first = self._lineage()[RELAXATIONS_FILE]["id"]
        self._relax(resume=True)
        self.assertEqual(self._lineage()[RELAXATIONS_FILE]["id"], first)

    def test_an_unrecorded_relaxations_log_of_other_draws_is_refused(self):
        from wyckoff_transformer.cli.protocol import StaleOutputError

        self._relaxations_log("Na4Cl4")
        with self.assertRaises(StaleOutputError):
            self._relax(resume=True)

    def test_score_refuses_outputs_of_different_lineages(self):
        from wyckoff_transformer.cli.protocol import (
            PYXTAL_TRIALS_FILE, RELAXATIONS_FILE, SCREEN_FILE, StaleOutputError,
            _write_lineage, genes_digest, require_consistent_lineage,
        )

        digest = genes_digest([self.CU])
        _write_lineage(self.out, SCREEN_FILE, {"id": "s", "parent": digest})
        _write_lineage(self.out, PYXTAL_TRIALS_FILE, {"id": "draws-2", "parent": digest})
        _write_lineage(self.out, RELAXATIONS_FILE, {"id": "r", "parent": "draws-2"})
        require_consistent_lineage(self.out, [self.CU])
        with self.assertRaisesRegex(StaleOutputError, "another gene file"):
            require_consistent_lineage(self.out, [self.NACL])
        _write_lineage(self.out, RELAXATIONS_FILE, {"id": "r", "parent": "draws-1"})
        with self.assertRaisesRegex(StaleOutputError, "relaxations.csv"):
            require_consistent_lineage(self.out, [self.CU])

    def test_a_run_from_before_lineage_is_scored(self):
        from wyckoff_transformer.cli.protocol import require_consistent_lineage

        require_consistent_lineage(self.out, [self.CU])
