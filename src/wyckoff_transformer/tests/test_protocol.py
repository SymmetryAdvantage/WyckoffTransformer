"""Tests for the de novo ranking protocol."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from wyckoff_transformer.cli.protocol import (
    _pin_visible_device,
    build_parser,
    resolve_devices,
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
        """Yb and the actinides are excluded from `scripts/compute_e_hull.py`.

        That exclusion belongs to the training labels, not here: a generated
        structure containing Yb has to be scored, not silently dropped.
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
        # Only gene 0 is at or below 0.1 eV/atom and survived every filter.
        self.assertEqual(report["metastable"], 1)
        self.assertAlmostEqual(report["metasun_per_sampled_gene"], 0.3)
        self.assertEqual(report["stable"], 0)
        self.assertAlmostEqual(report["sun_per_sampled_gene"], 0.0)

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
