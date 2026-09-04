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
    FINAL_CIF_SUFFIX,
    SYMMETRIC_CIF_SUFFIX,
    final_cif_suffix,
)
from wyckoff_transformer.evaluation.hull_mlips import (
    DEFAULT_HULL_MLIP,
    HULL_MLIPS,
    UnsupportedHullMlip,
    resolve_hull_mlip,
)
from wyckoff_transformer.evaluation.protocol import (
    GeneFingerprinter,
    GeneScreen,
    funnel,
    read_screen,
    screen_genes,
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
        self.assertEqual(args.n_trials, 1)
        self.assertFalse(args.release_symmetry)  # 1 trial x 2 stages

    def test_release_symmetry_can_be_turned_back_on(self):
        args = build_parser().parse_args(
            ["genes.json", "--output-dir", "out", "--release-symmetry"]
        )
        self.assertTrue(args.release_symmetry)

    def test_mlip_choices_are_restricted_to_published_hulls(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["genes.json", "--output-dir", "out", "--mlip", "chgnet"]
            )


class TestFinalCifSuffix(unittest.TestCase):
    def test_two_stage_run_ends_on_the_symmetric_stage(self):
        self.assertEqual(final_cif_suffix(release_symmetry=False), SYMMETRIC_CIF_SUFFIX)
        self.assertEqual(final_cif_suffix(release_symmetry=True), FINAL_CIF_SUFFIX)


class TestStepwiseRelaxStages(unittest.TestCase):
    """The 2-stage schedule must skip exactly one stage, and not silently no-op."""

    def _labels_for(self, **kwargs) -> list[str]:
        from wyckoff_transformer.cryspr import relaxer

        atoms = MagicMock()
        atoms.copy.return_value = atoms
        atoms.get_chemical_formula.return_value = "NaCl"
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

    def test_three_stages_by_default(self):
        self.assertEqual(
            self._labels_for(),
            ["1_fix-cell", "2_sym_cell+pos", "3_no-sym_cell+pos"],
        )

    def test_two_stages_when_symmetry_is_not_released(self):
        self.assertEqual(
            self._labels_for(release_symmetry=False),
            ["1_fix-cell", "2_sym_cell+pos"],
        )

    def test_dropping_both_symmetry_stages_is_refused(self):
        from wyckoff_transformer.cryspr import relaxer

        with self.assertRaises(ValueError):
            relaxer.stepwise_relax(
                atoms_in=MagicMock(), calculator=MagicMock(),
                fix_symmetry=False, release_symmetry=False,
            )


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
