"""The pieces the alex-mp-20 benchmark study added to the protocol, roe and gene screen.

DiffCSP++ starts (``--stage starts``), the single-relaxation schedule, novelty against a
dataset keyed by ``material_id``, and fire-control on a regressor that predicts e_hull.
"""
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.build import bulk

from wyckoff_transformer.cli.protocol import (
    OPTIONAL_STAGES,
    PYXTAL_FILE,
    PYXTAL_TRIALS_FILE,
    SCREEN_FILE,
    STARTS_FILE,
    build_parser,
)
from wyckoff_transformer.evaluation.protocol import GeneScreen, write_screen

#: Rock salt in its conventional cell: Na on 4a, Cl on 4b of Fm-3m.
NACL_GENE = {"group": 225, "sites": [["4a"], ["4b"]], "species": ["Na", "Cl"],
             "numIons": [4, 4]}


def _rocksalt() -> Atoms:
    return bulk("NaCl", "rocksalt", a=5.64, cubic=True)


class TestSingleSchedule(unittest.TestCase):
    def _labels_for(self, **kwargs) -> list[str]:
        from wyckoff_transformer.cryspr import relaxer

        atoms = MagicMock()
        atoms.copy.return_value = atoms
        atoms.get_chemical_formula.return_value = "NaCl"
        atoms.get_potential_energy.return_value = -1.0
        seen = []

        def fake_relaxer(*, label, **_):
            seen.append(label)
            return atoms

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(relaxer, "run_ase_relaxer", side_effect=fake_relaxer), \
                patch.object(relaxer, "write"):
            relaxer.stepwise_relax(atoms_in=atoms, calculator=MagicMock(), wdir=Path(tmp),
                                   **kwargs)
        return seen

    def test_single_is_exactly_one_unconstrained_relaxation(self):
        self.assertEqual(
            self._labels_for(fix_symmetry=False, warmup=False, rattle=False),
            ["3_no-sym_cell+pos"])

    def test_warmup_stays_on_by_default(self):
        self.assertEqual(self._labels_for(fix_symmetry=False, rattle=False),
                         ["1_fix-cell", "3_no-sym_cell+pos"])

    def test_relax_one_hands_the_single_schedule_to_relax_trial(self):
        from wyckoff_transformer.cli import protocol
        from wyckoff_transformer.cryspr import generator

        calls = {}

        def fake_relax_trial(**kwargs):
            calls.update(kwargs)
            raise RuntimeError("stop here")

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(generator, "relax_trial", side_effect=fake_relax_trial):
            row = protocol._relax_one(0, 0, _rocksalt(), tmp, 0.05, True, False, None,
                                      schedule="single", steps_limit=1000)
        self.assertEqual(row["status"], "failed")
        self.assertIs(calls["fix_symmetry"], False)
        self.assertIs(calls["warmup"], False)
        self.assertEqual(calls["steps_limit"], 1000)

    def test_cryspr_schedule_keeps_relax_trials_defaults(self):
        from wyckoff_transformer.cli import protocol
        from wyckoff_transformer.cryspr import generator

        calls = {}

        def fake_relax_trial(**kwargs):
            calls.update(kwargs)
            raise RuntimeError("stop here")

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(generator, "relax_trial", side_effect=fake_relax_trial):
            protocol._relax_one(0, 0, _rocksalt(), tmp, 0.05, True, True, None)
        self.assertNotIn("fix_symmetry", calls)
        self.assertNotIn("warmup", calls)


class TestCliAdditions(unittest.TestCase):
    def test_defaults_leave_the_published_protocol_unchanged(self):
        args = build_parser().parse_args(["genes.json", "--output-dir", "out"])
        self.assertEqual(args.relax_schedule, "cryspr")
        self.assertEqual(args.relax_steps, 500)
        self.assertIsNone(args.reference_id_column)
        self.assertTrue(args.starts_rattle)

    def test_starts_is_an_optional_stage(self):
        from wyckoff_transformer.cli.protocol import STAGES

        self.assertIn("starts", OPTIONAL_STAGES)
        self.assertNotIn("starts", STAGES)
        args = build_parser().parse_args(
            ["genes.json", "--output-dir", "out", "--stage", "starts", "--no-starts-rattle"])
        self.assertFalse(args.starts_rattle)


class TestReferenceIdColumn(unittest.TestCase):
    def test_ids_come_from_the_column_not_the_per_split_row_number(self):
        from wyckoff_transformer.evaluation import structure_novelty

        # alex-mp-20's index restarts in every split: row 0 of train and row 0 of val
        # are different materials.
        frames = [("train", pd.DataFrame({"material_id": ["mp-1"]}, index=[0])),
                  ("val", pd.DataFrame({"material_id": ["agm-2"]}, index=[0]))]
        def fake_iter_splits(cache, splits, columns):
            self.assertIn("material_id", columns)
            for split, frame in frames:
                yield split, frame.reindex(columns=columns)

        with patch.object(structure_novelty, "resolve_cache", side_effect=lambda c: c), \
                patch.object(structure_novelty, "cache_exists", return_value=True), \
                patch.object(structure_novelty, "iter_splits", side_effect=fake_iter_splits), \
                patch("wyckoff_transformer.evaluation.novelty.record_to_augmented_fingerprint",
                      return_value="F"):
            hits = structure_novelty.collect_reference_ids(
                ["F"], cache=Path("x"), splits=["train", "val"], id_column="material_id")
        self.assertEqual(hits, {"F": ["mp-1", "agm-2"]})

    def test_structures_are_read_by_the_named_id_column(self):
        from wyckoff_transformer.evaluation.structure_novelty import load_reference_structures

        from pymatgen.io.ase import AseAtomsAdaptor

        text = AseAtomsAdaptor.get_structure(_rocksalt()).to(fmt="cif")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.csv.gz"
            pd.DataFrame({"material_id": ["mp-22862", "mp-0"], "cif": [text, text]}).to_csv(
                path, index=False)
            found = load_reference_structures(["mp-22862"], lemat_cif_csv=path,
                                              id_column="material_id")
        self.assertEqual(list(found), ["mp-22862"])
        self.assertEqual(found["mp-22862"].composition.reduced_formula, "NaCl")


class TestStartChecks(unittest.TestCase):
    def test_a_structure_of_the_gene_passes(self):
        from wyckoff_transformer.diffcsp_bridge import check_start

        check = check_start(_rocksalt(), NACL_GENE)
        self.assertIsNone(check.error)
        self.assertEqual(check.n_atoms, 8)
        self.assertEqual(check.spacegroup, 225)

    def test_another_composition_is_refused(self):
        from wyckoff_transformer.diffcsp_bridge import check_start

        gene = dict(NACL_GENE, species=["K", "Cl"])
        self.assertIn("composition", check_start(_rocksalt(), gene).error)

    def test_overlapping_atoms_are_refused(self):
        from wyckoff_transformer.diffcsp_bridge import check_start

        atoms = _rocksalt()
        atoms.positions[1] = atoms.positions[0] + [0.1, 0, 0]
        self.assertIn("apart", check_start(atoms, NACL_GENE).error)

    def test_the_rattle_is_seeded_per_gene_and_trial(self):
        from wyckoff_transformer.diffcsp_bridge import rattle_start

        atoms = _rocksalt()
        first, again = rattle_start(atoms, 3, 0), rattle_start(atoms, 3, 0)
        other = rattle_start(atoms, 4, 0)
        np.testing.assert_allclose(first.positions, again.positions)
        self.assertFalse(np.allclose(first.positions, atoms.positions))
        self.assertFalse(np.allclose(first.positions, other.positions))
        self.assertFalse(np.allclose(first.cell[:], atoms.cell[:]))


class TestStageStarts(unittest.TestCase):
    def _run(self, tmp: Path, rattle: bool = True):
        from ase.io import write

        from wyckoff_transformer.cli import protocol

        genes = [NACL_GENE, dict(NACL_GENE, species=["K", "Cl"]), NACL_GENE]
        genes_path = tmp / "genes.json.gz"
        with gzip.open(genes_path, "wt") as handle:
            json.dump(genes, handle)
        out = tmp / "protocol"
        out.mkdir()
        # Genes 0 and 2 are the same gene; only the representative is reconstructed.
        write_screen(GeneScreen(n_sampled=3, valid=[0, 1, 2], counts={0: 2, 1: 1},
                                novel=[0, 1]), out / SCREEN_FILE)
        frames = []
        for gene in (0, 1):  # gene 1 gets a NaCl structure: the wrong composition
            atoms = _rocksalt()
            atoms.info = {"gene": gene, "trial": 0}
            frames.append(atoms)
        write(str(tmp / "starts.extxyz"), frames, format="extxyz")
        pd.DataFrame({"index": [0, 1], "trial": [0, 0], "status": ["ok", "ok"],
                      "error": ["", ""]}).to_csv(tmp / "starts.csv", index=False)
        args = SimpleNamespace(input=genes_path, output_dir=out, limit=None,
                               starts=tmp / "starts.extxyz", starts_log=tmp / "starts.csv",
                               starts_rattle=rattle, resume=False)
        protocol.stage_starts(args)
        return out

    def test_passing_starts_are_filed_and_failures_are_logged(self):
        from ase.io import read

        with tempfile.TemporaryDirectory() as tmp:
            out = self._run(Path(tmp))
            log = pd.read_csv(out / PYXTAL_TRIALS_FILE).set_index("index")
            self.assertEqual(log.loc[0, "status"], "ok")
            self.assertEqual(log.loc[1, "status"], "failed")
            self.assertIn("composition", log.loc[1, "error"])
            frames = read(str(out / PYXTAL_FILE), index=":")
            self.assertEqual([f.info["gene"] for f in frames], [0])
            self.assertFalse(np.allclose(frames[0].positions, _rocksalt().positions))
            diagnostics = pd.read_csv(out / STARTS_FILE).set_index("index")
            self.assertEqual(diagnostics.loc[0, "spacegroup_start"], 225)

    def test_no_rattle_files_the_start_as_given(self):
        from ase.io import read

        with tempfile.TemporaryDirectory() as tmp:
            out = self._run(Path(tmp), rattle=False)
            frame = read(str(out / PYXTAL_FILE), index=0)
            np.testing.assert_allclose(frame.positions, _rocksalt().positions, atol=1e-8)

    def test_a_gene_missing_from_the_log_is_refused(self):
        # The log lists only gene 0, so representative 1 has no record at all: the
        # starts were made for another gene file.
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(pd, "read_csv", return_value=pd.DataFrame(
                    {"index": [0], "trial": [0], "status": ["ok"], "error": [""]})):
                with self.assertRaisesRegex(ValueError, "no row"):
                    self._run(Path(tmp))


class TestDirectEHullRegressor(unittest.TestCase):
    def _regressor(self, values):
        from wyckoff_transformer.cascade.dataset import TargetClass

        regressor = MagicMock()
        regressor.target = TargetClass.Scalar
        regressor.scalar_loss = "mse"
        regressor.target_name = "gene_min_energy_above_hull"
        regressor.condition_features = ()
        regressor.device = torch.device("cpu")
        regressor.predict_scalars.return_value = (torch.tensor(values), None)
        return regressor

    def test_validate_accepts_both_targets_and_nothing_else(self):
        from wyckoff_transformer.cli.gene_screen import validate_regressor

        regressor = self._regressor([0.0])
        validate_regressor(regressor)
        regressor.target_name = "gene_min_formation_energy_per_atom"
        validate_regressor(regressor)
        regressor.target_name = "band_gap"
        with self.assertRaises(ValueError):
            validate_regressor(regressor)

    def test_the_prediction_is_the_score_and_no_hull_is_needed(self):
        from wyckoff_transformer.cli import gene_screen

        regressor = self._regressor([0.03, -0.01])
        genes = [NACL_GENE, dict(NACL_GENE, species=["K", "Cl"])]
        with patch.object(gene_screen, "filter_supported_tokens",
                          side_effect=lambda frame, _: (frame, [])), \
                patch.object(gene_screen, "build_tokenised_prediction_tensors"), \
                patch.object(gene_screen, "HullLookup") as lookup:
            scored = gene_screen.score_genes(genes, regressor, None).sort_index()
        lookup.assert_not_called()
        np.testing.assert_allclose(scored["score"], [0.03, -0.01], atol=1e-6)
        self.assertEqual(list(scored["predicted_below_hull"]), [False, True])
        self.assertTrue(scored["hull_energy"].isna().all())

    def test_the_filter_refuses_a_corrected_direct_energy(self):
        from wyckoff_transformer.roe.builtin import PredictedHullFilter

        self.assertIn("direct", PredictedHullFilter.HULLS)
        with self.assertRaises(ValueError):
            PredictedHullFilter(MagicMock(), None, hull="direct", energy="corrected",
                                known_genes=MagicMock())

    def test_the_filter_selects_the_lowest_predictions(self):
        from wyckoff_transformer.roe import builtin

        regressor = self._regressor([0.0])
        filt = builtin.PredictedHullFilter(regressor, None, hull="direct", select="top",
                                           top_k=1)
        scored = pd.DataFrame({"score": [0.05, 0.01], "formula": ["NaCl", "KCl"]},
                              index=[0, 1])
        variants = filt._variants(MagicMock(), [0, 1], scored)
        self.assertEqual(list(variants), ["predicted_e_hull_direct_raw"])
        self.assertEqual(list(variants["predicted_e_hull_direct_raw"]), [0.05, 0.01])


if __name__ == "__main__":
    unittest.main()
