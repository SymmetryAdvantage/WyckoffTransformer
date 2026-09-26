"""Tests for ``wyformer-cache-dataset``."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from wyckoff_transformer import dataset_manifest
from wyckoff_transformer.cli import cache_dataset as cli
from wyckoff_transformer.dataset_cache import load_cache

TOY_MANIFEST = """
name: toy
status: current
defaults:
  source: {method: dft, dft_settings: PBE_MP, correction: none}
  reference: lemat_bulk_pbe
fields:
  energy_above_hull:
    column: e_above_hull
    energy: {quantity: energy_above_hull, extent: per_atom}
  formation_energy_per_atom:
    column: e_form
    energy: {quantity: formation_energy, extent: per_atom}
duplicates:
  formation_energy_per_atom: e_form
"""


def use_manifests(test: unittest.TestCase, root: Path, **manifests: str) -> None:
    """Point the manifest registry at a directory holding just these, for one test."""
    directory = root / "manifests"
    directory.mkdir()
    for name, text in manifests.items():
        (directory / f"{name}.yaml").write_text(text)
    p = patch.object(dataset_manifest, "MANIFEST_DIR", directory)
    p.start()
    test.addCleanup(p.stop)
    dataset_manifest.load_manifest.cache_clear()
    test.addCleanup(dataset_manifest.load_manifest.cache_clear)


class TestColumnsToCarry(unittest.TestCase):
    """What travels from the CSV into the cache, and what may not."""

    HEADER = pd.DataFrame({
        "material_id": pd.Series(dtype=object),
        "formation_energy_per_atom": pd.Series(dtype=float),
        "max_force_missing": pd.Series(dtype=bool),
        "dft_run_type": pd.Series(dtype=object),
        "cif": pd.Series(dtype=object),
        "elements": pd.Series(dtype=object),
    })

    def test_every_label_column_is_carried_without_being_named(self):
        # The point of the default: a conditioning label that was built and then
        # left off a hand-written --scalar-columns never reached a tensor.
        self.assertEqual(
            cli.columns_to_carry(self.HEADER),
            ["material_id", "formation_energy_per_atom", "max_force_missing", "dft_run_type"])

    def test_the_structure_is_not_a_label(self):
        self.assertNotIn("cif", cli.columns_to_carry(self.HEADER))

    def test_a_column_the_symmetry_record_provides_is_skipped(self):
        # mp_20's `elements` is the formula's element list, not the per-site one
        # the record computes; carrying it would overwrite the record's.
        self.assertNotIn("elements", cli.columns_to_carry(self.HEADER))

    def test_naming_columns_carries_exactly_those(self):
        self.assertEqual(
            cli.columns_to_carry(self.HEADER, ["dft_run_type", "material_id"]),
            ["dft_run_type", "material_id"])

    def test_a_named_column_that_is_missing_is_an_error(self):
        with self.assertRaisesRegex(KeyError, "band_gap"):
            cli.columns_to_carry(self.HEADER, ["band_gap"])

    def test_a_named_column_that_would_overwrite_the_record_is_an_error(self):
        # Silently dropping it would be worse: the caller asked for it by name.
        with self.assertRaisesRegex(KeyError, "elements"):
            cli.columns_to_carry(self.HEADER, ["elements"])
        with self.assertRaisesRegex(KeyError, "cif"):
            cli.columns_to_carry(self.HEADER, ["cif"])

    def test_a_column_of_something_else_is_skipped_loudly(self):
        header = self.HEADER.assign(when=pd.Series(dtype="datetime64[ns]"))
        with self.assertLogs(cli.logger, "WARNING") as logs:
            carried = cli.columns_to_carry(header)
        self.assertNotIn("when", carried)
        self.assertIn("when", "\n".join(logs.output))


class TestSplitCsv(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_gzipped_or_not(self):
        (self.dir / "val.csv").touch()
        self.assertEqual(cli.split_csv(self.dir, "val"), self.dir / "val.csv")
        (self.dir / "val.csv.gz").touch()
        self.assertEqual(cli.split_csv(self.dir, "val"), self.dir / "val.csv.gz")

    def test_a_split_that_is_not_there(self):
        self.assertIsNone(cli.split_csv(self.dir, "test"))


class TestCacheSplit(unittest.TestCase):
    """The per-split pass, with symmetry determination stubbed out."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.csv = self.dir / "train.csv.gz"
        pd.DataFrame({
            "immutable_id": ["a", "b", "c", "d"],
            "cif": ["cif-a", "cif-b", "cif-c", "cif-d"],
            "energy_above_hull": [0.0, 0.1, 0.2, 0.3],
        }).set_index("immutable_id").to_csv(self.csv)

    @staticmethod
    def _record(sites):
        return {
            "site_symmetries": ["m"] * sites,
            "elements": ["Na"] * sites,
            "multiplicity": [1] * sites,
            "wyckoff_letters": ["a"] * sites,
            "sites_enumeration": [0] * sites,
            "dof": [0] * sites,
            "spacegroup_number": 1,
            "composition": None,
        }

    def _run(self, sites_by_cif, **kwargs):
        def fake(cif, **_):
            count = sites_by_cif[cif]
            return None if count is None else self._record(count)

        # Pool.map would have to pickle the local function, and the point here
        # is the bookkeeping around symmetrisation, not pyxtal.
        with patch.object(cli, "Pool") as pool:
            pool.return_value.__enter__.return_value.map = \
                lambda function, values, chunksize=None: [fake(v) for v in values]
            return cli.cache_split(
                self.csv, ["energy_above_hull"], None, 2, 0.1, 5.0, True,
                kwargs.get("max_sites"))

    def test_over_long_structures_are_dropped_not_truncated(self):
        frame = self._run({"cif-a": 2, "cif-b": 9, "cif-c": 3, "cif-d": 61}, max_sites=8)
        self.assertEqual(list(frame.index), ["a", "c"])
        # The kept rows are whole: nothing was cut down to the cap.
        self.assertEqual(list(frame["site_symmetries"].str.len()), [2, 3])

    def test_no_cap_keeps_everything(self):
        frame = self._run({"cif-a": 2, "cif-b": 9, "cif-c": 3, "cif-d": 61})
        self.assertEqual(list(frame.index), ["a", "b", "c", "d"])

    def test_a_structure_pyxtal_cannot_handle_is_dropped_with_a_count(self):
        # Raising would take a six-hour run down for one bad row.
        with self.assertLogs(cli.logger, "WARNING") as logs:
            frame = self._run({"cif-a": 2, "cif-b": None, "cif-c": 3, "cif-d": 4})
        self.assertEqual(list(frame.index), ["a", "c", "d"])
        self.assertIn("pyxtal could not handle", "\n".join(logs.output))

    def test_the_labels_land_on_the_right_rows(self):
        frame = self._run({"cif-a": 2, "cif-b": 9, "cif-c": 3, "cif-d": 61}, max_sites=8)
        self.assertEqual(list(frame["energy_above_hull"]), [0.0, 0.2])

    def test_nothing_surviving_is_an_error_not_an_empty_cache(self):
        with self.assertRaisesRegex(ValueError, "no structure survived"):
            self._run({f"cif-{k}": None for k in "abcd"})


class TestCacheDataset(unittest.TestCase):
    """The whole command, against a stubbed symmetriser."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.data = self.root / "data" / "toy"
        self.data.mkdir(parents=True)
        for split, rows in (("train", 3), ("val", 2), ("test", 2)):
            pd.DataFrame({
                "immutable_id": [f"{split}-{i}" for i in range(rows)],
                "cif": [f"cif-{split}-{i}" for i in range(rows)],
                "e_above_hull": [0.1 * i for i in range(rows)],
                "e_form": [-0.5 * i for i in range(rows)],
                # A builder's copy of e_form under the canonical name, which the
                # manifest says to verify and drop.
                "formation_energy_per_atom": [-0.5 * i for i in range(rows)],
                "elements": ["['Na']"] * rows,
            }).set_index("immutable_id").to_csv(self.data / f"{split}.csv.gz")
        use_manifests(self, self.root, toy=TOY_MANIFEST,
                      old_toy="name: old_toy\nstatus: obsolete\nreason: superseded by toy\n")

        frame_record = TestCacheSplit._record
        patcher = patch.object(cli, "Pool")
        pool = patcher.start()
        self.addCleanup(patcher.stop)
        pool.return_value.__enter__.return_value.map = \
            lambda function, values, chunksize=None: [frame_record(2) for _ in values]
        self.addCleanup(patch.stopall)
        for target, value in (("data_path", lambda name: self.data),
                              ("dataset_cache_dir", lambda name: self.root / "cache" / name)):
            p = patch.object(cli, target, value)
            p.start()
            self.addCleanup(p.stop)

    def test_every_split_is_cached(self):
        frames = cli.cache_dataset("toy")
        self.assertEqual({k: len(v) for k, v in frames.items()},
                         {"train": 3, "val": 2, "test": 2})
        self.assertEqual(sorted(load_cache(self.root / "cache" / "toy")),
                         ["test", "train", "val"])

    def test_labels_are_carried_without_being_named(self):
        frames = cli.cache_dataset("toy")
        self.assertIn("energy_above_hull", frames["train"].columns)

    def test_columns_get_their_canonical_names(self):
        frames = cli.cache_dataset("toy")
        columns = set(frames["train"].columns)
        self.assertTrue({"energy_above_hull", "formation_energy_per_atom"} <= columns)
        self.assertFalse({"e_above_hull", "e_form"} & columns)
        self.assertEqual(list(frames["train"]["formation_energy_per_atom"]), [0.0, -0.5, -1.0])

    def test_a_duplicate_that_differs_from_its_original_is_refused(self):
        path = self.data / "train.csv.gz"
        frame = pd.read_csv(path, index_col=0)
        frame["formation_energy_per_atom"] = 1.0
        frame.to_csv(path)
        with self.assertRaisesRegex(ValueError, "declared a copy"):
            cli.cache_dataset("toy")

    def test_the_field_definitions_are_recorded(self):
        from wyckoff_transformer.dataset_cache import build_info

        cli.cache_dataset("toy")
        recorded = build_info(self.root / "cache" / "toy", "train")["options"]
        self.assertEqual(recorded["manifest"], "toy")
        self.assertEqual(recorded["fields"],
                         dataset_manifest.load_manifest("toy").fields_record())
        dataset_manifest.check_cache_matches("toy", recorded["fields"])

    def test_an_obsolete_dataset_is_refused_unless_allowed(self):
        with self.assertRaises(dataset_manifest.ObsoleteDatasetError):
            cli.cache_dataset("old_toy")
        with self.assertLogs(dataset_manifest.logger, "WARNING"):
            frames = cli.cache_dataset("old_toy", allow_obsolete_dataset=True)
        # No manifest fields: nothing renamed, nothing recorded.
        self.assertIn("e_above_hull", frames["train"].columns)

    def test_a_split_that_is_absent_is_skipped_not_fatal(self):
        (self.data / "test.csv.gz").unlink()
        with self.assertLogs(cli.logger, "WARNING"):
            frames = cli.cache_dataset("toy")
        self.assertEqual(sorted(frames), ["train", "val"])

    def test_the_build_options_are_recorded_in_every_split(self):
        from wyckoff_transformer.dataset_cache import build_info

        cli.cache_dataset("toy", max_sites=61, symmetry_precision=0.2)
        records = build_info(self.root / "cache" / "toy")
        self.assertEqual(sorted(records), ["test", "train", "val"])
        for split, record in records.items():
            self.assertEqual(record["tool"], "wyformer-cache-dataset", split)
            self.assertEqual(record["options"]["max_sites"], 61, split)
            self.assertEqual(record["options"]["symmetry_precision"], 0.2, split)
            self.assertTrue(record["options"]["sort_by_letter"], split)

    def test_the_gene_minimum_records_which_splits_it_was_taken_over(self):
        # The minimum spans every split present at cache time, so the same flag
        # over two splits and over three gives a different target column.
        from wyckoff_transformer.dataset_cache import build_info

        (self.data / "test.csv.gz").unlink()
        with patch("wyckoff_transformer.gene_energy.add_observed_gene_minimum"), \
             self.assertLogs(cli.logger, "WARNING"):
            cli.cache_dataset("toy", observed_gene_minimum_target=True)
        record = build_info(self.root / "cache" / "toy", "train")
        self.assertEqual(record["options"]["observed_gene_minimum_over"], ["train", "val"])

    def test_the_gene_minimum_is_null_when_it_was_not_asked_for(self):
        from wyckoff_transformer.dataset_cache import build_info

        cli.cache_dataset("toy")
        record = build_info(self.root / "cache" / "toy", "train")
        self.assertIsNone(record["options"]["observed_gene_minimum_over"])

    def test_a_dataset_with_no_splits_at_all_is_an_error(self):
        for split in ("train", "val", "test"):
            (self.data / f"{split}.csv.gz").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "no train/val/test CSV"):
            cli.cache_dataset("toy")


class TestParser(unittest.TestCase):
    def test_there_is_no_way_to_truncate(self):
        # --max-wp is gone on purpose; it changed a structure's composition and
        # left its energy labels attached to the result.
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["toy", "--max-wp", "20"])

    def test_sites_are_sorted_by_letter_unless_asked_otherwise(self):
        self.assertTrue(cli.build_parser().parse_args(["toy"]).sort_by_letter)
        self.assertFalse(
            cli.build_parser().parse_args(["toy", "--no-sort-by-letter"]).sort_by_letter)


if __name__ == "__main__":
    unittest.main()
