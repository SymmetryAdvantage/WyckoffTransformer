"""The Parquet dataset cache, and its fallback to the format it replaced."""
import gzip
import pickle
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from pymatgen.core import Element

from wyckoff_transformer import dataset_cache as dc


def a_frame() -> pd.DataFrame:
    """A frame with one of every column shape the caching CLI produces."""
    return pd.DataFrame(
        {
            "site_symmetries": [["4/mmm", "m2m."], ["m"]],
            "elements": [[Element("Nd"), Element("Al")], [Element("O")]],
            "multiplicity": [[2, 8], [12]],
            "wyckoff_letters": [["a", "j"], ["f"]],
            "sites_enumeration": [[0, 1], [3]],
            "dof": [[0, 1], [2]],
            "spacegroup_number": [139, 8],
            "site_symmetries_augmented": [(("4/mmm", "m2m."), ("4/mmm", ".2.")), (("m",),)],
            "sites_enumeration_augmented": [frozenset({(0, 1), (1, 0)}), frozenset({(3,)})],
            "composition": [Counter({Element("Nd"): 2, Element("Al"): 8}),
                            Counter({Element("O"): 12})],
            "energy_above_hull": [0.0, 0.125],
            "run_type": ["GGA", "GGA+U"],
        },
        index=pd.Index(["mp-1", "mp-2"], name="material_id"),
    )


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = Path(tmp.name) / "toy"

    def assert_same(self, expected: pd.DataFrame, actual: pd.DataFrame):
        self.assertEqual(list(actual.columns), list(expected.columns))
        self.assertTrue(actual.index.equals(expected.index))
        self.assertEqual(actual.index.name, expected.index.name)
        for column in expected.columns:
            for left, right in zip(expected[column], actual[column]):
                self.assertEqual(type(left), type(right), column)
                self.assertEqual(left, right, column)

    def test_a_split_comes_back_as_the_objects_it_went_in_as(self):
        # Not merely equal: pyxtal records hold Element enums, Counters and a
        # frozenset of variants, and a reader that got lists back instead would
        # fingerprint a gene differently.
        frame = a_frame()
        dc.save_split(frame, self.cache, "train")
        self.assert_same(frame, dc.load_split(self.cache, "train"))

    def test_inner_types_survive_too(self):
        dc.save_split(a_frame(), self.cache, "train")
        row = dc.load_split(self.cache, "train").iloc[0]
        self.assertIsInstance(row["elements"][0], Element)
        self.assertIsInstance(row["composition"], Counter)
        self.assertIsInstance(next(iter(row["composition"])), Element)
        self.assertIsInstance(row["sites_enumeration_augmented"], frozenset)
        self.assertIsInstance(next(iter(row["sites_enumeration_augmented"])), tuple)
        self.assertIsInstance(row["site_symmetries_augmented"], tuple)
        self.assertIsInstance(row["site_symmetries_augmented"][0], tuple)

    def test_an_unnamed_index_stays_unnamed(self):
        frame = a_frame().reset_index(drop=True)
        dc.save_split(frame, self.cache, "train")
        loaded = dc.load_split(self.cache, "train")
        self.assertIsNone(loaded.index.name)
        self.assertTrue(loaded.index.equals(frame.index))

    def test_an_empty_split_round_trips(self):
        frame = a_frame().iloc[:0]
        dc.save_split(frame, self.cache, "test")
        loaded = dc.load_split(self.cache, "test")
        self.assertEqual(len(loaded), 0)
        self.assertEqual(list(loaded.columns), list(frame.columns))

    def test_a_missing_value_stays_missing(self):
        # get_composition_from_symmetry_sites returns None for a record it
        # cannot count, and that row must not come back as an empty Counter.
        frame = a_frame()
        frame.loc["mp-2", "composition"] = None
        dc.save_split(frame, self.cache, "train")
        self.assertIsNone(dc.load_split(self.cache, "train").loc["mp-2", "composition"])

    def test_the_whole_cache_round_trips(self):
        frames = {"train": a_frame(), "val": a_frame().iloc[:1]}
        dc.save_cache(frames, self.cache)
        loaded = dc.load_cache(self.cache)
        self.assertEqual(sorted(loaded), ["train", "val"])
        self.assert_same(frames["train"], loaded["train"])

    def test_writing_the_same_frame_twice_gives_the_same_bytes(self):
        frame = a_frame()
        dc.save_split(frame, self.cache, "train")
        first = dc.split_path(self.cache, "train").read_bytes()
        dc.save_split(frame, self.cache, "train")
        self.assertEqual(first, dc.split_path(self.cache, "train").read_bytes())

    def test_a_variant_set_is_written_sorted_whatever_order_it_iterates(self):
        # The bytes must follow the contents, not the insertion history of the
        # frame: two frames holding equal sets have to produce equal files.
        variants = [tuple(range(start, start + 3)) for start in range(0, 60, 3)]
        forwards = pd.DataFrame({"sites_enumeration_augmented": [frozenset(variants)]},
                                index=["mp-1"])
        backwards = pd.DataFrame({"sites_enumeration_augmented": [frozenset(reversed(variants))]},
                                 index=["mp-1"])
        dc.save_split(forwards, self.cache, "train")
        first = dc.split_path(self.cache, "train").read_bytes()
        dc.save_split(backwards, self.cache, "train")
        self.assertEqual(first, dc.split_path(self.cache, "train").read_bytes())
        def load():
            return dc.load_split(self.cache, "train")["sites_enumeration_augmented"].iloc[0]

        self.assertEqual(load(), frozenset(variants))
        # Rebuilding a frozenset re-hashes, so it does not iterate in the order
        # it was written in -- but it does iterate the same way every time, which
        # is what a tokenisation has to be able to rely on.
        self.assertEqual(list(load()), list(load()))


class TestBuildRecord(unittest.TestCase):
    """What built a split, recorded in the split."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = Path(tmp.name) / "toy"

    def test_the_record_comes_back_as_it_went_in(self):
        build = dc.provenance("a-tool", max_sites=61, sort_by_letter=True)
        dc.save_split(a_frame(), self.cache, "train", build)
        self.assertEqual(dc.build_info(self.cache, "train"), build)

    def test_provenance_names_the_code_that_ran(self):
        build = dc.provenance("a-tool", max_sites=61)
        self.assertEqual(build["tool"], "a-tool")
        self.assertEqual(build["options"], {"max_sites": 61})
        self.assertRegex(build["built"], r"^\d{4}-\d\d-\d\dT")
        # A checkout gives a commit; a wheel has no git to ask and says so.
        self.assertTrue(build["commit"] is None or len(build["commit"]) == 40)
        self.assertIsInstance(build["dirty"], (bool, type(None)))

    def test_a_split_written_without_one_records_nothing(self):
        # None means "not recorded", never "built with the defaults".
        dc.save_split(a_frame(), self.cache, "train")
        self.assertIsNone(dc.build_info(self.cache, "train"))

    def test_each_split_carries_its_own(self):
        # Splits are not always built together: a slice, or a resumed
        # conversion, writes them one at a time.
        dc.save_split(a_frame(), self.cache, "train", dc.provenance("first"))
        dc.save_split(a_frame(), self.cache, "test", dc.provenance("second"))
        self.assertEqual(
            {split: record["tool"] for split, record in dc.build_info(self.cache).items()},
            {"train": "first", "test": "second"})

    def test_save_cache_records_the_same_build_in_every_split(self):
        build = dc.provenance("a-tool", max_sites=8)
        dc.save_cache({"train": a_frame(), "val": a_frame()}, self.cache, build)
        self.assertEqual(dc.build_info(self.cache), {"train": build, "val": build})

    def test_a_record_that_is_not_plain_data_is_refused(self):
        # It rides in the schema metadata as JSON, so it has to survive JSON --
        # better a refusal at write time than a file nothing can open.
        with self.assertRaisesRegex(dc.CacheFormatError, "survive JSON"):
            dc.save_split(a_frame(), self.cache, "train", {"path": Path("/tmp")})

    def test_the_record_does_not_disturb_the_data(self):
        frame = a_frame()
        dc.save_split(frame, self.cache, "train", dc.provenance("a-tool"))
        loaded = dc.load_split(self.cache, "train")
        self.assertEqual(list(loaded.columns), list(frame.columns))
        self.assertTrue(loaded.index.equals(frame.index))

    def test_a_split_that_is_not_there_is_an_error(self):
        dc.save_split(a_frame(), self.cache, "train")
        with self.assertRaisesRegex(KeyError, "val"):
            dc.build_info(self.cache, "val")


class TestColumnSelection(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = Path(tmp.name) / "toy"
        dc.save_cache({"train": a_frame(), "test": a_frame()}, self.cache)

    def test_only_the_named_columns_come_back_in_the_order_named(self):
        loaded = dc.load_split(self.cache, "train", columns=("elements", "spacegroup_number"))
        self.assertEqual(list(loaded.columns), ["elements", "spacegroup_number"])
        self.assertEqual(loaded.index.name, "material_id")

    def test_no_columns_still_gives_the_index(self):
        loaded = dc.load_split(self.cache, "train", columns=())
        self.assertEqual(list(loaded.columns), [])
        self.assertEqual(list(loaded.index), ["mp-1", "mp-2"])

    def test_a_column_that_is_not_there_is_an_error(self):
        with self.assertRaisesRegex(KeyError, "band_gap"):
            dc.load_split(self.cache, "train", columns=("band_gap",))

    def test_only_the_named_splits_are_read(self):
        self.assertEqual(sorted(dc.load_cache(self.cache, splits=("test",))), ["test"])

    def test_a_split_that_is_not_there_is_an_error(self):
        with self.assertRaisesRegex(KeyError, "val"):
            dc.load_cache(self.cache, splits=("val",))

    def test_available_splits_reports_them_in_a_fixed_order(self):
        self.assertEqual(dc.available_splits(self.cache), ("train", "test"))


class TestTheSupersededFormat(unittest.TestCase):
    """Reading data.pkl.gz, which nothing writes any more."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = Path(tmp.name) / "toy"
        self.cache.mkdir(parents=True)
        self.frames = {"train": a_frame(), "test": a_frame().iloc[:1]}
        with gzip.open(dc.legacy_path(self.cache), "wb") as handle:
            pickle.dump(self.frames, handle)

    def test_a_cache_with_only_a_pickle_still_loads(self):
        loaded = dc.load_cache(self.cache)
        self.assertEqual(sorted(loaded), ["test", "train"])
        self.assertTrue(loaded["train"].equals(self.frames["train"]))

    def test_the_pickle_answers_available_splits_and_exists(self):
        self.assertEqual(dc.available_splits(self.cache), ("train", "test"))
        self.assertTrue(dc.cache_exists(self.cache))

    def test_a_parquet_split_wins_over_the_pickle(self):
        # A converted cache must not be read from the file it was converted
        # from, however long that file is kept around.
        replacement = a_frame().iloc[:1]
        dc.save_split(replacement, self.cache, "train")
        self.assertEqual(len(dc.load_split(self.cache, "train")), 1)
        self.assertEqual(len(dc.load_split(self.cache, "test")), 1)

    def test_columns_are_selected_from_the_pickle_too(self):
        loaded = dc.load_split(self.cache, "train", columns=("elements",))
        self.assertEqual(list(loaded.columns), ["elements"])

    def test_the_pickle_is_read_once_however_many_splits_are_wanted(self):
        # load_split has to unpickle the whole file to reach one split, so a
        # loop that called it per split would read 399 MB three times.
        with patch.object(dc, "load_legacy_cache", wraps=dc.load_legacy_cache) as read:
            splits = [split for split, _ in dc.iter_splits(self.cache)]
        self.assertEqual(splits, ["train", "test"])
        self.assertEqual(read.call_count, 1)

    def test_iter_splits_drops_each_frame_as_it_goes(self):
        import sys

        for _, frame in dc.iter_splits(self.cache):
            # The generator's own reference, this loop's, and getrefcount's.
            self.assertLessEqual(sys.getrefcount(frame), 4)

    def test_the_superseded_format_records_no_build(self):
        # Its options are not in the pickle and not anywhere else.
        self.assertEqual(dc.build_info(self.cache), {"train": None, "test": None})

    def test_the_pickles_path_names_the_same_cache(self):
        self.assertEqual(dc.as_cache_dir(dc.legacy_path(self.cache)), self.cache)
        self.assertEqual(sorted(dc.load_cache(dc.legacy_path(self.cache))), ["test", "train"])


class TestRefusals(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_an_empty_directory_is_a_missing_cache(self):
        with self.assertRaisesRegex(FileNotFoundError, "No dataset cache"):
            dc.load_cache(self.tmp)
        self.assertFalse(dc.cache_exists(self.tmp))
        self.assertEqual(dc.available_splits(self.tmp), ())

    def test_a_parquet_this_module_did_not_write_is_refused(self):
        # Without the metadata the Python type of every object column is a
        # guess, and guessing would hand a reader lists where it wanted a
        # frozenset of variants.
        dc.save_split(a_frame(), self.tmp, "train")
        path = dc.split_path(self.tmp, "train")
        table = pq.read_table(path)
        pq.write_table(table.replace_schema_metadata(None), path)
        with self.assertRaisesRegex(dc.CacheFormatError, "metadata"):
            dc.load_split(self.tmp, "train")

    def test_a_newer_format_version_is_refused(self):
        dc.save_split(a_frame(), self.tmp, "train")
        path = dc.split_path(self.tmp, "train")
        table = pq.read_table(path)
        metadata = dict(table.schema.metadata)
        raw = metadata[dc.METADATA_KEY].replace(
            b'"version": 1', f'"version": {dc.FORMAT_VERSION + 1}'.encode())
        metadata[dc.METADATA_KEY] = raw
        pq.write_table(table.replace_schema_metadata(metadata), path)
        with self.assertRaisesRegex(dc.CacheFormatError, "format version"):
            dc.load_split(self.tmp, "train")

    def test_an_index_that_collides_with_a_column_is_refused(self):
        frame = a_frame()
        frame.index.name = "elements"
        with self.assertRaisesRegex(dc.CacheFormatError, "also a column"):
            dc.save_split(frame, self.tmp, "train")

    def test_an_interrupted_write_leaves_no_half_file(self):
        # A rebuild that dies partway must leave the cache that was there, not
        # a truncated file that fails halfway through the next training run.
        frame = a_frame()
        dc.save_split(frame, self.tmp, "train")
        original = dc.split_path(self.tmp, "train").read_bytes()
        broken = frame.copy()
        broken.loc["mp-2", "elements"] = 5     # fails once the first chunk is written
        with patch.object(dc, "WRITE_CHUNK_ROWS", 1), self.assertRaises(TypeError):
            dc.save_split(broken, self.tmp, "train")
        self.assertEqual(dc.split_path(self.tmp, "train").read_bytes(), original)
        self.assertEqual(list(self.tmp.glob("*.partial")), [])


class TestPaths(unittest.TestCase):
    def test_either_spelling_of_a_cache_names_the_directory(self):
        self.assertEqual(dc.as_cache_dir("cache/x"), Path("cache/x"))
        self.assertEqual(dc.as_cache_dir("cache/x/data.pkl.gz"), Path("cache/x"))

    def test_a_split_is_named_after_itself(self):
        self.assertEqual(dc.split_path(Path("cache/x"), "val"), Path("cache/x/val.parquet"))


if __name__ == "__main__":
    unittest.main()
