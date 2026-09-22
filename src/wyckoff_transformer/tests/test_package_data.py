"""The Wyckoff mappings and engineers are committed package data, and must not change.

A model is only meaningful against the exact tables it was trained with: the harmonic
cluster engineer, for one, decides which Wyckoff letter a generated token means. These
tests fail when

- a committed file changes (``PACKAGE_DATA_SHA256``);
- the generator would now produce something else -- a dependency update is the usual
  cause: pyxtal's Wyckoff tables, scipy's spherical harmonics, scikit-learn's KMeans;
- a model directory stops carrying its own copy of the data, or loading one reads the
  package's instead.

If a change is intended, regenerate with ``python -m wyckoff_transformer.preprocess_wychoffs``,
review the diff, and update ``PACKAGE_DATA_SHA256``. Models trained before the change
keep working only through the copy saved with them.
"""
import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from .. import preprocess_wychoffs
from .. import tokenization as tok
from .. import wyckoff_processor
from ..wyckoff_processor import ENGINEERS_DIR, MODEL_ENGINEERS_DIRNAME, engineers_dir_for

PACKAGE_DIR = Path(preprocess_wychoffs.__file__).resolve().parent

#: Package-relative path -> sha256 of the committed bytes.
PACKAGE_DATA_SHA256 = {
    "wyckoffs_enumerated_by_ss.json": "ed3ad3777b1eec3273461f7e4dac44f32325d3a80c5813cb3404694334d4694b",
    "engineers/multiplicity.json": "e4126ef022763e4e349160c1123fba422d31524cf3e459ec10299c67dd0fbee4",
    "engineers/site_symmetry_ops.json": "5f954fdade2e5b92067cad2b5c500eb4c51b7da43c7a5ef3d700bfeae47b6d61",
    "engineers/site_symmetry_ops_id.json": "7b3259f2cad6132b7ed8c6d149d65f9d674c2147e7f22634bf5e67c2523a2e25",
    "engineers/site_symmetry_ops_id_table.json": "8df417f15924989d8f1b891eeee93eeb1dd53d164d5e649f750057c9b25fede1",
}

#: Regenerated floats may differ from the committed ones by this much. The signatures are
#: rounded to 1e-12 before they are written, so a different libm can still move the last
#: digit; every integer, string and key must match exactly.
FLOAT_TOLERANCE = 1e-10


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _differences(expected, actual, where="$"):
    """Yield the paths at which two JSON documents differ, floats compared with tolerance."""
    if isinstance(expected, float) or isinstance(actual, float):
        if not (isinstance(expected, (int, float)) and isinstance(actual, (int, float))
                and math.isclose(expected, actual, rel_tol=0, abs_tol=FLOAT_TOLERANCE)):
            yield f"{where}: {expected!r} != {actual!r}"
    elif isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            yield f"{where}: keys differ by {sorted(set(expected) ^ set(actual))[:5]}"
            return
        for key in expected:
            yield from _differences(expected[key], actual[key], f"{where}.{key}")
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            yield f"{where}: length {len(expected)} != {len(actual)}"
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            yield from _differences(left, right, f"{where}[{index}]")
    elif expected != actual:
        yield f"{where}: {expected!r} != {actual!r}"


class TestCommittedPackageData(unittest.TestCase):
    def test_files_are_exactly_the_committed_ones(self):
        on_disk = {path.relative_to(PACKAGE_DIR).as_posix() for path in
                   [PACKAGE_DIR / "wyckoffs_enumerated_by_ss.json", *ENGINEERS_DIR.glob("*.json")]
                   if path.exists()}
        self.assertEqual(on_disk, set(PACKAGE_DATA_SHA256))
        for relative, digest in PACKAGE_DATA_SHA256.items():
            with self.subTest(relative):
                self.assertEqual(_sha256(PACKAGE_DIR / relative), digest,
                                 f"{relative} changed; see this module's docstring")


class TestRegenerationReproducesCommittedData(unittest.TestCase):
    """Runs the whole generator (~12 s) into a temporary directory and compares."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            preprocess_wychoffs.enumerate_wychoffs_by_ss(
                output_file=cls.out / "wyckoffs_enumerated_by_ss.json",
                engineers_dir=cls.out / "engineers")
            preprocess_wychoffs.build_site_symmetry_ops_engineer(engineers_dir=cls.out / "engineers")
            preprocess_wychoffs.build_site_symmetry_ops_id_engineer(engineers_dir=cls.out / "engineers")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_committed_file_is_reproduced(self):
        for relative in PACKAGE_DATA_SHA256:
            with self.subTest(relative):
                committed = json.loads((PACKAGE_DIR / relative).read_text(encoding="utf-8"))
                regenerated = json.loads((self.out / relative).read_text(encoding="utf-8"))
                differences = list(_differences(committed, regenerated))
                self.assertEqual(
                    differences[:10], [],
                    f"{relative}: regeneration differs in {len(differences)} places. A "
                    f"dependency update is the likely cause; see this module's docstring.")


def _tiny_datasets() -> dict:
    rows = [
        {"spacegroup_number": 1, "elements": ["H", "O"],
         "site_symmetries": ["1", "1"], "sites_enumeration": [0, 0]},
        {"spacegroup_number": 2, "elements": ["Na", "Cl"],
         "site_symmetries": ["-1", "-1"], "sites_enumeration": [0, 1]},
    ]
    return {"train": pd.DataFrame(rows), "val": pd.DataFrame(rows[:1])}


def _multiplicity_config():
    return OmegaConf.create({
        "dtype": "int64",
        "include_stop": True,
        "token_fields": {
            "pure_categorical": ["elements", "site_symmetries", "sites_enumeration"],
            "engineered": {"multiplicity": {
                "type": "map",
                "inputs": ["spacegroup_number", "site_symmetries", "sites_enumeration"]}},
        },
        "sequence_fields": {"space_group": ["spacegroup_number"]},
    })


class TestModelCarriesItsPackageData(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.model_dir = Path(self._tmp.name) / "run"
        # pandarallel forks worker processes; the tokenisation here is tiny
        patcher = patch.object(tok.pandarallel, "initialize")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _train_processor(self):
        """What train_from_config leaves behind: package data, then the processor."""
        tok.save_package_data(self.model_dir)
        processor = tok.WyckoffProcessor.from_config(_multiplicity_config())
        with patch("pandas.DataFrame.parallel_apply", pd.DataFrame.apply, create=True):
            processor.tokenise_dataset(_tiny_datasets())
        processor.save_pretrained(self.model_dir)
        return processor

    def _retokenise(self):
        processor = tok.WyckoffProcessor(config=_multiplicity_config())
        with patch("pandas.DataFrame.parallel_apply", pd.DataFrame.apply, create=True):
            return processor.tokenise_dataset(
                _tiny_datasets(), tokenizer_path=self.model_dir / "wyckoff_processor.json")

    def test_save_package_data_copies_every_file(self):
        written = tok.save_package_data(self.model_dir)
        self.assertEqual(
            {path.relative_to(self.model_dir).as_posix() for path in written},
            set(PACKAGE_DATA_SHA256))
        for relative, digest in PACKAGE_DATA_SHA256.items():
            with self.subTest(relative):
                self.assertEqual(_sha256(self.model_dir / relative), digest)

    def test_a_model_directory_resolves_to_its_own_engineers(self):
        self.assertEqual(engineers_dir_for(self.model_dir), ENGINEERS_DIR)
        tok.save_package_data(self.model_dir)
        self.assertEqual(engineers_dir_for(self.model_dir), self.model_dir / MODEL_ENGINEERS_DIRNAME)
        self.assertEqual(engineers_dir_for(None), ENGINEERS_DIR)

    def test_a_model_loads_its_mappings_without_the_package(self):
        tok.save_package_data(self.model_dir)
        with patch.object(tok, "_PACKAGE_MAPPINGS_PATH", Path(self._tmp.name) / "absent.json"):
            mappings = tok.load_wyckoff_mappings(self.model_dir)
        self.assertEqual(mappings.letter_from_ss_enum[218]["-4.."], {0: "c", 1: "d"})

    def test_retokenising_reads_the_model_engineers_not_the_package(self):
        trained = self._train_processor()
        with patch.object(wyckoff_processor, "ENGINEERS_DIR", Path(self._tmp.name) / "absent"):
            _, _, token_engineers = self._retokenise()
        self.assertTrue(token_engineers["multiplicity"].db.equals(
            trained.token_engineers["multiplicity"].db))

    def test_retokenising_refuses_engineers_that_differ_from_the_saved_ones(self):
        self._train_processor()
        engineer_json = self.model_dir / MODEL_ENGINEERS_DIRNAME / "multiplicity.json"
        payload = json.loads(engineer_json.read_text(encoding="utf-8"))
        for entry in payload["entries"]:
            if entry[0] == [2, "-1", 1]:
                entry[1] += 1
        engineer_json.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from the one saved with the model"):
            self._retokenise()


if __name__ == "__main__":
    unittest.main()
