"""Tests for the cryspr subpackage.

Unit tests run without network access and without MACE installed by mocking
heavy dependencies.  Integration tests are marked ``needs_relax`` and require
``--run-relax`` to be passed to pytest.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

TEST_MODEL_URL = (
    "https://github.com/ACEsuit/mace-foundations/releases/download/"
    "mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model"
)

NACL_GENE = {
    "group": 225,
    "species": ["Na", "Cl"],
    "numIons": [4, 4],
    "sites": [["4a"], ["4b"]],
}


# ---------------------------------------------------------------------------
# resolve_model_path / _download_and_cache
# ---------------------------------------------------------------------------

class TestResolveModelPath(unittest.TestCase):
    def test_local_path_returned_unchanged(self):
        from wyckoff_transformer.cryspr.calculator import resolve_model_path
        result = resolve_model_path("/some/local/model.model")
        self.assertEqual(result, Path("/some/local/model.model"))

    def test_pathlib_path_returned_unchanged(self):
        from wyckoff_transformer.cryspr.calculator import resolve_model_path
        p = Path("/another/path.model")
        self.assertEqual(resolve_model_path(p), p)

    def test_http_url_triggers_download(self):
        from wyckoff_transformer.cryspr.calculator import resolve_model_path, _download_and_cache
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            with patch("urllib.request.urlretrieve") as mock_dl:
                mock_dl.side_effect = lambda url, dest: Path(dest).write_bytes(b"fake")
                with patch("wyckoff_transformer.cryspr.calculator._DEFAULT_CACHE_DIR", cache_dir):
                    result = resolve_model_path("http://example.com/model.model")
            self.assertTrue(result.exists())
            mock_dl.assert_called_once()

    def test_https_url_triggers_download(self):
        from wyckoff_transformer.cryspr.calculator import resolve_model_path
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            with patch("urllib.request.urlretrieve") as mock_dl:
                mock_dl.side_effect = lambda url, dest: Path(dest).write_bytes(b"fake")
                with patch("wyckoff_transformer.cryspr.calculator._DEFAULT_CACHE_DIR", cache_dir):
                    result = resolve_model_path("https://example.com/model.model")
            self.assertTrue(result.exists())
            mock_dl.assert_called_once()


class TestDownloadAndCache(unittest.TestCase):
    def test_file_written_to_cache(self):
        from wyckoff_transformer.cryspr.calculator import _download_and_cache
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            with patch("urllib.request.urlretrieve") as mock_dl:
                mock_dl.side_effect = lambda url, dest: Path(dest).write_bytes(b"data")
                result = _download_and_cache("https://example.com/m.model", cache_dir=cache_dir)
            self.assertTrue(result.exists())
            self.assertEqual(result.read_bytes(), b"data")

    def test_second_call_skips_download(self):
        from wyckoff_transformer.cryspr.calculator import _download_and_cache
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            with patch("urllib.request.urlretrieve") as mock_dl:
                mock_dl.side_effect = lambda url, dest: Path(dest).write_bytes(b"data")
                _download_and_cache("https://example.com/m.model", cache_dir=cache_dir)
                _download_and_cache("https://example.com/m.model", cache_dir=cache_dir)
            self.assertEqual(mock_dl.call_count, 1)

    def test_different_urls_get_different_files(self):
        from wyckoff_transformer.cryspr.calculator import _download_and_cache
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            with patch("urllib.request.urlretrieve") as mock_dl:
                mock_dl.side_effect = lambda url, dest: Path(dest).write_bytes(b"x")
                p1 = _download_and_cache("https://example.com/a.model", cache_dir=cache_dir)
                p2 = _download_and_cache("https://example.com/b.model", cache_dir=cache_dir)
            self.assertNotEqual(p1, p2)

    def test_failed_download_leaves_no_partial_file(self):
        from wyckoff_transformer.cryspr.calculator import _download_and_cache
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            with patch("urllib.request.urlretrieve", side_effect=OSError("network error")):
                with self.assertRaises(OSError):
                    _download_and_cache("https://example.com/m.model", cache_dir=cache_dir)
            # No .tmp file left behind
            self.assertEqual(list(cache_dir.glob("*.tmp")), [])


# ---------------------------------------------------------------------------
# single_pyxtal (mocked PyXtal)
# ---------------------------------------------------------------------------

class TestSinglePyxtal(unittest.TestCase):
    def test_returns_none_on_exception(self):
        from wyckoff_transformer.cryspr.generator import single_pyxtal
        with tempfile.TemporaryDirectory() as tmp:
            with patch("wyckoff_transformer.cryspr.generator.pyxtal") as MockPyXtal:
                MockPyXtal.return_value.from_random.side_effect = RuntimeError("fail")
                result = single_pyxtal(
                    wyckoffgene=NACL_GENE,
                    wdir=Path(tmp),
                )
        self.assertIsNone(result)

    def test_returns_atoms_on_success(self):
        from wyckoff_transformer.cryspr.generator import single_pyxtal
        from ase.build import bulk
        mock_atoms = bulk("NaCl", "rocksalt", a=5.64)

        with tempfile.TemporaryDirectory() as tmp:
            with patch("wyckoff_transformer.cryspr.generator.pyxtal") as MockPyXtal:
                inst = MockPyXtal.return_value
                inst.from_random.return_value = None
                inst.to_ase.return_value = mock_atoms
                inst.to_file.return_value = None
                result = single_pyxtal(
                    wyckoffgene=NACL_GENE,
                    wdir=Path(tmp),
                )
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# func_run — unit test (all trials fail)
# ---------------------------------------------------------------------------

class TestFuncRunAllFailed(unittest.TestCase):
    def test_returns_none_tuple_when_all_trials_fail(self):
        from wyckoff_transformer.cryspr.generator import func_run
        mock_calc = MagicMock()
        with patch("wyckoff_transformer.cryspr.generator.single_pyxtal", return_value=None):
            with tempfile.TemporaryDirectory() as tmp:
                result = func_run(
                    id_gene=0,
                    wyckoffgene=NACL_GENE,
                    calculator=mock_calc,
                    output_dir=Path(tmp),
                    n_trials=2,
                )
        self.assertEqual(result, (None, None, None, None))


# ---------------------------------------------------------------------------
# Integration test — requires --run-relax and network access
# ---------------------------------------------------------------------------

@pytest.mark.needs_relax
class TestFuncRunIntegration(unittest.TestCase):
    """Download a MACE model and run a single NaCl relaxation trial."""

    @classmethod
    def setUpClass(cls):
        from wyckoff_transformer.cryspr.calculator import build_mace_calculator
        cls.calculator = build_mace_calculator(model=TEST_MODEL_URL)

    def test_nacl_relaxation_produces_negative_energy(self):
        from wyckoff_transformer.cryspr.generator import func_run
        with tempfile.TemporaryDirectory() as tmp:
            atoms, formula, energy, energy_per_atom = func_run(
                id_gene=0,
                wyckoffgene=NACL_GENE,
                calculator=self.calculator,
                output_dir=Path(tmp),
                n_trials=1,
            )
        self.assertIsNotNone(atoms, "atoms should not be None for a successful relaxation")
        self.assertIsNotNone(formula)
        self.assertLess(energy, 0.0, "Relaxed NaCl energy should be negative")
        self.assertLess(energy_per_atom, 0.0)
