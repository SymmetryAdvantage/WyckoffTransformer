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
        self.assertEqual(result, (None, None, None, None, None))


# ---------------------------------------------------------------------------
# is_mace_calculator
# ---------------------------------------------------------------------------

def _calculator_from_module(module: str):
    """An object whose class claims to be defined in *module*."""
    klass = type("FakeCalculator", (), {"__module__": module})
    return klass()


class TestIsMaceCalculator(unittest.TestCase):
    def test_mace_calculator_module_is_recognised(self):
        from wyckoff_transformer.cryspr.generator import is_mace_calculator
        self.assertTrue(is_mace_calculator(_calculator_from_module("mace.calculators.mace")))

    def test_bare_mace_package_is_recognised(self):
        from wyckoff_transformer.cryspr.generator import is_mace_calculator
        self.assertTrue(is_mace_calculator(_calculator_from_module("mace")))

    def test_subclass_defined_outside_mace_is_recognised(self):
        """build_mace_calculator returns a local subclass, so the MRO must be walked."""
        from wyckoff_transformer.cryspr.generator import is_mace_calculator
        base = type("MACECalculator", (), {"__module__": "mace.calculators.mace"})
        derived = type("_IsolatedMACECalculator", (base,), {
            "__module__": "wyckoff_transformer.cryspr.calculator",
        })
        self.assertTrue(is_mace_calculator(derived()))

    def test_other_backends_are_not_recognised(self):
        from wyckoff_transformer.cryspr.generator import is_mace_calculator
        for module in ("tace.interface.ase.calculator", "upet.calculator",
                       "tensorpotential.calculator", "ase.calculators.emt"):
            with self.subTest(module=module):
                self.assertFalse(is_mace_calculator(_calculator_from_module(module)))

    def test_a_package_merely_starting_with_mace_is_not_recognised(self):
        """mace-torch is the only 'mace'; mace_layer or macetools are not it."""
        from wyckoff_transformer.cryspr.generator import is_mace_calculator
        for module in ("mace_extras.calculator", "macelike"):
            with self.subTest(module=module):
                self.assertFalse(is_mace_calculator(_calculator_from_module(module)))

    def test_mock_calculator_is_not_recognised(self):
        from wyckoff_transformer.cryspr.generator import is_mace_calculator
        self.assertFalse(is_mace_calculator(MagicMock()))

    def test_the_real_mace_calculator_class_is_recognised(self):
        from wyckoff_transformer.cryspr.generator import is_mace_calculator
        try:
            from mace.calculators import MACECalculator
        except ImportError:
            self.skipTest("mace-torch is not installed")
        # Instantiating MACECalculator needs a checkpoint; the detector only
        # reads the class hierarchy, so an uninitialised instance is enough.
        self.assertTrue(is_mace_calculator(MACECalculator.__new__(MACECalculator)))


# ---------------------------------------------------------------------------
# func_run — the clash guard is MACE-only
# ---------------------------------------------------------------------------

class TestFuncRunClashGuard(unittest.TestCase):
    """The guard exists for MACE's short-range collapse, so it engages only there.

    ``has_atomic_clash`` is patched to report a clash unconditionally: what is
    under test is whether func_run consults it and discards the trial, not the
    geometric criterion itself.
    """

    def _run(self, calculator, **kwargs):
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator
        from wyckoff_transformer.cryspr.generator import func_run

        relaxed = Atoms("Na2", positions=[[0, 0, 0], [2.8, 0, 0]],
                        cell=[5.6, 5.6, 5.6], pbc=True)
        relaxed.calc = SinglePointCalculator(relaxed, energy=-7.0)

        with tempfile.TemporaryDirectory() as tmp:
            with patch("wyckoff_transformer.cryspr.generator.single_pyxtal",
                       return_value=relaxed.copy()), \
                 patch("wyckoff_transformer.cryspr.generator.stepwise_relax",
                       return_value=relaxed), \
                 patch("wyckoff_transformer.cryspr.generator.has_atomic_clash",
                       return_value=True) as mock_clash:
                result = func_run(
                    id_gene=0,
                    wyckoffgene=NACL_GENE,
                    calculator=calculator,
                    output_dir=Path(tmp),
                    n_trials=1,
                    **kwargs,
                )
        return result, mock_clash

    def test_non_mace_calculator_skips_the_guard(self):
        result, mock_clash = self._run(_calculator_from_module("upet.calculator"))
        mock_clash.assert_not_called()
        self.assertIsNotNone(result[0], "the trial should have been kept")
        self.assertEqual(result[2], -7.0)

    def test_mace_calculator_engages_the_guard(self):
        result, mock_clash = self._run(_calculator_from_module("mace.calculators.mace"))
        mock_clash.assert_called_once()
        self.assertEqual(result, (None, None, None, None, None),
                         "the only trial clashed, so no structure survives")

    def test_clash_guard_true_forces_the_guard_on_a_non_mace_calculator(self):
        result, mock_clash = self._run(_calculator_from_module("upet.calculator"),
                                       clash_guard=True)
        mock_clash.assert_called_once()
        self.assertEqual(result, (None, None, None, None, None))

    def test_clash_guard_false_forces_the_guard_off_for_mace(self):
        result, mock_clash = self._run(_calculator_from_module("mace.calculators.mace"),
                                       clash_guard=False)
        mock_clash.assert_not_called()
        self.assertIsNotNone(result[0])


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
            atoms, formula, energy, energy_per_atom, cif = func_run(
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
        self.assertIsNotNone(cif, "CIF content should be returned for a successful relaxation")
        self.assertIn("_cell_length_a", cif)
