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
# pyxtal_tol_matrix
# ---------------------------------------------------------------------------

class TestPyxtalTolMatrix(unittest.TestCase):
    def test_the_default_factor_is_the_shipped_one(self):
        from wyckoff_transformer.cryspr.generator import (
            DEFAULT_PYXTAL_TOL_FACTOR,
            _DEFAULT_IADM,
            pyxtal_tol_matrix,
        )
        self.assertEqual(DEFAULT_PYXTAL_TOL_FACTOR, 1.3)
        self.assertIs(pyxtal_tol_matrix(), _DEFAULT_IADM)

    def test_the_same_factor_is_not_rebuilt(self):
        """The cache is the reason this helper exists.

        A `Tol_matrix(prototype="atomic")` fills a 100x100 radius array, which
        the generate stage would otherwise pay for once per draw.
        """
        from wyckoff_transformer.cryspr.generator import pyxtal_tol_matrix
        self.assertIs(pyxtal_tol_matrix(0.63), pyxtal_tol_matrix(0.63))
        self.assertIs(pyxtal_tol_matrix(0.63), pyxtal_tol_matrix(0.63000))

    def test_a_different_factor_gets_its_own_matrix(self):
        from wyckoff_transformer.cryspr.generator import pyxtal_tol_matrix
        strict, loose = pyxtal_tol_matrix(1.3), pyxtal_tol_matrix(0.4)
        self.assertIsNot(strict, loose)
        # The tolerance is linear in the factor, and that is what "permissive"
        # means here: the same pair, a shorter allowed contact.
        self.assertAlmostEqual(
            loose.get_tol(11, 17) / strict.get_tol(11, 17), 0.4 / 1.3, places=6
        )

    def test_the_clash_guard_is_left_alone(self):
        """A post-relaxation guard and a generation floor are different things.

        `has_atomic_clash` rejects MACE's collapse artifacts. If it followed
        `--pyxtal-tol-factor` down, a permissive draw would also switch off the
        guard that catches a *relaxed* structure collapsing, and the two
        effects would be inseparable.
        """
        import inspect

        from wyckoff_transformer.cryspr.generator import (
            _CLASH_IADM,
            has_atomic_clash,
            pyxtal_tol_matrix,
        )

        self.assertIs(
            inspect.signature(has_atomic_clash).parameters["iadm"].default,
            _CLASH_IADM,
        )
        # Still factor 1.1, and still its own object: it must not be served out
        # of the generation cache, where a caller could reach it by factor.
        self.assertAlmostEqual(
            _CLASH_IADM.get_tol(11, 17), pyxtal_tol_matrix(1.1).get_tol(11, 17)
        )
        self.assertIsNot(_CLASH_IADM, pyxtal_tol_matrix(1.1))


class TestPermissiveFactorDrawsCloserContacts(unittest.TestCase):
    """The factor has to change the draws, not just the manifest.

    Real PyXtal draws rather than a mock, because what is being asserted is a
    property of PyXtal's rejection sampling. `from_random` draws from its own
    RNG, which `single_pyxtal` does not expose a seed for, so the test is
    statistical: over 20 draws of a 12-DoF gene the permissive arm's closest
    contact is lower than the strict arm's in 3000 of 3000 bootstrap resamples
    of a 200-draw-per-arm measurement (strict: min 0.94, 10% below 1.0;
    permissive at 0.2: min 0.30, 48% below 1.0).
    """

    #: A gene with 12 free coordinates in a 16-atom cell, from the oracle
    #: cohort.  Free coordinates are the whole point: for a gene whose orbits
    #: are all fixed there is nothing for the floor to reject, and the arms
    #: would differ only through the cell PyXtal guesses.
    GENE = {
        "group": 33,
        "species": ["Cs", "Ag", "Te"],
        "numIons": [4, 4, 8],
        "sites": [["4a"], ["4a"], ["4a", "4a"]],
    }

    N_DRAWS = 20
    PERMISSIVE = 0.2

    @staticmethod
    def _closest_contact(atoms) -> float:
        """`min(d / (0.5*(r_a+r_b)))` over pairs of distinct atoms.

        Measured at factor 1.0 whatever the draw used, so the number is a
        physical ratio; in each arm's own units every arm's floor would be 1.0
        by construction.
        """
        import numpy as np
        from ase.neighborlist import neighbor_list
        from pyxtal.tolerance import Tol_matrix

        tm = Tol_matrix(prototype="atomic", factor=1.0)
        numbers = atoms.numbers
        unique = sorted({int(n) for n in numbers})
        # Grown until the nearest distinct pair is inside it: a loose draw can
        # have nothing at all within one tolerance.
        cutoff = 2.5 * max(tm.get_tol(a, b) for a in unique for b in unique)
        while True:
            first, second, dist = neighbor_list("ijd", atoms, cutoff)
            distinct = first != second
            if distinct.any() or cutoff > 40.0:
                break
            cutoff *= 2
        tols = np.array([tm.get_tol(int(numbers[a]), int(numbers[b]))
                         for a, b in zip(first, second)])
        return float(np.min((dist / tols)[distinct]))

    def _contacts(self, factor: float) -> list[float]:
        from wyckoff_transformer.cryspr.generator import pyxtal_tol_matrix, single_pyxtal

        contacts = []
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(self.N_DRAWS):
                atoms = single_pyxtal(
                    wyckoffgene=self.GENE,
                    iadm=pyxtal_tol_matrix(factor),
                    nlimit=30,
                    wdir=Path(tmp),
                )
                if atoms is not None:
                    contacts.append(self._closest_contact(atoms))
        return contacts

    def test_a_permissive_factor_admits_closer_contacts(self):
        from wyckoff_transformer.cryspr.generator import DEFAULT_PYXTAL_TOL_FACTOR

        strict = self._contacts(DEFAULT_PYXTAL_TOL_FACTOR)
        loose = self._contacts(self.PERMISSIVE)
        self.assertGreater(len(strict), 0, "PyXtal drew nothing at all")
        self.assertGreater(len(loose), 0, "PyXtal drew nothing at all")
        self.assertLess(min(loose), min(strict))
        # The nominal floor is 0.65*(r_a+r_b), i.e. 1.3 in these units, and the
        # strict arm still goes below 1.0 -- the floor biases the draw rather
        # than bounding it.  A permissive arm has to go well below it.
        self.assertLess(min(loose), 1.0)


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

        from wyckoff_transformer.cryspr.relaxer import RelaxStages

        with tempfile.TemporaryDirectory() as tmp:
            with patch("wyckoff_transformer.cryspr.generator.single_pyxtal",
                       return_value=relaxed.copy()), \
                 patch("wyckoff_transformer.cryspr.generator.stepwise_relax_stages",
                       return_value=RelaxStages(kept=relaxed, prerattle=relaxed)), \
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
