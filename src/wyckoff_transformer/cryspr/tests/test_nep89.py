"""NEP89, its Lennard-Jones fallback, and the selection the pre-screen makes.

Everything here runs offline except :class:`TestNep89Model`, which is marked
``needs_relax`` because it downloads the 15 MB model file and needs
``calorine``.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from wyckoff_transformer.cryspr.nep89 import (
    COULOMB_EV_ANGSTROM,
    NEP89_MODEL_SHA256,
    NEP89_MODEL_URL,
    Nep89WithFallback,
    ScreenedMorse,
    SpeciesLennardJones,
    ZBL_COEFFICIENTS,
    ZBL_EXPONENTS,
    ZBL_SCREENING_CONSTANT,
    ZBL_SCREENING_POWER,
    _lj_pair,
    nep89_elements,
)
from wyckoff_transformer.cryspr.prescreen import Candidate, select_candidates

#: A NEP header of the shape the real one has, so the parser can be tested
#: without the 15 MB download.
FAKE_HEADER = "nep4_zbl 3 Si O Na\nzbl 1 2\ncutoff 6 5 274 158\n"


def _write_model(directory: Path, header: str = FAKE_HEADER) -> Path:
    path = Path(directory) / "nep.txt"
    path.write_text(header, encoding="utf-8")
    return path


def _rattled_cell() -> Atoms:
    """A four-species cell with no symmetry, so no derivative is zero by accident."""
    atoms = Atoms(
        "NaClMgO",
        cell=np.diag([5.2, 5.4, 5.6]),
        pbc=True,
        scaled_positions=[
            (0.02, 0.03, 0.01), (0.51, 0.49, 0.02),
            (0.48, 0.02, 0.52), (0.03, 0.53, 0.49),
        ],
    )
    atoms.rattle(0.05, seed=3)
    return atoms


class TestLennardJonesDerivatives(unittest.TestCase):
    """The fallback relaxes a cell, so its forces and stress must be the real ones.

    A wrong sign or a missing factor here would not raise: the optimiser would
    simply walk the structure somewhere, and the only symptom would be a
    pre-relaxation that makes the scoring stage's job harder rather than easier.
    """

    STEP = 1e-5

    def setUp(self):
        self.atoms = _rattled_cell()
        self.atoms.calc = SpeciesLennardJones()

    def test_forces_match_finite_differences(self):
        analytic = self.atoms.get_forces()
        numeric = np.zeros_like(analytic)
        positions = self.atoms.get_positions()
        for i in range(len(self.atoms)):
            for axis in range(3):
                shifted = positions.copy()
                shifted[i, axis] += self.STEP
                self.atoms.set_positions(shifted)
                plus = self.atoms.get_potential_energy()
                shifted[i, axis] -= 2 * self.STEP
                self.atoms.set_positions(shifted)
                minus = self.atoms.get_potential_energy()
                numeric[i, axis] = -(plus - minus) / (2 * self.STEP)
        self.atoms.set_positions(positions)
        self.assertGreater(abs(analytic).max(), 1e-3)  # not a trivially flat point
        np.testing.assert_allclose(analytic, numeric, atol=1e-7)

    def test_stress_matches_finite_differences(self):
        analytic = self.atoms.get_stress(voigt=False)
        cell = self.atoms.cell.array.copy()
        scaled = self.atoms.get_scaled_positions().copy()
        volume = self.atoms.get_volume()
        numeric = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                strain = np.zeros((3, 3))
                strain[a, b] += self.STEP / 2
                strain[b, a] += self.STEP / 2
                self.atoms.set_cell(cell @ (np.eye(3) + strain), scale_atoms=True)
                plus = self.atoms.get_potential_energy()
                self.atoms.set_cell(cell @ (np.eye(3) - strain), scale_atoms=True)
                minus = self.atoms.get_potential_energy()
                numeric[a, b] = (plus - minus) / (2 * self.STEP) / volume
        self.atoms.set_cell(cell, scale_atoms=True)
        self.atoms.set_scaled_positions(scaled)
        self.assertGreater(abs(analytic).max(), 1e-4)
        np.testing.assert_allclose(analytic, numeric, atol=1e-8)


class TestLennardJonesShape(unittest.TestCase):
    def test_the_minimum_sits_at_the_sum_of_the_covalent_radii(self):
        """Which is the contact criterion PyXtal's tolerance matrix also uses."""
        from ase.data import atomic_numbers, covalent_radii

        calculator = SpeciesLennardJones()
        sigma, index = calculator.sigma_matrix([atomic_numbers["Na"], atomic_numbers["Cl"]])
        pair = sigma[index[atomic_numbers["Na"]], index[atomic_numbers["Cl"]]]
        expected = covalent_radii[atomic_numbers["Na"]] + covalent_radii[atomic_numbers["Cl"]]
        self.assertAlmostEqual(pair * 2 ** (1 / 6), expected, places=10)

    def test_species_get_different_sizes(self):
        """The whole reason not to use ASE's single-sigma LennardJones."""
        from ase.data import atomic_numbers

        calculator = SpeciesLennardJones()
        numbers = [atomic_numbers["H"], atomic_numbers["Cs"]]
        sigma, index = calculator.sigma_matrix(numbers)
        self.assertLess(
            sigma[index[atomic_numbers["H"]], index[atomic_numbers["H"]]],
            sigma[index[atomic_numbers["Cs"]], index[atomic_numbers["Cs"]]],
        )

    def test_the_energy_is_continuous_at_the_cutoff(self):
        sigma = np.array([2.0])
        cutoff = 2.5 * sigma
        energy, _ = _lj_pair(cutoff, sigma, 1.0, cutoff)
        self.assertAlmostEqual(float(energy[0]), 0.0, places=12)

    def test_overlapping_atoms_are_pushed_apart(self):
        """The one behaviour the pre-relaxation actually needs from the fallback."""
        atoms = Atoms("Si2", cell=np.diag([8.0, 8.0, 8.0]), pbc=True,
                      positions=[(0, 0, 0), (0.8, 0, 0)])
        atoms.calc = SpeciesLennardJones()
        force = atoms.get_forces()
        self.assertLess(force[0, 0], 0.0)   # atom 0 pushed away from atom 1
        self.assertGreater(force[1, 0], 0.0)

    def test_a_nonsense_epsilon_is_refused(self):
        with self.assertRaises(ValueError):
            SpeciesLennardJones(epsilon=0.0)
        with self.assertRaises(ValueError):
            SpeciesLennardJones(cutoff_sigma=1.0)


class TestElementParsing(unittest.TestCase):
    def test_the_element_set_comes_from_the_model_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(nep89_elements(_write_model(Path(tmp))), frozenset({"Si", "O", "Na"}))

    def test_a_header_that_lies_about_its_count_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_model(Path(tmp), "nep4_zbl 5 Si O Na\n")
            with self.assertRaises(ValueError):
                nep89_elements(path)

    def test_a_header_listing_a_non_element_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_model(Path(tmp), "nep4_zbl 2 Si Xx\n")
            with self.assertRaises(ValueError):
                nep89_elements(path)

    def test_a_missing_model_file_is_refused(self):
        from wyckoff_transformer.cryspr.nep89 import resolve_nep89_model

        with self.assertRaises(FileNotFoundError):
            resolve_nep89_model("/nonexistent/nep.txt")

    def test_the_checkpoint_is_pinned_by_commit_and_digest(self):
        """A branch URL would silently change every pre-relaxation in a study."""
        self.assertNotIn("/master/", NEP89_MODEL_URL)
        self.assertNotIn("/main/", NEP89_MODEL_URL)
        self.assertRegex(NEP89_MODEL_SHA256, r"^[0-9a-f]{64}$")


class TestFallbackDispatch(unittest.TestCase):
    """Which potential answers is decided per structure, from its element set."""

    def _calculator(self, elements):
        self._tmp = tempfile.TemporaryDirectory()
        return Nep89WithFallback(
            model=_write_model(Path(self._tmp.name)),
            elements=frozenset(elements),
        )

    def tearDown(self):
        if hasattr(self, "_tmp"):
            self._tmp.cleanup()

    def test_an_uncovered_element_falls_back_without_touching_nep(self):
        calculator = self._calculator({"Si"})
        atoms = Atoms("Po", cell=np.diag([3.35, 3.35, 3.35]), pbc=True)
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        self.assertEqual(calculator.last_backend, "fallback")
        self.assertEqual(calculator.unsupported_species(atoms), {"Po"})
        self.assertTrue(np.isfinite(energy))
        # The NEP backend was never constructed, so calorine need not be installed.
        self.assertIsNone(calculator._nep)

    def test_one_uncovered_atom_sends_the_whole_cell_to_the_fallback(self):
        """NEP is a many-body potential; there is no partial evaluation."""
        calculator = self._calculator({"Si"})
        atoms = Atoms("Si3Po", cell=np.diag([5.4, 5.4, 5.4]), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)])
        atoms.calc = calculator
        atoms.get_potential_energy()
        self.assertEqual(calculator.last_backend, "fallback")

    def test_coverage_is_reported_before_anything_is_evaluated(self):
        calculator = self._calculator({"Si", "O"})
        self.assertTrue(calculator.supports(Atoms("SiO2", cell=np.eye(3) * 5, pbc=True)))
        self.assertFalse(calculator.supports(Atoms("SiN", cell=np.eye(3) * 5, pbc=True)))


class TestPrescreenSelection(unittest.TestCase):
    """Deduplicate, then keep the budget's worth of lowest-energy survivors."""

    @staticmethod
    def _candidates(pairs):
        # matcher=False deduplicates on energy alone, so the structures are
        # never touched and can be placeholders.
        return [Candidate(trial=t, energy_per_atom=e, structure=None) for t, e in pairs]

    def test_the_lowest_energy_distinct_trials_are_selected(self):
        selection = select_candidates(
            self._candidates([(0, -1.0), (1, -3.0), (2, -2.0), (3, -0.5)]),
            budget=2, matcher=False, energy_tol=1e-6,
        )
        self.assertEqual(selection.selected, [1, 2])
        self.assertEqual(selection.rejected, [0, 3])
        self.assertEqual(selection.n_distinct, 4)

    def test_duplicates_collapse_onto_their_lowest_energy_member(self):
        """Which is the one the scoring potential should start from."""
        selection = select_candidates(
            self._candidates([(0, -2.0), (1, -2.0000001), (2, -1.0)]),
            budget=2, matcher=False, energy_tol=1e-3,
        )
        self.assertEqual(selection.n_distinct, 2)
        self.assertEqual(selection.duplicate_of, {0: 1})
        self.assertEqual(selection.selected, [1, 2])

    def test_a_wide_energy_gap_is_never_a_duplicate(self):
        """The gate that keeps the matcher off pairs it cannot possibly match."""
        selection = select_candidates(
            self._candidates([(0, -2.0), (1, -1.0)]),
            budget=2, matcher=False, energy_tol=1e-3,
        )
        self.assertEqual(selection.n_distinct, 2)
        self.assertEqual(selection.duplicate_of, {})

    def test_fewer_distinct_structures_than_budget_selects_them_all(self):
        selection = select_candidates(
            self._candidates([(0, -2.0), (1, -2.0)]),
            budget=3, matcher=False, energy_tol=1e-3,
        )
        self.assertEqual(selection.selected, [0])
        self.assertEqual(selection.rejected, [])

    def test_a_matcher_verdict_beats_the_energy_gate(self):
        """Degenerate but distinct structures must both survive."""
        class _NeverMatches:
            @staticmethod
            def fit(a, b):
                return False

        selection = select_candidates(
            self._candidates([(0, -2.0), (1, -2.0)]),
            budget=2, matcher=_NeverMatches(), energy_tol=1e-3,
        )
        self.assertEqual(selection.n_distinct, 2)
        self.assertEqual(selection.selected, [0, 1])

    def test_a_matcher_failure_keeps_both_rather_than_dropping_one(self):
        class _Raises:
            @staticmethod
            def fit(a, b):
                raise RuntimeError("no")

        selection = select_candidates(
            self._candidates([(0, -2.0), (1, -2.0)]),
            budget=2, matcher=_Raises(), energy_tol=1e-3,
        )
        self.assertEqual(selection.n_distinct, 2)

    def test_a_zero_budget_is_a_schedule_error(self):
        with self.assertRaises(ValueError):
            select_candidates(self._candidates([(0, -1.0)]), budget=0, matcher=False)


class TestPreRelaxationStage(unittest.TestCase):
    """The two-stage arm's first half, driven by the Lennard-Jones fallback.

    LJ stands in for NEP89 here on purpose: it is a real potential with real
    derivatives, it runs offline in milliseconds, and every property under test
    -- what gets written down, what the guard does, what a failure costs -- is
    about the stage rather than about the model.
    """

    @staticmethod
    def _draw(a: float = 6.5) -> Atoms:
        """A deliberately loose cell, like a PyXtal draw at tolerance factor 1.3."""
        return bulk("NaCl", "rocksalt", a=a)

    def test_the_stage_records_what_it_did_to_the_cell(self):
        from wyckoff_transformer.cryspr.generator import PRERELAX_VERDICT_FILE, prerelax

        with tempfile.TemporaryDirectory() as tmp:
            wdir = Path(tmp) / "prerelax"
            draw = self._draw()
            relaxed = prerelax(draw, SpeciesLennardJones(), wdir=wdir)
            verdict = json.loads((wdir / PRERELAX_VERDICT_FILE).read_text())
        self.assertAlmostEqual(verdict["volume_before"], draw.get_volume(), places=6)
        self.assertAlmostEqual(verdict["volume_after"], relaxed.get_volume(), places=6)
        self.assertAlmostEqual(
            verdict["volume_ratio"], verdict["volume_after"] / verdict["volume_before"],
            places=9,
        )
        self.assertFalse(verdict["rejected"])
        # The published pre-relaxation schedule: symmetric stages only.
        self.assertTrue(verdict["fix_symmetry"])
        self.assertFalse(verdict["release_symmetry"])
        self.assertFalse(verdict["rattle"])

    def test_the_result_carries_neither_calculator_nor_constraint(self):
        """Or the scoring potential would serve the cheap one's cached energy."""
        from wyckoff_transformer.cryspr.generator import prerelax

        with tempfile.TemporaryDirectory() as tmp:
            relaxed = prerelax(self._draw(), SpeciesLennardJones(), wdir=Path(tmp))
        self.assertIsNone(relaxed.calc)
        self.assertEqual(relaxed.constraints, [])

    def test_the_expansion_guard_returns_the_raw_draw(self):
        """A pre-relaxation that inflates the cell has moved away from the answer.

        Forced here by setting the limit below 1: whatever the stage did, the
        guard must hand back the untouched input rather than its output.
        """
        from wyckoff_transformer.cryspr.generator import PRERELAX_VERDICT_FILE, prerelax

        draw = self._draw()
        with tempfile.TemporaryDirectory() as tmp:
            wdir = Path(tmp) / "prerelax"
            kept = prerelax(draw, SpeciesLennardJones(), wdir=wdir, max_expansion=0.01)
            verdict = json.loads((wdir / PRERELAX_VERDICT_FILE).read_text())
        self.assertTrue(verdict["rejected"])
        self.assertEqual(verdict["max_expansion"], 0.01)
        np.testing.assert_allclose(kept.cell.array, draw.cell.array)

    def test_no_guard_by_default_keeps_whatever_the_potential_produced(self):
        from wyckoff_transformer.cryspr.generator import prerelax

        draw = self._draw()
        with tempfile.TemporaryDirectory() as tmp:
            kept = prerelax(draw, SpeciesLennardJones(), wdir=Path(tmp))
        self.assertFalse(np.allclose(kept.cell.array, draw.cell.array))

    def test_a_failure_costs_the_stage_and_not_the_trial(self):
        """The scoring relaxation can do the job without this stage."""
        from unittest.mock import MagicMock

        from wyckoff_transformer.cryspr.generator import prerelax

        broken = MagicMock()
        broken.get_potential_energy.side_effect = RuntimeError("no parameters")
        draw = self._draw()
        with tempfile.TemporaryDirectory() as tmp:
            kept = prerelax(draw, broken, wdir=Path(tmp))
        self.assertIs(kept, draw)

    def test_a_ranking_caller_gets_the_failure_instead(self):
        """The pre-screen ranks on this energy, so a silent no-op would mislead."""
        from unittest.mock import MagicMock

        from wyckoff_transformer.cryspr.generator import prerelax

        broken = MagicMock()
        broken.get_potential_energy.side_effect = RuntimeError("no parameters")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                prerelax(self._draw(), broken, wdir=Path(tmp), strict=True)

    def test_the_pre_relaxation_writes_into_its_own_directory(self):
        """So its stage CIFs cannot be confused with the scoring relaxation's.

        Both call stepwise_relax, which names CIFs by formula and stage, so a
        shared directory would collide stage for stage and the kept-CIF glob
        would stop being unambiguous.
        """
        from wyckoff_transformer.cryspr.generator import PRERELAX_DIR, relax_trial

        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = Path(tmp) / "trial-0"
            relaxed, energy, prerattle = relax_trial(
                atoms_in=self._draw(),
                calculator=SpeciesLennardJones(),
                trial_dir=trial_dir,
                release_symmetry=False,
                rattle=False,
                prerelax_calculator=SpeciesLennardJones(epsilon=0.5),
            )
            self.assertIsNotNone(relaxed)
            self.assertTrue((trial_dir / PRERELAX_DIR).is_dir())
            self.assertTrue(any((trial_dir / PRERELAX_DIR).glob("*.cif")))
            # Exactly one of each, in the trial directory itself.
            self.assertEqual(len(list(trial_dir.glob("*_kept.cif"))), 1)
            self.assertEqual(len(list(trial_dir.glob("*_prerattle.cif"))), 1)
            # With no rattle stage the two structures are the same one, and the
            # pre-rattle CIF is written anyway: a missing file would otherwise
            # be indistinguishable from a rattle that was never run.
            self.assertAlmostEqual(prerattle[1], energy, places=9)


class TestDigestVerification(unittest.TestCase):
    """A pinned checkpoint that is not verified is a pinned checkpoint in name only."""

    def test_a_cached_file_with_the_wrong_digest_is_refused(self):
        from wyckoff_transformer.cryspr.calculator import _download_and_cache, _sha256

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            url = "https://example.invalid/model.txt"
            # Pre-seed the cache entry the way a completed download would.
            import hashlib

            name = hashlib.sha256(url.encode()).hexdigest()[:16] + ".txt"
            (cache / name).write_text("not the model", encoding="utf-8")
            with self.assertRaises(ValueError):
                _download_and_cache(url, cache, expected_sha256="0" * 64)
            # ... and accepted when it does match, without hitting the network.
            digest = _sha256(cache / name)
            self.assertEqual(
                _download_and_cache(url, cache, expected_sha256=digest), cache / name
            )


@pytest.mark.needs_relax
class TestNep89Model(unittest.TestCase):
    """The real thing: downloads the model and needs ``calorine``."""

    @classmethod
    def setUpClass(cls):
        pytest.importorskip("calorine")

    def test_the_published_model_covers_89_elements(self):
        elements = nep89_elements()
        self.assertEqual(len(elements), 89)
        # The five below Pu that it omits, which is what the fallback is for.
        self.assertTrue({"Po", "At", "Rn", "Fr", "Ra"}.isdisjoint(elements))
        self.assertTrue({"H", "Si", "Fe", "U", "Pu"} <= elements)

    def test_a_covered_structure_gets_a_physical_energy(self):
        calculator = Nep89WithFallback()
        atoms = bulk("Si", "diamond", a=5.43)
        atoms.calc = calculator
        energy = atoms.get_potential_energy() / len(atoms)
        self.assertEqual(calculator.last_backend, "nep89")
        self.assertLess(abs(energy - (-5.4)), 0.5)  # PBE Si is about -5.42 eV/atom

    def test_one_calculator_serves_cells_of_different_size(self):
        """The C++ NEP object is sized to one structure; the wrapper rebuilds it."""
        calculator = Nep89WithFallback()
        first = bulk("Si", "diamond", a=5.43)
        first.calc = calculator
        per_atom = first.get_potential_energy() / len(first)
        second = bulk("Si", "diamond", a=5.43).repeat((2, 1, 1))
        second.calc = calculator
        self.assertAlmostEqual(second.get_potential_energy() / len(second), per_atom, places=6)

    def test_the_fallback_engages_for_an_element_it_omits(self):
        calculator = Nep89WithFallback()
        atoms = Atoms("Ra", cell=np.diag([5.15, 5.15, 5.15]), pbc=True)
        atoms.calc = calculator
        self.assertTrue(np.isfinite(atoms.get_potential_energy()))
        self.assertEqual(calculator.last_backend, "fallback")
        self.assertEqual(calculator.n_fallback, 1)


class TestSymmetricPerturbation(unittest.TestCase):
    """A symmetry-constrained search must move without leaving the space group."""

    @staticmethod
    def _cases():
        from ase import Atoms

        rutile = Atoms(
            "Ti2O4", cell=[4.6, 4.6, 2.95, 90, 90, 90], pbc=True,
            scaled_positions=[
                (0, 0, 0), (0.5, 0.5, 0.5), (0.305, 0.305, 0), (0.695, 0.695, 0),
                (0.805, 0.195, 0.5), (0.195, 0.805, 0.5),
            ],
        )
        return [
            ("NaCl", bulk("NaCl", "rocksalt", a=5.64)),
            ("Si", bulk("Si", "diamond", a=5.43)),
            ("rutile", rutile),
        ]

    def test_the_projected_step_keeps_the_space_group(self):
        from wyckoff_transformer.cryspr.relaxer import (
            _get_spacegroup_info,
            symmetric_perturb,
        )

        for name, atoms in self._cases():
            with self.subTest(structure=name):
                before = _get_spacegroup_info(atoms, 1e-3)[1]
                moved = symmetric_perturb(atoms, seed=0)
                self.assertEqual(_get_spacegroup_info(moved, 1e-3)[1], before)

    def test_the_unprojected_rattle_destroys_it(self):
        """Which is what stage 4 wants, and the opposite of what a search wants."""
        from wyckoff_transformer.cryspr.relaxer import (
            _get_spacegroup_info,
            perturb,
        )

        for name, atoms in self._cases():
            with self.subTest(structure=name):
                before = _get_spacegroup_info(atoms, 1e-3)[1]
                self.assertLess(
                    _get_spacegroup_info(perturb(atoms, seed=0), 1e-3)[1], before
                )

    def test_the_step_actually_moves_something(self):
        """A projection that returned the input would make the walk a fixed point."""
        from wyckoff_transformer.cryspr.relaxer import symmetric_perturb

        _, rutile = self._cases()[2]
        moved = symmetric_perturb(rutile, seed=0)
        self.assertGreater(
            np.abs(moved.get_positions() - rutile.get_positions()).max(), 1e-4
        )

    def test_the_constraint_travels_with_the_result(self):
        """Or the relaxation that follows would leave the subspace immediately."""
        from ase.constraints import FixSymmetry
        from wyckoff_transformer.cryspr.relaxer import symmetric_perturb

        moved = symmetric_perturb(self._cases()[0][1], seed=0)
        self.assertTrue(any(isinstance(c, FixSymmetry) for c in moved.constraints))


class TestBasinHopSymmetryAccounting(unittest.TestCase):
    """Symmetry *gained* is kept; only a genuine loss is a rejection.

    Comparing space-group numbers cannot tell the two apart, and on the oracle
    cohort 122 of 123 changes were increases -- a relaxation converging onto a
    supergroup, which still has every operation the gene's group had and is
    usually lower in energy. Rejecting on "the number changed" discarded exactly
    the good case.
    """

    def test_operation_counts_order_the_two_directions(self):
        from wyckoff_transformer.cryspr.basin_hopping import _n_symmetry_operations

        symmetric = bulk("Si", "diamond", a=5.43)
        broken = symmetric.copy()
        broken.rattle(0.1, seed=1)
        self.assertGreater(
            _n_symmetry_operations(symmetric, 1e-3),
            _n_symmetry_operations(broken, 1e-3),
        )

    def test_a_structure_spglib_cannot_read_counts_as_unknown(self):
        """0 rather than a crash, so one odd cell cannot end a walk."""
        from ase import Atoms
        from wyckoff_transformer.cryspr.basin_hopping import _n_symmetry_operations

        self.assertEqual(
            _n_symmetry_operations(Atoms("H", cell=[0, 0, 0], pbc=False), 1e-3), 0
        )


class TestScreenedMorseDerivatives(unittest.TestCase):
    """The shipped fallback relaxes cells, so its derivatives must be the real ones.

    Checked in the crowded regime as well as the ordinary one: the fallback's
    whole reason to exist is the draws that arrive overlapped, and a sign error
    that only shows up inside the ZBL core would be invisible in a well-behaved
    cell.
    """

    STEP = 1e-6

    def _check(self, atoms):
        atoms.calc = ScreenedMorse()
        analytic_f = atoms.get_forces()
        analytic_s = atoms.get_stress(voigt=False)
        positions = atoms.get_positions().copy()
        numeric_f = np.zeros_like(analytic_f)
        for i in range(len(atoms)):
            for axis in range(3):
                for sign in (1, -1):
                    shifted = positions.copy()
                    shifted[i, axis] += sign * self.STEP
                    atoms.set_positions(shifted)
                    numeric_f[i, axis] -= sign * atoms.get_potential_energy() / (2 * self.STEP)
        atoms.set_positions(positions)

        cell = atoms.cell.array.copy()
        scaled = atoms.get_scaled_positions().copy()
        volume = atoms.get_volume()
        numeric_s = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                for sign in (1, -1):
                    strain = np.zeros((3, 3))
                    strain[a, b] += sign * self.STEP / 2
                    strain[b, a] += sign * self.STEP / 2
                    atoms.set_cell(cell @ (np.eye(3) + strain), scale_atoms=True)
                    numeric_s[a, b] += (
                        sign * atoms.get_potential_energy() / (2 * self.STEP) / volume
                    )
        atoms.set_cell(cell, scale_atoms=True)
        atoms.set_scaled_positions(scaled)
        np.testing.assert_allclose(analytic_f, numeric_f, atol=1e-5)
        np.testing.assert_allclose(analytic_s, numeric_s, atol=1e-6)

    def test_an_ordinary_multi_element_cell(self):
        self._check(_rattled_cell())

    def test_a_cell_crowded_into_the_zbl_core(self):
        atoms = Atoms("Si4", cell=np.diag([3.0, 3.1, 3.2]), pbc=True,
                      scaled_positions=[(0, 0, 0), (0.18, 0.02, 0.01),
                                        (0.5, 0.5, 0.02), (0.02, 0.5, 0.5)])
        atoms.calc = ScreenedMorse()
        self.assertGreater(np.abs(atoms.get_forces()).max(), 100.0)  # deep in the core
        self._check(atoms)

    def test_a_pair_of_elements_nep89_omits(self):
        """Ra and Po are the fallback's actual job, not a hypothetical."""
        self._check(Atoms("RaPo", cell=np.diag([4.0, 4.1, 4.2]), pbc=True,
                          scaled_positions=[(0, 0, 0), (0.30, 0.02, 0.01)]))


class TestScreenedMorseShape(unittest.TestCase):
    """The two halves must each do their job, and hand over smoothly."""

    @staticmethod
    def _energy(sym_a, sym_b, r, calc=None):
        atoms = Atoms(sym_a + sym_b, cell=np.diag([16.0] * 3), pbc=True,
                      positions=[(0, 0, 0), (r, 0, 0)])
        atoms.calc = calc or ScreenedMorse()
        return float(atoms.get_potential_energy())

    def test_the_short_range_core_is_textbook_zbl(self):
        """Not an analogy: the same function NEP89 uses, so the handover is seamless."""
        from ase.data import atomic_numbers

        for a, b in (("Si", "Si"), ("Fe", "Fe"), ("U", "O")):
            z_i, z_j = atomic_numbers[a], atomic_numbers[b]
            screening = ZBL_SCREENING_CONSTANT / (
                z_i ** ZBL_SCREENING_POWER + z_j ** ZBL_SCREENING_POWER
            )
            for r in (0.1, 0.2):
                x = r / screening
                phi = sum(c * np.exp(-e * x)
                          for c, e in zip(ZBL_COEFFICIENTS, ZBL_EXPONENTS))
                expected = COULOMB_EV_ANGSTROM * z_i * z_j * phi / r
                with self.subTest(pair=f"{a}-{b}", r=r):
                    self.assertAlmostEqual(
                        self._energy(a, b, r) / expected, 1.0, places=3
                    )

    def test_the_minimum_sits_at_the_contact_distance(self):
        """Which is what lets a variable-cell relaxation contract a loose draw."""
        from ase.data import atomic_numbers, covalent_radii

        for a, b in (("Si", "Si"), ("H", "H"), ("Ra", "Ra"), ("Na", "Cl")):
            re = covalent_radii[atomic_numbers[a]] + covalent_radii[atomic_numbers[b]]
            grid = np.linspace(0.6 * re, 1.6 * re, 300)
            best = grid[int(np.argmin([self._energy(a, b, r) for r in grid]))]
            with self.subTest(pair=f"{a}-{b}"):
                self.assertAlmostEqual(best, re, delta=0.05 * re)

    def test_it_is_repulsive_all_the_way_in(self):
        for a, b in (("Si", "Si"), ("Ra", "Ra")):
            energies = [self._energy(a, b, r) for r in (0.1, 0.2, 0.4, 0.7)]
            with self.subTest(pair=f"{a}-{b}"):
                self.assertTrue(all(np.diff(energies) < 0))

    def test_it_is_orders_of_magnitude_tamer_than_the_lennard_jones_it_replaced(self):
        """The reason for the change: r^-12 reaches 1e16 eV where ZBL reaches 1e4."""
        zbl_like = self._energy("Si", "Si", 0.1)
        lj = self._energy("Si", "Si", 0.1, calc=SpeciesLennardJones())
        self.assertLess(zbl_like, 1e5)
        self.assertGreater(lj / zbl_like, 1e9)

    def test_the_handover_leaves_no_discontinuity(self):
        """A step in E or F would let an optimiser chase the seam."""
        from ase.data import atomic_numbers, covalent_radii

        re = 2 * covalent_radii[atomic_numbers["Si"]]
        grid = np.linspace(0.2 * re, 1.4 * re, 600)
        e = np.array([self._energy("Si", "Si", r) for r in grid])
        # A genuine discontinuity shows up as a second difference far larger
        # than its neighbours; a smooth blend keeps it comparable.
        d2 = np.abs(np.diff(e, 2))
        self.assertLess(d2.max() / np.median(d2[d2 > 0]), 5e3)

    def test_the_energy_vanishes_beyond_the_cutoff(self):
        # 8 A in a 16 A cell, so the periodic image is 8 A away too: at 12 A the
        # image would sit at 4 A, inside the 4.44 A cutoff, and the pair would
        # still interact.
        self.assertAlmostEqual(self._energy("Si", "Si", 8.0), 0.0, places=9)


class TestFallbackIsWiredIn(unittest.TestCase):
    def test_the_hybrid_uses_the_screened_fallback_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            calculator = Nep89WithFallback(
                model=_write_model(Path(tmp)), elements=frozenset({"Si"})
            )
            self.assertIsInstance(calculator.fallback, ScreenedMorse)
            atoms = Atoms("Ra", cell=np.diag([5.15] * 3), pbc=True)
            atoms.calc = calculator
            self.assertTrue(np.isfinite(atoms.get_potential_energy()))
            self.assertEqual(calculator.last_backend, "fallback")
            self.assertIn("ZBL", calculator.provenance()["prerelax_fallback_note"])
