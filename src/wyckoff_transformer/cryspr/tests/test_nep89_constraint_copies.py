"""Calculator snapshots omit constraints without changing constrained relaxation."""

from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms, FixSymmetry
from ase.optimize import BFGS

from wyckoff_transformer.cryspr.fast_bfgs import PositiveDefiniteBFGS
from wyckoff_transformer.cryspr.nep89 import Nep89WithFallback
from wyckoff_transformer.cryspr.relaxer import (
    _get_spacegroup_info,
    stepwise_relax_stages,
)


class RecordingCalculator(Calculator):
    implemented_properties = ("energy", "forces", "stress")

    def __init__(self):
        super().__init__()
        self.snapshots = []

    def set_atoms(self, atoms):
        self.set_atoms_constraints = list(atoms.constraints)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.snapshots.append(self.atoms)
        positions = self.atoms.positions
        self.results = {
            "energy": float(np.sum(positions**2)),
            "forces": -2 * positions,
            "stress": np.zeros(6),
        }


@pytest.mark.parametrize("covered", [False, True])
def test_inner_calculators_see_all_arrays_but_no_symmetry_constraint(tmp_path, covered):
    model = tmp_path / "nep.txt"
    model.write_text("nep4_zbl 1 Si\n", encoding="utf-8")
    inner = RecordingCalculator()
    calculator = Nep89WithFallback(
        model=model, elements=frozenset({"Si"}), fallback=inner,
    )
    if covered:
        calculator._nep = inner
    atoms = bulk("Si" if covered else "NaCl", "diamond" if covered else "rocksalt", a=5.8)
    atoms.new_array("custom_field", np.arange(len(atoms), dtype=float))
    atoms.set_masses(np.arange(len(atoms), dtype=float) + 20)
    atoms.info["source"] = "test"
    atoms.set_constraint(FixSymmetry(atoms))
    atoms.calc = calculator

    # A deepcopy anywhere in the calculator stack would fail this test.
    with patch.object(FixSymmetry, "__deepcopy__", side_effect=AssertionError("copied"), create=True):
        first = atoms.get_potential_energy()
        atoms.get_forces()
        assert len(inner.snapshots) == 1  # ASE result caching still works.
        atoms.positions[0, 0] += 0.01
        second = atoms.get_potential_energy()

    assert first != second
    assert len(inner.snapshots) == 2
    assert len(atoms.constraints) == 1
    assert isinstance(atoms.constraints[0], FixSymmetry)
    assert calculator.atoms.constraints == []
    assert all(snapshot.constraints == [] for snapshot in inner.snapshots)
    if covered:
        assert inner.set_atoms_constraints == []
    for snapshot in inner.snapshots:
        np.testing.assert_array_equal(snapshot.arrays["custom_field"], atoms.arrays["custom_field"])
        np.testing.assert_array_equal(snapshot.get_masses(), atoms.get_masses())
        assert snapshot.info == atoms.info
        assert snapshot.pbc.all()


def test_constraint_still_projects_forces_and_stress(tmp_path):
    model = tmp_path / "nep.txt"
    model.write_text("nep4_zbl 1 Si\n", encoding="utf-8")

    class AsymmetricCalculator(RecordingCalculator):
        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results["forces"] = np.arange(1, 3 * len(self.atoms) + 1).reshape(-1, 3).astype(float)
            self.results["stress"] = np.array([1., 2., 3., 4., 5., 6.])

    atoms = bulk("NaCl", "rocksalt", a=5.8)
    atoms.set_constraint(FixSymmetry(atoms))
    atoms.calc = Nep89WithFallback(model=model, elements=frozenset({"Si"}), fallback=AsymmetricCalculator())
    raw_forces = atoms.calc.get_forces(atoms).copy()
    raw_stress = atoms.calc.get_stress(atoms).copy()
    projected_forces = raw_forces.copy()
    projected_stress = raw_stress.copy()
    atoms.constraints[0].adjust_forces(atoms, projected_forces)
    atoms.constraints[0].adjust_stress(atoms, projected_stress)

    np.testing.assert_allclose(atoms.get_forces(), projected_forces, rtol=0, atol=1e-12)
    np.testing.assert_allclose(atoms.get_stress(), projected_stress, rtol=0, atol=1e-12)
    assert not np.allclose(raw_forces, projected_forces)
    assert not np.allclose(raw_stress, projected_stress)


def test_other_constraints_reach_fallback(tmp_path):
    model = tmp_path / "nep.txt"
    model.write_text("nep4_zbl 1 Si\n", encoding="utf-8")
    fallback = RecordingCalculator()
    atoms = bulk("NaCl", "rocksalt", a=5.8)
    atoms.set_constraint([FixAtoms(indices=[0]), FixSymmetry(atoms)])
    atoms.calc = Nep89WithFallback(model=model, elements=frozenset({"Si"}), fallback=fallback)
    atoms.get_potential_energy()

    assert len(atoms.constraints) == 2
    assert len(atoms.calc.atoms.constraints) == 1
    assert isinstance(atoms.calc.atoms.constraints[0], FixAtoms)
    assert atoms.calc.atoms.constraints[0] is not atoms.constraints[0]
    assert len(fallback.snapshots[0].constraints) == 1
    assert isinstance(fallback.snapshots[0].constraints[0], FixAtoms)


@pytest.mark.needs_relax
def test_real_nep_snapshot_is_constraint_free_and_keeps_force_projection():
    pytest.importorskip("calorine")
    atoms = _rutile()
    atoms.set_constraint(FixSymmetry(atoms))
    atoms.calc = Nep89WithFallback()
    raw = atoms.calc.get_forces(atoms).copy()
    constrained = atoms.get_forces()
    expected = raw.copy()
    atoms.constraints[0].adjust_forces(atoms, expected)
    np.testing.assert_allclose(constrained, expected, rtol=0, atol=1e-12)
    assert len(atoms.constraints) == 1
    assert atoms.calc.atoms.constraints == []
    assert atoms.calc._nep.atoms.constraints == []
    assert atoms.calc._nep._nepy_atoms.constraints == []


def _rutile():
    return Atoms(
        "Ti2O4", cell=[4.75, 4.75, 3.05, 90, 90, 90], pbc=True,
        scaled_positions=[
            (0, 0, 0), (0.5, 0.5, 0.5),
            (0.315, 0.315, 0), (0.685, 0.685, 0),
            (0.815, 0.185, 0.5), (0.185, 0.815, 0.5),
        ],
    )


@pytest.mark.needs_relax
@pytest.mark.parametrize("name,initial", [
    ("silicon", lambda: bulk("Si", "diamond", a=5.8)),
    ("silicon-32", lambda: bulk("Si", "diamond", a=5.8, cubic=True).repeat((2, 2, 1))),
    ("rutile", _rutile),
])
def test_combined_changes_match_original_relaxation_and_preserve_symmetry(
    tmp_path, name, initial,
):
    pytest.importorskip("calorine")
    atoms = initial()
    before = _get_spacegroup_info(atoms, symprec=1e-3)[1]
    assert before > 1

    # Identity reproduces the old calculator boundary; ASE BFGS reproduces
    # the old optimizer.  Both runs use the same real, pinned NEP89 model.
    with patch.object(Nep89WithFallback, "_calculator_geometry", staticmethod(lambda a: a)):
        original = stepwise_relax_stages(
            atoms, Nep89WithFallback(), optimizer=BFGS,
            fix_symmetry=True, release_symmetry=False, rattle=False,
            fmax=0.05, steps_limit=100, wdir=tmp_path / f"{name}-original",
        )
    combined = stepwise_relax_stages(
        atoms, Nep89WithFallback(), optimizer=PositiveDefiniteBFGS,
        fix_symmetry=True, release_symmetry=False, rattle=False,
        fmax=0.05, steps_limit=100, wdir=tmp_path / f"{name}-combined",
    )
    for result in (original, combined):
        assert isinstance(result.fixed_symmetry.constraints[0], FixSymmetry)
        assert _get_spacegroup_info(result.fixed_symmetry, symprec=1e-3)[1] == before
        assert np.max(np.abs(result.kept.get_forces())) < 0.05

    np.testing.assert_allclose(
        combined.fixed_symmetry_energy / len(atoms),
        original.fixed_symmetry_energy / len(atoms),
        rtol=0, atol=1e-5,
    )
    np.testing.assert_allclose(
        combined.fixed_symmetry.get_positions(),
        original.fixed_symmetry.get_positions(), rtol=0, atol=1e-3,
    )
    np.testing.assert_allclose(
        combined.fixed_symmetry.cell.array,
        original.fixed_symmetry.cell.array, rtol=0, atol=1e-3,
    )
