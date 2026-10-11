"""Invalid optimiser output must not enter spglib's native symmetry search."""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from wyckoff_transformer.cryspr import relaxer


@pytest.mark.parametrize("field,value", [("positions", np.nan), ("positions", np.inf),
                                        ("cell", np.nan), ("cell", np.inf)])
def test_nonfinite_geometry_is_rejected_before_spglib(monkeypatch, field, value):
    atoms = Atoms("Cu", positions=[[0., 0., 0.]], cell=np.eye(3), pbc=True)
    getattr(atoms, field)[0][0] = value

    def unexpected_call(*args, **kwargs):
        pytest.fail("Non-finite geometry reached native spglib")

    monkeypatch.setattr(relaxer.spglib, "get_symmetry_dataset", unexpected_call)
    with pytest.raises(ValueError, match="Non-finite"):
        relaxer._get_spacegroup_info(atoms, symprec=.01)


def test_finite_geometry_keeps_the_same_symmetry_inputs(monkeypatch):
    atoms = Atoms("Cu", positions=[[.3, .6, .9]], cell=np.eye(3) * 3, pbc=True)

    def symmetry_dataset(cell, *, symprec):
        np.testing.assert_array_equal(cell[0], atoms.cell.array)
        np.testing.assert_array_equal(cell[1], atoms.get_scaled_positions())
        np.testing.assert_array_equal(cell[2], atoms.numbers)
        assert symprec == .01
        return SimpleNamespace(international="Fm-3m", number=225)

    monkeypatch.setattr(relaxer.spglib, "get_symmetry_dataset", symmetry_dataset)
    assert relaxer._get_spacegroup_info(atoms, symprec=.01) == ("Fm-3m", 225)
