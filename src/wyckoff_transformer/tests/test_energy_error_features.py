"""Controls for size, periodic contacts and calculation metadata descriptors."""

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from wyckoff_transformer.evaluation.energy_error_features import (
    calculation_features, composition_keys, structure_features,
)


def test_reduced_formula_and_chemical_system_do_not_depend_on_cell_setting():
    assert composition_keys(["Na", "Cl"], [1, 1]) == composition_keys(["Cl", "Na"], [4, 4])
    assert composition_keys(["Na", "Cl"], [1, 1]) != composition_keys(["Na", "Cl"], [2, 1])


def test_intensive_geometry_and_gene_dof_survive_supercell():
    structure = Structure(Lattice.cubic(4), ["Na", "Cl"], [[0, 0, 0], [.5, .5, .5]])
    gene = dict(group=221, species=["Na", "Cl"], sites=[["1a"], ["1b"]])
    original = structure_features(structure, gene)
    structure.make_supercell([2, 1, 1])
    expanded = structure_features(structure, gene)
    assert expanded["n_atoms"] == 2 * original["n_atoms"]
    for key in ["volume_per_atom", "density", "covalent_sphere_fraction", "min_pair_distance", "min_covalent_distance_ratio"]:
        assert expanded[key] == pytest.approx(original[key])
    assert expanded["positional_dof"] == 0
    assert expanded["lattice_dof"] == 1


def test_one_atom_cell_includes_periodic_neighbors():
    structure = Structure(Lattice.cubic(3), ["Cu"], [[0, 0, 0]])
    gene = dict(group=221, species=["Cu"], sites=[["1a"]])
    features = structure_features(structure, gene)
    assert features["min_pair_distance"] == pytest.approx(3.)


def test_missing_settings_are_not_labelled_gga():
    missing = calculation_features(None, {"Fe": 1, "O": 1})
    assert np.isnan(missing["dft_uses_hubbard"])
    assert missing["dft_run_type"] == "unrecorded"
    recorded = calculation_features({"parameters": {"is_hubbard": False, "run_type": "GGA", "hubbards": {}}}, {"Fe": 1, "O": 1})
    assert recorded["dft_uses_hubbard"] == 0
    assert recorded["dft_hubbard_per_atom"] == 0


def test_gene_and_geometry_composition_mismatch_is_explicit():
    structure = Structure(Lattice.cubic(3), ["Cu"], [[0, 0, 0]])
    with pytest.raises(ValueError, match="different compositions"):
        structure_features(structure, dict(group=221, species=["Na"], sites=[["1a"]]))
