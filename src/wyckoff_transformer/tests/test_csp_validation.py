"""The pure parts of csp_validation; the sampling itself is exercised through a trained backbone."""
import pytest

from wyckoff_transformer.csp_validation import CSPValidation, CSPValidationConfig, hits_at_n, lattice_points


def test_hits_at_n_counts_a_formula_once_per_budget():
    hits = [[False, True, False], [False, False, False], [True, True, True]]
    assert hits_at_n(hits, (1, 2, 3)) == {1: 1, 2: 2, 3: 2}


@pytest.mark.parametrize("spacegroup, points", [
    (1, 1), (221, 1),      # primitive
    (12, 2), (38, 2),      # C2/m, Amm2: one-face centred
    (229, 2),              # Im-3m: body centred
    (225, 4),              # Fm-3m: face centred
    (166, 3), (146, 3),    # R-3m, R3 in PyXtal's hexagonal setting
])
def test_lattice_points(spacegroup, points):
    assert lattice_points(spacegroup) == points


def test_primitive_cell_count_scales_with_the_candidate_groups_centring():
    validation = CSPValidation.__new__(CSPValidation)  # _cell_atoms needs only these
    validation.lattice_points = {221: 1, 225: 4, 166: 3}
    task = {"n_atoms": 8, "n_primitive": 2}  # rocksalt: 8 atoms in Fm-3m, 2 in its primitive cell
    validation.config = CSPValidationConfig(period=1)
    assert [validation._cell_atoms(task, sg) for sg in (221, 225, 166)] == [2, 8, 6]
    validation.config = CSPValidationConfig(period=1, cell_count="conventional")
    assert validation._cell_atoms(task, 221) == 8
    validation.config = CSPValidationConfig(period=1, given_cell=False)
    assert validation._cell_atoms(task, 225) is None
