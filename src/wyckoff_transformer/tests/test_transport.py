"""Tests of the Li-transport descriptors and labels (wyckoff_transformer.transport)."""
import math

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from wyckoff_transformer.transport import scores
from wyckoff_transformer.transport.frameworks import (
    ANION_LABEL,
    CATION_LABEL,
    anion_elements,
    default_matcher,
    framework,
    group_frameworks,
)
from wyckoff_transformer.transport.pes import (
    Grid,
    allowed_points,
    map_relative_mask,
    minimal_percolation_energy,
    mobile_orbits,
    percolation_barriers,
    reachable,
    tile_centred,
    xi_descriptor,
)
from wyckoff_transformer.transport.sources import (
    gaussian_label_weights,
    nernst_einstein,
    parse_conductivity,
)


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def test_parse_conductivity():
    assert parse_conductivity("<1E-10") == (1e-10, "<")
    assert parse_conductivity(">1e-3") == (1e-3, ">")
    assert parse_conductivity(1.58e-6) == (1.58e-6, "")
    assert parse_conductivity("2.5e-4") == (2.5e-4, "")
    value, censoring = parse_conductivity(float("nan"))
    assert math.isnan(value) and censoring == ""


def test_nernst_einstein():
    # 10 Li in 1000 A^3 at 1000 K with D = 1e-5 cm^2/s: n = 1e28 m^-3,
    # sigma = n e^2 D / kT = 1e28 * (1.602e-19)^2 * 1e-9 / (1.381e-20) = 18.59 S/m.
    assert nernst_einstein(1e-5, 10, 1000.0, 1000.0) == pytest.approx(0.1859, rel=1e-3)


def test_gaussian_label_weights_hard_and_soft():
    w_pos, w_neg = gaussian_label_weights(np.array([0.5, 0.001, 0.05]), np.zeros(3), 0.1, 0.01)
    assert w_pos.tolist() == [1.0, 0.0, 0.0]
    assert w_neg.tolist() == [0.0, 1.0, 0.0]
    w_pos, w_neg = gaussian_label_weights(np.array([0.1]), np.array([0.05]), 0.1, 0.01)
    assert w_pos[0] == pytest.approx(0.5)
    assert w_neg[0] < 0.05


# --------------------------------------------------------------------------- #
# frameworks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("elements, anions", [
    (["Li", "P", "S"], {"S"}),
    (["Li", "S", "O"], {"O"}),
    (["Li", "B", "H"], {"H"}),
    (["Li", "N", "H"], {"N"}),
    (["Li", "P"], {"P"}),
    (["Li", "Al", "P"], {"P"}),
    (["Li", "Si", "P"], {"P"}),
    (["Li", "P", "S", "Cl"], {"S", "Cl"}),
    (["Li", "Ga", "Se", "O"], {"O"}),
    (["Li", "Y", "Cl"], {"Cl"}),
])
def test_anion_elements(elements, anions):
    assert anion_elements(elements) == anions


def _rocksalt(cation: str, anion: str, a: float) -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), [cation, anion],
                                     [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_framework_drops_li_and_anonymises():
    frame = framework(_rocksalt("Li", "Cl", 5.1))
    assert {site.specie.symbol for site in frame} == {ANION_LABEL}
    assert len(frame) == 4


def test_frameworks_of_different_chemistry_match():
    licl = framework(_rocksalt("Li", "Cl", 5.1))
    libr = framework(_rocksalt("Li", "Br", 5.5))
    lif = framework(_rocksalt("Li", "F", 4.0))
    assert default_matcher().fit(licl, libr)
    groups = group_frameworks({"LiCl": licl, "LiBr": libr, "LiF": lif})
    assert len(set(groups.values())) == 1


def test_disordered_sites():
    lattice = Lattice.cubic(6.0)
    structure = Structure(lattice, [{"Li": 0.8, "Ga": 0.2}, {"Ge": 0.5, "P": 0.5}, "S", "S"],
                          [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
    frame = framework(structure, anions=frozenset({"S"}))
    species = sorted(site.specie.symbol for site in frame)
    assert species == sorted([CATION_LABEL, ANION_LABEL, ANION_LABEL])


# --------------------------------------------------------------------------- #
# pes
# --------------------------------------------------------------------------- #
def test_grid_step_never_exceeds_target():
    grid = Grid.for_lattice(np.diag([4.0, 4.1, 10.0]), 0.25)
    assert grid.shape == (16, 17, 40)
    assert grid.frac_coords().shape == (16 * 17 * 40, 3)


def test_allowed_points_excludes_the_sphere():
    grid = Grid((10, 10, 10))
    allowed = allowed_points(np.eye(3) * 5.0, np.array([[0.0, 0.0, 0.0]]), grid, 1.2)
    frac = grid.frac_coords()
    cart = np.where(frac > 0.5, frac - 1, frac) * 5.0
    inside = np.linalg.norm(cart, axis=1) < 1.2
    assert not allowed[inside].any() and allowed[~inside].all()


def test_tile_centred_puts_start_in_the_middle():
    pes = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    tiled = tile_centred(pes, (1, 2, 3))
    assert tiled.shape == (9, 11, 13)
    assert tiled[4, 5, 6] == pes[1, 2, 3]
    assert tiled[0, 5, 6] == pes[1, 2, 3]     # a periodic replica on the face


def _channel_pes(n: int = 12, height: float = 1.0, barrier: float = 0.3) -> np.ndarray:
    """High everywhere except a channel along axis 0 with one bump of *barrier*."""
    pes = np.full((n, n, n), height)
    pes[:, n // 2, n // 2] = 0.0
    pes[n // 4, n // 2, n // 2] = barrier
    return pes


def test_reachable_and_mpe_on_a_channel():
    pes = _channel_pes()
    start = (0, 6, 6)
    tiled = tile_centred(pes, start)
    assert reachable(tiled, 0.2).sum() < reachable(tiled, 0.5).sum()
    assert minimal_percolation_energy(tiled) == pytest.approx(0.3, abs=0.01)


def test_percolation_barriers_on_a_channel():
    barriers = percolation_barriers(_channel_pes())
    assert barriers["E_1D"] == pytest.approx(0.3, abs=0.02)
    assert barriers["E_3D"] == pytest.approx(1.0, abs=0.02)


def test_map_relative_mask_follows_the_rotation():
    shape = (4, 4, 4)
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[4 + 1, 4, 4] = True                      # offset (1, 0, 0) from the centre
    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    mapped = map_relative_mask(mask, swap, shape)
    assert mapped[4, 4 + 1, 4] and mapped.sum() == 1


def test_mobile_orbits_maps_equivalent_li():
    structure = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.0), ["Li", "Cl"],
                                          [[0.5, 0, 0], [0, 0, 0]])
    li = [i for i, site in enumerate(structure) if site.specie.symbol == "Li"]
    orbits = mobile_orbits(structure, li)
    reps = {rep for rep, _ in orbits.values()}
    assert len(li) == 3 and len(reps) == 1
    for site, (rep, rotation) in orbits.items():
        image = rotation @ structure[rep].frac_coords
        delta = (image - structure[site].frac_coords) % 1.0
        assert np.allclose(np.minimum(delta, 1 - delta), 0, atol=1e-6) or site == rep


def test_xi_descriptor():
    assert xi_descriptor(0.0, 0.1) == 0.0
    assert xi_descriptor(10 ** -2.0, 10 ** -1.15) == pytest.approx(0.25)
    assert xi_descriptor(0.5, 0.5) > 0.99


# --------------------------------------------------------------------------- #
# scores
# --------------------------------------------------------------------------- #
def test_partial_auc_extremes():
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    perfect = np.arange(10, 0, -1, dtype=float)
    assert scores.partial_auc(perfect, labels, 1 - labels) == pytest.approx(1.0)
    assert scores.partial_auc(-perfect, labels, 1 - labels) == pytest.approx(0.0)
    ties = np.zeros(10)
    assert scores.partial_auc(ties, labels, 1 - labels) == pytest.approx(
        scores.random_partial_auc())


def test_bootstrap_interval_contains_point():
    rng = np.random.default_rng(1)
    labels = (rng.random(300) < 0.3).astype(float)
    score = labels + rng.normal(0, 0.7, 300)
    point, low, high = scores.bootstrap_partial_auc(score, labels, 1 - labels, n=200)
    assert low <= point <= high
    assert low > scores.random_partial_auc()


def test_rediscovery():
    percentiles = scores.percentile_ranks(np.array([10, 9, 8, 7, 1, 0, 6, 5, 4, 3, 2.0]))
    family = np.array(["a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b"], dtype=object)
    fast = np.ones(11, dtype=bool)
    result = scores.rediscovery(percentiles, family, fast, min_members=5, top_fraction=0.5)
    assert result["eligible"] == 2
    assert result["families"]["a"]["median_percentile"] < result["families"]["b"]["median_percentile"]


# --------------------------------------------------------------------------- #
# md
# --------------------------------------------------------------------------- #
def test_md_supercell_widths_and_size():
    from ase.build import bulk

    from wyckoff_transformer.transport.md import md_supercell

    atoms = bulk("Cu", "fcc", a=3.6)          # 1 atom, widths ~2.1 A
    supercell, _repeats = md_supercell(atoms, min_atoms=100, min_width=10.0)
    cell = np.asarray(supercell.cell)
    volume = abs(np.linalg.det(cell))
    widths = [volume / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
              for i in range(3)]
    assert min(widths) >= 10.0 - 1e-6
    assert len(supercell) >= 100


def test_windowed_diffusion_recovers_a_random_walk():
    from wyckoff_transformer.transport.md import windowed_diffusion

    rng = np.random.default_rng(0)
    d_true, dt = 0.05, 0.02                   # A^2/ps, ps
    steps = rng.normal(0, np.sqrt(2 * d_true * dt), size=(5001, 200, 3))
    positions = np.cumsum(steps, axis=0)
    d, _sem, windows = windowed_diffusion(positions, dt)
    assert windows == 6
    assert d == pytest.approx(d_true, rel=0.15)


def test_extrapolation_is_arrhenius():
    from wyckoff_transformer.transport.md import extrapolate

    assert extrapolate(1.0, 500.0, 500.0, 0.5) == pytest.approx(1.0)
    assert extrapolate(1.0, 500.0, 300.0, 0.5) < 1e-2


def test_centre_of_mass_drift_is_removed():
    from wyckoff_transformer.transport.md import (
        remove_centre_of_mass_drift,
        windowed_diffusion,
    )

    rng = np.random.default_rng(1)
    drift = (np.arange(3001) * 0.02 * 0.2)[:, None, None] * np.ones((1, 1, 3))  # 0.2 A/ps
    rigid = rng.normal(0, 0.05, size=(3001, 50, 3)) + drift
    cleaned = remove_centre_of_mass_drift(rigid, ["S"] * 50)
    assert windowed_diffusion(rigid, 0.02)[0] > 1e-2
    assert abs(windowed_diffusion(cleaned, 0.02)[0]) < 1e-3


# --------------------------------------------------------------------------- #
# screens
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("formula, expected", [
    ("Li3YCl6", "Cl-1 Li+1 Y+3"),
    ("Li2SO4", "Li+1 O-2 S+6"),           # S may be positive next to O
    ("LiGa(SeO3)2", "Ga+3 Li+1 O-2 Se+4"),
    ("LiCuS", "Cu+1 Li+1 S-2"),           # an ICSD-frequent, not "common", state
    ("LiBH4", "B+3 H-1 Li+1"),
    ("LiNH2", "H+1 Li+1 N-3"),
    ("LiNCl4", None),                     # N cannot be positive under a halide
    ("LiSI3", None),
    ("LiIO3", None),                      # halogens are always negative
    ("LiI3", None),
])
def test_generic_oxidation_states(formula, expected):
    from wyckoff_transformer.transport.screens import generic_oxidation_states

    assert generic_oxidation_states(formula) == expected


def test_bond_valence_roles():
    from wyckoff_transformer.transport.screens import bond_valence_roles

    states, reason = bond_valence_roles(_rocksalt("Li", "Cl", 5.1))
    assert (states, reason) == ("Cl-1 Li+1", "ok")


def test_environment_keys_keep_clean_sites_only():
    from wyckoff_transformer.transport.screens import environment_keys

    rows = [{"element": "Li", "ox": 1, "cn": 4, "geometry": "tetrahedron", "gap": 1.3,
             "angle_rms": 2.0},
            {"element": "Y", "ox": 3, "cn": 6, "geometry": "octahedron", "gap": 1.05,
             "angle_rms": 2.0},
            {"element": "Li", "ox": 1, "cn": 6, "geometry": "irregular", "gap": 1.5,
             "angle_rms": 25.0}]
    assert environment_keys(rows) == [("Li+1", 4, "tetrahedron")]
