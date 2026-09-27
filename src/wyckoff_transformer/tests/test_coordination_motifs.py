"""The coordination-motif screen on structures whose answers are textbook."""
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from wyckoff_transformer.evaluation.coordination_motifs import (
    analyse,
    classify_polyhedron,
    network_dimensionality,
)


def _p1_cif(sg, lattice, species, coords):
    s = Structure.from_spacegroup(sg, lattice, species, coords)
    return str(CifWriter(Structure.from_sites(s.sites)))


def _site(sites, element):
    return next(r for r in sites if r["element"] == element)


def test_cubic_perovskite():
    row, sites, pairs = analyse("t-1", _p1_cif("Pm-3m", Lattice.cubic(3.905), ["Sr", "Ti", "O"],
                                               [[0, 0, 0], [.5, .5, .5], [.5, .5, 0]]))
    assert row["status"] == "ok"
    assert row["tolerance_t"] == pytest.approx(1.0, abs=0.02)
    assert row["perovskite_corner_3d"]
    ti = _site(sites, "Ti")
    assert (ti["ox"], ti["cn"], ti["geometry"], ti["point_group"]) == (4, 6, "octahedron", "m-3m")
    assert not ti["polar_site"] and ti["offcentre"] < 1e-6
    modes = {p["mode"] for p in pairs if p["el1"] == p["el2"] == "Ti"}
    assert modes == {"corner"}


def test_rutile_edge_sharing():
    _, sites, pairs = analyse("t-2", _p1_cif("P4_2/mnm", Lattice.tetragonal(4.594, 2.959),
                                             ["Ti", "O"], [[0, 0, 0], [.3048, .3048, 0]]))
    assert _site(sites, "Ti")["point_group"] == "mmm"
    assert {p["mode"] for p in pairs} == {"corner", "edge"}


def test_corundum_face_sharing_and_off_centring():
    row, sites, pairs = analyse("t-3", _p1_cif("R-3c", Lattice.hexagonal(4.759, 12.99),
                                               ["Al", "O"], [[0, 0, .3521], [.3064, 0, .25]]))
    al = _site(sites, "Al")
    assert al["polar_site"] and al["point_group"] == "3"
    assert al["offcentre"] > 0.1                   # pushed away from the shared face
    assert "face" in {p["mode"] for p in pairs}
    assert row["face_sharing_d0_oct_dim"] == -1    # Al3+ is not a d0 z >= 4 cation


def test_square_planar_pt():
    _, sites, _ = analyse("t-4", _p1_cif("P4_2/mmc", Lattice.tetragonal(3.47, 6.11), ["Pt", "S"],
                                         [[0, .5, 0], [0, 0, .25]]))
    pt = _site(sites, "Pt")
    assert (pt["d_count"], pt["cn"], pt["geometry"]) == (8, 4, "square_planar")
    assert pt["pg_allows_square_planar"] and not pt["pg_allows_tetrahedron"]


def test_intermetallic_out_of_scope():
    row, sites, pairs = analyse("t-5", _p1_cif("Pm-3m", Lattice.cubic(3.0), ["Cu", "Zn"],
                                               [[0, 0, 0], [.5, .5, .5]]))
    assert row["status"] == "no_anion" and not sites and not pairs


@pytest.mark.parametrize("name,vectors", [
    ("tetrahedron", [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]),
    ("octahedron", [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]),
])
def test_classify_ideal(name, vectors):
    geom, rms = classify_polyhedron(np.array(vectors, float))
    assert geom == name and rms < 1e-6


def test_network_dimensionality():
    # a chain along x (one node linked to its own image), and an isolated dimer
    dims = network_dimensionality({0, 1, 2}, [(0, 0, (1, 0, 0)), (1, 2, (0, 0, 0))])
    assert dims == {0: 1, 1: 0, 2: 0}
