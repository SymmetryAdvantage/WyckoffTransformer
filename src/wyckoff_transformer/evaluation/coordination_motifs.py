"""Screen ionic crystal structures for the coordination-motif rules of
``docs/ideas/Theoretical & Crystallographic Rules Defining Unprecedented Coordination Motifs.md``.

Every structure gets one summary row, one row per symmetry-unique cation site, and
one row per (cation-pair class, sharing mode) it contains. The rarity tables are
computed from those rows by ``scripts/screen_coordination_motifs.py``; this module
only measures.

What is measured, and whether it needs the full structure or only the Wyckoff gene:

* oxidation states, d-electron count, HSAB class, Goldschmidt t and mu -- composition,
  hence gene level;
* site point group, whether it is polar, which ideal polyhedra it admits, number of
  Wyckoff orbits per element (Pauling's rule 5) -- gene level, read from the structure
  with the tolerance the genes were built with (0.1 A);
* coordination number, polyhedron shape, off-centring, bond-valence sums, polyhedral
  sharing modes and the dimensionality of face-sharing networks -- structure level.

Neighbours are anions within ``(1 + DISTANCE_TOLERANCE) * d_min`` of a cation, so a
site's ``gap`` (next-anion distance over last-bonded distance) says how well defined
its coordination number is. Shapes are assigned by comparing the sorted list of
ligand-cation-ligand angles with that of each ideal polyhedron of the same CN; this is
permutation invariant, cheap, and good enough to tell a tetrahedron from a square or
an octahedron from a trigonal prism, which is all the rules ask.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations

import numpy as np
import spglib
from pymatgen.core import Composition, Element, Lattice, Species
from pymatgen.analysis.bond_valence import BV_PARAMS
from pymatgen.optimization.neighbors import find_points_in_spheres

from wyckoff_transformer.evaluation.oxidation_state import (
    OXI_STATE_MAPPING_FILE,
    _load_json,
    compositional_oxi_state_guesses,
)

SYMPREC = 0.1                # what the Wyckoff genes were built with (data.py, tol=0.1)
NEIGHBOUR_CUTOFF = 4.5       # A; the search radius, not the bond criterion
DISTANCE_TOLERANCE = 0.25    # bonded if d <= (1 + tol) * d_min
IRREGULAR_ANGLE_RMS = 15.0   # degrees; worse than this against every reference -> "irregular"
BV_B = 0.37                  # Brown's softness parameter

ANIONS = {"N", "O", "F", "S", "Cl", "Se", "Br", "Te", "I"}
OXIDE_FLUORIDE = {"O", "F"}

# --- Chemistry tables -------------------------------------------------------------

#: d0 cations with z >= +4 (section 1C and 2A of the doc).
D0_HIGH_VALENT = {("Ti", 4), ("Zr", 4), ("Hf", 4), ("V", 5), ("Nb", 5), ("Ta", 5),
                  ("Cr", 6), ("Mo", 6), ("W", 6), ("Mn", 7), ("Tc", 7), ("Re", 7)}
#: Out-of-centre distortion strength of d0 octahedra (Halasyamani, Chem. Mater. 2004).
SOJT_D0 = {("Mo", 6): "strong", ("V", 5): "strong",
           ("W", 6): "intermediate", ("Ti", 4): "intermediate", ("Nb", 5): "intermediate",
           ("Ta", 5): "weak", ("Zr", 4): "weak", ("Hf", 4): "weak"}
#: ns2 lone-pair cations.
LONE_PAIR = {("Sn", 2), ("Sb", 3), ("Te", 4), ("Se", 4), ("I", 5), ("Bi", 3),
             ("Pb", 2), ("Tl", 1), ("As", 3), ("Ge", 2)}
D3_TARGETS = {("Cr", 3), ("Mn", 4)}
D8_LOW_SPIN_TARGETS = {("Pd", 2), ("Pt", 2), ("Au", 3)}

LANTHANOIDS = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
               "Tm", "Yb", "Lu"}
HARD_CATIONS = (
    {(el, 1) for el in ("H", "Li", "Na", "K", "Rb", "Cs")}
    | {(el, 2) for el in ("Be", "Mg", "Ca", "Sr", "Ba", "Mn")}
    | {(el, 3) for el in ("B", "Al", "Sc", "Y", "Ga", "In", "Cr", "Fe", "Co", "As")}
    | {(el, 3) for el in LANTHANOIDS}
    | {(el, 4) for el in ("Si", "Ge", "Sn", "Ti", "Zr", "Hf", "Th", "U", "Ce")}
    | D0_HIGH_VALENT
    | {("U", 6)}
)
SOFT_CATIONS = {("Cu", 1), ("Ag", 1), ("Au", 1), ("Au", 3), ("Tl", 1), ("Hg", 1),
                ("Hg", 2), ("Pd", 2), ("Pt", 2), ("Pt", 4), ("Cd", 2)}
HARD_ANIONS = {"O", "F"}      # N and Cl are borderline in many tables
SOFT_ANIONS = {"S", "Se", "Te", "I"}
PEROVSKITE_ANIONS = {"O", "F", "Cl", "Br", "I"}

POLAR_POINT_GROUPS = {"1", "2", "m", "mm2", "3", "3m", "4", "4mm", "6", "6mm"}
# Abstract subgroups of the ideal polyhedra's groups. Site symmetry must be one of
# them for the polyhedron to be symmetry-allowed on that site (necessary, since the
# orientation is not checked).
_TD = {"1", "2", "m", "3", "222", "mm2", "-4", "3m", "23", "-42m", "-4m2", "-43m"}
_D4H = {"1", "-1", "2", "m", "2/m", "222", "mm2", "mmm", "4", "-4", "4/m", "422",
        "4mm", "-42m", "-4m2", "4/mmm"}
_D3H = {"1", "2", "m", "3", "mm2", "-6", "32", "3m", "-6m2", "-62m"}
_OH = {"1", "-1", "2", "m", "2/m", "222", "mm2", "mmm", "4", "-4", "4/m", "422", "4mm",
       "-42m", "-4m2", "4/mmm", "3", "-3", "32", "3m", "-3m", "23", "m-3", "432",
       "-43m", "m-3m"}
ALLOWED_BY = {"tetrahedron": _TD, "square_planar": _D4H, "trigonal_prism": _D3H,
              "octahedron": _OH}

# --- Reference polyhedra ----------------------------------------------------------


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _sorted_angles(vectors: np.ndarray) -> np.ndarray:
    u = _unit(vectors)
    cos = np.clip(u @ u.T, -1.0, 1.0)
    iu = np.triu_indices(len(u), 1)
    return np.sort(np.degrees(np.arccos(cos[iu])))


def _prism(n_side: int, h_over_r: float, antiprism: bool = False) -> np.ndarray:
    phi = 2 * np.pi * np.arange(n_side) / n_side
    top = np.c_[np.cos(phi), np.sin(phi), np.full(n_side, h_over_r)]
    shift = np.pi / n_side if antiprism else 0.0
    bottom = np.c_[np.cos(phi + shift), np.sin(phi + shift), np.full(n_side, -h_over_r)]
    return np.r_[top, bottom]


_c120, _s120 = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
REFERENCES = {
    2: {"linear": [[0, 0, 1], [0, 0, -1]], "bent": [[0, 0, 1], [np.sin(np.radians(109.47)), 0,
                                                              np.cos(np.radians(109.47))]]},
    3: {"trigonal_planar": [[1, 0, 0], [_c120, _s120, 0], [_c120, -_s120, 0]],
        "trigonal_pyramid": [[1, 1, 1], [1, -1, -1], [-1, 1, -1]],
        "t_shaped": [[1, 0, 0], [-1, 0, 0], [0, 1, 0]]},
    4: {"tetrahedron": [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
        "square_planar": [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
        "seesaw": [[0, 0, 1], [0, 0, -1], [1, 0, 0], [_c120, _s120, 0]]},
    5: {"trigonal_bipyramid": [[0, 0, 1], [0, 0, -1], [1, 0, 0], [_c120, _s120, 0],
                               [_c120, -_s120, 0]],
        "square_pyramid": [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]]},
    6: {"octahedron": [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        # equilateral: edge a = r*sqrt(3), height a -> h/r = sqrt(3)/2
        "trigonal_prism": _prism(3, np.sqrt(3) / 2),
        "hexagonal_planar": [[np.cos(p), np.sin(p), 0] for p in np.arange(6) * np.pi / 3]},
    8: {"cube": [[x, y, z] for x in (1, -1) for y in (1, -1) for z in (1, -1)],
        "square_antiprism": _prism(4, 0.5946, antiprism=True),
        "hexagonal_bipyramid": [[np.cos(p), np.sin(p), 0] for p in np.arange(6) * np.pi / 3]
        + [[0, 0, 1], [0, 0, -1]]},
    12: {"cuboctahedron": [[a, b, 0] for a in (1, -1) for b in (1, -1)]
         + [[a, 0, b] for a in (1, -1) for b in (1, -1)]
         + [[0, a, b] for a in (1, -1) for b in (1, -1)],
         "anticuboctahedron": np.r_[_prism(3, 0.8165), [[np.cos(p), np.sin(p), 0]
                                                          for p in np.arange(6) * np.pi / 3]]},
}
REFERENCE_ANGLES = {cn: {name: _sorted_angles(np.asarray(v, float)) for name, v in refs.items()}
                    for cn, refs in REFERENCES.items()}


def classify_polyhedron(vectors: np.ndarray) -> tuple[str, float]:
    """Best-matching ideal polyhedron for ligand vectors, and the RMS angle misfit."""
    cn = len(vectors)
    refs = REFERENCE_ANGLES.get(cn)
    if cn < 2 or refs is None:
        return f"cn{cn}", float("nan")
    angles = _sorted_angles(vectors)
    best, best_rms = None, np.inf
    for name, ref in refs.items():
        rms = float(np.sqrt(np.mean((angles - ref) ** 2)))
        if rms < best_rms:
            best, best_rms = name, rms
    if best_rms > IRREGULAR_ANGLE_RMS:
        return f"irregular_cn{cn}", best_rms
    return best, best_rms


# --- Chemistry lookups ------------------------------------------------------------


@lru_cache(maxsize=None)
def d_electron_count(el: str, ox: int) -> int | None:
    """d count of a transition-metal cation (groups 3-12, lanthanoids excluded but La)."""
    e = Element(el)
    if e.is_lanthanoid and el != "La" or e.is_actinoid:
        return None
    if not 3 <= e.group <= 12 or e.row < 4:
        return None
    d = e.group - ox
    return d if 0 <= d <= 10 else None


def hsab_cation(el: str, ox: int) -> str:
    if (el, ox) in HARD_CATIONS:
        return "hard"
    if (el, ox) in SOFT_CATIONS:
        return "soft"
    return "borderline"


def hsab_anion(el: str) -> str:
    if el in HARD_ANIONS:
        return "hard"
    if el in SOFT_ANIONS:
        return "soft"
    return "borderline"


@lru_cache(maxsize=None)
def shannon_radius(el: str, ox: int, cn: str) -> float | None:
    try:
        return float(Species(el, ox).get_shannon_radius(cn, radius_type="ionic"))
    except Exception:
        # Several transition metals only list a high- or low-spin radius.
        try:
            radii = Species(el, ox)._data["Shannon radii"][str(ox)][cn]
            return float(next(iter(radii.values()))["ionic_radius"])
        except Exception:
            return None


@lru_cache(maxsize=None)
def largest_cn_radius(el: str, ox: int) -> float | None:
    """Shannon radius at the largest tabulated CN (the Goldschmidt A-site fallback)."""
    for cn in ("XII", "XI", "X", "IX", "VIII", "VII", "VI"):
        r = shannon_radius(el, ox, cn)
        if r is not None:
            return r
    return None


RADIUS_RATIO_BANDS = [(0.155, 2), (0.225, 3), (0.414, 4), (0.732, 6), (1.0, 8)]
CN_BANDS = [2, 3, 4, 6, 8, 12]


def radius_ratio_cn(ratio: float) -> int:
    for upper, cn in RADIUS_RATIO_BANDS:
        if ratio < upper:
            return cn
    return 12


def cn_band_index(cn: int) -> int:
    return int(np.argmin([abs(cn - b) for b in CN_BANDS]))


@lru_cache(maxsize=None)
def bond_valence_r0(cation: str, anion: str) -> float | None:
    """O'Keeffe & Brese (1991) R0 from the element parameters pymatgen ships."""
    try:
        p1, p2 = BV_PARAMS[Element(cation)], BV_PARAMS[Element(anion)]
    except KeyError:
        return None
    r1, c1, r2, c2 = p1["r"], p1["c"], p2["r"], p2["c"]
    return r1 + r2 - r1 * r2 * (np.sqrt(c1) - np.sqrt(c2)) ** 2 / (c1 * r1 + c2 * r2)


@lru_cache(maxsize=200_000)
def guess_oxidation_states(reduced_formula: str) -> tuple[tuple[str, int], ...] | None:
    """One integer oxidation state per element, or None.

    Uses the ICSD-prior ranking of LeMat-GenBench (as the charge-balance check does)
    and keeps the top solution only if every element has a single integer state:
    mixed-valence compounds are out of scope for rules that need a d count.
    """
    comp = Composition(reduced_formula)
    mapping = _load_json(OXI_STATE_MAPPING_FILE)
    override = {str(e): mapping[str(e)] for e in comp.elements if str(e) in mapping}
    try:
        sols = compositional_oxi_state_guesses(comp, all_oxi_states=False, max_sites=-1,
                                               oxi_states_override=override, target_charge=0)[0]
    except Exception:
        return None
    if not sols:
        return None
    best = sols[0]
    out = []
    for el, ox in best.items():
        if abs(ox - round(ox)) > 1e-6:
            return None
        out.append((str(el), int(round(ox))))
    return tuple(sorted(out))


# --- CIF ----------------------------------------------------------------------------


def parse_p1_cif(cif: str) -> tuple[Lattice, list[str], np.ndarray]:
    """Parse the pymatgen-written P1 CIFs of LeMat-Bulk without pymatgen's CifParser."""
    params = {}
    species, frac = [], []
    in_atoms = False
    header = []
    for line in cif.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("_cell_length_") or s.startswith("_cell_angle_"):
            key, value = s.split()[:2]
            params[key] = float(value)
            continue
        if s.startswith("_atom_site_"):
            in_atoms = True
            header.append(s)
            continue
        if in_atoms:
            if s.startswith("loop_") or s.startswith("_"):
                break
            parts = s.split()
            row = dict(zip(header, parts))
            if float(row.get("_atom_site_occupancy", 1.0)) < 0.999:
                raise ValueError("partial occupancy")
            species.append(row["_atom_site_type_symbol"])
            frac.append([float(row["_atom_site_fract_x"]), float(row["_atom_site_fract_y"]),
                         float(row["_atom_site_fract_z"])])
    if params.get("_symmetry_Int_Tables_number", 1) != 1 and "P 1" not in cif:
        raise ValueError("not a P1 CIF")
    lattice = Lattice.from_parameters(
        params["_cell_length_a"], params["_cell_length_b"], params["_cell_length_c"],
        params["_cell_angle_alpha"], params["_cell_angle_beta"], params["_cell_angle_gamma"])
    return lattice, species, np.asarray(frac, float)


# --- Symmetry -------------------------------------------------------------------------


def site_symmetry(lattice: Lattice, species: list[str], frac: np.ndarray):
    """(equivalent_atoms, space group number, per-atom point-group symbol) via spglib."""
    numbers = [Element(s).Z for s in species]
    cell = (lattice.matrix, frac, numbers)
    ds = spglib.get_symmetry_dataset(cell, symprec=SYMPREC)
    if ds is None:
        n = len(species)
        return np.arange(n), 1, ["1"] * n
    rots, trans = ds.rotations, ds.translations
    point_groups = {}
    for rep in np.unique(ds.equivalent_atoms):
        x = frac[rep]
        diff = np.einsum("nij,j->ni", rots, x) + trans - x
        diff -= np.round(diff)
        dist = np.linalg.norm(diff @ lattice.matrix, axis=1)
        stab = rots[dist < 2 * SYMPREC]
        try:
            point_groups[rep] = spglib.get_pointgroup(stab)[0].strip()
        except Exception:
            point_groups[rep] = "?"
    return ds.equivalent_atoms, int(ds.number), [point_groups[e] for e in ds.equivalent_atoms]


# --- Periodic-graph dimensionality ------------------------------------------------------


def network_dimensionality(nodes: set[int], edges: list[tuple[int, int, tuple]]) -> dict[int, int]:
    """Dimensionality (0-3) of each connected component of a periodic quotient graph.

    ``edges`` are (i, j, T): node i in the home cell is linked to node j in cell T.
    Returns node -> dimensionality of its component.
    """
    adj = defaultdict(list)
    for i, j, t in edges:
        t = np.asarray(t, int)
        adj[i].append((j, t))
        adj[j].append((i, -t))
    result = {}
    seen = set()
    for start in nodes:
        if start in seen:
            continue
        offsets = {start: np.zeros(3, int)}
        stack = [start]
        cycles = []
        while stack:
            u = stack.pop()
            for v, t in adj[u]:
                target = offsets[u] + t
                if v not in offsets:
                    offsets[v] = target
                    stack.append(v)
                else:
                    delta = target - offsets[v]
                    if delta.any():
                        cycles.append(delta)
        dim = int(np.linalg.matrix_rank(np.array(cycles))) if cycles else 0
        for v in offsets:
            result[v] = dim
            seen.add(v)
    return result


# --- The analysis ---------------------------------------------------------------------


def analyse(material_id: str, cif: str):
    """Return (structure_row, site_rows, pair_rows) for one structure.

    Structures outside scope (no anion, no single-valued oxidation-state assignment,
    an anion element with a non-negative state) get a structure row with ``status``
    explaining why and no other rows.
    """
    lattice, species, frac = parse_p1_cif(cif)
    comp = Composition(Counter(species))
    row = {"material_id": material_id, "source": re.match(r"[A-Za-z]*", material_id).group(0),
           "reduced_formula": comp.reduced_formula, "n_atoms": len(species),
           "n_elements": len(comp.elements), "status": "ok"}
    elements = {str(e) for e in comp.elements}
    if not elements & ANIONS:
        row["status"] = "no_anion"
        return row, [], []
    oxi = guess_oxidation_states(comp.reduced_formula)
    if oxi is None:
        row["status"] = "no_single_valence_oxidation_states"
        return row, [], []
    oxi = dict(oxi)
    negatives = {el for el, z in oxi.items() if z < 0}
    if not negatives or not negatives <= ANIONS or any(z == 0 for z in oxi.values()):
        row["status"] = "oxidation_states_not_ionic"
        return row, [], []

    equivalent, sg, point_group = site_symmetry(lattice, species, frac)
    row["space_group"] = sg
    # Rule 5: Wyckoff orbits per element.
    orbits = defaultdict(set)
    for i, el in enumerate(species):
        orbits[el].add(int(equivalent[i]))
    row["max_orbits_per_element"] = max(len(v) for v in orbits.values())
    row["orbits_per_element"] = len(set(equivalent)) / len(elements)

    cart = lattice.get_cartesian_coords(frac)
    is_anion = np.array([oxi[s] < 0 for s in species])
    anion_idx = np.flatnonzero(is_anion)
    cation_idx = np.flatnonzero(~is_anion)
    if len(cation_idx) == 0:
        row["status"] = "no_cation"
        return row, [], []

    centres, points, images, dists = find_points_in_spheres(
        cart[anion_idx], cart[cation_idx], NEIGHBOUR_CUTOFF, np.array([1, 1, 1], dtype=np.int64),
        lattice.matrix, tol=1e-8)
    images = np.rint(images).astype(int)

    by_cation = defaultdict(list)
    for c, p, img, d in zip(centres, points, images, dists):
        if d > 0.5:
            by_cation[int(c)].append((d, int(anion_idx[p]), tuple(img)))

    # Bonds: cation -> list of (anion, image, distance)
    bonds = {}
    site_info = {}
    for ci, c in enumerate(cation_idx):
        nb = sorted(by_cation.get(ci, []))
        if not nb:
            bonds[c] = []
            continue
        d_min = nb[0][0]
        bonded = [x for x in nb if x[0] <= (1 + DISTANCE_TOLERANCE) * d_min]
        next_d = nb[len(bonded)][0] if len(nb) > len(bonded) else np.inf
        bonds[c] = [(a, img, d) for d, a, img in bonded]
        site_info[c] = {"gap": next_d / bonded[-1][0]}
    if not any(bonds.values()):
        row["status"] = "no_bonds"
        return row, [], []

    # Anion-side view and bond valences
    anion_bonds = defaultdict(list)          # anion -> [(cation, image of cation)]
    bv_sum = defaultdict(float)
    ebs_sum = defaultdict(float)
    for c, bl in bonds.items():
        el_c, z_c = species[c], oxi[species[c]]
        cn = len(bl)
        for a, img, d in bl:
            anion_bonds[a].append((c, tuple(-np.asarray(img))))
            r0 = bond_valence_r0(el_c, species[a])
            s = np.exp((r0 - d) / BV_B) if r0 is not None else np.nan
            bv_sum[c] += s
            bv_sum[a] += s
            ebs_sum[a] += z_c / cn

    # Per-cation polyhedron
    polyhedra = {}
    for c, bl in bonds.items():
        if not bl:
            continue
        vecs = np.array([cart[a] + np.asarray(img) @ lattice.matrix - cart[c] for a, img, _ in bl])
        geom, rms = classify_polyhedron(vecs)
        ligands = {species[a] for a, _, _ in bl}
        dists_c = np.array([d for _, _, d in bl])
        offcentre = float(np.linalg.norm(vecs.mean(axis=0)))
        delta_d = np.nan
        if len(bl) == 6 and geom == "octahedron":
            u = _unit(vecs)
            cos = u @ u.T
            used, delta_d = set(), 0.0
            for i in range(6):
                if i in used:
                    continue
                j = int(np.argmin(cos[i]))
                used |= {i, j}
                delta_d += abs(dists_c[i] - dists_c[j]) / max(abs(cos[i, j]), 1e-3)
        polyhedra[c] = {"cn": len(bl), "geometry": geom, "angle_rms": rms,
                        "ligands_of": ligands <= OXIDE_FLUORIDE,
                        "ligand_classes": {hsab_anion(x) for x in ligands},
                        "offcentre": offcentre, "delta_d": delta_d,
                        "mean_d": float(dists_c.mean())}

    # Sharing between cation polyhedra: count shared anions per pair class.
    shared = defaultdict(int)
    for a, cl in anion_bonds.items():
        for (c1, t1), (c2, t2) in combinations(cl, 2):
            dt = tuple(np.asarray(t2) - np.asarray(t1))
            if (c1, dt) > (c2, tuple(-x for x in dt)) if c1 == c2 else c1 > c2:
                c1, c2, dt = c2, c1, tuple(-x for x in dt)
            if c1 == c2 and not any(dt):
                continue
            shared[(c1, c2, dt)] += 1

    def _mode(n):
        return "corner" if n == 1 else "edge" if n == 2 else "face"

    # --- site rows (one per symmetry-unique cation) ---
    site_rows = []
    for c in cation_idx:
        if equivalent[c] != c or c not in polyhedra:
            continue
        el, z = species[c], oxi[species[c]]
        pol = polyhedra[c]
        pg = point_group[c]
        dcount = d_electron_count(el, z)
        r_c = shannon_radius(el, z, "VI")
        anion_els = sorted({species[a] for a, _, _ in bonds[c]})
        r_a = shannon_radius(anion_els[0], oxi[anion_els[0]], "VI") if len(anion_els) == 1 else None
        rr_cn = radius_ratio_cn(r_c / r_a) if r_c and r_a else None
        site_rows.append({
            "material_id": material_id, "site": int(c), "element": el, "ox": z,
            "multiplicity": int((equivalent == c).sum()), "d_count": dcount,
            "cn": pol["cn"], "geometry": pol["geometry"], "angle_rms": pol["angle_rms"],
            "gap": site_info[c]["gap"], "ligands": "".join(anion_els),
            "ligands_of": pol["ligands_of"], "point_group": pg,
            "polar_site": pg in POLAR_POINT_GROUPS,
            **{f"pg_allows_{k}": pg in v for k, v in ALLOWED_BY.items()},
            "offcentre": pol["offcentre"], "delta_d": pol["delta_d"], "mean_d": pol["mean_d"],
            "bvs": bv_sum[c], "radius_ratio_cn": rr_cn,
            "hsab": hsab_cation(el, z),
            "ligand_hsab": "/".join(sorted(pol["ligand_classes"])),
        })

    # --- pair rows, aggregated per (species pair, mode) ---
    pair_agg = {}
    fs_nodes, fs_edges = set(), []          # face-sharing d0 high-valent octahedra
    bb_edges = []                           # all B-B links, for the perovskite test
    for (c1, c2, dt), n in shared.items():
        if c1 not in polyhedra or c2 not in polyhedra:
            continue
        p1, p2 = polyhedra[c1], polyhedra[c2]
        k1 = (species[c1], oxi[species[c1]], p1["cn"], p1["geometry"], p1["ligands_of"])
        k2 = (species[c2], oxi[species[c2]], p2["cn"], p2["geometry"], p2["ligands_of"])
        if k2 < k1:
            k1, k2 = k2, k1
        mode = _mode(n)
        d_mm = float(np.linalg.norm(cart[c2] + np.asarray(dt) @ lattice.matrix - cart[c1]))
        key = (k1, k2, mode)
        cnt, dmin = pair_agg.get(key, (0, np.inf))
        pair_agg[key] = (cnt + 1, min(dmin, d_mm))
        both_d0 = ((species[c1], oxi[species[c1]]) in D0_HIGH_VALENT
                   and (species[c2], oxi[species[c2]]) in D0_HIGH_VALENT)
        if (mode == "face" and both_d0 and p1["cn"] == 6 and p2["cn"] == 6
                and p1["ligands_of"] and p2["ligands_of"]):
            fs_nodes |= {c1, c2}
            fs_edges.append((c1, c2, dt))
        bb_edges.append((c1, c2, dt, mode))

    pair_rows = [{"material_id": material_id,
                  "el1": k1[0], "ox1": k1[1], "cn1": k1[2], "geom1": k1[3], "of1": k1[4],
                  "el2": k2[0], "ox2": k2[1], "cn2": k2[2], "geom2": k2[3], "of2": k2[4],
                  "mode": mode, "n_links": cnt, "min_d_mm": dmin}
                 for (k1, k2, mode), (cnt, dmin) in pair_agg.items()]

    row["face_sharing_d0_oct_dim"] = (max(network_dimensionality(fs_nodes, fs_edges).values())
                                      if fs_nodes else -1)

    # --- Rule 2: anion bond-valence and electrostatic-strength sums ---
    anion_dev_bvs, anion_dev_ebs = [], []
    for a in anion_idx:
        if a not in anion_bonds:
            continue
        v = abs(oxi[species[a]])
        anion_dev_bvs.append((bv_sum[a] - v) / v)
        anion_dev_ebs.append((ebs_sum[a] - v) / v)
    cation_dev = [(bv_sum[c] - oxi[species[c]]) for c in polyhedra]
    all_dev = cation_dev + [bv_sum[a] - abs(oxi[species[a]]) for a in anion_idx if a in anion_bonds]
    row["gii"] = float(np.sqrt(np.nanmean(np.square(all_dev)))) if all_dev else np.nan
    row["max_anion_ebs_dev"] = float(np.max(np.abs(anion_dev_ebs))) if anion_dev_ebs else np.nan
    row["max_anion_bvs_dev"] = float(np.nanmax(np.abs(anion_dev_bvs))) if anion_dev_bvs else np.nan
    row["unbonded_anions"] = int(len(anion_idx) - len(anion_bonds))

    # --- Section 3: HSAB inversion ---
    anion_classes = {hsab_anion(el) for el in negatives}
    row["mixed_hard_soft_anions"] = {"hard", "soft"} <= anion_classes
    hard_in_soft = soft_in_hard = False
    for c, pol in polyhedra.items():
        cls = hsab_cation(species[c], oxi[species[c]])
        if cls == "hard" and pol["ligand_classes"] == {"soft"}:
            hard_in_soft = True
        if cls == "soft" and pol["ligand_classes"] == {"hard"}:
            soft_in_hard = True
    row["hsab_hard_cation_soft_cage"] = hard_in_soft and row["mixed_hard_soft_anions"]
    row["hsab_soft_cation_hard_cage"] = soft_in_hard and row["mixed_hard_soft_anions"]
    row["hsab_inversion"] = hard_in_soft and soft_in_hard and row["mixed_hard_soft_anions"]

    # --- Section 4: Goldschmidt ---
    row["tolerance_t"] = row["octahedral_mu"] = np.nan
    row["perovskite_corner_3d"] = False
    red = comp.reduced_composition
    if len(red) == 3 and len(negatives) == 1:
        (x_el,) = negatives
        cats = [str(e) for e in red.elements if str(e) != x_el]
        amounts = sorted([red[c] for c in cats])
        if x_el in PEROVSKITE_ANIONS and amounts == [1, 1] and red[x_el] == 3:
            r_x = shannon_radius(x_el, oxi[x_el], "VI")
            radii = {c: (largest_cn_radius(c, oxi[c]), shannon_radius(c, oxi[c], "VI")) for c in cats}
            a_el, b_el = sorted(cats, key=lambda c: radii[c][0] or 0, reverse=True)
            r_a, r_b = radii[a_el][0], radii[b_el][1]
            if r_x and r_a and r_b:
                row["tolerance_t"] = (r_a + r_x) / (np.sqrt(2) * (r_b + r_x))
                row["octahedral_mu"] = r_b / r_x
                b_sites = {c for c in polyhedra if species[c] == b_el}
                b_links = [(c1, c2, dt, m) for c1, c2, dt, m in bb_edges
                           if c1 in b_sites and c2 in b_sites]
                if (b_sites and all(polyhedra[c]["cn"] == 6 for c in b_sites)
                        and b_links and all(m == "corner" for *_, m in b_links)):
                    degree = defaultdict(int)
                    for c1, c2, _, _ in b_links:
                        degree[c1] += 1
                        degree[c2] += 1
                    dims = network_dimensionality(b_sites, [(c1, c2, dt) for c1, c2, dt, _ in b_links])
                    row["perovskite_corner_3d"] = (all(degree[c] == 6 for c in b_sites)
                                                   and min(dims.values()) == 3)
    return row, site_rows, pair_rows
