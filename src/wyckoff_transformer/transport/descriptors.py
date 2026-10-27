"""Cheap structure descriptors of Li transport, for the Phase 1 benchmark.

* **BVSE percolation barriers** (``bvlain``, Dembitskiy et al.): the 1D/2D/3D
  percolation energies of a bond-valence site-energy map of Li in the structure.
* **Framework connectivity** (Jun et al., Nat. Mater. 21, 924, 2022): whether the
  non-Li cation polyhedra share only corners with each other (or nothing), the
  dimensionality of the network they form, and the share of anions in finite,
  topologically isolated groups (PS4, BH4, YCl6 units).
* **Anion framework** (Wang et al., Nat. Mater. 14, 1026, 2015): whether the anion
  sublattice matches bcc, fcc or hcp, and the volume per anion.
* **Li environment**: coordination number, bond-length distortion index and off-centring
  of every Li site, and the shortest Li-Li and Li-anion separations Sendek et al. build
  their model on.

Bonds use the rule of :mod:`wyckoff_transformer.evaluation.coordination_motifs`: an anion
is bonded to a cation if its distance divided by the pair's bond-valence R0 is within
25% of the shortest such ratio.  Anion roles come from
:func:`wyckoff_transformer.transport.frameworks.structure_anions`.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from itertools import combinations

import numpy as np
from pymatgen.core import Lattice, Structure

from wyckoff_transformer.evaluation.coordination_motifs import (
    DISTANCE_TOLERANCE,
    NEIGHBOUR_CUTOFF,
    bond_valence_r0,
    network_dimensionality,
)
from wyckoff_transformer.transport.frameworks import structure_anions

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# BVSE
# --------------------------------------------------------------------------- #
def bvse_barriers(structure: Structure, mobile_ion: str = "Li1+", resolution: float = 0.2,
                  r_cut: float = 10.0, encut: float = 10.0) -> dict[str, float]:
    """``bvlain``'s BVSE percolation barriers, eV; NaN when oxidation states fail."""
    import warnings

    from bvlain import Lain

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lain = Lain(verbose=False)
        try:
            lain.read_structure(structure.copy(), oxi_check=True)
            lain.bvse_distribution(mobile_ion=mobile_ion, r_cut=r_cut, resolution=resolution)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            # No bond-valence oxidation states, or no softBV parameters for one of the
            # ions (bvlain raises KeyError for e.g. Sb3-).
            logger.debug("BVSE: cannot map %s (%r)", structure.formula, exc)
            return {"bvse_E_1D": math.nan, "bvse_E_2D": math.nan, "bvse_E_3D": math.nan}
        barriers = lain.percolation_barriers(encut=encut)
    return {f"bvse_{key}": float(value) for key, value in barriers.items()}


# --------------------------------------------------------------------------- #
# Bonds
# --------------------------------------------------------------------------- #
def cation_anion_bonds(structure: Structure, anions: frozenset[str]
                       ) -> dict[int, list[tuple[int, tuple[int, int, int], float]]]:
    """Cation site -> ``[(anion site, image of the anion, distance)]``."""
    from pymatgen.optimization.neighbors import find_points_in_spheres

    species = [site.specie.symbol for site in structure]
    cart = structure.cart_coords
    is_anion = np.array([s in anions for s in species])
    anion_idx = np.flatnonzero(is_anion)
    cation_idx = np.flatnonzero(~is_anion)
    if not len(anion_idx) or not len(cation_idx):
        return {int(c): [] for c in cation_idx}
    centres, points, images, dists = find_points_in_spheres(
        np.ascontiguousarray(cart[anion_idx]), np.ascontiguousarray(cart[cation_idx]),
        NEIGHBOUR_CUTOFF, np.array([1, 1, 1], dtype=np.int64),
        np.ascontiguousarray(structure.lattice.matrix), tol=1e-8)
    images = np.rint(images).astype(int)
    by_cation = defaultdict(list)
    for c, p, img, d in zip(centres, points, images, dists):
        if d > 0.5:
            by_cation[int(c)].append((float(d), int(anion_idx[p]), tuple(int(x) for x in img)))
    bonds = {}
    for ci, c in enumerate(cation_idx):
        found = by_cation.get(ci, [])
        if not found:
            bonds[int(c)] = []
            continue
        r0 = [bond_valence_r0(species[c], species[a]) for _, a, _ in found]
        if any(r is None for r in r0):
            r0 = [1.0] * len(found)
        ranked = sorted((d / r, d, a, img) for (d, a, img), r in zip(found, r0))
        q_min = ranked[0][0]
        bonds[int(c)] = [(a, img, d) for q, d, a, img in ranked
                         if q <= (1 + DISTANCE_TOLERANCE) * q_min]
    return bonds


# --------------------------------------------------------------------------- #
# Anion framework
# --------------------------------------------------------------------------- #
_IDEAL = {
    "bcc": Structure(Lattice.cubic(3.0), ["Rn", "Rn"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    "fcc": Structure(Lattice.cubic(3.0), ["Rn"] * 4,
                     [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]),
    "hcp": Structure(Lattice.hexagonal(3.0, 3.0 * math.sqrt(8 / 3)), ["Rn", "Rn"],
                     [[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]]),
}


def anion_lattice_type(structure: Structure, anions: frozenset[str],
                       ltol: float = 0.3, stol: float = 0.5, angle_tol: float = 10.0) -> str:
    """``bcc``, ``fcc``, ``hcp`` or ``other``: the ideal lattice the anions match best.

    Every anion becomes one species, so mixed-anion sublattices are judged as one.
    The tolerances are looser than for structure identity because the anion sublattice
    of a real compound is distorted; Wang et al. matched with the same tool.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher

    sites = [site for site in structure if site.specie.symbol in anions]
    if not sites:
        return "none"
    sublattice = Structure(structure.lattice, ["Rn"] * len(sites),
                           [site.frac_coords for site in sites])
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol, primitive_cell=True,
                               scale=True, attempt_supercell=True)
    best, best_rms = "other", math.inf
    for name, ideal in _IDEAL.items():
        try:
            rms = matcher.get_rms_dist(ideal, sublattice)
        except (ValueError, IndexError):
            rms = None
        if rms is not None and rms[0] < best_rms:
            best, best_rms = name, rms[0]
    return best


# --------------------------------------------------------------------------- #
# Gene level
# --------------------------------------------------------------------------- #
def gene_descriptors(structure: Structure, anions: frozenset[str], mobile: str = "Li",
                     tol: float = 0.1) -> dict:
    """What a Wyckoff gene alone fixes about Li: content, anion softness, Li sites.

    The gene is read with pyxtal at the tolerance the training genes were built with
    (:func:`wyckoff_transformer.data.kick_pyxtal_until_it_works`, 0.1 A).
    """
    from pymatgen.core import Element

    composition = structure.composition
    out = {"li_fraction": composition.get_atomic_fraction(mobile)}
    anion_atoms = [(Element(a).X, composition[a]) for a in anions]
    if anion_atoms:
        out["anion_electronegativity"] = (sum(x * n for x, n in anion_atoms)
                                          / sum(n for _, n in anion_atoms))
    try:
        from wyckoff_transformer.data import kick_pyxtal_until_it_works

        gene = kick_pyxtal_until_it_works(structure, tol=tol, attempts=6)
    except Exception as exc:  # noqa: BLE001 - pyxtal fails in many ways
        logger.debug("gene of %s: %r", structure.formula, exc)
        return out
    li_sites = [site for site in gene.atom_sites if site.specie == mobile]
    out["gene_spacegroup"] = gene.group.number
    out["n_li_orbits"] = len(li_sites)
    if li_sites:
        out["li_free_site_fraction"] = float(np.mean([site.wp.get_dof() > 0
                                                      for site in li_sites]))
    return out


# --------------------------------------------------------------------------- #
# All of them
# --------------------------------------------------------------------------- #
def coarse_descriptors(structure: Structure, mobile: str = "Li") -> dict:
    """Every cheap descriptor except BVSE, gene-level ones included, as one flat dict."""
    anions = structure_anions(structure, mobile)
    species = [site.specie.symbol for site in structure]
    n_anions = sum(s in anions for s in species)
    out: dict = {"anions": "".join(sorted(anions)), "n_anions": n_anions,
                 "volume_per_anion": structure.volume / n_anions if n_anions else math.nan,
                 "volume_per_atom": structure.volume / len(structure),
                 **gene_descriptors(structure, anions, mobile)}
    if not n_anions:
        return out
    out["anion_lattice"] = anion_lattice_type(structure, anions)
    bonds = cation_anion_bonds(structure, anions)
    lattice = structure.lattice
    cart = structure.cart_coords

    # Li environments.
    li_sites = [i for i, s in enumerate(species) if s == mobile]
    cns, distortions, offcentres, li_anion = [], [], [], []
    for i in li_sites:
        bl = bonds.get(i, [])
        if not bl:
            continue
        d = np.array([x[2] for x in bl])
        vectors = np.array([cart[a] + np.asarray(img) @ lattice.matrix - cart[i]
                            for a, img, _ in bl])
        cns.append(len(bl))
        distortions.append(float(np.mean(np.abs(d - d.mean()) / d.mean())))
        offcentres.append(float(np.linalg.norm(vectors.mean(axis=0))))
        li_anion.append(float(d.min()))
    if cns:
        out.update({"li_cn_mean": float(np.mean(cns)), "li_cn_min": int(min(cns)),
                    "li_cn_max": int(max(cns)),
                    "li_low_cn_fraction": float(np.mean(np.array(cns) <= 3)),
                    "li_distortion_mean": float(np.mean(distortions)),
                    "li_distortion_max": float(np.max(distortions)),
                    "li_offcentre_mean": float(np.mean(offcentres)),
                    "li_anion_shortest_mean": float(np.mean(li_anion))})
    if len(li_sites) > 1:
        from pymatgen.optimization.neighbors import find_points_in_spheres

        li_cart = np.ascontiguousarray(cart[li_sites])
        c, _, _, d = find_points_in_spheres(
            li_cart, li_cart, 8.0, np.array([1, 1, 1], dtype=np.int64),
            np.ascontiguousarray(lattice.matrix), tol=1e-8)
        shortest = defaultdict(lambda: math.inf)
        for centre, dist in zip(c, d):
            if dist > 0.5:
                shortest[int(centre)] = min(shortest[int(centre)], float(dist))
        values = [v for v in shortest.values() if math.isfinite(v)]
        if values:
            out["li_li_shortest_mean"] = float(np.mean(values))
            out["li_li_shortest_min"] = float(np.min(values))

    # Framework: non-Li cation polyhedra and the anions they share.
    framework_cations = {c for c in bonds if species[c] != mobile and bonds[c]}
    anion_owners = defaultdict(list)          # anion -> [(cation, image of the cation)]
    for c in framework_cations:
        for a, img, _ in bonds[c]:
            anion_owners[a].append((c, tuple(-x for x in img)))
    shared = defaultdict(int)
    for owners in anion_owners.values():
        for (c1, t1), (c2, t2) in combinations(owners, 2):
            dt = tuple(b - a for a, b in zip(t1, t2))
            if c1 == c2 and not any(dt):
                continue
            key = (c1, c2, dt) if (c1, dt) <= (c2, tuple(-x for x in dt)) else \
                (c2, c1, tuple(-x for x in dt))
            shared[key] += 1
    modes = defaultdict(int)
    for n in shared.values():
        modes["corner" if n == 1 else "edge" if n == 2 else "face"] += 1
    out.update({"framework_corner_links": modes["corner"],
                "framework_edge_links": modes["edge"],
                "framework_face_links": modes["face"],
                "corner_sharing_or_isolated": int(modes["edge"] == 0 and modes["face"] == 0)})
    # Network of framework cations and their anions: its dimensionality and the share
    # of anions in finite groups.
    nodes = set(framework_cations) | set(anion_owners)
    edges = [(c, a, img) for c in framework_cations for a, img, _ in bonds[c]]
    if nodes:
        dims = network_dimensionality(nodes, edges)
        out["framework_dimensionality"] = int(max(dims.values()))
        anion_dims = [dims.get(i, 0) for i, s in enumerate(species) if s in anions]
        out["isolated_anion_fraction"] = float(np.mean([d == 0 for d in anion_dims]))
    else:
        out["framework_dimensionality"] = 0
        out["isolated_anion_fraction"] = 1.0
    return out
