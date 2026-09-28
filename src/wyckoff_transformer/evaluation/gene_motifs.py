"""Coordination motifs that a Wyckoff gene guarantees, and prototype keys.

The structure-level screen (:mod:`wyckoff_transformer.evaluation.coordination_motifs`,
``docs/coordination_motif_screen.md``) needs a relaxed structure. This module asks what
the *gene* already decides: a gene fixes each site's point group, so it can rule a
polyhedron out, and for a few prototypes it fixes how the polyhedra are connected. A
motif flagged here holds in every structure built from the gene, provided the
relaxation keeps the symmetry. Structures are then needed only to test stability.

Flags, all restricted to compounds whose anions are only O and F. Oxidation states come
from the same single-valence ICSD-prior guess as the structure screen.

* ``fs_d0_chain_2h``: the 2H hexagonal-perovskite prototype (BaNiO3 type, P6_3/mmc,
  A 2c/2d, B 2a, X 6h) with a d0 B cation of z >= +4. The B octahedra share faces in
  infinite chains along c, so this is an extended face-sharing d0 network. The
  structure screen found none among 381k near-hull structures.
* ``perovskite_t_out``: a cubic (Pm-3m) or GdFeO3-type (Pnma) perovskite gene whose
  Goldschmidt t lies outside [0.70, 1.15]. The corner-sharing BX6 network is only
  guaranteed if the relaxation keeps the prototype.
* ``cr3_no_octahedron``: Cr3+ on a site whose point group admits no octahedron.
* ``d8_no_square``: Pd2+, Pt2+ or Au3+ on a site whose point group admits no square
  plane.
* ``sojt_pinned_strong`` / ``sojt_pinned_d0`` / ``lone_pair_pinned``: an SOJT d0 cation
  (strong = Mo6+ and V5+), or an ns2 cation, on a non-polar site, where symmetry
  forbids off-centring. The gene cannot tell whether the d0 site is octahedral.

Prototype key: :func:`~wyckoff_transformer.evaluation.novelty.record_to_strict_AFLOW_fingerprint`
(space group plus the multiset of Wyckoff orbits per element, with elements
anonymised and equivalent settings merged), hashed to 16 bytes. Two genes with the
same key are the same structure type, whatever their elements.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from functools import lru_cache

import numpy as np

from wyckoff_transformer.evaluation.coordination_motifs import (
    ALLOWED_BY,
    D0_HIGH_VALENT,
    LONE_PAIR,
    POLAR_POINT_GROUPS,
    SOJT_D0,
    guess_oxidation_states,
    largest_cn_radius,
    shannon_radius,
)
from wyckoff_transformer.evaluation.novelty import (
    record_to_augmented_fingerprint,
    record_to_strict_AFLOW_fingerprint,
)

OF_ANIONS = {"O", "F"}
#: Elements a flag can be about; a gene with none of them skips the oxidation guess.
TARGET_ELEMENTS = ({el for el, _ in D0_HIGH_VALENT} | {el for el, _ in LONE_PAIR}
                   | {"Cr", "Pd", "Pt", "Au"})

# --- site point groups ------------------------------------------------------------------


@lru_cache(maxsize=None)
def _site_point_groups(space_group: int) -> dict[str, str]:
    """Wyckoff letter -> unoriented site point group (spglib notation)."""
    import spglib
    from pyxtal.symmetry import Group

    rng = np.random.default_rng(0)
    group = Group(space_group)
    general = group[0].ops
    out = {}
    for wp in group.Wyckoff_positions:
        # A generic point of the orbit, so no accidental extra stabiliser.
        p0 = wp.ops[0].operate(rng.random(3))
        rots = []
        for op in general:
            d = op.operate(p0) - p0
            d -= np.round(d)
            if np.abs(d).max() < 1e-6:
                rots.append(np.rint(op.rotation_matrix).astype("intc"))
        out[wp.letter] = spglib.get_pointgroup(np.array(rots, dtype="intc"))[0].strip()
    return out


def site_point_group(space_group: int, letter: str) -> str:
    return _site_point_groups(int(space_group))[letter]


# --- keys -------------------------------------------------------------------------------


def _canonical(x):
    """Nested frozensets/tuples -> a deterministic nested tuple (for hashing)."""
    if isinstance(x, frozenset):
        return ("S",) + tuple(sorted((_canonical(e) for e in x), key=repr))
    if isinstance(x, tuple):
        return ("T",) + tuple(_canonical(e) for e in x)
    return x


def digest(fingerprint) -> bytes:
    return hashlib.blake2b(repr(_canonical(fingerprint)).encode(), digest_size=16).digest()


def prototype_key(record) -> bytes:
    return digest(record_to_strict_AFLOW_fingerprint(record))


def gene_key(record) -> bytes:
    return digest(record_to_augmented_fingerprint(record))


# --- reference prototypes ---------------------------------------------------------------

_REFERENCE_GENES = {
    "2H_BaNiO3": {"group": 194, "sites": [["2d"], ["2a"], ["6h"]],
                  "species": ["Ba", "Ni", "O"], "numIons": [2, 2, 6]},
    "cubic_perovskite": {"group": 221, "sites": [["1a"], ["1b"], ["3c"]],
                         "species": ["Sr", "Ti", "O"], "numIons": [1, 1, 3]},
    "GdFeO3": {"group": 62, "sites": [["4c"], ["4b"], ["4c", "8d"]],
               "species": ["Gd", "Fe", "O"], "numIons": [4, 4, 12]},
}


@lru_cache(maxsize=1)
def reference_prototype_keys() -> dict[bytes, str]:
    from wyckoff_transformer.evaluation.protocol import GeneFingerprinter

    fp = GeneFingerprinter()
    return {prototype_key(fp.record(g)): name for name, g in _REFERENCE_GENES.items()}


# --- flags ------------------------------------------------------------------------------

FLAGS = ("fs_d0_chain_2h", "perovskite_t_out", "cr3_no_octahedron", "d8_no_square",
         "sojt_pinned_strong", "sojt_pinned_d0", "lone_pair_pinned")


def _composition(record) -> dict[str, int]:
    comp = defaultdict(int)
    for el, m in zip(record["elements"], record["multiplicity"]):
        comp[str(el)] += int(m)
    return dict(comp)


def _reduced_formula(comp: dict[str, int]) -> str:
    from math import gcd
    from functools import reduce
    g = reduce(gcd, comp.values())
    return "".join(f"{el}{n // g}" for el, n in sorted(comp.items()))


def gene_flags(record) -> dict:
    """Gene-level motif flags for one record (augmented Wyckoff record).

    ``record`` needs ``spacegroup_number``, ``elements``, ``multiplicity`` and
    ``wyckoff_letters``; the prototype key needs the augmented fields too.
    """
    out = {f: False for f in FLAGS}
    out["tolerance_t"] = np.nan
    out["oxidation_ok"] = False
    elements = [str(e) for e in record["elements"]]
    present = set(elements)
    if not (present & OF_ANIONS) or not (present & TARGET_ELEMENTS):
        return out
    comp = _composition(record)
    oxi = guess_oxidation_states(_reduced_formula(comp))
    if oxi is None:
        return out
    oxi = dict(oxi)
    anions = {el for el, z in oxi.items() if z < 0}
    if not anions or not anions <= OF_ANIONS:
        return out
    out["oxidation_ok"] = True
    sg = int(record["spacegroup_number"])
    pgs = [site_point_group(sg, letter) for letter in record["wyckoff_letters"]]

    for el, pg in zip(elements, pgs):
        z = oxi[el]
        if (el, z) == ("Cr", 3) and pg not in ALLOWED_BY["octahedron"]:
            out["cr3_no_octahedron"] = True
        if (el, z) in {("Pd", 2), ("Pt", 2), ("Au", 3)} and pg not in ALLOWED_BY["square_planar"]:
            out["d8_no_square"] = True
        if pg not in POLAR_POINT_GROUPS:
            cls = SOJT_D0.get((el, z))
            if cls == "strong":
                out["sojt_pinned_strong"] = True
            if cls is not None:
                out["sojt_pinned_d0"] = True
            if (el, z) in LONE_PAIR:
                out["lone_pair_pinned"] = True

    proto = reference_prototype_keys().get(prototype_key(record)) \
        if "site_symmetries_augmented" in record else None
    out["reference_prototype"] = proto
    if proto == "2H_BaNiO3":
        b = [el for el, pg in zip(elements, pgs) if pg == "-3m"]
        if b and (b[0], oxi[b[0]]) in D0_HIGH_VALENT:
            out["fs_d0_chain_2h"] = True
    if proto in ("cubic_perovskite", "GdFeO3") and len(anions) == 1:
        (x,) = anions
        cats = [el for el in comp if el != x]
        if len(cats) == 2:
            r = {c: (largest_cn_radius(c, oxi[c]), shannon_radius(c, oxi[c], "VI")) for c in cats}
            a_el, b_el = sorted(cats, key=lambda c: r[c][0] or 0, reverse=True)
            r_x = shannon_radius(x, oxi[x], "VI")
            if r[a_el][0] and r[b_el][1] and r_x:
                t = (r[a_el][0] + r_x) / (np.sqrt(2) * (r[b_el][1] + r_x))
                out["tolerance_t"] = t
                out["perovskite_t_out"] = bool(t < 0.70 or t > 1.15)
    return out
