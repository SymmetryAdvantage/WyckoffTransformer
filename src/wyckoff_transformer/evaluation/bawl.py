"""BAWL structure fingerprinting, as LeMat-GenBench computes it.

BAWL -- Bonding Algorithm Weisfeiler-Lehman -- hashes a structure by its bonding
graph, composition, and space group.  It is what LeMat-GenBench's uniqueness and
novelty metrics compare, so our fingerprints have to be byte-identical to theirs
or the numbers stop being comparable.

This is a narrowed port of ``material_hasher.hasher.bawl``, keeping only the
configuration the benchmark actually uses -- Weisfeiler-Lehman over an EconNN
bonding graph, with the SPGLib symmetry label -- and dropping the AFLOW and moyo
label backends and the abstract-hasher plumbing.  The Weisfeiler-Lehman hash
itself is not reimplemented: it stays in ``structuregraph-helpers``, the same
PyPI package material-hasher delegates to.

``tests/test_genbench_equivalence.py`` asserts string equality against
material-hasher's own ``BAWLHasher`` when it is installed.

Attribution: derived from LeMaterial/material-hasher (Apache-2.0),
https://github.com/LeMaterial/material-hasher.

References:
    Siron et al., LeMat-Bulk, AI for Accelerated Materials Design, ICLR 2025.
    Ongari et al., J. Chem. Eng. Data 67.7 (2022) 1743-1756.
"""
from __future__ import annotations

import logging
from typing import Optional

from networkx import Graph
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import EconNN, NearNeighbors
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from structuregraph_helpers.hash import generate_hash

logger = logging.getLogger(__name__)

#: EconNN settings LeMat-GenBench passes to the bonding algorithm.  Changing any
#: of these changes every fingerprint.
BONDING_KWARGS = {"tol": 0.2, "cutoff": 10, "use_fictive_radius": True}

#: Symmetry tolerance for the SPGLib label.
SYMPREC = 0.01

#: Weisfeiler-Lehman iterations, node-decorated and not edge-decorated, matching
#: material-hasher's ``generate_hash(graph, True, False, 100)``.
WL_ITERATIONS = 100


def structure_graph(
    structure: Structure,
    bonding_kwargs: Optional[dict] = None,
    bonding_algorithm: type[NearNeighbors] = EconNN,
) -> Graph:
    """Bonding graph of *structure*, decorated the way BAWL expects.

    Nodes carry the element name and edges the periodic image offset, so the
    hash distinguishes structures that share a topology but differ in what sits
    on it or in how bonds wrap the cell.
    """
    graph = StructureGraph.with_local_env_strategy(
        structure=structure,
        strategy=bonding_algorithm(**(BONDING_KWARGS if bonding_kwargs is None else bonding_kwargs)),
    )
    for index, site in enumerate(structure):
        graph.graph.nodes[index]["specie"] = site.specie.name
    for edge in graph.graph.edges:
        graph.graph.edges[edge]["voltage"] = graph.graph.edges[edge]["to_jimage"]
    return graph.graph


def spglib_symmetry_label(structure: Structure, symprec: float = SYMPREC) -> Optional[int]:
    """Space group number, or ``None`` when symmetry detection fails."""
    try:
        return SpacegroupAnalyzer(structure, symprec).get_symmetry_dataset().number
    except Exception as exc:
        logger.warning("Could not determine the symmetry label: %s", exc)
        return None


class BawlFingerprinter:
    """Computes BAWL fingerprints.

    Args:
        shorten: Drop the symmetry label from the hash.  ``True`` reproduces
            LeMat-GenBench's ``"short-bawl"``, which is what its leaderboard
            configuration uses; ``False`` reproduces ``"bawl"``.
        include_composition: Include the reduced formula.  The benchmark sets
            this on for both variants.
    """

    def __init__(self, shorten: bool = True, include_composition: bool = True) -> None:
        self.shorten = shorten
        self.include_composition = include_composition

    def components(self, structure: Structure) -> dict:
        """The pieces of the hash, in the order they are joined."""
        data = {"bonding_graph_hash": generate_hash(
            structure_graph(structure), True, False, WL_ITERATIONS
        )}
        if not self.shorten:
            data["symmetry_label"] = spglib_symmetry_label(structure)
        if self.include_composition:
            data["composition"] = structure.composition.formula.replace(" ", "")
        return data

    def __call__(self, structure: Structure) -> str:
        """Fingerprint *structure*.

        ``None`` components become empty strings rather than the literal
        "None", matching material-hasher, so a structure whose symmetry could
        not be determined still yields a stable hash.
        """
        return "_".join(
            str(value) if value is not None else ""
            for value in self.components(structure).values()
        )


def get_fingerprint(structure: Structure, fingerprinter: BawlFingerprinter) -> Optional[str]:
    """Fingerprint *structure*, returning ``None`` if it cannot be computed.

    Mirrors LeMat-GenBench's helper, including its use of a ``fingerprint``
    entry in ``structure.properties`` as a cache.
    """
    cached = structure.properties.get("fingerprint")
    if cached is not None:
        return cached
    try:
        return fingerprinter(structure)
    except Exception as exc:
        logger.warning("Fingerprinting failed: %s", exc)
        return None
