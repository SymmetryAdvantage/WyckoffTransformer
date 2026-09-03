"""Energy above the LeMat-Bulk MLIP convex hull.

This reimplements what LeMat-GenBench's ``get_energy_above_hull`` does, without
depending on it.  There is little to reimplement: the thermodynamics is
pymatgen's :class:`~pymatgen.analysis.phase_diagram.PhaseDiagram`, and the
reference is the published ``LeMaterial/LeMat-Bulk-MLIP-Hull`` dataset.  What
the benchmark adds is only the step of restricting the phase diagram to the
compositional subspace of the query, which is a set-containment test.

Because both inputs are external and versioned -- pymatgen and a pinned
HuggingFace dataset -- the numbers here do not drift when LeMat-GenBench
changes.  ``tests/test_genbench_equivalence.py`` pins them against the
benchmark's own implementation.

The hull is *self-consistent*: each MLIP's energies are referenced to a hull
built from that same MLIP.  Mixing them is what
:mod:`wyckoff_transformer.evaluation.hull_mlips` exists to prevent.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element

from wyckoff_transformer.evaluation.hull_mlips import HULL_REPO_ID, resolve_hull_mlip

logger = logging.getLogger(__name__)

#: Highest atomic number the one-hot composition encoding covers.
N_ELEMENTS = 118


def _one_hot(elements) -> np.ndarray:
    """Indicator over atomic numbers, matching LeMat-GenBench's encoding."""
    vector = np.zeros(N_ELEMENTS, dtype=np.int64)
    for element in elements:
        obj = Element(element) if isinstance(element, str) else element
        vector[int(obj.number) - 1] = 1
    return vector


class HullEnergyCalculator:
    """Computes ``e_above_hull`` against one published LeMat-Bulk MLIP hull.

    The reference table is loaded once and encoded into a dense indicator
    matrix, so each query costs one matrix-vector product plus a phase diagram
    over the handful of entries that survive.

    Args:
        mlip: A runnable hull name, e.g. ``"orb_conserv_inf"``.
        parquet: The published parquet.  Downloaded from
            :data:`~wyckoff_transformer.evaluation.hull_mlips.HULL_REPO_ID`
            when omitted, which uses the local HuggingFace cache if present.
    """

    def __init__(self, mlip: str, parquet: Optional[Path] = None) -> None:
        self.spec = resolve_hull_mlip(mlip)
        self.entries = self._load(self.spec.hull_type, parquet)
        # Row i is the indicator of the elements present in entry i.  A hull
        # entry is usable for a query only if it introduces no element the
        # query lacks, i.e. its indicator has no overlap with the complement of
        # the query's -- exactly LeMat-GenBench's `(all_compositions @
        # forbidden) == 0`.
        self._composition_matrix = np.stack(
            [_one_hot(species) for species in self.entries["species_at_sites"]]
        )
        logger.info(
            "Loaded %d %s hull entries", len(self.entries), self.spec.hull_type
        )

    @staticmethod
    def _load(hull_type: str, parquet: Optional[Path]) -> pd.DataFrame:
        if parquet is None:
            from huggingface_hub import hf_hub_download

            parquet = hf_hub_download(
                repo_id=HULL_REPO_ID,
                filename=f"data/{hull_type}-00000-of-00001.parquet",
                repo_type="dataset",
            )
        frame = pd.read_parquet(parquet, columns=["species_at_sites", "energy"])
        return frame.dropna(subset=["energy"]).reset_index(drop=True)

    def subspace(self, composition: Composition) -> pd.DataFrame:
        """Hull entries whose elements are all present in *composition*."""
        forbidden = 1 - _one_hot(composition.elements)
        return self.entries.loc[(self._composition_matrix @ forbidden) == 0]

    def energy_above_hull(
        self, total_energy: float, composition: Composition
    ) -> float:
        """Energy above hull in eV/atom.

        Args:
            total_energy: Total energy of the structure in eV, from the *same*
                potential that defines this hull.
            composition: Its composition.  Charged species are reduced to
                neutral elements, as the reference entries are neutral.

        Returns:
            eV/atom above the hull; negative below it.

        Raises:
            ValueError: If the hull holds no entry in this composition's
                subspace, or the phase diagram cannot be built.
        """
        subspace = self.subspace(composition)
        entries = [
            PDEntry(Composition(Counter(row["species_at_sites"])), row["energy"])
            for _, row in subspace.iterrows()
        ]
        if not entries:
            raise ValueError(
                f"No {self.spec.hull_type} hull entries within the composition "
                f"subspace of {composition.formula}"
            )

        neutral: dict = {}
        for element, count in composition.as_dict().items():
            if isinstance(element, str):
                base = element.rstrip("+-0123456789") if (
                    "+" in element or "-" in element
                ) else element
            else:
                base = getattr(element, "element", element)
            neutral[base] = neutral.get(base, 0) + count

        try:
            diagram = PhaseDiagram(entries)
            query = PDEntry(Composition(neutral), total_energy)
            return diagram.get_decomp_and_e_above_hull(query, allow_negative=True)[1]
        except Exception as exc:
            raise ValueError(
                f"Failed to compute energy above hull for {composition.formula}: {exc}"
            ) from exc
