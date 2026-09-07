"""Turn the formula table into tensors, keeping the two feature sets apart.

The split enforced here is the whole point of the model. **Chemistry** -- which
elements, in what proportion -- is what the location head sees, and it is the
only thing allowed to determine the estimate of ``f*(X)``. **Provenance** -- how
many entries each search process contributed -- is what the excess-scale head
sees, and it never reaches the location.

That is an exclusion restriction written as an architecture. Selection into this
archive is not ignorable: a formula was computed because somebody expected it to
be interesting, which is entangled with the quantity being estimated. A model
free to read effort into the floor will learn that singleton formulas have high
energy -- true in the archive, since 69% of formulas are one-shot substitution
products, and exactly wrong for a formula nobody has tried yet. Denying the
location head the effort channels is the strongest available guard against
learning the exploration paradox as if it were chemistry.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from math import comb
from typing import Dict, FrozenSet, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from wyckoff_transformer.csp import parse_formula
from wyckoff_transformer.formula_energy.dataset import KINDS

#: Atomic numbers run 1..118 and 0 is the padding slot, so the embedding needs
#: 119 rows and no tokeniser has to be built, loaded or kept in sync.
N_ELEMENT_SLOTS = 119
PADDING_ID = 0

#: Effort and provenance, in a fixed order. Counts arrive as ``log1p`` because
#: the difference between one entry and two says far more about how well searched
#: a formula is than the difference between forty and forty-one.
FORMULA_FEATURES: Tuple[str, ...] = (
    *(f"log1p_n_{kind}" for kind in KINDS),
    "log1p_n_rows",
    "log1p_n_cell_sizes",
    "log1p_n_icsd",
    "has_icsd",
    "max_force_min",
    "max_force_median",
)

#: How well explored the *neighbourhood* is, which the formula-level features
#: cannot say: two thirds of formulas have a single entry, so their own counts
#: are nearly constant, and a formula nobody has computed has none at all. These
#: are the only provenance channel that is populated when screening a novel
#: composition.
#:
#: Arity is controlled combinatorially. A system of ``a`` elements contains
#: exactly ``C(a, k)`` subsystems of size ``k``, so a raw count over all
#: subsystems grows with arity mechanically -- measured, its rank correlation
#: with arity is +0.72, and the exact-system count runs the other way at -0.74.
#: Dividing the entries drawn from size-``k`` subsystems by ``C(a, k)`` gives the
#: mean entries per ``k``-element subsystem, which is arity-free by construction:
#: the same correlations fall to -0.07 and +0.07.
SYSTEM_FEATURES: Tuple[str, ...] = (
    "log1p_sys_entries_per_binary",
    "log1p_sys_entries_per_ternary",
    "log1p_sys_hull_per_binary",
    "log1p_sys_hull_per_ternary",
)

PROVENANCE_FEATURES: Tuple[str, ...] = (*FORMULA_FEATURES, *SYSTEM_FEATURES)


def _atomic_numbers(symbols: Iterable[str]) -> List[int]:
    from pymatgen.core.periodic_table import Element  # noqa: PLC0415

    return [Element(symbol).Z for symbol in symbols]


def composition_tensors(
    formulas: Sequence[str],
    max_elements: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduced-formula strings into padded element ids, fractions and a mask.

    Fractions rather than counts, so the representation is invariant to how the
    formula was written: ``BaTiO3`` and ``Ba2Ti2O6`` give identical tensors, which
    they must, because they are the same compound and share a floor. Elements are
    emitted in a fixed order, and the encoder is permutation invariant anyway.

    Args:
        formulas: Canonical keys from :func:`~.dataset.formula_key`.
        max_elements: Pad width. Defaults to the widest formula given.

    Returns:
        ``(element_ids, fractions, padding_mask)``, each ``[N, max_elements]``;
        the mask is ``True`` where the slot is padding, matching the convention
        of :class:`torch.nn.TransformerEncoder`.
    """
    parsed = [parse_formula(formula) for formula in formulas]
    if not parsed:
        raise ValueError("No formulas given")
    width = max_elements or max(len(counts) for counts in parsed)

    element_ids = np.zeros((len(parsed), width), dtype=np.int64)
    fractions = np.zeros((len(parsed), width), dtype=np.float32)
    for row, counts in enumerate(parsed):
        if len(counts) > width:
            raise ValueError(f"{formulas[row]!r} has {len(counts)} elements, above the pad width {width}")
        symbols = sorted(counts)
        total = sum(counts.values())
        element_ids[row, :len(symbols)] = _atomic_numbers(symbols)
        fractions[row, :len(symbols)] = [counts[symbol] / total for symbol in symbols]

    ids = torch.from_numpy(element_ids)
    return ids, torch.from_numpy(fractions), ids == PADDING_ID


def provenance_tensor(
    table: pd.DataFrame,
    names: Sequence[str] = PROVENANCE_FEATURES,
) -> torch.Tensor:
    """The effort channels of the formula table, in ``names`` order.

    Args:
        table: Output of :func:`~.dataset.build_formula_table`, optionally with
            the columns :class:`SystemDensity` adds.
        names: Which features to emit. Defaults to all of them; a saved ensemble
            passes the list it was trained with, so an older checkpoint keeps
            loading after the set grows.

    Returns:
        ``[N, len(names)]`` float32.
    """
    columns = {
        **{f"log1p_n_{kind}": np.log1p(table[f"n_{kind}"].to_numpy(dtype=np.float64)) for kind in KINDS},
        "log1p_n_rows": np.log1p(table["n_rows"].to_numpy(dtype=np.float64)),
        "log1p_n_cell_sizes": np.log1p(table["n_cell_sizes"].to_numpy(dtype=np.float64)),
        "log1p_n_icsd": np.log1p(table["n_icsd"].to_numpy(dtype=np.float64)),
        "has_icsd": table["has_icsd"].to_numpy(dtype=np.float64),
        "max_force_min": table["max_force_min"].to_numpy(dtype=np.float64),
        "max_force_median": table["max_force_median"].to_numpy(dtype=np.float64),
    }
    for name in SYSTEM_FEATURES:
        if name in table.columns:
            columns[name] = table[name].to_numpy(dtype=np.float64)
    missing = [name for name in names if name not in columns]
    if missing:
        raise KeyError(f"Table lacks the columns for {missing}; run SystemDensity.attach first")
    stacked = np.stack([columns[name] for name in names], axis=1)
    return torch.from_numpy(np.nan_to_num(stacked)).float()


class SystemDensity:
    """How much the archive holds around a composition, per subsystem.

    Built once from the archive, then queried for any formula -- including one
    nobody has computed, which is the point: every other provenance channel is
    identically zero there.
    """

    #: Subsystem sizes the densities are reported for. Unary counts are constant
    #: at one entry per element and carry nothing; sizes above three are rare
    #: enough that the mean is dominated by empty subsystems.
    SIZES = (2, 3)

    def __init__(self, entries: Dict[FrozenSet[str], int], hull_entries: Dict[FrozenSet[str], int]) -> None:
        self.entries = entries
        self.hull_entries = hull_entries

    @classmethod
    def from_table(cls, table: pd.DataFrame, tolerance: float = 1e-6) -> "SystemDensity":
        """Count entries, and hull-defining entries, per chemical system."""
        on_hull = table["e_form_min"] <= table["e_hull_at_composition"] + tolerance
        def counts(frame):
            return {frozenset(str(key).split("-")): int(value)
                    for key, value in frame.groupby("chemsys").size().items()}
        return cls(counts(table), counts(table[on_hull]))

    @lru_cache(maxsize=200_000)
    def _for_system(self, elements: Tuple[str, ...]) -> Tuple[float, ...]:
        arity = len(elements)
        values = []
        for source in (self.entries, self.hull_entries):
            for size in self.SIZES:
                if arity < size:
                    values.append(0.0)
                    continue
                total = sum(source.get(frozenset(subset), 0)
                            for subset in combinations(elements, size))
                values.append(total / comb(arity, size))
        return tuple(values)

    def columns(self, formulas: Sequence[str]) -> pd.DataFrame:
        """``[N, 4]`` of ``log1p`` densities, indexed as ``formulas``."""
        rows = np.empty((len(formulas), len(SYSTEM_FEATURES)))
        for position, formula in enumerate(formulas):
            rows[position] = self._for_system(tuple(sorted(parse_formula(formula))))
        # Order matches SYSTEM_FEATURES: entries per binary, per ternary, then
        # the same two restricted to hull-defining entries.
        order = [0, 1, 2, 3]
        return pd.DataFrame(np.log1p(rows[:, order]), columns=list(SYSTEM_FEATURES),
                            index=pd.Index(formulas))

    def attach(self, table: pd.DataFrame) -> pd.DataFrame:
        """Add the density columns to a table indexed by reduced formula."""
        return table.join(self.columns(table.index.tolist()))
