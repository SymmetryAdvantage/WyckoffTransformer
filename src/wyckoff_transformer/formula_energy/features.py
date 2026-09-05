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

from typing import Iterable, List, Sequence, Tuple

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
PROVENANCE_FEATURES: Tuple[str, ...] = (
    *(f"log1p_n_{kind}" for kind in KINDS),
    "log1p_n_rows",
    "log1p_n_cell_sizes",
    "log1p_n_icsd",
    "has_icsd",
    "max_force_min",
    "max_force_median",
)


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


def provenance_tensor(table: pd.DataFrame) -> torch.Tensor:
    """The effort channels of the formula table, in :data:`PROVENANCE_FEATURES` order.

    Args:
        table: Output of :func:`~.dataset.build_formula_table`.

    Returns:
        ``[N, len(PROVENANCE_FEATURES)]`` float32.
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
    stacked = np.stack([columns[name] for name in PROVENANCE_FEATURES], axis=1)
    return torch.from_numpy(np.nan_to_num(stacked)).float()
