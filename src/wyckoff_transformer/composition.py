"""Conditioning WyFormer on the target chemical formula.

`csp.ConstrainedDecoder` guarantees the composition by masking: it removes every
choice that would strand the target, so the formula comes out exact. What it
cannot do is tell the model *which* formula it is building. The distribution the
decoder samples from is the backbone's unconditional one, renormalised over the
feasible set, so among the legal continuations nothing prefers the ones this
composition actually adopts.

This module supplies the missing input. The composition becomes a fixed-width
vector over the element vocabulary, which joins whatever scalar the model is
already conditioned on and reaches every encoder layer through the existing
AdaLN path -- the same route `energy_above_hull` takes.

The representation
------------------

A composition is a map from element to atom count in the conventional cell. Two
things are worth separating in it:

- **what** the compound is made of, as fractions that sum to one. This is
  z-invariant: BaTiO3 and Ba2Ti2O6 give the same vector, which is right, because
  they are the same chemistry and a model that had to learn them as unrelated
  inputs would be wasting capacity.
- **how much** of it is in the cell, as ``log1p`` of the total atom count. Cell
  size is not chemistry, but it is not ignorable either -- it bounds how many
  Wyckoff positions the gene needs -- and on a log scale a 4-atom cell and a
  200-atom one are a few units apart rather than two orders of magnitude.

Together they are a bijection with the raw counts, so nothing is lost, and both
channels land in a range an `nn.Linear` into AdaLN can use without the largest
compositions dominating the modulation.

Widths follow the element tokeniser, service tokens included. Those columns stay
zero -- STOP is not an element -- which costs a few unused weights and keeps the
vector indexable by token id with no offset table to get wrong.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

#: Channels beyond the per-element fractions: currently just log1p(total atoms).
COMPOSITION_EXTRA_CHANNELS = 1

#: Field name the densified vector is stored under, and the name a config's
#: `condition_feature` should *not* use -- this one is built, not read from disk.
COMPOSITION_FIELD = "composition_vector"

#: The ragged per-example fields the tokeniser emits for a `counters: composition`
#: entry, which this module densifies.
COMPOSITION_SOURCE_FIELDS = ("composition_tokens", "composition_counts")


def composition_conditioning_dim(n_elements: int, size_channel: bool = True) -> int:
    """Width of the vector `composition_vector` produces for a given vocabulary."""
    return n_elements + (COMPOSITION_EXTRA_CHANNELS if size_channel else 0)


def composition_vector(
    element_tokens: Sequence[int] | Tensor,
    counts: Sequence[float] | Tensor,
    n_elements: int,
    device: Optional[torch.device] = None,
    size_channel: bool = True,
) -> Tensor:
    """One composition as element fractions, optionally then log1p(size).

    Args:
        element_tokens: Element token ids present in the composition.
        counts: Atoms of each, in the same order. Conventional-cell counts.
        n_elements: Size of the element vocabulary, service tokens included.
        device: Where to build the tensor.
        size_channel: Append log1p of the total atom count. Right for de novo
            generation, where asking for a cell of a given size is meaningful.
            Wrong for CSP, where the cell size is what the model is being asked
            to choose: `csp.ConstrainedDecoder` decides ``z`` by where the model
            emits STOP, which is after the conditioning has already been built,
            so a model trained with this channel has to be decoded one ``z`` at a
            time and cannot spread its own probability over the alternatives.

    Raises:
        ValueError: on a token outside the vocabulary, a non-positive count, or
            an empty composition -- each of which would otherwise produce a
            silently wrong conditioning vector.
    """
    tokens = torch.as_tensor(element_tokens, dtype=torch.int64, device=device).reshape(-1)
    amounts = torch.as_tensor(counts, dtype=torch.float32, device=device).reshape(-1)
    if tokens.numel() != amounts.numel():
        raise ValueError(
            f"{tokens.numel()} element tokens against {amounts.numel()} counts")
    if tokens.numel() == 0:
        raise ValueError("Empty composition")
    if bool(((tokens < 0) | (tokens >= n_elements)).any()):
        raise ValueError(
            f"Element token outside a vocabulary of {n_elements}: "
            f"{tokens[(tokens < 0) | (tokens >= n_elements)].tolist()}")
    if bool((amounts <= 0).any()):
        raise ValueError(f"Non-positive atom count in {amounts.tolist()}")

    vector = torch.zeros(composition_conditioning_dim(n_elements, size_channel),
                         dtype=torch.float32, device=device)
    total = amounts.sum()
    # index_add rather than assignment: a composition should never list an element
    # twice, but if a caller's parser does, summing is the sane reading.
    vector[:n_elements].index_add_(0, tokens, amounts / total)
    if size_channel:
        vector[n_elements] = torch.log1p(total)
    return vector


def composition_vectors(
    element_tokens: Sequence[Sequence[int] | Tensor],
    counts: Sequence[Sequence[float] | Tensor],
    n_elements: int,
    device: Optional[torch.device] = None,
    size_channel: bool = True,
) -> Tensor:
    """`composition_vector` over a whole split, into one dense block.

    The tokeniser stores compositions as one ragged tensor per structure, since
    different compounds have different numbers of elements. This is where they
    become the dense block the conditioning path indexes by row.
    """
    if len(element_tokens) != len(counts):
        raise ValueError(
            f"{len(element_tokens)} compositions against {len(counts)} count vectors")
    if not element_tokens:
        raise ValueError("No compositions given")
    rows = [composition_vector(tokens, amounts, n_elements, device=device,
                               size_channel=size_channel)
            for tokens, amounts in zip(element_tokens, counts)]
    return torch.stack(rows)


def attach_composition_vector(
    data: Dict,
    n_elements: int,
    device: Optional[torch.device] = None,
    field: str = COMPOSITION_FIELD,
    size_channel: bool = True,
) -> Dict:
    """Add the densified composition to a split's data dict, in place.

    Does nothing if the field is already present, so a dataset that was cached
    with it, or one already prepared by an earlier call, is left alone.

    Raises:
        KeyError: if the tokeniser did not emit the composition counters. The fix
            is a `counters: composition` entry under `sequence_fields` in the
            tokeniser config, and a re-cache.
    """
    if field in data:
        return data
    missing = [name for name in COMPOSITION_SOURCE_FIELDS if name not in data]
    if missing:
        raise KeyError(
            f"Composition conditioning needs {missing} in the tokenised data. Add "
            "'counters: {composition: elements}' under sequence_fields in the "
            "tokeniser config and re-cache the dataset.")
    data[field] = composition_vectors(
        data["composition_tokens"], data["composition_counts"], n_elements, device=device,
        size_channel=size_channel)
    return data


def composition_vector_for_formula(
    formula_counts: Dict[str, float],
    elements_tokeniser,
    device: Optional[torch.device] = None,
    size_channel: bool = True,
) -> Tensor:
    """The conditioning vector for a formula given as element symbols to counts.

    The inference-side counterpart of `attach_composition_vector`: what
    ``wyformer-csp`` builds from ``--formula`` so the model is conditioned on the
    same representation it was trained on.
    """
    from wyckoff_transformer.csp import _element_token  # noqa: PLC0415

    tokens = [_element_token(symbol, elements_tokeniser) for symbol in formula_counts]
    return composition_vector(
        tokens, list(formula_counts.values()), len(elements_tokeniser), device=device,
        size_channel=size_channel)


def describe(vector: Tensor, elements_tokeniser) -> List[str]:
    """Read a conditioning vector back as symbols and fractions, for debugging."""
    n_elements = len(elements_tokeniser)
    to_token = getattr(elements_tokeniser, "to_token", None)
    parts = []
    for index in torch.nonzero(vector[:n_elements]).reshape(-1).tolist():
        symbol = str(to_token[index]) if to_token is not None else f"#{index}"
        parts.append(f"{symbol}={vector[index]:.3f}")
    if vector.shape[-1] > n_elements:
        parts.append(f"log1p(atoms)={vector[n_elements]:.3f}")
    return parts
