"""Conditioning WyFormer on the chemical system: which elements, not how many.

CSP mode fixes the formula. This is the relaxed variant of it: the caller names a
set of allowed elements -- Ba-Ti-O -- and leaves the stoichiometry, the cell size
and the space group to the model. It is the question a chemist asks more often
than the CSP one, because a chemical system is what a synthesis campaign is
scoped by, while a formula is already an answer.

Two things change with respect to `wyckoff_transformer.composition`, and they
change together:

- **The conditioning vector loses the counts.** A composition is a map from
  element to atom count, encoded as fractions plus `log1p` of the cell size. A
  chemical system is a *set*, so its encoding is the indicator of that set over
  the element vocabulary: one where the element is allowed, zero elsewhere.
  Nothing says how much of it there is, which is exactly the information the
  relaxed mode declines to supply.
- **The decoding constraint loses its arithmetic.** `csp.ConstrainedDecoder`
  exists because a composition is only reachable through a particular sum of
  Wyckoff multiplicities, so most prefixes strand it and have to be masked ahead
  of time. A *set* strands nothing: every element of it can be placed at any
  multiplicity, any number of times. So the constraint is a plain mask on the
  `elements` logits, which `WyckoffGenerator.generate_tensors` already applies
  through `allowed_element_set`, and every draw is accepted.

The L1 norm of the vector is the arity of the system -- 3 for Ba-Ti-O -- so the
model is told how many elements it is working with, without being told which
proportions. That is deliberate: a binary system and a quaternary one have very
different gene-length distributions, and an encoding that hid the difference
would make the model infer it from the space group alone.

What the label means at training time
-------------------------------------

Every training structure carries the set of elements it actually contains, so
the model learns ``p(gene | exactly these elements are present)``. It does *not*
learn ``p(gene | a subset of these elements)``: nothing in the training data
pairs a gene with a system larger than its own.

That is the honest reading of the label, and it has a consequence worth stating
plainly, because it is the one thing about this mode that can surprise. Asking
for Ba-Ti-O asks for a compound containing all three. If what you want is "any
oxide of barium and titanium, including BaO and TiO2", the conditioning vector
is the wrong instrument -- pass the wider set to the sampler's mask
(``--allowed-elements``) and condition on nothing, or run the model once per
subsystem. The generator's mask permits any subset of the requested set, so a
model that decides on a subset is not prevented from producing it; it is only
never encouraged to.

Widths follow the element tokeniser, service tokens included, for the same
reason `composition` does it: the vector stays indexable by token id with no
offset table to get wrong, at the cost of three columns that are always zero.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

#: Field name the indicator vector is stored under, and the name a config's
#: `condition_feature` should *not* use -- this one is built, not read from disk.
CHEMICAL_SYSTEM_FIELD = "chemical_system_vector"

#: The ragged per-example field the tokeniser emits for a `counters: composition`
#: entry. Only the keys are read here; the counts are what this mode drops.
CHEMICAL_SYSTEM_SOURCE_FIELD = "composition_tokens"

#: How ``Ba-Ti-O`` is spelled on a command line, and in `describe`.
SYSTEM_DELIMITER = "-"


def chemical_system_conditioning_dim(n_elements: int) -> int:
    """Width of the vector `chemical_system_vector` produces for a given vocabulary."""
    return n_elements


def chemical_system_vector(
    element_tokens: Sequence[int] | Tensor,
    n_elements: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    """One chemical system as an indicator over the element vocabulary.

    Args:
        element_tokens: Element token ids in the system. Repeats are idempotent --
            a set has no multiplicity -- which is what makes this safe to build
            from a composition's keys or from a per-site element list alike.
        n_elements: Size of the element vocabulary, service tokens included.
        device: Where to build the tensor.

    Raises:
        ValueError: on a token outside the vocabulary, or an empty system. A
            structure with no elements is not a structure, and an empty vector
            would be indistinguishable from an unconditional row.
    """
    tokens = torch.as_tensor(element_tokens, dtype=torch.int64, device=device).reshape(-1)
    if tokens.numel() == 0:
        raise ValueError("Empty chemical system")
    if bool(((tokens < 0) | (tokens >= n_elements)).any()):
        raise ValueError(
            f"Element token outside a vocabulary of {n_elements}: "
            f"{tokens[(tokens < 0) | (tokens >= n_elements)].tolist()}")

    vector = torch.zeros(chemical_system_conditioning_dim(n_elements),
                         dtype=torch.float32, device=device)
    # index_fill_, not index_add_: the counts are precisely what this mode drops, so
    # an element named twice must not come out as a two.
    vector.index_fill_(0, tokens, 1.0)
    return vector


def chemical_system_vectors(
    element_tokens: Sequence[Sequence[int] | Tensor],
    n_elements: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    """`chemical_system_vector` over a whole split, into one dense block.

    The tokeniser stores each structure's elements as one ragged tensor, since
    compounds differ in how many they have. This is where they become the dense
    block the conditioning path indexes by row.
    """
    if not len(element_tokens):
        raise ValueError("No chemical systems given")
    return torch.stack([chemical_system_vector(tokens, n_elements, device=device)
                        for tokens in element_tokens])


def attach_chemical_system_vector(
    data: Dict,
    n_elements: int,
    device: Optional[torch.device] = None,
    field: str = CHEMICAL_SYSTEM_FIELD,
) -> Dict:
    """Add the densified chemical system to a split's data dict, in place.

    Does nothing if the field is already present, so a dataset that was cached
    with it, or one already prepared by an earlier call, is left alone.

    Reads the composition counters the tokeniser already emits and keeps only
    their keys: the same cache serves both this mode and `composition`, so
    switching between them needs no re-cache.

    Raises:
        KeyError: if the tokeniser did not emit the composition counters.
    """
    if field in data:
        return data
    if CHEMICAL_SYSTEM_SOURCE_FIELD not in data:
        raise KeyError(
            f"Chemical system conditioning needs {CHEMICAL_SYSTEM_SOURCE_FIELD!r} in the "
            "tokenised data. Add 'counters: {composition: elements}' under "
            "sequence_fields in the tokeniser config and re-cache the dataset.")
    data[field] = chemical_system_vectors(
        data[CHEMICAL_SYSTEM_SOURCE_FIELD], n_elements, device=device)
    return data


def parse_chemical_system(
    system: str | Iterable[str],
    elements_tokeniser,
    delimiter: str = SYSTEM_DELIMITER,
) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    """``"Ba-Ti-O"`` into its symbols and their token ids, deduplicated.

    Returns the symbols in the order given -- so a log line reads back the way
    the caller wrote it -- and the tokens sorted, since a set has no order and a
    sorted tuple is the form the sampler's mask wants.

    Raises:
        ValueError: on an empty system.
        KeyError: on an element the model has never seen, which is worth failing
            on rather than silently conditioning on a system missing a component.
    """
    from wyckoff_transformer.csp import _element_token  # noqa: PLC0415

    if isinstance(system, str):
        raw = [part.strip() for part in system.split(delimiter)]
    else:
        raw = [str(part).strip() for part in system]
    symbols: List[str] = []
    for symbol in raw:
        if not symbol:
            continue
        if symbol not in symbols:
            symbols.append(symbol)
    if not symbols:
        raise ValueError(f"Empty chemical system: {system!r}")
    tokens = tuple(sorted(_element_token(symbol, elements_tokeniser) for symbol in symbols))
    return tuple(symbols), tokens


def chemical_system_vector_for_elements(
    system: str | Iterable[str],
    elements_tokeniser,
    device: Optional[torch.device] = None,
    delimiter: str = SYSTEM_DELIMITER,
) -> Tensor:
    """The conditioning vector for a system given as element symbols.

    The inference-side counterpart of `attach_chemical_system_vector`: what
    ``wyformer-generate --chemical-system`` builds, so training and sampling see
    one representation.
    """
    _, tokens = parse_chemical_system(system, elements_tokeniser, delimiter=delimiter)
    return chemical_system_vector(tokens, len(elements_tokeniser), device=device)


def describe(vector: Tensor, elements_tokeniser, delimiter: str = SYSTEM_DELIMITER) -> str:
    """Read a conditioning vector back as ``Ba-Ti-O``, for logs and debugging."""
    n_elements = len(elements_tokeniser)
    to_token = getattr(elements_tokeniser, "to_token", None)
    parts = []
    for index in torch.nonzero(vector[:n_elements]).reshape(-1).tolist():
        parts.append(str(to_token[index]) if to_token is not None else f"#{index}")
    return delimiter.join(parts)
