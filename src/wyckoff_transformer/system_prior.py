"""Sampling (chemical system, space group) pairs inside an element palette.

`chemical_system` conditions the model on *one* system the caller names. This is
what stands in front of it when the caller does not name one: a user supplies a
small set of **required** elements and a large set of **allowed** ones -- 1-3
against 20-40 is the expected shape -- and every generated structure needs a
system and a space group of its own, drawn from something that knows which
combinations are real.

Why not keep filtering at sampling time
---------------------------------------

The mode this replaces samples unconditionally and throws away what does not
match. That is biased twice over, and both biases are measurable on
`lemat_bulk_fmax1`'s training split.

- **The space group is drawn from the wrong distribution.** It is this
  architecture's start token, so it is fixed *before* any element is chosen and
  cannot react to the palette. Over a 36-element palette the space groups of the
  structures that survive the filter carry KL 0.10 nats against the unconditional
  distribution, rising to 0.21 when Li is required and 0.48 for Li-S. Per system
  -- which is what a model conditioned on the system actually sees -- the
  row-weighted mean KL is 0.92 nats over the 55334 systems with at least 20
  rows. Li-Mn-Co-O crystallises in P1, P-1, C2 and C2/m; the unconditional mode
  starts most of its batch in 123, 216 and 12, and no amount of element masking
  moves a start token.
- **The element mask does not respect likelihoods.** Masking `elements` logits
  makes every allowed element reachable, but nothing in the batch says which
  *combinations* of them occur: the mask is flat over the palette while real
  chemistry is not, and the sampler cannot prefer Li-Mn-O over Li-Kr-O except
  through whatever the unconditioned model already believes. Forcing the
  required elements in by argmax, which is what the current decoder does when
  `required_element_set` is non-empty, is a third distortion on top.

Rejection is also expensive where it is not wrong. The share of training rows
that satisfy a query is 11.4% for a bare 36-element palette, 1.4% once Li is
required and 0.07% for Li-S: 9x to 1400x oversampling to fill a batch.

What this module does instead
-----------------------------

It reads `p(system, space group)` off the training split as counts, restricts it
to the feasible region ``required <= S <= allowed``, and samples pairs from the
restricted, renormalised distribution. Each drawn pair is then a complete
conditioning input for one structure: the system goes in through
`chemical_system.chemical_system_vector`, the space group is the start token, and
the element mask is the system itself rather than the whole palette.

Two things have to be smoothed, and both were fitted against the held-out split
rather than chosen (see docs/chemical_system_sampler.md for the sweeps).

- **`p(G | S)` is shrunk toward a back-off.** The median system has 54 training
  rows spread over 228 space groups, so raw counts are nearly all zeros: the
  unsmoothed table scores -150.9 nats/row on held-out data against -3.26 for the
  shrunk one. The back-off is a product of experts -- the geometric mean of the
  per-element space group distributions, times the arity's, divided by the global
  one -- which reaches -2.00 nats/row on held-out rows whose system was never
  seen, against -2.23 for arity alone, -2.41 for the elements alone and -2.69 for
  the global distribution. `kappa` = 100 pseudo-counts is the optimum and the
  curve is flat from 50 to 150.
- **Systems that were never observed still have to be reachable.** 5.5% of
  held-out rows are in a system absent from training, so an empirical-only
  sampler would be certain of something false. `novel_fraction` mixes in a
  proposal that draws an arity and then elements without replacement, both
  weighted by *distinct system* counts rather than row counts: -14.55 nats
  against -15.98 for the naive uniform-over-palette version on held-out novel
  systems. It defaults to the novelty rate measured on the held-out split at
  build time, so the sampler is as adventurous as the data says the world is.

What it does not do
-------------------

Nothing here estimates `p(gene | system, space group)` -- that is the model's
job, and this is only its input distribution. And a drawn system is a *request*:
the element mask permits any subset of it, so the model can answer a Li-Mn-Co-O
request with a Li-Mn-O structure. `SystemDraws` records what was asked for; what
was produced has to be read off the generated structures.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

#: How ``Ba-Ti-O`` is spelled on a command line. Shared with `chemical_system`.
SYSTEM_DELIMITER = "-"

#: Pseudo-counts of back-off mixed into every `p(G | S)`. Fitted on the held-out
#: split of `lemat_bulk_fmax1`: -3.2646 nats/row at 100, -3.2648 at 50, -3.2704 at
#: 150, so the optimum is a plateau and the exact value is not load-bearing.
DEFAULT_SG_KAPPA = 100.0

#: Pseudo-counts of the global space group distribution mixed into the per-element
#: and per-arity back-off tables. The held-out likelihood is flat in this from 1 to
#: 1000 (-1.9965 to -1.9993); it is here to keep the geometric mean finite for an
#: element that has never been seen in a given space group, not to fit anything.
DEFAULT_BACKOFF_BETA = 10.0

#: Element tokens that are not elements. Excluded from every system and palette.
SERVICE_TOKENS = ("MASK", "STOP", "PAD")

_LANE_BITS = 64
_ARTIFACT_VERSION = 1


def _lanes(n_elements: int) -> int:
    return (n_elements + _LANE_BITS - 1) // _LANE_BITS


def _mask_from_tokens(tokens: Iterable[int], n_elements: int) -> np.ndarray:
    """One element set as a bitmask over the element vocabulary.

    Subset tests are the inner loop of every query -- 662k systems against one
    palette -- so systems are stored as bits and tested with two vectorised
    ands rather than as Python sets.
    """
    mask = np.zeros(_lanes(n_elements), dtype=np.uint64)
    for token in tokens:
        token = int(token)
        if not 0 <= token < n_elements:
            raise ValueError(f"Element token {token} outside a vocabulary of {n_elements}")
        mask[token // _LANE_BITS] |= np.uint64(1) << np.uint64(token % _LANE_BITS)
    return mask


def _normalise(weights: np.ndarray) -> np.ndarray:
    total = weights.sum()
    if not total > 0:
        raise ValueError("Cannot normalise a distribution with no mass")
    return weights / total


def _apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """``p ** (1 / T)``, renormalised. T = 1 is the distribution itself, larger is flatter."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    if temperature == 1.0:
        return probabilities
    return _normalise(np.power(probabilities, 1.0 / temperature))


def _resolve_rng(rng: Union[None, int, np.random.Generator]) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


@dataclass(frozen=True)
class SystemSpaceGroupCell:
    """One (system, space group) request and how many structures asked for it."""
    symbols: Tuple[str, ...]
    element_tokens: Tuple[int, ...]
    space_group: int
    count: int
    novel: bool

    @property
    def system(self) -> str:
        return SYSTEM_DELIMITER.join(self.symbols)


@dataclass
class SystemDraws:
    """`n` sampled requests, one per structure to generate.

    Rows are in sampling order and deliberately shuffled across the empirical and
    novel components, so truncating the batch does not truncate one of them.
    """
    element_tokens: Tuple[Tuple[int, ...], ...]
    space_groups: np.ndarray
    is_novel: np.ndarray
    element_symbols: Tuple[str, ...]
    query: Dict

    def __len__(self) -> int:
        return len(self.element_tokens)

    def symbols(self, row: int) -> Tuple[str, ...]:
        return tuple(self.element_symbols[token] for token in self.element_tokens[row])

    def systems(self) -> List[str]:
        """``["Li-Mn-O", ...]``, one per row, for logs and provenance."""
        return [SYSTEM_DELIMITER.join(self.symbols(row)) for row in range(len(self))]

    def grouped(self) -> List[SystemSpaceGroupCell]:
        """The draws as unique (system, space group) cells with multiplicities.

        Not how a plan is executed -- `generate_tensors` takes the whole batch at
        once through `element_mask` -- but how it is read: 1000 structures over 329
        systems is a table, and a list of 1000 rows is not. `manifest` is built from
        this, and a caller that wants one generator call per system has it here.
        Sorted by descending count, then by system, so the plan reads the same way
        twice.
        """
        cells: Dict[Tuple[Tuple[int, ...], int], List[int]] = {}
        for row in range(len(self)):
            key = (self.element_tokens[row], int(self.space_groups[row]))
            entry = cells.setdefault(key, [0, 0])
            entry[0] += 1
            entry[1] += int(self.is_novel[row])
        out = [
            SystemSpaceGroupCell(
                symbols=tuple(self.element_symbols[token] for token in tokens),
                element_tokens=tokens,
                space_group=space_group,
                count=count,
                novel=bool(novel),
            )
            for (tokens, space_group), (count, novel) in cells.items()
        ]
        out.sort(key=lambda cell: (-cell.count, cell.symbols, cell.space_group))
        return out

    def conditioning_block(self, n_elements: int, device=None):
        """The chemical-system conditioning rows, ``(n, n_elements)``.

        Built through `chemical_system.chemical_system_vector`, not reimplemented:
        the sampler must hand the model the same representation training used.
        """
        from wyckoff_transformer.chemical_system import chemical_system_vector  # noqa: PLC0415
        import torch  # noqa: PLC0415

        return torch.stack([
            chemical_system_vector(tokens, n_elements, device=device)
            for tokens in self.element_tokens
        ])

    def element_mask(self, n_elements: int, stop_token: Optional[int] = None, device=None):
        """Per-row mask over the element vocabulary, ``(n, n_elements)`` of bool.

        One row's mask is its own system, not the whole palette: the palette says
        what the campaign may contain, the drawn system says what *this* structure
        was asked for, and letting the rest of the palette back in at decoding time
        would undo the sampling this module exists to do.

        `stop_token` is admitted when given, since a sequence that cannot stop
        cannot terminate. It is looked up from the stored vocabulary when omitted.
        """
        import torch  # noqa: PLC0415

        if stop_token is None and "STOP" in self.element_symbols:
            stop_token = self.element_symbols.index("STOP")
        mask = torch.zeros((len(self), n_elements), dtype=torch.bool, device=device)
        for row, tokens in enumerate(self.element_tokens):
            mask[row, list(tokens)] = True
            if stop_token is not None:
                mask[row, stop_token] = True
        return mask

    def start_tensor(self, space_group_tokeniser, start_type: str, device=None, dtype=None):
        """The per-row start tokens for the drawn space groups.

        `start_type` follows `CascadeTransformer`: ``one_hot`` for the
        `SpaceGroupEncoder`'s symmetry-matrix encoding, ``categorial`` for a plain
        token id.
        """
        import torch  # noqa: PLC0415

        space_groups = [int(value) for value in self.space_groups]
        unknown = sorted({sg for sg in space_groups if sg not in space_group_tokeniser})
        if unknown:
            raise ValueError(
                f"Space groups {unknown} are not in the model's vocabulary; it was trained "
                f"on {len(space_group_tokeniser)} of them. The prior and the model must be "
                "built from the same dataset.")
        if start_type == "one_hot":
            return space_group_tokeniser.encode_spacegroups(
                space_groups, dtype=torch.float32 if dtype is None else dtype, device=device)
        if start_type == "categorial":
            return torch.tensor([space_group_tokeniser[sg] for sg in space_groups],
                                dtype=torch.int64 if dtype is None else dtype, device=device)
        raise ValueError(f"Unsupported start type {start_type!r}")

    def manifest(self) -> Dict:
        """The plan as JSON: the query, the vocabulary, and the cells with multiplicities.

        The element symbols travel with the plan because the cells name element
        *tokens*, which mean nothing without the vocabulary that assigned them. A
        consumer can then check the plan against the model it is about to feed
        rather than silently generating for the wrong elements.
        """
        return {
            "query": self.query,
            "element_symbols": list(self.element_symbols),
            "n_structures": len(self),
            "cells": [
                {
                    "chemical_system": cell.system,
                    "element_tokens": list(cell.element_tokens),
                    "space_group": cell.space_group,
                    "n_structures": cell.count,
                    "novel_system": cell.novel,
                }
                for cell in self.grouped()
            ],
        }

    @classmethod
    def from_manifest(
        cls, manifest: Dict, element_symbols: Optional[Sequence[str]] = None
    ) -> "SystemDraws":
        """Rebuild draws from `manifest`, expanding the cells back into rows.

        `element_symbols` defaults to the vocabulary the manifest carries, and is
        checked against it when given: a plan read against the wrong vocabulary would
        decode every system into different elements without failing.
        """
        stored = manifest.get("element_symbols")
        if element_symbols is None:
            if stored is None:
                raise ValueError(
                    "This plan carries no element vocabulary; pass element_symbols "
                    "from the model whose tokeniser wrote it.")
            element_symbols = stored
        elif stored is not None and list(stored) != [str(s) for s in element_symbols]:
            raise ValueError(
                "The plan was drawn against a different element vocabulary than the one "
                "given: it names "
                f"{len(stored)} tokens against {len(list(element_symbols))}. The prior and "
                "the model have to come from the same dataset.")
        tokens: List[Tuple[int, ...]] = []
        space_groups: List[int] = []
        novel: List[bool] = []
        for cell in manifest["cells"]:
            for _ in range(int(cell["n_structures"])):
                tokens.append(tuple(cell["element_tokens"]))
                space_groups.append(int(cell["space_group"]))
                novel.append(bool(cell["novel_system"]))
        return cls(
            element_tokens=tuple(tokens),
            space_groups=np.asarray(space_groups, dtype=np.int64),
            is_novel=np.asarray(novel, dtype=bool),
            element_symbols=tuple(element_symbols),
            query=manifest.get("query", {}),
        )

    def summary(self, top: int = 10) -> str:
        """A few lines a human can check the plan against."""
        cells = self.grouped()
        systems: Dict[Tuple[str, ...], int] = {}
        for cell in cells:
            systems[cell.symbols] = systems.get(cell.symbols, 0) + cell.count
        lines = [
            f"{len(self)} structures over {len(systems)} systems and {len(cells)} "
            f"(system, space group) cells; {int(self.is_novel.sum())} rows in systems "
            "absent from the training data",
        ]
        for symbols, count in sorted(systems.items(), key=lambda item: (-item[1], item[0]))[:top]:
            lines.append(f"  {SYSTEM_DELIMITER.join(symbols):24s} {count:6d}")
        return "\n".join(lines)


@dataclass(frozen=True)
class FeasibleSystems:
    """The observed systems a query admits, and the weight each carries."""
    indices: np.ndarray
    weights: np.ndarray
    counts: np.ndarray
    total_rows: float
    corpus_rows: float

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def acceptance_rate(self) -> float:
        """Share of the training corpus a rejection sampler would have kept."""
        return float(self.total_rows / self.corpus_rows) if self.corpus_rows else 0.0


class SystemSpaceGroupPrior:
    """`p(chemical system, space group)` as counts, queryable under a palette.

    Built from a tokenised dataset by `from_tensor_cache`, or from rows in memory
    by `from_rows`. The tables are element-token indexed and space-group-number
    keyed, so a prior is tied to the vocabulary of the dataset it was built from
    and `start_tensor` refuses a model that disagrees.
    """

    def __init__(
        self,
        element_symbols: Sequence[str],
        space_groups: np.ndarray,
        system_token_data: np.ndarray,
        system_token_offsets: np.ndarray,
        system_masks: np.ndarray,
        system_counts: np.ndarray,
        cell_offsets: np.ndarray,
        cell_space_group_index: np.ndarray,
        cell_counts: np.ndarray,
        element_sg_counts: np.ndarray,
        arity_sg_counts: np.ndarray,
        global_sg_counts: np.ndarray,
        system_arity_counts: np.ndarray,
        element_system_counts: np.ndarray,
        metadata: Optional[Dict] = None,
    ):
        self.element_symbols = tuple(str(symbol) for symbol in element_symbols)
        self.space_groups = np.asarray(space_groups, dtype=np.int64)
        self.system_token_data = np.asarray(system_token_data, dtype=np.int32)
        self.system_token_offsets = np.asarray(system_token_offsets, dtype=np.int64)
        self.system_masks = np.asarray(system_masks, dtype=np.uint64)
        self.system_counts = np.asarray(system_counts, dtype=np.float64)
        self.cell_offsets = np.asarray(cell_offsets, dtype=np.int64)
        self.cell_space_group_index = np.asarray(cell_space_group_index, dtype=np.int32)
        self.cell_counts = np.asarray(cell_counts, dtype=np.float64)
        self.element_sg_counts = np.asarray(element_sg_counts, dtype=np.float64)
        self.arity_sg_counts = np.asarray(arity_sg_counts, dtype=np.float64)
        self.global_sg_counts = np.asarray(global_sg_counts, dtype=np.float64)
        self.system_arity_counts = np.asarray(system_arity_counts, dtype=np.float64)
        self.element_system_counts = np.asarray(element_system_counts, dtype=np.float64)
        self.metadata = dict(metadata or {})

        self.n_elements = len(self.element_symbols)
        self.n_space_groups = len(self.space_groups)
        self._symbol_to_token = {symbol: index for index, symbol in enumerate(self.element_symbols)}
        self._sg_to_index = {int(sg): index for index, sg in enumerate(self.space_groups)}
        self._system_lookup: Optional[Dict[Tuple[int, ...], int]] = None
        self._service_tokens = frozenset(
            self._symbol_to_token[name] for name in SERVICE_TOKENS if name in self._symbol_to_token)

        global_p = _normalise(self.global_sg_counts)
        beta = float(self.metadata.get("backoff_beta", DEFAULT_BACKOFF_BETA))
        self._log_global = np.log(global_p)
        self._log_element = np.log(
            (self.element_sg_counts + beta * global_p)
            / (self.element_sg_counts.sum(axis=1, keepdims=True) + beta))
        self._log_arity = np.log(
            (self.arity_sg_counts + beta * global_p)
            / (self.arity_sg_counts.sum(axis=1, keepdims=True) + beta))

    # ------------------------------------------------------------------ building

    @classmethod
    def from_rows(
        cls,
        systems: Sequence[Sequence[int]],
        space_groups: Sequence[int],
        element_symbols: Sequence[str],
        weights: Optional[Sequence[float]] = None,
        held_out_systems: Optional[Sequence[Sequence[int]]] = None,
        metadata: Optional[Dict] = None,
    ) -> "SystemSpaceGroupPrior":
        """Count `(system, space group)` over rows already in memory.

        Args:
            systems: One iterable of element tokens per structure. Repeats are
                idempotent: a system is a set.
            space_groups: The real space group number of each structure.
            element_symbols: The element vocabulary, indexed by token, service
                tokens included, exactly as the model's tokeniser has it.
            weights: Per-row weight, defaulting to one. A row filter that keeps
                only the stable polymorphs is expressed by dropping rows, not by
                weighting them; this is here for importance weights.
            held_out_systems: Systems of a split not counted into the tables. Used
                only to measure how often a system is new, which becomes the
                default `novel_fraction`.
        """
        if len(systems) != len(space_groups):
            raise ValueError(
                f"{len(systems)} systems against {len(space_groups)} space groups")
        if not len(systems):
            raise ValueError("No rows to build a prior from")
        n_elements = len(element_symbols)
        row_weights = (np.ones(len(systems), dtype=np.float64) if weights is None
                       else np.asarray(weights, dtype=np.float64))

        sg_values = np.asarray([int(sg) for sg in space_groups], dtype=np.int64)
        unique_sgs = np.unique(sg_values)
        sg_to_index = {int(sg): index for index, sg in enumerate(unique_sgs)}
        n_sg = len(unique_sgs)

        keys: List[Tuple[int, ...]] = []
        index_of: Dict[Tuple[int, ...], int] = {}
        row_system = np.empty(len(systems), dtype=np.int64)
        for row, tokens in enumerate(systems):
            key = tuple(sorted({int(token) for token in tokens}))
            if not key:
                raise ValueError(f"Row {row} has an empty chemical system")
            if key[0] < 0 or key[-1] >= n_elements:
                raise ValueError(f"Row {row} has an element token outside the vocabulary")
            index = index_of.get(key)
            if index is None:
                index = len(keys)
                index_of[key] = index
                keys.append(key)
            row_system[row] = index

        n_systems = len(keys)
        max_arity = max(len(key) for key in keys)
        system_counts = np.zeros(n_systems, dtype=np.float64)
        np.add.at(system_counts, row_system, row_weights)

        cells: Dict[Tuple[int, int], float] = {}
        for row in range(len(systems)):
            key = (int(row_system[row]), sg_to_index[int(sg_values[row])])
            cells[key] = cells.get(key, 0.0) + float(row_weights[row])
        ordered = sorted(cells.items())
        cell_system = np.asarray([key[0] for key, _ in ordered], dtype=np.int64)
        cell_sg = np.asarray([key[1] for key, _ in ordered], dtype=np.int32)
        cell_counts = np.asarray([value for _, value in ordered], dtype=np.float64)
        cell_offsets = np.zeros(n_systems + 1, dtype=np.int64)
        np.add.at(cell_offsets, cell_system + 1, 1)
        cell_offsets = np.cumsum(cell_offsets)

        element_sg_counts = np.zeros((n_elements, n_sg), dtype=np.float64)
        arity_sg_counts = np.zeros((max_arity + 1, n_sg), dtype=np.float64)
        global_sg_counts = np.zeros(n_sg, dtype=np.float64)
        system_arity_counts = np.zeros(max_arity + 1, dtype=np.float64)
        element_system_counts = np.zeros(n_elements, dtype=np.float64)
        for index, key in enumerate(keys):
            system_arity_counts[len(key)] += 1.0
            for token in key:
                element_system_counts[token] += 1.0
            start, stop = cell_offsets[index], cell_offsets[index + 1]
            sgs = cell_sg[start:stop]
            counts = cell_counts[start:stop]
            global_sg_counts[sgs] += counts
            arity_sg_counts[len(key), sgs] += counts
            for token in key:
                element_sg_counts[token, sgs] += counts

        system_masks = np.stack([_mask_from_tokens(key, n_elements) for key in keys])
        token_offsets = np.zeros(n_systems + 1, dtype=np.int64)
        for index, key in enumerate(keys):
            token_offsets[index + 1] = token_offsets[index] + len(key)
        token_data = np.concatenate([np.asarray(key, dtype=np.int32) for key in keys])

        meta = dict(metadata or {})
        meta.setdefault("backoff_beta", DEFAULT_BACKOFF_BETA)
        meta["n_rows"] = float(row_weights.sum())
        meta["n_systems"] = int(n_systems)
        if held_out_systems is not None and len(held_out_systems):
            unseen = sum(
                1 for tokens in held_out_systems
                if tuple(sorted({int(token) for token in tokens})) not in index_of)
            meta["held_out_novelty_rate"] = unseen / len(held_out_systems)
            meta["held_out_rows"] = int(len(held_out_systems))

        return cls(
            element_symbols=element_symbols,
            space_groups=unique_sgs,
            system_token_data=token_data,
            system_token_offsets=token_offsets,
            system_masks=system_masks,
            system_counts=system_counts,
            cell_offsets=cell_offsets,
            cell_space_group_index=cell_sg,
            cell_counts=cell_counts,
            element_sg_counts=element_sg_counts,
            arity_sg_counts=arity_sg_counts,
            global_sg_counts=global_sg_counts,
            system_arity_counts=system_arity_counts,
            element_system_counts=element_system_counts,
            metadata=meta,
        )

    # ----------------------------------------------------------------- artifacts

    def save(self, path: Union[str, Path]) -> Path:
        """Write the tables to a single ``.npz``."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            version=np.asarray(_ARTIFACT_VERSION),
            element_symbols=np.asarray(self.element_symbols, dtype=object),
            space_groups=self.space_groups,
            system_token_data=self.system_token_data,
            system_token_offsets=self.system_token_offsets,
            system_masks=self.system_masks,
            system_counts=self.system_counts,
            cell_offsets=self.cell_offsets,
            cell_space_group_index=self.cell_space_group_index,
            cell_counts=self.cell_counts,
            element_sg_counts=self.element_sg_counts,
            arity_sg_counts=self.arity_sg_counts,
            global_sg_counts=self.global_sg_counts,
            system_arity_counts=self.system_arity_counts,
            element_system_counts=self.element_system_counts,
            metadata=np.asarray(json.dumps(self.metadata)),
        )
        return destination

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SystemSpaceGroupPrior":
        with np.load(Path(path), allow_pickle=True) as stored:
            version = int(stored["version"])
            if version != _ARTIFACT_VERSION:
                raise ValueError(
                    f"System prior at {path} is version {version}, this build reads "
                    f"{_ARTIFACT_VERSION}. Rebuild it with `wyformer-system-prior build`.")
            return cls(
                element_symbols=[str(symbol) for symbol in stored["element_symbols"]],
                space_groups=stored["space_groups"],
                system_token_data=stored["system_token_data"],
                system_token_offsets=stored["system_token_offsets"],
                system_masks=stored["system_masks"],
                system_counts=stored["system_counts"],
                cell_offsets=stored["cell_offsets"],
                cell_space_group_index=stored["cell_space_group_index"],
                cell_counts=stored["cell_counts"],
                element_sg_counts=stored["element_sg_counts"],
                arity_sg_counts=stored["arity_sg_counts"],
                global_sg_counts=stored["global_sg_counts"],
                system_arity_counts=stored["system_arity_counts"],
                element_system_counts=stored["element_system_counts"],
                metadata=json.loads(str(stored["metadata"])),
            )

    # ------------------------------------------------------------------- lookups

    @property
    def element_tokens(self) -> Tuple[int, ...]:
        """Every token that names a real element, service tokens excluded."""
        return tuple(token for token in range(self.n_elements)
                     if token not in self._service_tokens)

    def token(self, symbol: str) -> int:
        """``"Li"`` to its element token, as the model's tokeniser has it."""
        try:
            return self._symbol_to_token[symbol]
        except KeyError:
            raise KeyError(
                f"Element {symbol!r} is not in the vocabulary this prior was built "
                f"from ({self.n_elements} tokens)") from None

    def parse_elements(self, elements: Union[None, str, Iterable[str]]) -> Tuple[int, ...]:
        """``"Li-S"`` or ``["Li", "S"]`` to sorted, deduplicated element tokens.

        `None` and the empty string are the empty set, which is what an unrestricted
        `required` means; `allowed` treats it separately.
        """
        if elements is None:
            return ()
        if isinstance(elements, str):
            parts = [part.strip() for part in elements.split(SYSTEM_DELIMITER)]
        else:
            parts = [str(part).strip() for part in elements]
        return tuple(sorted({self.token(part) for part in parts if part}))

    def system_tokens(self, index: int) -> Tuple[int, ...]:
        start, stop = self.system_token_offsets[index], self.system_token_offsets[index + 1]
        return tuple(int(token) for token in self.system_token_data[start:stop])

    def system_symbols(self, index: int) -> Tuple[str, ...]:
        return tuple(self.element_symbols[token] for token in self.system_tokens(index))

    def index_of(self, tokens: Iterable[int]) -> Optional[int]:
        """Where a system sits in the tables, or `None` if it was never observed."""
        if self._system_lookup is None:
            self._system_lookup = {
                self.system_tokens(index): index
                for index in range(len(self.system_counts))
            }
        return self._system_lookup.get(tuple(sorted({int(token) for token in tokens})))

    # ------------------------------------------------------------------- queries

    def feasible_systems(
        self,
        required: Union[None, str, Iterable[str], Iterable[int]] = None,
        allowed: Union[None, str, Iterable[str], Iterable[int]] = None,
        min_arity: Optional[int] = None,
        max_arity: Optional[int] = None,
        system_temperature: float = 1.0,
    ) -> FeasibleSystems:
        """Observed systems with ``required <= S <= allowed``, and their weights.

        `allowed` of `None` means the whole vocabulary, which is the "required
        elements only" query. Arity bounds narrow it further: with 30 allowed
        elements the observed feasible set is dominated by quaternaries, and a
        campaign that wants ternaries should say so here rather than discard them
        after generation.
        """
        required_tokens, allowed_tokens = self._resolve_query(required, allowed)
        required_mask = _mask_from_tokens(required_tokens, self.n_elements)
        allowed_mask = _mask_from_tokens(allowed_tokens, self.n_elements)

        outside = (self.system_masks & ~allowed_mask)
        keep = ~outside.any(axis=1)
        if required_tokens:
            has_required = ((self.system_masks & required_mask) == required_mask).all(axis=1)
            keep &= has_required
        if min_arity is not None or max_arity is not None:
            arity = np.diff(self.system_token_offsets)
            if min_arity is not None:
                keep &= arity >= min_arity
            if max_arity is not None:
                keep &= arity <= max_arity

        indices = np.nonzero(keep)[0]
        counts = self.system_counts[indices]
        total = float(counts.sum())
        weights = (_apply_temperature(_normalise(counts), system_temperature)
                   if len(indices) else np.zeros(0, dtype=np.float64))
        return FeasibleSystems(
            indices=indices,
            weights=weights,
            counts=counts,
            total_rows=total,
            corpus_rows=float(self.system_counts.sum()),
        )

    def space_group_probabilities(
        self,
        tokens: Iterable[int],
        kappa: float = DEFAULT_SG_KAPPA,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """`p(G | S)` over `space_groups`, shrunk toward the back-off.

        An unobserved system has no counts, so the same expression returns the
        pure back-off for it -- the novel component needs no separate code path.
        """
        if kappa < 0:
            raise ValueError(f"kappa must be non-negative, got {kappa}")
        tokens = tuple(sorted({int(token) for token in tokens}))
        back_off = self.back_off_probabilities(tokens)
        counts = np.zeros(self.n_space_groups, dtype=np.float64)
        index = self.index_of(tokens)
        if index is not None:
            start, stop = self.cell_offsets[index], self.cell_offsets[index + 1]
            counts[self.cell_space_group_index[start:stop]] = self.cell_counts[start:stop]
        total = counts.sum()
        if total + kappa <= 0:
            raise ValueError(
                "kappa = 0 leaves an unobserved system with no space group distribution")
        return _apply_temperature((counts + kappa * back_off) / (total + kappa), temperature)

    def back_off_probabilities(self, tokens: Iterable[int]) -> np.ndarray:
        """`p(G | S)` for a system the counts cannot speak about.

        A product of experts against the global distribution as the reference
        measure: the geometric mean over the system's elements of `p(G | element)`,
        times `p(G | arity)`, divided by `p(G)`. Held-out likelihood on rows whose
        system is unseen: -2.00 nats, against -2.23 for the arity expert alone,
        -2.41 for the element one and -2.69 for the global distribution.
        """
        tokens = tuple(sorted({int(token) for token in tokens}))
        if not tokens:
            raise ValueError("An empty chemical system has no space group distribution")
        arity = min(len(tokens), len(self._log_arity) - 1)
        logits = (self._log_element[list(tokens)].mean(axis=0)
                  + self._log_arity[arity] - self._log_global)
        logits -= logits.max()
        return _normalise(np.exp(logits))

    # ------------------------------------------------------------------ sampling

    def sample(
        self,
        n_structures: int,
        required: Union[None, str, Iterable[str], Iterable[int]] = None,
        allowed: Union[None, str, Iterable[str], Iterable[int]] = None,
        *,
        novel_fraction: Optional[float] = None,
        system_temperature: float = 1.0,
        sg_kappa: float = DEFAULT_SG_KAPPA,
        sg_temperature: float = 1.0,
        min_arity: Optional[int] = None,
        max_arity: Optional[int] = None,
        rng: Union[None, int, np.random.Generator] = None,
    ) -> SystemDraws:
        """Draw `n_structures` (system, space group) requests inside the palette.

        Args:
            n_structures: How many structures the batch will hold. One request per
                structure: the point of the mode is that they differ.
            required: Elements every structure must be asked to contain.
            allowed: The palette. `None` is the whole vocabulary.
            novel_fraction: Share of rows drawn from the novel-system proposal
                rather than from the counts. `None` takes the novelty rate measured
                on the held-out split when the prior was built, so the sampler is
                as adventurous as the data says the world is; raise it deliberately
                to explore, set it to zero to stay strictly on observed systems.
            system_temperature: Flattens `p(S)`. 1 is the empirical distribution,
                larger tends to uniform over feasible systems -- which is a claim
                that the corpus's attention is not evidence, defensible for a
                database assembled by campaigns rather than by sampling nature.
            sg_kappa: Back-off pseudo-counts in `p(G | S)`.
            sg_temperature: Flattens `p(G | S)` the same way.
            min_arity, max_arity: Bounds on how many elements a drawn system has.
            rng: Seed or generator. Seed it to make a campaign reproducible.

        Raises:
            ValueError: if the query admits nothing -- no observed system inside
                the palette and no novel component to fall back on.
        """
        if n_structures <= 0:
            raise ValueError(f"n_structures must be positive, got {n_structures}")
        generator = _resolve_rng(rng)
        required_tokens, allowed_tokens = self._resolve_query(required, allowed)
        feasible = self.feasible_systems(
            required_tokens, allowed_tokens, min_arity=min_arity, max_arity=max_arity,
            system_temperature=system_temperature)
        if novel_fraction is None:
            novel_fraction = float(self.metadata.get("held_out_novelty_rate", 0.0))
        if not 0.0 <= novel_fraction <= 1.0:
            raise ValueError(f"novel_fraction must be in [0, 1], got {novel_fraction}")

        arity_low, arity_high = self._arity_bounds(
            required_tokens, allowed_tokens, min_arity, max_arity)
        novel_possible = arity_low <= arity_high and self.system_arity_counts[
            arity_low:arity_high + 1].sum() > 0
        if not len(feasible):
            if not novel_possible or novel_fraction <= 0:
                raise ValueError(
                    "No training system satisfies this query: "
                    f"required={self._describe(required_tokens) or '(none)'}, "
                    f"allowed={self._describe(allowed_tokens)}"
                    + ("" if novel_possible else ", and no arity is admissible")
                    + ". Widen the palette, relax the arity bounds, or pass "
                      "novel_fraction > 0 to propose systems the data has not seen.")
            logger.warning(
                "No observed system satisfies the query; drawing every row from the "
                "novel-system proposal")
            novel_fraction = 1.0
        if novel_fraction > 0 and not novel_possible:
            logger.warning(
                "The novel-system proposal cannot produce a system within the arity "
                "bounds; drawing every row from the observed systems")
            novel_fraction = 0.0

        n_novel = int(generator.binomial(n_structures, novel_fraction))
        n_empirical = n_structures - n_novel

        drawn: List[Tuple[int, ...]] = []
        novel_flags: List[bool] = []
        if n_empirical:
            picks = generator.choice(len(feasible), size=n_empirical, p=feasible.weights)
            for index in picks:
                drawn.append(self.system_tokens(int(feasible.indices[index])))
                novel_flags.append(False)
        if n_novel:
            novel_systems = self._sample_novel_systems(
                n_novel, required_tokens, allowed_tokens, arity_low, arity_high, generator)
            drawn.extend(novel_systems)
            novel_flags.extend([True] * len(novel_systems))
            if len(novel_systems) < n_novel:
                shortfall = n_novel - len(novel_systems)
                if not len(feasible):
                    raise ValueError(
                        "The novel proposal could not find a system this palette admits "
                        "and the data has not seen, and there is no observed system to "
                        "fall back on. Widen the palette or relax the arity bounds.")
                logger.warning(
                    "The novel proposal kept landing on systems the data already has; "
                    "%i of %i novel rows fell back to the observed distribution",
                    shortfall, n_novel)
                picks = generator.choice(len(feasible), size=shortfall, p=feasible.weights)
                for index in picks:
                    drawn.append(self.system_tokens(int(feasible.indices[index])))
                    novel_flags.append(False)

        space_groups = self._sample_space_groups(drawn, sg_kappa, sg_temperature, generator)
        order = generator.permutation(len(drawn))
        return SystemDraws(
            element_tokens=tuple(drawn[index] for index in order),
            space_groups=space_groups[order],
            is_novel=np.asarray(novel_flags, dtype=bool)[order],
            element_symbols=self.element_symbols,
            query={
                "required": self._describe(required_tokens),
                "allowed": self._describe(allowed_tokens),
                "n_structures": int(n_structures),
                "novel_fraction": float(novel_fraction),
                "system_temperature": float(system_temperature),
                "sg_kappa": float(sg_kappa),
                "sg_temperature": float(sg_temperature),
                "min_arity": min_arity,
                "max_arity": max_arity,
                "source": self.metadata.get("source"),
            },
        )

    # ------------------------------------------------------------------ internals

    def _resolve_query(self, required, allowed) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        required_tokens = self._as_tokens(required)
        if allowed is None:
            allowed_tokens = self.element_tokens
        else:
            allowed_tokens = self._as_tokens(allowed)
            if not allowed_tokens:
                raise ValueError("The allowed set is empty")
        service = set(required_tokens) & self._service_tokens
        if service:
            raise ValueError(
                f"{sorted(self.element_symbols[token] for token in service)} are service "
                "tokens, not elements")
        allowed_tokens = tuple(token for token in allowed_tokens
                               if token not in self._service_tokens)
        missing = set(required_tokens) - set(allowed_tokens)
        if missing:
            raise ValueError(
                f"Required elements {sorted(self.element_symbols[token] for token in missing)} "
                "are not in the allowed set. Every required element has to be allowed; "
                "the two sets are a floor and a ceiling on the same system.")
        return required_tokens, tuple(sorted(allowed_tokens))

    def _as_tokens(self, elements) -> Tuple[int, ...]:
        if elements is None:
            return ()
        if isinstance(elements, str):
            return self.parse_elements(elements)
        elements = list(elements)
        if not elements:
            return ()
        if all(isinstance(item, (int, np.integer)) for item in elements):
            for token in elements:
                if not 0 <= int(token) < self.n_elements:
                    raise ValueError(
                        f"Element token {int(token)} outside a vocabulary of {self.n_elements}")
            return tuple(sorted({int(token) for token in elements}))
        return self.parse_elements(elements)

    def _describe(self, tokens: Iterable[int]) -> str:
        return SYSTEM_DELIMITER.join(self.element_symbols[token] for token in tokens)

    def _arity_bounds(self, required_tokens, allowed_tokens, min_arity, max_arity):
        low = max(1, len(required_tokens), min_arity or 1)
        high = min(len(allowed_tokens), len(self.system_arity_counts) - 1)
        if max_arity is not None:
            high = min(high, max_arity)
        return low, high

    def _sample_novel_systems(
        self,
        n_systems: int,
        required_tokens: Tuple[int, ...],
        allowed_tokens: Tuple[int, ...],
        arity_low: int,
        arity_high: int,
        generator: np.random.Generator,
    ) -> List[Tuple[int, ...]]:
        """Propose systems the training data does not contain.

        An arity first, then the remaining elements without replacement, both
        weighted by how many *distinct* systems each already appears in rather than
        by how many rows: a corpus that enumerated ten thousand quaternary oxides
        says more about its own campaign than about how often oxygen belongs in a
        new system. Measured on the held-out systems that training never saw,
        distinct-system weighting scores -14.55 nats against -15.98 for the naive
        row-weighted uniform pair.

        Draws that land on an already-observed system are rejected, so the
        component is genuinely the complement of the empirical one and the mixture
        is a mixture. The retry budget is finite: a palette whose systems are
        nearly all observed returns short, and `sample` tops the batch up from the
        counts rather than looping.
        """
        free_tokens = np.asarray([token for token in allowed_tokens
                                  if token not in set(required_tokens)], dtype=np.int64)
        arity_weights = self.system_arity_counts[arity_low:arity_high + 1].copy()
        # An arity needs enough free elements to reach it.
        for offset, arity in enumerate(range(arity_low, arity_high + 1)):
            if arity - len(required_tokens) > len(free_tokens):
                arity_weights[offset] = 0.0
        if not arity_weights.sum() > 0:
            return []
        arity_probabilities = _normalise(arity_weights)

        if len(free_tokens):
            element_weights = self.element_system_counts[free_tokens].astype(np.float64)
            if not element_weights.sum() > 0:
                element_weights = np.ones(len(free_tokens), dtype=np.float64)
            element_probabilities = _normalise(element_weights)
        else:
            # The palette is exactly the required set: the only candidate is that set
            # itself, which is novel only if the data has never seen it.
            element_probabilities = None

        systems: List[Tuple[int, ...]] = []
        budget = 20 * n_systems + 100
        while len(systems) < n_systems and budget > 0:
            budget -= 1
            arity = arity_low + int(generator.choice(len(arity_probabilities), p=arity_probabilities))
            n_free = arity - len(required_tokens)
            if n_free > len(free_tokens):
                continue
            if n_free > 0:
                picked = generator.choice(
                    len(free_tokens), size=n_free, replace=False, p=element_probabilities)
                extra = {int(free_tokens[index]) for index in picked}
            else:
                extra = set()
            candidate = tuple(sorted(set(required_tokens) | extra))
            if self.index_of(candidate) is not None:
                continue
            systems.append(candidate)
        return systems

    def _sample_space_groups(
        self,
        systems: Sequence[Tuple[int, ...]],
        kappa: float,
        temperature: float,
        generator: np.random.Generator,
    ) -> np.ndarray:
        """A space group per row, drawn from that row's `p(G | S)`.

        Rows sharing a system share a distribution, so it is built once per distinct
        system and their space groups drawn in one multinomial.
        """
        rows_by_system: Dict[Tuple[int, ...], List[int]] = {}
        for row, tokens in enumerate(systems):
            rows_by_system.setdefault(tokens, []).append(row)
        out = np.empty(len(systems), dtype=np.int64)
        for tokens, rows in rows_by_system.items():
            probabilities = self.space_group_probabilities(tokens, kappa, temperature)
            picks = generator.choice(self.n_space_groups, size=len(rows), p=probabilities)
            out[rows] = self.space_groups[picks]
        return out

    # ------------------------------------------------------------------ reporting

    def describe_query(
        self,
        required: Union[None, str, Iterable[str], Iterable[int]] = None,
        allowed: Union[None, str, Iterable[str], Iterable[int]] = None,
        top: int = 15,
        min_arity: Optional[int] = None,
        max_arity: Optional[int] = None,
        system_temperature: float = 1.0,
        sg_kappa: float = DEFAULT_SG_KAPPA,
    ) -> str:
        """What a query admits, before spending a GPU hour on it."""
        required_tokens, allowed_tokens = self._resolve_query(required, allowed)
        feasible = self.feasible_systems(
            required_tokens, allowed_tokens, min_arity=min_arity, max_arity=max_arity,
            system_temperature=system_temperature)
        lines = [
            f"required: {self._describe(required_tokens) or '(none)'}",
            f"allowed:  {self._describe(allowed_tokens)} ({len(allowed_tokens)} elements)",
            f"observed feasible systems: {len(feasible)} "
            f"({feasible.total_rows:.0f} training rows, "
            f"{100 * feasible.acceptance_rate:.2f}% of the corpus -- the acceptance rate "
            "a rejection sampler would have had)",
        ]
        if not len(feasible):
            lines.append("nothing observed; every row would come from the novel proposal")
            return "\n".join(lines)
        arity = np.diff(self.system_token_offsets)[feasible.indices]
        by_arity = {int(value): int((arity == value).sum()) for value in np.unique(arity)}
        lines.append(f"by arity: {by_arity}")
        order = np.argsort(-feasible.weights)[:top]
        lines.append(f"top {min(top, len(order))} systems by sampling weight:")
        for position in order:
            index = int(feasible.indices[position])
            probabilities = self.space_group_probabilities(
                self.system_tokens(index), kappa=sg_kappa)
            best = np.argsort(-probabilities)[:4]
            lines.append(
                f"  {'-'.join(self.system_symbols(index)):22s} "
                f"p={feasible.weights[position]:.4f} rows={self.system_counts[index]:7.0f}  "
                + " ".join(f"{int(self.space_groups[i])}:{probabilities[i]:.2f}" for i in best))
        marginal = np.zeros(self.n_space_groups, dtype=np.float64)
        for position in range(len(feasible)):
            index = int(feasible.indices[position])
            start, stop = self.cell_offsets[index], self.cell_offsets[index + 1]
            marginal[self.cell_space_group_index[start:stop]] += (
                feasible.weights[position]
                * self.cell_counts[start:stop] / self.system_counts[index])
        global_p = _normalise(self.global_sg_counts)
        kl = float(np.sum(np.where(marginal > 0, marginal * np.log(
            np.where(marginal > 0, marginal, 1.0) / global_p), 0.0)))
        best = np.argsort(-marginal)[:6]
        lines.append(
            "space groups under this palette: "
            + " ".join(f"{int(self.space_groups[i])}:{marginal[i]:.3f}" for i in best))
        best_global = np.argsort(-global_p)[:6]
        lines.append(
            "         unconditionally:        "
            + " ".join(f"{int(self.space_groups[i])}:{global_p[i]:.3f}" for i in best_global))
        lines.append(f"KL(palette || unconditional) = {kl:.3f} nats")
        return "\n".join(lines)


# ---------------------------------------------------------------- cache reading

def _resolve_cache_path(dataset: str, cache_root: Optional[Path] = None) -> Path:
    """`cache/<dataset>`, tolerating the dash/underscore spelling as `generate` does."""
    if cache_root is None:
        cache_root = Path.cwd() / "cache"
    for candidate in (dataset, dataset.replace("-", "_"), dataset.replace("_", "-")):
        path = Path(cache_root) / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Dataset {dataset!r} not found under {cache_root}")


def _resolve_cache_files(cache_path: Path, config_name: Optional[str] = None) -> Tuple[Path, Path]:
    """The tensor/tokeniser pair for a tokeniser config, or the only one there is."""
    from wyckoff_transformer.tokenization import TENSOR_CACHE_SUFFIX  # noqa: PLC0415

    tensor_dir, tokeniser_dir = cache_path / "tensors", cache_path / "tokenisers"
    if not tensor_dir.exists() or not tokeniser_dir.exists():
        raise FileNotFoundError(f"{cache_path} has no 'tensors' and 'tokenisers' directories")
    pairs = []
    for tensor_path in sorted(tensor_dir.glob(f"*{TENSOR_CACHE_SUFFIX}")):
        tokeniser_path = tokeniser_dir / f"{tensor_path.stem}.json"
        if tokeniser_path.exists():
            pairs.append((tensor_path, tokeniser_path))
    if config_name is not None:
        for tensor_path, tokeniser_path in pairs:
            if tensor_path.stem == config_name:
                return tensor_path, tokeniser_path
        raise FileNotFoundError(
            f"Tokeniser config {config_name!r} not cached under {cache_path}; "
            f"found {[path.stem for path, _ in pairs]}")
    if not pairs:
        raise FileNotFoundError(f"No tensor/tokeniser pair under {cache_path}")
    if len(pairs) > 1:
        raise ValueError(
            f"{cache_path} holds several tokeniser configs "
            f"({[path.stem for path, _ in pairs]}); name one. It has to be the config the "
            "model was trained with, or the element tokens will not line up.")
    return pairs[0]


def _read_cache_node(handle, structure: Dict, split: str, field: str):
    """One field of one split out of a tensor cache, without loading the rest.

    The caches are 6.5 GB and this needs two columns of them, so the safetensors
    structure metadata is walked by hand rather than going through
    `load_tensor_cache`, which materialises every tensor in the file.
    """
    try:
        node = structure["items"][split]["items"][field]
    except KeyError:
        raise KeyError(
            f"Field {field!r} is not in split {split!r} of this tensor cache. "
            f"Available: {sorted(structure['items'].get(split, {}).get('items', {}))}") from None
    if node["type"] == "tensor":
        return handle.get_tensor(node["key"]).numpy()
    if node["type"] == "tensor_list" and node.get("storage") == "concat":
        return (handle.get_tensor(node["data"]).numpy(),
                handle.get_tensor(node["lengths"]).numpy())
    raise ValueError(f"Field {field!r} has an unsupported cache layout: {node}")


def _decode_space_groups(encoded: np.ndarray, tokeniser) -> np.ndarray:
    """Cached start tokens back to space group numbers.

    Two layouts reach here: a 1-D tensor of token ids from an `EnumeratingTokeniser`,
    and the `SpaceGroupEncoder`'s symmetry-matrix rows. The reference for the second
    is re-encoded at the cache's own dtype and matched bytewise, so nothing depends
    on how the encoding floats round.
    """
    if encoded.ndim == 1:
        return np.asarray([int(tokeniser.to_token[int(index)]) for index in encoded],
                          dtype=np.int64)
    if not hasattr(tokeniser, "np_dict"):
        raise ValueError("A 2-D start tensor needs a SpaceGroupEncoder to decode it")
    numbers = list(tokeniser.np_dict.keys())
    reference = np.stack([np.asarray(tokeniser.np_dict[sg]) for sg in numbers]).astype(encoded.dtype)
    lookup = {row.tobytes(): int(sg) for row, sg in zip(reference, numbers)}
    if len(lookup) != len(numbers):
        raise ValueError("The space group encoding is not unique at the cache's dtype")
    out = np.empty(len(encoded), dtype=np.int64)
    for row_index, row in enumerate(encoded):
        try:
            out[row_index] = lookup[row.tobytes()]
        except KeyError:
            raise ValueError(
                f"Row {row_index} of the cached start tensor is not any known space "
                "group encoding") from None
    return out


def _split_systems(token_data: np.ndarray, lengths: np.ndarray) -> List[Tuple[int, ...]]:
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    return [tuple(sorted(set(token_data[offsets[row]:offsets[row + 1]].tolist())))
            for row in range(len(lengths))]


def prior_from_tensor_cache(
    dataset: str,
    config_name: Optional[str] = None,
    cache_root: Optional[Union[str, Path]] = None,
    splits: Sequence[str] = ("train",),
    held_out_splits: Sequence[str] = ("val",),
    max_values: Optional[Dict[str, float]] = None,
    composition_field: str = "composition_tokens",
    start_field: str = "spacegroup_number",
) -> SystemSpaceGroupPrior:
    """Build a prior from a tokenised dataset in `cache/`.

    The tables come from the *tokenised* cache rather than from the dataframe on
    purpose: it is the same file the trainer reads, so the element tokens, the
    space group vocabulary and the train/val split are the model's own by
    construction, and a prior can never quietly disagree with the checkpoint it is
    sampling for.

    Args:
        dataset: Directory under `cache/`, e.g. `lemat_bulk_fmax1`.
        config_name: Tokeniser config stem, when the dataset has more than one.
        splits: Splits counted into the tables. The default is training only:
            the prior is meant to be the model's own input distribution, and
            counting validation into it would make the novelty rate below a
            measurement of nothing.
        held_out_splits: Splits used only to measure how often a system is new,
            which becomes the default `novel_fraction`.
        max_values: Upper bounds a row must satisfy to be counted, by field name,
            e.g. ``{"energy_above_hull": 0.1}``. This changes the question the
            prior answers, from "where does this palette put atoms" to "where does
            it put atoms *stably*", and is worth doing when the campaign is going
            to condition on a low hull energy anyway.
    """
    from safetensors import safe_open  # noqa: PLC0415

    from wyckoff_transformer.wyckoff_processor import WyckoffProcessor  # noqa: PLC0415

    cache_path = _resolve_cache_path(dataset, Path(cache_root) if cache_root else None)
    tensor_path, tokeniser_path = _resolve_cache_files(cache_path, config_name)
    tokenisers = WyckoffProcessor.from_pretrained(tokeniser_path).tokenisers
    elements_tokeniser = tokenisers["elements"]
    element_symbols = [str(symbol) for symbol in elements_tokeniser.to_token]
    sg_tokeniser = tokenisers[start_field]

    systems: List[Tuple[int, ...]] = []
    space_groups: List[int] = []
    kept, seen = 0, 0
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        structure = json.loads(metadata["wyckoff_transformer_tensor_cache_structure"])
        for split in splits:
            token_data, lengths = _read_cache_node(handle, structure, split, composition_field)
            split_systems = _split_systems(token_data, lengths)
            split_sgs = _decode_space_groups(
                _read_cache_node(handle, structure, split, start_field), sg_tokeniser)
            keep = np.ones(len(split_systems), dtype=bool)
            for field, bound in (max_values or {}).items():
                values = _read_cache_node(handle, structure, split, field).reshape(len(keep), -1)
                keep &= (values <= bound).all(axis=1)
            seen += len(split_systems)
            kept += int(keep.sum())
            systems.extend(system for system, take in zip(split_systems, keep) if take)
            space_groups.extend(int(sg) for sg, take in zip(split_sgs, keep) if take)
            logger.info("%s/%s: %i rows, %i kept", dataset, split, len(split_systems), keep.sum())
        held_out: List[Tuple[int, ...]] = []
        for split in held_out_splits:
            if split not in structure["items"]:
                logger.warning("Held-out split %r absent from the cache", split)
                continue
            token_data, lengths = _read_cache_node(handle, structure, split, composition_field)
            split_systems = _split_systems(token_data, lengths)
            # The same filter, or the novelty rate stops being one: against a prior
            # built from stable rows only, an unfiltered held-out split would count
            # "this system has no stable member in training" as "this system is new".
            keep = np.ones(len(split_systems), dtype=bool)
            for field, bound in (max_values or {}).items():
                values = _read_cache_node(handle, structure, split, field).reshape(len(keep), -1)
                keep &= (values <= bound).all(axis=1)
            held_out.extend(system for system, take in zip(split_systems, keep) if take)

    if not systems:
        raise ValueError(
            f"No row of {dataset} survived the filters {max_values}; the prior would be empty")
    return SystemSpaceGroupPrior.from_rows(
        systems=systems,
        space_groups=space_groups,
        element_symbols=element_symbols,
        held_out_systems=held_out or None,
        metadata={
            "source": f"{dataset}/{tensor_path.stem}",
            "dataset": dataset,
            "tokeniser_config": tensor_path.stem,
            "splits": list(splits),
            "held_out_splits": list(held_out_splits),
            "max_values": dict(max_values or {}),
            "rows_seen": int(seen),
            "rows_kept": int(kept),
        },
    )
