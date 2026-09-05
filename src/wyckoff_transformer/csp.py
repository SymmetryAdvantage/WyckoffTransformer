"""Crystal structure prediction: sampling a Wyckoff gene for a *given* composition.

De novo generation asks the model for any plausible crystal.  CSP fixes the
formula and asks for the structure it adopts, which changes the sampling problem
in two ways.

The composition becomes a hard constraint.  A gene's composition is not a free
choice made at the end -- it is the sum of the multiplicities of the Wyckoff
positions assigned to each element, decided one site at a time.  Ancestral
sampling with a rejection test at the end wastes most of its draws: a decoder
that has placed a 4-fold position for an element needing 6 more atoms can only
finish if the space group offers positions summing to exactly 6, and usually it
has already made that impossible several sites earlier.  `SpaceGroupCombinatorics`
answers the reachability question exactly, so the decoder can mask any choice
that strands the composition, and every sequence it emits has the target formula
by construction.

Selection replaces novelty.  There is a right answer, so candidates are ranked
rather than filtered, and the ranking signal is a regressor trained with the
censored likelihood in `wyckoff_transformer.censored` -- an estimate of
``min(E | gene)``, the best energy the gene could reach, which is the correct
quantity when the downstream reconstruction gets several attempts at the gene's
manifold.

Two conditioning modes, both supplied by the caller as `condition_value`:

(a) ``Delta_E_polymorph = 0``: the gene whose optimum is the ground-state
    polymorph of this composition.  The CSP target stated exactly.
(b) the formation energy that places the composition on the convex hull, for a
    backbone conditioned on formation energy.  An affine reparametrisation of
    ``e_hull = 0`` per composition.

The multiplicity rule
---------------------

A Wyckoff position with no positional freedom (``dof == 0``) is a fixed set of
points, so a structure can occupy it at most once; a position with ``dof > 0``
is a continuous orbit and can be occupied repeatedly at different coordinates.
That is the rule `WyckoffProcessor.pyxtal_notation_to_sites` enforces when it
turns a gene into a structure, and it is what makes reachability interesting:
the reusable multiplicities form an unbounded coin problem, and the fixed ones a
scarce resource shared between the elements.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

#: Largest atom count per element the reachability tables are built for. Conventional
#: cells in the training sets stay far below this; a composition above it is rejected
#: rather than silently truncated.
MAX_ATOMS_PER_ELEMENT = 256

#: Ceiling on nodes explored by the exact special-position search before it gives up
#: and reports "feasible". Erring towards feasible only lets a doomed sequence run to
#: the end, where it is dropped; erring the other way would forbid legal structures.
FEASIBILITY_NODE_BUDGET = 4096


@dataclass(frozen=True)
class WyckoffPosition:
    """One Wyckoff position of one space group, in the model's token space."""
    ss_token: int
    enum_token: int
    letter: str
    multiplicity: int
    dof: int

    @property
    def reusable(self) -> bool:
        """Whether a structure may occupy this position more than once.

        True exactly when the position has positional freedom: the occupations
        then sit at different coordinates and are different orbits.
        """
        return self.dof > 0


def _unbounded_reachability(
    coins: Sequence[int], limit: int) -> Tuple[np.ndarray, np.ndarray]:
    """Which totals a multiset of reusable multiplicities can hit, and in how few.

    Returns ``(reachable, min_sites)`` over ``0..limit``. ``min_sites`` is the
    fewest positions summing to the total, and `limit + 1` where unreachable, so
    it can be summed across elements and compared with the remaining slots.
    """
    reachable = np.zeros(limit + 1, dtype=bool)
    min_sites = np.full(limit + 1, limit + 1, dtype=np.int32)
    reachable[0] = True
    min_sites[0] = 0
    for total in range(1, limit + 1):
        for coin in coins:
            if coin <= total and reachable[total - coin]:
                reachable[total] = True
                if min_sites[total - coin] + 1 < min_sites[total]:
                    min_sites[total] = min_sites[total - coin] + 1
    return reachable, min_sites


class SpaceGroupCombinatorics:
    """Which per-element atom counts a space group can build, and with what.

    Splits the space group's positions into the reusable ones, which form an
    unbounded coin system each element can draw on independently, and the fixed
    ones, which are a scarce resource the elements compete for.  That split is
    what makes an exact answer cheap in the common case: if every element's
    deficit is a sum of reusable multiplicities, the elements do not interact
    and the state is feasible with no search at all.

    Args:
        sg_number: The real space group number, for diagnostics.
        positions: Every Wyckoff position of the group.
        max_atoms: Table size; see `MAX_ATOMS_PER_ELEMENT`.
    """

    def __init__(
        self,
        sg_number: int,
        positions: Sequence[WyckoffPosition],
        max_atoms: int = MAX_ATOMS_PER_ELEMENT,
    ):
        self.sg_number = sg_number
        self.positions = tuple(positions)
        self.max_atoms = max_atoms
        self.by_key = {(p.ss_token, p.enum_token): p for p in positions}
        self.reusable = tuple(p for p in positions if p.reusable)
        self.fixed = tuple(p for p in positions if not p.reusable)

        reusable_mults = sorted({p.multiplicity for p in self.reusable})
        # Sufficient: reachable with reusable positions alone means the elements never
        # compete, because a reusable position can serve all of them at once.
        self.reachable_reusable, self.min_sites_reusable = _unbounded_reachability(
            reusable_mults, max_atoms)
        # Necessary: nothing is reachable that is not reachable when the once-only rule
        # on fixed positions is dropped, so failing this prunes soundly.
        all_mults = sorted({p.multiplicity for p in positions})
        self.reachable_any, self.min_sites_any = _unbounded_reachability(all_mults, max_atoms)

    @property
    def smallest_multiplicity(self) -> int:
        return min(p.multiplicity for p in self.positions)

    def can_complete(
        self,
        deficits: Sequence[int],
        used_fixed: frozenset,
        slots_left: int,
    ) -> bool:
        """Can the remaining atom deficits be covered exactly?

        Args:
            deficits: Atoms still owed, one entry per element of the target.
            used_fixed: ``(ss_token, enum_token)`` keys of fixed positions already
                occupied, which are no longer available.
            slots_left: Sequence positions left before `max_sequence_length`.

        Returns:
            False only when completion is provably impossible.  The bounded search
            in the middle band reports True when it runs out of budget, so a False
            is trustworthy and a True is occasionally optimistic.
        """
        if any(d < 0 for d in deficits):
            return False
        outstanding = [d for d in deficits if d > 0]
        if not outstanding:
            return True
        if any(d > self.max_atoms for d in outstanding):
            return False
        if slots_left <= 0:
            return False

        # Necessary condition, and the cheapest: every deficit must be some sum of the
        # group's multiplicities, and the sites that takes must fit in the slots left.
        if not all(self.reachable_any[d] for d in outstanding):
            return False
        if sum(self.min_sites_any[d] for d in outstanding) > slots_left:
            return False

        # Sufficient condition. Reusable positions are unlimited and shared, so if each
        # element can be finished out of them alone there is nothing to arbitrate.
        if (all(self.reachable_reusable[d] for d in outstanding)
                and sum(self.min_sites_reusable[d] for d in outstanding) <= slots_left):
            return True

        # Otherwise at least one element needs a fixed position, and they are scarce.
        available = [p for p in self.fixed if (p.ss_token, p.enum_token) not in used_fixed]
        return self._search_fixed(tuple(outstanding), available, slots_left)

    def _search_fixed(
        self,
        deficits: Tuple[int, ...],
        available: List[WyckoffPosition],
        slots_left: int,
    ) -> bool:
        """Exact assignment of scarce fixed positions to elements, depth first.

        Each fixed position goes to one element or to none; an element is finished
        as soon as its remainder is reachable from the reusable positions alone.
        Distinct positions of equal multiplicity are interchangeable, so only one
        of each multiplicity is tried at a given depth.
        """
        budget = [FEASIBILITY_NODE_BUDGET]
        seen = set()

        def recurse(index: int, remaining: Tuple[int, ...], slots: int) -> bool:
            if all(r == 0 or self.reachable_reusable[r] for r in remaining):
                if sum(0 if r == 0 else self.min_sites_reusable[r] for r in remaining) <= slots:
                    return True
            if index >= len(available) or slots <= 0:
                return False
            budget[0] -= 1
            if budget[0] <= 0:
                logger.debug("Feasibility search budget exhausted in space group %d",
                             self.sg_number)
                return True
            state = (index, remaining, slots)
            if state in seen:
                return False
            seen.add(state)
            # Prune on the relaxed condition, which stays valid at every depth.
            if not all(r == 0 or self.reachable_any[r] for r in remaining):
                return False

            tried_multiplicities = set()
            for offset in range(index, len(available)):
                multiplicity = available[offset].multiplicity
                if multiplicity in tried_multiplicities:
                    continue
                tried_multiplicities.add(multiplicity)
                for element, owed in enumerate(remaining):
                    if owed < multiplicity:
                        continue
                    nxt = list(remaining)
                    nxt[element] = owed - multiplicity
                    if recurse(offset + 1, tuple(nxt), slots - 1):
                        return True
            return False

        return recurse(0, deficits, slots_left)


def group_multiplicity_db(multiplicity_db) -> Dict:
    """Index the multiplicity table by its space group key.

    The first index level holds whatever the start tokeniser produces, which for
    a one-hot `SpaceGroupEncoder` is a tuple of the encoded vector and for an
    enumerating tokeniser a plain integer.  Pandas cannot be asked for "the rows
    whose first level equals this tuple" without the tuple being read as a
    multi-level key, so the table is walked once and grouped here instead. It
    holds a couple of thousand rows, so the pass is not worth avoiding.
    """
    grouped: Dict = {}
    for key, multiplicity in multiplicity_db.items():
        sg_key, ss_token, enum_token = key
        grouped.setdefault(sg_key, []).append((ss_token, enum_token, multiplicity))
    return grouped


def start_to_db_key(start: Tensor):
    """The multiplicity table's space group key for a start tensor.

    Mirrors what `WyckoffGenerator.generate_tensors` does when it feeds the
    engineers: a multi-valued start token becomes a tuple, a scalar one stays a
    scalar.
    """
    flat = start.reshape(-1) if start.dim() <= 1 else start.reshape(start.shape[0], -1)[0]
    if flat.numel() == 1:
        return flat.item()
    return tuple(flat.tolist())


def build_combinatorics(
    sg_number: int,
    sg_token,
    multiplicity_db,
    letter_from_ss_enum_idx: Dict,
    wp_index: Dict,
    ss_tokeniser,
    max_atoms: int = MAX_ATOMS_PER_ELEMENT,
    grouped_db: Optional[Dict] = None,
) -> SpaceGroupCombinatorics:
    """Assemble one space group's positions from the tokenisers and tables.

    The multiplicity engineer is the authority on which ``(site symmetry,
    enumeration)`` token pairs exist in the group -- it is what the generator's
    own validity check consults -- while `wp_index` supplies the degrees of
    freedom that decide whether a position can be occupied twice.

    Args:
        sg_token: The start-tensor key into the multiplicity table, as
            `start_to_db_key` builds it.
        grouped_db: Output of `group_multiplicity_db`, to avoid regrouping the
            table once per space group.
    """
    positions = []
    grouped = grouped_db if grouped_db is not None else group_multiplicity_db(multiplicity_db)
    group_db = grouped.get(sg_token)
    if not group_db:
        raise KeyError(f"Space group {sg_number} is absent from the multiplicity table")
    for ss_token, enum_token, multiplicity in group_db:
        ss_string = ss_tokeniser.to_token[ss_token]
        try:
            letter = letter_from_ss_enum_idx[sg_number][ss_string][enum_token]
            dof = wp_index[sg_number][ss_string][letter][1]
        except KeyError:
            # Present in the multiplicity table but not a real position of this group.
            continue
        positions.append(WyckoffPosition(
            ss_token=int(ss_token), enum_token=int(enum_token), letter=letter,
            multiplicity=int(multiplicity), dof=int(dof)))
    if not positions:
        raise ValueError(f"No Wyckoff positions recovered for space group {sg_number}")
    return SpaceGroupCombinatorics(sg_number, positions, max_atoms=max_atoms)


_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> Counter:
    """``"Na2Cl2"`` or ``"BaTiO3"`` into a Counter of element symbol to count."""
    counts: Counter = Counter()
    position = 0
    for match in _FORMULA_TOKEN.finditer(formula.strip()):
        if match.start() != position:
            raise ValueError(f"Cannot parse formula {formula!r} at offset {position}")
        position = match.end()
        symbol, digits = match.groups()
        counts[symbol] += int(digits) if digits else 1
    if position != len(formula.strip()) or not counts:
        raise ValueError(f"Cannot parse formula {formula!r}")
    return counts


@dataclass
class CompositionTarget:
    """A conventional-cell composition, in element-token space.

    Multiplicities count atoms in the conventional cell, so this is a *cell*
    content and not a reduced formula: ``NaCl`` at ``z=4`` is the target, not
    ``NaCl``.  `for_formula` builds one from a reduced formula and a number of
    formula units.
    """
    symbols: Tuple[str, ...]
    counts: Tuple[int, ...]
    element_tokens: Tuple[int, ...]

    @classmethod
    def for_formula(cls, formula: str, z: int, elements_tokeniser) -> "CompositionTarget":
        reduced = parse_formula(formula)
        symbols, counts, tokens = [], [], []
        for symbol, count in sorted(reduced.items()):
            symbols.append(symbol)
            counts.append(count * z)
            tokens.append(_element_token(symbol, elements_tokeniser))
        return cls(tuple(symbols), tuple(counts), tuple(tokens))

    @property
    def total_atoms(self) -> int:
        return sum(self.counts)

    def __str__(self) -> str:
        return "".join(f"{s}{c}" for s, c in zip(self.symbols, self.counts))


def _element_token(symbol: str, elements_tokeniser) -> int:
    """Element symbol to token, across the several key conventions in use."""
    from pymatgen.core import Element  # noqa: PLC0415
    for key in (Element(symbol), f"Element {symbol}", symbol):
        try:
            if key in elements_tokeniser:
                return elements_tokeniser[key]
        except TypeError:
            continue
    raise KeyError(f"Element {symbol!r} is not in the model's vocabulary")


@dataclass
class BeamState:
    """One partially decoded gene under an exact composition constraint."""
    deficits: List[int]
    used_fixed: frozenset = frozenset()
    rows: List[Tuple[int, int, int]] = field(default_factory=list)
    log_prob: float = 0.0
    finished: bool = False

    @property
    def n_sites(self) -> int:
        return len(self.rows)

    def place(self, element_index: int, position: WyckoffPosition,
              element_token: int, log_prob: float) -> "BeamState":
        deficits = list(self.deficits)
        deficits[element_index] -= position.multiplicity
        used = self.used_fixed
        if not position.reusable:
            used = used | {(position.ss_token, position.enum_token)}
        return BeamState(
            deficits=deficits,
            used_fixed=used,
            rows=self.rows + [(element_token, position.ss_token, position.enum_token)],
            log_prob=self.log_prob + log_prob,
            finished=all(d == 0 for d in deficits),
        )

    def legal_positions(
        self,
        element_index: int,
        combinatorics: SpaceGroupCombinatorics,
        max_sites: int,
    ) -> List[WyckoffPosition]:
        """Positions this element may take next without stranding the composition.

        A position is legal when it does not overshoot the element's deficit, is
        still available if it is a fixed one, and leaves a state that can still be
        completed within the remaining sequence slots.
        """
        owed = self.deficits[element_index]
        slots_left = max_sites - self.n_sites - 1
        legal = []
        for position in combinatorics.positions:
            if position.multiplicity > owed:
                continue
            if not position.reusable and (position.ss_token, position.enum_token) in self.used_fixed:
                continue
            deficits = list(self.deficits)
            deficits[element_index] = owed - position.multiplicity
            used = self.used_fixed
            if not position.reusable:
                used = used | {(position.ss_token, position.enum_token)}
            if combinatorics.can_complete(deficits, used, slots_left):
                legal.append(position)
        return legal

    def live_elements(
        self,
        combinatorics: SpaceGroupCombinatorics,
        max_sites: int,
    ) -> List[int]:
        """Indices of elements that can legally receive the next site."""
        return [index for index, owed in enumerate(self.deficits)
                if owed > 0 and self.legal_positions(index, combinatorics, max_sites)]



@dataclass
class CSPCandidate:
    """A completed gene with the target composition."""
    sg_number: int
    rows: Tuple[Tuple[int, int, int], ...]
    log_prob: float
    n_sites: int
    #: Predicted ``min(E | gene)`` from the censored regressor; None until ranked.
    predicted_energy: Optional[float] = None

    def normalised_log_prob(self, length_penalty: float = 1.0) -> float:
        """Sequence log-probability divided by ``n_sites ** length_penalty``.

        The raw sum has one term per site, so it prefers few-site genes for
        reasons that have nothing to do with them being better structures.
        """
        if length_penalty == 0 or self.n_sites == 0:
            return self.log_prob
        return self.log_prob / (self.n_sites ** length_penalty)


class ConstrainedDecoder:
    """Decodes Wyckoff genes whose composition is fixed in advance.

    Wraps a trained backbone and the per-space-group multiplicity tables, and
    exposes two strategies over the same masking machinery.

    ``sample`` draws ancestrally from the masked distribution.  Because the mask
    already removes every choice that strands the composition, the acceptance
    rate is one: each draw is a sample from the model's own distribution
    restricted to genes with the target formula.  This is the strategy to pair
    with reranking, since the candidates stay diverse by construction.

    ``beam`` keeps the ``beam_width`` most likely prefixes at every cascade
    field.  It returns the model's modal genes rather than a sample of them,
    which is the right thing when the backbone's likelihood is what you trust
    and the wrong thing when the candidates are about to be reranked by
    something else: beams sharing a prefix differ only in their last sites, so a
    wide beam buys much less spread over the relaxation budget than the same
    number of samples.

    Args:
        model: A trained `CascadeTransformer` in generation mode.
        cascade_order: Field order the model was trained with.
        cascade_is_target: Which of those fields the model predicts.
        tokenisers, token_engineers, masks, stops: As `WyckoffGenerator` takes them.
        max_sequence_len: Hard cap on sites per gene.
        device: Defaults to the model's.
        max_atoms: Reachability table size; see `MAX_ATOMS_PER_ELEMENT`.
    """

    def __init__(
        self,
        model,
        cascade_order: Sequence[str],
        cascade_is_target: Dict[str, bool],
        tokenisers: Dict,
        token_engineers: Dict,
        masks: Dict,
        stops: Optional[Dict],
        max_sequence_len: int,
        device: Optional[torch.device] = None,
        max_atoms: int = MAX_ATOMS_PER_ELEMENT,
    ):
        self.model = model
        self.cascade_order = tuple(cascade_order)
        self.cascade_is_target = dict(cascade_is_target)
        self.tokenisers = tokenisers
        self.token_engineers = token_engineers
        self.masks = masks
        self.stops = stops
        self.max_sequence_len = int(max_sequence_len)
        self.device = device or next(model.parameters()).device
        self.max_atoms = max_atoms
        for required in ("elements", "site_symmetries", "sites_enumeration"):
            if required not in self.cascade_order:
                raise NotImplementedError(
                    "Composition-constrained decoding needs the site-symmetry cascade "
                    f"(elements, site_symmetries, sites_enumeration); {required} is missing")
        self.field_index = {name: i for i, name in enumerate(self.cascade_order)}
        self.engineered_fields = [name for name in self.cascade_order
                                  if not self.cascade_is_target.get(name, False)]
        unsupported = set(self.engineered_fields) - {"multiplicity"}
        if unsupported:
            raise NotImplementedError(
                f"Non-target cascade fields {sorted(unsupported)} are not supported in CSP mode")
        self._wp_index: Optional[Dict] = None
        self._letter_index: Optional[Dict] = None
        self._combinatorics: Dict[int, SpaceGroupCombinatorics] = {}
        self._grouped_db: Optional[Dict] = None

    @property
    def wp_index(self) -> Dict:
        """Wyckoff positions by space group. Built on first use: it costs a couple of
        seconds and is not needed at all when the tables are supplied directly."""
        if self._wp_index is None:
            from wyckoff_transformer.tokenization import get_wp_index  # noqa: PLC0415
            self._wp_index = get_wp_index()
        return self._wp_index

    @property
    def letter_index(self) -> Dict:
        """Wyckoff letter by space group, site symmetry and enumeration token."""
        if self._letter_index is None:
            self._letter_index = self.tokenisers["sites_enumeration"].get_letter_from_ss_enum_idx()
        return self._letter_index

    def combinatorics(self, sg_number: int, sg_token) -> SpaceGroupCombinatorics:
        """The reachability tables for one space group, built once and cached."""
        if sg_number not in self._combinatorics:
            if self._grouped_db is None:
                self._grouped_db = group_multiplicity_db(
                    self.token_engineers["multiplicity"].db)
            self._combinatorics[sg_number] = build_combinatorics(
                sg_number=sg_number,
                sg_token=start_to_db_key(sg_token) if isinstance(sg_token, Tensor) else sg_token,
                multiplicity_db=None,
                letter_from_ss_enum_idx=self.letter_index,
                wp_index=self.wp_index,
                ss_tokeniser=self.tokenisers["site_symmetries"],
                max_atoms=self.max_atoms,
                grouped_db=self._grouped_db)
        return self._combinatorics[sg_number]

    def _cascade_tensors(
        self,
        states: Sequence[BeamState],
        width: int,
        combinatorics: SpaceGroupCombinatorics,
    ) -> List[Tensor]:
        """Materialise beam states as the cascade tensors the model reads.

        Slots beyond a state's placed sites stay at MASK, which is how the model
        is told which position it is being asked to predict.
        """
        tensors = [
            torch.full((len(states), width), int(self.masks[name]),
                       dtype=torch.int64, device=self.device)
            for name in self.cascade_order]
        element_index = self.field_index["elements"]
        ss_index = self.field_index["site_symmetries"]
        enum_index = self.field_index["sites_enumeration"]
        multiplicity_index = self.field_index.get("multiplicity")
        for row, state in enumerate(states):
            for site, (element, ss, enum) in enumerate(state.rows):
                tensors[element_index][row, site] = element
                tensors[ss_index][row, site] = ss
                tensors[enum_index][row, site] = enum
                if multiplicity_index is not None:
                    tensors[multiplicity_index][row, site] = \
                        combinatorics.by_key[(ss, enum)].multiplicity
        return tensors

    @torch.no_grad()
    def _log_probs(
        self,
        start: Tensor,
        states: Sequence[BeamState],
        site: int,
        cascade_index: int,
        combinatorics: SpaceGroupCombinatorics,
        temperature: float,
        cond: Optional[Tensor],
    ) -> Tensor:
        """Log probabilities over one cascade field, for every state, at one site."""
        tensors = self._cascade_tensors(states, site + 1, combinatorics)
        batch_start = start.expand(len(states), *start.shape[1:]) if start.dim() > 1 \
            else start.expand(len(states))
        batch_cond = cond.expand(len(states), *cond.shape[1:]) if cond is not None else None
        logits = self.model(batch_start, tensors, None, cascade_index, cond=batch_cond)
        return torch.log_softmax(logits.float() / temperature, dim=-1)

    @staticmethod
    def _masked(log_probs: Tensor, allowed: Sequence[int]) -> Tensor:
        """Renormalise a row over `allowed` alone; everything else gets -inf."""
        masked = torch.full_like(log_probs, float("-inf"))
        index = torch.as_tensor(sorted(allowed), dtype=torch.long, device=log_probs.device)
        masked[index] = log_probs[index]
        return masked - torch.logsumexp(masked, dim=0)

    def _partial_tensors(
        self,
        proposals: Sequence["_Proposal"],
        width: int,
        combinatorics: SpaceGroupCombinatorics,
    ) -> List[Tensor]:
        """Cascade tensors for states that have a partly-chosen row at `width - 1`."""
        tensors = self._cascade_tensors([p.state for p in proposals], width, combinatorics)
        site = width - 1
        for row, proposal in enumerate(proposals):
            if proposal.element_token is not None:
                tensors[self.field_index["elements"]][row, site] = proposal.element_token
            if proposal.ss_token is not None:
                tensors[self.field_index["site_symmetries"]][row, site] = proposal.ss_token
        return tensors

    @torch.no_grad()
    def _stage_log_probs(
        self,
        start: Tensor,
        proposals: Sequence["_Proposal"],
        site: int,
        field: str,
        combinatorics: SpaceGroupCombinatorics,
        temperature: float,
        cond: Optional[Tensor],
    ) -> Tensor:
        tensors = self._partial_tensors(proposals, site + 1, combinatorics)
        n = len(proposals)
        batch_start = start.expand(n, *start.shape[1:]) if start.dim() > 1 else start.expand(n)
        batch_cond = cond.expand(n, *cond.shape[1:]) if cond is not None else None
        logits = self.model(batch_start, tensors, None, self.field_index[field], cond=batch_cond)
        return torch.log_softmax(logits.float() / temperature, dim=-1)

    @torch.no_grad()
    def decode(
        self,
        start: Tensor,
        sg_number: int,
        target: CompositionTarget,
        n_candidates: int = 64,
        strategy: str = "sample",
        beam_width: Optional[int] = None,
        temperature: float = 1.0,
        cond: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> List[CSPCandidate]:
        """Decode genes with exactly `target`'s composition in one space group.

        Args:
            start: The space group start token, shape ``[1]`` or ``[1, d]``.
            sg_number: The real space group number the start token encodes.
            target: The conventional-cell composition every gene must have.
            n_candidates: Genes to return (``sample``), or paths to carry
                (``beam``, where it also defaults `beam_width`).
            strategy: ``"sample"`` or ``"beam"``.
            beam_width: Paths kept per cascade field under ``beam``.
            temperature: Softmax temperature; below 1 sharpens.
            cond: Conditioning vector, shape ``[1, condition_dim]``. For CSP this
                carries ``Delta_E_polymorph = 0`` or the hull-target energy.
            generator: RNG for reproducible sampling.

        Returns:
            Completed candidates, most likely first.  Every one has the target
            composition; the list is shorter than requested only if the space
            group cannot express it at all.
        """
        if strategy not in ("sample", "beam"):
            raise ValueError(f"Unknown strategy: {strategy}")
        combinatorics = self.combinatorics(sg_number, start)
        width = beam_width or n_candidates
        self.model.eval()

        if not combinatorics.can_complete(list(target.counts), frozenset(), self.max_sequence_len):
            logger.info("Space group %d cannot build %s", sg_number, target)
            return []

        live = [BeamState(deficits=list(target.counts))
                for _ in range(n_candidates if strategy == "sample" else 1)]
        finished: List[BeamState] = []

        for site in range(self.max_sequence_len):
            if not live:
                break
            proposals = self._choose_elements(
                start, live, site, target, combinatorics, temperature, cond, strategy,
                width, generator)
            proposals = self._choose_field(
                start, proposals, site, "site_symmetries", combinatorics, temperature, cond,
                strategy, width, generator)
            proposals = self._choose_field(
                start, proposals, site, "sites_enumeration", combinatorics, temperature, cond,
                strategy, width, generator)
            live = []
            for proposal in proposals:
                position = combinatorics.by_key[(proposal.ss_token, proposal.enum_token)]
                state = proposal.state.place(
                    proposal.element_index, position, proposal.element_token,
                    proposal.log_prob_delta)
                (finished if state.finished else live).append(state)
            if strategy == "beam" and len(live) > width:
                live.sort(key=lambda s: s.log_prob, reverse=True)
                live = live[:width]

        if live:
            logger.info("%d paths hit the %d-site cap without completing %s in space group %d",
                        len(live), self.max_sequence_len, target, sg_number)
        candidates = [
            CSPCandidate(sg_number=sg_number, rows=tuple(state.rows),
                         log_prob=state.log_prob, n_sites=state.n_sites)
            for state in finished]
        candidates.sort(key=lambda c: c.log_prob, reverse=True)
        return candidates[:n_candidates]

    def _choose_elements(
        self, start, live, site, target, combinatorics, temperature, cond, strategy,
        width, generator) -> List["_Proposal"]:
        """Pick which element receives the site, over elements that can still take one."""
        seeds = [_Proposal(state=state) for state in live]
        log_probs = self._stage_log_probs(
            start, seeds, site, "elements", combinatorics, temperature, cond)
        options = []
        for row, state in enumerate(live):
            allowed_indices = state.live_elements(combinatorics, self.max_sequence_len)
            if not allowed_indices:
                continue
            allowed_tokens = [target.element_tokens[i] for i in allowed_indices]
            masked = self._masked(log_probs[row], allowed_tokens)
            options.append((row, state, allowed_indices, allowed_tokens, masked))
        return self._resolve(
            options, strategy, width, generator,
            build=lambda state, token, index_map, delta: _Proposal(
                state=state, element_index=index_map[token], element_token=token,
                log_prob_delta=delta),
            index_maps=[dict(zip(tokens, indices))
                        for _, _, indices, tokens, _ in options])

    def _choose_field(
        self, start, proposals, site, field, combinatorics, temperature, cond, strategy,
        width, generator) -> List["_Proposal"]:
        """Pick a site symmetry, then an enumeration, among positions still legal."""
        if not proposals:
            return []
        log_probs = self._stage_log_probs(
            start, proposals, site, field, combinatorics, temperature, cond)
        options = []
        for row, proposal in enumerate(proposals):
            legal = proposal.state.legal_positions(
                proposal.element_index, combinatorics, self.max_sequence_len)
            if field == "sites_enumeration":
                legal = [p for p in legal if p.ss_token == proposal.ss_token]
            allowed = sorted({p.ss_token if field == "site_symmetries" else p.enum_token
                              for p in legal})
            if not allowed:
                continue
            options.append((row, proposal, None, allowed, self._masked(log_probs[row], allowed)))
        return self._resolve(
            options, strategy, width, generator,
            build=lambda proposal, token, _, delta: proposal.with_token(field, token, delta),
            index_maps=[None] * len(options))

    @staticmethod
    def _resolve(options, strategy, width, generator, build, index_maps):
        """Sample one continuation per path, or keep the `width` best across all of them."""
        if strategy == "sample":
            resolved = []
            for (_, carrier, _, _, masked), index_map in zip(options, index_maps):
                token = int(torch.multinomial(
                    masked.exp(), num_samples=1, generator=generator).item())
                resolved.append(build(carrier, token, index_map, float(masked[token])))
            return resolved
        scored = []
        for (_, carrier, _, allowed, masked), index_map in zip(options, index_maps):
            # The element stage carries raw states, the later stages carry proposals that
            # already hold this row's partial log-probability.
            base = carrier.log_prob if isinstance(carrier, BeamState) \
                else carrier.state.log_prob + carrier.log_prob_delta
            for token in allowed:
                delta = float(masked[token])
                scored.append((base + delta, carrier, token, index_map, delta))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [build(carrier, token, index_map, delta)
                for _, carrier, token, index_map, delta in scored[:width]]


@dataclass
class _Proposal:
    """A path partway through one site's cascade."""
    state: BeamState
    element_index: Optional[int] = None
    element_token: Optional[int] = None
    ss_token: Optional[int] = None
    enum_token: Optional[int] = None
    log_prob_delta: float = 0.0

    def with_token(self, field: str, token: int, delta: float) -> "_Proposal":
        updated = _Proposal(
            state=self.state, element_index=self.element_index,
            element_token=self.element_token, ss_token=self.ss_token,
            enum_token=self.enum_token, log_prob_delta=self.log_prob_delta + delta)
        setattr(updated, "ss_token" if field == "site_symmetries" else "enum_token", token)
        return updated


def candidates_to_tensors(
    candidates: Sequence[CSPCandidate],
    start: Tensor,
    combinatorics: SpaceGroupCombinatorics,
    cascade_order: Sequence[str],
    pads: Dict,
    stops: Dict,
    augmented_fields: Optional[Sequence[str]] = None,
    include_stop: bool = True,
    sequence_length: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Tensor]:
    """Pack decoded genes into the tokenised form a trained model reads.

    Sites are written in decoding order, then a STOP if the model was trained
    with one, then PAD to a common width -- the layout the dataset derives its
    padding mask from.

    Args:
        candidates: Genes to pack; they need not share a length.
        start: The space group start token, shape ``[1]`` or ``[1, d]``.
        combinatorics: Tables for their space group, for the multiplicity field.
        cascade_order: Field order of the *reading* model, which need not be the
            decoder's as long as the two share a tokeniser.
        pads, stops: Per-field service tokens, as `WyckoffTrainer` holds them.
        augmented_fields: Fields the reader expects an augmented variant of. One
            variant is supplied, the decoded gene itself.
        include_stop: Whether the reader's training data carried a STOP token.
        sequence_length: Padded width; defaults to what the genes need.
        device: Defaults to `start`'s.

    Returns:
        A dict suitable for `WyckoffTrainer.predict_scalars`.
    """
    device = device or start.device
    n_sites = max((c.n_sites for c in candidates), default=0)
    width = sequence_length or (n_sites + (1 if include_stop else 0))
    if width < n_sites + (1 if include_stop else 0):
        raise ValueError(
            f"sequence_length {width} cannot hold a {n_sites}-site gene"
            + (" plus its STOP" if include_stop else ""))

    tensors: Dict[str, Tensor] = {}
    for name in cascade_order:
        tensors[name] = torch.full((len(candidates), width), int(pads[name]),
                                   dtype=torch.int64, device=device)
    for row, candidate in enumerate(candidates):
        for site, (element, ss, enum) in enumerate(candidate.rows):
            values = {"elements": element, "site_symmetries": ss, "sites_enumeration": enum,
                      "multiplicity": combinatorics.by_key[(ss, enum)].multiplicity}
            for name in cascade_order:
                if name not in values:
                    raise NotImplementedError(f"Cannot pack cascade field {name!r}")
                tensors[name][row, site] = values[name]
        if include_stop:
            for name in cascade_order:
                tensors[name][row, candidate.n_sites] = int(stops[name])

    data: Dict[str, Tensor] = dict(tensors)
    for name in augmented_fields or ():
        # One variant per example: the gene as decoded. The reader averages over
        # variants, and with a single one that average is the gene itself.
        data[f"{name}_augmented"] = [[tensors[name][row]] for row in range(len(candidates))]
    data["pure_sequence_length"] = torch.tensor(
        [c.n_sites for c in candidates], dtype=torch.int64, device=device)
    start_row = start.reshape(1, -1) if start.dim() > 1 else start.reshape(1)
    data["start"] = start_row.expand(len(candidates), *start_row.shape[1:]).clone()
    return data


def rank_by_predicted_minimum(
    candidates: Sequence[CSPCandidate],
    trainer,
    start: Tensor,
    combinatorics: SpaceGroupCombinatorics,
    augmentation_samples: int = 1,
) -> List[CSPCandidate]:
    """Order candidates by a censored regressor's estimate of ``min(E | gene)``.

    This is the selection step of CSP mode.  The regressor must have been fitted
    with ``scalar_loss="censored"``: an MSE-fitted one predicts ``E[E | gene]``,
    which ranks genes by the average structure on their manifold rather than by
    the best, and so systematically prefers genes with little positional freedom
    -- exactly the ones whose average and best coincide.

    Returns a new list, lowest predicted energy first, with `predicted_energy`
    filled in.  The input order (model likelihood) is not consulted; blend the
    two yourself via `CSPCandidate.normalised_log_prob` if you want both.
    """
    if not candidates:
        return []
    if getattr(trainer, "scalar_loss", "mse") != "censored":
        logger.warning(
            "Ranking with a %s regressor: it estimates the mean energy of the gene's "
            "structures, not the minimum, which biases selection towards low-dof genes.",
            getattr(trainer, "scalar_loss", "mse"))
    include_stop = True
    if trainer.tokeniser_config is not None:
        include_stop = bool(trainer.tokeniser_config.get("include_stop", True))
    data = candidates_to_tensors(
        candidates, start=start, combinatorics=combinatorics,
        cascade_order=trainer.cascade_order, pads=trainer.pad_dict, stops=trainer.stops_dict,
        augmented_fields=trainer.augmented_fields, include_stop=include_stop,
        sequence_length=trainer.max_sequence_length, device=trainer.device)
    data[trainer.start_name] = data.pop("start")
    predictions, _ = trainer.predict_scalars(data, augmentation_samples=augmentation_samples)
    ranked = []
    for candidate, energy in zip(candidates, predictions.reshape(-1).tolist()):
        ranked.append(CSPCandidate(
            sg_number=candidate.sg_number, rows=candidate.rows, log_prob=candidate.log_prob,
            n_sites=candidate.n_sites, predicted_energy=float(energy)))
    ranked.sort(key=lambda c: c.predicted_energy)
    return ranked


def candidates_to_pyxtal(
    candidates: Sequence[CSPCandidate],
    trainer,
    start: Tensor,
    combinatorics: SpaceGroupCombinatorics,
) -> List[Optional[dict]]:
    """Convert decoded genes into the pyxtal dicts the rest of the pipeline reads.

    Same shape as `WyckoffTrainer.generate_structures` returns, so CSP output
    goes straight into ``wyformer-cryspr`` and ``wyformer-protocol``.  Entries
    the processor rejects come back as None and should be dropped by the caller;
    with the composition mask in place that should not happen, and it is worth
    noticing when it does.
    """
    from functools import partial  # noqa: PLC0415

    from wyckoff_transformer.tokenization import get_wp_index, load_wyckoff_mappings  # noqa: PLC0415

    if not candidates:
        return []
    data = candidates_to_tensors(
        candidates, start=start, combinatorics=combinatorics,
        cascade_order=trainer.cascade_order, pads=trainer.pad_dict, stops=trainer.stops_dict,
        include_stop=True, device=torch.device("cpu"))
    stacked = torch.stack([data[name] for name in trainer.cascade_order], dim=-1)
    to_pyxtal = partial(
        trainer.processor.tensor_to_pyxtal,
        cascade_order=trainer.cascade_order,
        letter_from_ss_enum_idx=trainer.tokenisers["sites_enumeration"].get_letter_from_ss_enum_idx(
            trainer.run_path),
        ss_from_letter=load_wyckoff_mappings(trainer.run_path).ss_from_letter,
        wp_index=get_wp_index())
    starts = data["start"].cpu()
    return [to_pyxtal(starts[row], stacked[row]) for row in range(len(candidates))]
