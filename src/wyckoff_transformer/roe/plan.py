"""The four rules of engagement, and the assembled run that executes one.

A mode is a *name for an ordering of slots*, not for a set of implementations.
``fire-control`` says "screen, then the predicted hull filter, then reconstruct";
which screen and which energy predictor fill those slots is the caller's choice,
and swapping one for another does not change what the mode means or what it can
be compared against.

Where a quantity already has a name in the literature -- ``e_hull``, novelty,
uniqueness, formation energy, SUN, MetaSUN -- it keeps it.  The naval vocabulary
names the one thing that had no name: the modes themselves.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd

from wyckoff_transformer.roe.cohort import Cohort, timed
from wyckoff_transformer.roe.components import (
    GeneFilter,
    GeneSource,
    Reconstructor,
    SystemSampler,
    check_requirements,
)

logger = logging.getLogger(__name__)

#: Slots a mode can name, and what belongs in each.
#:
#: ``screen``
#:     Uniqueness and novelty against the reference archive.  A free lookup.
#: ``energy``
#:     A predicted formation energy compared with the hull at the gene's
#:     composition.  One forward pass per gene.
#: ``surprisal``
#:     The generator's own log-density, the reference-free novelty estimator of
#:     ``docs/generative_novelty_screen.md``.  No mode uses it by default; it is
#:     a slot so that an arm which does is a mode rather than a patch.
FILTER_SLOTS = ("screen", "energy", "surprisal")

#: Structural validity is not a slot.  A formally illegal gene is not a gene the
#: reconstructor could draw if it wanted to -- PyXtal has nothing to place -- so
#: every mode drops it, and none gets credit for doing so.  It is reported, never
#: chosen.
ALWAYS_APPLIED = ("validity",)


@dataclass(frozen=True)
class RulesOfEngagement:
    """One mode: which slots run, in which order, and how cost is charged.

    Attributes:
        name: The mode's name, as the CLI takes it.
        summary: One line, for ``wyformer-roe list``.
        filters: Slot names in the order they run.  Order is part of the mode:
            ``fire-control`` screens before it ranges and ``torpedo-run`` ranges
            before it screens, which is a different experiment even with
            identical components.
        needs_sampler: Whether a target is chosen before generating.
        charge_duplicates: Whether the cost accounting charges the mode for the
            duplicate genes it never noticed were duplicates.  True only where no
            uniqueness screen runs; see
            :func:`~wyckoff_transformer.roe.report.engagement_report`.
        rationale: Why the mode exists, for the manifest and the docs.
    """

    name: str
    summary: str
    filters: tuple[str, ...]
    needs_sampler: bool
    charge_duplicates: bool
    rationale: str = ""

    def __post_init__(self) -> None:
        unknown = [slot for slot in self.filters if slot not in FILTER_SLOTS]
        if unknown:
            raise ValueError(f"{self.name}: unknown filter slot(s) {unknown}; known: {list(FILTER_SLOTS)}")
        if len(set(self.filters)) != len(self.filters):
            raise ValueError(f"{self.name}: a slot is named more than once: {self.filters}")


BROADSIDE = RulesOfEngagement(
    name="broadside",
    summary="Generate and reconstruct. No screen, no ranging.",
    filters=(),
    needs_sampler=False,
    charge_duplicates=True,
    rationale=(
        "The baseline every other mode is measured against, and the only one whose "
        "reconstruction budget is spent exactly as the generator drew it -- duplicates "
        "included. It is what a mode costs when a relaxation is assumed cheap."
    ),
)

FIRE_DISCIPLINE = RulesOfEngagement(
    name="fire-discipline",
    summary="Screen for uniqueness and novelty, then reconstruct.",
    filters=("screen",),
    needs_sampler=False,
    charge_duplicates=False,
    rationale=(
        "A lookup against the reference archive costs nothing and removes the two "
        "kinds of wasted reconstruction that need no model to recognise: the same "
        "gene twice, and a gene the archive already holds."
    ),
)

FIRE_CONTROL = RulesOfEngagement(
    name="fire-control",
    summary="Screen, then keep only genes predicted below the hull, then reconstruct.",
    filters=("screen", "energy"),
    needs_sampler=False,
    charge_duplicates=False,
    rationale=(
        "Screening first and ranking second is the ordering that worked in "
        "docs/generative_novelty_screen.md: an energy ranker left to itself finds "
        "low-lying genes partly by finding compositions the archive already holds, "
        "so the overlap is removed before the energy has a chance to reward it."
    ),
)

TORPEDO_RUN = RulesOfEngagement(
    name="torpedo-run",
    summary="Sample a chemical system, generate into it, range, screen, reconstruct.",
    filters=("energy", "screen"),
    needs_sampler=True,
    charge_duplicates=False,
    rationale=(
        "The only mode that chooses what to aim at. The chemical system is sampled "
        "first and the generator is conditioned on it, so the cohort is spread over "
        "the systems a campaign cares about rather than over the ones the training "
        "corpus was largest in. Ranging runs before the screen here because the "
        "predicted hull is the cheaper of the two once the reference fingerprints "
        "are not already resident -- and because within a named system most genes "
        "are novel, so the screen has less to remove and less reason to run first."
    ),
)

#: Every mode by name.
RULES_OF_ENGAGEMENT: dict[str, RulesOfEngagement] = {
    roe.name: roe for roe in (BROADSIDE, FIRE_DISCIPLINE, FIRE_CONTROL, TORPEDO_RUN)
}


def resolve(name: str) -> RulesOfEngagement:
    """The mode called *name*, or a ``ValueError`` naming the ones that exist."""
    try:
        return RULES_OF_ENGAGEMENT[name]
    except KeyError:
        raise ValueError(
            f"No rules of engagement called {name!r}. Known modes: "
            f"{', '.join(RULES_OF_ENGAGEMENT)}."
        ) from None


@dataclass
class Engagement:
    """A mode with its slots filled: the runnable object.

    Attributes:
        roe: Which mode this is.
        source: What produces the genes.
        filters: Slot name -> the filter filling it.  Must be exactly the slots
            the mode names.
        sampler: Required by, and only accepted by, a mode that chooses a target.
        reconstructor: Omitted to stop at the filtered gene file, which is what a
            run that will reconstruct on another machine wants.
    """

    roe: RulesOfEngagement
    source: GeneSource
    filters: Mapping[str, GeneFilter] = field(default_factory=dict)
    sampler: Optional[SystemSampler] = None
    reconstructor: Optional[Reconstructor] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Refuse an assembly before anything expensive has run."""
        named = tuple(self.roe.filters)
        given = tuple(self.filters)
        missing = [slot for slot in named if slot not in given]
        extra = [slot for slot in given if slot not in named]
        if missing:
            raise ValueError(
                f"{self.roe.name} runs the {list(named)} filter(s); nothing was given for "
                f"{missing}.")
        if extra:
            raise ValueError(
                f"{self.roe.name} runs the {list(named)} filter(s), so {extra} has no slot "
                f"in it. A mode is defined by which filters it runs -- pick the mode that "
                f"names them, or define a new one.")
        for slot, gene_filter in self.filters.items():
            declared = getattr(gene_filter, "slot", slot)
            if declared != slot:
                raise ValueError(
                    f"{gene_filter.name!r} is a {declared!r} filter and was given the "
                    f"{slot!r} slot.")
        if self.roe.needs_sampler and self.sampler is None:
            raise ValueError(
                f"{self.roe.name} chooses a target before generating, so it needs a "
                "chemical-system sampler.")
        if not self.roe.needs_sampler and self.sampler is not None:
            raise ValueError(
                f"{self.roe.name} does not choose a target; a sampler would make it a "
                f"{TORPEDO_RUN.name}.")
        if getattr(self.source, "requires_plan", False) and self.sampler is None:
            raise ValueError(
                f"The gene source {self.source.name!r} generates into a planned chemical "
                "system and was given no sampler to plan one.")
        check_requirements(self.ordered_filters())
        self.rank_column()

    def ordered_filters(self) -> list[GeneFilter]:
        return [self.filters[slot] for slot in self.roe.filters]

    def rank_column(self) -> Optional[str]:
        """The column the budget ranks on, when a filter left the cut to it."""
        columns = [column for gene_filter in self.ordered_filters()
                   if (column := getattr(gene_filter, "rank_column", None))]
        if len(columns) > 1:
            raise ValueError(f"More than one filter asks the budget to rank: {columns}")
        return columns[0] if columns else None

    def describe(self) -> dict:
        return {
            "rules_of_engagement": self.roe.name,
            "filters": list(self.roe.filters),
            "always_applied": list(ALWAYS_APPLIED),
            "charge_duplicates": self.roe.charge_duplicates,
            "components": {
                "sampler": self.sampler.describe() if self.sampler is not None else None,
                "source": self.source.describe(),
                **{f"filter:{slot}": self.filters[slot].describe() for slot in self.roe.filters},
                "reconstructor": (
                    self.reconstructor.describe() if self.reconstructor is not None else None),
            },
        }

    # ------------------------------------------------------------------ #
    def _draw(self, n_structures: int) -> tuple[list[dict], Optional[dict], float]:
        """One batch: plan it if the mode aims, then draw it."""
        attempts = getattr(self.source, "attempts", None)
        n_attempts = attempts(n_structures) if attempts is not None else n_structures

        plan = None
        plan_record = None
        elapsed_planning = 0.0
        if self.sampler is not None:
            with timed() as elapsed:
                # The sampler plans one request per *draw*, not per gene wanted:
                # formal validity removes some of what the generator emits, so a
                # plan the size of the target would come back short.
                plan = self.sampler.plan(n_attempts)
            elapsed_planning = elapsed()
            plan_record = {
                "stage": "sampler", "component": self.sampler.name,
                "planned": len(plan), "seconds": round(elapsed_planning, 3),
                "detail": self.sampler.describe(),
            }
        with timed() as elapsed:
            genes = self.source.draw(n_structures, plan)
        return genes, plan_record, elapsed()

    def _filter(self, genes: list[dict], plan_record: Optional[dict],
                source_seconds: float) -> Cohort:
        """A fresh cohort over *genes*, run through validity and every slot.

        Rebuilt from scratch on every top-up round rather than filtered
        incrementally.  Uniqueness is a property of the whole cohort -- a gene
        drawn in round three can be a duplicate of one from round one -- so a
        filter that only ever saw the newest batch would report a uniqueness
        rate that improves with the number of rounds.
        """
        from wyckoff_transformer.roe.builtin import apply_validity

        cohort = Cohort.from_genes(genes, provenance=self.describe())
        if plan_record is not None:
            cohort.history.append(_record_from_dict(plan_record, cohort))
        cohort.record(
            "source", self.source.name, seconds=source_seconds,
            detail=self.source.describe())
        # Always, before any slot: a formally illegal gene is not a reconstruction
        # any mode could have attempted, so no mode is credited with skipping it.
        apply_validity(cohort)
        for gene_filter in self.ordered_filters():
            gene_filter.apply(cohort)
        return cohort

    def run(
        self,
        n_structures: int,
        output_dir: Path,
        target_engaged: Optional[int] = None,
        max_rounds: int = 12,
        max_sampled: Optional[int] = None,
    ) -> Cohort:
        """Draw, filter, write, and -- if a reconstructor was given -- reconstruct.

        Args:
            n_structures: Genes to draw in the first round.
            output_dir: Where the cohort, the manifest and the reconstruction go.
            target_engaged: Keep drawing until this many genes survive every
                filter, which is what makes two modes comparable: the
                reconstruction is the expensive part and the budget, not the
                draw, is what a campaign is limited by.  ``None`` draws once.
            max_rounds: Give up after this many top-up rounds rather than
                drawing forever against a target the filters will not reach.
            max_sampled: Give up once this many genes have been drawn.  Defaults
                to 50x the target, which is far past any sensible pass rate.

        The cohort is written *before* the reconstruction, not after: the
        reconstruction is hours and can fail on hardware, and what the mode
        selected is the part that must survive that.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        genes: list[dict] = []
        plan_record = None
        source_seconds = 0.0
        rounds = 0
        ceiling = max_sampled if max_sampled is not None else (
            50 * target_engaged if target_engaged else None)

        while True:
            rounds += 1
            want = n_structures if rounds == 1 else self._top_up_size(
                cohort, target_engaged, n_structures)
            batch, batch_plan, batch_seconds = self._draw(want)
            source_seconds += batch_seconds
            if rounds == 1:
                plan_record = batch_plan
            before = len(genes)
            genes.extend(batch)
            logger.info("Round %d: drew %d genes (%d total)", rounds, len(batch), len(genes))
            cohort = self._filter(genes, plan_record, source_seconds)

            if target_engaged is None or cohort.n_kept >= target_engaged:
                break
            if len(genes) == before:
                logger.warning(
                    "The source produced no new genes; stopping at %d engaged of the "
                    "%d asked for.", cohort.n_kept, target_engaged)
                break
            if rounds >= max_rounds:
                logger.warning(
                    "Stopping after %d rounds with %d engaged of the %d asked for.",
                    rounds, cohort.n_kept, target_engaged)
                break
            if ceiling is not None and len(genes) >= ceiling:
                logger.warning(
                    "Stopping at %d genes drawn with %d engaged of the %d asked for.",
                    len(genes), cohort.n_kept, target_engaged)
                break

        rank_column = self.rank_column()
        if target_engaged is not None and cohort.n_kept > target_engaged and rank_column:
            # A filter left the cut to the budget: keep the best-ranked survivors of
            # *every* filter.  Ranking before a screen and cutting there would spend
            # the budget on duplicates and known genes the screen then removes.
            # Every draw was needed to rank, so none is dropped from the cohort.
            kept = cohort.kept_indices()
            values = pd.to_numeric(cohort.table.loc[kept, rank_column], errors="coerce")
            order = values.fillna(float("inf")).sort_values(kind="stable")
            chosen = order.index[:target_engaged]
            finite = values.loc[chosen].dropna()
            cohort.record(
                "filter", "budget", passed=[int(i) for i in chosen],
                detail={"target_engaged": target_engaged, "rank_column": rank_column,
                        "dropped": len(kept) - len(chosen),
                        "selection_cut": float(finite.max()) if len(finite) else None})
        elif target_engaged is not None and cohort.n_kept > target_engaged:
            # Truncate to the budget so every mode reconstructs the same number.
            # The last-drawn survivors go, not a random subset: the cohort is in
            # sampling order and keeping a prefix keeps it a sample.
            #
            # The draws *after* the one that filled the budget are then dropped
            # from the cohort entirely rather than marked, because they are not
            # genes the mode rejected -- they are genes it would never have drawn.
            # Leaving them in would charge a mode for generation it did not need
            # and dilute every per-sampled-gene rate it reports, by 10% for a
            # broadside that drew 1100 to keep 1000.  Re-filtering the prefix is
            # safe: dropping later genes can only remove duplicates of earlier
            # ones, so the first `target_engaged` survivors are the same genes.
            needed = cohort.kept_indices()[target_engaged - 1] + 1
            if needed < len(genes):
                logger.info(
                    "The budget was filled by draw %d of %d; the rest were never needed",
                    needed, len(genes))
                cohort = self._filter(genes[:needed], plan_record, source_seconds)
            surplus = cohort.kept_indices()[target_engaged:]
            if surplus:
                cohort.record(
                    "filter", "budget", passed=cohort.kept_indices()[:target_engaged],
                    detail={"target_engaged": target_engaged, "dropped": len(surplus)})

        written = cohort.write(output_dir, extra_manifest={
            "rounds": rounds, "target_engaged": target_engaged})
        logger.info("Cohort written to %s", written["manifest"])

        if self.reconstructor is not None:
            funnel = self.reconstructor.reconstruct(cohort, output_dir)
            if funnel:
                with open(output_dir / "funnel_from_reconstructor.json", "wt") as handle:
                    json.dump(funnel, handle, indent=1)
        return cohort

    @staticmethod
    def _top_up_size(cohort: Cohort, target_engaged: Optional[int], fallback: int) -> int:
        """How many to draw next, from the pass rate observed so far.

        Deliberately generous (a 30% margin, and never fewer than a tenth of the
        first batch): another round costs a generation pass, and overshooting
        costs nothing because the cohort is truncated to the budget anyway.

        Capped at ten times the first batch, because the estimate divides by a
        pass rate: a mode whose filters have let almost nothing through so far
        would otherwise ask for a batch large enough to exhaust the machine on
        the strength of two or three survivors.
        """
        if target_engaged is None:
            return fallback
        short = target_engaged - cohort.n_kept
        rate = cohort.n_kept / cohort.n_sampled if cohort.n_sampled else 0.0
        # Nothing has passed yet, so there is no rate to extrapolate from: draw
        # another batch the size of the first rather than only the shortfall,
        # which would keep failing for the same reason the first batch did.
        estimate = max(fallback, short) if rate <= 0 else int(round(short / rate * 1.3))
        return max(fallback // 10, min(estimate, 10 * fallback))


def _record_from_dict(record: dict, cohort: Cohort):
    from wyckoff_transformer.roe.cohort import StageRecord

    return StageRecord(
        stage=record["stage"],
        component=record["component"],
        considered=record["planned"],
        passed=record["planned"],
        weight_considered=record["planned"],
        weight_passed=record["planned"],
        seconds=record["seconds"],
        detail=record["detail"],
    )
