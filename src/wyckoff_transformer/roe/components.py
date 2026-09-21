"""What a rules-of-engagement mode is assembled from.

Four kinds of component, each a ``Protocol`` rather than a base class so that an
adapter over existing code is a component by having the right methods, not by
inheriting from anything:

``SystemSampler``
    Chooses what to aim at: a chemical system and a space group per structure.
    Only the modes that select a target before generating use one.

``GeneSource``
    Produces Wyckoff genes.  WyFormer is one; a gene file on disk is another,
    which is how a cohort already drawn can be re-filtered without re-sampling.

``GeneFilter``
    Narrows a cohort, and annotates every gene it looked at.  Filters compose by
    intersection and must only ever narrow what they were given
    (:class:`~wyckoff_transformer.roe.cohort.ResurrectionError`).

``Reconstructor``
    Turns genes into structures and scores them -- CrySPR today.  This is the
    expensive one, and every filter exists to spend less of it.

A component also declares what it needs, so a mode can be refused before a GPU
hour is spent rather than after: ``requires_plan`` on a source, and
``provides`` / ``requires`` on a filter, which name the cohort columns it writes
and the ones it expects to find.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import pandas as pd

from wyckoff_transformer.roe.cohort import Cohort

#: The kinds, in the order a mode runs them.
STAGE_KINDS = ("sampler", "source", "filter", "reconstructor")


@runtime_checkable
class SystemSampler(Protocol):
    """Draws the (chemical system, space group) request for each structure."""

    name: str

    def plan(self, n_structures: int) -> Any:
        """A :class:`~wyckoff_transformer.system_prior.SystemDraws` of length *n*."""

    def describe(self) -> dict:
        """What went into the plan, for the manifest."""


@runtime_checkable
class GeneSource(Protocol):
    """Produces the magazine: the genes a mode will spend its budget on."""

    name: str
    #: True when the source cannot run without a :class:`SystemSampler`'s plan.
    requires_plan: bool

    def draw(self, n_structures: int, plan: Optional[Any] = None) -> list[dict]:
        """PyXtal-notation genes, in sampling order."""

    def describe(self) -> dict:
        ...


@runtime_checkable
class GeneFilter(Protocol):
    """Narrows a cohort and annotates it.

    ``apply`` mutates *cohort* through
    :meth:`~wyckoff_transformer.roe.cohort.Cohort.record` and returns nothing:
    the cohort is the run's single mutable object, and a filter that returned a
    new one would invite a caller to keep the wrong copy.
    """

    name: str
    #: The mode slot this filter answers for -- see
    #: :data:`~wyckoff_transformer.roe.plan.FILTER_SLOTS`.  A mode names slots,
    #: not implementations, which is what lets one be swapped for another.
    slot: str
    #: Cohort columns this filter writes.
    provides: tuple[str, ...]
    #: Cohort columns it expects to already be there.  Checked before the run.
    requires: tuple[str, ...]

    def apply(self, cohort: Cohort) -> None:
        ...

    def describe(self) -> dict:
        ...


@runtime_checkable
class Reconstructor(Protocol):
    """Turns kept genes into relaxed, scored structures."""

    name: str

    def reconstruct(self, cohort: Cohort, output_dir: Path) -> dict:
        """Run the reconstruction and return its funnel, or ``{}`` if it wrote none."""

    def describe(self) -> dict:
        ...


def check_requirements(filters: Sequence[GeneFilter]) -> None:
    """Refuse a filter chain whose inputs nothing upstream produces.

    Cheap, and it fires before the first model is loaded: a chain that ranks on
    a column no filter writes would otherwise fail after the cohort has been
    generated, which on a conditioned model is the expensive half of a mode that
    has not yet relaxed anything.
    """
    available: set[str] = set()
    for gene_filter in filters:
        missing = [column for column in gene_filter.requires if column not in available]
        if missing:
            raise ValueError(
                f"Filter {gene_filter.name!r} needs cohort column(s) {missing}, which no "
                f"filter before it provides (available: {sorted(available) or 'none'})."
            )
        available.update(gene_filter.provides)


def kept_frame(cohort: Cohort, columns: Sequence[str] = ()) -> pd.DataFrame:
    """The rows a filter should look at: those still kept, and nothing else."""
    frame = cohort.table.loc[cohort.table["kept"]]
    return frame[list(columns)] if columns else frame
