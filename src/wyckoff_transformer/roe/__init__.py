"""Rules of engagement: the inference modes WyFormer generates under.

A mode is an ordered composition of four kinds of component -- a chemical-system
sampler, a gene source, zero or more gene filters, and a reconstructor -- over a
:class:`~wyckoff_transformer.roe.cohort.Cohort` that is never re-indexed and
never has rows removed.  The four named modes are in
:mod:`wyckoff_transformer.roe.plan`; what they are for is in
``docs/rules_of_engagement.md``.

Nothing here re-implements a screen, a generator or a relaxation.  Every
component in :mod:`wyckoff_transformer.roe.builtin` is an adapter over code that
already exists and is already tested, and the value this package adds is the
composition, the per-sampled-gene accounting, and the manifest that says which
components a cohort actually passed through.
"""
from wyckoff_transformer.roe.cohort import Cohort, StageRecord
from wyckoff_transformer.roe.components import (
    GeneFilter,
    GeneSource,
    Reconstructor,
    SystemSampler,
)
from wyckoff_transformer.roe.plan import (
    RULES_OF_ENGAGEMENT,
    Engagement,
    RulesOfEngagement,
    resolve,
)

__all__ = [
    "RULES_OF_ENGAGEMENT",
    "Cohort",
    "Engagement",
    "GeneFilter",
    "GeneSource",
    "Reconstructor",
    "RulesOfEngagement",
    "StageRecord",
    "SystemSampler",
    "resolve",
]
