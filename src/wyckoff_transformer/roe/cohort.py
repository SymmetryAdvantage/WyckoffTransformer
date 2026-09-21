"""The cohort of genes a rules-of-engagement run carries through its filters.

One rule governs everything here: **a filter marks, it never drops**.  The
cohort keeps every gene it was born with, at the index it was born at, and each
filter writes its own verdict column plus its contribution to ``kept``.  Two
things follow, and both are the reason the class exists rather than a list that
gets shorter:

* every rate stays expressible per *sampled* gene, which is the denominator the
  ranking protocol reports on and the only one that can be compared across
  modes (``docs/de_novo_ranking_protocol.md``);
* the cost a mode avoided is recoverable.  The whole argument for filtering is
  that a relaxation is expensive and a prediction is not, and that argument
  cannot be checked against a cohort that has forgotten what it discarded.
"""
from __future__ import annotations

import gzip
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: The filtered gene file a reconstructor is handed.
ENGAGED_GENES_FILE = "engaged_genes.json.gz"
#: Per-gene verdicts, one row per *sampled* gene.
COHORT_FILE = "cohort.csv"
#: What produced the cohort and what it passed through.
MANIFEST_FILE = "roe_manifest.json"

#: Columns every cohort has from birth.
#:
#: ``weight`` is how many sampled genes a row stands for.  It is 1 everywhere
#: until a uniqueness screen collapses duplicates onto their representative, at
#: which point the representative carries the count and the duplicates carry 0 --
#: so ``weight`` always sums to the number of sampled genes, whatever has run.
BASE_COLUMNS = ("kept", "weight")


@dataclass
class StageRecord:
    """What one component did to the cohort, for the manifest.

    Attributes:
        stage: The component's kind -- ``sampler``, ``source``, ``filter`` or
            ``reconstructor``.
        component: The component's own name, e.g. ``gene_screen``.
        considered: Rows still kept when the stage ran.  A filter downstream of
            another does not see, and is not charged for, what the first removed.
        passed: Rows still kept when it finished.
        weight_considered: The same in sampled genes rather than rows.
        weight_passed: The same in sampled genes rather than rows.
        seconds: Wall-clock, so a filter's cost can be compared with the
            relaxations it saved.
        detail: Whatever the component wants recorded: thresholds, model ids,
            reference identity.
    """

    stage: str
    component: str
    considered: int
    passed: int
    weight_considered: int
    weight_passed: int
    seconds: float
    detail: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "stage": self.stage,
            "component": self.component,
            "considered": self.considered,
            "passed": self.passed,
            "weight_considered": self.weight_considered,
            "weight_passed": self.weight_passed,
            "seconds": round(self.seconds, 3),
            "detail": self.detail,
        }


class ResurrectionError(RuntimeError):
    """A filter passed a row an earlier one had already removed.

    Filters compose by intersection, so this is always a bug in the filter
    rather than a state the cohort should absorb: a mode's yield would then
    depend on the order its filters happened to run in for reasons other than
    the ones it intends.
    """


@dataclass
class Cohort:
    """Genes, their verdicts, and the record of what produced both.

    Attributes:
        genes: PyXtal-notation genes, in sampling order.  Never reordered or
            truncated; the index into this list is a gene's identity for the
            whole run and is what ``cohort.csv`` is keyed by.
        table: One row per gene.  Filters add columns; :data:`BASE_COLUMNS` are
            always present.
        history: One :class:`StageRecord` per component that ran.
        provenance: What drew the genes -- model, conditioning, plan, seed.
    """

    genes: list[dict]
    table: pd.DataFrame
    history: list[StageRecord] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    @classmethod
    def from_genes(cls, genes: Sequence[dict], provenance: Optional[dict] = None) -> "Cohort":
        genes = list(genes)
        table = pd.DataFrame(
            {
                "kept": np.ones(len(genes), dtype=bool),
                "weight": np.ones(len(genes), dtype=np.int64),
            },
            index=pd.RangeIndex(len(genes), name="index"),
        )
        return cls(genes=genes, table=table, provenance=dict(provenance or {}))

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    @property
    def n_sampled(self) -> int:
        """Genes drawn.  The denominator of every rate a mode reports."""
        return len(self.genes)

    @property
    def kept(self) -> pd.Series:
        return self.table["kept"]

    @property
    def n_kept(self) -> int:
        return int(self.table["kept"].sum())

    @property
    def weight_kept(self) -> int:
        """Sampled genes the kept rows stand for.

        Equal to :attr:`n_kept` until a uniqueness screen runs, and larger
        afterwards: a representative that four sampled genes collapsed onto is
        one relaxation and four draws.
        """
        return int(self.table.loc[self.table["kept"], "weight"].sum())

    def kept_indices(self) -> list[int]:
        return self.table.index[self.table["kept"]].tolist()

    def kept_genes(self) -> list[dict]:
        return [self.genes[i] for i in self.kept_indices()]

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def record(
        self,
        stage: str,
        component: str,
        *,
        passed: Optional[Iterable[int]] = None,
        columns: Optional[Mapping[str, Any]] = None,
        weights: Optional[Mapping[int, int]] = None,
        seconds: float = 0.0,
        detail: Optional[dict] = None,
    ) -> StageRecord:
        """Apply one component's verdict.

        Args:
            stage: The component's kind, for :class:`StageRecord`.
            component: The component's name.
            passed: Indices the component keeps.  ``None`` keeps everything it
                was given, which is what a component that only annotates -- a
                score written for later ranking -- passes.  An index that was
                already removed raises :class:`ResurrectionError`.
            columns: Column name -> values, either a mapping keyed by index or
                anything ``pd.Series`` accepts.  Written for the whole frame,
                with NA where the component had nothing to say.
            weights: Index -> how many sampled genes that row now stands for.
                Only a uniqueness screen sets this.
            seconds: Wall-clock the component took.
            detail: Recorded verbatim in the manifest.

        Returns:
            The record, also appended to :attr:`history`.
        """
        considered = self.table.index[self.table["kept"]]
        weight_considered = int(self.table.loc[considered, "weight"].sum())

        if columns:
            for name, values in columns.items():
                series = values if isinstance(values, pd.Series) else pd.Series(values)
                if series.empty:
                    # Reindexing an empty Series would give the column object dtype and
                    # lose whatever the component meant it to hold.
                    self.table[name] = pd.Series(pd.NA, index=self.table.index, dtype=series.dtype)
                else:
                    self.table[name] = series.reindex(self.table.index)

        if weights is not None:
            weight_column = pd.Series(weights, dtype="int64").reindex(self.table.index)
            # A row the screen said nothing about keeps the weight it had: a screen
            # that only looked at the kept rows must not zero the ones it skipped.
            self.table["weight"] = weight_column.fillna(self.table["weight"]).astype("int64")

        if passed is not None:
            passed_index = pd.Index(sorted(set(int(i) for i in passed)), dtype="int64")
            unknown = passed_index.difference(self.table.index)
            if len(unknown):
                raise KeyError(f"{component} passed indices that are not in the cohort: {list(unknown)[:5]}")
            resurrected = passed_index.difference(considered)
            if len(resurrected):
                raise ResurrectionError(
                    f"{component} passed {len(resurrected)} gene(s) an earlier stage had "
                    f"already removed, e.g. {list(resurrected)[:5]}. Filters compose by "
                    "intersection; a filter must only ever narrow what it was given."
                )
            verdict = pd.Series(False, index=self.table.index)
            verdict.loc[passed_index] = True
            # A row an earlier stage removed gets NA rather than False: this stage
            # did not look at it, and recording a verdict it never reached would
            # make the column unreadable as "what this filter decided".
            looked_at = pd.Series(self.table.index.isin(considered), index=self.table.index)
            self.table[f"passed_{component}"] = (
                verdict.astype("boolean").where(looked_at, pd.NA))
            self.table["kept"] = self.table["kept"] & verdict

        record = StageRecord(
            stage=stage,
            component=component,
            considered=len(considered),
            passed=self.n_kept,
            weight_considered=weight_considered,
            weight_passed=self.weight_kept,
            seconds=seconds,
            detail=dict(detail or {}),
        )
        self.history.append(record)
        logger.info(
            "%s (%s): %d of %d genes kept (%d of %d sampled), %.1fs",
            component, stage, record.passed, record.considered,
            record.weight_passed, record.weight_considered, seconds,
        )
        return record

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def manifest(self, extra: Optional[dict] = None) -> dict:
        return {
            "sampled": self.n_sampled,
            "kept": self.n_kept,
            "kept_weight": self.weight_kept,
            "provenance": self.provenance,
            "stages": [record.summary() for record in self.history],
            **(extra or {}),
        }

    def write(self, output_dir: Path, extra_manifest: Optional[dict] = None) -> dict[str, Path]:
        """Write the filtered gene file, the per-gene table and the manifest.

        The gene file holds only the kept genes, because that is what a
        reconstructor is asked to spend its budget on; ``engaged_index`` in the
        table says where each one landed in it, so a funnel row can always be
        traced back to the sampled gene it came from.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        kept = self.kept_indices()
        engaged = pd.Series(pd.NA, index=self.table.index, dtype="Int64")
        engaged.loc[kept] = range(len(kept))
        self.table["engaged_index"] = engaged

        genes_path = output_dir / ENGAGED_GENES_FILE
        with gzip.open(genes_path, "wt", encoding="utf-8") as handle:
            json.dump([self.genes[i] for i in kept], handle)

        table_path = output_dir / COHORT_FILE
        self.table.to_csv(table_path)

        manifest_path = output_dir / MANIFEST_FILE
        with open(manifest_path, "wt", encoding="utf-8") as handle:
            json.dump(self.manifest(extra_manifest), handle, indent=1)

        logger.info("Wrote %d of %d genes to %s", len(kept), self.n_sampled, genes_path)
        return {"genes": genes_path, "table": table_path, "manifest": manifest_path}


def timed():
    """``with timed() as elapsed:`` -- ``elapsed()`` gives the seconds so far."""
    class _Timer:
        def __enter__(self):
            self._start = time.time()
            return self

        def __exit__(self, *exc):
            self._seconds = time.time() - self._start
            return False

        def __call__(self) -> float:
            return getattr(self, "_seconds", time.time() - self._start)

    return _Timer()
