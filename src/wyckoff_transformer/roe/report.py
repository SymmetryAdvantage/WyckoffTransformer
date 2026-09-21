"""What a mode cost and what it found, restated per sampled gene.

The protocol reports rates against the cohort it was handed.  A mode filters
before handing one over, so those rates answer "how good were the genes that got
through", which is not the question a mode is chosen on.  This module restates
them against the genes the mode *drew*, and puts the reconstruction budget next
to them, because the only reason to filter is that a reconstruction costs more
than a prediction.

The restatement re-uses
:func:`~wyckoff_transformer.evaluation.protocol.funnel_structure_metrics`
verbatim, with a :class:`~wyckoff_transformer.evaluation.protocol.GeneScreen`
built from the cohort's own weights.  MetaSUN must mean exactly what it means in
``docs/de_novo_ranking_protocol.md``; a second implementation of the masks would
eventually mean something slightly else.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from wyckoff_transformer.evaluation.protocol import GeneScreen, funnel_structure_metrics
from wyckoff_transformer.roe.cohort import Cohort

logger = logging.getLogger(__name__)

REPORT_FILE = "engagement.json"


def _read(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, index_col=0)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s (%s)", path, exc)
        return None


def cohort_screen(cohort: Cohort) -> GeneScreen:
    """A ``GeneScreen`` over the *engaged* genes, weighted by the cohort.

    ``counts`` is keyed by the row's position in the engaged gene file, which is
    what the protocol's outputs are indexed by, and valued by how many sampled
    genes that row stands for.  ``n_sampled`` is the whole cohort.  Feeding those
    two to the protocol's own metric function is what turns its per-engaged-gene
    rates into per-sampled-gene ones.
    """
    if "engaged_index" not in cohort.table.columns:
        raise ValueError(
            "The cohort has not been written yet, so nothing knows which engaged row "
            "each gene became. Call Cohort.write first.")
    engaged = cohort.table.loc[cohort.table["kept"], ["engaged_index", "weight"]]
    counts = {
        int(row.engaged_index): int(row.weight)
        for row in engaged.itertuples()
        if pd.notna(row.engaged_index)
    }
    return GeneScreen(
        n_sampled=cohort.n_sampled,
        valid=sorted(counts),
        counts=counts,
    )


def cohort_gene_metrics(cohort: Cohort) -> dict:
    """The gene-level funnel as the *mode* saw it, per gene drawn.

    Distinct from ``protocol_funnel["gene"]``, which is the protocol's own
    screen re-run on the genes the mode handed over -- an audit of the survivors,
    so for a mode with a screen its uniqueness and novelty are near 1.0 by
    construction. This is the view where those numbers still mean something:
    the denominator is every gene drawn, and a mode that never screened reports
    ``None`` rather than a number it did not measure.
    """
    table = cohort.table
    sampled = cohort.n_sampled
    valid = int(table["passed_validity"].fillna(False).astype(bool).sum()) \
        if "passed_validity" in table.columns else None

    if "unique_representative" in table.columns:
        representatives = table["unique_representative"].fillna(False).astype(bool)
        unique_gene = int(representatives.sum())
    else:
        representatives, unique_gene = None, None

    novel = gene_known = sampled_novel = sampled_known = None
    if representatives is not None and "gene_novel" in table.columns:
        is_novel = table["gene_novel"].fillna(False).astype(bool)
        novel = int((representatives & is_novel).sum())
        gene_known = int((representatives & ~is_novel).sum())
        weights = table["duplicates"].fillna(0).astype("int64")
        sampled_novel = int(weights[representatives & is_novel].sum())
        sampled_known = int(weights[representatives & ~is_novel].sum())

    def ratio(numerator, denominator):
        if numerator is None or not denominator:
            return None
        return numerator / denominator

    return {
        "sampled": sampled,
        "valid_gene": valid,
        "unique_gene": unique_gene,
        "gene_novel": novel,
        "gene_known": gene_known,
        "sampled_novel": sampled_novel,
        "sampled_known": sampled_known,
        "valid_gene_rate": ratio(valid, sampled),
        "unique_gene_rate": ratio(unique_gene, sampled),
        "gene_novelty_rate": ratio(novel, unique_gene),
        "engaged": cohort.n_kept,
        "engaged_rate": ratio(cohort.n_kept, sampled),
    }


def reconstruction_cost(
    cohort: Cohort,
    relaxations: Optional[pd.DataFrame],
    charge_duplicates: bool,
) -> dict:
    """Trials run, and trials a faithful campaign at this mode would have paid for.

    They differ for exactly one mode.  ``broadside`` runs no uniqueness screen,
    so a campaign at those rules would have reconstructed every duplicate as
    though it were a new gene; the protocol nonetheless deduplicates, because
    that is what it does.  Charging the duplicates recovers the cost without
    spending it.

    That makes the cost exact and the *yield* a lower bound: the duplicate draws
    would have been extra reconstruction trials of the same gene, and extra
    trials sometimes find a lower minimum (``docs/de_novo_ranking_protocol.md``,
    the trial schedule).  So ``broadside``'s true yield is at or above what is
    reported here, and its advantage over the other modes -- which is none --
    is not overstated by this choice.
    """
    trials_run = None if relaxations is None else int(len(relaxations))
    if relaxations is None or "index" not in relaxations.reset_index().columns:
        per_gene = None
    else:
        reset = relaxations.reset_index()
        column = "index" if "index" in reset.columns else reset.columns[0]
        per_gene = reset.groupby(column).size()

    charged = trials_run
    if charge_duplicates and per_gene is not None:
        weights = cohort_screen(cohort).counts
        charged = int(sum(count * weights.get(int(gene), 1) for gene, count in per_gene.items()))

    return {
        "trials_run": trials_run,
        "trials_charged": charged,
        "charge_duplicates": charge_duplicates,
        "genes_reconstructed": None if per_gene is None else int(len(per_gene)),
    }


def engagement_report(
    cohort: Cohort,
    roe,
    output_dir: Path,
    protocol_subdir: str = "protocol",
) -> dict:
    """Assemble the mode's own report from the cohort and the protocol outputs.

    Everything is reported twice where the two differ: as the protocol computed
    it, per gene the protocol was given, and restated per gene the mode drew.
    A mode is chosen on the second and debugged on the first.
    """
    output_dir = Path(output_dir)
    protocol_dir = output_dir / protocol_subdir

    structures = _read(protocol_dir / "structures.csv")
    structures_fixed = _read(protocol_dir / "structures_fixed_symmetry.csv")
    relaxations = _read(protocol_dir / "relaxations.csv")

    protocol_funnel = None
    funnel_path = protocol_dir / "funnel.json"
    if funnel_path.is_file():
        with open(funnel_path, "rt", encoding="utf-8") as handle:
            protocol_funnel = json.load(handle)

    per_sampled = {"gene": cohort_gene_metrics(cohort)}
    if structures is not None:
        screen = cohort_screen(cohort)
        per_sampled["free"] = funnel_structure_metrics(screen, structures)
        if structures_fixed is not None:
            per_sampled["fixed_symmetry"] = funnel_structure_metrics(screen, structures_fixed)

    cost = reconstruction_cost(cohort, relaxations, roe.charge_duplicates)

    report = {
        "rules_of_engagement": roe.name,
        "summary": roe.summary,
        "cohort": {
            "sampled": cohort.n_sampled,
            "engaged": cohort.n_kept,
            "engaged_weight": cohort.weight_kept,
            "filter_seconds": round(
                sum(record.seconds for record in cohort.history
                    if record.stage in ("filter", "validity")), 3),
        },
        "stages": [record.summary() for record in cohort.history],
        "cost": cost,
        "per_sampled_gene": per_sampled,
        "protocol_funnel": protocol_funnel,
    }

    # Rounds per hit: the number the whole escalation exists to move.
    for section, metrics in per_sampled.items():
        if section == "gene":
            continue
        hits = {
            "metasun": metrics.get("metastable_among_novel"),
            "sun": metrics.get("stable_among_novel"),
        }
        charged = cost.get("trials_charged")
        report.setdefault("trials_per_hit", {})[section] = {
            label: (charged / count if charged and count else None)
            for label, count in hits.items()
        }
        report["trials_per_hit"][section].update(
            {f"{label}_hits": count for label, count in hits.items()})

    return report


def write_report(report: dict, output_dir: Path) -> Path:
    path = Path(output_dir) / REPORT_FILE
    with open(path, "wt", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    return path
