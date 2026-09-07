"""Does screening the formula before spending a structure search on it pay off?

The screener ranks *compositions*; the protocol scores *structures*. This joins
the two: take a protocol run, ask the screener what it would have said about each
generated gene's formula before anything was relaxed, and measure whether the
structures it would have kept are more often metastable, stable, or MetaSUN than
the run as a whole.

The comparison is retrospective on purpose. Every gene is relaxed, so the top
slice and the whole set are measured on the same structures and the only thing
that varies is the order -- which is what a prefilter changes. Enrichment is the
rate in the top *k* over the rate in the whole run, so 1.0 is no better than
spending the same budget on a random subset of what the generator produced.

Two energy models meet here and it is worth being explicit. The screener is
fitted on PBE formation energies and scores against the PBE hull; the protocol
relaxes with an MLIP and scores against the matching MLIP hull. Their
disagreement is noise in the join, and noise here can only dilute an enrichment,
never manufacture one.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from wyckoff_transformer.csp import parse_formula
from wyckoff_transformer.formula_energy import screen as screening
from wyckoff_transformer.formula_energy import train as T
from wyckoff_transformer.formula_energy.dataset import DEFAULT_TABLE, formula_key
from wyckoff_transformer.formula_energy.features import SystemDensity

logger = logging.getLogger(__name__)

DEFAULT_ENSEMBLE = Path("runs/formula_energy/ensemble.pt")
DEFAULT_REFERENCE = Path("data/lemat-bulk/lemat_pbe_ehull.csv.gz")

#: Entries this far above the hull or lower are kept as reference; everything
#: else is discarded before any phase diagram is built. A convex hull is defined
#: by the entries *on* it, so this changes no answer, and it takes the reference
#: from 4.7M rows to roughly a hundred thousand -- the difference between
#: building a diagram per chemical system in minutes and in hours.
REFERENCE_TOLERANCE = 0.05

#: The outcomes a screening run is trying to raise, as predicates on a protocol
#: ``structures.csv`` row. ``metastable`` and ``stable`` are the raw energy
#: criteria; the SUN pair additionally demands the structure be valid, unique and
#: novel, which is what the leaderboard counts.
OUTCOMES = {
    "metastable": lambda f: f["e_above_hull"] <= 0.1,
    "stable": lambda f: f["e_above_hull"] <= 0.0,
    "metasun": lambda f: (f["e_above_hull"] <= 0.1) & f["valid_structure"]
                         & f["unique_structure"] & f["novel_structure"],
    "sun": lambda f: (f["e_above_hull"] <= 0.0) & f["valid_structure"]
                     & f["unique_structure"] & f["novel_structure"],
}


def formulas_from_genes(path: Path) -> List[str]:
    """The composition of each gene, straight from the sampled Wyckoff assignment.

    A gene whose reconstruction failed has no formula in ``structures.csv`` -- the
    column describes a structure that does not exist -- but it does have a
    composition, and it did consume a slot of the budget. Reading it here keeps
    those rows in the denominator as the misses they are, instead of quietly
    improving every rate by dropping them.
    """
    import gzip, json  # noqa: PLC0415

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        genes = json.load(handle)
    return [
        "".join(f"{symbol}{count}" for symbol, count in zip(gene["species"], gene["numIons"]))
        for gene in genes
    ]


def reduced_keys(formulas: Sequence[str]) -> List[Optional[str]]:
    """Protocol formulas (cell formulas like ``Al4Mn2Nd12``) to reduced keys.

    Returns ``None`` where the formula is missing or unparseable rather than
    raising, so one failed reconstruction does not take the analysis with it.
    """
    keys: List[Optional[str]] = []
    for text in formulas:
        try:
            keys.append(formula_key(parse_formula(str(text))))
        except (ValueError, TypeError):
            keys.append(None)
    return keys


def hull_energies(
    keys: Sequence[Optional[str]],
    table: Optional[pd.DataFrame] = None,
    reference: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """The PBE hull's formation energy at each composition.

    Looked up in the formula table where the archive already holds the
    composition -- ``e_form - e_hull`` is exactly this and costs nothing -- and
    computed from a phase diagram only for the compositions nobody has computed,
    which is the case a generator is supposed to produce.
    """
    values = np.full(len(keys), np.nan)
    if table is not None:
        known = table["e_hull_at_composition"]
        values = pd.Index(keys).map(known).to_numpy(dtype=float)
    missing = np.isnan(values) & np.array([key is not None for key in keys])
    logger.info("%d of %d formulas are in the archive; %d need a phase diagram",
                int((~missing).sum()), len(keys), int(missing.sum()))
    if missing.any():
        if reference is None:
            raise ValueError("Some formulas are not in the table and no reference was given")
        lookup = screening.HullLookup(reference)
        for position in np.flatnonzero(missing):
            try:
                values[position] = lookup.hull_energy_per_atom(keys[position])
            except (ValueError, KeyError) as problem:
                logger.warning("no hull for %s: %s", keys[position], problem)
    return values


def load_reference(path: Path = DEFAULT_REFERENCE,
                   tolerance: float = REFERENCE_TOLERANCE) -> pd.DataFrame:
    """The hull-defining subset of the archive, in the shape a phase diagram wants."""
    frame = pd.read_csv(
        path, usecols=["immutable_id", "full_formula", "chemsys", "energy_corrected", "e_hull"],
        low_memory=False,
    )
    for column in ("energy_corrected", "e_hull"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["full_formula", "chemsys", "energy_corrected", "e_hull"])
    kept = frame[frame["e_hull"] <= tolerance]
    logger.info("reference: %d of %d rows within %g of the hull", len(kept), len(frame), tolerance)
    return kept.drop(columns=["e_hull"]).set_index("immutable_id")


def score_structures(
    structures: pd.DataFrame,
    models: Sequence,
    table: Optional[pd.DataFrame] = None,
    reference: Optional[pd.DataFrame] = None,
    device: Optional[torch.device] = None,
    genes: Optional[Path] = None,
    density: Optional[SystemDensity] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Attach the screener's verdict to each row of a protocol ``structures.csv``.

    Args:
        genes: The gene file the run was made from. Used to recover the
            composition of rows whose reconstruction failed.
        density: Neighbourhood densities, built from the archive. The only
            provenance channel a never-computed composition can answer.
        feature_names: The provenance features the ensemble expects, from
            :func:`~.train.load_ensemble`.
    """
    from wyckoff_transformer.formula_energy.features import PROVENANCE_FEATURES  # noqa: PLC0415

    feature_names = list(feature_names or PROVENANCE_FEATURES)
    device = device or torch.device("cpu")
    frame = structures.copy()
    keys = reduced_keys(frame["formula"])
    if genes is not None:
        from_genes = reduced_keys(formulas_from_genes(genes))
        missing = [position for position, key in enumerate(keys) if key is None]
        for position in missing:
            gene_index = int(frame["index"].iloc[position])
            if 0 <= gene_index < len(from_genes):
                keys[position] = from_genes[gene_index]
        logger.info("recovered %d formulas from the gene file", len(missing))
    frame["reduced_formula"] = keys
    if frame["reduced_formula"].isna().any():
        logger.warning("%d rows have no usable formula and cannot be ranked",
                       int(frame["reduced_formula"].isna().sum()))
    # Whether the archive already holds this composition. Worth carrying because
    # it is the obvious confound: a screener that merely sorted known formulas
    # from novel ones would show an enrichment for an uninteresting reason, in
    # either direction, and the slice table reports the split so that can be seen.
    frame["formula_known"] = (
        frame["reduced_formula"].isin(table.index) if table is not None else False
    )
    frame["hull"] = hull_energies(frame["reduced_formula"].tolist(), table, reference)

    usable = frame["reduced_formula"].notna().to_numpy()
    frame["location"] = np.nan
    frame["sigma_epistemic"] = np.nan
    keys_usable = frame.loc[usable, "reduced_formula"].tolist()
    system = density.columns(keys_usable) if density is not None else None
    data = T.prepare_formulas(keys_usable, hull=frame.loc[usable, "hull"].to_numpy(),
                              system=system, feature_names=feature_names).to(device)
    predicted = T.predict(models, data)
    frame.loc[usable, "location"] = predicted["location"].to_numpy()
    frame.loc[usable, "sigma_epistemic"] = predicted["sigma_epistemic"].to_numpy()
    # Lower is more promising, in both cases. The adjusted score is Wren's rule:
    # demand that the floor clear the hull by its own uncertainty.
    frame["score_naive"] = frame["location"] - frame["hull"]
    frame["score_adjusted"] = frame["location"] + frame["sigma_epistemic"] - frame["hull"]
    return frame


def wilson_interval(hits: int, trials: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a rate. Honest at the small counts this produces.

    A thousand genes yields on the order of ten stable structures, so a top-slice
    rate is a handful of events and the normal approximation is not usable. The
    interval is what says whether an enrichment above one means anything.
    """
    if trials == 0:
        return (float("nan"), float("nan"))
    rate = hits / trials
    denominator = 1 + z * z / trials
    centre = (rate + z * z / (2 * trials)) / denominator
    spread = z * np.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def slice_table(
    scored: pd.DataFrame,
    score_column: str = "score_adjusted",
    budgets: Sequence[int] = (100, 250, 500),
    novel_only: bool = False,
) -> pd.DataFrame:
    """Rate and enrichment for each outcome among the top *k* by screener score.

    Args:
        novel_only: Restrict to genes whose composition the archive does not
            already hold. Worth reporting separately, because the hull energy at a
            never-computed composition is a different kind of number from the hull
            energy at a computed one -- it is set by the surrounding tie-lines
            rather than by a compound sitting on it -- so a ranking can separate
            known formulas from novel ones and show an enrichment that is really
            just that separation. Rediscovering a known compound is not what the
            structure-search budget is for, so this subset is the decision-relevant
            one.
    """
    # A formula with no hull is one the reference could not cover; it cannot be
    # ranked, and dropping it from both the numerator and the base rate keeps the
    # enrichment honest.
    usable = scored[scored[score_column].notna()]
    if novel_only:
        if "formula_known" not in usable.columns:
            raise ValueError("novel_only needs the formula_known column from score_structures")
        usable = usable[~usable["formula_known"].astype(bool)]
    if usable.empty:
        raise ValueError("No rows carry a score; was the hull lookup able to cover any formula?")
    # A gene whose reconstruction failed has no energy. It is a real outcome of
    # spending the budget -- a miss -- so it stays in the denominator as one.
    filled = usable.fillna({"e_above_hull": np.inf})
    order = filled.sort_values(score_column, kind="stable")
    rows = []
    for name, predicate in OUTCOMES.items():
        base = predicate(filled).to_numpy(dtype=bool).mean()
        ordered = predicate(order).to_numpy(dtype=bool)
        for budget in [*budgets, len(order)]:
            if budget > len(order):
                continue
            hits = int(ordered[:budget].sum())
            low, high = wilson_interval(hits, budget)
            row = {
                "outcome": name, "budget": budget, "hits": hits,
                "rate": hits / budget, "base_rate": base,
                "enrichment": (hits / budget) / base if base > 0 else np.nan,
                # The enrichment a rate at each end of the interval would give:
                # if this bracket spans 1.0, the slice is not distinguishable
                # from spending the same budget at random.
                "enrichment_low": low / base if base > 0 else np.nan,
                "enrichment_high": high / base if base > 0 else np.nan,
            }
            if "formula_known" in order.columns:
                row["share_known"] = float(order["formula_known"].to_numpy()[:budget].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("structures", type=Path, help="a protocol run's structures.csv")
    parser.add_argument("--ensemble", type=Path, default=DEFAULT_ENSEMBLE)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--genes", type=Path, default=None,
                        help="the gene file the run came from, to recover failed reconstructions")
    parser.add_argument("--out", type=Path, default=None, help="write the scored rows here")
    parser.add_argument("--budgets", type=int, nargs="+", default=(100, 250, 500))
    parser.add_argument("--reference-tolerance", type=float, default=REFERENCE_TOLERANCE,
                        help="keep reference entries within this of the hull; see the constant")
    parser.add_argument("--device", type=torch.device, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # reduce_formula announces every reduction. Generated cell formulas are
    # reducible more often than not, so at INFO this buries the tables.
    logging.getLogger("wyckoff_transformer.csp").setLevel(logging.WARNING)

    device = args.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    structures = pd.read_csv(args.structures)
    table = pd.read_parquet(args.table, columns=["e_hull_at_composition"])
    reference = load_reference(args.reference, args.reference_tolerance)

    models, _, feature_names = T.load_ensemble(args.ensemble, device)
    density = None
    if any(name.startswith("log1p_sys_") for name in feature_names):
        logger.info("building neighbourhood densities from the archive")
        density = SystemDensity.from_table(
            pd.read_parquet(args.table, columns=["e_form_min", "e_hull_at_composition", "chemsys"]))
    scored = score_structures(structures, models, table, reference, device, args.genes,
                              density, feature_names)
    if args.out:
        scored.to_csv(args.out, index=False)

    for column in ("score_naive", "score_adjusted"):
        for novel_only in (False, True):
            scope = "novel formulas only" if novel_only else "all genes"
            print(f"\n--- ranked by {column}, {scope} ---")
            budgets = args.budgets
            if novel_only:
                # The novel subset is smaller, so the same absolute budgets would
                # mostly be dropped; scale them to keep the same shape of question.
                share = float((~scored["formula_known"].astype(bool)).mean())
                budgets = [max(10, int(round(budget * share))) for budget in args.budgets]
            print(slice_table(scored, column, budgets, novel_only).to_string(index=False))


if __name__ == "__main__":
    main()
