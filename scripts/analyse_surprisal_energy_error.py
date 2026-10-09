"""Does the generator's surprisal say where the energy regressor is wrong?

Fire-control ranks genes by a regressor's predicted ``e_hull``. The best arm of
docs/generative_novelty_screen.md kept only the least surprising 30% of the
novel genes before that ranking ran; this asks whether the surprisal works by
predicting the regressor's error.

Two outcomes, both ORB ``e_above_hull`` against the LeMat-Bulk ORB hull, from
the pool's protocol run:

``fixed_symmetry``
    relaxed with the gene's symmetry held, so the structure is still the gene's
    Wyckoff representation. This is the gene's own energy, the quantity the
    regressor predicts (its target is the lowest formation energy among the
    gene's structures).
``free``
    the structure the protocol keeps, after the symmetry release and the rattle.
    It is what a selection is scored on.

``residual = realized - predicted``; positive means the regressor was
optimistic. The prediction is against the PBE hull and the outcome against the
ORB one, so a global offset is expected; the median residual is removed and
what is studied is the spread around it.

Only gene-novel representatives are read: a gene LeMat-Bulk holds may have its
minimum among the regressor's training targets, and its error would say nothing
about the genes a screen is for.

    python scripts/analyse_surprisal_energy_error.py $WYFORMER_RUNS/roe_surprisal/cfg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from wyckoff_transformer.evaluation.protocol import METASTABLE_THRESHOLD

#: An error this large moves a gene across the whole metastability window.
LARGE_ERROR = 0.1
#: The selection fire-control makes at B = 1000.
TOP = 1000


def _auc(score: pd.Series, label: pd.Series) -> Optional[float]:
    usable = score.notna() & label.notna()
    values = score[usable].to_numpy(dtype=float)
    positive = label[usable].to_numpy(dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    ranks = pd.Series(values).rank().to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _rho(a: pd.Series, b: pd.Series) -> Optional[float]:
    usable = a.notna() & b.notna()
    if usable.sum() < 10:
        return None
    return float(spearmanr(a[usable], b[usable])[0])


def stratified_rho(score: pd.Series, target: pd.Series, strata: pd.Series) -> Optional[float]:
    """Spearman within each stratum, averaged with the stratum sizes as weights.

    Surprisal grows with the number of sites, and so may the error; inside a
    stratum of equal site count that route is closed.
    """
    total, weight = 0.0, 0
    for _, group in pd.DataFrame({"s": score, "t": target, "k": strata}).dropna().groupby("k"):
        if len(group) >= 20 and group["s"].nunique() > 1 and group["t"].nunique() > 1:
            total += len(group) * spearmanr(group["s"], group["t"])[0]
            weight += len(group)
    return total / weight if weight else None


def load(root: Path) -> pd.DataFrame:
    genes = pd.read_csv(root / "analysis" / "genes.csv.gz", index_col=0)
    reps = genes[genes["is_representative"] & genes["gene_novel"]].copy()
    fixed = pd.read_csv(root / "protocol" / "structures_fixed_symmetry.csv", index_col=0)
    reps["e_hull_fixed_symmetry"] = fixed["e_above_hull"].reindex(reps.index).astype(float)
    reps["e_hull_free"] = reps["e_above_hull"].astype(float)
    variants = [c.removeprefix("surprisal_") for c in reps.columns if c.startswith("surprisal_")]
    first = variants[0]
    sites = pd.read_csv(root / "scores" / f"gene_novelty_{first}.csv", index_col=0)["n_sites"]
    reps["n_sites"] = sites.reindex(reps.index)
    return reps


def study(reps: pd.DataFrame, variant: str, target: str) -> dict:
    surprisal = reps[f"surprisal_{variant}"]
    predicted = reps["predicted_e_hull"]
    realized = reps[f"e_hull_{target}"]
    usable = surprisal.notna() & predicted.notna() & realized.notna()
    d = reps[usable].copy()
    d["residual"] = realized[usable] - predicted[usable]
    offset = float(d["residual"].median())
    d["centred"] = d["residual"] - offset
    d["abs_error"] = d["centred"].abs()
    d["large"] = d["abs_error"] > LARGE_ERROR
    d["optimistic_miss"] = d["centred"] > LARGE_ERROR
    d["s"] = d[f"surprisal_{variant}"]

    scores = {
        "surprisal": d["s"],
        "surprisal_per_site": d["s"] / d["n_sites"],
        "n_sites": d["n_sites"].astype(float),
        "n_atoms": d["n_atoms"].astype(float),
        "predicted_e_hull": d["predicted_e_hull"],
    }
    report = {
        "n": int(len(d)),
        "median_offset": offset,
        "mean_abs_error": float(d["abs_error"].mean()),
        "rank_correlation_predicted_realized": _rho(d["predicted_e_hull"], d[f"e_hull_{target}"]),
        "large_error_rate": float(d["large"].mean()),
        "predictors": {
            name: {
                "spearman_abs_error": _rho(score, d["abs_error"]),
                "spearman_signed_residual": _rho(score, d["centred"]),
                "auc_large_error": _auc(score, d["large"]),
                "auc_optimistic_miss": _auc(score, d["optimistic_miss"]),
            }
            for name, score in scores.items()
        },
        "surprisal_within_equal_site_count": {
            "spearman_abs_error": stratified_rho(d["s"], d["abs_error"], d["n_sites"]),
            "spearman_signed_residual": stratified_rho(d["s"], d["centred"], d["n_sites"]),
        },
    }

    d["decile"] = pd.qcut(d["s"], 10, labels=False, duplicates="drop")
    report["deciles"] = [
        {
            "decile": int(k), "n": int(len(g)),
            "median_surprisal": float(g["s"].median()),
            "median_signed_residual": float(g["centred"].median()),
            "mean_abs_error": float(g["abs_error"].mean()),
            "p90_abs_error": float(g["abs_error"].quantile(0.9)),
            "large_error_rate": float(g["large"].mean()),
            "optimistic_miss_rate": float(g["optimistic_miss"].mean()),
            "rank_correlation_predicted_realized": _rho(
                g["predicted_e_hull"], g[f"e_hull_{target}"]),
            "metastable": float((g[f"e_hull_{target}"] <= METASTABLE_THRESHOLD).mean()),
        }
        for k, g in d.groupby("decile")
    ]

    # What fire-control actually selects: the lowest predictions. Split them by
    # where they sit in the surprisal distribution of all novel genes.
    top = d.nsmallest(TOP, "predicted_e_hull").copy()
    tercile = pd.qcut(d["s"], 3, labels=["low", "middle", "high"])
    top["tercile"] = tercile.reindex(top.index)
    report["fire_control_top"] = {
        str(k): {
            "n": int(len(g)),
            "median_predicted_e_hull": float(g["predicted_e_hull"].median()),
            "median_realized_e_hull": float(g[f"e_hull_{target}"].median()),
            "median_signed_residual": float(g["centred"].median()),
            "optimistic_miss_rate": float(g["optimistic_miss"].mean()),
            "metastable": float((g[f"e_hull_{target}"] <= METASTABLE_THRESHOLD).mean()),
        }
        for k, g in top.groupby("tercile", observed=True)
    }
    return report


def _f(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def markdown(name: str, variant: str, report: dict) -> str:
    lines = [f"### {name}, surprisal `{variant}`\n"]
    for target in ("fixed_symmetry", "free"):
        r = report[target]
        lines.append(
            f"**{target}**: {r['n']} novel genes, offset {r['median_offset']:+.3f}, "
            f"mean |error| {r['mean_abs_error']:.3f}, rank correlation predicted-realized "
            f"{_f(r['rank_correlation_predicted_realized'])}, |error| > {LARGE_ERROR}: "
            f"{r['large_error_rate']:.3f}\n")
        lines.append("| predictor | rho with \\|error\\| | rho with signed error | "
                     "AUC \\|error\\| > 0.1 | AUC optimistic miss |")
        lines.append("|---|---|---|---|---|")
        for predictor, v in r["predictors"].items():
            lines.append(f"| {predictor} | {_f(v['spearman_abs_error'])} | "
                         f"{_f(v['spearman_signed_residual'])} | {_f(v['auc_large_error'])} | "
                         f"{_f(v['auc_optimistic_miss'])} |")
        w = r["surprisal_within_equal_site_count"]
        lines.append(f"| surprisal, within equal site count | {_f(w['spearman_abs_error'])} | "
                     f"{_f(w['spearman_signed_residual'])} | | |\n")
        lines.append("| surprisal decile | median signed error | mean \\|error\\| | p90 \\|error\\| "
                     "| optimistic miss | rank corr. predicted-realized | metastable |")
        lines.append("|---|---|---|---|---|---|---|")
        for dec in r["deciles"]:
            lines.append(
                f"| {dec['decile']} | {dec['median_signed_residual']:+.3f} | "
                f"{dec['mean_abs_error']:.3f} | {dec['p90_abs_error']:.3f} | "
                f"{dec['optimistic_miss_rate']:.3f} | "
                f"{_f(dec['rank_correlation_predicted_realized'])} | {dec['metastable']:.3f} |")
        lines.append(f"\nThe {TOP} lowest predictions, by surprisal tercile of the novel genes:\n")
        lines.append("| tercile | n | median predicted | median realized | optimistic miss | metastable |")
        lines.append("|---|---|---|---|---|---|")
        for k, v in r["fire_control_top"].items():
            lines.append(f"| {k} | {v['n']} | {v['median_predicted_e_hull']:.3f} | "
                         f"{v['median_realized_e_hull']:.3f} | {v['optimistic_miss_rate']:.3f} | "
                         f"{v['metastable']:.3f} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", type=Path, nargs="+",
                        help="Pool roots already replayed by analyse_roe_surprisal.py.")
    args = parser.parse_args(argv)
    for root in args.roots:
        reps = load(root)
        variants = [c.removeprefix("surprisal_") for c in reps.columns
                    if c.startswith("surprisal_")]
        report = {variant: {target: study(reps, variant, target)
                            for target in ("fixed_symmetry", "free")}
                  for variant in variants}
        (root / "analysis" / "energy_error.json").write_text(json.dumps(report, indent=1))
        text = "\n".join(markdown(root.name, v, report[v]) for v in variants)
        (root / "analysis" / "energy_error.md").write_text(text)
        print(text)


if __name__ == "__main__":
    main()
