"""Does the DFT fixed-hull screen enrich a fixed relaxation budget for (Meta)SUN?

The screen (`wyformer-dft-screen`) ranks a pool of genes before any relaxation;
the protocol (`wyformer-protocol`) relaxes the pool and says which genes actually
came out SUN. Running both over the *same* pool turns the screen's claim into a
measurement: take the top N by each score, and compare the SUN rate in that slice
with the rate over the whole pool, which is what a random draw of N would give.

    python scripts/analyse_dft_screen_uplift.py generated/<run> [--budgets 250,500,1000]

Rates are reported against two denominators, as docs/dft_fixed_hull_attack.md asks
for: per submitted structure (the selection question) and per sampled gene (the
generator question, which selection cannot move).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: The score columns `wyformer-dft-screen` writes, best first, plus the two
#: single-estimator ablations the doc calls for.
SCORES = (
    "joint_score_adjusted",
    "joint_score_naive",
    "composition_score_adjusted",
    "composition_score_naive",
    "gene_score",
)
#: The score the per-bin funnel is cut on: the conservative joint score is the
#: screen's headline ranking, so its bins are the ones worth reading.
DECILE_SCORE = "joint_score_adjusted"
METASTABLE_THRESHOLD = 0.1
STABLE_THRESHOLD = 0.0


def outcomes(protocol_dir: Path) -> pd.DataFrame:
    """Per representative gene: the SUN indicators, and what it stands for."""
    frame = pd.read_csv(protocol_dir / "structures.csv", index_col="index")
    screen = json.loads((protocol_dir / "screen.json").read_text(encoding="utf-8"))
    counts = {int(k): int(v) for k, v in screen["counts"].items()}

    surviving = pd.Series(True, index=frame.index)
    for column in ("has_structure", "valid_structure", "unique_structure", "novel_structure"):
        if column not in frame.columns:
            raise ValueError(
                f"{protocol_dir/'structures.csv'} has no {column!r}: run --stage score")
        surviving &= frame[column].fillna(False).astype(bool)

    energy = frame["e_above_hull"]
    result = pd.DataFrame(index=frame.index)
    result["metastable"] = energy.notna() & (energy <= METASTABLE_THRESHOLD)
    result["stable"] = energy.notna() & (energy <= STABLE_THRESHOLD)
    result["metasun"] = surviving & result["metastable"]
    result["sun"] = surviving & result["stable"]
    result["e_above_hull"] = energy
    # Kept unfused as well: (M)SUN is a product of stability and novelty, and the
    # two move in opposite directions under this screen, so a bare uplift number
    # cannot say which term a ranking is winning or losing on.
    for column in ("valid_structure", "novel_structure"):
        result[column] = frame[column].fillna(False).astype(bool)
    # A representative stands for every sampled gene with its fingerprint, so a
    # per-sampled-gene rate weights it by that count.
    result["sampled"] = pd.Series(counts).reindex(result.index).fillna(1).astype(int)
    return result


def _rate(selected: pd.DataFrame, column: str, n_sampled_total: int) -> dict:
    """One arm's hit rate against both denominators."""
    hits = selected[column]
    return {
        "n_submitted": int(len(selected)),
        "hits": int(hits.sum()),
        "per_submitted": float(hits.mean()) if len(selected) else float("nan"),
        "per_sampled_gene": float(
            (selected.loc[hits, "sampled"].sum()) / n_sampled_total),
    }


def _random_baseline(frame: pd.DataFrame, column: str, budget: int,
                     draws: int, rng: np.random.Generator) -> dict:
    """What a random draw of `budget` genes from the same pool gives.

    Sampling rather than using the pool mean, because the comparison the arms are
    judged against needs the spread of a draw of this size, not just its centre.
    """
    values = frame[column].to_numpy(dtype=bool)
    picks = rng.integers(0, len(values), size=(draws, budget))
    rates = values[picks].mean(axis=1)
    return {
        "n_submitted": budget,
        "per_submitted": float(rates.mean()),
        "per_submitted_p5": float(np.quantile(rates, 0.05)),
        "per_submitted_p95": float(np.quantile(rates, 0.95)),
    }


def _p_value(selected_hits: int, selected_n: int, rest_hits: int, rest_n: int) -> float:
    """One-sided Fisher exact: is the selected slice richer than the rest?"""
    from scipy.stats import fisher_exact  # noqa: PLC0415

    table = [[selected_hits, selected_n - selected_hits],
             [rest_hits, rest_n - rest_hits]]
    return float(fisher_exact(table, alternative="greater")[1])


def rank_quality(frame: pd.DataFrame, bins: int = 10) -> dict:
    """How well each score orders the pool, before any budget is imposed.

    A top-N uplift conflates the screen's skill with where the budget happens to
    fall. The rank correlation against the achieved hull distance is the skill on
    its own, and the per-bin funnel shows what the ranking costs elsewhere.
    """
    from scipy.stats import spearmanr  # noqa: PLC0415

    scored = frame[frame["valid_structure"] & frame["e_above_hull"].notna()]
    correlations = {}
    for score in SCORES:
        rho, p_value = spearmanr(scored[score], scored["e_above_hull"])
        correlations[score] = {"spearman_rho": float(rho), "p_value": float(p_value),
                               "n": int(len(scored))}

    ranked = frame.copy()
    ranked["bin"] = pd.qcut(ranked[DECILE_SCORE], bins, labels=False)
    grouped = ranked.groupby("bin")
    bin_rows = [
        {
            "bin": int(index),
            "n": int(len(group)),
            "median_e_above_hull": float(group["e_above_hull"].median()),
            "valid_structure": float(group["valid_structure"].mean()),
            "novel_structure": float(group["novel_structure"].mean()),
            "metastable": float(group["metastable"].mean()),
            "metasun": float(group["metasun"].mean()),
            "sun": float(group["sun"].mean()),
            "formula_known": float(group["formula_known"].mean()),
        }
        for index, group in grouped
    ]
    return {"spearman_vs_e_above_hull": correlations,
            "binned_by": DECILE_SCORE, "bins": bin_rows}


def analyse(pool: Path, budgets: tuple[int, ...], draws: int, seed: int) -> dict:
    screen = pd.read_csv(pool / "dft_screen.csv", index_col="index")
    result = outcomes(pool / "protocol")
    n_sampled_total = int(result["sampled"].sum())

    # Only genes with a relaxation outcome can be scored, and the screen ranks the
    # sampled pool including duplicates; join on the representatives.
    columns = list(SCORES) + (["formula_known"] if "formula_known" in screen else [])
    frame = result.join(screen[columns], how="inner")
    if "formula_known" in frame:
        frame["formula_known"] = frame["formula_known"].fillna(False).astype(bool)
    missing = [c for c in SCORES if frame[c].isna().all()]
    if missing:
        raise ValueError(f"the screen has no values for {missing}")

    rng = np.random.default_rng(seed)
    report: dict = {
        "pool": str(pool),
        "n_relaxed_representatives": int(len(frame)),
        "n_sampled_genes": n_sampled_total,
        "pool_rate": {
            metric: _rate(frame, metric, n_sampled_total)
            for metric in ("metasun", "sun")
        },
        "rank_quality": rank_quality(frame),
        "budgets": {},
    }
    # The screen finds low-lying structures partly by finding compositions the
    # reference set already holds, so ranking the raw pool spends its skill on
    # genes novelty will reject. Ranking inside the novel-formula subset is the
    # same screen with that overlap removed.
    novel_only = frame[~frame["formula_known"]] if "formula_known" in frame else None
    if novel_only is not None:
        report["n_novel_formula"] = int(len(novel_only))

    for budget in budgets:
        if budget > len(frame):
            continue
        entry: dict = {}
        for metric in ("metasun", "sun"):
            arms = {"random": _random_baseline(frame, metric, budget, draws, rng)}
            for score in SCORES:
                # Lower is better: the score is an energy margin above the hull.
                selected = frame.nsmallest(budget, score, keep="first")
                rest = frame.drop(index=selected.index)
                arm = _rate(selected, metric, n_sampled_total)
                arm["uplift_vs_pool"] = (
                    arm["per_submitted"] / frame[metric].mean()
                    if frame[metric].mean() else float("nan"))
                arm["p_value_vs_rest"] = _p_value(
                    arm["hits"], len(selected),
                    int(rest[metric].sum()), len(rest))
                arms[score] = arm
            if novel_only is not None:
                restricted = {}
                for score in SCORES:
                    selected = novel_only.nsmallest(budget, score, keep="first")
                    if len(selected) < budget:
                        continue
                    rest = frame.drop(index=selected.index)
                    arm = _rate(selected, metric, n_sampled_total)
                    arm["uplift_vs_pool"] = (
                        arm["per_submitted"] / frame[metric].mean()
                        if frame[metric].mean() else float("nan"))
                    arm["p_value_vs_rest"] = _p_value(
                        arm["hits"], len(selected),
                        int(rest[metric].sum()), len(rest))
                    # Flags the point where the budget has eaten the subset and
                    # the arm has decayed into that subset's base rate.
                    arm["fraction_of_subset"] = float(budget / len(novel_only))
                    restricted[score] = arm
                arms["novel_formula_only"] = restricted
            entry[metric] = arms
        report["budgets"][budget] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pool", type=Path,
                        help="directory holding dft_screen.csv and protocol/")
    parser.add_argument("--budgets", type=str, default="250,500,1000,2000")
    parser.add_argument("--draws", type=int, default=10000,
                        help="random-baseline draws per budget")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="write the report as JSON (default: <pool>/dft_screen_uplift.json)")
    args = parser.parse_args()

    budgets = tuple(int(b) for b in args.budgets.split(",") if b.strip())
    report = analyse(args.pool, budgets, args.draws, args.seed)
    out = args.out or args.pool / "dft_screen_uplift.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    pool_rates = report["pool_rate"]
    print(f"{report['n_relaxed_representatives']} relaxed representatives, "
          f"{report['n_sampled_genes']} sampled genes")
    print(f"pool MetaSUN {pool_rates['metasun']['per_submitted']:.4f} "
          f"| SUN {pool_rates['sun']['per_submitted']:.4f}")

    quality = report["rank_quality"]
    print("\n--- Spearman(score, e_above_hull), lower score should mean lower hull ---")
    for name, stats in quality["spearman_vs_e_above_hull"].items():
        print(f"  {name:28s} rho={stats['spearman_rho']:+.4f}  p={stats['p_value']:.3g}")
    print(f"\n--- funnel by {quality['binned_by']} bin (0 = best score) ---")
    print(f"  {'bin':>3} {'med e_hull':>11} {'valid':>7} {'novel':>7} "
          f"{'metastab':>9} {'MetaSUN':>8} {'known f.':>9}")
    for row in quality["bins"]:
        print(f"  {row['bin']:>3} {row['median_e_above_hull']:>11.3f} "
              f"{row['valid_structure']:>7.1%} {row['novel_structure']:>7.1%} "
              f"{row['metastable']:>9.1%} {row['metasun']:>8.1%} "
              f"{row['formula_known']:>9.1%}")

    for budget, entry in report["budgets"].items():
        for metric, arms in entry.items():
            print(f"\n--- {metric} at budget {budget} (per submitted) ---")
            for name, arm in arms.items():
                if name == "random":
                    print(f"  {name:28s} {arm['per_submitted']:.4f} "
                          f"[{arm['per_submitted_p5']:.4f}, {arm['per_submitted_p95']:.4f}]")
                elif name == "novel_formula_only":
                    for score, restricted in arm.items():
                        print(f"  {'novel-only ' + score:28s} "
                              f"{restricted['per_submitted']:.4f} "
                              f"({restricted['uplift_vs_pool']:.2f}x, "
                              f"p={restricted['p_value_vs_rest']:.3g}, "
                              f"{restricted['fraction_of_subset']:.0%} of subset)")
                else:
                    print(f"  {name:28s} {arm['per_submitted']:.4f} "
                          f"({arm['uplift_vs_pool']:.2f}x, p={arm['p_value_vs_rest']:.3g})")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
