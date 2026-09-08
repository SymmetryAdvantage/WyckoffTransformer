"""Does the generator's own surprisal buy novelty the energy screen throws away?

`scripts/analyse_dft_screen_uplift.py` measured the first lever and found its
limit: the DFT screen ranks stability well, but it buys that ranking with
compositions the reference archive already holds, so MetaSUN barely moves. The
fix it demonstrated -- drop the genes whose fingerprint is already in the
reference set, *then* rank -- is a binary filter that can only remove the 29% of
the pool that lookup can see.

`wyformer-gene-novelty` supplies a continuous version of the same lever:
-log p(gene) under the generative model. This script asks whether it earns its
place, on the same pool and against the same relaxation outcomes:

1. does surprisal predict novelty at all -- both gene-fingerprint novelty, which
   is free, and post-relaxation structure novelty, which is not;
2. what does it cost in stability, since a rare gene is also an odd one;
3. and does thresholding on it beat the free lookup, or only reproduce it.

    python scripts/analyse_novelty_screen.py generated/<run>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_dft_screen_uplift import (  # noqa: E402
    _p_value,
    _random_baseline,
    _rate,
    outcomes,
)

#: The energy arm the joint screen is built on: the headline of the first lever.
ENERGY_SCORE = "joint_score_adjusted"
#: The novelty arms. `surprisal` is the model's log-density of the whole gene;
#: `surprisal_per_site` divides that by the site count, which is a different
#: score rather than a normalisation of the same one -- the two are almost
#: rank-uncorrelated, because gene size dominates the unnormalised quantity.
NOVELTY_SCORES = ("surprisal", "surprisal_per_site")
#: Fractions of the pool the novelty filter keeps before the energy ranking runs.
#: 1.0 is the naive arm; 0.71 is roughly what the free fingerprint lookup keeps
#: on this pool, so the two are directly comparable.
KEEP_FRACTIONS = (1.0, 0.7, 0.5, 0.3, 0.2)


def load_novelty(pool: Path) -> pd.DataFrame:
    """The surprisal columns, for the genes the generator could score."""
    frame = pd.read_csv(pool / "gene_novelty.csv", index_col="index")
    missing = [column for column in NOVELTY_SCORES if column not in frame]
    if missing:
        raise ValueError(f"{pool/'gene_novelty.csv'} has no {missing}: rerun wyformer-gene-novelty")
    return frame


def _auc(score: pd.Series, label: pd.Series) -> dict:
    """Rank-AUC of `score` against a binary `label`, by the Mann-Whitney identity.

    AUC is the readable statistic here: it is the probability that a randomly
    chosen novel gene scores above a randomly chosen known one, which is exactly
    the question a screen asks of a novelty estimator, and it is invariant to any
    monotone rescaling of the score.
    """
    usable = score.notna() & label.notna()
    values = score[usable].to_numpy(dtype=float)
    positive = label[usable].to_numpy(dtype=bool)
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return {"auc": float("nan"), "n_positive": n_positive, "n_negative": n_negative}
    ranks = pd.Series(values).rank().to_numpy()
    auc = (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    return {"auc": float(auc), "n_positive": n_positive, "n_negative": n_negative}


def estimator_quality(frame: pd.DataFrame) -> dict:
    """Is the surprisal a novelty estimator, and what does it cost in stability?"""
    from scipy.stats import spearmanr  # noqa: PLC0415

    report: dict = {"auc": {}, "spearman": {}, "bins": {}}
    for score in NOVELTY_SCORES:
        report["auc"][score] = {
            "gene_novel": _auc(frame[score], frame["gene_novel"]),
            "novel_structure": _auc(frame[score], frame["novel_structure"]),
            # Novelty that survives everything else is what (M)SUN pays for.
            "metasun": _auc(frame[score], frame["metasun"]),
        }
        # The decisive comparison. Gene novelty is a free lookup, so a proxy for
        # it cannot beat it; the estimator earns its place only if it says
        # something the lookup cannot -- which of the genes the lookup has
        # already called novel go on to relax into a novel *structure*, and
        # which of them come out (M)SUN.
        novel = frame[frame["gene_novel"]]
        report["auc"][score]["novel_structure_given_gene_novel"] = _auc(
            novel[score], novel["novel_structure"])
        report["auc"][score]["metasun_given_gene_novel"] = _auc(
            novel[score], novel["metasun"])
        # Which term of (M)SUN the conditional signal is actually made of. If the
        # two agree, the estimator is behaving as a stability prior inside the
        # novel subset rather than as a novelty estimator at all.
        report["auc"][score]["metastable_given_gene_novel"] = _auc(
            novel[score], novel["metastable"])
        scored = frame[frame[score].notna() & frame["e_above_hull"].notna()]
        rho, p_value = spearmanr(scored[score], scored["e_above_hull"])
        report["spearman"][score] = {
            "vs_e_above_hull": {"rho": float(rho), "p": float(p_value), "n": int(len(scored))}}
        paired = frame[frame[score].notna() & frame[ENERGY_SCORE].notna()]
        rho, p_value = spearmanr(paired[score], paired[ENERGY_SCORE])
        report["spearman"][score]["vs_energy_score"] = {
            "rho": float(rho), "p": float(p_value), "n": int(len(paired))}

        ranked = frame[frame[score].notna()].copy()
        ranked["bin"] = pd.qcut(ranked[score], 10, labels=False, duplicates="drop")
        report["bins"][score] = [
            {
                "bin": int(index),
                "n": int(len(group)),
                "median_surprisal": float(group[score].median()),
                "median_e_above_hull": float(group["e_above_hull"].median()),
                "gene_novel": float(group["gene_novel"].mean()),
                "novel_structure": float(group["novel_structure"].mean()),
                "metastable": float(group["metastable"].mean()),
                "metasun": float(group["metasun"].mean()),
                "sun": float(group["sun"].mean()),
            }
            for index, group in ranked.groupby("bin")
        ]
    return report


def _arm(selected: pd.DataFrame, frame: pd.DataFrame, metric: str,
         n_sampled_total: int) -> dict:
    """One selection's hit rate, and what it is worth against the whole pool."""
    rest = frame.drop(index=selected.index)
    arm = _rate(selected, metric, n_sampled_total)
    pool_rate = frame[metric].mean()
    pool_per_relaxation = frame[metric].sum() / frame["n_trials"].sum()
    arm["uplift_vs_pool"] = (
        arm["per_submitted"] / pool_rate if pool_rate else float("nan"))
    arm["uplift_per_relaxation"] = (
        arm["per_relaxation"] / pool_per_relaxation if pool_per_relaxation else float("nan"))
    arm["p_value_vs_rest"] = _p_value(
        arm["hits"], len(selected), int(rest[metric].sum()), len(rest))
    return arm


def _filtered_then_ranked(frame: pd.DataFrame, novelty: str, keep: float,
                          budget: int) -> pd.DataFrame | None:
    """Keep the `keep` most surprising fraction, then spend the budget on energy.

    The order matters and is the point of the experiment: ranking by energy first
    and filtering after would spend the ranking on genes the novelty term is
    about to reject, which is precisely how the first lever lost its gains.
    """
    usable = frame[frame[novelty].notna()]
    n_keep = int(round(keep * len(usable)))
    if n_keep < budget:
        return None
    subset = usable.nlargest(n_keep, novelty, keep="first")
    return subset.nsmallest(budget, ENERGY_SCORE, keep="first")


def _rank_fused(frame: pd.DataFrame, novelty: str, weight: float,
                budget: int) -> pd.DataFrame:
    """Rank by a weighted sum of the two scores' normalised ranks.

    A soft alternative to the threshold: no gene is excluded outright, so a very
    low-energy candidate can still buy its way in on stability alone.
    """
    usable = frame[frame[novelty].notna() & frame[ENERGY_SCORE].notna()]
    energy_rank = usable[ENERGY_SCORE].rank(pct=True)
    novelty_rank = (-usable[novelty]).rank(pct=True)
    fused = (1.0 - weight) * energy_rank + weight * novelty_rank
    return usable.loc[fused.nsmallest(budget, keep="first").index]


#: Quantile bands of `surprisal` the lookup-free sweep keeps before ranking by
#: energy. `(0.0, 1.0)` is the energy screen alone; a band with `hi < 1` drops the
#: genes the model finds most improbable, which the funnel says relax nowhere.
BAND_LOW = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
BAND_HIGH = (0.7, 0.8, 0.9, 1.0)
#: Weights on the novelty rank in the soft alternative to the band. The hard
#: filter excludes; this only reweights, so a very low-energy gene can still buy
#: its way in on stability alone.
FUSE_WEIGHTS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)


def _band(frame: pd.DataFrame, novelty: str, low: float, high: float) -> pd.DataFrame:
    """The genes whose surprisal falls in the quantile band [low, high]."""
    usable = frame[frame[novelty].notna()]
    quantile = usable[novelty].rank(pct=True)
    return usable[(quantile > low) & (quantile <= high)]


def lookup_free_sweep(frame: pd.DataFrame, budgets: tuple[int, ...],
                      n_sampled_total: int, novelty: str = "surprisal",
                      splits: int = 40, seed: int = 0) -> dict:
    """The two-estimator screen on its own: energy and likelihood, no reference set.

    This is the arm that matters when there is no archive to deduplicate against,
    and it is swept properly rather than at a few round numbers, because the
    funnel implies the optimum is a *band*: the most typical genes are the ones
    already in the archive, and the most surprising ones relax nowhere, so both
    tails should go.

    Reporting the best cell of a swept grid on the same pool it was chosen on is
    a selection bias, so each budget also carries a split-half estimate: the band
    is chosen on one random half and scored on the other, over `splits` draws.
    The gap between the two is how much of the grid maximum is real.
    """
    rng = np.random.default_rng(seed)
    usable = frame[frame[novelty].notna() & frame[ENERGY_SCORE].notna()]
    report: dict = {"novelty_score": novelty, "budgets": {}}

    for budget in budgets:
        grid = []
        for low in BAND_LOW:
            for high in BAND_HIGH:
                if high - low < 0.15:
                    continue
                selected = _band(usable, novelty, low, high)
                if len(selected) < budget:
                    continue
                picked = selected.nsmallest(budget, ENERGY_SCORE, keep="first")
                arm = _arm(picked, frame, "metasun", n_sampled_total)
                arm.update({"low": low, "high": high})
                grid.append(arm)
        if not grid:
            continue
        best = max(grid, key=lambda row: row["per_submitted"])

        # Split-half: choose the band on one half, spend the budget on the other.
        half_budget = max(1, budget // 2)
        held_out, chosen = [], []
        for _ in range(splits):
            shuffled = rng.permutation(usable.index.to_numpy())
            first = usable.loc[shuffled[: len(shuffled) // 2]]
            second = usable.loc[shuffled[len(shuffled) // 2:]]
            candidates = []
            for low in BAND_LOW:
                for high in BAND_HIGH:
                    if high - low < 0.15:
                        continue
                    picked = _band(first, novelty, low, high)
                    if len(picked) < half_budget:
                        continue
                    rate = picked.nsmallest(half_budget, ENERGY_SCORE, keep="first")[
                        "metasun"].mean()
                    candidates.append((rate, low, high))
            if not candidates:
                continue
            _, low, high = max(candidates)
            picked = _band(second, novelty, low, high)
            if len(picked) < half_budget:
                continue
            held_out.append(float(picked.nsmallest(
                half_budget, ENERGY_SCORE, keep="first")["metasun"].mean()))
            chosen.append((low, high))
        pool_rate = frame["metasun"].mean()
        fused = []
        for weight in FUSE_WEIGHTS:
            arm = _arm(_rank_fused(usable, novelty, weight, budget), frame, "metasun",
                       n_sampled_total)
            arm["weight"] = weight
            fused.append(arm)
        report["budgets"][budget] = {
            "grid": grid,
            "rank_fusion": fused,
            "best_in_sample": best,
            "held_out": {
                "n_splits": len(held_out),
                "per_submitted": float(np.mean(held_out)) if held_out else float("nan"),
                "per_submitted_p5": float(np.quantile(held_out, 0.05)) if held_out else float("nan"),
                "uplift_vs_pool": (float(np.mean(held_out)) / pool_rate) if held_out and pool_rate
                                  else float("nan"),
                "modal_band": max(set(chosen), key=chosen.count) if chosen else None,
            },
        }
    return report


def analyse(pool: Path, budgets: tuple[int, ...], draws: int, seed: int) -> dict:
    screen = pd.read_csv(pool / "dft_screen.csv", index_col="index")
    novelty = load_novelty(pool)
    result = outcomes(pool / "protocol")
    n_sampled_total = int(result["sampled"].sum())

    frame = result.join(screen[[ENERGY_SCORE]], how="inner").join(
        novelty[list(NOVELTY_SCORES) + ["n_sites", "log_representations"]], how="left")
    scorable = frame[list(NOVELTY_SCORES)].notna().all(axis=1)
    rng = np.random.default_rng(seed)

    report: dict = {
        "pool": str(pool),
        "n_relaxed_representatives": int(len(frame)),
        "n_scored_by_generator": int(scorable.sum()),
        "n_sampled_genes": n_sampled_total,
        "pool_rate": {
            metric: _rate(frame, metric, n_sampled_total) for metric in ("metasun", "sun")},
        "gene_novel_rate": float(frame["gene_novel"].mean()),
        "estimator_quality": estimator_quality(frame[scorable]),
        "lookup_free_sweep": lookup_free_sweep(
            frame[scorable], budgets, n_sampled_total, seed=seed),
        "budgets": {},
    }

    for budget in budgets:
        if budget > int(scorable.sum()):
            continue
        entry: dict = {}
        for metric in ("metasun", "sun"):
            arms = {"random": _random_baseline(frame, metric, budget, draws, rng)}
            # The two single-lever baselines this has to beat.
            arms["energy_only"] = _arm(
                frame.nsmallest(budget, ENERGY_SCORE, keep="first"), frame, metric,
                n_sampled_total)
            arms["fingerprint_novel_then_energy"] = _arm(
                frame[frame["gene_novel"]].nsmallest(budget, ENERGY_SCORE, keep="first"),
                frame, metric, n_sampled_total)
            for novelty_score in NOVELTY_SCORES:
                arms[f"{novelty_score}_only"] = _arm(
                    frame[scorable].nlargest(budget, novelty_score, keep="first"),
                    frame, metric, n_sampled_total)
                for keep in KEEP_FRACTIONS:
                    selected = _filtered_then_ranked(
                        frame[scorable], novelty_score, keep, budget)
                    if selected is None:
                        continue
                    arms[f"{novelty_score}_top{keep:g}_then_energy"] = _arm(
                        selected, frame, metric, n_sampled_total)
                # Both levers plus the free lookup: does the model's density add
                # anything the fingerprint set does not already say? Both
                # directions are asked, because within the novel set the two
                # objectives point opposite ways -- the most surprising genes are
                # the most novel and the least likely to relax anywhere good.
                both = frame[scorable & frame["gene_novel"]]
                for keep in (0.5, 0.3):
                    selected = _filtered_then_ranked(both, novelty_score, keep, budget)
                    if selected is not None:
                        arms[f"fingerprint_novel_{novelty_score}_top{keep:g}_then_energy"] = _arm(
                            selected, frame, metric, n_sampled_total)
                    selected = _filtered_then_ranked(
                        both.assign(**{f"_least_{novelty_score}": -both[novelty_score]}),
                        f"_least_{novelty_score}", keep, budget)
                    if selected is not None:
                        arms[f"fingerprint_novel_least_{novelty_score}{keep:g}_then_energy"] = \
                            _arm(selected, frame, metric, n_sampled_total)
                for weight in (0.25, 0.5):
                    arms[f"{novelty_score}_rankfuse_w{weight:g}"] = _arm(
                        _rank_fused(frame[scorable], novelty_score, weight, budget),
                        frame, metric, n_sampled_total)
            entry[metric] = arms
        report["budgets"][budget] = entry
    return report


def _print_quality(report: dict) -> None:
    quality = report["estimator_quality"]
    print("\n--- is surprisal a novelty estimator? (AUC, 0.5 = no signal) ---")
    for score, entries in quality["auc"].items():
        parts = " ".join(
            f"{name} {stats['auc']:.3f}" for name, stats in entries.items())
        print(f"  {score:22s} {parts}")
    print("\n--- what it costs (Spearman) ---")
    for score, entries in quality["spearman"].items():
        hull = entries["vs_e_above_hull"]
        energy = entries["vs_energy_score"]
        print(f"  {score:22s} vs e_above_hull {hull['rho']:+.3f}  "
              f"vs {ENERGY_SCORE} {energy['rho']:+.3f}")
    for score, rows in quality["bins"].items():
        print(f"\n--- funnel by {score} decile (9 = most surprising) ---")
        print(f"  {'bin':>3} {'surprisal':>10} {'med e_hull':>11} {'gene nov':>9} "
              f"{'str. nov':>9} {'metastab':>9} {'MetaSUN':>8} {'SUN':>7}")
        for row in rows:
            print(f"  {row['bin']:>3} {row['median_surprisal']:>10.2f} "
                  f"{row['median_e_above_hull']:>11.3f} {row['gene_novel']:>9.1%} "
                  f"{row['novel_structure']:>9.1%} {row['metastable']:>9.1%} "
                  f"{row['metasun']:>8.1%} {row['sun']:>7.1%}")


def _print_sweep(report: dict) -> None:
    """The no-reference-set arm: what the two estimators are worth by themselves."""
    sweep = report["lookup_free_sweep"]
    pool_rate = report["pool_rate"]["metasun"]["per_submitted"]
    print("\n" + "=" * 78)
    print("LOOKUP-FREE: energy + likelihood only, surprisal quantile band swept")
    print("=" * 78)
    for budget, entry in sweep["budgets"].items():
        grid = entry["grid"]
        highs = sorted({row["high"] for row in grid})
        lows = sorted({row["low"] for row in grid})
        print(f"\nMetaSUN uplift at budget {budget} "
              f"(rows: drop this bottom fraction; cols: keep up to this quantile)")
        print("        " + "".join(f"{high:>8.2f}" for high in highs))
        for low in lows:
            cells = []
            for high in highs:
                match = [r for r in grid if r["low"] == low and r["high"] == high]
                cells.append(f"{match[0]['uplift_vs_pool']:>8.2f}" if match else f"{'-':>8}")
            print(f"  {low:>5.2f} " + "".join(cells))
        best = entry["best_in_sample"]
        out = entry["held_out"]
        print(f"  best in sample   band ({best['low']:.2f}, {best['high']:.2f})  "
              f"{best['per_submitted']:.4f}  {best['uplift_vs_pool']:.2f}x")
        print(f"  split-half held out  {out['per_submitted']:.4f}  "
              f"{out['uplift_vs_pool']:.2f}x over {out['n_splits']} splits, "
              f"modal band {out['modal_band']}")
        fused = " ".join(
            f"w={row['weight']:g}:{row['uplift_vs_pool']:.2f}x" for row in entry["rank_fusion"])
        print(f"  soft rank fusion  {fused}")
    print(f"\n  (pool MetaSUN {pool_rate:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pool", type=Path,
                        help="directory holding dft_screen.csv, gene_novelty.csv and protocol/")
    parser.add_argument("--budgets", type=str, default="250,500,1000")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="write the report as JSON (default: <pool>/novelty_screen.json)")
    args = parser.parse_args()

    budgets = tuple(int(b) for b in args.budgets.split(",") if b.strip())
    report = analyse(args.pool, budgets, args.draws, args.seed)
    out = args.out or args.pool / "novelty_screen.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{report['n_relaxed_representatives']} relaxed representatives, "
          f"{report['n_scored_by_generator']} scored by the generator; "
          f"{report['gene_novel_rate']:.1%} novel by fingerprint")
    pool_rates = report["pool_rate"]
    print(f"pool MetaSUN {pool_rates['metasun']['per_submitted']:.4f} "
          f"| SUN {pool_rates['sun']['per_submitted']:.4f}")
    _print_quality(report)
    _print_sweep(report)

    for budget, entry in report["budgets"].items():
        for metric, arms in entry.items():
            print(f"\n--- {metric} at budget {budget} (per submitted) ---")
            baseline = arms["random"]
            print(f"  {'random':44s} {baseline['per_submitted']:.4f} "
                  f"[{baseline['per_submitted_p5']:.4f}, {baseline['per_submitted_p95']:.4f}]")
            for name, arm in arms.items():
                if name == "random":
                    continue
                print(f"  {name:44s} {arm['per_submitted']:.4f} "
                      f"({arm['uplift_vs_pool']:.2f}x gene, "
                      f"{arm['uplift_per_relaxation']:.2f}x relax, "
                      f"p={arm['p_value_vs_rest']:.3g})")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
