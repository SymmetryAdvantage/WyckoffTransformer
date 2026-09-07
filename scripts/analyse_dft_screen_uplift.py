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
    result["metasun"] = surviving & energy.notna() & (energy <= METASTABLE_THRESHOLD)
    result["sun"] = surviving & energy.notna() & (energy <= STABLE_THRESHOLD)
    result["e_above_hull"] = energy
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


def analyse(pool: Path, budgets: tuple[int, ...], draws: int, seed: int) -> dict:
    screen = pd.read_csv(pool / "dft_screen.csv", index_col="index")
    result = outcomes(pool / "protocol")
    n_sampled_total = int(result["sampled"].sum())

    # Only genes with a relaxation outcome can be scored, and the screen ranks the
    # sampled pool including duplicates; join on the representatives.
    frame = result.join(screen[list(SCORES)], how="inner")
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
        "budgets": {},
    }

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
    for budget, entry in report["budgets"].items():
        for metric, arms in entry.items():
            print(f"\n--- {metric} at budget {budget} (per submitted) ---")
            for name, arm in arms.items():
                if name == "random":
                    print(f"  {name:28s} {arm['per_submitted']:.4f} "
                          f"[{arm['per_submitted_p5']:.4f}, {arm['per_submitted_p95']:.4f}]")
                else:
                    print(f"  {name:28s} {arm['per_submitted']:.4f} "
                          f"({arm['uplift_vs_pool']:.2f}x, p={arm['p_value_vs_rest']:.3g})")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
