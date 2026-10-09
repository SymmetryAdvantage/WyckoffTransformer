"""Correct the energy ranking by the surprisal instead of cutting a surprisal band.

docs/generative_novelty_screen.md found that the generator's surprisal predicts
the energy regressor's error, mostly its optimism. A predictable bias can be
corrected rather than cut. Two corrections, both fitted on relaxed genes alone:
relaxations and the hull, which fire-control needs anyway, and no novelty label.

``shift``
    ``predicted e_hull + f(surprisal)``, with *f* the isotonic (non-decreasing)
    fit of the residual ``realized - predicted`` against surprisal. Ranks by
    the expected realized ``e_hull``.
``probability``
    ``P(metastable | predicted, surprisal)``, a logistic regression on
    ``(p, s, p*s, s^2)``, standardised. Ranks by the probability of the outcome
    MetaSUN counts, which a skewed, heteroscedastic error can rank differently
    from its mean.

A correction repairs the *upper* edge of the band: the improbable genes the
regressor is optimistic about. It is not a novelty lever, so it is tried three
ways:

* after the lookup, in place of the hard least-surprising-30% cut;
* after dropping the most typical 40% of the pool by surprisal (the band's lower
  edge), with no reference at all;
* alone, with no novelty lever, as the control.

Evaluated out of sample, two ways:

``cross-fit``
    the pool's unique genes split at random into halves; the correction is fitted
    on one half's relaxed genes, known and novel alike, and every arm is selected
    and scored on the other half at half the budget, which keeps the selection
    strength of the full pool at the full budget. 40 splits.
``transfer``
    fitted on one backbone's whole pool and applied to the other's.

    python scripts/analyse_surprisal_correction.py \\
        $WYFORMER_RUNS/roe_surprisal/cfg $WYFORMER_RUNS/roe_surprisal/uncond
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wyckoff_transformer.evaluation.protocol import METASTABLE_THRESHOLD

BUDGET = 1000
SPLITS = 40
#: Fixed in advance, as in analyse_roe_surprisal.py.
BAND = (0.40, 0.80)
TYPICAL_CUT = BAND[0]
LEAST_SURPRISING = 0.30
#: How many relaxed genes the calibration needs: subsamples of the fitting half.
CALIBRATION_SIZES = (250, 1000, None)


def load(root: Path, track: str) -> pd.DataFrame:
    directory = "analysis" if track == "free" else f"analysis_{track}"
    genes = pd.read_csv(root / directory / "genes.csv.gz", index_col=0)
    reps = genes[genes["is_representative"]].copy()
    column = "surprisal_w1" if "surprisal_w1" in reps else next(
        c for c in reps.columns if c.startswith("surprisal_"))
    reps["s"] = reps[column]
    reps["p"] = reps["predicted_e_hull"]
    reps["realized"] = reps["e_above_hull"]
    reps["metastable"] = reps["metastable"].astype(bool)
    reps["metasun"] = reps["metasun"].astype(bool)
    reps["sun"] = reps["sun"].astype(bool)
    reps["gene_novel"] = reps["gene_novel"].astype(bool)
    reps.attrs["surprisal"] = column
    return reps


# --------------------------------------------------------------------------- #
# Corrections
# --------------------------------------------------------------------------- #
def fit_shift(calibration: pd.DataFrame) -> Callable[[pd.DataFrame], pd.Series]:
    """Lower is better: the expected realized e_hull."""
    usable = calibration[["s", "p", "realized"]].dropna()
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(usable["s"], usable["realized"] - usable["p"])

    def score(part: pd.DataFrame) -> pd.Series:
        shift = pd.Series(model.predict(part["s"].fillna(part["s"].max())), index=part.index)
        return part["p"] + shift
    return score


def _features(frame: pd.DataFrame) -> np.ndarray:
    p, s = frame["p"].to_numpy(), frame["s"].to_numpy()
    return np.column_stack([p, s, p * s, s * s])


def _fit_logistic(calibration: pd.DataFrame, label: str) -> Callable[[pd.DataFrame], pd.Series]:
    usable = calibration[["s", "p"]].notna().all(axis=1)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(_features(calibration[usable]), calibration.loc[usable, label])

    def score(part: pd.DataFrame) -> pd.Series:
        ok = part[["s", "p"]].notna().all(axis=1)
        out = pd.Series(np.inf, index=part.index)
        out[ok] = -model.predict_proba(_features(part[ok]))[:, 1]
        return out
    return score


def fit_probability(calibration: pd.DataFrame) -> Callable[[pd.DataFrame], pd.Series]:
    """Lower is better: minus P(metastable | p, s).

    Every relaxed representative is a calibration row; one whose relaxation
    failed or whose structure is invalid is a non-metastable outcome, as the
    protocol counts it. No novelty label is read.
    """
    return _fit_logistic(calibration, "metastable")


def fit_metasun(calibration: pd.DataFrame) -> Callable[[pd.DataFrame], pd.Series]:
    """Lower is better: minus P(MetaSUN | p, s).

    The one correction that reads novelty: its labels were judged against the
    reference, once, on the calibration pool. Selection then needs no lookup.
    """
    return _fit_logistic(calibration, "metasun")


CORRECTIONS = {"shift": fit_shift, "probability": fit_probability, "metasun": fit_metasun}


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def _quantile(values: pd.Series, eligible: pd.Series) -> pd.Series:
    return values[eligible & values.notna()].rank(pct=True).reindex(values.index)


def _take(score: pd.Series, mask: pd.Series, budget: int) -> pd.Index:
    return score[mask].fillna(np.inf).sort_values(kind="stable").index[:budget]


def arms(part: pd.DataFrame, budget: int, scores: dict) -> dict:
    """Arm name -> the selected representatives, all within *part*."""
    everyone = pd.Series(True, index=part.index)
    novel = part["gene_novel"]
    q = _quantile(part["s"], everyone)
    q_novel = _quantile(part["s"], novel)
    band = (q > BAND[0]) & (q <= BAND[1])
    atypical = q > TYPICAL_CUT
    plausible = novel & (q_novel <= LEAST_SURPRISING)
    selected = {
        "energy": _take(part["p"], everyone, budget),
        "lookup": _take(part["p"], novel, budget),
        "lookup+plausible": _take(part["p"], plausible, budget),
        "surprisal band": _take(part["p"], band, budget),
    }
    for name, score in scores.items():
        corrected = score(part)
        selected[f"lookup+{name}"] = _take(corrected, novel, budget)
        selected[f"typical cut+{name}"] = _take(corrected, atypical, budget)
        selected[f"{name} alone"] = _take(corrected, everyone, budget)
    return selected


#: Which arms need the reference set, for the tables.
NEEDS_LOOKUP = ("lookup",)


def outcome(part: pd.DataFrame, chosen: pd.Index, budget: int) -> dict:
    rows = part.loc[chosen]
    return {
        "metasun": float(rows["metasun"].sum() / budget),
        "sun": float(rows["sun"].sum() / budget),
        "metastable": float(rows["metastable"].sum() / budget),
        "gene_novel": float(rows["gene_novel"].mean()) if len(rows) else 0.0,
        "filled": int(len(rows)),
    }


def cross_fit(reps: pd.DataFrame, budget: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    half = budget // 2
    results: dict = {}
    for _ in range(SPLITS):
        mask = rng.random(len(reps)) < 0.5
        fit_half, test_half = reps[mask], reps[~mask]
        for size in CALIBRATION_SIZES:
            calibration = fit_half if size is None else fit_half.sample(
                n=min(size, len(fit_half)), random_state=int(rng.integers(1 << 31)))
            scores = {name: fit(calibration) for name, fit in CORRECTIONS.items()}
            label = "all" if size is None else str(size)
            for arm, chosen in arms(test_half, half, scores).items():
                # The smaller calibrations only change the corrected arms.
                if size is not None and not any(c in arm for c in CORRECTIONS):
                    continue
                results.setdefault(label, {}).setdefault(arm, []).append(
                    outcome(test_half, chosen, half))
    summary = {}
    for label, by_arm in results.items():
        summary[label] = {}
        for arm, runs in by_arm.items():
            values = np.array([r["metasun"] for r in runs])
            summary[label][arm] = {
                "metasun_mean": float(values.mean()),
                "metasun_p10": float(np.quantile(values, 0.1)),
                "metasun_p90": float(np.quantile(values, 0.9)),
                "sun_mean": float(np.mean([r["sun"] for r in runs])),
                "gene_novel_mean": float(np.mean([r["gene_novel"] for r in runs])),
                "per_split": values.tolist(),
            }
    # Paired against the arm each correction is meant to replace, split by split.
    full = summary["all"]
    summary["paired"] = {}
    for new, old in (("lookup+shift", "lookup+plausible"), ("lookup+probability", "lookup+plausible"),
                     ("lookup+shift", "lookup"), ("lookup+probability", "lookup"),
                     ("typical cut+shift", "surprisal band"),
                     ("typical cut+probability", "surprisal band"),
                     ("typical cut+shift", "lookup"), ("typical cut+probability", "lookup")):
        diff = np.array(full[new]["per_split"]) - np.array(full[old]["per_split"])
        summary["paired"][f"{new} - {old}"] = {
            "mean": float(diff.mean()), "wins": int((diff > 0).sum()), "splits": len(diff)}
    return summary


def transfer(source: pd.DataFrame, target: pd.DataFrame, budget: int) -> dict:
    scores = {name: fit(source) for name, fit in CORRECTIONS.items()}
    return {arm: outcome(target, chosen, budget)
            for arm, chosen in arms(target, budget, scores).items()}


def in_sample(reps: pd.DataFrame, budget: int) -> dict:
    scores = {name: fit(reps) for name, fit in CORRECTIONS.items()}
    return {arm: outcome(reps, chosen, budget)
            for arm, chosen in arms(reps, budget, scores).items()}


# --------------------------------------------------------------------------- #
def markdown(report: dict) -> str:
    pools = list(report["pools"])
    arm_names = list(report["pools"][pools[0]]["cross_fit"]["all"])
    lines = [f"### Cross-fit: MetaSUN per slot, {SPLITS} half/half splits, "
             f"{report['budget'] // 2} slots per half (mean, p10-p90)\n"]
    lines.append("| arm | reference | " + " | ".join(pools) + " |")
    lines.append("|---|---|" + "---|" * len(pools))
    for arm in arm_names:
        cells = []
        for pool in pools:
            v = report["pools"][pool]["cross_fit"]["all"][arm]
            cells.append(f"{v['metasun_mean']:.3f} ({v['metasun_p10']:.3f}-{v['metasun_p90']:.3f})")
        needs = ("lookup" if arm.startswith(NEEDS_LOOKUP)
                 else "calibration" if "metasun" in arm else "no")
        lines.append(f"| {arm} | {needs} | " + " | ".join(cells) + " |")
    lines.append("\n### Paired, split by split\n")
    lines.append("| comparison | " + " | ".join(f"{p} mean (wins/{SPLITS})" for p in pools) + " |")
    lines.append("|---|" + "---|" * len(pools))
    for key in report["pools"][pools[0]]["cross_fit"]["paired"]:
        cells = [f"{report['pools'][p]['cross_fit']['paired'][key]['mean']:+.3f} "
                 f"({report['pools'][p]['cross_fit']['paired'][key]['wins']})" for p in pools]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines.append("\n### Calibration size: cross-fit MetaSUN with the correction fitted on n relaxed genes\n")
    sizes = [s for s in report["pools"][pools[0]]["cross_fit"] if s not in ("paired",)]
    corrected = [a for a in arm_names if any(c in a for c in CORRECTIONS)]
    lines.append("| arm | " + " | ".join(f"{p} n={s}" for p in pools for s in sizes) + " |")
    lines.append("|---|" + "---|" * (len(pools) * len(sizes)))
    for arm in corrected:
        cells = [f"{report['pools'][p]['cross_fit'][s][arm]['metasun_mean']:.3f}"
                 for p in pools for s in sizes]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.append(f"\n### Full pool, B = {report['budget']}: in-sample fit, and fitted on the other backbone\n")
    lines.append("| arm | " + " | ".join(f"{p} in-sample | {p} transferred" for p in pools) + " |")
    lines.append("|---|" + "---|" * (2 * len(pools)))
    for arm in arm_names:
        cells = []
        for p in pools:
            cells.append(f"{report['pools'][p]['in_sample'][arm]['metasun']:.3f}")
            cells.append(f"{report['pools'][p]['transferred'][arm]['metasun']:.3f}")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", type=Path, nargs=2,
                        help="Two pool roots replayed by analyse_roe_surprisal.py; each "
                             "is also the other's transfer source.")
    parser.add_argument("--track", choices=("free", "fixed_symmetry"), default="free")
    parser.add_argument("--budget", type=int, default=BUDGET,
                        help="Full-pool budget; the cross-fit spends half of it per half.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write correction_B<budget>.{json,md} (default: the "
                             "first root's analysis directory).")
    args = parser.parse_args(argv)

    pools = {root.name: load(root, args.track) for root in args.roots}
    names = list(pools)
    report = {"track": args.track, "budget": args.budget, "splits": SPLITS, "band": list(BAND),
              "metastable_threshold": METASTABLE_THRESHOLD, "pools": {}}
    for name in names:
        other = names[1] if name == names[0] else names[0]
        reps = pools[name]
        report["pools"][name] = {
            "surprisal": reps.attrs["surprisal"],
            "representatives": int(len(reps)),
            "cross_fit": cross_fit(reps, args.budget),
            "in_sample": in_sample(reps, args.budget),
            "transferred": transfer(pools[other], reps, args.budget),
            "transferred_from": other,
        }
    directory = "analysis" if args.track == "free" else f"analysis_{args.track}"
    out = args.out or args.roots[0] / directory
    stem = f"correction_B{args.budget}"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=1))
    text = markdown(report)
    (out / f"{stem}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
