"""Run the comparison: does fitting the bound as a bound beat regressing it?

Everything is trained in the *shallow world* -- Materials Project and OQMD, with
Alexandria's mass-substitution campaign held out -- and scored on whether it
predicted the formulas Alexandria later put below the shallow hull. Training and
evaluation therefore stand in the same relation to each other as a real screening
run stands to the future: the model sees a state of knowledge, and is judged on
what the next search found.

Five models, one split, one answer key:

* the censored ensemble, which estimates the floor under the shallow bound;
* ``g_D``, the same encoder under MSE on every shallow formula;
* ``g_C``, the same encoder under MSE on the formulas whose experimentally
  observed structure is the shallow archive's own minimum;
* ``g_D - g_C``, the headroom signal the two were proposed to produce;
* Magpie descriptors under gradient boosting, and the chemical-system lookup.

Each is scored with the naive triage rule and with Wren's uncertainty-adjusted
one, because the difference between them was worth 15 points of precision in the
only prospective campaign anyone has published.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from wyckoff_transformer.csp import parse_formula
from wyckoff_transformer.formula_energy import answer_key as ak
from wyckoff_transformer.formula_energy import baselines, metrics
from wyckoff_transformer.formula_energy import train as T

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("data/formula_energy/experiment")


#: Budgets for the ranking comparison: how many structure searches one could afford.
BUDGETS = (100, 500, 1_000, 5_000)


def _threshold_rows(
    name: str,
    location: np.ndarray,
    hull: np.ndarray,
    discovered: np.ndarray,
    sigma: Optional[np.ndarray],
    truth: Optional[np.ndarray] = None,
) -> List[Dict[str, float]]:
    """A model that estimates a floor, scored under both triage rules.

    Only models producing an energy on the same scale as the hull appear here.
    ``g_D - g_C`` does not: in the proposal it is evidence that headroom exists,
    not an estimate of where the floor is, so it is scored by ranking instead.
    """
    rows = []
    for rule, spread in (("naive", None), ("uncertainty-adjusted", sigma)):
        if rule == "uncertainty-adjusted" and spread is None:
            continue
        row = {"model": name, "rule": rule}
        row.update(metrics.screening_metrics(location, hull, discovered, epistemic_sigma=spread))
        if truth is not None:
            row["mae_vs_deep_min"] = float(np.abs(location - truth).mean())
        rows.append(row)
    return rows


def _ranking_rows(name: str, rank_score: np.ndarray, discovered: np.ndarray) -> List[Dict[str, float]]:
    """Every model, ranked, at a budget it could actually be run at.

    This is the comparison that includes the proposal's headroom signal on equal
    terms: whatever a model produces, sort by it and count how many of the top k
    were real discoveries. Lower ``rank_score`` must mean more promising.
    """
    # A budget larger than the candidate pool is meaningless, and on a small test
    # split every fixed budget can be. Fall back to the whole pool so the ranking
    # comparison still reports something.
    budgets = [budget for budget in BUDGETS if budget <= len(rank_score)] or [len(rank_score)]
    curve = metrics.enrichment_curve(rank_score, discovered, budgets=budgets)
    return [{"model": name, **row} for row in curve.to_dict("records")]


def run(
    shallow: pd.DataFrame,
    key: pd.DataFrame,
    device: torch.device,
    n_models: int = 10,
    epochs: int = 40,
    quick: bool = False,
    noise: Optional[float] = None,
) -> pd.DataFrame:
    """Train everything and score it against the answer key on the test split."""
    scored = shallow.index.intersection(key.index)
    shallow = shallow.loc[scored]
    key = key.loc[scored]
    # Derived from the formulas rather than read from a column: the pad width has
    # to be shared across the three splits and the baselines' own subsets, so it
    # is a property of the corpus, and deriving it keeps this runnable on any
    # table with the right index.
    max_elements = max(len(parse_formula(formula)) for formula in shallow.index)

    train_rows = shallow["split"] == "train"
    val_rows = shallow["split"] == "val"
    test_rows = shallow["split"] == "test"
    logger.info("train %d, val %d, test %d formulas",
                int(train_rows.sum()), int(val_rows.sum()), int(test_rows.sum()))

    common = dict(d_model=128 if quick else 256, n_layers=2 if quick else 3,
                  epochs=3 if quick else epochs, batch_size=1024)
    if noise is not None:
        common["noise"] = noise
    train_data = T.prepare(shallow[train_rows], max_elements=max_elements).to(device)
    val_data = T.prepare(shallow[val_rows], max_elements=max_elements).to(device)
    test_data = T.prepare(shallow[test_rows], max_elements=max_elements).to(device)

    test_key = key[test_rows]
    hull = test_key["shallow_e_hull"].to_numpy()
    discovered = test_key["discovered"].to_numpy()
    deep_min = test_key["deep_e_form_min"].to_numpy()
    rows: List[Dict[str, float]] = []
    ranked: List[Dict[str, float]] = []

    logger.info("=== censored ensemble ===")
    censored_models, _ = T.train_ensemble(
        train_data, val_data, T.TrainConfig(loss="censored", **common),
        device, n_models=1 if quick else n_models,
    )
    censored = T.predict(censored_models, test_data)
    rows += _threshold_rows("censored", censored["location"].to_numpy(), hull, discovered,
                            censored["sigma_epistemic"].to_numpy(), deep_min)
    ranked += _ranking_rows(
        "censored",
        censored["location"].to_numpy() + censored["sigma_epistemic"].to_numpy() - hull, discovered)

    logger.info("=== g_D: MSE on every formula ===")
    g_d_models, _ = T.train_ensemble(
        train_data, val_data, T.TrainConfig(loss="mse", **common),
        device, n_models=1 if quick else min(n_models, 5),
    )
    g_d = T.predict(g_d_models, test_data)
    rows += _threshold_rows("g_D", g_d["location"].to_numpy(), hull, discovered,
                            g_d["sigma_epistemic"].to_numpy(), deep_min)
    ranked += _ranking_rows("g_D", g_d["location"].to_numpy() - hull, discovered)

    logger.info("=== g_C: MSE where the observed structure is the archive minimum ===")
    exact = shallow["has_icsd"] & (shallow["icsd_excess"].fillna(np.inf) <= 1e-9)
    logger.info("g_C support: %d formulas (%.3f%% of the corpus)",
                int(exact.sum()), 100 * exact.mean())
    g_c_models, _ = baselines.fit_neural_baseline(
        shallow, "train", "val", T.TrainConfig(loss="mse", **common), device,
        subset=exact, max_elements=max_elements,
    )
    g_c = T.predict(g_c_models, test_data)
    rows += _threshold_rows("g_C", g_c["location"].to_numpy(), hull, discovered, None, deep_min)
    ranked += _ranking_rows("g_C", g_c["location"].to_numpy() - hull, discovered)

    # The proposal's headroom signal, scored the way it was meant: a large
    # positive g_D - g_C says the all-data fit sits well above the experimentally
    # anchored one, so there is room underneath. Negated, because the ranking
    # convention is that lower is more promising.
    ranked += _ranking_rows(
        "g_D - g_C", -baselines.headroom_signal(g_d, g_c).to_numpy(), discovered)

    logger.info("=== Magpie + gradient boosting ===")
    gbdt = baselines.fit_gbdt(
        baselines.magpie_features(train_data.to(torch.device("cpu"))),
        shallow.loc[train_rows, "e_form_min"].to_numpy(),
        max_iter=50 if quick else 400,
    )
    magpie = gbdt.predict(baselines.magpie_features(test_data.to(torch.device("cpu"))))
    rows += _threshold_rows("magpie+gbdt", magpie, hull, discovered, None, deep_min)
    ranked += _ranking_rows("magpie+gbdt", magpie - hull, discovered)

    logger.info("=== chemical-system lookup ===")
    lookup = baselines.chemsys_mean_lookup(shallow[train_rows], shallow[test_rows])
    rows += _threshold_rows("chemsys mean", lookup, hull, discovered, None, deep_min)
    ranked += _ranking_rows("chemsys mean", lookup - hull, discovered)

    frame = pd.DataFrame(rows)
    frame.attrs["ranking"] = ranked
    frame.attrs["calibration"] = metrics.expected_calibration_error(
        metrics.probability_below_hull(
            censored["location"].to_numpy(), censored["sigma_epistemic"].to_numpy(), hull),
        discovered,
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shallow-table", type=Path, default=ak.SHALLOW_TABLE)
    parser.add_argument("--answer-key", type=Path, default=ak.ANSWER_KEY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", type=torch.device, default=None)
    parser.add_argument("--models", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--noise", type=float, default=None,
                        help="label-noise scale for the censored loss; see TrainConfig.noise")
    parser.add_argument("--quick", action="store_true",
                        help="one small model and three epochs, to check the wiring")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    device = args.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)
    frame = run(
        pd.read_parquet(args.shallow_table), pd.read_parquet(args.answer_key),
        device, args.models, args.epochs, args.quick, args.noise,
    )
    ranking = pd.DataFrame(frame.attrs["ranking"])
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "thresholds.csv", index=False)
    ranking.to_csv(args.out / "ranking.csv", index=False)
    (args.out / "extras.json").write_text(json.dumps(
        {"calibration_error": frame.attrs["calibration"]}, indent=2))
    print("--- triage rules ---")
    print(frame.to_string(index=False))
    print("\n--- enrichment at a fixed budget ---")
    print(ranking.pivot(index="model", columns="budget", values="enrichment").to_string())
    print(f"\nexpected calibration error (censored): {frame.attrs['calibration']:.4f}")


if __name__ == "__main__":
    main()
