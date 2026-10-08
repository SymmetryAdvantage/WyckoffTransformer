#!/usr/bin/env python3
"""Censored vs MSE gene-energy predictors, scored against each gene's observed minimum.

    python scripts/evaluate_censored_energy.py <out_dir> <run_id>... [--device cuda]

Every run is asked for every val and test structure of lemat_bulk_fmax1_stress, tokenised
from its original Wyckoff description by the run's own processor. The reference is
gene_min_formation_energy_per_atom: the lowest formation energy LeMat-Bulk holds for the
gene (all splits). That is the MSE models' training target and an upper bound on what the
censored model estimates, m(g) = min(E | gene) -- so the two are expected to differ in a
specific way: the censored location should sit at the observed minimum for well-sampled
genes and below it for genes seen once.

Rows are stratified by
- n_obs: how many LeMat-Bulk structures share the gene, counted over all splits by the
  proxy (reduced composition, gene minimum) -- every row of a gene carries the same minimum,
  and two genes of one composition with bit-identical minima are vanishingly rare;
- icsd_backed (lemat_bulk_fmax1_stress_icsd).

Per model and stratum: MAE and mean signed error against the minimum, the share of rows
whose prediction lies above the minimum by more than 25 meV/atom (a floor above a structure
that exists), and Spearman correlation with the minimum.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from wyckoff_transformer.cli.csp import load_trainer
from wyckoff_transformer.dataset_cache import dataset_cache_dir, load_split
from wyckoff_transformer.paths import runs_root
from wyckoff_transformer.prediction import build_tokenised_prediction_tensors, filter_supported_tokens

DATASET = "lemat_bulk_fmax1_stress"
TARGET = "gene_min_formation_energy_per_atom"
COLUMNS = ["spacegroup_number", "elements", "site_symmetries", "sites_enumeration",
           "multiplicity", "composition", TARGET]
CHUNK = 20000
ABOVE = 0.025


def composition_key(composition) -> str:
    items = composition.items() if isinstance(composition, dict) else composition
    counts = {str(k): int(v) for k, v in items}
    g = np.gcd.reduce(list(counts.values()))
    return ",".join(f"{k}{v // g}" for k, v in sorted(counts.items()))


def gene_counts(frames: dict[str, pd.DataFrame]) -> pd.Series:
    keys = []
    for frame in frames.values():
        keys.append(frame["composition"].map(composition_key) + "|" + frame[TARGET].map(repr))
    return pd.concat(keys).value_counts()


def predict(run_id: str, frame: pd.DataFrame, device: torch.device) -> pd.Series:
    trainer = load_trainer(device, model_path=runs_root() / run_id)
    out = pd.Series(np.nan, index=frame.index)
    for start in range(0, len(frame), CHUNK):
        chunk = frame.iloc[start:start + CHUNK]
        supported, _ = filter_supported_tokens(chunk, trainer)
        tensors = build_tokenised_prediction_tensors(supported, trainer)
        with torch.no_grad():
            mean, _ = trainer.predict_scalars(tensors, augmentation_samples=1)
        out.loc[supported.index] = mean.float().cpu().numpy()
    return out


def metrics(pred: pd.Series, ref: pd.Series) -> dict:
    ok = pred.notna() & ref.notna()
    p, r = pred[ok], ref[ok]
    err = p - r
    return {"n": int(ok.sum()), "mae": float(err.abs().mean()), "bias": float(err.mean()),
            "above_min": float((err > ABOVE).mean()),
            "spearman": float(spearmanr(p, r).statistic) if len(p) > 2 else float("nan")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cache = dataset_cache_dir(DATASET)
    counts_frames = {s: load_split(cache, s, columns=["composition", TARGET]) for s in ("train", "val", "test")}
    counts = gene_counts(counts_frames)
    del counts_frames
    icsd_cache = dataset_cache_dir(f"{DATASET}_icsd")

    frames = []
    for split in ("val", "test"):
        frame = load_split(cache, split, columns=COLUMNS).dropna(subset=[TARGET])
        frame["icsd_backed"] = load_split(icsd_cache, split, columns=["icsd_backed"])["icsd_backed"].reindex(frame.index)
        frame["n_obs"] = (frame["composition"].map(composition_key) + "|" + frame[TARGET].map(repr)).map(counts)
        frame["split"] = split
        frames.append(frame)
    frame = pd.concat(frames)
    frame.index = frame["split"] + ":" + frame.index.astype(str)

    for run_id in args.runs:
        print(f"predicting with {run_id}", flush=True)
        frame[run_id] = predict(run_id, frame, device)

    frame["n_obs_bin"] = pd.cut(frame["n_obs"], [0, 1, 4, 10**9], labels=["1", "2-4", ">=5"])
    rows = []
    strata = {"all": pd.Series(True, index=frame.index)}
    for flag in (0.0, 1.0):
        strata[f"icsd={int(flag)}"] = frame["icsd_backed"] == flag
    for label in ("1", "2-4", ">=5"):
        strata[f"n_obs={label}"] = frame["n_obs_bin"] == label
    for split in ("val", "test"):
        for name, mask in strata.items():
            sub = frame[(frame["split"] == split) & mask]
            for run_id in args.runs:
                rows.append({"split": split, "stratum": name, "run": run_id, **metrics(sub[run_id], sub[TARGET])})
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "metrics.csv", index=False)
    frame.drop(columns=["elements", "site_symmetries", "sites_enumeration", "multiplicity", "composition"]).to_parquet(
        args.out_dir / "predictions.parquet")
    (args.out_dir / "runs.json").write_text(json.dumps(args.runs))
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table.pivot_table(index=["split", "stratum"], columns="run",
                                values=["n", "mae", "bias", "above_min", "spearman"]).round(4).to_string())


if __name__ == "__main__":
    main()
