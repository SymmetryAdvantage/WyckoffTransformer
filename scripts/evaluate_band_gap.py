"""Test-split metrics of a gene-level band-gap regressor, through the screening path.

The trainer's own ``loss.epoch.test.mae`` scores one random augmentation per row. This
scores the test split the way ``cli.gene_screen.predict_gene_scalars`` scores a
generated gene -- the cache's records tokenised by ``build_tokenised_prediction_tensors``
and averaged over ``--augmentation-samples`` equivalent descriptions -- and adds what a
band-gap *screen* needs: precision and recall of "gap >= threshold".

    .venv/bin/python scripts/evaluate_band_gap.py ksvnn3oj mp_2026_gga_gap --device cuda
    .venv/bin/python scripts/evaluate_band_gap.py ksvnn3oj mp_20260928_r2scan_gap  # cross
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import torch

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT
from wyckoff_transformer.cli.csp import load_trainer
from wyckoff_transformer.dataset_cache import dataset_cache_dir, load_split
from wyckoff_transformer.prediction import (
    build_tokenised_prediction_tensors,
    filter_supported_tokens,
)

RECORD_COLUMNS = ["spacegroup_number", "elements", "site_symmetries", "sites_enumeration",
                  "multiplicity", "site_symmetries_augmented", "sites_enumeration_augmented",
                  "composition", "band_gap"]


def metrics(prediction: np.ndarray, target: np.ndarray, thresholds) -> dict:
    out = {"n": len(target),
           "mae": float(np.abs(prediction - target).mean()),
           "rmse": float(np.sqrt(((prediction - target) ** 2).mean())),
           "mae_gap_above_0": float(np.abs(prediction - target)[target > 0].mean())}
    for threshold in thresholds:
        predicted, actual = prediction >= threshold, target >= threshold
        tp = int((predicted & actual).sum())
        out[f"precision@{threshold}"] = tp / max(int(predicted.sum()), 1)
        out[f"recall@{threshold}"] = tp / max(int(actual.sum()), 1)
        out[f"accuracy@{threshold}"] = float((predicted == actual).mean())
    return out


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wandb_run")
    parser.add_argument("dataset", help="Dataset whose test split to score.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--augmentation-samples", type=int, default=8)
    parser.add_argument("--batch", type=int, default=5000)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--out", default=None, help="Write the metrics here as JSON.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    regressor = load_trainer(device=args.device, wandb_run=args.wandb_run,
                             wandb_entity=WANDB_ENTITY, wandb_project=WANDB_PROJECT)
    if hasattr(regressor.model, "_orig_mod"):
        regressor.model = regressor.model._orig_mod
    regressor.model.eval()
    records = load_split(dataset_cache_dir(args.dataset), args.split, columns=RECORD_COLUMNS)
    supported, dropped = filter_supported_tokens(records, regressor)
    predictions = []
    for start in range(0, len(supported), args.batch):
        part = supported.iloc[start:start + args.batch]
        tensors = build_tokenised_prediction_tensors(part, regressor)
        mean, _ = regressor.predict_scalars(
            tensors, augmentation_samples=args.augmentation_samples)
        predictions.append(mean.detach().float().cpu().numpy())
    prediction = np.concatenate(predictions)
    result = {"wandb_run": args.wandb_run, "dataset": args.dataset, "split": args.split,
              "augmentation_samples": args.augmentation_samples,
              "outside_vocabulary": len(dropped),
              **metrics(prediction, supported["band_gap"].to_numpy(dtype=float),
                        args.thresholds)}
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
