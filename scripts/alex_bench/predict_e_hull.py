"""Predicted gene-minimum e_hull for every gene in a file, for validating the predictor.

Fire-control only sees predictions for the genes it ranks; here every arm's engaged genes
get one, so the predictor can be checked against ORB on unselected cohorts too.

    .venv/bin/python scripts/alex_bench/predict_e_hull.py GENES.json.gz OUT.csv \
        --regressor-path RUNS/gene_min_ehull_... [--device cuda]
"""
import argparse
from pathlib import Path

import torch

from wyckoff_transformer.cli.csp import load_trainer
from wyckoff_transformer.cli.gene_screen import predicts_e_hull, score_genes
from wyckoff_transformer.evaluation.protocol import load_genes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("genes", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--regressor-path", type=Path, required=True)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    args = parser.parse_args()
    regressor = load_trainer(device=args.device, model_path=args.regressor_path)
    if not predicts_e_hull(regressor):
        raise SystemExit(f"{args.regressor_path} does not predict e_hull directly")
    scored = score_genes(load_genes(args.genes), regressor, None).sort_index()
    scored = scored.rename(columns={"score": "predicted_e_hull"})
    scored[["formula", "predicted_e_hull", "reason"]].to_csv(args.out)
    print(f"{scored['predicted_e_hull'].notna().sum()}/{len(scored)} genes scored -> {args.out}")


if __name__ == "__main__":
    main()
