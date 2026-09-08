"""``wyformer-gene-novelty``: score generated genes by the generator's own surprisal.

The second lever of the two-lever screen.  ``wyformer-dft-screen`` says which
genes are likely to sit low against the fixed DFT hull; this says which of them
the model does not already produce by the thousand, and therefore which are
worth a relaxation slot in a campaign whose objective is *new* stable matter.

    wyformer-gene-novelty genes.json.gz --model-path runs/e9ywwsie \\
        --condition energy_above_hull=0 --out gene_novelty.csv

Score it at the condition the pool was generated at.  The density a sample came
from is the conditional one, and a novelty number read off a different
conditioning is a number about a different generator.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT
from wyckoff_transformer.cli import resolve_condition_values
from wyckoff_transformer.cli.csp import load_trainer
from wyckoff_transformer.evaluation.protocol import load_genes
from wyckoff_transformer.gene_likelihood import (
    LIKELIHOOD_COLUMNS,
    records_from_genes,
    score_gene_likelihood,
)

logger = logging.getLogger(__name__)


def score_genes(
    genes,
    trainer,
    condition_values=None,
    permutation_samples: int = 8,
    seed: int = 0,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """One row per sampled gene, unscorable ones kept with a reason."""
    records, reasons = records_from_genes(genes, trainer)
    output = pd.DataFrame(index=pd.RangeIndex(len(genes), name="index"))
    for column in LIKELIHOOD_COLUMNS:
        output[column] = float("nan")
    output["reason"] = reasons
    if records.empty:
        return output

    cond = None
    if trainer.condition_features:
        if condition_values is None:
            raise ValueError(
                f"This generator is conditioned on {list(trainer.condition_features)}. "
                "Pass --condition NAME=VALUE at the value the pool was generated at: an "
                "unconditional score would be a density the pool was not drawn from.")
        cond = trainer.build_condition_from_values(
            condition_values, len(records), device=trainer.device)
    scored = score_gene_likelihood(
        records,
        trainer,
        cond=cond,
        permutation_samples=permutation_samples,
        seed=seed,
        batch_size=batch_size,
    )
    output.loc[scored.index, list(LIKELIHOOD_COLUMNS)] = scored
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Wyckoff genes by the generative model's surprisal.")
    parser.add_argument("genes", type=Path, help="JSON or JSON.GZ list of PyXtal-notation genes.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-path", type=Path, help="Directory holding the generator.")
    source.add_argument("--model-wandb-run", type=str, help="W&B run holding the generator.")
    source.add_argument("--hf-model", type=str, help="HuggingFace model id.")
    parser.add_argument("--wandb-entity", type=str, default=WANDB_ENTITY)
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--condition", action="append", default=None, metavar="NAME=VALUE",
                        help="Conditioning value the pool was generated at; repeatable.")
    parser.add_argument("--condition-value", type=float, default=None,
                        help="Shorthand for a model conditioned on one feature.")
    parser.add_argument("--permutation-samples", type=int, default=8,
                        help="Representations drawn per gene; the bound tightens with this.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Genes per forward sweep (default: the whole pool).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("gene_novelty.csv"))
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    trainer = load_trainer(
        device=args.device,
        model_path=args.model_path,
        wandb_run=args.model_wandb_run,
        hf_model=args.hf_model,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    )
    condition_values = resolve_condition_values(
        trainer, args.condition, args.condition_value)
    genes = load_genes(args.genes)
    scored = score_genes(
        genes,
        trainer,
        condition_values=condition_values,
        permutation_samples=args.permutation_samples,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out)
    usable = scored["surprisal"].notna()
    print(f"{int(usable.sum())} of {len(scored)} genes scored: {args.out}")
    if usable.any():
        surprisal = scored.loc[usable, "surprisal"]
        spread = scored.loc[usable, "std_representation_log_likelihood"].median()
        print(f"surprisal (nats): median {surprisal.median():.2f}, "
              f"p5 {surprisal.quantile(0.05):.2f}, p95 {surprisal.quantile(0.95):.2f}")
        print(f"median spread across representation draws: {spread:.2f} nats")


if __name__ == "__main__":
    main()
