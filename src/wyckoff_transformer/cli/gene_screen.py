"""``wyformer-gene-screen``: rank generated Wyckoff genes against the PBE hull.

The scalar regressor predicts the lowest formation energy observed for a
Wyckoff gene.  Its ``max_force`` input is fixed to zero here because a generated
gene has not been relaxed yet; screening asks whether its clean realisation can
fall below the hull at its composition.

    wyformer-gene-screen genes.json.gz --regressor-path runs/gene_energy \\
        --reference data/lemat-bulk/lemat_pbe_ehull.csv.gz --out screened.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT
from wyckoff_transformer.cascade.dataset import TargetClass
from wyckoff_transformer.cli.csp import load_trainer
from wyckoff_transformer.evaluation.protocol import GeneFingerprinter, load_genes
from wyckoff_transformer.formula_energy.screen import HullLookup
from wyckoff_transformer.gene_energy import (
    GENE_MIN_FORMATION_ENERGY_COLUMN,
    build_clean_relaxation_condition,
)
from wyckoff_transformer.prediction import (
    build_tokenised_prediction_tensors,
    filter_supported_tokens,
)

logger = logging.getLogger(__name__)

DEFAULT_REFERENCE = Path("data/lemat-bulk/lemat_pbe_ehull.csv.gz")
REFERENCE_COLUMNS = ("immutable_id", "full_formula", "chemsys", "energy_corrected")


def _formula_from_record(record: dict) -> str:
    """Render a conventional-cell formula from a validated symmetry record."""
    parts = []
    for element, raw_count in sorted(
        record["composition"].items(),
        key=lambda item: str(item[0]),
    ):
        count = int(raw_count)
        if count <= 0 or count != raw_count:
            raise ValueError(f"Invalid composition count {raw_count!r} for {element!s}")
        symbol = getattr(element, "symbol", str(element))
        parts.append(f"{symbol}{count}")
    if not parts:
        raise ValueError("A Wyckoff gene has no occupied positions")
    return "".join(parts)


def load_reference(path: Path) -> pd.DataFrame:
    """Read the PBE entries required to construct convex hulls."""
    reference = pd.read_csv(path, usecols=list(REFERENCE_COLUMNS), low_memory=False)
    reference["energy_corrected"] = pd.to_numeric(
        reference["energy_corrected"],
        errors="coerce",
    )
    reference = reference.dropna(subset=["full_formula", "chemsys", "energy_corrected"])
    return reference.set_index("immutable_id")


def validate_regressor(regressor) -> None:
    """Reject checkpoints whose target cannot support the clean gene screen."""
    if regressor.target != TargetClass.Scalar:
        raise ValueError("The gene-energy screener requires a Scalar regressor.")
    if getattr(regressor, "scalar_loss", "mse") != "mse":
        raise ValueError(
            "The initial gene-energy screener requires scalar_loss='mse', fitted to "
            f"{GENE_MIN_FORMATION_ENERGY_COLUMN!r}.")
    if getattr(regressor, "target_name", None) != GENE_MIN_FORMATION_ENERGY_COLUMN:
        raise ValueError(
            "The regressor target must be "
            f"{GENE_MIN_FORMATION_ENERGY_COLUMN!r}, not "
            f"{getattr(regressor, 'target_name', None)!r}.")
    # This validates the exact condition-feature layout and keeps the force-at-zero
    # policy in one place shared with CSP reranking.
    build_clean_relaxation_condition(regressor, 1, device=regressor.device)


def score_genes(
    genes: Sequence[dict],
    regressor,
    reference: pd.DataFrame,
    augmentation_samples: int = 1,
) -> pd.DataFrame:
    """Predict clean formation energies and compare every usable gene to its hull."""
    validate_regressor(regressor)
    output = pd.DataFrame(index=pd.RangeIndex(len(genes), name="index"))
    output["formula"] = pd.NA
    output["predicted_formation_energy"] = np.nan
    output["hull_energy"] = np.nan
    output["score"] = np.nan
    output["predicted_below_hull"] = pd.Series(pd.NA, index=output.index, dtype="boolean")
    output["reason"] = pd.NA

    fingerprinter = GeneFingerprinter()
    records = []
    for index, gene in enumerate(genes):
        try:
            record = fingerprinter.record(gene)
            # A repeated element must have all its occupied positions summed.
            composition = {}
            for element, count in zip(record["elements"], record["multiplicity"]):
                composition[element] = composition.get(element, 0) + count
            record["composition"] = composition
            record["formula"] = _formula_from_record(record)
        except (KeyError, TypeError, ValueError) as error:
            output.loc[index, "reason"] = f"{type(error).__name__}: {error}"
            continue
        record["source_index"] = index
        records.append(record)

    if not records:
        return output
    records_frame = pd.DataFrame.from_records(records).set_index("source_index")
    supported, unsupported = filter_supported_tokens(records_frame, regressor)
    if unsupported:
        output.loc[unsupported, "reason"] = "Gene is outside the regressor vocabulary"
    if supported.empty:
        return output

    output.loc[supported.index, "formula"] = supported["formula"]
    tensors = build_tokenised_prediction_tensors(supported, regressor)
    cond = build_clean_relaxation_condition(regressor, len(supported), device=regressor.device)
    prediction, _ = regressor.predict_scalars(
        tensors,
        augmentation_samples=augmentation_samples,
        cond=cond,
    )
    output.loc[supported.index, "predicted_formation_energy"] = (
        prediction.detach().cpu().numpy()
    )

    lookup = HullLookup(reference)
    hulls = {}
    for formula in supported["formula"].unique():
        try:
            hulls[formula] = lookup.hull_energy_per_atom(formula)
        except (KeyError, ValueError) as error:
            hulls[formula] = np.nan
            logger.warning("No hull for %s: %s", formula, error)
    output.loc[supported.index, "hull_energy"] = supported["formula"].map(hulls)

    usable = output["predicted_formation_energy"].notna() & output["hull_energy"].notna()
    output.loc[usable, "score"] = (
        output.loc[usable, "predicted_formation_energy"]
        - output.loc[usable, "hull_energy"]
    )
    output.loc[usable, "predicted_below_hull"] = output.loc[usable, "score"] < 0
    return output.sort_values("score", kind="stable", na_position="last")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict clean formation energy for Wyckoff genes and compare it to the hull.",
    )
    parser.add_argument("genes", type=Path, help="JSON or JSON.GZ list of PyXtal-notation genes.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--regressor-path", type=Path, help="Directory holding the trained critic.")
    source.add_argument("--regressor-wandb-run", type=str, help="W&B run holding the critic.")
    parser.add_argument("--wandb-entity", type=str, default=WANDB_ENTITY)
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                        help="LeMat-Bulk PBE CSV defining the formation-energy hull.")
    parser.add_argument("--out", type=Path, default=Path("gene_screen.csv"))
    parser.add_argument("--top", type=int, default=None, help="Keep this many best scores.")
    parser.add_argument("--below-hull-only", action="store_true",
                        help="Write only genes whose predicted energy is below the hull.")
    parser.add_argument("--augmentation-samples", type=int, default=1)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    regressor = load_trainer(
        device=args.device,
        model_path=args.regressor_path,
        wandb_run=args.regressor_wandb_run,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    )
    scored = score_genes(
        load_genes(args.genes),
        regressor,
        load_reference(args.reference),
        augmentation_samples=args.augmentation_samples,
    )
    if args.below_hull_only:
        scored = scored[scored["predicted_below_hull"].eq(True)]
    if args.top is not None:
        scored = scored.head(args.top)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out)
    predicted = scored["predicted_below_hull"].eq(True).sum()
    print(f"{predicted} of {len(scored)} written genes are predicted below the hull: {args.out}")


if __name__ == "__main__":
    main()
