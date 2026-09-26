"""Rank generated Wyckoff genes against the fixed LeMat-Bulk DFT hull.

This composes two existing, deliberately distinct estimators:

* the formula ensemble estimates ``f*(X)``, the latent energy floor at a
  composition, from provenance-aware censored bounds;
* the gene regressor estimates the lowest PBE formation energy observed for a
  Wyckoff gene, used as that gene's attainable energy.

Both predictions are compared independently with the same immutable PBE hull,
and all four energy inputs -- the two models' targets, the formula table and the
hull reference -- must carry the same energy definition
(:func:`check_energy_fields`; ``docs/energy_fields.md``).
The conservative joint score is their maximum, so a joint pass requires both
estimators to put the candidate below the hull.  No MLIP energies, relaxation,
or active-learning updates are used here.

Example::

    wyformer-dft-screen genes.json.gz --formula-ensemble runs/formula_energy/ensemble.pt --regressor-path runs/gene_energy --out dft_screen.csv --top 1000
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence, cast

import numpy as np
import pandas as pd
import torch

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT
from wyckoff_transformer.cli import gene_screen
from wyckoff_transformer.cli.csp import load_trainer
from wyckoff_transformer.dataset_manifest import file_formation_energy_field
from wyckoff_transformer.energy_fields import EnergyField, check_compatible
from wyckoff_transformer.evaluation.protocol import load_genes
from wyckoff_transformer.field_provenance import target_field
from wyckoff_transformer.formula_energy import train as formula_train
from wyckoff_transformer.formula_energy.dataset import DEFAULT_TABLE
from wyckoff_transformer.formula_energy.features import SYSTEM_FEATURES, SystemDensity
from wyckoff_transformer.formula_energy.metrics import probability_below_hull
from wyckoff_transformer.formula_energy.prefilter import reduced_keys
from wyckoff_transformer.paths import resolve_store_path

logger = logging.getLogger(__name__)

DEFAULT_FORMULA_ENSEMBLE = Path("runs/formula_energy/ensemble.pt")
DEFAULT_REFERENCE = gene_screen.DEFAULT_REFERENCE
RANKING_COLUMNS = (
    "composition_score_naive",
    "composition_score_adjusted",
    "gene_score",
    "joint_score_naive",
    "joint_score_adjusted",
)


def validate_formula_ensemble(
    config,
    feature_names: Sequence[str],
) -> None:
    """Require a DFT floor estimator usable on never-computed formulas.

    Its energy definition is checked separately, by :func:`check_energy_fields`.
    """
    if config.loss != "censored":
        raise ValueError(
            "The DFT attack screen requires a censored formula ensemble estimating "
            "the latent composition floor; an MSE ensemble estimates the observed "
            "archive bound instead."
        )
    missing = set(config.location_features) - set(feature_names)
    if missing:
        raise ValueError(
            "The formula checkpoint names location features absent from its saved "
            f"feature layout: {sorted(missing)}"
        )
    unavailable = set(config.location_features) - set(SYSTEM_FEATURES)
    if unavailable:
        raise ValueError(
            "The formula floor reads candidate-specific provenance that a novel "
            f"composition cannot supply: {sorted(unavailable)}. Only chemistry and "
            "system-neighbourhood features are valid for this screen."
        )


def check_energy_fields(
    gene_regressor,
    formula_provenance,
    formula_table_path: Path,
    reference_path: Path,
    allow_incompatible_energy: bool = False,
) -> list[str]:
    """Refuse to combine energies that do not share one definition.

    The hull reference fixes the definition; the gene regressor's target, the
    formula ensemble's target and the formula table must each be a formation
    energy on the same DFT settings, correction and reference entry set.  A file
    no dataset manifest names, or a model with no recorded or inferable
    provenance, has an unknown definition and is refused too.

    Returns:
        The differences, when ``allow_incompatible_energy`` let them through.
    """
    reference = file_formation_energy_field(reference_path)
    formula_target = None
    if formula_provenance is not None and formula_provenance["fields"].get("target"):
        formula_target = EnergyField.from_dict(formula_provenance["fields"]["target"])
    lines = []
    for what, field in (
            ("gene regressor target", target_field(gene_regressor.field_provenance or {})),
            ("formula ensemble target", formula_target),
            (f"formula table {formula_table_path}",
             file_formation_energy_field(formula_table_path))):
        lines += check_compatible(
            reference, field, f"DFT hull reference {reference_path} vs {what}",
            allow=allow_incompatible_energy)
    return lines


def _verdict(score: pd.Series) -> pd.Series:
    """Nullable on-or-below-hull verdict for a score in eV/atom."""
    result = pd.Series(pd.NA, index=score.index, dtype="boolean")
    usable = score.notna()
    result.loc[usable] = score.loc[usable] <= 0.0
    return result


def _formula_hulls(frame: pd.DataFrame) -> pd.Series:
    """One fixed-hull value per reduced formula, rejecting inconsistent inputs."""
    usable = frame[frame["reduced_formula"].notna() & frame["hull_energy"].notna()]
    if usable.empty:
        return pd.Series(dtype=float)
    grouped = usable.groupby("reduced_formula", sort=False)["hull_energy"]
    maximum = cast(pd.Series, grouped.max())
    minimum = cast(pd.Series, grouped.min())
    spread = maximum - minimum
    if (spread > 1e-6).any():
        offenders = spread[spread > 1e-6].nlargest(3).to_dict()
        raise ValueError(
            "The fixed DFT hull is inconsistent within a reduced formula; "
            f"worst spreads: {offenders}"
        )
    return cast(pd.Series, grouped.first())


def predict_composition_floors(
    gene_scores: pd.DataFrame,
    models: Sequence,
    config,
    feature_names: Sequence[str],
    formula_table: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    """Predict the composition floor once for each formula in ``gene_scores``."""
    validate_formula_ensemble(config, feature_names)
    frame = gene_scores.copy()
    frame["reduced_formula"] = reduced_keys(frame["formula"].tolist())
    hulls = _formula_hulls(frame)
    if hulls.empty:
        return pd.DataFrame(
            columns=pd.Index(np.asarray(
                ["location", "sigma_epistemic", "scale", "target", "hull"],
                dtype=object,
            )),
            index=pd.Index(np.asarray([], dtype=object), name="reduced_formula"),
        )

    formulas = [str(formula) for formula in hulls.index]
    system = None
    if set(feature_names) & set(SYSTEM_FEATURES):
        density = SystemDensity.from_table(formula_table)
        system = density.columns(formulas)
    data = formula_train.prepare_formulas(
        formulas,
        hull=hulls.to_numpy(dtype=float),
        system=system,
        feature_names=feature_names,
    ).to(device)
    return formula_train.predict(models, data)


def combine_scores(
    gene_scores: pd.DataFrame,
    composition_predictions: pd.DataFrame,
    formula_table: pd.DataFrame,
    rank_by: str = "joint_score_adjusted",
) -> pd.DataFrame:
    """Keep both estimands separate and form conservative joint DFT-hull scores."""
    if rank_by not in RANKING_COLUMNS:
        raise ValueError(f"Unknown ranking column {rank_by!r}; expected one of {RANKING_COLUMNS}")

    frame = gene_scores.copy()
    frame["reduced_formula"] = reduced_keys(frame["formula"].tolist())
    reduced_formula = cast(pd.Series, frame["reduced_formula"])
    frame["formula_known"] = reduced_formula.isin(formula_table.index.to_numpy(copy=False))

    renamed = composition_predictions.rename(columns={
        "location": "composition_floor",
        "sigma_epistemic": "composition_sigma_epistemic",
        "scale": "composition_excess_scale",
    })
    for column in (
        "composition_floor",
        "composition_sigma_epistemic",
        "composition_excess_scale",
    ):
        if column in renamed:
            prediction_column = cast(pd.Series, renamed[column])
            frame[column] = reduced_formula.map(prediction_column)
        else:
            frame[column] = np.nan

    # Explicit aliases make the estimands visible without changing the output of
    # the existing gene-only command.
    frame["dft_hull_energy"] = frame["hull_energy"]
    if "e_hull_at_composition" in formula_table:
        table_hull = reduced_formula.map(cast(pd.Series, formula_table["e_hull_at_composition"]))
        comparable = table_hull.notna() & frame["dft_hull_energy"].notna()
        disagreement = (
            table_hull[comparable] - frame.loc[comparable, "dft_hull_energy"]
        ).abs()
        if (disagreement > 1e-6).any():
            worst = disagreement.nlargest(3).to_dict()
            raise ValueError(
                "The formula table and fixed DFT reference use inconsistent hull "
                f"energies; worst absolute differences: {worst}"
            )
    frame["gene_attainable_energy"] = frame["predicted_formation_energy"]
    frame["gene_score"] = frame["gene_attainable_energy"] - frame["dft_hull_energy"]
    frame["composition_score_naive"] = (
        frame["composition_floor"] - frame["dft_hull_energy"]
    )
    frame["composition_score_adjusted"] = (
        frame["composition_floor"]
        + frame["composition_sigma_epistemic"]
        - frame["dft_hull_energy"]
    )
    frame["composition_p_below_hull"] = probability_below_hull(
        frame["composition_floor"].to_numpy(dtype=float),
        frame["composition_sigma_epistemic"].to_numpy(dtype=float),
        frame["dft_hull_energy"].to_numpy(dtype=float),
    )

    # np.maximum deliberately propagates NaN. A missing component is not evidence
    # that the other estimator alone passed the joint screen.
    frame["joint_score_naive"] = np.maximum(
        frame["composition_score_naive"].to_numpy(dtype=float),
        frame["gene_score"].to_numpy(dtype=float),
    )
    frame["joint_score_adjusted"] = np.maximum(
        frame["composition_score_adjusted"].to_numpy(dtype=float),
        frame["gene_score"].to_numpy(dtype=float),
    )

    for column in RANKING_COLUMNS:
        frame[f"{column}_below_hull"] = _verdict(cast(pd.Series, frame[column]))
    return frame.sort_values(rank_by, kind="stable", na_position="last")


def score_dft_genes(
    genes: Sequence[dict],
    gene_regressor,
    formula_models: Sequence,
    formula_config,
    formula_feature_names: Sequence[str],
    reference: pd.DataFrame,
    formula_table: pd.DataFrame,
    device: torch.device,
    augmentation_samples: int = 1,
    rank_by: str = "joint_score_adjusted",
) -> pd.DataFrame:
    """Score genes with the two DFT-trained estimators against one fixed PBE hull.

    The caller checks first that their energies agree (:func:`check_energy_fields`).
    """
    gene_scores = gene_screen.score_genes(
        genes,
        gene_regressor,
        reference,
        augmentation_samples=augmentation_samples,
    )
    composition_predictions = predict_composition_floors(
        gene_scores,
        formula_models,
        formula_config,
        formula_feature_names,
        formula_table,
        device,
    )
    return combine_scores(gene_scores, composition_predictions, formula_table, rank_by)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("genes", type=Path, help="JSON or JSON.GZ list of PyXtal-notation genes")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--regressor-path", type=Path,
                        help="directory holding the DFT gene-energy regressor")
    source.add_argument("--regressor-wandb-run", type=str,
                        help="W&B run holding the DFT gene-energy regressor")
    parser.add_argument("--wandb-entity", type=str, default=WANDB_ENTITY)
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--formula-ensemble", type=Path, default=DEFAULT_FORMULA_ENSEMBLE,
                        help="censored DFT formula-floor ensemble")
    parser.add_argument("--formula-table", type=Path, default=DEFAULT_TABLE,
                        help="formula table used to train the floor model")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                        help="immutable LeMat-Bulk PBE CSV defining the DFT hull")
    parser.add_argument("--out", type=Path, default=Path("dft_screen.csv"))
    parser.add_argument("--rank-by", choices=RANKING_COLUMNS,
                        default="joint_score_adjusted")
    parser.add_argument("--top", type=int, default=None,
                        help="write only the highest-ranked candidates")
    parser.add_argument("--joint-below-hull-only", action="store_true",
                        help="keep only candidates passing the adjusted joint screen")
    parser.add_argument("--augmentation-samples", type=int, default=1)
    parser.add_argument("--device", type=torch.device, default=None)
    parser.add_argument(
        "--allow-incompatible-energy", "--allow-unverified-energy-scale",
        dest="allow_incompatible_energy",
        action="store_true",
        help="combine energy inputs whose definitions differ or are unknown; the "
             "differences are logged. --allow-unverified-energy-scale is the old name.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(message)s",
    )
    device = args.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gene_regressor = load_trainer(
        device=device,
        model_path=args.regressor_path,
        wandb_run=args.regressor_wandb_run,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    )
    formula_models, formula_config, feature_names = formula_train.load_ensemble(
        args.formula_ensemble,
        device,
    )
    check_energy_fields(
        gene_regressor,
        formula_train.load_ensemble_field_provenance(args.formula_ensemble),
        args.formula_table,
        args.reference,
        allow_incompatible_energy=args.allow_incompatible_energy,
    )
    formula_table = pd.read_parquet(
        resolve_store_path(args.formula_table),
        columns=["e_form_min", "e_hull_at_composition", "chemsys"],
    )
    scored = score_dft_genes(
        load_genes(args.genes),
        gene_regressor,
        formula_models,
        formula_config,
        feature_names,
        gene_screen.load_reference(args.reference),
        formula_table,
        device,
        augmentation_samples=args.augmentation_samples,
        rank_by=args.rank_by,
    )
    if args.joint_below_hull_only:
        joint_verdict = cast(pd.Series, scored["joint_score_adjusted_below_hull"])
        scored = scored[joint_verdict.eq(True)]
    if args.top is not None:
        scored = scored.head(args.top)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out)
    joint_verdict = cast(pd.Series, scored["joint_score_adjusted_below_hull"])
    passing = int(joint_verdict.eq(True).sum())
    print(f"{passing} of {len(scored)} written genes pass the adjusted joint DFT screen: {args.out}")


if __name__ == "__main__":
    main()
