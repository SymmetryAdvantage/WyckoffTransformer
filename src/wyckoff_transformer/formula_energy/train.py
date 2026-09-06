"""Fit the censored-minimum model, as a deep ensemble.

The training loop is here rather than in :class:`~wyckoff_transformer.trainer.WyckoffTrainer`
because that class is built around cascade sequences over Wyckoff sites -- masked
positions, ``known_seq_len``, augmentation over equivalent settings -- and a
chemical formula has none of those. What is reused is the part that matters:
:class:`wyckoff_transformer.censored.CensoredMinLoss`, unchanged, because the
likelihood is the same one level up.

Ten models from different initialisations, which is Wren's protocol and the
reason it can screen at all. A single network gives a point estimate, and ranking
millions of candidates by a point estimate selects for the largest errors. The
ensemble's disagreement about where the floor lies is the epistemic term that
turns the ranking into a probability and lets the triage rule ask for a margin.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from torch import nn

from wyckoff_transformer.censored import CensoredMinLoss
from wyckoff_transformer.formula_energy.encoder import FormulaEnergyModel
from wyckoff_transformer.formula_energy.features import PROVENANCE_FEATURES, composition_tensors, provenance_tensor

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Everything that is not data. Defaults follow Wren where it had an opinion."""

    d_model: int = 256
    n_layers: int = 3
    n_heads: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    head_widths: Sequence[int] = (256, 256, 128, 64)
    detach_scale_trunk: bool = False
    #: ``"censored"`` fits the archive minimum as a bound and estimates the floor
    #: beneath it. ``"mse"`` regresses the bound itself, which is what the two
    #: comparison models g_C and g_D do -- same encoder, same features, same
    #: split, so the likelihood is the only thing that differs between them.
    loss: str = "censored"
    #: Label noise of the reference energies, eV/atom. **Not** the 0.01 that
    #: ``censored.DEFAULT_NOISE`` uses at the gene level: measured here, that
    #: value breaks the fit. It enters the likelihood as ``t = (observed -
    #: location) / noise``, so against formation energies spanning several eV it
    #: makes ``t`` of order 500 at initialisation; the model compensates by
    #: inflating the excess scale and drives the floor far below the data. A
    #: sweep on the shallow world, one model, 15 epochs, MAE against the deep
    #: minimum:
    #:
    #: === ======== ====== ========== ==============
    #: loss noise    MAE    violation  flagged below hull
    #: === ======== ====== ========== ==============
    #: mse  --       0.158  0.477      0.121
    #: cens 0.01     0.410  0.022      0.732
    #: cens 0.05     0.195  0.130      0.378
    #: cens 0.10     0.169  0.289      0.246
    #: cens 0.25     0.195  0.736      0.031
    #: === ======== ====== ========== ==============
    #:
    #: 0.10 is also the physically defensible figure: cross-source DFT
    #: disagreement across MP, OQMD and Alexandria is of that order, and the
    #: within-formula spread of ``e_hull`` in LeMat-Bulk has a median of 0.142.
    noise: float = 0.10
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    epochs: int = 40
    patience: int = 5
    clip_grad_norm: float = 1.0
    #: Provenance features to withhold, by name. Emptying the tuple is the
    #: ablation that asks whether the ICSD flag earns its place.
    drop_provenance: Sequence[str] = ()
    seed: int = 0


class MseOnLocation(nn.Module):
    """Plain MSE on the location column, wearing :class:`CensoredMinLoss`'s interface.

    Keeping the two-column head under an MSE objective means the censored and
    uncensored fits differ in exactly one thing -- the objective -- rather than in
    architecture, initialisation and output shape as well.
    """

    n_outputs = 2

    def split(self, prediction: Tensor) -> Tuple[Tensor, Tensor]:
        return prediction[..., 0], prediction[..., 1]

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.nn.functional.mse_loss(prediction[..., 0], target)


def make_criterion(config: "TrainConfig") -> nn.Module:
    """The objective named by ``config.loss``."""
    if config.loss == "censored":
        return CensoredMinLoss(noise=config.noise, predict_scale=True)
    if config.loss == "mse":
        return MseOnLocation()
    raise ValueError(f"Unknown loss: {config.loss!r}, expected 'censored' or 'mse'")


@dataclass
class FormulaData:
    """Padded tensors for one split, plus the labels and the hull to screen against."""

    element_ids: Tensor
    fractions: Tensor
    padding_mask: Tensor
    provenance: Tensor
    target: Tensor
    hull: Tensor
    formulas: pd.Index = field(default_factory=pd.Index)

    def __len__(self) -> int:
        return len(self.target)

    def to(self, device: torch.device) -> "FormulaData":
        return FormulaData(
            self.element_ids.to(device), self.fractions.to(device), self.padding_mask.to(device),
            self.provenance.to(device), self.target.to(device), self.hull.to(device), self.formulas,
        )

    def index(self, selection: Tensor) -> "FormulaData":
        return FormulaData(
            self.element_ids[selection], self.fractions[selection], self.padding_mask[selection],
            self.provenance[selection], self.target[selection], self.hull[selection], self.formulas,
        )


def prepare(
    table: pd.DataFrame,
    target_column: str = "e_form_min",
    hull_column: str = "e_hull_at_composition",
    max_elements: Optional[int] = None,
    drop_provenance: Sequence[str] = (),
    feature_names: Sequence[str] = PROVENANCE_FEATURES,
) -> FormulaData:
    """Formula table into tensors.

    Args:
        table: Output of :func:`~.dataset.build_formula_table`.
        target_column: The bound being fitted -- the archive's lowest energy.
        hull_column: What a discovery has to beat.
        max_elements: Pad width; must be shared across splits.
        drop_provenance: Names from :data:`~.features.PROVENANCE_FEATURES` to zero
            out, for ablations.
    """
    element_ids, fractions, padding_mask = composition_tensors(table.index, max_elements)
    provenance = provenance_tensor(table, feature_names)
    if drop_provenance:
        unknown = set(drop_provenance) - set(feature_names)
        if unknown:
            raise ValueError(f"Unknown provenance features: {sorted(unknown)}")
        for name in drop_provenance:
            provenance[:, list(feature_names).index(name)] = 0.0
    return FormulaData(
        element_ids, fractions, padding_mask, provenance,
        torch.tensor(table[target_column].to_numpy(), dtype=torch.float32),
        torch.tensor(table[hull_column].to_numpy(), dtype=torch.float32),
        table.index,
    )


def prepare_formulas(
    formulas: Sequence[str],
    hull: Optional[np.ndarray] = None,
    max_elements: Optional[int] = None,
    system: Optional[pd.DataFrame] = None,
    feature_names: Sequence[str] = PROVENANCE_FEATURES,
) -> FormulaData:
    """Tensors for formulas that are not in the archive.

    Provenance is all zeros, which is not a gap to be imputed but the truth:
    nobody has computed this composition, so no search process has contributed
    anything. The location head does not read provenance in any case, which is
    what makes a never-computed formula answerable at all.
    """
    element_ids, fractions, padding_mask = composition_tensors(formulas, max_elements)
    count = len(element_ids)
    provenance = torch.zeros(count, len(feature_names))
    if system is not None:
        # The neighbourhood densities are the one channel a never-computed
        # composition can still answer, so they are filled where available.
        for position, name in enumerate(feature_names):
            if name in system.columns:
                provenance[:, position] = torch.tensor(
                    system[name].to_numpy(dtype=np.float32), dtype=torch.float32)
    return FormulaData(
        element_ids, fractions, padding_mask,
        provenance,
        torch.full((count,), float("nan")),
        torch.tensor(np.zeros(count) if hull is None else np.asarray(hull), dtype=torch.float32),
        pd.Index(formulas),
    )


def _forward(model: FormulaEnergyModel, data: FormulaData, rows: Tensor) -> Tensor:
    return model(data.element_ids[rows], data.fractions[rows], data.padding_mask[rows], data.provenance[rows])


@torch.no_grad()
def evaluate(model: FormulaEnergyModel, data: FormulaData, criterion: nn.Module,
             batch_size: int = 4096) -> Dict[str, float]:
    """Held-out NLL, plus how far the fitted floor sits below the observed bound.

    ``excess`` is the diagnostic that separates this from an MSE fit: it should be
    positive, because the floor is meant to lie *under* the lowest thing anyone
    found, and a floor sitting on top of the observations means the censoring is
    not being fitted.
    """
    model.eval()
    total, excess, violations, absolute = 0.0, 0.0, 0.0, 0.0
    for start in range(0, len(data), batch_size):
        rows = torch.arange(start, min(start + batch_size, len(data)), device=data.target.device)
        prediction = _forward(model, data, rows)
        location, _ = criterion.split(prediction)
        total += criterion(prediction, data.target[rows]).item() * len(rows)
        excess += (data.target[rows] - location).sum().item()
        absolute += (data.target[rows] - location).abs().sum().item()
        violations += (data.target[rows] < location).sum().item()
    return {
        "nll": total / len(data),
        "excess": excess / len(data),
        "mae_vs_bound": absolute / len(data),
        "violation": violations / len(data),
    }


def train_one(
    train: FormulaData,
    val: FormulaData,
    config: TrainConfig,
    device: torch.device,
    seed: Optional[int] = None,
) -> Tuple[FormulaEnergyModel, List[Dict[str, float]]]:
    """Fit one member of the ensemble, early-stopping on validation NLL."""
    torch.manual_seed(config.seed if seed is None else seed)
    model = FormulaEnergyModel(
        n_provenance=len(PROVENANCE_FEATURES), d_model=config.d_model, n_layers=config.n_layers,
        n_heads=config.n_heads, dim_feedforward=config.dim_feedforward, dropout=config.dropout,
        head_widths=config.head_widths, detach_scale_trunk=config.detach_scale_trunk,
        # Start the floor at the mean of the bounds it is meant to sit under, so
        # the first epochs are spent learning chemistry rather than the offset.
        location_bias=float(train.target.mean()),
    ).to(device)
    criterion = make_criterion(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

    history: List[Dict[str, float]] = []
    best_nll, best_state, since_best = math.inf, None, 0
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(train), device=device)
        running = 0.0
        for start in range(0, len(train), config.batch_size):
            rows = order[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(_forward(model, train, rows), train.target[rows])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()
            running += loss.item() * len(rows)

        measured = evaluate(model, val, criterion)
        measured["epoch"] = epoch
        measured["train_nll"] = running / len(train)
        history.append(measured)
        logger.info("epoch %d train %.4f val %.4f excess %.4f violation %.4f",
                    epoch, measured["train_nll"], measured["nll"], measured["excess"], measured["violation"])

        if measured["nll"] < best_nll - 1e-5:
            best_nll, since_best = measured["nll"], 0
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= config.patience:
                logger.info("stopping at epoch %d; best val NLL %.4f", epoch, best_nll)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_ensemble(
    train: FormulaData,
    val: FormulaData,
    config: TrainConfig,
    device: torch.device,
    n_models: int = 10,
) -> Tuple[List[FormulaEnergyModel], List[List[Dict[str, float]]]]:
    """Wren's deep ensemble: the same fit from ``n_models`` different starts."""
    models, histories = [], []
    for member in range(n_models):
        logger.info("=== ensemble member %d of %d ===", member + 1, n_models)
        model, history = train_one(train, val, config, device, seed=config.seed + member)
        models.append(model)
        histories.append(history)
    return models, histories


@torch.no_grad()
def predict(
    models: Sequence[FormulaEnergyModel],
    data: FormulaData,
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Ensemble prediction: the floor, how much the members disagree, and the excess.

    ``sigma_epistemic`` is the spread of the members' estimates of the floor. That
    is the quantity the screening rule needs -- it is uncertainty about *where the
    floor is*, whereas ``scale`` describes how far observed structures scatter
    above it and says nothing about whether the floor is where we think.
    """
    if not models:
        raise ValueError("No models given")
    locations, scales = [], []
    for model in models:
        model.eval()
        member_location, member_scale = [], []
        for start in range(0, len(data), batch_size):
            rows = torch.arange(start, min(start + batch_size, len(data)), device=data.target.device)
            prediction = _forward(model, data, rows)
            member_location.append(prediction[:, 0].cpu())
            member_scale.append(prediction[:, 1].exp().cpu())
        locations.append(torch.cat(member_location))
        scales.append(torch.cat(member_scale))

    stacked = torch.stack(locations)
    return pd.DataFrame({
        "location": stacked.mean(dim=0).numpy(),
        "sigma_epistemic": stacked.std(dim=0, unbiased=len(models) > 1).numpy(),
        "scale": torch.stack(scales).mean(dim=0).numpy(),
        "target": data.target.cpu().numpy(),
        "hull": data.hull.cpu().numpy(),
    }, index=data.formulas)


def save_ensemble(models: Sequence[FormulaEnergyModel], path: Path, config: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": config.__dict__,
        "provenance_features": list(PROVENANCE_FEATURES),
        "state_dicts": [model.state_dict() for model in models],
    }, path)


def main() -> None:
    """Fit an ensemble on a formula table and save it, for screening.

    ``experiment.py`` trains models to compare them and throws them away; this
    trains one to keep. Screening a *generated* structure wants the model fitted
    on the whole archive, not the shallow world, because the hull it will be
    compared against is the whole archive's.
    """
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--table", type=Path, default=Path("data/formula_energy/formula_table.parquet"))
    parser.add_argument("--out", type=Path, default=Path("runs/formula_energy/ensemble.pt"))
    parser.add_argument("--models", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--noise", type=float, default=TrainConfig.noise)
    parser.add_argument("--loss", choices=("censored", "mse"), default="censored")
    parser.add_argument("--drop-provenance", nargs="*", default=None,
                        help="zero these provenance features; the ablation switch")
    parser.add_argument("--device", type=torch.device, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from wyckoff_transformer.csp import parse_formula  # noqa: PLC0415

    table = pd.read_parquet(args.table)
    device = args.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_elements = max(len(parse_formula(formula)) for formula in table.index)
    config = TrainConfig(loss=args.loss, noise=args.noise, epochs=args.epochs,
                         drop_provenance=tuple(args.drop_provenance or ()))
    train = prepare(table[table["split"] == "train"], max_elements=max_elements,
                    drop_provenance=config.drop_provenance).to(device)
    val = prepare(table[table["split"] == "val"], max_elements=max_elements,
                  drop_provenance=config.drop_provenance).to(device)
    logger.info("train %d, val %d formulas, pad width %d, device %s",
                len(train), len(val), max_elements, device)

    models, _ = train_ensemble(train, val, config, device, n_models=args.models)
    save_ensemble(models, args.out, config)
    print(f"wrote {args.out} ({args.models} models, pad width {max_elements})")


def load_ensemble(
    path: Path, device: torch.device
) -> Tuple[List[FormulaEnergyModel], TrainConfig, List[str]]:
    """Returns the members, the config, and the provenance features they expect.

    The feature list travels with the checkpoint so a model trained before the
    set grew keeps loading, and so a caller cannot silently hand it a vector of
    the wrong width in the wrong order.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    config = TrainConfig(**payload["config"])
    models = []
    for state in payload["state_dicts"]:
        model = FormulaEnergyModel(
            n_provenance=len(payload["provenance_features"]), d_model=config.d_model,
            n_layers=config.n_layers, n_heads=config.n_heads, dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, head_widths=config.head_widths,
            detach_scale_trunk=config.detach_scale_trunk,
        ).to(device)
        model.load_state_dict(state)
        models.append(model.eval())
    return models, config, list(payload["provenance_features"])


if __name__ == "__main__":
    main()
