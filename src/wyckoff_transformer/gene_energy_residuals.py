"""What the gene-energy regressor gets wrong on genes the archive already holds.

For a gene LeMat-Bulk contains, the regressor's error is not an unknown: the DFT
gene-minimum formation energy is right there, so ``prediction - DFT`` is observed.
Two uses follow, and this module provides both.

**A gene the archive holds needs no prediction.** Its DFT energy is known, so a
selection should use that and not the regressor's guess. :class:`KnownGeneEnergies`
is the lookup, keyed by the same 128-bit gene key the novelty screen uses.

**Residuals are local.** A regressor that is 50 meV/atom too high across a
chemical system misplaces every candidate in it against the DFT hull by the same
amount, and a hull is exactly the comparison where a per-system offset matters.
:class:`ResidualCorrection` estimates that offset by hierarchical shrinkage:

    b(E)     = (sum_{el(i) = E} r_i + kappa * b_sub(E)) / (n_E + kappa)
    b_sub(E) = (sum_{el(i) < E} r_i + kappa * b_0)    / (n_sub + kappa)

with ``b_0`` the global mean residual. A system with many residuals of its own
gets its own mean; one with none backs off to what its subsystems say, then to
the global offset.

**Only honest residuals are used.** The target is a minimum over the whole
fingerprint class across every split, so a validation gene whose fingerprint
also occurs in ``train`` had its target shown to the model. The residual table
keeps only val/test genes absent from ``train``, and :func:`validate` fits the
correction on val and scores it on test before a run is allowed to rely on it.

    python -m wyckoff_transformer.gene_energy_residuals \\
        --regressor-path <run-dir> --out-dir <dir> --device cuda:0
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

TARGET = "gene_min_formation_energy_per_atom"
DEFAULT_DATASET = "lemat_bulk_fmax1_stress"
RESIDUALS_FILE = "residuals.parquet"
KNOWN_GENES_FILE = "known_gene_energies.npz"
VALIDATION_FILE = "residual_validation.json"
DEFAULT_KAPPAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
#: How much held-out MAE the correction must remove before a run relies on it.
MIN_RELATIVE_GAIN = 0.01

#: What the regressor reads, plus what the residual is measured against.
_PREDICTION_COLUMNS = (
    "spacegroup_number", "elements", "site_symmetries", "sites_enumeration",
    "multiplicity", "composition", "site_symmetries_augmented",
    "sites_enumeration_augmented", TARGET, "energy_above_hull",
)
_KEY_COLUMNS = (
    "spacegroup_number", "elements", "site_symmetries_augmented",
    "sites_enumeration_augmented", TARGET,
)


def _symbol(element) -> str:
    symbol = getattr(element, "symbol", None)
    if symbol is not None:
        return str(symbol)
    text = str(element)
    return text.split()[-1] if text.startswith("Element") else text


def chemical_system(elements: Iterable) -> str:
    """``"Li-Mn-O"``-style key, sorted alphabetically, for any spelling of the elements."""
    return "-".join(sorted({_symbol(element) for element in elements}))


# --------------------------------------------------------------------------- #
# Known genes
# --------------------------------------------------------------------------- #
class KnownGeneEnergies:
    """Gene key -> DFT gene-minimum formation energy, for every gene in the archive."""

    def __init__(self, low: np.ndarray, high: np.ndarray, energy: np.ndarray) -> None:
        order = np.argsort(low, kind="stable")
        self.low = np.asarray(low, dtype=np.int64)[order]
        self.high = np.asarray(high, dtype=np.int64)[order]
        self.energy = np.asarray(energy, dtype=np.float64)[order]

    def __len__(self) -> int:
        return len(self.low)

    @classmethod
    def load(cls, path: Path) -> "KnownGeneEnergies":
        with np.load(path) as stored:
            return cls(stored["low"], stored["high"], stored["energy"])

    def save(self, path: Path) -> Path:
        np.savez(path, low=self.low, high=self.high, energy=self.energy)
        return path

    def lookup(self, keys: Sequence) -> np.ndarray:
        """DFT energy per key, NaN for a gene the archive does not hold."""
        if len(keys) == 0:
            return np.zeros(0)
        keys = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
        out = np.full(len(keys), np.nan)
        left = np.searchsorted(self.low, keys[:, 0], side="left")
        right = np.searchsorted(self.low, keys[:, 0], side="right")
        for row, (start, stop) in enumerate(zip(left, right)):
            for position in range(start, stop):
                if self.high[position] == keys[row, 1]:
                    out[row] = self.energy[position]
                    break
        return out


# --------------------------------------------------------------------------- #
# The correction
# --------------------------------------------------------------------------- #
class ResidualCorrection:
    """Per-chemical-system offset of the regressor, shrunk toward its subsystems.

    Args:
        chemsys: One ``"A-B-C"`` per residual.
        residuals: ``prediction - DFT`` per residual, eV/atom.
        kappa: Pseudo-count of the shrinkage. Larger trusts a system's own
            residuals less.
    """

    def __init__(self, chemsys: Sequence[str], residuals: Sequence[float], kappa: float) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        frame = pd.DataFrame({"chemsys": list(chemsys), "r": np.asarray(residuals, float)})
        frame = frame[np.isfinite(frame["r"])]
        grouped = frame.groupby("chemsys")["r"].agg(["sum", "count"])
        self.sums = grouped["sum"].to_dict()
        self.counts = grouped["count"].to_dict()
        self.kappa = float(kappa)
        self.global_mean = float(frame["r"].mean()) if len(frame) else 0.0
        self.n = int(len(frame))

    def bias(self, chemsys: str) -> float:
        elements = chemsys.split("-")
        own_sum = self.sums.get(chemsys, 0.0)
        own_n = self.counts.get(chemsys, 0)
        sub_sum, sub_n = 0.0, 0
        for size in range(1, len(elements)):
            for subset in combinations(sorted(elements), size):
                key = "-".join(subset)
                sub_sum += self.sums.get(key, 0.0)
                sub_n += self.counts.get(key, 0)
        b_sub = (sub_sum + self.kappa * self.global_mean) / (sub_n + self.kappa)
        return (own_sum + self.kappa * b_sub) / (own_n + self.kappa)

    def biases(self, chemsys: Iterable[str]) -> np.ndarray:
        cache: dict = {}
        out = []
        for key in chemsys:
            if key not in cache:
                cache[key] = self.bias(key)
            out.append(cache[key])
        return np.asarray(out, dtype=np.float64)

    def describe(self) -> dict:
        return {"kappa": self.kappa, "global_mean": self.global_mean,
                "residuals": self.n, "systems": len(self.counts)}


def _auc(scores: np.ndarray, positive: np.ndarray) -> Optional[float]:
    """Probability a positive ranks below a negative (lower score = better)."""
    positive = np.asarray(positive, bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    ranks = pd.Series(-np.asarray(scores, float)).rank().to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _metrics(frame: pd.DataFrame, correction: np.ndarray, threshold: float) -> dict:
    error = frame["residual"].to_numpy() - correction
    metrics = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mean_error": float(np.mean(error)),
    }
    if "dft_e_hull" in frame:
        dft = frame["dft_e_hull"].to_numpy()
        usable = np.isfinite(dft)
        predicted = dft + error
        metrics[f"auc_e_hull_le_{threshold:g}"] = _auc(
            predicted[usable], dft[usable] <= threshold)
    return metrics


def validate(
    residuals: pd.DataFrame,
    kappas: Sequence[float] = DEFAULT_KAPPAS,
    threshold: float = 0.05,
) -> dict:
    """Fit on the val residuals, score on the test ones, for each kappa.

    ``auc_e_hull_le_<t>`` is the ranking a selection actually uses: the chance a
    test gene with DFT ``e_hull <= t`` gets a lower predicted ``e_hull`` than one
    without, where the predicted ``e_hull`` is the DFT one plus the (corrected)
    residual -- the hull at a composition the archive holds cancels out.
    """
    fit = residuals[residuals["split"] == "val"]
    test = residuals[residuals["split"] == "test"]
    report = {
        "fit_residuals": int(len(fit)), "test_residuals": int(len(test)),
        "threshold": threshold,
        "raw": _metrics(test, np.zeros(len(test)), threshold),
        "global_offset": _metrics(
            test, np.full(len(test), float(fit["residual"].mean())), threshold),
        "kappa": {},
    }
    for kappa in kappas:
        correction = ResidualCorrection(fit["chemsys"], fit["residual"], kappa)
        report["kappa"][str(kappa)] = _metrics(
            test, correction.biases(test["chemsys"]), threshold)
    best = min(report["kappa"], key=lambda k: report["kappa"][k]["mae"])
    report["best_kappa"] = float(best)
    # A pre-registered bar: at least 1% off the held-out MAE. The kappa is chosen
    # on the same test set, so a smaller gain is indistinguishable from that choice.
    report["min_relative_gain"] = MIN_RELATIVE_GAIN
    report["correction_helps"] = bool(
        report["kappa"][best]["mae"] < (1 - MIN_RELATIVE_GAIN) * report["raw"]["mae"])
    return report


def load_correction(directory: Path, kappa: Optional[float] = None) -> Optional[ResidualCorrection]:
    """The correction fitted on every honest residual, at the validated kappa.

    Returns ``None`` when validation found it does not help on held-out genes and
    no kappa was forced: a correction that makes test predictions worse is not
    one a run should select with.
    """
    directory = Path(directory)
    with open(directory / VALIDATION_FILE, "rt", encoding="utf-8") as handle:
        report = json.load(handle)
    if kappa is None:
        if not report["correction_helps"]:
            logger.warning(
                "The residual correction did not beat the raw prediction on held-out "
                "genes (%s); running without it.", directory / VALIDATION_FILE)
            return None
        kappa = report["best_kappa"]
    residuals = pd.read_parquet(directory / RESIDUALS_FILE, columns=["chemsys", "residual"])
    correction = ResidualCorrection(residuals["chemsys"], residuals["residual"], kappa)
    logger.info("Residual correction: %s", correction.describe())
    return correction


# --------------------------------------------------------------------------- #
# Building the table
# --------------------------------------------------------------------------- #
def _keys(frame: pd.DataFrame) -> np.ndarray:
    from wyckoff_transformer.evaluation.gene_hash import keys_from_frame  # noqa: PLC0415

    return keys_from_frame(frame).numpy()


def predict_frame(frame: pd.DataFrame, regressor, chunk: int = 20000,
                  augmentation_samples: int = 1) -> np.ndarray:
    """Regressor predictions for Wyckoff records, NaN where outside its vocabulary.

    The same path ``cli.gene_screen.score_genes`` takes, including the clean
    relaxation condition, so a residual measures the error a candidate would get.
    """
    from wyckoff_transformer.gene_energy import build_clean_relaxation_condition  # noqa: PLC0415
    from wyckoff_transformer.prediction import (  # noqa: PLC0415
        build_tokenised_prediction_tensors,
        filter_supported_tokens,
    )

    out = pd.Series(np.nan, index=frame.index)
    for start in range(0, len(frame), chunk):
        part = frame.iloc[start:start + chunk]
        try:
            supported, _ = filter_supported_tokens(part, regressor)
        except ValueError:
            continue
        if supported.empty:
            continue
        tensors = build_tokenised_prediction_tensors(supported, regressor)
        cond = build_clean_relaxation_condition(regressor, len(supported), device=regressor.device)
        with torch.no_grad():
            mean, _ = regressor.predict_scalars(
                tensors, augmentation_samples=augmentation_samples, cond=cond)
        out.loc[supported.index] = mean.float().cpu().numpy()
        logger.info("Predicted %d / %d", min(start + chunk, len(frame)), len(frame))
    return out.to_numpy()


def build(
    regressor,
    out_dir: Path,
    dataset: str = DEFAULT_DATASET,
    max_rows: Optional[int] = None,
    augmentation_samples: int = 1,
    kappas: Sequence[float] = DEFAULT_KAPPAS,
) -> dict:
    """Write the known-gene energies, the honest residuals and their validation."""
    from wyckoff_transformer.dataset_cache import dataset_cache_dir, load_split  # noqa: PLC0415

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = dataset_cache_dir(dataset)

    known_low, known_high, known_energy = [], [], []
    train = load_split(cache, "train", columns=list(_KEY_COLUMNS))
    if max_rows is not None and len(train) > max_rows:
        train = train.sample(n=max_rows, random_state=0)
    train_keys = _keys(train)
    known_low.append(train_keys[:, 0])
    known_high.append(train_keys[:, 1])
    known_energy.append(train[TARGET].to_numpy(dtype=np.float64))
    logger.info("Keyed %d train rows", len(train))
    del train
    gc.collect()
    train_set = set(map(tuple, train_keys.tolist()))

    held_out = []
    for split in ("val", "test"):
        frame = load_split(cache, split, columns=list(_PREDICTION_COLUMNS))
        if max_rows is not None and len(frame) > max_rows:
            frame = frame.sample(n=max_rows, random_state=0)
        keys = _keys(frame)
        known_low.append(keys[:, 0])
        known_high.append(keys[:, 1])
        known_energy.append(frame[TARGET].to_numpy(dtype=np.float64))
        frame = frame.assign(split=split, key_low=keys[:, 0], key_high=keys[:, 1])
        frame["seen_in_train"] = [tuple(k) in train_set for k in keys.tolist()]
        held_out.append(frame)
    held_out = pd.concat(held_out)

    # Known genes: one energy per key (the target is already the class minimum).
    low = np.concatenate(known_low)
    high = np.concatenate(known_high)
    energy = np.concatenate(known_energy)
    unique = pd.DataFrame({"low": low, "high": high, "energy": energy}).dropna()
    unique = unique.drop_duplicates(["low", "high"])
    KnownGeneEnergies(unique["low"], unique["high"], unique["energy"]).save(
        out_dir / KNOWN_GENES_FILE)
    logger.info("%d known genes written", len(unique))

    # Honest residuals: one row per gene never seen in train, its rows' lowest
    # DFT e_hull standing for the gene's.
    unseen = held_out[~held_out["seen_in_train"]].dropna(subset=[TARGET])
    dft_e_hull = unseen.groupby(["key_low", "key_high"])["energy_above_hull"].min()
    unseen = unseen.drop_duplicates(["key_low", "key_high"]).copy()
    unseen["dft_e_hull"] = dft_e_hull.reindex(
        pd.MultiIndex.from_frame(unseen[["key_low", "key_high"]])).to_numpy()
    logger.info("%d of %d held-out rows are genes train never saw",
                len(unseen), len(held_out))
    unseen["prediction"] = predict_frame(unseen, regressor,
                                         augmentation_samples=augmentation_samples)
    unseen["chemsys"] = [chemical_system(elements) for elements in unseen["elements"]]
    table = pd.DataFrame({
        "key_low": unseen["key_low"].to_numpy(),
        "key_high": unseen["key_high"].to_numpy(),
        "split": unseen["split"].to_numpy(),
        "chemsys": unseen["chemsys"].to_numpy(),
        "spacegroup_number": unseen["spacegroup_number"].to_numpy(),
        "dft": unseen[TARGET].to_numpy(dtype=np.float64),
        "dft_e_hull": unseen["dft_e_hull"].to_numpy(dtype=np.float64),
        "prediction": unseen["prediction"].to_numpy(dtype=np.float64),
    })
    table["residual"] = table["prediction"] - table["dft"]
    table = table.dropna(subset=["residual"]).reset_index(drop=True)
    table.to_parquet(out_dir / RESIDUALS_FILE)

    report = validate(table, kappas)
    report.update({
        "dataset": dataset, "max_rows": max_rows,
        "augmentation_samples": augmentation_samples,
        "held_out_rows": int(len(held_out)), "unseen_genes": int(len(unseen)),
        "known_genes": int(len(unique)),
    })
    with open(out_dir / VALIDATION_FILE, "wt", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    from wyckoff_transformer.cli.csp import load_trainer  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--regressor-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Subsample each split; for smoke tests only.")
    parser.add_argument("--augmentation-samples", type=int, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    regressor = load_trainer(device=args.device, model_path=args.regressor_path)
    report = build(regressor, args.out_dir, args.dataset, args.max_rows,
                   args.augmentation_samples)
    printable = {k: v for k, v in report.items() if k != "kappa"}
    print(json.dumps(printable, indent=1))
    for kappa, metrics in report["kappa"].items():
        print(f"kappa={kappa:>6}: {metrics}")


if __name__ == "__main__":
    main()
