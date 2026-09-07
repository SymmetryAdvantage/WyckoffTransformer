import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import wandb
from omegaconf import OmegaConf

import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT, wandb_run_path
from wyckoff_transformer.data import (
    read_cif,
    compute_symmetry_sites,
    pyxtal_notation_to_sites,
    get_composition_from_symmetry_sites,
)
from wyckoff_transformer.preprocess_wychoffs import get_augmentation_dict
from wyckoff_transformer.prediction import (
    build_tokenised_prediction_tensors,
    filter_supported_tokens,
)
from wyckoff_transformer.tokenization import load_wyckoff_mappings
from wyckoff_transformer.trainer import WyckoffTrainer


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict scalar properties using a pretrained Wyckoff Transformer.")
    parser.add_argument("input_path", type=Path, help="Input file containing structures.")
    parser.add_argument("output", type=Path, help="Where to write predictions (.csv or .csv.gz).")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--wandb-run", type=str, help="W&B run id to load the model from.")
    parser.add_argument("--wandb-entity", type=str, default=WANDB_ENTITY,
                        help="W&B entity holding --wandb-run")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT, help="W&B project")
    source_group.add_argument("--model-path", type=Path,
                              help="Directory with best_model_params.pt and tokeniser artifacts.")
    parser.add_argument("--input-type", choices=["cif", "pyxtal"], required=True,
                        help="Whether the input file contains raw CIFs or pyxtal dictionaries.")
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"),
                        help="Device used for inference.")
    parser.add_argument("--augmentation-samples", type=int, default=1,
                        help="Number of augmentation draws per structure.")
    parser.add_argument("--retain-samples", action="store_true",
                        help="Store individual augmentation samples in the output.")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Number of parallel jobs for symmetry detection.")
    parser.add_argument("--symmetry-precision", type=float, default=0.1,
                        help="Tolerance used when deriving symmetry from CIF inputs.")
    parser.add_argument("--symmetry-a-tol", type=float, default=5.0,
                        help="Angular tolerance used when deriving symmetry from CIF inputs.")
    parser.add_argument("--use-cached-tensors", action="store_true",
                        help="Load cached training tensors instead of recomputing them.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity.")
    return parser.parse_args()


def ensure_run_artifacts(run: wandb.apis.public.Run, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    required_files = ["best_model_params.pt", "tokenizers.pkl.gz", "token_engineers.pkl.gz"]
    for file_name in required_files:
        target_path = run_dir / file_name
        if target_path.exists():
            continue
        wandb_file = run.file(file_name)
        if wandb_file is None:
            raise FileNotFoundError(f"{file_name} not found among W&B artifacts for run {run.id}")
        logger.info("Downloading %s to %s", file_name, target_path)
        wandb_file.download(root=str(run_dir), replace=True)


def load_config_from_run(run: wandb.apis.public.Run) -> OmegaConf:
    wandb_config = OmegaConf.create(dict(run.config))
    base_config_name = wandb_config.get("base_config", None)
    if base_config_name:
        base_config = OmegaConf.load(
            Path(__file__).parent.parent / "yamls" / "models" / f"{base_config_name}.yaml")
        return OmegaConf.merge(base_config, wandb_config)
    return wandb_config


def load_wandb_model(
    run_id: str,
    device: torch.device,
    use_cached_tensors: bool,
    entity: str = WANDB_ENTITY,
    project: str = WANDB_PROJECT) -> Tuple[WyckoffTrainer, OmegaConf]:

    wandb_run = wandb.Api().run(wandb_run_path(run_id, entity, project))
    run_dir = Path(__file__).parent.parent / "runs" / run_id
    ensure_run_artifacts(wandb_run, run_dir)
    config = load_config_from_run(wandb_run)
    trainer = WyckoffTrainer.from_config(
        config,
        device=device,
        use_cached_tensors=use_cached_tensors,
        run_path=run_dir)
    state_dict = torch.load(trainer.run_path / "best_model_params.pt",
                            map_location=device,
                            weights_only=True)
    trainer.model.load_state_dict(state_dict)
    return trainer, config


def load_local_model(
    model_path: Path,
    device: torch.device,
    use_cached_tensors: bool) -> Tuple[WyckoffTrainer, OmegaConf]:

    config_path = model_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    config = OmegaConf.load(config_path)
    trainer = WyckoffTrainer.from_config(
        config,
        device=device,
        use_cached_tensors=use_cached_tensors,
        run_path=model_path)
    state_dict = torch.load(model_path / "best_model_params.pt",
                            map_location=device,
                            weights_only=True)
    trainer.model.load_state_dict(state_dict)
    return trainer, config


def read_cif_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    if "cif" not in df.columns:
        raise ValueError("Input CSV must contain a 'cif' column.")
    with pd.option_context("mode.chained_assignment", None):
        logger.info("Parsing %d CIF entries.", len(df))
        df["structure"] = df["cif"].map(read_cif)
    return df


def read_pyxtal_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        opener = gzip.open
        mode = "rt"
    else:
        opener = open
        mode = "rt"
    with opener(path, mode, encoding="ascii") as fh:
        records = json.load(fh)
    if not isinstance(records, list):
        raise ValueError("PyXtal input must be a list of dictionaries.")
    df = pd.DataFrame.from_records(records)
    if "group" not in df.columns or "sites" not in df.columns or "species" not in df.columns:
        raise ValueError("PyXtal input must contain 'group', 'sites', and 'species' keys.")

    _m = load_wyckoff_mappings()
    wychoffs_enumerated_by_ss = _m.enum_from_ss_letter
    ss_from_letter = _m.ss_from_letter
    augmentation_dict = get_augmentation_dict()
    def converter(row):
        return pyxtal_notation_to_sites(
            row,
            wychoffs_enumerated_by_ss=wychoffs_enumerated_by_ss,
            ss_from_letter=ss_from_letter,
            wychoffs_augmentation=augmentation_dict)
    logger.info("Converting %d pyxtal structures to symmetry sites.", len(df))
    converted = df.apply(converter, axis=1, result_type="expand")
    converted.index = df.index
    converted["composition"] = converted.apply(get_composition_from_symmetry_sites, axis=1)
    return converted


def derive_symmetry_from_structures(
    df: pd.DataFrame,
    n_jobs: int | None,
    symmetry_precision: float,
    symmetry_a_tol: float) -> pd.DataFrame:

    symmetry_data = compute_symmetry_sites(
        {"prediction": df},
        n_jobs=n_jobs,
        symmetry_precision=symmetry_precision,
        symmetry_a_tol=symmetry_a_tol)
    return symmetry_data["prediction"]


def sanitise_symmetry_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "site_symmetries",
        "elements",
        "sites_enumeration",
        "multiplicity",
        "spacegroup_number",
        "composition",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required symmetry columns: {missing}")

    mask = df["site_symmetries"].notna() & df["elements"].notna() & df["sites_enumeration"].notna()
    filtered = df.loc[mask].copy()
    if filtered.empty:
        raise ValueError("No valid symmetry records were produced.")

    def normalise_augmented(entry):
        if isinstance(entry, (set, frozenset)):
            return [list(variant) for variant in entry]
        if isinstance(entry, list):
            return [list(variant) for variant in entry]
        return []

    if "sites_enumeration_augmented" in filtered.columns:
        filtered["sites_enumeration_augmented"] = filtered["sites_enumeration_augmented"].apply(normalise_augmented)
    else:
        filtered["sites_enumeration_augmented"] = [[] for _ in range(len(filtered))]

    if "composition" in filtered.columns:
        filtered["composition"] = filtered["composition"].apply(lambda comp: comp if comp is not None else {})
    else:
        filtered["composition"] = [{} for _ in range(len(filtered))]
    return filtered


def run_prediction(args: argparse.Namespace) -> None:
    if args.log_level:
        logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if args.device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    if args.wandb_run:
        trainer, _ = load_wandb_model(args.wandb_run, args.device, args.use_cached_tensors,
                                      args.wandb_entity, args.wandb_project)
    else:
        trainer, _ = load_local_model(args.model_path, args.device, args.use_cached_tensors)

    dropped_indices: List = []
    original_index_order: List = []
    if args.input_type == "cif":
        cif_df = read_cif_dataframe(args.input_path)
        symmetry_df = derive_symmetry_from_structures(
            cif_df,
            n_jobs=args.n_jobs,
            symmetry_precision=args.symmetry_precision,
            symmetry_a_tol=args.symmetry_a_tol)
        processed_df = sanitise_symmetry_dataframe(symmetry_df)
        missing = cif_df.index.difference(processed_df.index)
        if not missing.empty:
            raise ValueError(f"Failed to process symmetry for {len(missing)} structures from the input.")
        processed_df = processed_df.loc[cif_df.index]
        original_index_order = processed_df.index.tolist()
        processed_df, dropped_indices = filter_supported_tokens(processed_df, trainer)
        cif_df = cif_df.loc[processed_df.index]
    else:
        symmetry_df = read_pyxtal_dataframe(args.input_path)
        processed_df = sanitise_symmetry_dataframe(symmetry_df)
        original_index_order = processed_df.index.tolist()
        processed_df, dropped_indices = filter_supported_tokens(processed_df, trainer)

    logger.info("Prepared %d structures for prediction.", len(processed_df))
    tokenised_data = build_tokenised_prediction_tensors(processed_df, trainer)
    mean_predictions, all_predictions = trainer.predict_scalars(
        tokenised_data,
        augmentation_samples=args.augmentation_samples)

    mean_predictions = mean_predictions.detach().cpu()
    all_predictions = all_predictions.detach().cpu()
    results = pd.DataFrame(index=processed_df.index)
    results["prediction_mean"] = mean_predictions.numpy()
    if args.augmentation_samples > 1:
        results["prediction_std"] = all_predictions.std(dim=0, unbiased=False).numpy()
        if args.retain_samples:
            for sample_idx in range(args.augmentation_samples):
                results[f"prediction_sample_{sample_idx}"] = all_predictions[sample_idx].numpy()

    if original_index_order:
        results = results.reindex(original_index_order)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output)
    logger.info("Saved predictions for %d structures to %s", len(results), args.output)


def main():
    args = parse_args()
    run_prediction(args)


if __name__ == "__main__":
    main()
