"""Push a trained WyckoffTransformer model to HuggingFace Hub.

Usage:
    push_to_hub.py <repo_id> --wandb-run <run_path>
    push_to_hub.py <repo_id> --model-path <path>
"""
import argparse
import tempfile
from pathlib import Path

import wandb
from huggingface_hub import HfApi

REQUIRED_FILES = [
    "best_model_params.pt",
    "wyckoff_processor.json",
    "config.yaml",
    "spacegroup_distribution.json",
    "wyckoffs_enumerated_by_ss.json",
]


def download_wandb_artifacts(run_path: str, target_dir: Path) -> None:
    """Download model artifacts from a W&B run into target_dir.

    Fetches each artifact directly by name, avoiding a full scan of all
    logged artifacts. The latest version of the model artifact is used
    (it is the checkpoint with the best validation loss).
    """
    api = wandb.Api()
    run_id = run_path.split("/")[-1]
    entity_project = "/".join(run_path.split("/")[:2])

    # Each tuple: (artifact_name, artifact_type, [files to extract])
    artifact_specs = [
        (f"best_model_{run_id}", "model", ["best_model_params.pt"]),
        (f"processors_{run_id}", "processors", ["wyckoff_processor.json", "wyckoffs_enumerated_by_ss.json"]),
        (f"run_config_{run_id}", "config", ["config.yaml"]),
        (f"spacegroup_distribution_{run_id}", "dataset_stats", ["spacegroup_distribution.json"]),
    ]

    target_dir.mkdir(parents=True, exist_ok=True)
    for artifact_name, artifact_type, filenames in artifact_specs:
        artifact = api.artifact(f"{entity_project}/{artifact_name}:latest", type=artifact_type)
        artifact_dir = Path(artifact.download(
            root=str(target_dir / f"_artifact_{artifact_name}")
        ))
        for filename in filenames:
            src = artifact_dir / filename
            if not src.exists():
                matches = list(artifact_dir.glob(f"**/{filename}"))
                if not matches:
                    raise FileNotFoundError(
                        f"File '{filename}' not found inside downloaded artifact '{artifact_name}'."
                    )
                src = matches[0]
            (target_dir / filename).write_bytes(src.read_bytes())
            print(f"  Downloaded {filename} from {artifact_name}")


def push_model_dir_to_hub(model_dir: Path, repo_id: str) -> None:
    """Upload all required model files from model_dir to a HuggingFace repo."""
    for filename in REQUIRED_FILES:
        path = model_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Required file '{filename}' not found in '{model_dir}'."
            )

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=REQUIRED_FILES,
    )
    print(f"Model pushed to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Push a trained WyckoffTransformer model to HuggingFace Hub."
    )
    parser.add_argument("repo_id", type=str, help="HuggingFace repo ID, e.g. 'username/model-name'.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wandb-run", type=str,
           help="Full W&B run path to download artifacts from, "
                "e.g. 'symmetry-advantage/WyckoffTransformer/hx9ecapn'.")
    source.add_argument("--model-path", type=Path, help="Local model directory with required files.")
    args = parser.parse_args()

    if args.wandb_run:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            print(f"Downloading artifacts from W&B run '{args.wandb_run}'...")
            download_wandb_artifacts(args.wandb_run, model_dir)
            push_model_dir_to_hub(model_dir, args.repo_id)
    else:
        push_model_dir_to_hub(args.model_path, args.repo_id)


if __name__ == "__main__":
    main()
