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
]


def download_wandb_artifacts(run_path: str, target_dir: Path) -> None:
    """Download model artifacts from a W&B run into target_dir.

    Searches all logged artifacts for the required files by filename,
    taking the latest version of any artifact that contains each file.
    For the model weights, the latest artifact version is used (it is
    the checkpoint with the best validation loss).
    """
    api = wandb.Api()
    run = api.run(run_path)

    # Build a map: filename -> latest artifact that contains it
    file_to_artifact: dict[str, object] = {}
    for artifact in run.logged_artifacts():
        artifact_files = {f.name for f in artifact.files()}
        for required in REQUIRED_FILES:
            if required in artifact_files:
                # Overwrite to keep the latest version (artifacts are
                # yielded in log order, so the last assignment wins)
                file_to_artifact[required] = artifact

    missing = set(REQUIRED_FILES) - set(file_to_artifact)
    if missing:
        raise FileNotFoundError(
            f"Could not find all required files in W&B run '{run_path}'. "
            f"Missing: {missing}"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, artifact in file_to_artifact.items():
        artifact_dir = Path(artifact.download(
            root=str(target_dir / f"_artifact_{artifact.name.replace(':', '_')}")
        ))
        src = artifact_dir / filename
        if not src.exists():
            # Some artifacts store files in a flat layout without subdirs
            matches = list(artifact_dir.glob(f"**/{filename}"))
            if not matches:
                raise FileNotFoundError(
                    f"File '{filename}' not found inside downloaded artifact '{artifact.name}'."
                )
            src = matches[0]
        (target_dir / filename).write_bytes(src.read_bytes())
        print(f"  Downloaded {filename} from {artifact.name}")


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
