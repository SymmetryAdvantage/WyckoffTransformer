#!/usr/bin/env python3
"""Publish MLIP compatibility probes and their dated study notes to W&B."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import wandb

from wyckoff_transformer.paths import wandb_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--file", action="append", type=Path, required=True)
    args = parser.parse_args()
    files = [path.resolve() for path in args.file]
    for path in files:
        if not path.is_file():
            parser.error(f"Missing compatibility record: {path}")
    if len({path.name for path in files}) != len(files):
        parser.error("Compatibility record basenames must be unique")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run = wandb.init(entity="symmetry-advantage", project="WyckoffTransformer",
                     id="mlipbias1", resume="allow", dir=str(wandb_dir()),
                     job_type="mlip_bias_compatibility")
    artifact = wandb.Artifact(
        args.artifact_name, type="mlip_bias_compatibility",
        metadata={"recorded_utc": datetime.now(timezone.utc).isoformat(),
                  "wyformer_commit": commit, "files": [path.name for path in files]},
    )
    for path in files:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact).wait()
    run.finish()
    print(f"Uploaded {args.artifact_name} with {len(files)} files")


if __name__ == "__main__":
    main()
