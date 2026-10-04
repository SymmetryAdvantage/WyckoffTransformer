"""Fetch pinned WyFormer checkpoints from W&B into the runs store, for ``--model-path``.

Each run is given as ``RUN_ID[:vN]``; ``vN`` pins the ``best_model_<RUN_ID>`` artifact
version, so a resumed run cannot swap the weights under a study. Without it the latest
version is taken, and the version actually fetched is recorded in ``fetched.json``.
``config.yaml`` is written too: ``load_trainer(model_path=...)`` reads it, and
``protocol_wandb.ensure_run_files`` does not fetch it.

    .venv/bin/python scripts/alex_bench/fetch_wandb_runs.py RUN_ID:v42 RUN_ID ...
"""
import argparse
import json
import logging
from pathlib import Path

import wandb
from omegaconf import OmegaConf

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT
from wyckoff_transformer.paths import runs_root

logger = logging.getLogger(__name__)

#: Artifact name prefixes and the versions to take; best_model's is overridable.
ARTIFACTS = ("processors", "spacegroup_distribution", "best_model")


def fetch(api: wandb.Api, entity: str, project: str, spec: str, runs: Path) -> dict:
    run_id, _, version = spec.partition(":")
    run_dir = runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"run": run_id, "artifacts": {}}
    for prefix in ARTIFACTS:
        alias = (version or "latest") if prefix == "best_model" else "v0"
        artifact = api.artifact(f"{entity}/{project}/{prefix}_{run_id}:{alias}")
        names = {f.name for f in artifact.files()}
        missing = [name for name in names if not (run_dir / name).exists()]
        if missing:
            logger.info("Downloading %s:%s -> %s", artifact.name, artifact.version, run_dir)
            artifact.download(root=str(run_dir))
        record["artifacts"][prefix] = f"{artifact.name.split(':')[0]}:{artifact.version}"
    # A pinned best_model must not be silently replaced by a different one already on disk.
    previous = run_dir / "fetched.json"
    if previous.is_file():
        old = json.loads(previous.read_text())["artifacts"].get("best_model")
        if old != record["artifacts"]["best_model"]:
            raise RuntimeError(
                f"{run_dir} already holds {old}, not {record['artifacts']['best_model']}; "
                "remove best_model_params.pt and fetched.json to switch versions")
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        run = api.run(f"{entity}/{project}/{run_id}")
        OmegaConf.save(OmegaConf.create(dict(run.config)), config_path)
    previous.write_text(json.dumps(record, indent=1) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="RUN_ID or RUN_ID:vN")
    parser.add_argument("--entity", default=WANDB_ENTITY)
    parser.add_argument("--project", default=WANDB_PROJECT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    api = wandb.Api()
    for spec in args.runs:
        print(json.dumps(fetch(api, args.entity, args.project, spec, runs_root())))


if __name__ == "__main__":
    main()
