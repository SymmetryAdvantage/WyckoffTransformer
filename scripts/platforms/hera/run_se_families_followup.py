"""Run the Liverpool follow-up through reporting, archiving every finished stage.

Launch from the checkout root with its .venv/bin/python. Re-running resumes the
campaign's existing per-system and per-trial outputs. A file lock prevents two
drivers from operating on the same run.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import wandb
import yaml

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT
from wyckoff_transformer.paths import data_path, runs_root, wandb_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "yamls/campaigns/solid_electrolyte_families_hera_20261103.yaml"))
    stages = ["preflight", "generate", "bandgap", "select", "relax", "report"]
    parser.add_argument("--start-stage", choices=stages, default="preflight",
                        help="Resume at this stage using the preceding saved outputs.")
    args = parser.parse_args()
    checkout = Path.cwd()
    sys.path.insert(0, str(checkout / "scripts"))
    import solid_electrolyte_campaign as campaign

    config = yaml.safe_load(args.config.read_text())
    root = runs_root() / config["campaign_id"]
    root.mkdir(parents=True, exist_ok=True)
    with (root / "driver.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        patch = subprocess.check_output(["git", "diff", "HEAD"], text=True)
        (root / "launch.patch").write_text(patch)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", WANDB_MODE="online",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        os.environ["WANDB_MODE"] = "online"
        with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
                        id=config["campaign_id"], name=config["campaign_id"],
                        resume="allow", job_type="campaign", dir=wandb_dir(),
                        config={**config, "code_commit": commit, "code_dirty": bool(patch)}) as run:
            run.summary["stage"] = "preflight"
            known = campaign.load_additional_known(config)
            campaign.load_obelix(config)
            systems = campaign.expand_systems(config)
            original = campaign.expand_systems(campaign.load_config(campaign.DEFAULT_CONFIG))
            if set(systems.system) & set(original.system):
                raise ValueError("Follow-up repeats an October system")
            assert len(systems) == 19
            audit = []
            for spec in config["screens"]["known_compositions"]:
                with data_path(spec["path"]).open(encoding="utf-8-sig", newline="") as handle:
                    from pymatgen.core import Composition
                    for row in csv.DictReader(handle):
                        row["reduced_formula"] = Composition(
                            row["composition"].strip().strip('"'), strict=True).reduced_formula
                        row["mixture_notation"] = '"' in row["composition"]
                        audit.append(row)
            pd.DataFrame(audit).to_csv(root / "source_formula_audit.csv", index=False)
            run.summary["source_records"] = len(audit)
            run.summary["source_formulas"] = len(known)
            artifact = wandb.Artifact(f"{config['campaign_id']}-inputs", type="campaign_inputs")
            for path in (args.config, Path(__file__), checkout / "scripts/solid_electrolyte_campaign.py",
                         checkout / "scripts/kappagen_gene_campaign.py",
                         checkout / "docs/solid_electrolyte_campaign.md",
                         root / "launch.patch", root / "source_formula_audit.csv"):
                artifact.add_file(str(path), name=path.name)
            for spec in [config["screens"]["obelix"], *config["screens"]["known_compositions"]]:
                artifact.add_file(str(data_path(spec["path"])))
            run.log_artifact(artifact).wait()
            # Vocabulary probes run in a separate process so model memory is released.
            for stage in stages[stages.index(args.start_stage):]:
                run.summary["stage"] = stage
                command = [sys.executable, "scripts/solid_electrolyte_campaign.py", stage,
                           "--config", str(args.config), "--device", "cuda:0"]
                if stage == "relax":
                    command += ["--cpu-lanes", "2", "--cpu-cores", "12", "--no-remote"]
                if stage == "report":
                    command += ["--report-workers", "12"]
                print("Launching:", " ".join(command), flush=True)
                with (root / f"{stage}.log").open("a") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            env=environment, check=False)
                run.save(str(root / f"{stage}.log"), base_path=str(root), policy="now")
                outputs = wandb.Artifact(f"{config['campaign_id']}-{stage}", type="campaign_stage")
                if stage == "preflight" and (root / "preflight.json").exists():
                    outputs.add_file(str(root / "preflight.json"))
                for directory in {"generate": ["raw", "systems"], "bandgap": ["systems"],
                                  "select": ["selected", "protocol"],
                                  "relax": ["protocol"], "report": ["report"]}.get(stage, []):
                    target = root / directory
                    if not target.exists():
                        continue
                    if directory == "protocol":
                        for run_dir in target.iterdir():
                            for name in campaign.PROTOCOL_UPLOADS:
                                path = run_dir / name
                                if path.is_dir():
                                    outputs.add_dir(str(path), name=f"protocol/{run_dir.name}/{name}")
                                elif path.is_file():
                                    outputs.add_file(str(path), name=f"protocol/{run_dir.name}/{name}")
                    else:
                        outputs.add_dir(str(target), name=directory)
                outputs.add_file(str(root / f"{stage}.log"))
                run.log_artifact(outputs).wait()
                if result.returncode:
                    run.summary["failed_stage"] = stage
                    raise RuntimeError(f"{stage} failed: see {root / (stage + '.log')}")
                run.summary[f"{stage}_complete"] = True
            run.summary["stage"] = "complete"
            if run.summary.get("failed_stage") is not None:
                del run.summary["failed_stage"]
            for name in ("shortlist", "tiers_by_family"):
                table = pd.read_csv(root / "report" / f"{name}.csv")
                run.log({name: wandb.Table(dataframe=table)})


if __name__ == "__main__":
    main()
