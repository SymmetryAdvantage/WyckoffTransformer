#!/usr/bin/env python3
"""Resume a full E_hull × classifier-free-guidance de novo protocol grid.

Each arm is uploaded as its own protocol artifact.  A final sweep artifact holds
the three comparison tables and a compact, machine-readable grid.  The local
output directory is only a resumable working copy of those W&B artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from wyckoff_transformer import WANDB_PROJECT
from wyckoff_transformer.paths import wandb_dir

TARGETS = ("0", "0.05", "0.1")
SCALES = (0, 1, 2, 3, 4, 5)
LOGGER = logging.getLogger("cfg_ehull_grid")


def arm_name(target: str, scale: int) -> str:
    return f"cfg-grid-e{target.replace('.', 'p')}-w{scale}"


def oversample(scale: int) -> float:
    # The w=5, E_hull=0 target may have low formal validity.
    return max(1.5, min(10.0, 1.5 * (1 + 0.8 * max(0, scale - 1))))


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Running %s", " ".join(command))
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND " + " ".join(command) + "\n")
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    if result.returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-35:])
        raise RuntimeError(f"{command[0]} exited {result.returncode}; {log_path}:\n{tail}")


def run_arm(run_id: str, root: Path, target: str, scale: int, cpu_threads: int,
            devices: str, gen_device: str, env: dict[str, str]) -> Path:
    name = arm_name(target, scale)
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "grid.log"
    common = [
        str(Path(sys.executable).with_name("wyformer-protocol-wandb")), run_id,
        "--output-dir", str(directory),
        "--condition", f"energy_above_hull={target}",
        "--guidance-scale", str(scale),
        "--arm", name,
        "--n-genes", "1000",
        "--gen-device", gen_device,
        "--pyxtal-cores", str(max(1, cpu_threads - 2)),
        "--devices", devices,
        "--workers-per-device", "1",
    ]
    if not (directory / "screen.json").is_file():
        already_generated = (directory / "wyckoff_genes.json.gz").is_file()
        factor = oversample(scale)
        while True:
            args = [*common, "--stages", "screen", "--no-upload"]
            if already_generated:
                args.append("--skip-generate")
            else:
                args += ["--oversample", str(factor)]
            try:
                run_command(args, log, env)
                break
            except RuntimeError as error:
                if already_generated or "Raise --oversample" not in str(error) or factor >= 24:
                    raise
                factor = min(24.0, factor * 2)
                LOGGER.warning("%s needed more valid genes; retrying with oversample %.1f", name, factor)

    uploaded = directory / "grid-uploaded.json"
    if not uploaded.is_file():
        # The protocol resumes at trial granularity.  Rerun both stages even if
        # files exist: a partial relaxation also has relaxations.csv.
        run_command([*common, "--skip-generate", "--stages", "generate,relax", "--no-upload"],
                    log, env)
        run_command([*common, "--skip-generate", "--stages", "score"], log, env)
        uploaded.write_text(json.dumps({"arm": name, "run": run_id}) + "\n")
    LOGGER.info("Completed %s", name)
    return directory


def write_tables(run_id: str, root: Path, env: dict[str, str]) -> Path:
    tables = root / "tables"
    tables.mkdir(exist_ok=True)
    rows = []
    for target in TARGETS:
        flat_arms = [
            part for scale in SCALES
            for part in ("--arm", f"w{scale}={root / arm_name(target, scale)}")
        ]
        target_dir = tables / f"e{target.replace('.', 'p')}"
        run_command([
            sys.executable, "scripts/analyse_guidance_sweep.py", "table",
            *flat_arms, "--reference", "w1", "--output-dir", str(target_dir),
        ], target_dir / "table.log", env)
        for scale in SCALES:
            directory = root / arm_name(target, scale)
            funnel = json.loads((directory / "funnel.json").read_text())
            manifest = json.loads((directory / "manifest.json").read_text())
            gene = funnel["gene"]
            free = funnel["free"]
            fixed = funnel["fixed_symmetry"]
            sampled = gene["sampled"]
            rows.append({
                "target_e_hull_ev_per_atom": target,
                "guidance_scale": scale,
                "arm": arm_name(target, scale),
                "sampled": sampled,
                "formal_gene_validity": manifest["formal_gene_validity"],
                "gene_novel_per_sampled": gene["sampled_novel"] / sampled,
                "free_valid_per_sampled": free["valid_structure"] / sampled,
                "free_metastable_per_sampled": free["metastable"] / sampled,
                "free_stable_per_sampled": free["stable"] / sampled,
                "free_metasun_per_sampled": free["metastable_among_novel"] / sampled,
                "free_sun_per_sampled": free["stable_among_novel"] / sampled,
                "fixed_metasun_per_sampled": fixed["metastable_among_novel"] / sampled,
            })
    with (tables / "grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (tables / "grid.json").write_text(json.dumps({"run": run_id, "arms": rows}, indent=2) + "\n")
    return tables


def upload_tables(run_id: str, tables: Path) -> None:
    import wandb

    run = wandb.init(dir=wandb_dir(), project=WANDB_PROJECT, entity="symmetry-advantage",
                     id=run_id, resume="must")
    try:
        artifact = wandb.Artifact(f"cfg_ehull_grid_{run_id}", type="protocol_sweep",
                                  metadata={"targets": list(TARGETS), "scales": list(SCALES),
                                            "n_genes_per_arm": 1000})
        artifact.add_dir(str(tables), name="tables")
        run.log_artifact(artifact)
    finally:
        run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=10)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--gen-device", default="cuda:0")
    args = parser.parse_args()
    if not 1 <= args.cpu_threads <= 10:
        parser.error("--cpu-threads must be between 1 and 10")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[:args.cpu_threads])
    LOGGER.info("CPU affinity %s; one relaxation worker per %s", allowed[:args.cpu_threads], args.devices)
    env = os.environ.copy()
    env.update({key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env["WANDB_ENTITY"] = "symmetry-advantage"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        for scale in SCALES:
            run_arm(args.run_id, args.output_dir, target, scale, args.cpu_threads,
                    args.devices, args.gen_device, env)
    tables = write_tables(args.run_id, args.output_dir, env)
    upload_tables(args.run_id, tables)
    LOGGER.info("Grid complete: %s", tables)


if __name__ == "__main__":
    main()
