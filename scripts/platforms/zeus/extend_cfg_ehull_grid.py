#!/usr/bin/env python3
"""Continue a completed CFG E_hull grid to higher guidance scales.

Each new arm runs the full 1000-gene protocol and is uploaded independently.
After each arm, a versioned W&B sweep artifact records the combined table and
the first sustained downturn in free MetaSUN and SUN for each target.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import run_cfg_ehull_grid as grid
from wyckoff_transformer import WANDB_PROJECT
from wyckoff_transformer.paths import wandb_dir

LOGGER = logging.getLogger("cfg_ehull_high_w")
METRICS = ("free_metasun_per_sampled", "free_sun_per_sampled")


def wait_for_base_grid(service: str, root: Path) -> None:
    """Do not contend with the first grid, including any automatic restarts."""
    last_state = None
    while True:
        result = subprocess.run(
            ["systemctl", "--user", "show", service, "-p", "ActiveState", "-p", "Result"],
            capture_output=True, text=True, check=True,
        )
        state = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        status = (state.get("ActiveState"), state.get("Result"))
        if status != last_state:
            LOGGER.info("Base grid service %s: %s", service, status)
            last_state = status
        if status == ("inactive", "success") and (root / "tables" / "grid.json").is_file():
            LOGGER.info("Base grid completed and its tables are present")
            return
        time.sleep(30)


def arm_row(root: Path, target: str, scale: int) -> dict | None:
    directory = root / grid.arm_name(target, scale)
    if not (directory / "grid-uploaded.json").is_file():
        return None
    funnel = json.loads((directory / "funnel.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    sampled = funnel["gene"]["sampled"]
    free = funnel["free"]
    return {
        "target_e_hull_ev_per_atom": target,
        "guidance_scale": scale,
        "arm": grid.arm_name(target, scale),
        "sampled": sampled,
        "formal_gene_validity": manifest["formal_gene_validity"],
        "gene_novel_per_sampled": funnel["gene"]["sampled_novel"] / sampled,
        "free_valid_per_sampled": free["valid_structure"] / sampled,
        "free_metastable_per_sampled": free["metastable"] / sampled,
        "free_stable_per_sampled": free["stable"] / sampled,
        "free_metasun_per_sampled": free["metastable_among_novel"] / sampled,
        "free_sun_per_sampled": free["stable_among_novel"] / sampled,
    }


def rows_for_target(root: Path, target: str) -> list[dict]:
    rows = []
    for scale in itertools.count():
        row = arm_row(root, target, scale)
        if row is None:
            break
        rows.append(row)
    return rows


def first_downturn(rows: list[dict], metric: str) -> int | None:
    """First two post-grid arms below the best value before both arms."""
    for index in range(7, len(rows)):
        earlier_best = max(row[metric] for row in rows[: index - 1])
        if rows[index - 1][metric] < earlier_best and rows[index][metric] < earlier_best:
            return rows[index]["guidance_scale"]
    return None


def downturns(rows: list[dict]) -> dict[str, int | None]:
    return {metric: first_downturn(rows, metric) for metric in METRICS}


def write_snapshot(run_id: str, root: Path, limits: dict) -> Path:
    tables = root / "high_w_tables"
    tables.mkdir(exist_ok=True)
    per_target = {target: rows_for_target(root, target) for target in grid.TARGETS}
    rows = [row for target in grid.TARGETS for row in per_target[target]]
    with (tables / "grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "run": run_id,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip(),
        "n_genes_per_arm": 1000,
        "rule": "For each target and each metric, the first two arms at w>=6 "
                "below the best preceding value mark an observed downturn. "
                "Stop that target after downturns in both free MetaSUN and free SUN.",
        "downturns": {target: downturns(values) for target, values in per_target.items()},
        "generation_limits": limits,
        "arms": rows,
    }
    (tables / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return tables


def upload_snapshot(run_id: str, tables: Path) -> None:
    import wandb

    run = wandb.init(dir=wandb_dir(), project=WANDB_PROJECT,
                     entity="symmetry-advantage", id=run_id, resume="must")
    try:
        artifact = wandb.Artifact(f"cfg_ehull_high_w_{run_id}", type="protocol_sweep",
                                  metadata={"targets": list(grid.TARGETS),
                                            "first_added_scale": 6, "n_genes_per_arm": 1000})
        artifact.add_dir(str(tables), name="tables")
        run.log_artifact(artifact)
    finally:
        run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-service", default="wyformer-cfg-grid.service")
    parser.add_argument("--cpu-threads", type=int, default=10)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--workers-per-device", type=int, default=2)
    parser.add_argument("--gen-device", default="cuda:0")
    args = parser.parse_args()
    if not 1 <= args.cpu_threads <= 10:
        parser.error("--cpu-threads must be between 1 and 10")
    if not 1 <= args.workers_per_device <= 2:
        parser.error("--workers-per-device must be between 1 and 2")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    wait_for_base_grid(args.base_service, args.output_dir)

    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[:args.cpu_threads])
    LOGGER.info("CPU affinity %s; %d relaxation workers per %s",
                allowed[:args.cpu_threads], args.workers_per_device, args.devices)
    env = os.environ.copy()
    env.update({key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env["WANDB_ENTITY"] = "symmetry-advantage"

    limit_file = args.output_dir / "high_w_generation_limits.json"
    limits = json.loads(limit_file.read_text()) if limit_file.is_file() else {}
    # Also recovers an arm uploaded immediately before a supervisor restart.
    upload_snapshot(args.run_id, write_snapshot(args.run_id, args.output_dir, limits))
    for scale in itertools.count(6):
        for target in grid.TARGETS:
            previous = rows_for_target(args.output_dir, target)
            if target in limits or all(value is not None for value in downturns(previous).values()):
                continue
            if len(previous) > scale:
                continue
            if len(previous) != scale:
                raise RuntimeError(f"Expected {scale} completed arms for target {target}; "
                                   f"found {len(previous)}")
            if previous[-1]["formal_gene_validity"] <= 0:
                raise RuntimeError(f"Previous formal validity is zero for target {target}")
            factor = min(100.0, max(grid.oversample(scale),
                                    1.35 / previous[-1]["formal_gene_validity"]))
            LOGGER.info("Starting E_hull=%s, w=%d with oversample %.2f", target, scale, factor)
            try:
                grid.run_arm(args.run_id, args.output_dir, target, scale,
                             args.cpu_threads, args.devices, args.workers_per_device,
                             args.gen_device, env, initial_oversample=factor,
                             max_oversample=100.0)
            except RuntimeError as error:
                if "Raise --oversample" not in str(error):
                    raise
                limits[target] = {"scale": scale, "max_oversample": 100,
                                  "reason": "Could not obtain 1000 formally valid genes "
                                            "within 100000 attempted draws"}
                limit_file.write_text(json.dumps(limits, indent=2) + "\n")
                LOGGER.warning("Generation limit reached for E_hull=%s, w=%d", target, scale)
            tables = write_snapshot(args.run_id, args.output_dir, limits)
            upload_snapshot(args.run_id, tables)
            LOGGER.info("Uploaded high-w table through E_hull=%s, w=%d", target, scale)
        if all(target in limits or all(value is not None for value in downturns(
                rows_for_target(args.output_dir, target)).values()) for target in grid.TARGETS):
            LOGGER.info("All targets have observed downturns or reached the generation limit")
            return


if __name__ == "__main__":
    main()
