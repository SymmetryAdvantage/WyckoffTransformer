#!/usr/bin/env python3
"""Stop the adaptive CFG sweep after its current arm and publish its conclusion."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import extend_cfg_ehull_grid as sweep

LOGGER = logging.getLogger("finish_cfg_ehull_grid")


def write_conclusion(run_id: str, root: Path, final_target: str, final_scale: int,
                     tables: Path) -> None:
    rows_by_target = {target: sweep.rows_for_target(root, target)
                      for target in sweep.grid.TARGETS}
    lines = [
        "# CFG E_hull guidance grid: final observed results",
        "",
        f"Date: {datetime.now(timezone.utc).date().isoformat()} (UTC).",
        f"W&B run: https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/{run_id}",
        "Evaluation commit: " + subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip(),
        f"Stopped at the user's request after target {final_target}, w={final_scale}",
        "completed and uploaded. Each arm sampled 1,000 genes and used the full",
        "de novo ranking protocol. The percentages below use free relaxation and",
        "are per sampled gene.",
        "",
        "| Target E_hull (eV/atom) | Best observed MetaSUN | Best observed SUN | Last arm | Last MetaSUN | Last SUN |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for target, rows in rows_by_target.items():
        meta = max(rows, key=lambda row: row["free_metasun_per_sampled"])
        sun = max(rows, key=lambda row: row["free_sun_per_sampled"])
        last = rows[-1]
        lines.append(
            f"| {target} | {100 * meta['free_metasun_per_sampled']:.1f}% "
            f"(w={meta['guidance_scale']}) | {100 * sun['free_sun_per_sampled']:.1f}% "
            f"(w={sun['guidance_scale']}) | w={last['guidance_scale']} | "
            f"{100 * last['free_metasun_per_sampled']:.1f}% | "
            f"{100 * last['free_sun_per_sampled']:.1f}% |"
        )
    lines += [
        "",
        "These are observed maxima over the sampled arms, not estimates of",
        "statistically distinct optima. SUN is based on few stable novel structures",
        "per 1,000 genes and is especially noisy. Higher guidance also lowers the",
        "formal validity of raw draws, so the cost of obtaining 1,000 valid genes",
        "increases. The complete per-arm results are in grid.csv; each arm's",
        "generated genes, PyXtal structures, relaxed CIFs, trial and structure",
        "tables, and manifest are in its W&B protocol_eval artifact.",
        "",
    ]
    (tables / "conclusion.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--service", default="wyformer-cfg-high-w.service")
    parser.add_argument("--target", default="0.05")
    parser.add_argument("--scale", type=int, default=11)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    marker = args.output_dir / sweep.grid.arm_name(args.target, args.scale) / "grid-uploaded.json"
    LOGGER.info("Waiting for completed and uploaded arm %s", marker.parent.name)
    while not marker.is_file():
        time.sleep(0.5)
    LOGGER.info("Arm upload marker present; stopping adaptive sweep before another arm")
    subprocess.run(["systemctl", "--user", "stop", args.service], check=True)

    limit_file = args.output_dir / "high_w_generation_limits.json"
    limits = json.loads(limit_file.read_text()) if limit_file.is_file() else {}
    tables = sweep.write_snapshot(args.run_id, args.output_dir, limits)
    report_path = tables / "report.json"
    report = json.loads(report_path.read_text())
    report["stopped_at_user_request"] = {
        "target_e_hull_ev_per_atom": args.target,
        "guidance_scale": args.scale,
        "reason": "No higher guidance scales requested after the current arm",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_conclusion(args.run_id, args.output_dir, args.target, args.scale, tables)
    sweep.upload_snapshot(args.run_id, tables)
    LOGGER.info("Final conclusion uploaded: %s", tables / "conclusion.md")


if __name__ == "__main__":
    main()
