"""Upload protocol evaluation artifact for ehull_adamw_wsd_5x-20260912-115321 (target E_hull=0.05).

Ensures W&B run summary is NOT modified or overwritten.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import wandb
from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT, wandb_run_path
from wyckoff_transformer.paths import wandb_dir
from wyckoff_transformer.cli import protocol as protocol_cli
from wyckoff_transformer.cli.protocol_wandb import ARTIFACT_TYPE, GENES_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RUN_ID = "ehull_adamw_wsd_5x-20260912-115321"
OUTPUT_DIR = Path("generated/ehull_adamw_wsd_5x-20260912-115321/protocol")


def main() -> None:
    api = wandb.Api()
    run_path_str = wandb_run_path(RUN_ID, WANDB_ENTITY, WANDB_PROJECT)
    api_run = api.run(run_path_str)
    
    # Capture summary beforehand
    initial_summary = dict(api_run.summary)
    logger.info("Captured initial summary (%d keys)", len(initial_summary))
    
    manifest_path = OUTPUT_DIR / protocol_cli.MANIFEST_FILE
    metadata = {}
    if manifest_path.is_file():
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    # Add clear target labels
    metadata["target"] = "E_hull=0.05"
    metadata["condition"] = "energy_above_hull=0.05"
    metadata["condition_feature"] = "energy_above_hull"
    metadata["condition_target"] = 0.05
    metadata["description"] = "De novo ranking protocol evaluation for ehull_adamw_wsd_5x-20260912-115321 sampled at E_hull=0.05 target"

    # Connect to the run
    run = wandb.init(
        dir=wandb_dir(),
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        id=RUN_ID,
        resume="must",
    )

    def populate_artifact(art: wandb.Artifact) -> None:
        for name in (
            GENES_FILE,
            protocol_cli.SCREEN_FILE,
            protocol_cli.PYXTAL_FILE,
            protocol_cli.PYXTAL_TRIALS_FILE,
            protocol_cli.RELAXATIONS_FILE,
            protocol_cli.PRESCREEN_TRIALS_FILE,
            protocol_cli.PRESCREEN_SELECTION_FILE,
            protocol_cli.STRUCTURES_FILE,
            protocol_cli.STRUCTURES_FIXED_FILE,
            protocol_cli.FUNNEL_FILE,
            protocol_cli.MANIFEST_FILE,
        ):
            p = OUTPUT_DIR / name
            if p.is_file():
                art.add_file(str(p), name=name)
        cif_dir = OUTPUT_DIR / protocol_cli.CIF_DIR
        if cif_dir.is_dir():
            art.add_dir(str(cif_dir), name=protocol_cli.CIF_DIR)
        cif_fixed_dir = OUTPUT_DIR / protocol_cli.CIF_FIXED_DIR
        if cif_fixed_dir.is_dir():
            art.add_dir(str(cif_fixed_dir), name=protocol_cli.CIF_FIXED_DIR)

    # 1. Dedicated artifact collection clearly labelled in its name
    art_dedicated = wandb.Artifact(
        name=f"protocol_{RUN_ID}_ehull005",
        type=ARTIFACT_TYPE,
        description="De novo ranking protocol evaluation for ehull_adamw_wsd_5x-20260912-115321 sampled at E_hull=0.05 target",
        metadata=metadata,
    )
    populate_artifact(art_dedicated)
    logger.info("Logging artifact %s...", art_dedicated.name)
    run.log_artifact(art_dedicated, aliases=["latest", "ehull005", "ehull_0.05", "target_0.05", "E_hull=0.05"])

    # 2. In canonical collection as a version with explicit target aliases
    art_canonical = wandb.Artifact(
        name=f"protocol_{RUN_ID}",
        type=ARTIFACT_TYPE,
        description="De novo ranking protocol evaluation for ehull_adamw_wsd_5x-20260912-115321 sampled at E_hull=0.05 target",
        metadata=metadata,
    )
    populate_artifact(art_canonical)
    logger.info("Logging artifact %s...", art_canonical.name)
    run.log_artifact(art_canonical, aliases=["ehull005", "ehull_0.05", "target_0.05", "E_hull=0.05"])

    logger.info("Finishing wandb run...")
    run.finish()

    # Re-verify summary via wandb API
    refreshed_run = api.run(run_path_str)
    current_summary = dict(refreshed_run.summary)
    
    # Check if any protocol summary keys were touched
    summary_diff = {k: v for k, v in current_summary.items() if initial_summary.get(k) != v}
    if summary_diff:
        logger.warning("Summary keys changed: %s. Restoring initial summary...", list(summary_diff.keys()))
        for k in list(refreshed_run.summary.keys()):
            if k not in initial_summary:
                del refreshed_run.summary[k]
        for k, v in initial_summary.items():
            refreshed_run.summary[k] = v
        refreshed_run.summary.update()
        logger.info("Restored initial summary successfully.")
    else:
        logger.info("Verified: run summary was NOT modified.")

    # In canonical collection protocol_<run_id>, keep 'latest' on v2 (baseline e_hull=0)
    try:
        art_v2 = api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/protocol_{RUN_ID}:v2")
        if "latest" not in art_v2.aliases:
            logger.info("Re-assigning 'latest' alias to protocol_%s:v2 (baseline E_hull=0)", RUN_ID)
            art_v2.aliases.append("latest")
            art_v2.save()
        art_v3 = api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/protocol_{RUN_ID}:v3")
        if "latest" in art_v3.aliases:
            logger.info("Removing 'latest' alias from protocol_%s:v3 (E_hull=0.05)", RUN_ID)
            art_v3.aliases.remove("latest")
            art_v3.save()
    except Exception as exc:
        logger.warning("Could not adjust aliases via API: %s", exc)

    logger.info("Upload complete!")


if __name__ == "__main__":
    main()
