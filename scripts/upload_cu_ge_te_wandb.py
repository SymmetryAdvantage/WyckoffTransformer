"""Upload Cu-Ge-Te convex hull mapping campaign outputs to W&B.

Logs summary metrics, plots, and a versioned artifact containing the
full dataset, candidate selections, relaxed structures, and ground-state CIFs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import wandb

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT
from wyckoff_transformer.paths import wandb_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("upload_cu_ge_te_wandb")

ARTIFACTS_DIR = Path("artifacts/cu_ge_te_hull")


def main() -> None:
    logger.info("Initializing W&B run...")
    run = wandb.init(
        dir=wandb_dir(),
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        id="pglsoqms",
        resume="allow",
        name="cu_ge_te_hull_mapping_20260929",
        job_type="hull_mapping",
        tags=["cu-ge-te", "convex_hull", "orb_conserv_inf", "nep89", "discovery"],
        notes="De novo convex hull mapping of the Cu-Ge-Te system using 100k generated genes, NEP 89 pre-relaxation, and ORB relaxation.",
    )

    # 1. Read summary metrics from the hull entries CSV
    entries_csv = ARTIFACTS_DIR / "cu_ge_te_hull_entries.csv"
    if entries_csv.is_file():
        df = pd.read_csv(entries_csv)
        n_total = len(df)
        n_relaxed = len(df[df["source"] == "generated"])
        n_reference = len(df[df["source"] == "reference"])
        n_stable = len(df[df["is_stable"]])
        n_metastable_50 = len(df[df["is_metastable_50meV"]])
        n_metastable_100 = len(df[df["is_metastable_100meV"]])

        run.summary.update({
            "system": "Cu-Ge-Te",
            "backbone_model": "chemsys_sg_uncond_adanmw_wsd-20260925-030937",
            "regressor_model": "min_energy_adamw_wsd-20260924-102431",
            "relaxer": "orb_conserv_inf",
            "prerelaxer": "nep89",
            "n_generated_genes": 100000,
            "n_unique_genes": 29659,
            "n_reconstructed_genes": 10000,
            "n_initial_pyxtal_structures": 489371,
            "n_prescreen_selected": 48660,
            "n_orb_relaxed": n_relaxed,
            "n_total_hull_entries": n_total,
            "n_orb_reference_entries": n_reference,
            "n_ground_state_phases": n_stable,
            "n_metastable_phases_50meV": n_metastable_50,
            "n_metastable_phases_100meV": n_metastable_100,
            "ground_state_formulas": [
                "Cu", "Ge", "Te", "Cu3GeTe4", "GeTe", "CuTe", "Cu3Te2", "Cu5Ge"
            ],
            "paper_material_e_above_hull_eV": 0.0897,
            "paper_material_formula": "Cu5Ge2Te7",
            "paper_material_sg": 5,
        })
        logger.info("Updated run summary with hull metrics.")

    # 2. Log publication figure
    plot_path = ARTIFACTS_DIR / "cu_ge_te_hull.png"
    if plot_path.is_file():
        run.log({"cu_ge_te_hull_plot": wandb.Image(str(plot_path), caption="Cu-Ge-Te Ternary Convex Hull & Pseudobinary Cut")})
        logger.info("Logged hull plot image to W&B.")

    # 3. Create and populate artifact
    artifact = wandb.Artifact(
        name="cu_ge_te_hull_campaign",
        type="hull_mapping_results",
        description="Complete dataset, candidate structures, relaxations, and ground-state CIFs for Cu-Ge-Te hull exploration.",
        metadata={
            "system": "Cu-Ge-Te",
            "elements": ["Cu", "Ge", "Te"],
            "date": "2026-09-29",
            "n_entries": 48671,
            "n_relaxed": 48629,
        },
    )

    # Core summary and plots
    artifact.add_file(str(plot_path), name="cu_ge_te_hull.png")
    artifact.add_file(str(entries_csv), name="cu_ge_te_hull_entries.csv")

    # Gene files
    for fname in [
        "selected_10000_genes.json.gz",
        "scored_unique_genes.csv.gz",
        "cu_ge_te_unique_genes.json.gz",
        "cu_ge_te_100k_genes.json.gz",
        "cu_ge_te_system_plan.json",
        "pipeline.log",
    ]:
        p = ARTIFACTS_DIR / fname
        if p.is_file():
            artifact.add_file(str(p), name=fname)

    # Reconstruction summaries
    recon_dir = ARTIFACTS_DIR / "reconstruction"
    for fname in [
        "prescreen_selection.csv",
        "relaxations.csv",
        "structures.csv",
        "structures_fixed_symmetry.csv",
        "manifest.json",
        "screen.json",
        "cifs.tar.gz",
    ]:
        p = recon_dir / fname
        if p.is_file():
            artifact.add_file(str(p), name=f"reconstruction/{fname}")

    # Key phases CIFs
    key_phases_dir = ARTIFACTS_DIR / "key_phases"
    if key_phases_dir.is_dir():
        artifact.add_dir(str(key_phases_dir), name="key_phases")

    logger.info("Uploading artifact to W&B...")
    run.log_artifact(artifact)

    run_url = run.url
    run_id = run.id
    logger.info("Run finished successfully: id=%s url=%s", run_id, run_url)
    run.finish()


if __name__ == "__main__":
    main()
