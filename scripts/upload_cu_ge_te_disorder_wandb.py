"""Archive the Cu-Ge-Te disorder-parent inference and sensitivity run in W&B."""

from __future__ import annotations

import json
from pathlib import Path

import wandb

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT
from wyckoff_transformer.paths import wandb_dir

ROOT = Path("artifacts/cu_ge_te_hull")
BASELINE = ROOT / "disorder_parents"
SENSITIVITY = ROOT / "disorder_parents_symprec010"


def main() -> None:
    baseline = json.loads((BASELINE / "analysis.json").read_text())
    sensitivity = json.loads((SENSITIVITY / "analysis.json").read_text())
    run = wandb.init(
        dir=wandb_dir(), entity=WANDB_ENTITY, project=WANDB_PROJECT,
        job_type="disorder_parent_inference",
        name="cu_ge_te_disorder_parents_20260929",
        tags=["cu-ge-te", "disorder", "orb_conserv_inf", "hull_mapping"],
        config={
            "source_campaign_run": "pglsoqms",
            "source_campaign_commit": "54c955b",
            "analysis_base_commit": "a50d319",
            "max_ehull_eV": baseline["max_ehull"],
            "symprec_baseline_angstrom": baseline["symprec_angstrom"],
            "symprec_sensitivity_angstrom": sensitivity["symprec_angstrom"],
        },
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    run.summary.update({
        "selected_structures": baseline["selected_structures"],
        "candidate_memberships": baseline["candidate_memberships"],
        "candidate_parent_geometries": baseline["candidate_families"],
        "cross_formula_parents": baseline["families_with_multiple_formulas"],
        "stronger_cross_formula_parents": baseline[
            "families_with_at_least_three_formulas_and_ten_structures"
        ],
        "sensitivity_candidate_parent_geometries": sensitivity["candidate_families"],
        "sensitivity_cross_formula_parents": sensitivity["families_with_multiple_formulas"],
    })
    artifact = wandb.Artifact(
        "cu_ge_te_disorder_parents_20260929", type="disorder_parent_inference",
        description=(
            "Substitutional parent-framework hypotheses for generated Cu-Ge-Te "
            "ORB relaxations within 100 meV/atom of the campaign hull; "
            "includes a symmetry-tolerance sensitivity run and representative CIFs. "
            "These are not experimentally confirmed disordered materials."
        ),
        metadata={
            "source_campaign_run": "pglsoqms",
            "selected_structures": baseline["selected_structures"],
            "baseline_symprec_angstrom": baseline["symprec_angstrom"],
            "sensitivity_symprec_angstrom": sensitivity["symprec_angstrom"],
        },
    )
    artifact.add_dir(str(BASELINE), name="symprec015")
    artifact.add_dir(str(SENSITIVITY), name="symprec010")
    artifact.add_file("docs/cu_ge_te_disorder_parent_inference_20260929.md")
    artifact.add_file("scripts/infer_cu_ge_te_disorder_parents.py")
    artifact.add_file("src/wyckoff_transformer/tests/test_disorder_parent_inference.py")
    logged = run.log_artifact(artifact)
    logged.wait()
    print(f"W&B run: {run.url}")
    print(f"W&B artifact: {logged.name}")
    run.finish()


if __name__ == "__main__":
    main()
