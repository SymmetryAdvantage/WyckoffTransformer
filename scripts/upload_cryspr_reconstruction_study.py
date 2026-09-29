"""Archive the 2026-09-05 CrySPR reconstruction study in W&B.

The small analysis files stay individually browsable. The 87,687 per-trial
files and target CIFs are bundled to keep the artifact manageable while
preserving the complete original output tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

import wandb

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT
from wyckoff_transformer.paths import wandb_dir


ARTIFACT_NAME = "cryspr_reconstruction_fidelity_20260905"
CODE_COMMIT = "dc241d5d84886d280014db832dfdd7ff18eccf6b"
REPORT_COMMIT = "c82d82f08e1a22b7f8bf1528f798777f6af16bbb"
EXPECTED_VERDICTS = {
    "recovered": 761,
    "sampled_not_selected": 28,
    "lower_energy_alternative": 16,
    "missed": 186,
    "generation_failed": 8,
}
REQUIRED_FILES = (
    "STUDY_REPORT.md",
    "data/sample_draw_distributions.json",
    "data/sampled_1000_targets_raw.parquet",
    "data/relaxed_targets.parquet",
    "data/unique_wyckoff_genes.json.gz",
    "data/reconstruction_results.pkl",
    "tables/results_genes.csv",
    "tables/headline_metrics.json",
    "tables/breakdown_dof_positional.csv",
    "tables/breakdown_dof_total.csv",
    "tables/breakdown_nsites.csv",
    "tables/breakdown_n_wyckoff_sites.csv",
    "tables/breakdown_crystal_system.csv",
    "tables/breakdown_spacegroup.csv",
    "tables/breakdown_symmetry_attempt.csv",
)


def validate(study_dir: Path) -> dict:
    missing = [name for name in REQUIRED_FILES if not (study_dir / name).is_file()]
    missing += [name for name in ("cryspr", "targets") if not (study_dir / name).is_dir()]
    if missing:
        raise ValueError(f"Incomplete study output: missing {', '.join(missing)}")

    headline = json.loads((study_dir / "tables/headline_metrics.json").read_text())
    with (study_dir / "tables/results_genes.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    verdicts = dict(Counter(row["verdict"] for row in rows))
    if len(rows) != 999 or headline["total_genes"] != 999:
        raise ValueError("Expected exactly 999 scored genes")
    if verdicts != EXPECTED_VERDICTS or headline["verdict_counts"] != EXPECTED_VERDICTS:
        raise ValueError("Per-gene verdicts differ from the published report")
    if (headline["total_matching_trials_s3"], headline["total_matching_trials_s4"]) != (4912, 5011):
        raise ValueError("Stage 3/4 match counts differ from the published report")

    n_targets = sum(1 for path in (study_dir / "targets").iterdir() if path.is_file())
    n_trials = sum(1 for _ in (study_dir / "cryspr").glob("*/trial-*"))
    if n_targets != 1000 or n_trials != 9990:
        raise ValueError(f"Expected 1000 target CIFs and 9990 trials; found {n_targets}, {n_trials}")
    return headline


def archive_tree(directory: Path, destination: Path) -> None:
    with tarfile.open(destination, mode="w:gz", compresslevel=1) as archive:
        archive.add(directory, arcname=directory.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True,
                        help="Directory produced by run_cryspr_reconstruction_study.py")
    args = parser.parse_args()
    study_dir = args.study_dir.resolve()
    headline = validate(study_dir)

    run = wandb.init(
        dir=wandb_dir(),
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=ARTIFACT_NAME,
        job_type="study_archive",
        tags=["cryspr", "reconstruction", "lemat-bulk", "orb-v3"],
        config={"study_date": "2026-09-05", "code_commit": CODE_COMMIT,
                "report_commit": REPORT_COMMIT, "n_trials_per_gene": 10},
    )
    if run is None:
        raise RuntimeError("W&B did not create a run")

    artifact = wandb.Artifact(
        name=ARTIFACT_NAME,
        type="study_dataset",
        description="Complete original inputs, per-gene results, target structures, and all CrySPR trial files for the 2026-09-05 reconstruction fidelity study.",
        metadata={"study_date": "2026-09-05", "code_commit": CODE_COMMIT,
                  "report_commit": REPORT_COMMIT, "n_genes": 999,
                  "n_targets": 1000, "n_trials": 9990,
                  "verdict_counts": EXPECTED_VERDICTS},
    )
    artifact.add_file(str(study_dir / "STUDY_REPORT.md"))
    artifact.add_dir(str(study_dir / "data"), name="data")
    artifact.add_dir(str(study_dir / "tables"), name="tables")
    run.summary.update(headline)

    # W&B snapshots local files when logging. Keep both archives alive until
    # the upload has finished, including any retry made by the SDK.
    with tempfile.TemporaryDirectory(prefix="cryspr-reconstruction-") as temp:
        for tree in ("targets", "cryspr"):
            archive_path = Path(temp) / f"{tree}.tar.gz"
            archive_tree(study_dir / tree, archive_path)
            artifact.add_file(str(archive_path), name=archive_path.name)
        logged = run.log_artifact(artifact)
        logged.wait()
        print(f"run_url={run.url}")
        print(f"artifact={WANDB_ENTITY}/{WANDB_PROJECT}/{ARTIFACT_NAME}:{logged.version}")
    run.finish()


if __name__ == "__main__":
    main()
