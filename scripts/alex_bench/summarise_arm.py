"""One summary row per evaluated arm of the alex-mp-20 benchmark study.

The decision metric is MSUN *per submitted structure*: every gene an arm hands to
DiffCSP++ is one submitted structure, so the denominator is the number of engaged genes,
duplicates included. The protocol's funnel counts each unique gene once, which is what a
submission gets credit for -- a duplicate gene is a duplicate structure, not a second hit.

    .venv/bin/python scripts/alex_bench/summarise_arm.py ARM_DIR --setting S --arm A \
        [--summary SUMMARY.csv]

ARM_DIR holds ``engaged_genes.json.gz``, ``cohort.csv`` (from ``wyformer-roe``) and
``protocol/`` (the protocol run). Writes ``ARM_DIR/summary.json`` and appends to
``--summary``.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from wyckoff_transformer.evaluation.protocol import load_genes
from wyckoff_transformer.formula_energy.prefilter import wilson_interval

DIFFCSP_REPO = Path("/home/kna/DiffCSPNew")


def _commit(repo: Path) -> str:
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "-uno"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - provenance, never a reason to fail
        return "unknown"


def _rate(prefix: str, hits: int, n: int) -> dict:
    low, high = wilson_interval(hits, n)
    return {prefix: hits / n if n else float("nan"),
            f"{prefix}_low": low, f"{prefix}_high": high}


def summarise(arm_dir: Path, setting: str, arm: str) -> dict:
    protocol = arm_dir / "protocol"
    n = len(load_genes(arm_dir / "engaged_genes.json.gz"))
    funnel = json.loads((protocol / "funnel.json").read_text())
    manifest = json.loads((protocol / "manifest.json").read_text())
    structures = pd.read_csv(protocol / "structures.csv").set_index("index")
    row = {"setting": setting, "arm": arm, "n_submitted": n,
           "unique_genes": funnel["gene"]["unique_gene"],
           "gene_novel": funnel["gene"]["gene_novel"]}

    starts = pd.read_csv(protocol / "starts.csv")
    row["start_failed"] = int((starts["status"] != "ok").sum())
    for readout, tag in (("free", ""), ("fixed_symmetry", "as_submitted_")):
        counts = funnel[readout]
        for key in ("structure", "valid_structure", "unique_structure", "novel_structure",
                    "metastable", "stable"):
            row[f"{tag}{key}"] = counts[key]
        row.update(_rate(f"{tag}msun_per_submitted", counts["metastable_among_novel"], n))
        row.update(_rate(f"{tag}sun_per_submitted", counts["stable_among_novel"], n))

    relaxations = pd.read_csv(protocol / "relaxations.csv")
    row["relax_hours"] = round(relaxations["seconds"].sum() / 3600, 3)
    row["mean_atoms"] = round(float(structures["n_atoms"].mean()), 2)

    cohort_path = arm_dir / "cohort.csv"
    if cohort_path.is_file():
        cohort = pd.read_csv(cohort_path).set_index("index")
        engaged = cohort[cohort["engaged_index"].notna()] if "engaged_index" in cohort else cohort
        row["sampled_for_arm"] = int(len(cohort))
        if "predicted_e_hull" in cohort and engaged["predicted_e_hull"].notna().any():
            row["selection_cut_e_hull"] = float(engaged["predicted_e_hull"].max())

    # Indexed like the protocol's genes: the position in engaged_genes.json.gz.
    predicted_path = arm_dir / "predicted_e_hull.csv"
    if predicted_path.is_file():
        predicted = pd.read_csv(predicted_path).set_index("index")["predicted_e_hull"]
        row["mean_predicted_e_hull"] = float(predicted.mean())
        joined = structures.join(predicted, how="inner")
        joined = joined[joined["valid_structure"].eq(True)]
        joined = joined.dropna(subset=["predicted_e_hull", "e_above_hull"])
        if len(joined) > 2:
            row["predictor_spearman"] = float(
                joined["predicted_e_hull"].corr(joined["e_above_hull"], method="spearman"))
            novel = joined[joined["novel_structure"].eq(True)]
            if len(novel) > 2:
                row["predictor_spearman_novel"] = float(
                    novel["predicted_e_hull"].corr(novel["e_above_hull"], method="spearman"))

    row.update({
        "mlip": manifest.get("mlip"), "relax_schedule": manifest.get("relax_schedule"),
        "novelty_reference": manifest.get("reference_cache"),
        "wyformer_commit": _commit(Path(__file__).resolve().parents[2]),
        "diffcsp_commit": _commit(DIFFCSP_REPO),
    })
    for key in ("generator", "gen_args", "regressor"):
        meta = arm_dir / "arm.json"
        if meta.is_file():
            row[key] = json.loads(meta.read_text()).get(key)
    return {k: (round(v, 5) if isinstance(v, float) and np.isfinite(v) else v)
            for k, v in row.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("arm_dir", type=Path)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()
    row = summarise(args.arm_dir, args.setting, args.arm)
    (args.arm_dir / "summary.json").write_text(json.dumps(row, indent=1) + "\n")
    if args.summary is not None:
        frame = pd.DataFrame([row])
        if args.summary.is_file():
            frame = pd.concat([pd.read_csv(args.summary), frame], ignore_index=True)
            frame = frame.drop_duplicates(["setting", "arm"], keep="last")
        frame.to_csv(args.summary, index=False)
    print(json.dumps({k: row[k] for k in ("setting", "arm", "n_submitted",
                                          "msun_per_submitted", "sun_per_submitted")}))


if __name__ == "__main__":
    main()
