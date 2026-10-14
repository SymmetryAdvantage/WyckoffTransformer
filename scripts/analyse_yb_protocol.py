#!/usr/bin/env python3
"""Audit Yb exclusion in W&B protocol_eval artifacts bearing the latest alias.

Read-only with respect to source runs. Downloads only scoring inputs; verifies
the published funnel before removing Yb samples from numerator and denominator.
Outputs are working copies; log them as an analysis artifact for preservation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import wandb

from wyckoff_transformer.dataset_cache import as_cache_dir
from wyckoff_transformer.evaluation.protocol import (
    funnel_structure_metrics,
    load_genes,
    read_screen,
)
from wyckoff_transformer.paths import runs_root, wandb_dir

PROJECT = "symmetry-advantage/WyckoffTransformer"
FILES = (
    "manifest.json",
    "screen.json",
    "funnel.json",
    "wyckoff_genes.json.gz",
    "structures.csv",
    "structures_fixed_symmetry.csv",
)


def analyse(artifact, root: Path, run_summaries: dict) -> tuple[list[dict], dict]:
    artifact_name = artifact.name.rsplit(":", 1)[0] + ":" + artifact.version
    directory = root / "inputs" / artifact_name
    directory.mkdir(parents=True, exist_ok=True)
    if not all((directory / name).exists() for name in FILES):
        names = {entry.name for entry in artifact.files()}
        for name in FILES:
            if name in names and not (directory / name).exists():
                artifact.get_entry(name).download(root=str(directory))
    manifest = json.loads((directory / "manifest.json").read_text())
    funnel = json.loads((directory / "funnel.json").read_text())
    screen = read_screen(directory / "screen.json")
    genes = load_genes(directory / "wyckoff_genes.json.gz")
    assert len(genes) == screen.n_sampled
    yb = {i for i, gene in enumerate(genes) if "Yb" in gene["species"]}
    n_yb = len(yb)
    collection = artifact_name.rsplit(":", 1)[0]
    suffix = collection.removeprefix("protocol_")
    source_run = suffix.split(".", 1)[0].removesuffix("_ehull005")
    summary_prefix = (
        "protocol/" if "." not in suffix else f"protocol_{suffix.split('.', 1)[1]}/"
    )
    baseline = "." not in suffix and not suffix.endswith("_ehull005")
    rows = []
    for section, name in (
        ("free", "structures.csv"),
        ("fixed_symmetry", "structures_fixed_symmetry.csv"),
    ):
        if not (directory / name).exists():
            continue
        frame = pd.read_csv(directory / name).set_index("index")
        assert frame.index.is_unique
        metrics = funnel_structure_metrics(screen, frame)
        # Validate counts and weighted rates using the production implementation.
        for key in (
            "metasun_per_sampled_gene",
            "sun_per_sampled_gene",
            "metastable_among_novel",
        ):
            assert abs(metrics[key] - funnel[section][key]) < 1e-12, (
                artifact.name,
                section,
                key,
                metrics[key],
                funnel[section][key],
            )
        mask = pd.Series(True, index=frame.index)
        for col in (
            "has_structure",
            "valid_structure",
            "unique_structure",
            "novel_structure",
        ):
            assert frame[col].dropna().isin([True, False]).all(), (artifact.name, col)
            mask &= frame[col].fillna(False).astype(bool)
        weights = pd.Series(screen.counts).reindex(frame.index).fillna(0).astype(int)
        is_yb = frame.index.isin(yb)
        msun = mask & frame.e_above_hull.notna() & frame.e_above_hull.le(0.1)
        sun = mask & frame.e_above_hull.notna() & frame.e_above_hull.le(0.0)
        hits = int(weights[msun].sum())
        yb_hits = int(weights[msun & is_yb].sum())
        remaining = screen.n_sampled - n_yb
        non_yb_rate = (hits - yb_hits) / remaining if remaining else None
        yb_rate = yb_hits / n_yb if n_yb else None
        published = run_summaries.get(source_run, {}).get(
            f"{summary_prefix}{section}/metasun_per_sampled_gene"
        )
        row = {
            "artifact": artifact_name,
            "source_run": source_run,
            "baseline": baseline,
            "section": section,
            "n_sampled": screen.n_sampled,
            "n_yb": n_yb,
            "msun_hits": hits,
            "yb_msun_hits": yb_hits,
            "msun_rate": hits / screen.n_sampled,
            "yb_msun_rate": yb_rate,
            "non_yb_msun_rate": non_yb_rate,
            "delta_percentage_points": 100 * (non_yb_rate - hits / screen.n_sampled)
            if non_yb_rate is not None
            else None,
            "sun_hits": int(weights[sun].sum()),
            "yb_sun_hits": int(weights[sun & is_yb].sum()),
            "yb_representatives": int(is_yb.sum()),
            "yb_novel_hits": int(weights[mask & is_yb].sum()),
            "yb_no_hull": int(
                (
                    frame.e_above_hull.isna()
                    & is_yb
                    & frame.has_structure.fillna(False)
                ).sum()
            ),
            "yb_median_ehull": frame.loc[is_yb, "e_above_hull"].median(),
            "wandb_summary_rate": published,
            "summary_matches": published is not None
            and abs(published - hits / screen.n_sampled) < 1e-12,
            "reference_cache": manifest.get("reference_cache"),
            "mlip": manifest.get("mlip"),
            "hull_type": manifest.get("hull", {}).get("hull_type"),
        }
        rows.append(row)
    info = {
        "name": artifact_name,
        "id": artifact.id,
        "digest": artifact.digest,
        "created_at": artifact.created_at,
        "source_run": source_run,
        "metadata": dict(artifact.metadata),
        "manifest": manifest,
    }
    print(
        f"{artifact_name}: Yb={n_yb}; free MSUN {rows[0]['msun_rate']:.3f} -> {rows[0]['non_yb_msun_rate']:.6f}",
        flush=True,
    )
    return rows, info


def main() -> None:
    pd.set_option("future.no_silent_downcasting", True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--log",
        action="store_true",
        help="Preserve the audit in a new W&B analysis run.",
    )
    args = parser.parse_args()
    root = args.output_dir or runs_root() / "yb_protocol_audit" / datetime.now(
        UTC
    ).strftime("%Y%m%dT%H%M%SZ")
    root.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=90)
    summaries = {
        r.id: dict(r.summary)
        for r in api.runs(
            PROJECT,
            filters={
                "summary_metrics.protocol/free/metasun_per_sampled_gene": {
                    "$exists": True
                }
            },
        )
    }
    collections = list(
        api.artifact_type("protocol_eval", project=PROJECT).collections()
    )
    with ThreadPoolExecutor(max_workers=6) as executor:
        artifacts = list(
            executor.map(lambda c: api.artifact(f"{PROJECT}/{c.name}:latest"), collections)
        )
        results = list(executor.map(lambda a: analyse(a, root, summaries), artifacts))
    rows = [row for result, _ in results for row in result]
    table = pd.DataFrame(rows).sort_values(
        ["baseline", "section", "source_run", "artifact"],
        ascending=[False, True, True, True],
    )
    table.to_csv(root / "yb_exclusion.csv", index=False)
    provenance = {
        "audit_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "project": PROJECT,
        "artifact_selection": "latest alias of every protocol_eval collection; this can differ from the highest version number",
        "definition": "has_structure & valid_structure & unique_structure & novel_structure & e_above_hull <= 0.1; weighted by screen.counts; denominator includes all sampled genes",
        "exclusion": "Remove every sampled gene containing Yb from numerator and denominator; retain recorded verdicts and energies",
        "sources": [info for _, info in results],
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"OUTPUT {root}", flush=True)
    print(
        table.loc[
            (table.section == "free") & table.baseline,
            [
                "source_run",
                "n_sampled",
                "n_yb",
                "yb_msun_hits",
                "msun_rate",
                "non_yb_msun_rate",
                "delta_percentage_points",
                "summary_matches",
            ],
        ].to_string(index=False)
    )
    if args.log:
        log_audit(root)


def log_audit(root: Path) -> None:
    """Publish a completed audit without changing any evaluated run."""
    table = pd.read_csv(root / "yb_exclusion.csv")
    table["reference_cache_canonical"] = table.reference_cache.map(
        lambda p: str(as_cache_dir(p))
    )
    table.to_csv(root / "yb_exclusion.csv", index=False)
    provenance = json.loads((root / "provenance.json").read_text())
    pooled = []
    # Keep different novelty references and potentials in separate pools.
    for (baseline, section, reference, mlip), group in table.groupby(
        ["baseline", "section", "reference_cache_canonical", "mlip"]
    ):
        n = int(group.n_sampled.sum())
        ny = int(group.n_yb.sum())
        hits = int(group.msun_hits.sum())
        hy = int(group.yb_msun_hits.sum())
        pooled.append(
            {
                "baseline": bool(baseline),
                "section": section,
                "reference_cache": reference,
                "mlip": mlip,
                "artifacts": len(group),
                "n_sampled": n,
                "n_yb": ny,
                "msun_hits": hits,
                "yb_msun_hits": hy,
                "msun_rate": hits / n,
                "yb_msun_rate": hy / ny if ny else None,
                "non_yb_msun_rate": (hits - hy) / (n - ny),
                "delta_percentage_points": 100 * ((hits - hy) / (n - ny) - hits / n),
            }
        )
    (root / "pooled.json").write_text(json.dumps(pooled, indent=2) + "\n")
    with wandb.init(
        entity="symmetry-advantage",
        project="WyckoffTransformer",
        dir=wandb_dir(),
        job_type="analysis",
        name="yb-protocol-audit-" + root.name,
        config={
            "git_commit": provenance["git_commit"],
            "source_artifact_count": table.artifact.nunique(),
            "definition": provenance["definition"],
            "exclusion": provenance["exclusion"],
        },
        mode="online",
    ) as run:
        artifact = wandb.Artifact(
            "yb-protocol-audit-" + root.name,
            type="protocol_analysis",
            metadata={
                "git_commit": provenance["git_commit"],
                "audit_utc": provenance["audit_utc"],
                "source_artifact_count": int(table.artifact.nunique()),
            },
        )
        for name in ("yb_exclusion.csv", "provenance.json", "pooled.json"):
            artifact.add_file(str(root / name), name=name)
        for source in provenance["sources"]:
            artifact.add_dir(
                str(root / "inputs" / source["name"]), name="inputs/" + source["name"]
            )
        artifact.add_file(
            "scripts/analyse_yb_protocol.py", name="analyse_yb_protocol.py"
        )
        run.log_artifact(artifact)
        run.log({"yb_exclusion": wandb.Table(dataframe=table)})
        for row in pooled:
            prefix = ("baseline" if row["baseline"] else "arms") + "/" + row["section"]
            for key in (
                "artifacts",
                "n_sampled",
                "n_yb",
                "msun_hits",
                "yb_msun_hits",
                "msun_rate",
                "yb_msun_rate",
                "non_yb_msun_rate",
                "delta_percentage_points",
            ):
                run.summary[prefix + "/" + key] = row[key]
        print("AUDIT_RUN", run.url, flush=True)


if __name__ == "__main__":
    main()
