"""Replay Yb exclusion on current ROE artifacts without changing source runs."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import wandb
from analyse_roe_surprisal import _as_bool, build_arms, evaluate

from wyckoff_transformer.evaluation.protocol import (
    funnel_structure_metrics,
    load_genes,
    read_screen,
)
from wyckoff_transformer.paths import runs_root, wandb_dir

PROJECT = "symmetry-advantage/WyckoffTransformer"
MODERN = (
    "roe_surprisal_cfg-20261008",
    "roe_surprisal_uncond-20261008",
    "roe_surprisal_cfg_xl-20261012",
    "roe_surprisal_cfg_xl_w3-20261012",
    "roe_surprisal_cfg_xl_w2-20261012",
    "roe_surprisal_cfg_xl_w1-20261012",
)
LEGACY = {
    "roe_cfg_c0p05_w3": ("broadside", "fire-discipline", "fire-control"),
    "roe_torpedo_chemsys_sg_uncond_adanmw_wsd-20260924": ("joint", "reference"),
    "roe_torpedo_chemsys_sg_uncond_adanmw_wsd-20260924_pilot": ("joint", "reference"),
}


def download(name: str, root: Path):
    api = wandb.Api(timeout=90)
    artifact = api.artifact(f"{PROJECT}/{name}:latest")
    pinned = name + ":" + artifact.version
    directory = root / "inputs" / pinned
    if name in MODERN:
        files = [
            "pool/wyckoff_genes.json.gz",
            "pool/pool_manifest.json",
            "analysis/genes.csv.gz",
            "analysis/report.json",
            "protocol/structures.csv",
            "protocol/screen.json",
            "protocol/manifest.json",
        ]
    else:
        files = ["pool/wyckoff_genes.json.gz"]
        for arm in LEGACY[name]:
            files.extend(
                [
                    f"{arm}/engaged_genes.json.gz",
                    f"{arm}/protocol/structures.csv",
                    f"{arm}/protocol/screen.json",
                    f"{arm}/protocol/funnel.json",
                    f"{arm}/protocol/manifest.json",
                ]
            )
    for filename in files:
        if not (directory / filename).exists():
            artifact.get_entry(filename).download(root=str(directory))
    print("Downloaded", pinned, flush=True)
    return directory, {
        "name": pinned,
        "id": artifact.id,
        "digest": artifact.digest,
        "metadata": dict(artifact.metadata),
        "files": files,
    }


def modern(directory: Path, name: str):
    genes = load_genes(directory / "pool/wyckoff_genes.json.gz")
    frame = pd.read_csv(directory / "analysis/genes.csv.gz", index_col=0)
    frame["has_yb"] = ["Yb" in genes[int(i)]["species"] for i in frame.index]
    frame["source_index"] = frame.index
    for col in (
        "is_representative",
        "gene_novel",
        "metasun",
        "sun",
        "metastable",
        "novel_structure",
    ):
        frame[col] = _as_bool(frame[col])
    screen = read_screen(directory / "protocol/screen.json")
    assert len(genes) == len(frame) == screen.n_sampled
    assert (
        Counter(frame.loc[frame.representative.ge(0), "representative"])
        == screen.counts
    )
    assert frame.loc[frame.is_representative].index.tolist() == sorted(screen.counts)
    structures = pd.read_csv(directory / "protocol/structures.csv", index_col=0)
    for col in (
        "has_structure",
        "valid_structure",
        "unique_structure",
        "novel_structure",
        "relaxed_fingerprint_changed",
    ):
        structures[col] = _as_bool(structures[col])
    structures["metasun"] = structures[
        ["has_structure", "valid_structure", "unique_structure", "novel_structure"]
    ].all(axis=1) & structures.e_above_hull.le(0.1)
    report = json.loads((directory / "analysis/report.json").read_text())
    assert structures.metasun.sum() == report["pool"]["metasun_unique"]
    mapped = structures.metasun.reindex(frame.representative).fillna(False).to_numpy()
    assert (mapped == frame.metasun.to_numpy()).all()
    # Removing all samples of a Yb fingerprint leaves the first representative
    # of every remaining class intact. Keep its original id for outcome lookup;
    # reset only sample ordering so draws consumed is measured on the clean pool.
    clean = frame.loc[~frame.has_yb].reset_index(drop=True)
    variants = report["variants"]
    rows = []
    for budget_text, published in report["arms"].items():
        budget = int(budget_text)
        before = build_arms(frame, budget, variants)
        after = build_arms(clean, budget, variants)
        for arm, (slots, consumed, _, _) in before.items():
            original = evaluate(frame, structures, slots, consumed, budget, None)
            for key in ("metasun", "sun", "filled", "duplicate_slots"):
                assert original[key] == published[arm][key], (name, budget, arm, key)
            original_rows = frame.loc[slots]
            retained_slots = original_rows.index[~original_rows.has_yb].tolist()
            ny = int(original_rows.has_yb.sum())
            retained_budget = budget - ny
            filtered = evaluate(
                frame, structures, retained_slots, consumed, retained_budget, None
            )
            new_slots, new_consumed, _, _ = after[arm]
            replay = evaluate(clean, structures, new_slots, new_consumed, budget, None)
            old_reps = set(
                original_rows.representative[original_rows.representative.ge(0)]
            )
            yb_reps = set(original_rows.loc[original_rows.has_yb, "representative"])
            yb_hits = int(
                structures.reindex(sorted(old_reps & yb_reps))
                .metasun.fillna(False)
                .sum()
            )
            new_reps = set(clean.loc[new_slots, "representative"])
            lost = int(
                structures.reindex(sorted(old_reps - new_reps))
                .metasun.fillna(False)
                .sum()
            )
            gained = int(
                structures.reindex(sorted(new_reps - old_reps))
                .metasun.fillna(False)
                .sum()
            )
            assert replay["metasun"] - original["metasun"] == gained - lost
            rows.append(
                {
                    "run": name,
                    "budget": budget,
                    "arm": arm,
                    "original_msun": original["metasun_rate"],
                    "selected_yb": ny,
                    "selected_yb_msun": yb_hits,
                    "filter_only_msun": filtered["metasun_rate"],
                    "replay_msun": replay["metasun_rate"],
                    "replay_delta_pp": 100
                    * (replay["metasun_rate"] - original["metasun_rate"]),
                    "hits_lost": lost,
                    "hits_gained": gained,
                    "original_filled": original["filled"],
                    "replay_filled": replay["filled"],
                    "original_sun": original["sun_rate"],
                    "replay_sun": replay["sun_rate"],
                    "original_msun_per_relax_hour": original["metasun_per_relax_hour"],
                    "replay_msun_per_relax_hour": replay["metasun_per_relax_hour"],
                }
            )
    pool = {
        "run": name,
        "sampled": len(frame),
        "yb_sampled": int(frame.has_yb.sum()),
        "msun_sampled": int(frame.metasun.sum()),
        "yb_msun_sampled": int(frame.loc[frame.has_yb, "metasun"].sum()),
        "msun_rate": float(frame.metasun.mean()),
        "non_yb_msun_rate": float(clean.metasun.mean()),
        "yb_representatives": int((frame.has_yb & frame.is_representative).sum()),
    }
    print(
        name,
        "Yb",
        pool["yb_sampled"],
        "pool MSUN",
        pool["msun_rate"],
        "->",
        pool["non_yb_msun_rate"],
        flush=True,
    )
    return rows, pool


def legacy(directory: Path, name: str):
    pool_genes = load_genes(directory / "pool/wyckoff_genes.json.gz")
    pool_yb = sum("Yb" in gene["species"] for gene in pool_genes)
    rows = []
    for arm in LEGACY[name]:
        genes = load_genes(directory / arm / "engaged_genes.json.gz")
        screen = read_screen(directory / arm / "protocol/screen.json")
        frame = pd.read_csv(directory / arm / "protocol/structures.csv", index_col=0)
        for col in (
            "has_structure",
            "valid_structure",
            "unique_structure",
            "novel_structure",
            "relaxed_fingerprint_changed",
        ):
            frame[col] = _as_bool(frame[col])
        published = json.loads((directory / arm / "protocol/funnel.json").read_text())
        original = funnel_structure_metrics(screen, frame)
        assert (
            abs(
                original["metasun_per_sampled_gene"]
                - published["free"]["metasun_per_sampled_gene"]
            )
            < 1e-12
        )
        yb = {i for i, gene in enumerate(genes) if "Yb" in gene["species"]}
        mask = frame[
            ["has_structure", "valid_structure", "unique_structure", "novel_structure"]
        ].all(axis=1) & frame.e_above_hull.le(0.1)
        weights = pd.Series(screen.counts).reindex(frame.index).fillna(0)
        hits = int(weights[mask].sum())
        yb_hits = int(weights[mask & frame.index.isin(yb)].sum())
        rows.append(
            {
                "run": name,
                "arm": arm,
                "sampled": screen.n_sampled,
                "pool_yb": pool_yb,
                "selected_yb": len(yb),
                "selected_yb_msun": yb_hits,
                "original_msun": original["metasun_per_sampled_gene"],
                "filter_only_msun": (hits - yb_hits) / (screen.n_sampled - len(yb)),
                "replay_msun": original["metasun_per_sampled_gene"]
                if pool_yb == 0
                else None,
            }
        )
    print(
        name, "pool Yb", pool_yb, "arm Yb", [r["selected_yb"] for r in rows], flush=True
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    pd.set_option("future.no_silent_downcasting", True)
    root = args.output_dir or runs_root() / "yb_roe_audit" / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    root.mkdir(parents=True, exist_ok=True)
    names = [*MODERN, *LEGACY]
    with ThreadPoolExecutor(max_workers=6) as pool:
        sources = list(pool.map(lambda name: download(name, root), names))
    rows, pools, older = [], [], []
    for name, (directory, _) in zip(names, sources):
        if name in MODERN:
            table, stats = modern(directory, name)
            rows.extend(table)
            pools.append(stats)
        else:
            older.extend(legacy(directory, name))
    pd.DataFrame(rows).to_csv(root / "roe_yb_replay.csv", index=False)
    pd.DataFrame(pools).to_csv(root / "pools.csv", index=False)
    pd.DataFrame(older).to_csv(root / "legacy_yb.csv", index=False)
    provenance = {
        "audit_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "sources": [info for _, info in sources],
        "method": "Remove all Yb samples before build_arms; recompute quantile bands and fill the original reconstruction budget from surviving pool. Stored predictions and relaxation outcomes unchanged. Also report deletion from the original selections with reduced denominator. Standard MetaSUN only; no new strict StructureMatcher deduplication.",
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    main_table = pd.DataFrame(rows).query("budget == 1000")
    print(
        main_table[
            [
                "run",
                "arm",
                "selected_yb",
                "selected_yb_msun",
                "original_msun",
                "replay_msun",
                "replay_delta_pp",
            ]
        ].to_string(index=False)
    )
    print("OUTPUT", root, flush=True)
    if args.log:
        with wandb.init(
            entity="symmetry-advantage",
            project="WyckoffTransformer",
            dir=wandb_dir(),
            name="yb-roe-audit-" + root.name,
            job_type="analysis",
            mode="online",
            config={
                "git_commit": provenance["git_commit"],
                "method": provenance["method"],
            },
        ) as run:
            artifact = wandb.Artifact(
                "yb-roe-audit-" + root.name,
                type="protocol_analysis",
                metadata={
                    "git_commit": provenance["git_commit"],
                    "audit_utc": provenance["audit_utc"],
                },
            )
            artifact.add_dir(str(root))
            artifact.add_file("scripts/analyse_yb_roe.py", name="analyse_yb_roe.py")
            artifact.add_file(
                "scripts/analyse_roe_surprisal.py", name="analyse_roe_surprisal.py"
            )
            run.log_artifact(artifact)
            run.log({"B1000_yb_replay": wandb.Table(dataframe=main_table)})
            print("AUDIT_RUN", run.url, flush=True)


if __name__ == "__main__":
    main()
