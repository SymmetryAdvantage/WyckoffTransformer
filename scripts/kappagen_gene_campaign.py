"""Generate and screen Wyckoff genes for the kappagen ternary systems and their binaries.

For every ternary system in the list and every distinct binary subsystem of them:
draw N genes from the chemical-system conditioned generator, deduplicate them by
the canonical 128-bit gene key, and score the unique ones with the gene-minimum
formation-energy regressor against the LeMat-Bulk PBE hull. ``merge`` then
deduplicates across systems (a pool conditioned on A-B-C can answer with A-B) and
keeps the genes predicted within ``--max-e-hull`` of the hull.

    run_in_singularity.sh python scripts/kappagen_gene_campaign.py generate \\
        --out $WYFORMER_RUNS/kappagen_genes --shard 0 --n-shards 4 --device cuda
    run_in_singularity.sh python scripts/kappagen_gene_campaign.py merge \\
        --out $WYFORMER_RUNS/kappagen_genes --wandb

Every system writes ``systems/<system>.parquet`` when done, and a system whose file
exists is skipped, so a shard re-submitted after its walltime resumes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wyckoff_transformer.chemical_system import parse_chemical_system
from wyckoff_transformer.cli.csp import load_trainer
from wyckoff_transformer.cli.gene_screen import (
    DEFAULT_REFERENCE,
    check_regressor_reference,
    load_reference,
    score_genes,
)
from wyckoff_transformer.cli.generate import chemical_system_vector_for_generation
from wyckoff_transformer.evaluation.gene_hash import gene_key
from wyckoff_transformer.evaluation.protocol import GeneFingerprinter
from wyckoff_transformer.formula_energy.screen import HullLookup
from wyckoff_transformer.paths import runs_root

logger = logging.getLogger("kappagen_gene_campaign")

GENERATOR_RUN = "chemsys_sg_uncond_adanmw_wsd-20260925-030937"
REGRESSOR_RUN = "min_energy_adamw_wsd_h7x_xl-20260930-030055"
SYSTEMS_LIST = Path("docs/archive/final_kappagen_plus_si_sparse_or_disordered_ternary_systems.md")


def campaign_systems(path: Path) -> list[str]:
    """The ternaries in *path*, then every distinct binary subsystem of them."""
    ternaries = []
    for line in path.read_text().splitlines():
        parts = line.strip().split("-")
        if len(parts) == 3 and all(parts):
            ternaries.append("-".join(sorted(parts)))
    ternaries = sorted(set(ternaries))
    binaries = sorted({"-".join(pair) for system in ternaries
                       for pair in combinations(system.split("-"), 2)})
    return ternaries + binaries


def system_seed(system: str, base: int) -> int:
    digest = hashlib.blake2b(f"{base}:{system}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def generate_pool(trainer, system: str, n_genes: int, batch: int, seed: int) -> list[dict]:
    """*n_genes* draws conditioned on, and masked to, exactly *system*."""
    torch.manual_seed(seed)
    elements_tokeniser = trainer.tokenisers["elements"]
    _, tokens = parse_chemical_system(system, elements_tokeniser)
    # allowed_element_set alone does not switch the element constraint on in
    # generate_structures (only a required set or a per-row mask does), so the mask.
    row_mask = torch.zeros(len(elements_tokeniser), dtype=torch.bool, device=trainer.device)
    row_mask[list(tokens)] = True
    row_mask[elements_tokeniser.stop_token] = True
    genes = []
    for start in range(0, n_genes, batch):
        rows = min(batch, n_genes - start)
        cond = chemical_system_vector_for_generation(
            system, elements_tokeniser, rows, trainer.device)
        with torch.no_grad():
            genes.extend(trainer.generate_structures(
                n_structures=rows, calibrate=False, composition_cond=cond,
                allowed_element_mask=row_mask.unsqueeze(0).expand(rows, -1)))
    return genes


def dedup_pool(genes: list[dict], fingerprinter: GeneFingerprinter) -> pd.DataFrame:
    """One row per distinct gene key, the first occurrence kept, with its count."""
    rows = []
    for gene in genes:
        try:
            key = gene_key(fingerprinter.record(gene))
        except (KeyError, TypeError, ValueError):
            continue
        rows.append((key[0], key[1], gene))
    frame = pd.DataFrame(rows, columns=["key_low", "key_high", "gene"])
    counts = frame.groupby(["key_low", "key_high"]).size().rename("count")
    frame = frame.drop_duplicates(["key_low", "key_high"], keep="first")
    frame = frame.join(counts, on=["key_low", "key_high"]).reset_index(drop=True)
    return frame


def run_generate(args) -> None:
    systems = campaign_systems(args.systems_list)
    mine = args.only.split(",") if args.only else systems[args.shard::args.n_shards]
    out_dir = args.out / "systems"
    raw_dir = args.out / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in mine if not (out_dir / f"{s}.parquet").exists()]
    logger.info("%d systems in all, %d in shard %d/%d, %d left",
                len(systems), len(mine), args.shard, args.n_shards, len(todo))
    if not todo:
        return

    generator = load_trainer(device=args.device, model_path=runs_root() / GENERATOR_RUN)
    if hasattr(generator.model, "_orig_mod"):
        generator.model = generator.model._orig_mod
    generator.model.eval()
    regressor = load_trainer(device=args.device, model_path=runs_root() / REGRESSOR_RUN)
    check_regressor_reference(regressor, args.reference)
    reference = load_reference(args.reference)
    lookup = HullLookup(reference)
    fingerprinter = GeneFingerprinter()

    for system in todo:
        t0 = time.time()
        genes = generate_pool(generator, system, args.n_genes, args.batch,
                              system_seed(system, args.seed))
        with gzip.open(raw_dir / f"{system}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(genes, handle)
        t1 = time.time()
        pool = dedup_pool(genes, fingerprinter)
        t2 = time.time()
        scored = []
        for start in range(0, len(pool), args.score_batch):
            part = pool.iloc[start:start + args.score_batch]
            result = score_genes(list(part["gene"]), regressor, reference, hull_lookup=lookup)
            result = result.sort_index()
            result.index = part.index
            scored.append(result)
        scored = pd.concat(scored) if scored else pd.DataFrame()
        pool = pool.join(scored[["formula", "predicted_formation_energy", "hull_energy",
                                 "score", "reason"]])
        pool["requested_system"] = system
        pool["chemsys"] = pool["gene"].map(lambda g: "-".join(sorted(set(g["species"]))))
        pool["spacegroup_number"] = pool["gene"].map(lambda g: int(g["group"]))
        pool["n_atoms"] = pool["gene"].map(lambda g: int(sum(g["numIons"])))
        pool["gene"] = pool["gene"].map(json.dumps)
        tmp = out_dir / f"{system}.parquet.tmp"
        pool.to_parquet(tmp, index=False)
        tmp.rename(out_dir / f"{system}.parquet")
        logger.info(
            "%s: %d valid of %d draws, %d unique, %d scored, %d with e_hull <= %.2f "
            "(generate %.0f s, dedup %.0f s, score %.0f s)",
            system, len(genes), args.n_genes, len(pool), int(pool["score"].notna().sum()),
            int((pool["score"] <= args.max_e_hull).sum()), args.max_e_hull,
            t1 - t0, t2 - t1, time.time() - t2)


def run_merge(args) -> None:
    systems = campaign_systems(args.systems_list)
    out_dir = args.out / "systems"
    missing = [s for s in systems if not (out_dir / f"{s}.parquet").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} systems not done yet, e.g. {missing[:5]}")
    frames = [pd.read_parquet(out_dir / f"{s}.parquet") for s in systems]
    pools = pd.concat(frames, ignore_index=True)
    raw_valid = {s: int(f["count"].sum()) for s, f in zip(systems, frames)}

    # Across systems: the first system in campaign order keeps the gene, counts add up.
    grouped = pools.groupby(["key_low", "key_high"], sort=False)
    total_count = grouped["count"].transform("sum")
    requested_by = grouped["requested_system"].transform(lambda s: ",".join(sorted(set(s))))
    pools = pools.assign(count=total_count, requested_by=requested_by)
    unique = pools.drop_duplicates(["key_low", "key_high"], keep="first").reset_index(drop=True)
    # A pool may answer with a unary or, rarely, a subsystem of another target; only
    # genes whose own system is a campaign system are kept.
    unique["in_campaign"] = unique["chemsys"].isin(set(systems))
    selected = unique[unique["in_campaign"] & (unique["score"] <= args.max_e_hull)]
    selected = selected.sort_values("score", kind="stable")

    unique.drop(columns=["gene"]).to_parquet(args.out / "unique_genes_scored.parquet",
                                             index=False)
    selected.to_parquet(args.out / "selected_genes.parquet", index=False)
    with gzip.open(args.out / "selected_genes.json.gz", "wt", encoding="utf-8") as handle:
        json.dump([json.loads(g) for g in selected["gene"]], handle)

    summary = pd.DataFrame({
        "requested_system": systems,
        "arity": [len(s.split("-")) for s in systems],
        "draws": args.n_genes,
        "valid": [raw_valid[s] for s in systems],
        "unique_in_pool": [len(f) for f in frames],
        "unique_after_merge": unique.groupby("requested_system").size().reindex(systems)
                                    .fillna(0).astype(int).values,
        "selected": selected.groupby("requested_system").size().reindex(systems)
                            .fillna(0).astype(int).values,
    })
    by_chemsys = selected.groupby("chemsys").size().rename("selected").reset_index()
    summary.to_csv(args.out / "summary_by_requested_system.csv", index=False)
    by_chemsys.to_csv(args.out / "selected_by_chemsys.csv", index=False)
    totals = {
        "n_systems": len(systems),
        "n_ternary_systems": int((summary["arity"] == 3).sum()),
        "n_binary_systems": int((summary["arity"] == 2).sum()),
        "n_draws": int(len(systems) * args.n_genes),
        "n_valid": int(summary["valid"].sum()),
        "n_unique_within_pools": int(summary["unique_in_pool"].sum()),
        "n_unique": int(len(unique)),
        "n_scored": int(unique["score"].notna().sum()),
        "n_unique_in_campaign_systems": int(unique["in_campaign"].sum()),
        "n_selected": int(len(selected)),
        "n_selected_ternary": int((selected["chemsys"].str.count("-") == 2).sum()),
        "n_selected_binary": int((selected["chemsys"].str.count("-") == 1).sum()),
        "max_e_hull": args.max_e_hull,
        "generator": GENERATOR_RUN,
        "regressor": REGRESSOR_RUN,
        "reference": str(args.reference),
    }
    (args.out / "totals.json").write_text(json.dumps(totals, indent=1))
    logger.info("Totals: %s", totals)

    if args.wandb:
        upload(args, totals, summary, by_chemsys)


def upload(args, totals: dict, summary: pd.DataFrame, by_chemsys: pd.DataFrame) -> None:
    import wandb  # noqa: PLC0415

    from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT  # noqa: PLC0415
    from wyckoff_transformer.paths import wandb_dir  # noqa: PLC0415

    run = wandb.init(
        dir=wandb_dir(), entity=WANDB_ENTITY, project=WANDB_PROJECT,
        name=f"{args.out.name}_{time.strftime('%Y%m%d')}", job_type="gene_campaign",
        tags=["kappagen", "chemsys", "gene_screen"],
        config={**totals, "n_genes_per_system": args.n_genes, "seed": args.seed},
        notes="100k genes per kappagen ternary and per binary subsystem, deduplicated, "
              f"kept at predicted e_hull <= {args.max_e_hull} eV/atom.")
    run.summary.update(totals)
    run.log({"summary_by_requested_system": wandb.Table(dataframe=summary),
             "selected_by_chemsys": wandb.Table(dataframe=by_chemsys)})
    artifact = wandb.Artifact(
        "kappagen_selected_genes", type="wyckoff_genes",
        description=f"Genes predicted within {args.max_e_hull} eV/atom of the PBE hull.",
        metadata=totals)
    for name in ("selected_genes.json.gz", "selected_genes.parquet",
                 "unique_genes_scored.parquet", "summary_by_requested_system.csv",
                 "selected_by_chemsys.csv", "totals.json"):
        artifact.add_file(str(args.out / name))
    run.log_artifact(artifact)
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("generate", "merge", "list"))
    parser.add_argument("--out", type=Path, default=runs_root() / "kappagen_genes")
    parser.add_argument("--systems-list", type=Path, default=SYSTEMS_LIST)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--n-genes", type=int, default=100_000)
    parser.add_argument("--batch", type=int, default=25_000, help="Draws per forward pass.")
    parser.add_argument("--score-batch", type=int, default=20_000)
    parser.add_argument("--max-e-hull", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated systems to run instead of the shard.")
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("wyckoff_transformer.wyckoff_processor").setLevel(logging.WARNING)
    if args.stage == "list":
        for system in campaign_systems(args.systems_list):
            print(system)
    elif args.stage == "generate":
        run_generate(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
