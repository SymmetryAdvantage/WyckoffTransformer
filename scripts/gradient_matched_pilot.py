#!/usr/bin/env python3
"""Pilot: energy a LeMat-Bulk row would release had its relaxation converged fully.

For each sampled row, conservative ORB supplies only the curvature: its energy surface
is pinned to the archived DFT forces and stress at the archived geometry
(:mod:`wyckoff_transformer.cryspr.gradient_matched`), and three estimates of the drop
are recorded -- a symmetry-fixed relaxation, a Newton step, and the steepest-descent
lower bound.

Stages, each resumable from ``--output-dir``:

1. ``row_stats.parquet``  per-row force/stress presence and magnitude for the archive
   (or ``--row-stats`` to reuse an existing audit with the same columns);
2. ``sample.parquet``     the stratified sample with structures, forces and stress;
3. ``results/*.jsonl``    one line per row, appended as each finishes;
4. ``results.parquet``    everything merged.

Rows: ``lemat_bulk_fmax1`` only, with forces and stress present, at most
``--max-sites`` sites, stratified by source x max |F| bin x max |stress| bin.

Example (zeus, GPU 0)::

    CUDA_VISIBLE_DEVICES=0 python scripts/gradient_matched_pilot.py --device cuda --workers 6
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from wyckoff_transformer.paths import cache_root, data_root

RAW = data_root() / "lemat-bulk" / "raw" / "data.parquet"
FMAX1 = data_root() / "lemat_bulk_fmax1"
logger = logging.getLogger("gradient_matched_pilot")

#: eV/A. The first bin is exactly zero: rows with no free positional parameter.
FORCE_EDGES = [0.0, 0.005, 0.02, 0.1, 1.0]
FORCE_LABELS = ["0", "(0,0.005]", "(0.005,0.02]", "(0.02,0.1]", "(0.1,1]"]
#: kBar, on max |stress component|. Medians are ~2 kBar for Alexandria and MP and ~9 for
#: OQMD, whose p90 is 33.
STRESS_EDGES = [0.0, 1.0, 5.0, 20.0, np.inf]
STRESS_LABELS = ["[0,1]", "(1,5]", "(5,20]", ">20"]
MAX_NEIGHBORS = 120  # orb-v3 adapter default; beyond it the knn graph truncates
NEIGHBOR_RADIUS = 6.0


def _source(ids: pd.Series) -> pd.Series:
    text = ids.astype(str)
    return pd.Series(np.where(text.str.startswith("mp-"), "mp",
                              np.where(text.str.startswith("agm"), "agm", "oqmd")), index=ids.index)


def _max_abs_nested(col) -> np.ndarray:
    arr = col.combine_chunks()
    n = len(arr)
    inner = arr.flatten()
    row_of_inner = pc.list_parent_indices(arr).to_numpy()
    values = inner.flatten().fill_null(np.nan).to_numpy(zero_copy_only=False)
    row_of_value = row_of_inner[pc.list_parent_indices(inner).to_numpy()]
    out = np.full(n, -np.inf)
    ok = np.isfinite(values)
    np.maximum.at(out, row_of_value[ok], np.abs(values[ok]))
    out[out == -np.inf] = np.nan
    return out


def build_row_stats(path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(RAW)
    parts = []
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=["immutable_id", "nsites", "forces", "stress_tensor"])
        parts.append(pd.DataFrame({
            "immutable_id": t.column("immutable_id").to_pandas(),
            "nsites": t.column("nsites").to_numpy(),
            "max_abs_force": _max_abs_nested(t.column("forces")),
            "max_abs_stress": _max_abs_nested(t.column("stress_tensor")),
        }))
    stats = pd.concat(parts, ignore_index=True)
    stats.to_parquet(path, index=False)
    return stats


def load_fmax1_ids(path: Path | None) -> pd.Index:
    if path is not None:
        return pd.Index(pd.read_parquet(path)["immutable_id"])
    ids = [pd.read_csv(FMAX1 / f"{split}.csv.gz", usecols=["immutable_id"])["immutable_id"]
           for split in ("train", "val", "test")]
    return pd.Index(pd.concat(ids))


def build_sample(args, out: Path) -> pd.DataFrame:
    stats_path = args.output_dir / "row_stats.parquet"
    if args.row_stats is not None:
        stats = pd.read_parquet(args.row_stats)
    elif stats_path.exists():
        stats = pd.read_parquet(stats_path)
    else:
        logger.info("computing per-row force/stress statistics over the archive")
        stats = build_row_stats(stats_path)
    # Positional index in the raw parquet, needed to read the rows back.
    stats = stats.reset_index(drop=True)
    stats["raw_index"] = np.arange(len(stats))
    keep = stats["immutable_id"].isin(load_fmax1_ids(args.fmax1_ids))
    keep &= stats["max_abs_force"].notna() & stats["max_abs_stress"].notna()
    total = int(keep.sum())
    keep &= stats["nsites"] <= args.max_sites
    logger.info("fmax1 rows with forces and stress: %d; at most %d sites: %d (%.2f%% excluded)",
                total, args.max_sites, int(keep.sum()), 100 * (1 - keep.sum() / total))
    pool = stats[keep].copy()
    pool["source"] = _source(pool["immutable_id"])
    pool["force_bin"] = np.where(
        pool["max_abs_force"] == 0, FORCE_LABELS[0],
        pd.cut(pool["max_abs_force"], FORCE_EDGES, labels=FORCE_LABELS[1:], include_lowest=False).astype(str))
    pool["stress_bin"] = pd.cut(pool["max_abs_stress"], STRESS_EDGES, labels=STRESS_LABELS,
                                include_lowest=True).astype(str)
    counts = pool.groupby(["source", "force_bin", "stress_bin"]).size()
    logger.info("stratum population:\n%s", counts.unstack("stress_bin").to_string())
    sample = (pool.groupby(["source", "force_bin", "stress_bin"], group_keys=False)
              .apply(lambda g: g.sample(min(len(g), args.n_per_stratum), random_state=args.seed)))
    sample = sample.merge(counts.rename("stratum_population").reset_index(),
                          on=["source", "force_bin", "stress_bin"])

    pf = pq.ParquetFile(RAW)
    starts = np.cumsum([0] + [pf.metadata.row_group(g).num_rows for g in range(pf.metadata.num_row_groups)])
    sample["row_group"] = np.searchsorted(starts, sample["raw_index"].to_numpy(), side="right") - 1
    columns = ["immutable_id", "lattice_vectors", "cartesian_site_positions", "species_at_sites",
               "forces", "stress_tensor", "energy"]
    parts = []
    for g, rows in sample.groupby("row_group"):
        t = pf.read_row_group(int(g), columns=columns).take(rows["raw_index"].to_numpy() - starts[g])
        parts.append(t.to_pandas())
    structures = pd.concat(parts)
    for name in ("lattice_vectors", "cartesian_site_positions", "forces", "stress_tensor"):
        structures[name] = structures[name].map(lambda v: np.stack(v).tolist())
    structures["species_at_sites"] = structures["species_at_sites"].map(list)
    sample = sample.merge(structures, on="immutable_id", validate="one_to_one")
    sample = sample.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sample.to_parquet(out, index=False)
    logger.info("wrote %s: %d rows", out, len(sample))
    return sample


def _spacegroup(atoms, symprec: float) -> int:
    import spglib
    dataset = spglib.get_symmetry_dataset((atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers),
                                          symprec=symprec)
    return int(dataset.number) if dataset is not None else -1


def _orb_batch_evaluator(model, adapter, device, max_atoms_per_batch: int):
    import torch

    def evaluate(structures):
        results = []
        start = 0
        while start < len(structures):
            chunk, n = [], 0
            while start < len(structures) and (not chunk or n + len(structures[start]) <= max_atoms_per_batch):
                chunk.append(structures[start]); n += len(structures[start]); start += 1
            with torch.enable_grad():
                batch = adapter.from_ase_atoms_list(chunk, device=device)
                out = model.predict(batch.to(device), split=True)
            for i in range(len(chunk)):
                stress = out[model.grad_stress_name][i].detach().cpu().numpy().reshape(6)
                results.append((float(out["energy"][i].item()),
                                out[model.grad_forces_name][i].detach().cpu().numpy(),
                                np.array([[stress[0], stress[5], stress[4]],
                                          [stress[5], stress[1], stress[3]],
                                          [stress[4], stress[3], stress[2]]])))
        return results
    return evaluate


def process_row(row, calc, batch_eval, args) -> dict:
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    from wyckoff_transformer.cryspr import gradient_matched as gm

    def matrix(nested) -> np.ndarray:
        # parquet hands nested lists back as object arrays of arrays
        return np.array([np.asarray(r, dtype=float) for r in nested])

    t0 = time.time()
    atoms = Atoms(symbols=list(row["species_at_sites"]), positions=matrix(row["cartesian_site_positions"]),
                  cell=matrix(row["lattice_vectors"]), pbc=True)
    n = len(atoms)
    target_forces = matrix(row["forces"])
    target_stress = gm.lemat_stress_to_ase(matrix(row["stress_tensor"]))
    record = {k: row[k] for k in ("immutable_id", "source", "force_bin", "stress_bin", "nsites",
                                   "max_abs_force", "max_abs_stress", "stratum_population")}
    counts = np.bincount(neighbor_list("i", atoms, NEIGHBOR_RADIUS), minlength=n)
    record["max_neighbors"] = int(counts.max())

    corrected = gm.GradientMatchedCalculator.from_reference(calc, atoms, target_forces, target_stress)
    match = corrected.match
    record["orb_dft_force_rms_diff"] = float(np.sqrt(np.mean(match.delta_forces ** 2)))
    record["orb_dft_stress_diff_kbar"] = float(np.abs(match.delta_stress).max() * gm.KBAR_PER_EV_PER_A3)

    coords = gm.GeneralizedCoordinates(atoms, symprec=args.symprec)
    record["n_operations"] = coords.n_operations
    evaluator = gm.corrected_evaluator(batch_eval, match)
    t1 = time.time()
    newton = gm.newton_estimate(coords, evaluator, step=args.fd_step, soft_curvature=args.soft_curvature)
    record.update(
        n_modes=newton.n_modes, n_position_modes=coords.n_position_modes,
        sym_gradient_norm=newton.gradient_norm,
        newton_mev=1e3 * newton.drop / n, newton_positions_mev=1e3 * newton.drop_positions / n,
        newton_cell_mev=1e3 * newton.drop_cell / n, newton_soft_share=newton.soft_share,
        n_soft=newton.n_soft, n_negative=newton.n_negative, min_curvature=newton.min_curvature)
    if newton.gradient_norm > 0:
        direction = coords.basis @ newton.reduced_gradient
        curvature = gm.curvature_along(coords, evaluator, direction, step=args.fd_step)
    else:
        curvature = np.nan
    record["sd_curvature"] = curvature
    record["sd_bound_mev"] = 1e3 * gm.steepest_descent_bound(newton.gradient_norm, curvature) / n
    record["newton_seconds"] = time.time() - t1

    t2 = time.time()
    relaxed = gm.relax_estimate(atoms, corrected, symprec=args.symprec, fmax=args.fmax,
                                steps=args.max_steps, trust_displacement=args.trust_displacement,
                                trust_strain=args.trust_strain)
    record.update(
        relax_mev=1e3 * relaxed.drop / n, relax_converged=relaxed.converged, relax_steps=relaxed.steps,
        max_displacement=relaxed.max_displacement, max_strain=relaxed.max_strain,
        trust_exceeded=relaxed.trust_exceeded, relax_aborted=relaxed.aborted,
        relax_seconds=time.time() - t2)
    for tol in (1e-3, 0.1):
        before, after = _spacegroup(atoms, tol), _spacegroup(relaxed.final, tol)
        record[f"sg_before_{tol:g}"] = before
        record[f"sg_after_{tol:g}"] = after
    record["seconds"] = time.time() - t0
    return record


def worker(rank: int, rows: list[dict], args) -> None:
    import torch
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s w{rank} %(message)s")
    if args.device == "cpu":
        torch.set_num_threads(args.threads_per_worker)
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    model, adapter = getattr(pretrained, args.model)(device=args.device, precision="float64", compile=False)
    model = model.eval()
    calc = ORBCalculator(model, adapter, device=args.device)
    batch_eval = _orb_batch_evaluator(model, adapter, args.device, args.max_atoms_per_batch)
    out_path = args.output_dir / "results" / f"worker_{rank}.jsonl"
    with open(out_path, "a") as sink:
        for i, row in enumerate(rows):
            try:
                record = process_row(row, calc, batch_eval, args)
            except Exception as exc:  # recorded, not fatal: one bad row must not stop a shard
                record = {"immutable_id": row["immutable_id"], "error": repr(exc),
                          "traceback": traceback.format_exc(limit=5)}
            record["model"] = args.model
            if args.device.startswith("cuda"):
                # The finite-difference batch of one large cell must not stay cached on a
                # shared card while this worker relaxes small ones.
                record["peak_gpu_mib"] = torch.cuda.max_memory_allocated() / 2**20
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            sink.write(json.dumps(record, default=float) + "\n")
            sink.flush()
            if i % 10 == 0:
                logger.info("%d/%d done", i + 1, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=cache_root() / "gradient_matched" / "pilot")
    parser.add_argument("--row-stats", type=Path, default=None,
                        help="Existing per-row audit with immutable_id, nsites, max_abs_force, "
                             "max_abs_stress, in raw-parquet row order.")
    parser.add_argument("--fmax1-ids", type=Path, default=None,
                        help="Parquet with an immutable_id column; defaults to reading the split CSVs.")
    parser.add_argument("--n-per-stratum", type=int, default=17)
    parser.add_argument("--max-sites", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N sample rows.")
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--model", default="orb_v3_conservative_inf_mpa")
    parser.add_argument("--symprec", type=float, default=1e-3)
    parser.add_argument("--fmax", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fd-step", type=float, default=1e-3)
    parser.add_argument("--soft-curvature", type=float, default=0.1)
    parser.add_argument("--trust-displacement", type=float, default=0.15)
    parser.add_argument("--trust-strain", type=float, default=0.02)
    parser.add_argument("--max-atoms-per-batch", type=int, default=20000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results").mkdir(exist_ok=True)
    sample_path = args.output_dir / "sample.parquet"
    sample = pd.read_parquet(sample_path) if sample_path.exists() else build_sample(args, sample_path)
    if args.sample_only:
        return
    if args.limit is not None:
        sample = sample.iloc[:args.limit]

    done = set()
    for shard in (args.output_dir / "results").glob("*.jsonl"):
        with open(shard) as handle:
            # Errored rows are retried on the next run; the merge below keeps the success.
            done.update(r["immutable_id"] for r in map(json.loads, filter(str.strip, handle))
                        if "error" not in r)
    todo = sample[~sample["immutable_id"].isin(done)]
    logger.info("%d rows in sample, %d done, %d to do", len(sample), len(done), len(todo))
    if len(todo):
        # Largest structures first across workers, so no single shard gets all of them.
        todo = todo.sort_values("nsites", ascending=False)
        rows = todo.to_dict("records")
        shards = [rows[r::args.workers] for r in range(args.workers)]
        existing = {int(p.stem.split("_")[1]) for p in (args.output_dir / "results").glob("worker_*.jsonl")}
        ranks = sorted(set(range(max(existing | {-1}) + 1 + args.workers)) - existing)[:args.workers]
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=worker, args=(rank, shard, args)) for rank, shard in zip(ranks, shards) if shard]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

    records = []
    for shard in sorted((args.output_dir / "results").glob("*.jsonl")):
        with open(shard) as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    results = pd.DataFrame(records)
    if "error" not in results:
        results["error"] = None
    # One row per id: the success if there is one, else the latest error.
    results = (results.assign(_ok=results["error"].isna())
               .sort_values("_ok", kind="stable").drop_duplicates("immutable_id", keep="last")
               .drop(columns="_ok"))
    results.to_parquet(args.output_dir / "results.parquet", index=False)
    logger.info("wrote %s: %d rows, %d errors", args.output_dir / "results.parquet", len(results),
                int(results.get("error", pd.Series(dtype=object)).notna().sum()))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
