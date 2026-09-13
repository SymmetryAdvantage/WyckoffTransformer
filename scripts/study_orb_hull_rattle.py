"""Study: Lower-energy polymorph discovery on the ORB convex hull via rattling.

Objectives:
1. Sample 1000 structures from LeMaterial/LeMat-Bulk-MLIP-Hull-All on the ORB hull.
2. Subject them to our pipeline:
   (a) relax with ORB while preserving symmetry
   (b) rattle and relax unconstrained
3. Check how this affects energy, Wyckoff gene, and structure.
Quantify for how many structures on the hull it is possible to find a lower-energy
polymorph simply by rattling.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import spglib
from ase import Atoms
from ase.filters import FrechetCellFilter as CellFilter
from ase.io import write as ase_write
from huggingface_hub import hf_hub_download
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# Ensure local project imports work
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_cryspr_reconstruction_study import (
    apply_runtime_patches,
    build_patched_orb_calculator,
)
from wyckoff_transformer.cryspr.relaxer import (
    RATTLE_ACCEPT_EV_PER_ATOM,
    RATTLE_STDEV,
    RATTLE_STRAIN_STDEV,
    _get_spacegroup_info,
    perturb,
    run_ase_relaxer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(processName)s) %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orb_hull_rattle_study")

CRYSTAL_SYSTEMS = [
    (1, 2, "triclinic"),
    (3, 15, "monoclinic"),
    (16, 74, "orthorhombic"),
    (75, 142, "tetragonal"),
    (143, 167, "trigonal"),
    (168, 194, "hexagonal"),
    (195, 230, "cubic"),
]


def get_crystal_system(spg_num: int) -> str:
    for low, high, name in CRYSTAL_SYSTEMS:
        if low <= spg_num <= high:
            return name
    return "unknown"


def get_wyckoff_positions_string(atoms: Atoms, symprec: float = 1e-3) -> str:
    """Return a sorted canonical string of Wyckoff positions (e.g. 'Fe: 4a, 8f; O: 12c')."""
    try:
        dataset = spglib.get_symmetry_dataset(
            (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers),
            symprec=symprec,
        )
        if dataset is None:
            return "unknown"
        wyckoffs = getattr(dataset, "wyckoffs", None)
        if wyckoffs is None:
            wyckoffs = dataset["wyckoffs"]
        symbols = atoms.get_chemical_symbols()
        
        # Group by element
        elem_wps: Dict[str, List[str]] = {}
        for sym, wp in zip(symbols, wyckoffs):
            elem_wps.setdefault(sym, []).append(wp)
            
        parts = []
        for sym in sorted(elem_wps.keys()):
            # Count multiplicities of each letter
            counts = Counter(elem_wps[sym])
            w_str = ", ".join(f"{cnt}{letter}" for letter, cnt in sorted(counts.items()))
            parts.append(f"{sym}: {w_str}")
        return "; ".join(parts)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Step 1: Sampling 1000 structures from LeMat-Bulk-MLIP-Hull-All
# ---------------------------------------------------------------------------
def sample_hull_structures(
    output_dir: Path,
    n_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_file = output_dir / f"sampled_{n_samples}_orb_hull_targets.parquet"
    if sample_file.exists():
        logger.info("Found existing sampled targets at %s", sample_file)
        return pd.read_parquet(sample_file)

    logger.info("Loading LeMaterial/LeMat-Bulk-MLIP-Hull-All shards to find structures on ORB hull...")
    p0 = hf_hub_download(
        "LeMaterial/LeMat-Bulk-MLIP-Hull-All",
        "data/train-00000-of-00002.parquet",
        repo_type="dataset",
    )
    p1 = hf_hub_download(
        "LeMaterial/LeMat-Bulk-MLIP-Hull-All",
        "data/train-00001-of-00002.parquet",
        repo_type="dataset",
    )

    dfs = []
    for p in [p0, p1]:
        table = pq.read_table(
            p,
            columns=[
                "immutable_id",
                "nsites",
                "orb_conserv_inf_hull",
                "orb_conserv_inf_energy",
                "true_energy",
                "dft_hull",
            ],
        )
        mask = pc.equal(table["orb_conserv_inf_hull"], 0.0)
        dfs.append(table.filter(mask).to_pandas())

    df_hull = pd.concat(dfs, ignore_index=True)
    logger.info(
        "Found %d candidate structures with orb_conserv_inf_hull == 0.0",
        len(df_hull),
    )

    sampled = df_hull.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    sampled_ids = set(sampled["immutable_id"])

    logger.info("Extracting geometries for %d sampled targets from LeMat-Bulk shards...", n_samples)
    t0 = time.time()
    found_rows = []
    for shard_idx in range(17):
        filename = f"compatible_pbe/train-{shard_idx:05d}-of-00017.parquet"
        shard_path = hf_hub_download(
            repo_id="LeMaterial/LeMat-Bulk",
            filename=filename,
            repo_type="dataset",
            local_files_only=True,
        )
        table = pq.read_table(
            shard_path,
            columns=[
                "immutable_id",
                "lattice_vectors",
                "cartesian_site_positions",
                "species_at_sites",
                "nsites",
                "chemical_formula_reduced",
            ],
            filters=[("immutable_id", "in", sampled_ids)],
        )
        df_shard = table.to_pandas()
        if len(df_shard) > 0:
            found_rows.append(df_shard)
            logger.info(
                "Shard %d: matched %d rows (total: %d/%d)",
                shard_idx,
                len(df_shard),
                sum(len(m) for m in found_rows),
                n_samples,
            )
        if sum(len(m) for m in found_rows) == n_samples:
            break

    df_geom = pd.concat(found_rows, ignore_index=True)
    # Merge metadata
    df_merged = df_geom.merge(
        sampled[[
            "immutable_id",
            "orb_conserv_inf_hull",
            "orb_conserv_inf_energy",
            "true_energy",
            "dft_hull",
        ]],
        on="immutable_id",
    )
    logger.info("Extraction completed in %.2fs. Saving sampled targets to %s", time.time() - t0, sample_file)
    df_merged.to_parquet(sample_file, index=False)
    df_merged.drop(columns=["lattice_vectors", "cartesian_site_positions", "species_at_sites"]).to_csv(
        output_dir / f"sampled_{n_samples}_orb_hull_targets.csv", index=False
    )
    return df_merged


# ---------------------------------------------------------------------------
# Step 2: Multi-GPU Worker Pipeline
# ---------------------------------------------------------------------------
def _worker_loop(
    worker_id: int,
    gpu_id: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    output_dir: Path,
    fmax: float = 0.05,
    steps_limit: int = 500,
):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    apply_runtime_patches()
    import torch
    torch.set_num_threads(1)
    
    try:
        calc = build_patched_orb_calculator(gpu_id)
    except Exception as exc:
        logger.exception("Worker %d on %s failed to initialize ORB: %s", worker_id, gpu_id, exc)
        result_queue.put(("FATAL", worker_id, gpu_id, str(exc)))
        return

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0)
    adaptor = AseAtomsAdaptor()

    while True:
        task = task_queue.get()
        if task is None:
            break

        idx, row_dict = task
        imm_id = row_dict["immutable_id"]
        formula = row_dict["chemical_formula_reduced"]
        species = list(row_dict["species_at_sites"])
        lat = np.array(row_dict["lattice_vectors"])
        coords = np.array(row_dict["cartesian_site_positions"])
        n_atoms = len(species)

        wdir = output_dir / "structures" / imm_id
        wdir.mkdir(parents=True, exist_ok=True)

        try:
            t_start = time.time()
            # 1. Initial Atoms
            atoms_init = Atoms(symbols=species, positions=coords, cell=lat, pbc=True)
            spg_init_sym, spg_init_num = _get_spacegroup_info(atoms_init, symprec=1e-3)
            vol_init = atoms_init.get_volume()

            # 2. Step (a): Symmetry-preserved relaxation
            # Warm-up (fix cell)
            atoms_warmup = run_ase_relaxer(
                atoms_in=atoms_init,
                calculator=calc,
                fix_symmetry=True,
                cell_filter=None,
                fmax=fmax,
                steps_limit=steps_limit,
                wdir=wdir,
                label="1_fix-cell",
                logfile=wdir / "1_fix-cell.log",
            )
            # Symmetric cell + positions
            atoms_sym = run_ase_relaxer(
                atoms_in=atoms_warmup,
                calculator=calc,
                fix_symmetry=True,
                cell_filter=CellFilter,
                fmax=fmax,
                steps_limit=steps_limit,
                wdir=wdir,
                label="2_sym_cell+pos",
                logfile=wdir / "2_sym_cell+pos.log",
            )
            e_sym_tot = float(atoms_sym.get_potential_energy())
            e_sym_per_atom = e_sym_tot / n_atoms
            spg_sym_sym, spg_sym_num = _get_spacegroup_info(atoms_sym, symprec=1e-3)
            vol_sym = atoms_sym.get_volume()
            wyckoff_sym_str = get_wyckoff_positions_string(atoms_sym, symprec=1e-3)

            # 3. Intermediate step: unconstrained relaxation before rattle (stage 3)
            atoms_nosym = run_ase_relaxer(
                atoms_in=atoms_sym,
                calculator=calc,
                fix_symmetry=False,
                cell_filter=CellFilter,
                fmax=fmax,
                steps_limit=steps_limit,
                wdir=wdir,
                label="3_no-sym_cell+pos",
                logfile=wdir / "3_no-sym_cell+pos.log",
            )
            e_nosym_tot = float(atoms_nosym.get_potential_energy())
            e_nosym_per_atom = e_nosym_tot / n_atoms

            # 4. Step (b): Rattle and unconstrained relaxation (stage 4)
            seed = int(42 + idx)
            perturbed = perturb(
                atoms_nosym,
                rattle_stdev=RATTLE_STDEV,
                strain_stdev=RATTLE_STRAIN_STDEV,
                seed=seed,
            )
            atoms_rattle = run_ase_relaxer(
                atoms_in=perturbed,
                calculator=calc,
                fix_symmetry=False,
                cell_filter=CellFilter,
                fmax=fmax,
                steps_limit=steps_limit,
                wdir=wdir,
                label="4_rattle_no-sym",
                logfile=wdir / "4_rattle_no-sym.log",
            )
            e_rattle_tot = float(atoms_rattle.get_potential_energy())
            e_rattle_per_atom = e_rattle_tot / n_atoms
            spg_rattle_sym, spg_rattle_num = _get_spacegroup_info(atoms_rattle, symprec=1e-3)
            vol_rattle = atoms_rattle.get_volume()
            wyckoff_rattle_str = get_wyckoff_positions_string(atoms_rattle, symprec=1e-3)

            # 5. Energy and structure comparisons
            delta_e_rattle = e_rattle_per_atom - e_sym_per_atom
            delta_e_nosym = e_nosym_per_atom - e_sym_per_atom
            
            # Pymatgen StructureMatcher comparison
            struct_sym = adaptor.get_structure(atoms_sym)
            struct_rattle = adaptor.get_structure(atoms_rattle)
            struct_matched = bool(matcher.fit(struct_sym, struct_rattle))
            rms_dist = matcher.get_rms_dist(struct_sym, struct_rattle)
            matcher_rmsd = float(rms_dist[0]) if rms_dist is not None else np.nan
            matcher_max_dist = float(rms_dist[1]) if rms_dist is not None else np.nan

            t_elapsed = time.time() - t_start

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            res = {
                "idx": idx,
                "immutable_id": imm_id,
                "chemical_formula_reduced": formula,
                "nsites": n_atoms,
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "elapsed_seconds": round(t_elapsed, 2),
                # Energy
                "e_initial_dft": row_dict.get("true_energy"),
                "e_initial_orb": row_dict.get("orb_conserv_inf_energy"),
                "e_sym_total": e_sym_tot,
                "e_sym_per_atom": e_sym_per_atom,
                "e_nosym_per_atom": e_nosym_per_atom,
                "e_rattle_total": e_rattle_tot,
                "e_rattle_per_atom": e_rattle_per_atom,
                "delta_e_rattle_ev": delta_e_rattle,
                "delta_e_rattle_mev": delta_e_rattle * 1000.0,
                "delta_e_nosym_mev": delta_e_nosym * 1000.0,
                "lower_energy_any": bool(delta_e_rattle < 0.0),
                "lower_energy_1mev": bool(delta_e_rattle < -0.001),
                "lower_energy_5mev": bool(delta_e_rattle < -0.005),
                "lower_energy_10mev": bool(delta_e_rattle < -0.010),
                "lower_energy_50mev": bool(delta_e_rattle < -0.050),
                "lower_energy_100mev": bool(delta_e_rattle < -0.100),
                # Symmetry & Wyckoff
                "spg_initial_num": spg_init_num,
                "spg_initial_symbol": spg_init_sym,
                "spg_sym_num": spg_sym_num,
                "spg_sym_symbol": spg_sym_sym,
                "spg_sym_crystal_system": get_crystal_system(spg_sym_num),
                "spg_rattle_num": spg_rattle_num,
                "spg_rattle_symbol": spg_rattle_sym,
                "spg_rattle_crystal_system": get_crystal_system(spg_rattle_num),
                "spg_changed": bool(spg_sym_num != spg_rattle_num),
                "crystal_system_changed": bool(get_crystal_system(spg_sym_num) != get_crystal_system(spg_rattle_num)),
                "wyckoff_sym": wyckoff_sym_str,
                "wyckoff_rattle": wyckoff_rattle_str,
                "wyckoff_changed": bool(wyckoff_sym_str != wyckoff_rattle_str),
                # Structure
                "vol_sym": vol_sym,
                "vol_rattle": vol_rattle,
                "delta_vol_pct": ((vol_rattle - vol_sym) / vol_sym) * 100.0,
                "struct_matched": struct_matched,
                "matcher_rmsd": matcher_rmsd,
                "matcher_max_dist": matcher_max_dist,
                "distinct_polymorph_1mev": bool((not struct_matched) and (delta_e_rattle < -0.001)),
                "status": "success",
            }
            result_queue.put(("RESULT", res))

        except Exception as exc:
            logger.exception("Error relaxing structure %s (idx=%d): %s", imm_id, idx, exc)
            result_queue.put((
                "ERROR",
                {
                    "idx": idx,
                    "immutable_id": imm_id,
                    "chemical_formula_reduced": formula,
                    "nsites": n_atoms,
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "status": f"error: {exc}",
                },
            ))


def run_orb_hull_rattle_study(
    df_targets: pd.DataFrame,
    output_dir: Path,
    devices: List[str] = ("cuda:0", "cuda:0", "cuda:1", "cuda:1", "cuda:2"),
    fmax: float = 0.05,
    steps_limit: int = 500,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "results.csv"
    results_parquet = output_dir / "results.parquet"
    checkpoint_csv = output_dir / "results_in_progress.csv"

    completed_ids = set()
    existing_results = []
    if results_parquet.exists():
        df_ex = pd.read_parquet(results_parquet)
        completed_ids = set(df_ex["immutable_id"])
        existing_results = df_ex.to_dict(orient="records")
        logger.info("Found %d completed results in %s", len(completed_ids), results_parquet)
    elif checkpoint_csv.exists():
        df_ex = pd.read_csv(checkpoint_csv)
        completed_ids = set(df_ex["immutable_id"])
        existing_results = df_ex.to_dict(orient="records")
        logger.info("Resuming from checkpoint with %d results in %s", len(completed_ids), checkpoint_csv)

    remaining_tasks = []
    for idx, row in df_targets.iterrows():
        if row["immutable_id"] not in completed_ids:
            row_dict = {
                "immutable_id": row["immutable_id"],
                "chemical_formula_reduced": row["chemical_formula_reduced"],
                "species_at_sites": row["species_at_sites"],
                "lattice_vectors": row["lattice_vectors"].tolist() if hasattr(row["lattice_vectors"], "tolist") else row["lattice_vectors"],
                "cartesian_site_positions": row["cartesian_site_positions"].tolist() if hasattr(row["cartesian_site_positions"], "tolist") else row["cartesian_site_positions"],
                "true_energy": row["true_energy"],
                "orb_conserv_inf_energy": row["orb_conserv_inf_energy"],
            }
            remaining_tasks.append((idx, row_dict))

    logger.info(
        "Starting relaxation study: %d total targets, %d already done, %d remaining across %d workers (%s)",
        len(df_targets),
        len(completed_ids),
        len(remaining_tasks),
        len(devices),
        devices,
    )

    if not remaining_tasks:
        logger.info("All tasks are already completed!")
        return pd.read_parquet(results_parquet) if results_parquet.exists() else pd.read_csv(checkpoint_csv)

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    for task in remaining_tasks:
        task_queue.put(task)
    for _ in devices:
        task_queue.put(None)

    processes = []
    for w_id, dev in enumerate(devices):
        p = mp.Process(
            target=_worker_loop,
            args=(w_id, dev, task_queue, result_queue, output_dir, fmax, steps_limit),
            name=f"Worker-{w_id}-{dev}",
        )
        p.start()
        processes.append(p)

    all_results = list(existing_results)
    n_total_remaining = len(remaining_tasks)
    n_processed = 0
    t0 = time.time()
    lower_1mev_count = sum(1 for r in all_results if r.get("lower_energy_1mev", False))

    while n_processed < n_total_remaining:
        msg_type, payload = result_queue.get()
        if msg_type == "FATAL":
            logger.error("A worker encountered a fatal error: %s. Terminating all workers.", payload)
            for p in processes:
                p.terminate()
            raise RuntimeError(f"Fatal worker failure: {payload}")

        all_results.append(payload)
        n_processed += 1
        if payload.get("lower_energy_1mev", False):
            lower_1mev_count += 1

        if n_processed % 10 == 0 or n_processed == n_total_remaining:
            elapsed = time.time() - t0
            rate = n_processed / elapsed if elapsed > 0 else 0
            eta_mins = (n_total_remaining - n_processed) / rate / 60.0 if rate > 0 else 0
            logger.info(
                "Progress: %d/%d (%.1f%%) | %.2f struct/s (%.1f/min) | ETA: %.1f min | Lower-energy (>=1 meV): %d (%.1f%%)",
                n_processed,
                n_total_remaining,
                100.0 * n_processed / n_total_remaining,
                rate,
                rate * 60.0,
                eta_mins,
                lower_1mev_count,
                100.0 * lower_1mev_count / len(all_results),
            )
            # Periodic checkpointing
            df_checkpoint = pd.DataFrame(all_results)
            df_checkpoint.to_csv(checkpoint_csv, index=False)

    for p in processes:
        p.join()

    logger.info("All workers finished! Saving final results...")
    df_final = pd.DataFrame(all_results)
    df_final.to_csv(results_csv, index=False)
    df_final.to_parquet(results_parquet, index=False)
    logger.info("Results saved to %s and %s", results_csv, results_parquet)
    return df_final


# ---------------------------------------------------------------------------
# Step 3: Analysis & Statistical Report Generation
# ---------------------------------------------------------------------------
def compute_and_save_analysis(output_dir: Path):
    results_parquet = output_dir / "results.parquet"
    if not results_parquet.exists():
        results_parquet = output_dir / "results.csv"
        df = pd.read_csv(results_parquet)
    else:
        df = pd.read_parquet(results_parquet)

    valid = df[df["status"] == "success"].copy()
    n_total = len(df)
    n_valid = len(valid)

    logger.info("Computing analysis over %d valid structures...", n_valid)

    # 1. Energy Analysis
    dE = valid["delta_e_rattle_mev"]
    dE_ev = valid["delta_e_rattle_ev"]
    
    n_lower_any = int((valid["delta_e_rattle_ev"] < 0.0).sum())
    n_lower_1mev = int((valid["delta_e_rattle_ev"] <= -0.001).sum())
    n_lower_5mev = int((valid["delta_e_rattle_ev"] <= -0.005).sum())
    n_lower_10mev = int((valid["delta_e_rattle_ev"] <= -0.010).sum())
    n_lower_50mev = int((valid["delta_e_rattle_ev"] <= -0.050).sum())
    n_lower_100mev = int((valid["delta_e_rattle_ev"] <= -0.100).sum())

    n_nosym_lower_1mev = int((valid["delta_e_nosym_mev"] <= -1.0).sum())

    drops = valid[valid["delta_e_rattle_ev"] <= -0.001]
    
    # 2. Symmetry Analysis
    spg_changed = valid[valid["spg_changed"]]
    cs_changed = valid[valid["crystal_system_changed"]]
    wyckoff_changed = valid[valid["wyckoff_changed"]]

    # Symmetry changes among lower energy drops
    drops_spg_changed = drops[drops["spg_changed"]]
    drops_wyckoff_changed = drops[drops["wyckoff_changed"]]

    # By crystal system
    cs_breakdown = {}
    for cs in ["cubic", "hexagonal", "trigonal", "tetragonal", "orthorhombic", "monoclinic", "triclinic"]:
        sub = valid[valid["spg_sym_crystal_system"] == cs]
        if len(sub) > 0:
            sub_drops = sub[sub["delta_e_rattle_ev"] <= -0.001]
            cs_breakdown[cs] = {
                "total": len(sub),
                "share_pct": round(100.0 * len(sub) / n_valid, 2),
                "lower_1mev_count": len(sub_drops),
                "lower_1mev_pct": round(100.0 * len(sub_drops) / len(sub), 2),
                "median_drop_mev": round(float(sub_drops["delta_e_rattle_mev"].median()), 2) if len(sub_drops) > 0 else 0.0,
            }

    # 3. Structural Divergence
    struct_matched_count = int(valid["struct_matched"].sum())
    struct_unmatched_count = n_valid - struct_matched_count
    distinct_polymorphs = int(valid["distinct_polymorph_1mev"].sum())

    summary = {
        "cohort": {
            "total_sampled": n_total,
            "valid_evaluated": n_valid,
            "failed": n_total - n_valid,
        },
        "energy": {
            "lower_energy_any_count": n_lower_any,
            "lower_energy_any_pct": round(100.0 * n_lower_any / n_valid, 2),
            "lower_energy_1mev_count": n_lower_1mev,
            "lower_energy_1mev_pct": round(100.0 * n_lower_1mev / n_valid, 2),
            "lower_energy_5mev_count": n_lower_5mev,
            "lower_energy_5mev_pct": round(100.0 * n_lower_5mev / n_valid, 2),
            "lower_energy_10mev_count": n_lower_10mev,
            "lower_energy_10mev_pct": round(100.0 * n_lower_10mev / n_valid, 2),
            "lower_energy_50mev_count": n_lower_50mev,
            "lower_energy_50mev_pct": round(100.0 * n_lower_50mev / n_valid, 2),
            "lower_energy_100mev_count": n_lower_100mev,
            "lower_energy_100mev_pct": round(100.0 * n_lower_100mev / n_valid, 2),
            "unconstrained_without_rattle_1mev_count": n_nosym_lower_1mev,
            "unconstrained_without_rattle_1mev_pct": round(100.0 * n_nosym_lower_1mev / n_valid, 2),
            "delta_e_mev_all": {
                "mean": round(float(dE.mean()), 3),
                "std": round(float(dE.std()), 3),
                "min": round(float(dE.min()), 3),
                "p10": round(float(dE.quantile(0.10)), 3),
                "p25": round(float(dE.quantile(0.25)), 3),
                "median": round(float(dE.median()), 3),
                "p75": round(float(dE.quantile(0.75)), 3),
                "max": round(float(dE.max()), 3),
            },
            "delta_e_mev_drops_only": {
                "count": len(drops),
                "mean": round(float(drops["delta_e_rattle_mev"].mean()), 3) if len(drops) > 0 else 0.0,
                "median": round(float(drops["delta_e_rattle_mev"].median()), 3) if len(drops) > 0 else 0.0,
                "min": round(float(drops["delta_e_rattle_mev"].min()), 3) if len(drops) > 0 else 0.0,
                "max": round(float(drops["delta_e_rattle_mev"].max()), 3) if len(drops) > 0 else 0.0,
            },
        },
        "symmetry_and_wyckoff": {
            "spg_changed_count": len(spg_changed),
            "spg_changed_pct": round(100.0 * len(spg_changed) / n_valid, 2),
            "crystal_system_changed_count": len(cs_changed),
            "crystal_system_changed_pct": round(100.0 * len(cs_changed) / n_valid, 2),
            "wyckoff_changed_count": len(wyckoff_changed),
            "wyckoff_changed_pct": round(100.0 * len(wyckoff_changed) / n_valid, 2),
            "drops_spg_changed_count": len(drops_spg_changed),
            "drops_spg_changed_pct": round(100.0 * len(drops_spg_changed) / len(drops), 2) if len(drops) > 0 else 0.0,
            "drops_wyckoff_changed_count": len(drops_wyckoff_changed),
            "drops_wyckoff_changed_pct": round(100.0 * len(drops_wyckoff_changed) / len(drops), 2) if len(drops) > 0 else 0.0,
            "crystal_systems": cs_breakdown,
        },
        "structure": {
            "struct_matched_count": struct_matched_count,
            "struct_matched_pct": round(100.0 * struct_matched_count / n_valid, 2),
            "distinct_polymorph_1mev_count": distinct_polymorphs,
            "distinct_polymorph_1mev_pct": round(100.0 * distinct_polymorphs / n_valid, 2),
            "median_volume_change_pct": round(float(valid["delta_vol_pct"].abs().median()), 3),
            "median_volume_change_pct_drops": round(float(drops["delta_vol_pct"].abs().median()), 3) if len(drops) > 0 else 0.0,
        },
    }

    summary_json = output_dir / "summary_metrics.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary metrics written to %s", summary_json)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Study ORB hull rattling stability.")
    parser.add_argument("--output-dir", type=Path, default=Path("generated/studies/orb_hull_rattle"))
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=str, default="cuda:0,cuda:0,cuda:1,cuda:1,cuda:2")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps-limit", type=int, default=500)
    parser.add_argument("--skip-relax", action="store_true", help="Skip relaxation and re-run analysis only")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    if not args.skip_relax:
        df_targets = sample_hull_structures(
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            seed=args.seed,
        )
        run_orb_hull_rattle_study(
            df_targets=df_targets,
            output_dir=args.output_dir,
            devices=devices,
            fmax=args.fmax,
            steps_limit=args.steps_limit,
        )

    compute_and_save_analysis(args.output_dir)
    logger.info("Study completed successfully!")


if __name__ == "__main__":
    main()
