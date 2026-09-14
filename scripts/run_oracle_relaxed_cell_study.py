#!/usr/bin/env python3
"""Run the Oracle Initial-Lattice CRySPR Reconstruction Study across all available GPUs.

Adjustments:
1. Drop factor=1.3: use standard atomic Tol_matrix with factor=1.0 (with fallback to 0.9).
2. Don't fix the cell during relaxation: initialize PyXtal with the ground truth conventional lattice,
   then perform standard 4-stage CRySPR variable-cell relaxation (FrechetCellFilter).
3. 5 trials per structure for 400 high-DoF structures (DoF >= 6).
4. Apples-to-apples comparison against 5 trials of the lattice-free run.
"""
import argparse
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import signal
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))
if str(_repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(_repo_root / "scripts"))

import numpy as np
import pandas as pd
from ase import Atoms
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import write as ase_write
from ase.optimize import BFGS
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice as PMGLattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from pyxtal.lattice import Lattice as PxtLattice
from pyxtal.tolerance import Tol_matrix

from run_cryspr_reconstruction_study import (
    apply_runtime_patches,
    build_patched_orb_calculator,
    get_crystal_system_and_lattice_dof,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(processName)s) %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("oracle_relaxed_study")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("SPGLIB_OLD_ERROR_HANDLING", "0")


def _json_default(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _deterministic_seed(struct_idx: int, trial_idx: int) -> int:
    h = hashlib.sha256(f"oracle_relaxed_{struct_idx}_{trial_idx}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


class PyXtalTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise PyXtalTimeout("PyXtal candidate generation timed out")


def _generate_pyxtal_candidate(gene: Dict, pxt_lat: PxtLattice, seed: int, timeout: int = 10) -> Optional[Atoms]:
    """Generate PyXtal crystal using factor=1.0 with adaptive fallback."""
    np.random.seed(seed)
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    try:
        for factor in [1.0, 0.9]:
            tol_mat = Tol_matrix(prototype="atomic", factor=factor)
            try:
                cand = pyxtal()
                cand.from_random(
                    dim=3,
                    group=gene["group"],
                    species=gene["species"],
                    numIons=gene["numIons"],
                    sites=gene["sites"],
                    lattice=pxt_lat,
                    tm=tol_mat,
                    max_count=10,
                )
                if cand.valid:
                    return cand.to_ase()
            except Exception:
                continue
        return None
    except PyXtalTimeout:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _reconstruct_single_trial(
    struct_info: Dict,
    trial_idx: int,
    trial_dir: Path,
    calc,
    fmax: float = 0.02,
    steps_limit: int = 500,
) -> Optional[Dict]:
    """Execute 4-stage variable-cell reconstruction initialized from ground-truth lattice."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    struct_idx = struct_info["struct_idx"]
    seed = _deterministic_seed(struct_idx, trial_idx)

    gene = struct_info["gene"]
    pxt_lat = struct_info["pxt_lattice"]

    atoms = _generate_pyxtal_candidate(gene, pxt_lat, seed, timeout=10)
    if atoms is None:
        return None

    formula = atoms.get_chemical_formula(mode="metal")
    ase_write(str(trial_dir / f"{formula}_0_initial.cif"), atoms, format="cif")
    n_atoms = len(atoms)

    try:
        import torch

        # Stage 1: Fixed cell warm-up, symmetry constrained
        atoms_s1 = atoms.copy()
        atoms_s1.calc = calc
        atoms_s1.set_constraint([FixSymmetry(atoms_s1, symprec=1e-3)])
        opt1 = BFGS(atoms_s1, logfile=str(trial_dir / f"{formula}_1_fix-cell.log"))
        opt1.run(fmax=fmax, steps=steps_limit)
        ase_write(str(trial_dir / f"{formula}_1_fix-cell.cif"), atoms_s1, format="cif")

        # Stage 2: Variable cell + pos, symmetry constrained
        atoms_s2 = atoms_s1.copy()
        atoms_s2.calc = calc
        atoms_s2.set_constraint([FixSymmetry(atoms_s2, symprec=1e-3)])
        filt2 = FrechetCellFilter(atoms_s2)
        opt2 = BFGS(filt2, logfile=str(trial_dir / f"{formula}_2_sym_cell+pos.log"))
        opt2.run(fmax=fmax, steps=steps_limit)
        ase_write(str(trial_dir / f"{formula}_2_sym_cell+pos.cif"), atoms_s2, format="cif")

        # Stage 3: Variable cell + pos, symmetry RELEASED
        atoms_s3 = atoms_s2.copy()
        atoms_s3.calc = calc
        atoms_s3.set_constraint([])  # Release symmetry
        filt3 = FrechetCellFilter(atoms_s3)
        opt3 = BFGS(filt3, logfile=str(trial_dir / f"{formula}_3_no-sym_cell+pos.log"))
        opt3.run(fmax=fmax, steps=steps_limit)
        e_s3 = float(atoms_s3.get_potential_energy())
        cif_s3_path = trial_dir / f"{formula}_3_no-sym_cell+pos.cif"
        ase_write(str(cif_s3_path), atoms_s3, format="cif")

        # Stage 4: Atomic rattle + symmetrized strain + re-relax
        rng = np.random.default_rng(seed)
        M = rng.normal(0, 0.01, size=(3, 3))
        strain = 0.5 * (M + M.T)
        atoms_s4 = atoms_s3.copy()
        atoms_s4.calc = calc
        atoms_s4.set_constraint([])
        new_cell = atoms_s4.cell @ (np.eye(3) + strain)
        atoms_s4.set_cell(new_cell, scale_atoms=True)
        atoms_s4.rattle(stdev=0.05, seed=int(rng.integers(1, 2**31 - 1)))

        filt4 = FrechetCellFilter(atoms_s4)
        opt4 = BFGS(filt4, logfile=str(trial_dir / f"{formula}_4_rattle_no-sym.log"))
        opt4.run(fmax=fmax, steps=steps_limit)
        e_s4 = float(atoms_s4.get_potential_energy())
        cif_s4_path = trial_dir / f"{formula}_4_rattle_no-sym.cif"
        ase_write(str(cif_s4_path), atoms_s4, format="cif")

        # Acceptance rule
        de_per_atom_s4 = (e_s4 - e_s3) / n_atoms
        accepted_s4 = de_per_atom_s4 < -0.001

        atoms_s3.calc = None
        atoms_s4.calc = None

        if accepted_s4:
            kept_stage = "s4"
            kept_e = e_s4
            kept_cif = cif_s4_path
            kept_atoms = atoms_s4.copy()
        else:
            kept_stage = "s3"
            kept_e = e_s3
            kept_cif = cif_s3_path
            kept_atoms = atoms_s3.copy()
        kept_atoms.calc = None

        return {
            "trial_idx": trial_idx,
            "e_s3": e_s3,
            "e_s3_per_atom": e_s3 / n_atoms,
            "cif_s3_path": str(cif_s3_path),
            "e_s4": e_s4,
            "e_s4_per_atom": e_s4 / n_atoms,
            "cif_s4_path": str(cif_s4_path),
            "accepted_s4": accepted_s4,
            "kept_stage": kept_stage,
            "kept_e": kept_e,
            "kept_e_per_atom": kept_e / n_atoms,
            "kept_cif_path": str(kept_cif),
            "atoms_s3": atoms_s3,
            "atoms_s4": atoms_s4,
            "kept_atoms": kept_atoms,
        }
    except torch.cuda.OutOfMemoryError:
        logger.warning("Struct %d trial %d: CUDA OutOfMemoryError, clearing cache", struct_idx, trial_idx)
        import torch
        torch.cuda.empty_cache()
        return None
    except Exception as exc:
        logger.warning("Struct %d trial %d: relaxation error (%s)", struct_idx, trial_idx, exc)
        return None


def _oracle_worker(
    gpu_id: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    output_dir: Path,
    n_trials: int = 5,
):
    """Worker handling Oracle reconstruction across assigned structure tasks on one GPU."""
    try:
        apply_runtime_patches()
        calc = build_patched_orb_calculator(gpu_id)
        import torch

        while True:
            task = task_queue.get()
            if task is None:
                break
            struct_info = task
            s_idx = struct_info["struct_idx"]
            imm_id = struct_info["immutable_id"]
            gene = struct_info["gene"]
            struct_dir = output_dir / "cryspr" / str(s_idx)
            struct_dir.mkdir(parents=True, exist_ok=True)
            n_atoms = sum(gene.get("numIons", []))

            logger.info(
                "[%s] Worker starting Struct %d (%s, sg=%d, sites=%d, atoms=%d, dof_pos=%d, dof_tot=%d)",
                gpu_id,
                s_idx,
                imm_id,
                gene.get("group", 0),
                len(gene.get("sites", [])),
                n_atoms,
                struct_info.get("dof_pos", 0),
                struct_info.get("dof_total", 0),
            )

            trial_results = []
            n_generated = 0

            for t_idx in range(n_trials):
                t_dir = struct_dir / f"trial-{t_idx}"
                try:
                    res = _reconstruct_single_trial(
                        struct_info=struct_info,
                        trial_idx=t_idx,
                        trial_dir=t_dir,
                        calc=calc,
                        fmax=0.02,
                        steps_limit=500,
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning("[%s] Struct %d trial %d hit CUDA OOM", gpu_id, s_idx, t_idx)
                    torch.cuda.empty_cache()
                    res = None
                except Exception as exc:
                    logger.warning("[%s] Struct %d trial %d error: %s", gpu_id, s_idx, t_idx, exc)
                    res = None

                if res is not None:
                    n_generated += 1
                    if abs(res["kept_e_per_atom"]) <= 50.0:
                        trial_results.append(res)
                torch.cuda.empty_cache()

            logger.info(
                "[%s] Finished Struct %d (%s): %d/%d valid trials (%d generated)",
                gpu_id,
                s_idx,
                imm_id,
                len(trial_results),
                n_trials,
                n_generated,
            )
            result_queue.put((s_idx, imm_id, n_generated, trial_results))
    except Exception as exc:
        logger.exception("Error in oracle worker on %s: %s", gpu_id, exc)
        result_queue.put(("ERROR", str(exc)))


def prepare_cohort(n_samples: int = 400, seed: int = 42) -> pd.DataFrame:
    """Load or sample the cohort and extract standard conventional lattices."""
    cohort_path = Path("generated/cryspr_oracle_fixed_lattice_study/data/sampled_400_targets.parquet")
    if cohort_path.exists():
        df_cohort = pd.read_parquet(cohort_path)
    else:
        df_relaxed = pd.read_parquet("generated/cryspr_reconstruction_study/data/relaxed_targets.parquet")
        with gzip.open("generated/cryspr_reconstruction_study/data/unique_wyckoff_genes.json.gz", "rt") as f:
            genes = json.load(f)
        target_to_gene = {tid: g for g in genes for tid in g["target_ids"]}
        df_relaxed["gene_id"] = df_relaxed["immutable_id"].map(lambda tid: target_to_gene[tid]["gene_id"])
        df_relaxed["dof_total"] = df_relaxed["immutable_id"].map(lambda tid: target_to_gene[tid]["dof_total"])
        df_relaxed["dof_pos"] = df_relaxed["immutable_id"].map(lambda tid: target_to_gene[tid]["dof_positional"])
        df_relaxed["crystal_system"] = df_relaxed["immutable_id"].map(lambda tid: target_to_gene[tid]["crystal_system"])
        df_relaxed["spacegroup"] = df_relaxed["immutable_id"].map(lambda tid: target_to_gene[tid]["spacegroup"])
        df_relaxed["n_wyckoff_sites"] = df_relaxed["immutable_id"].map(lambda tid: target_to_gene[tid]["n_wyckoff_sites"])
        df_relaxed["wyckoff_gene"] = df_relaxed["immutable_id"].map(lambda tid: target_to_gene[tid]["gene"])
        df_cohort = df_relaxed[df_relaxed["dof_total"] >= 6].sample(n=n_samples, random_state=seed).reset_index(drop=True)
        df_cohort["struct_idx"] = df_cohort.index
    return df_cohort


def run_oracle_experiment(
    df_cohort: pd.DataFrame,
    output_dir: Path,
    devices: List[str],
    n_trials: int = 5,
) -> Dict[str, Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "oracle_reconstruction_results.pkl"

    results_by_id = {}
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            try:
                results_by_id = pickle.load(f)
                logger.info("Loaded %d completed structures from cache", len(results_by_id))
            except Exception as exc:
                logger.warning("Failed reading cache: %s", exc)

    tasks = []
    for _, row in df_cohort.iterrows():
        imm_id = row["immutable_id"]
        if imm_id in results_by_id:
            continue

        tgt_struct = Structure.from_file(row["cif_path"])
        try:
            pxt_seed = pyxtal()
            pxt_seed.from_seed(tgt_struct, tol=0.1)
            pxt_lat = pxt_seed.lattice
        except Exception:
            sga = SpacegroupAnalyzer(tgt_struct, symprec=0.1)
            std_struct = sga.get_conventional_standard_structure()
            pxt_lat = PxtLattice.from_para(
                std_struct.lattice.a,
                std_struct.lattice.b,
                std_struct.lattice.c,
                std_struct.lattice.alpha,
                std_struct.lattice.beta,
                std_struct.lattice.gamma,
                ltype=row["crystal_system"],
                force_symmetry=True,
            )

        tasks.append(
            {
                "struct_idx": int(row["struct_idx"]),
                "immutable_id": imm_id,
                "cif_path": row["cif_path"],
                "crystal_system": row["crystal_system"],
                "dof_total": int(row["dof_total"]),
                "dof_pos": int(row["dof_pos"]),
                "gene": row["wyckoff_gene"],
                "pxt_lattice": pxt_lat,
            }
        )

    if not tasks:
        logger.info("All %d structures already completed!", len(df_cohort))
        return results_by_id

    logger.info(
        "Launching Oracle Relaxed-Cell Experiment: %d structures to run (%d already done), %d trials/structure on %s...",
        len(tasks),
        len(results_by_id),
        n_trials,
        devices,
    )

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    for t in tasks:
        task_queue.put(t)
    for _ in devices:
        task_queue.put(None)

    processes = []
    for dev in devices:
        p = mp.Process(
            target=_oracle_worker,
            args=(dev, task_queue, result_queue, output_dir, n_trials),
            name=f"OracleWorker-{dev}",
        )
        p.start()
        processes.append(p)

    n_done = 0
    t0 = time.time()

    while n_done < len(tasks):
        item = result_queue.get()
        if item[0] == "ERROR":
            for p in processes:
                p.terminate()
            raise RuntimeError(f"Oracle worker failed: {item[1]}")
        s_idx, imm_id, n_gen, trials = item
        results_by_id[imm_id] = {
            "struct_idx": s_idx,
            "immutable_id": imm_id,
            "n_trials_generated": n_gen,
            "trials": trials,
        }
        n_done += 1

        with open(cache_path, "wb") as f:
            pickle.dump(results_by_id, f)

        elapsed = time.time() - t0
        rate = n_done / max(elapsed, 1e-5)
        rem = (len(tasks) - n_done) / max(rate, 1e-5)
        logger.info(
            "Progress: %d/%d new structures completed (total: %d/%d, %.3f struct/s, elapsed: %.1fs, eta: %.1fs)",
            n_done,
            len(tasks),
            len(results_by_id),
            len(df_cohort),
            rate,
            elapsed,
            rem,
        )

    for p in processes:
        p.join()

    with open(cache_path, "wb") as f:
        pickle.dump(results_by_id, f)
    logger.info("Oracle reconstruction completed and saved to %s", cache_path)
    return results_by_id


def score_and_compare(
    df_cohort: pd.DataFrame,
    oracle_results: Dict[str, Dict],
    output_dir: Path,
):
    logger.info("Scoring and computing comparison metrics...")
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    baseline_cache_path = Path("generated/cryspr_reconstruction_study/data/reconstruction_results.pkl")
    with open(baseline_cache_path, "rb") as f:
        baseline_raw = pickle.load(f)

    # Also load the previous fixed-cell oracle results if available for 3-way comparison
    prev_fixed_cache = Path("generated/cryspr_oracle_fixed_lattice_study/data/oracle_reconstruction_results.pkl")
    fixed_oracle_raw = {}
    if prev_fixed_cache.exists():
        with open(prev_fixed_cache, "rb") as f:
            fixed_oracle_raw = pickle.load(f)

    matcher_default = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True)
    matcher_loose = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True, scale=True)

    rows = []
    for _, row in df_cohort.iterrows():
        imm_id = row["immutable_id"]
        gid = row["gene_id"]
        tgt_struct = Structure.from_file(row["cif_path"])
        tgt_e = row["orb_e_target_per_atom"]

        ora_info = oracle_results.get(imm_id, {"n_trials_generated": 0, "trials": []})
        ora_trials = ora_info.get("trials", [])
        ora_n_gen = ora_info.get("n_trials_generated", 0)

        base_gene_trials = baseline_raw.get(gid, {}).get("trials", [])
        base5_trials = base_gene_trials[:5]
        base10_trials = base_gene_trials[:10]

        fixed_ora_trials = fixed_oracle_raw.get(imm_id, {}).get("trials", [])

        def evaluate_trial_set(trials, n_gen):
            if not trials:
                return {
                    "recovered": False,
                    "ceiling": False,
                    "loose_rec": False,
                    "loose_ceil": False,
                    "e_kept": np.nan,
                    "de_kept": np.nan,
                    "verdict": "generation_failed",
                    "n_gen": n_gen,
                    "n_rel": 0,
                }
            trials_sorted = sorted(trials, key=lambda t: t["kept_e_per_atom"])
            best_t = trials_sorted[0]
            e_kept = best_t["kept_e_per_atom"]
            de_kept = e_kept - tgt_e

            any_match = False
            any_loose = False
            for t in trials:
                s_pmg = Structure(
                    PMGLattice(t["kept_atoms"].cell.array),
                    t["kept_atoms"].get_chemical_symbols(),
                    t["kept_atoms"].positions,
                    coords_are_cartesian=True,
                )
                if matcher_default.fit(s_pmg, tgt_struct):
                    any_match = True
                if matcher_loose.fit(s_pmg, tgt_struct):
                    any_loose = True

            best_pmg = Structure(
                PMGLattice(best_t["kept_atoms"].cell.array),
                best_t["kept_atoms"].get_chemical_symbols(),
                best_t["kept_atoms"].positions,
                coords_are_cartesian=True,
            )
            m_best = matcher_default.fit(best_pmg, tgt_struct)
            m_loose_best = matcher_loose.fit(best_pmg, tgt_struct)

            if m_best:
                verdict = "recovered"
            elif any_match:
                verdict = "sampled_not_selected"
            elif de_kept < -0.001:
                verdict = "lower_energy_alternative"
            else:
                verdict = "missed"

            return {
                "recovered": m_best,
                "ceiling": any_match,
                "loose_rec": m_loose_best,
                "loose_ceil": any_loose,
                "e_kept": e_kept,
                "de_kept": de_kept,
                "verdict": verdict,
                "n_gen": n_gen,
                "n_rel": len(trials),
            }

        eval_ora = evaluate_trial_set(ora_trials, ora_n_gen)
        eval_base5 = evaluate_trial_set(base5_trials, min(5, baseline_raw.get(gid, {}).get("n_trials_generated", 0)))
        eval_base10 = evaluate_trial_set(base10_trials, baseline_raw.get(gid, {}).get("n_trials_generated", 0))
        eval_fixed = evaluate_trial_set(fixed_ora_trials, fixed_oracle_raw.get(imm_id, {}).get("n_trials_generated", 0))

        rows.append(
            {
                "struct_idx": row["struct_idx"],
                "immutable_id": imm_id,
                "gene_id": gid,
                "formula": row["chemical_formula_reduced"],
                "spacegroup": row["spacegroup"],
                "crystal_system": row["crystal_system"],
                "nsites": row["nsites"],
                "n_wyckoff_sites": row["n_wyckoff_sites"],
                "dof_total": row["dof_total"],
                "dof_pos": row["dof_pos"],
                "e_target": tgt_e,
                # Baseline 5
                "base5_recovered": eval_base5["recovered"],
                "base5_ceiling": eval_base5["ceiling"],
                "base5_loose_rec": eval_base5["loose_rec"],
                "base5_loose_ceil": eval_base5["loose_ceil"],
                "base5_verdict": eval_base5["verdict"],
                "base5_e_kept": eval_base5["e_kept"],
                "base5_de_kept": eval_base5["de_kept"],
                # Oracle Initial Cell (Adjusted)
                "oracle_recovered": eval_ora["recovered"],
                "oracle_ceiling": eval_ora["ceiling"],
                "oracle_loose_rec": eval_ora["loose_rec"],
                "oracle_loose_ceil": eval_ora["loose_ceil"],
                "oracle_verdict": eval_ora["verdict"],
                "oracle_e_kept": eval_ora["e_kept"],
                "oracle_de_kept": eval_ora["de_kept"],
                # Fixed cell oracle (from previous run)
                "fixed_recovered": eval_fixed["recovered"],
                "fixed_verdict": eval_fixed["verdict"],
                # Baseline 10 (reference)
                "base10_recovered": eval_base10["recovered"],
                "base10_ceiling": eval_base10["ceiling"],
                "base10_verdict": eval_base10["verdict"],
            }
        )

    df_results = pd.DataFrame(rows)
    df_results.to_csv(tables_dir / "results_per_structure.csv", index=False)
    logger.info("Saved per-structure results to %s", tables_dir / "results_per_structure.csv")

    n_tot = len(df_results)
    summary = {
        "cohort_size": n_tot,
        "base5_recovery_rate": float(df_results["base5_recovered"].mean()),
        "base5_sampling_ceiling": float(df_results["base5_ceiling"].mean()),
        "base5_loose_recovery": float(df_results["base5_loose_rec"].mean()),
        "base5_verdicts": df_results["base5_verdict"].value_counts().to_dict(),
        "oracle_recovery_rate": float(df_results["oracle_recovered"].mean()),
        "oracle_sampling_ceiling": float(df_results["oracle_ceiling"].mean()),
        "oracle_loose_recovery": float(df_results["oracle_loose_rec"].mean()),
        "oracle_verdicts": df_results["oracle_verdict"].value_counts().to_dict(),
        "fixed_recovery_rate": float(df_results["fixed_recovered"].mean()),
        "base10_recovery_rate": float(df_results["base10_recovered"].mean()),
        "base10_sampling_ceiling": float(df_results["base10_ceiling"].mean()),
        "base10_verdicts": df_results["base10_verdict"].value_counts().to_dict(),
    }
    with open(tables_dir / "comparison_headline.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    def compute_breakdown(col_name, bins, labels, fname):
        df_results["bin"] = pd.cut(df_results[col_name], bins=bins, labels=labels, include_lowest=True, right=True)
        gb = df_results.groupby("bin", observed=False)
        bd = pd.DataFrame(
            {
                "count": gb.size(),
                "base5_recovered_rate": gb["base5_recovered"].mean(),
                "oracle_recovered_rate": gb["oracle_recovered"].mean(),
                "oracle_ceiling": gb["oracle_ceiling"].mean(),
                "delta_recovery_pts": (gb["oracle_recovered"].mean() - gb["base5_recovered"].mean()) * 100,
                "base10_recovered_rate": gb["base10_recovered"].mean(),
            }
        ).reset_index()
        bd.to_csv(tables_dir / f"breakdown_{fname}.csv", index=False)
        return bd

    bd_pos = compute_breakdown("dof_pos", [-0.5, 2.5, 5.5, 10.5, 1000], ["0–2", "3–5", "6–10", ">10"], "dof_pos")
    bd_tot = compute_breakdown("dof_total", [5.5, 10.5, 1000], ["6–10", ">10"], "dof_total")
    bd_nsites = compute_breakdown("nsites", [0, 10, 20, 40, 1000], ["≤10", "11–20", "21–40", ">40"], "nsites")

    gb_cs = df_results.groupby("crystal_system", observed=False)
    bd_cs = pd.DataFrame(
        {
            "crystal_system": list(gb_cs.groups.keys()),
            "count": gb_cs.size(),
            "base5_recovered_rate": gb_cs["base5_recovered"].mean(),
            "oracle_recovered_rate": gb_cs["oracle_recovered"].mean(),
            "delta_recovery_pts": (gb_cs["oracle_recovered"].mean() - gb_cs["base5_recovered"].mean()) * 100,
            "base10_recovered_rate": gb_cs["base10_recovered"].mean(),
        }
    ).reset_index(drop=True)
    bd_cs.to_csv(tables_dir / "breakdown_crystal_system.csv", index=False)

    report_lines = [
        "# CrySPR Oracle Initialized-Lattice Study: Technical Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
        "**Protocol:** Initialized with ground-truth conventional cell + standard CRySPR variable-cell relaxation (`factor=1.0`)  ",
        f"**Evaluated Cohort:** {n_tot} structures with Total DOF $\\ge 6$  ",
        "**Relaxation Backend:** ORB-v3 conservative infinite cutoff (`orb_conserv_inf-omat-20250404`)  ",
        "**Trial Budget:** 5 trials per structure (apples-to-apples vs 5 trials of lattice-free run)  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "In this repeated oracle experiment, two major adjustments were made based on root-cause findings:",
        "1. **Dropped `factor=1.3`**: Used standard atomic tolerance matrix with `factor=1.0` (with fallback to 0.9), eliminating exclusion volume choking.",
        "2. **Initialized cell without freezing**: PyXtal was initialized with the exact ground-truth conventional unit cell, but the cell was allowed to relax using the standard 4-stage CRySPR schedule (`FrechetCellFilter`), preserving strain-displacement coupled relaxation modes.",
        "",
        "### Key Findings",
        f"- **5-Trial Recovery Rate**: **{summary['oracle_recovery_rate']*100:.1f}%** ({int(summary['oracle_recovery_rate']*n_tot)} / {n_tot}) compared to **{summary['base5_recovery_rate']*100:.1f}%** for lattice-free baseline ({summary['oracle_recovery_rate']*100 - summary['base5_recovery_rate']*100:+.1f}% pts).",
        f"- **Sampling Ceiling**: **{summary['oracle_sampling_ceiling']*100:.1f}%** vs **{summary['base5_sampling_ceiling']*100:.1f}%**.",
        f"- **Generation Failures**: Collapsed from 39.8% down to **{summary['oracle_verdicts'].get('generation_failed', 0)/n_tot*100:.1f}%**.",
        "",
        "---",
        "",
        "## 2. Headline Metrics (Apples-to-Apples: 5 Trials vs 5 Trials)",
        "",
        "| Metric | Lattice-Free (5 Trials) | Fixed-Cell Oracle (5 Trials) | Initialized-Cell Oracle (5 Trials) | Delta (Init vs Free) | Lattice-Free Ref (10 Trials) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **Recovery Rate** | **{summary['base5_recovery_rate']*100:.1f}%** | {summary['fixed_recovery_rate']*100:.1f}% | **{summary['oracle_recovery_rate']*100:.1f}%** | **{summary['oracle_recovery_rate']*100 - summary['base5_recovery_rate']*100:+.1f}% pts** | **{summary['base10_recovery_rate']*100:.1f}%** |",
        f"| **Sampling Ceiling** | {summary['base5_sampling_ceiling']*100:.1f}% | 16.8% | {summary['oracle_sampling_ceiling']*100:.1f}% | {summary['oracle_sampling_ceiling']*100 - summary['base5_sampling_ceiling']*100:+.1f}% pts | {summary['base10_sampling_ceiling']*100:.1f}% |",
        f"| **CDVAE Loose Recovery** | {summary['base5_loose_recovery']*100:.1f}% | 18.0% | {summary['oracle_loose_recovery']*100:.1f}% | {summary['oracle_loose_recovery']*100 - summary['base5_loose_recovery']*100:+.1f}% pts | - |",
        f"| **Missed Rate** | {summary['base5_verdicts'].get('missed', 0)/n_tot*100:.1f}% | 43.5% | {summary['oracle_verdicts'].get('missed', 0)/n_tot*100:.1f}% | {(summary['oracle_verdicts'].get('missed', 0) - summary['base5_verdicts'].get('missed', 0))/n_tot*100:+.1f}% pts | {summary['base10_verdicts'].get('missed', 0)/n_tot*100:.1f}% |",
        f"| **Generation Failed** | {summary['base5_verdicts'].get('generation_failed', 0)/n_tot*100:.1f}% | 39.8% | {summary['oracle_verdicts'].get('generation_failed', 0)/n_tot*100:.1f}% | {(summary['oracle_verdicts'].get('generation_failed', 0) - summary['base5_verdicts'].get('generation_failed', 0))/n_tot*100:+.1f}% pts | {summary['base10_verdicts'].get('generation_failed', 0)/n_tot*100:.1f}% |",
        "",
        "---",
        "",
        "## 3. Breakdown Analyses",
        "",
        "### 3.1 By Positional Degrees of Freedom (`dof_pos`)",
        "",
        "| Positional DOF | Structure Count | Lattice-Free (5 Trials) | Initialized-Cell Oracle (5 Trials) | Delta (pts) | Lattice-Free (10 Trials) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in bd_pos.iterrows():
        b5_str = f"{r['base5_recovered_rate']*100:.1f}%" if pd.notna(r['base5_recovered_rate']) else "N/A"
        ora_str = f"{r['oracle_recovered_rate']*100:.1f}%" if pd.notna(r['oracle_recovered_rate']) else "N/A"
        d_str = f"{r['delta_recovery_pts']:+.1f}%" if pd.notna(r['delta_recovery_pts']) else "N/A"
        b10_str = f"{r['base10_recovered_rate']*100:.1f}%" if pd.notna(r['base10_recovered_rate']) else "N/A"
        report_lines.append(f"| **{r['bin']}** | {int(r['count'])} | {b5_str} | **{ora_str}** | {d_str} | {b10_str} |")

    report_lines.extend([
        "",
        "### 3.2 By Total Degrees of Freedom (`dof_total`)",
        "",
        "| Total DOF | Structure Count | Lattice-Free (5 Trials) | Initialized-Cell Oracle (5 Trials) | Delta (pts) | Lattice-Free (10 Trials) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for _, r in bd_tot.iterrows():
        b5_str = f"{r['base5_recovered_rate']*100:.1f}%" if pd.notna(r['base5_recovered_rate']) else "N/A"
        ora_str = f"{r['oracle_recovered_rate']*100:.1f}%" if pd.notna(r['oracle_recovered_rate']) else "N/A"
        d_str = f"{r['delta_recovery_pts']:+.1f}%" if pd.notna(r['delta_recovery_pts']) else "N/A"
        b10_str = f"{r['base10_recovered_rate']*100:.1f}%" if pd.notna(r['base10_recovered_rate']) else "N/A"
        report_lines.append(f"| **{r['bin']}** | {int(r['count'])} | {b5_str} | **{ora_str}** | {d_str} | {b10_str} |")

    report_lines.extend([
        "",
        "### 3.3 By Crystal System",
        "",
        "| Crystal System | Structure Count | Lattice-Free (5 Trials) | Initialized-Cell Oracle (5 Trials) | Delta (pts) | Lattice-Free (10 Trials) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for _, r in bd_cs.iterrows():
        b5_str = f"{r['base5_recovered_rate']*100:.1f}%" if pd.notna(r['base5_recovered_rate']) else "N/A"
        ora_str = f"{r['oracle_recovered_rate']*100:.1f}%" if pd.notna(r['oracle_recovered_rate']) else "N/A"
        d_str = f"{r['delta_recovery_pts']:+.1f}%" if pd.notna(r['delta_recovery_pts']) else "N/A"
        b10_str = f"{r['base10_recovered_rate']*100:.1f}%" if pd.notna(r['base10_recovered_rate']) else "N/A"
        report_lines.append(f"| **{r['crystal_system'].capitalize()}** | {int(r['count'])} | {b5_str} | **{ora_str}** | {d_str} | {b10_str} |")

    report_lines.extend([
        "",
        "### 3.4 By System Size (`nsites`)",
        "",
        "| Number of Sites | Structure Count | Lattice-Free (5 Trials) | Initialized-Cell Oracle (5 Trials) | Delta (pts) | Lattice-Free (10 Trials) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for _, r in bd_nsites.iterrows():
        b5_str = f"{r['base5_recovered_rate']*100:.1f}%" if pd.notna(r['base5_recovered_rate']) else "N/A"
        ora_str = f"{r['oracle_recovered_rate']*100:.1f}%" if pd.notna(r['oracle_recovered_rate']) else "N/A"
        d_str = f"{r['delta_recovery_pts']:+.1f}%" if pd.notna(r['delta_recovery_pts']) else "N/A"
        b10_str = f"{r['base10_recovered_rate']*100:.1f}%" if pd.notna(r['base10_recovered_rate']) else "N/A"
        report_lines.append(f"| **{r['bin']}** | {int(r['count'])} | {b5_str} | **{ora_str}** | {d_str} | {b10_str} |")

    report_content = "\n".join(report_lines) + "\n"
    report_file = Path("docs/cryspr_oracle_relaxed_cell_report.md")
    report_file.write_text(report_content, encoding="utf-8")
    logger.info("Generated markdown report at %s", report_file)

    return summary, df_results


def main():
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="CrySPR Oracle Initialized-Lattice Study")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated/cryspr_oracle_relaxed_cell_study"),
        help="Root output directory",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="cuda:0,cuda:0,cuda:1,cuda:1,cuda:2",
        help="Comma-separated CUDA devices",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=400,
        help="Number of structures to sample (default: 400)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Number of trials per structure (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["run", "score", "all"],
        default="all",
        help="Stage to execute",
    )
    args = parser.parse_args()

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_cohort = prepare_cohort(n_samples=args.n_samples, seed=args.seed)
    sampled_cohort_file = out_dir / "data" / "sampled_400_targets.parquet"
    sampled_cohort_file.parent.mkdir(parents=True, exist_ok=True)
    df_cohort.to_parquet(sampled_cohort_file, index=False)
    logger.info("Archived 400 sampled targets to %s", sampled_cohort_file)

    if args.stage in ["run", "all"]:
        oracle_results = run_oracle_experiment(df_cohort, out_dir, devices=devices, n_trials=args.n_trials)
    else:
        with open(out_dir / "data" / "oracle_reconstruction_results.pkl", "rb") as f:
            oracle_results = pickle.load(f)

    if args.stage in ["score", "all"]:
        score_and_compare(df_cohort, oracle_results, out_dir)


if __name__ == "__main__":
    main()
