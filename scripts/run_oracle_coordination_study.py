#!/usr/bin/env python3
"""Run the Oracle Coordination Number Guided CRySPR Reconstruction Study across available GPUs.

Hypothesis: Knowing the ground-truth coordination numbers allows rapid permissive PyXtal sampling
(factor=1.0, 0% timeouts) paired with pre-relaxation CN filtering to select the candidate that starts
inside the correct coordination basin, breaking the trade-off between initialization failure and high final energy.

Protocol:
1. Ground-truth conventional unit cell initialization (factor=1.0, fallback 0.9).
2. For each trial, draw M=10 candidate structures with PyXtal.
3. Score each candidate by MAE against Oracle Coordination Numbers (CrystalNN).
4. Select the best CN-matching candidate for relaxation.
5. Execute standard 4-stage variable-cell CRySPR relaxation (FrechetCellFilter).
6. 5 trials per structure across all 400 high-DoF structures (Total DoF >= 6).
7. Apples-to-apples 5-way comparison against:
   - base5 (lattice-free baseline, 5 trials)
   - base10 (lattice-free baseline, 10 trials)
   - fixed5 (fixed-cell oracle, 5 trials)
   - relaxed_oracle5 (initialized-cell oracle without CN guidance, 5 trials)
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
from pymatgen.analysis.local_env import CrystalNN
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
logger = logging.getLogger("oracle_cn_study")

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


def _deterministic_seed(struct_idx: int, trial_idx: int, sub_idx: int = 0) -> int:
    h = hashlib.sha256(f"oracle_cn_{struct_idx}_{trial_idx}_{sub_idx}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


class PyXtalTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise PyXtalTimeout("PyXtal candidate generation timed out")


def compute_species_cn(struct: Structure, cnn: CrystalNN) -> Dict[str, float]:
    cns_by_elem = defaultdict(list)
    for i, site in enumerate(struct):
        try:
            cn = cnn.get_cn(struct, i)
        except Exception:
            cn = 0
        cns_by_elem[site.specie.symbol].append(cn)
    return {elem: float(np.mean(vals)) for elem, vals in cns_by_elem.items()}


def compute_cn_mae(cn_dict_candidate: Dict[str, float], cn_dict_target: Dict[str, float]) -> float:
    all_elems = set(cn_dict_target.keys())
    if not all_elems:
        return 999.0
    errs = [abs(cn_dict_candidate.get(e, 0.0) - cn_dict_target[e]) for e in all_elems]
    return float(np.mean(errs))


def _generate_cn_guided_candidate(
    gene: Dict,
    pxt_lat: PxtLattice,
    gt_cn_dict: Dict[str, float],
    cnn: CrystalNN,
    struct_idx: int,
    trial_idx: int,
    n_candidates: int = 10,
    timeout: int = 15,
) -> Tuple[Optional[Atoms], float, Dict[str, float]]:
    """Sample n_candidates permissive PyXtal candidates and select the one with lowest CN error."""
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    candidates = []
    try:
        for factor in [1.0, 0.9]:
            tol_mat = Tol_matrix(prototype="atomic", factor=factor)
            for m in range(n_candidates):
                sub_seed = _deterministic_seed(struct_idx, trial_idx, m)
                np.random.seed(sub_seed)
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
                        max_count=4,
                    )
                    if cand.valid:
                        pmg_cand = cand.to_pymatgen()
                        cand_cn = compute_species_cn(pmg_cand, cnn)
                        err = compute_cn_mae(cand_cn, gt_cn_dict)
                        candidates.append((err, cand.to_ase(), cand_cn))
                        if len(candidates) >= n_candidates:
                            break
                except Exception:
                    continue
            if candidates:
                break
    except PyXtalTimeout:
        pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if not candidates:
        return None, 999.0, {}

    candidates.sort(key=lambda x: x[0])
    best_err, best_atoms, best_cn = candidates[0]
    return best_atoms, best_err, best_cn


def _reconstruct_single_trial(
    struct_info: Dict,
    trial_idx: int,
    trial_dir: Path,
    calc,
    cnn: CrystalNN,
    fmax: float = 0.02,
    steps_limit: int = 500,
) -> Optional[Dict]:
    """Execute CN-guided variable-cell reconstruction initialized from ground-truth lattice."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    struct_idx = struct_info["struct_idx"]
    seed = _deterministic_seed(struct_idx, trial_idx)

    gene = struct_info["gene"]
    pxt_lat = struct_info["pxt_lattice"]
    gt_cn = struct_info["gt_cn"]

    atoms, init_cn_err, init_cn_dict = _generate_cn_guided_candidate(
        gene, pxt_lat, gt_cn, cnn, struct_idx, trial_idx, n_candidates=10, timeout=15
    )
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
            "init_cn_err": init_cn_err,
            "init_cn_dict": init_cn_dict,
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
    """Worker handling CN-guided Oracle reconstruction across assigned structure tasks on one GPU."""
    try:
        apply_runtime_patches()
        calc = build_patched_orb_calculator(gpu_id)
        cnn = CrystalNN()
        import torch

        while True:
            task = task_queue.get()
            if task is None:
                break
            struct_info = task
            s_idx = struct_info["struct_idx"]
            imm_id = struct_info["immutable_id"]
            struct_dir = output_dir / "cryspr" / str(s_idx)

            trial_results = []
            n_gen = 0
            for t_idx in range(n_trials):
                t_dir = struct_dir / f"trial-{t_idx}"
                # Resume trial if s3 or s4 cif exists
                res = None
                s3_cifs = list(t_dir.glob("*_3_no-sym_cell+pos.cif"))
                if s3_cifs and (t_dir / f"{s3_cifs[0].stem.split('_')[0]}_0_initial.cif").exists():
                    try:
                        atoms_s3 = Structure.from_file(s3_cifs[0]).to_ase_atoms()
                        log3 = t_dir / f"{s3_cifs[0].stem.split('_')[0]}_3_no-sym_cell+pos.log"
                        e_s3 = None
                        if log3.exists():
                            for line in log3.read_text().splitlines():
                                parts = line.strip().split()
                                if len(parts) >= 4 and parts[0] == "BFGS:":
                                    e_s3 = float(parts[3])
                        if e_s3 is not None:
                            n_at = len(atoms_s3)
                            res = {
                                "trial_idx": t_idx,
                                "init_cn_err": 0.0,
                                "init_cn_dict": {},
                                "e_s3": e_s3,
                                "e_s3_per_atom": e_s3 / n_at,
                                "cif_s3_path": str(s3_cifs[0]),
                                "e_s4": e_s3,
                                "e_s4_per_atom": e_s3 / n_at,
                                "cif_s4_path": str(s3_cifs[0]),
                                "accepted_s4": False,
                                "kept_stage": "s3",
                                "kept_e": e_s3,
                                "kept_e_per_atom": e_s3 / n_at,
                                "kept_cif_path": str(s3_cifs[0]),
                                "atoms_s3": atoms_s3,
                                "atoms_s4": atoms_s3,
                                "kept_atoms": atoms_s3,
                            }
                    except Exception:
                        res = None

                if res is None:
                    res = _reconstruct_single_trial(
                        struct_info=struct_info,
                        trial_idx=t_idx,
                        trial_dir=t_dir,
                        calc=calc,
                        cnn=cnn,
                    )

                if res is not None:
                    trial_results.append(res)
                    n_gen += 1

            result_queue.put(
                {
                    "struct_idx": s_idx,
                    "immutable_id": imm_id,
                    "n_trials_generated": n_gen,
                    "trials": trial_results,
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as exc:
        logger.exception("Error in CN oracle worker on %s: %s", gpu_id, exc)
    finally:
        logger.info("Worker on %s finished", gpu_id)


def prepare_cohort(n_samples: int = 400, seed: int = 42) -> pd.DataFrame:
    cohort_path = Path("generated/cryspr_oracle_fixed_lattice_study/data/sampled_400_targets.parquet")
    df_cohort = pd.read_parquet(cohort_path)
    if n_samples is not None and n_samples < len(df_cohort):
        df_cohort = df_cohort.iloc[:n_samples].reset_index(drop=True)
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

    cnn = CrystalNN()
    tasks = []
    for _, row in df_cohort.iterrows():
        imm_id = row["immutable_id"]
        if imm_id in results_by_id:
            continue

        tgt_struct = Structure.from_file(row["cif_path"])
        gt_cn = compute_species_cn(tgt_struct, cnn)

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
                "gt_cn": gt_cn,
            }
        )

    if not tasks:
        logger.info("All %d structures already completed!", len(df_cohort))
        return results_by_id

    logger.info(
        "Launching CN-Guided Oracle Experiment: %d structures to run (%d already done), %d trials/structure on %s...",
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
        )
        p.start()
        processes.append(p)

    n_done = 0
    t0 = time.time()
    while n_done < len(tasks):
        res = result_queue.get()
        results_by_id[res["immutable_id"]] = res
        n_done += 1

        if n_done % 10 == 0 or n_done == len(tasks):
            with open(cache_path, "wb") as f:
                pickle.dump(results_by_id, f)

        elapsed = time.time() - t0
        rate = n_done / max(elapsed, 1e-5)
        rem = (len(tasks) - n_done) / max(rate, 1e-5)
        logger.info(
            "Progress: %d/%d completed (total: %d/%d, %.3f struct/s, elapsed: %.1fs, eta: %.1fs)",
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
    logger.info("CN Oracle reconstruction completed and saved to %s", cache_path)
    return results_by_id


def score_and_compare(
    df_cohort: pd.DataFrame,
    cn_oracle_results: Dict[str, Dict],
    output_dir: Path,
):
    logger.info("Scoring and computing multi-way comparison metrics...")
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    baseline_cache_path = Path("generated/cryspr_reconstruction_study/data/reconstruction_results.pkl")
    with open(baseline_cache_path, "rb") as f:
        baseline_raw = pickle.load(f)

    fixed_cache = Path("generated/cryspr_oracle_fixed_lattice_study/data/oracle_reconstruction_results.pkl")
    fixed_oracle_raw = {}
    if fixed_cache.exists():
        with open(fixed_cache, "rb") as f:
            fixed_oracle_raw = pickle.load(f)

    relaxed_cache = Path("generated/cryspr_oracle_relaxed_cell_study/data/oracle_reconstruction_results.pkl")
    relaxed_oracle_raw = {}
    if relaxed_cache.exists():
        with open(relaxed_cache, "rb") as f:
            relaxed_oracle_raw = pickle.load(f)

    matcher_default = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True)
    matcher_loose = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True, scale=True)

    rows = []
    for _, row in df_cohort.iterrows():
        imm_id = row["immutable_id"]
        gid = row["gene_id"]
        tgt_struct = Structure.from_file(row["cif_path"])
        tgt_e = row["orb_e_target_per_atom"]

        cn_info = cn_oracle_results.get(imm_id, {"n_trials_generated": 0, "trials": []})
        cn_trials = cn_info.get("trials", [])
        cn_n_gen = cn_info.get("n_trials_generated", 0)

        base_gene_trials = baseline_raw.get(gid, {}).get("trials", [])
        base5_trials = base_gene_trials[:5]
        base10_trials = base_gene_trials[:10]

        fixed_trials = fixed_oracle_raw.get(imm_id, {}).get("trials", [])
        relaxed_trials = relaxed_oracle_raw.get(imm_id, {}).get("trials", [])

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

        eval_cn = evaluate_trial_set(cn_trials, cn_n_gen)
        eval_base5 = evaluate_trial_set(base5_trials, min(5, len(base_gene_trials)))
        eval_base10 = evaluate_trial_set(base10_trials, min(10, len(base_gene_trials)))
        eval_fixed = evaluate_trial_set(fixed_trials, len(fixed_trials))
        eval_relaxed = evaluate_trial_set(relaxed_trials, len(relaxed_trials))

        rows.append(
            {
                "struct_idx": row["struct_idx"],
                "immutable_id": imm_id,
                "formula": row["chemical_formula_reduced"],
                "spacegroup": row["spacegroup"],
                "crystal_system": row["crystal_system"],
                "nsites": row["nsites"],
                "dof_total": row["dof_total"],
                "dof_pos": row["dof_pos"],
                "e_target": tgt_e,
                "base5_recovered": eval_base5["recovered"],
                "base5_ceiling": eval_base5["ceiling"],
                "base5_verdict": eval_base5["verdict"],
                "base5_de_kept": eval_base5["de_kept"],
                "fixed_recovered": eval_fixed["recovered"],
                "fixed_verdict": eval_fixed["verdict"],
                "relaxed_recovered": eval_relaxed["recovered"],
                "relaxed_verdict": eval_relaxed["verdict"],
                "cn_recovered": eval_cn["recovered"],
                "cn_ceiling": eval_cn["ceiling"],
                "cn_loose_rec": eval_cn["loose_rec"],
                "cn_verdict": eval_cn["verdict"],
                "cn_e_kept": eval_cn["e_kept"],
                "cn_de_kept": eval_cn["de_kept"],
                "base10_recovered": eval_base10["recovered"],
                "base10_verdict": eval_base10["verdict"],
            }
        )

    df_results = pd.DataFrame(rows)
    df_results.to_csv(tables_dir / "results_per_structure.csv", index=False)

    summary = {
        "cohort_size": len(df_results),
        "base5_recovery_rate": float(df_results["base5_recovered"].mean()),
        "base5_sampling_ceiling": float(df_results["base5_ceiling"].mean()),
        "fixed_recovery_rate": float(df_results["fixed_recovered"].mean()),
        "relaxed_oracle_recovery_rate": float(df_results["relaxed_recovered"].mean()),
        "cn_oracle_recovery_rate": float(df_results["cn_recovered"].mean()),
        "cn_oracle_sampling_ceiling": float(df_results["cn_ceiling"].mean()),
        "cn_oracle_loose_recovery": float(df_results["cn_loose_rec"].mean()),
        "cn_oracle_verdicts": df_results["cn_verdict"].value_counts().to_dict(),
        "base10_recovery_rate": float(df_results["base10_recovered"].mean()),
    }

    with open(tables_dir / "comparison_headline.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    # Print summary
    logger.info("=" * 70)
    logger.info("CRySPR ORACLE COORDINATION NUMBER STUDY HEADLINE RESULTS")
    logger.info("=" * 70)
    logger.info("Cohort size: %d structures (DoF >= 6)", summary["cohort_size"])
    logger.info("Lattice-Free Baseline (5 trials):   %.1f%%", summary["base5_recovery_rate"] * 100)
    logger.info("Fixed-Cell Oracle (5 trials):        %.1f%%", summary["fixed_recovery_rate"] * 100)
    logger.info("Initialized-Cell Oracle (5 trials):  %.1f%%", summary["relaxed_oracle_recovery_rate"] * 100)
    logger.info("CN-GUIDED ORACLE (5 trials):         %.1f%%", summary["cn_oracle_recovery_rate"] * 100)
    logger.info("CN-Guided Sampling Ceiling:          %.1f%%", summary["cn_oracle_sampling_ceiling"] * 100)
    logger.info("Lattice-Free Reference (10 trials): %.1f%%", summary["base10_recovery_rate"] * 100)
    logger.info("=" * 70)

    return summary, df_results


def main():
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="CrySPR Oracle Coordination Number Study")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated/cryspr_oracle_coordination_study"),
        help="Root output directory",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="cuda:0,cuda:0,cuda:1,cuda:1",
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
        help="Random seed (default: 42)",
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

    if args.stage in ["run", "all"]:
        oracle_results = run_oracle_experiment(df_cohort, out_dir, devices=devices, n_trials=args.n_trials)
    else:
        with open(out_dir / "data" / "oracle_reconstruction_results.pkl", "rb") as f:
            oracle_results = pickle.load(f)

    if args.stage in ["score", "all"]:
        score_and_compare(df_cohort, oracle_results, out_dir)


if __name__ == "__main__":
    main()
