#!/usr/bin/env python3
"""Run the Lattice-Free Coordination Number Guided CRySPR Reconstruction Study.

Hypothesis: In the lattice-free baseline (factor=1.3), PyXtal samples loose, unjammed unit cells
(V ~ 1.5 - 2.0x V_GT) which relax via compressive annealing (50.8% recovery).
By drawing M=10 candidate structures with factor=1.3 and selecting the candidate with the lowest
coordination number error (CrystalNN MAE vs Oracle Ground Truth) BEFORE relaxation, we initialize
in the correct coordination basin while retaining the unjammed, expansive volume that allows
compressive annealing to work smoothly.

Protocol:
1. For each trial, draw M=10 candidate structures with PyXtal using factor=1.3 (lattice-free).
2. Score each candidate by MAE against Oracle Coordination Numbers (CrystalNN).
3. Select the best CN-matching candidate for relaxation.
4. Execute standard 4-stage variable-cell CRySPR relaxation (FrechetCellFilter, ORB-v3).
5. 5 trials per structure across high-DoF structures (Total DoF >= 6).
6. Apples-to-apples multi-way comparison against:
   - base5 (lattice-free baseline, 5 trials, 50.8%)
   - base10 (lattice-free baseline, 10 trials, 60.8%)
   - fixed5 (fixed-cell oracle, 5 trials, 16.2%)
   - relaxed_oracle5 (dense initialized-cell oracle without CN guidance, 5 trials, 19.0%)
   - cn_oracle5 (dense initialized-cell oracle with CN guidance, 5 trials, 16.2%)
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
from pymatgen.core import Structure
from pyxtal import pyxtal
from pyxtal.tolerance import Tol_matrix

from run_cryspr_reconstruction_study import (
    apply_runtime_patches,
    build_patched_orb_calculator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(processName)s) %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lf_cn_study")

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
    h = hashlib.sha256(f"lf_cn_{struct_idx}_{trial_idx}_{sub_idx}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


class PyXtalTimeout(BaseException):
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


def _generate_lattice_free_cn_candidate(
    gene: Dict,
    gt_cn_dict: Dict[str, float],
    cnn: CrystalNN,
    struct_idx: int,
    trial_idx: int,
    n_candidates: int = 10,
    factor: float = 1.3,
    timeout: int = 15,
) -> Tuple[Optional[Atoms], float, Dict[str, float]]:
    """Sample n_candidates lattice-free PyXtal candidates (factor=1.3) and select lowest CN error."""
    t_start = time.time()
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    candidates = []
    try:
        for f in [factor, 1.0]:  # fallback to 1.0 if 1.3 fails on high-DoF structures
            if time.time() - t_start > timeout:
                break
            tol_mat = Tol_matrix(prototype="atomic", factor=f)
            for m in range(n_candidates):
                rem_time = timeout - (time.time() - t_start)
                if rem_time <= 0:
                    break
                signal.alarm(max(1, int(rem_time)))
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
                except PyXtalTimeout:
                    raise
                except Exception:
                    continue
            if candidates:
                break
    except PyXtalTimeout:
        logger.warning("Struct %d trial %d: PyXtal generation timed out (>%ds)", struct_idx, trial_idx, timeout)
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
    n_candidates: int = 10,
    factor: float = 1.3,
    fmax: float = 0.02,
    steps_limit: int = 500,
) -> Optional[Dict]:
    """Execute CN-guided variable-cell reconstruction with factor=1.3 lattice-free candidate sampling."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    struct_idx = struct_info["struct_idx"]
    seed = _deterministic_seed(struct_idx, trial_idx)

    gene = struct_info["gene"]
    gt_cn = struct_info["gt_cn"]

    atoms, init_cn_err, init_cn_dict = _generate_lattice_free_cn_candidate(
        gene=gene,
        gt_cn_dict=gt_cn,
        cnn=cnn,
        struct_idx=struct_idx,
        trial_idx=trial_idx,
        n_candidates=n_candidates,
        factor=factor,
        timeout=15,
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
        atoms_s3.set_constraint([])
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

        accepted_s4 = bool(e_s4 < (e_s3 - 1e-4))
        kept_stage = "s4" if accepted_s4 else "s3"
        kept_e = e_s4 if accepted_s4 else e_s3
        kept_cif = cif_s4_path if accepted_s4 else cif_s3_path
        kept_atoms = atoms_s4.copy() if accepted_s4 else atoms_s3.copy()

        atoms_s3.calc = None
        atoms_s4.calc = None
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

    except Exception as exc:
        logger.warning("Struct %d trial %d: relaxation error (%s)", struct_idx, trial_idx, exc)
        return None


def _lf_cn_worker(
    gpu_id: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    output_dir: Path,
    n_trials: int = 5,
    n_candidates: int = 10,
    factor: float = 1.3,
):
    """Worker handling Lattice-Free CN-guided reconstruction on one GPU."""
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
                res = None
                # Check for cached completion
                s3_cifs = list(t_dir.glob("*_3_no-sym_cell+pos.cif"))
                s4_cifs = list(t_dir.glob("*_4_rattle_no-sym.cif"))
                if s3_cifs and (t_dir / f"{s3_cifs[0].stem.split('_')[0]}_0_initial.cif").exists():
                    try:
                        atoms_s3 = Structure.from_file(s3_cifs[0]).to_ase_atoms()
                        formula_stem = s3_cifs[0].stem.split('_')[0]
                        log3 = t_dir / f"{formula_stem}_3_no-sym_cell+pos.log"
                        e_s3 = None
                        if log3.exists():
                            for line in log3.read_text().splitlines():
                                parts = line.strip().split()
                                if len(parts) >= 4 and parts[0] == "BFGS:":
                                    try:
                                        e_s3 = float(parts[3])
                                    except ValueError:
                                        pass
                        if e_s3 is not None:
                            n_at = len(atoms_s3)
                            e_s4 = e_s3
                            cif_s4 = s3_cifs[0]
                            atoms_s4 = atoms_s3
                            accepted_s4 = False
                            if s4_cifs:
                                log4 = t_dir / f"{formula_stem}_4_rattle_no-sym.log"
                                if log4.exists():
                                    for line in log4.read_text().splitlines():
                                        parts = line.strip().split()
                                        if len(parts) >= 4 and parts[0] == "BFGS:":
                                            try:
                                                e_s4 = float(parts[3])
                                            except ValueError:
                                                pass
                                    if e_s4 < (e_s3 - 1e-4):
                                        accepted_s4 = True
                                        cif_s4 = s4_cifs[0]
                                        atoms_s4 = Structure.from_file(cif_s4).to_ase_atoms()

                            kept_stage = "s4" if accepted_s4 else "s3"
                            kept_e = e_s4 if accepted_s4 else e_s3
                            kept_cif = cif_s4 if accepted_s4 else s3_cifs[0]
                            kept_atoms = atoms_s4 if accepted_s4 else atoms_s3

                            res = {
                                "trial_idx": t_idx,
                                "init_cn_err": 0.0,
                                "init_cn_dict": {},
                                "e_s3": e_s3,
                                "e_s3_per_atom": e_s3 / n_at,
                                "cif_s3_path": str(s3_cifs[0]),
                                "e_s4": e_s4,
                                "e_s4_per_atom": e_s4 / n_at,
                                "cif_s4_path": str(cif_s4),
                                "accepted_s4": accepted_s4,
                                "kept_stage": kept_stage,
                                "kept_e": kept_e,
                                "kept_e_per_atom": kept_e / n_at,
                                "kept_cif_path": str(kept_cif),
                                "atoms_s3": atoms_s3,
                                "atoms_s4": atoms_s4,
                                "kept_atoms": kept_atoms,
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
                        n_candidates=n_candidates,
                        factor=factor,
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
        logger.exception("Error in Lattice-Free CN worker on %s: %s", gpu_id, exc)
    finally:
        logger.info("Worker on %s finished", gpu_id)


def prepare_cohort(n_samples: int = 400, seed: int = 42) -> pd.DataFrame:
    cohort_path = Path("generated/cryspr_oracle_fixed_lattice_study/data/sampled_400_targets.parquet")
    df_cohort = pd.read_parquet(cohort_path)
    if n_samples is not None and n_samples < len(df_cohort):
        df_cohort = df_cohort.iloc[:n_samples].reset_index(drop=True)
    return df_cohort


def run_experiment(
    df_cohort: pd.DataFrame,
    output_dir: Path,
    devices: List[str],
    n_trials: int = 5,
    n_candidates: int = 10,
    factor: float = 1.3,
) -> Dict[str, Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / "lattice_free_cn_results.pkl"

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

        tasks.append(
            {
                "struct_idx": int(row["struct_idx"]),
                "immutable_id": imm_id,
                "cif_path": row["cif_path"],
                "crystal_system": row["crystal_system"],
                "dof_total": int(row["dof_total"]),
                "dof_pos": int(row["dof_pos"]),
                "gene": row["wyckoff_gene"],
                "gt_cn": gt_cn,
            }
        )

    if not tasks:
        logger.info("All %d structures already completed!", len(df_cohort))
        return results_by_id

    logger.info(
        "Launching Lattice-Free CN Experiment (factor=%.1f, M=%d): %d to run (%d done), %d trials on %s...",
        factor,
        n_candidates,
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
            target=_lf_cn_worker,
            args=(dev, task_queue, result_queue, output_dir, n_trials, n_candidates, factor),
        )
        p.start()
        processes.append(p)

    n_done = 0
    t0 = time.time()
    while n_done < len(tasks):
        res = result_queue.get()
        results_by_id[res["immutable_id"]] = res
        n_done += 1

        if n_done % 5 == 0 or n_done == len(tasks):
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

    return results_by_id


def score_and_compare(df_cohort: pd.DataFrame, lf_cn_results: Dict[str, Dict], output_dir: Path):
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    ref_cn_study = Path("generated/cryspr_oracle_coordination_study/tables/results_per_structure.csv")
    df_ref = pd.read_csv(ref_cn_study)
    ref_by_id = {row["immutable_id"]: row for _, row in df_ref.iterrows()}

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0, primitive_cell=True)

    records = []
    verdicts = defaultdict(int)

    for _, row in df_cohort.iterrows():
        imm_id = row["immutable_id"]
        s_idx = int(row["struct_idx"])
        ref_row = ref_by_id.get(imm_id, {})

        tgt_struct = Structure.from_file(row["cif_path"])
        e_tgt = float(row["orb_e_target_per_atom"])

        result = lf_cn_results.get(imm_id)
        if result is None or not result.get("trials"):
            verdict = "generation_failed"
            recovered = False
            ceiling = False
            e_kept = None
            de_kept = None
        else:
            trials = result["trials"]
            valid_trials = [t for t in trials if t.get("kept_e_per_atom") is not None]

            if not valid_trials:
                verdict = "generation_failed"
                recovered = False
                ceiling = False
                e_kept = None
                de_kept = None
            else:
                best_trial = min(valid_trials, key=lambda t: t["kept_e_per_atom"])
                e_kept = best_trial["kept_e_per_atom"]
                de_kept = e_kept - e_tgt

                trial_matches = []
                for t in valid_trials:
                    atoms = t.get("kept_atoms")
                    if atoms is not None:
                        try:
                            pmg_s = Structure(
                                lattice=atoms.get_cell(),
                                species=atoms.get_chemical_symbols(),
                                coords=atoms.get_positions(),
                                coords_are_cartesian=True,
                            )
                            trial_matches.append(bool(matcher.fit(pmg_s, tgt_struct)))
                        except Exception:
                            trial_matches.append(False)
                    elif t.get("kept_cif_path") and Path(t["kept_cif_path"]).exists():
                        try:
                            pmg_s = Structure.from_file(t["kept_cif_path"])
                            trial_matches.append(bool(matcher.fit(pmg_s, tgt_struct)))
                        except Exception:
                            trial_matches.append(False)
                    else:
                        trial_matches.append(False)

                ceiling = any(trial_matches)

                best_atoms = best_trial.get("kept_atoms")
                best_match = False
                if best_atoms is not None:
                    try:
                        pmg_best = Structure(
                            lattice=best_atoms.get_cell(),
                            species=best_atoms.get_chemical_symbols(),
                            coords=best_atoms.get_positions(),
                            coords_are_cartesian=True,
                        )
                        best_match = bool(matcher.fit(pmg_best, tgt_struct))
                    except Exception:
                        best_match = False
                elif best_trial.get("kept_cif_path") and Path(best_trial["kept_cif_path"]).exists():
                    try:
                        pmg_best = Structure.from_file(best_trial["kept_cif_path"])
                        best_match = bool(matcher.fit(pmg_best, tgt_struct))
                    except Exception:
                        best_match = False

                recovered = best_match
                if recovered:
                    verdict = "recovered"
                elif de_kept < -0.001:
                    verdict = "lower_energy_alternative"
                elif ceiling and not recovered:
                    verdict = "sampled_not_selected"
                else:
                    verdict = "missed"

        verdicts[verdict] += 1
        records.append(
            {
                "struct_idx": s_idx,
                "immutable_id": imm_id,
                "formula": row["chemical_formula_reduced"],
                "crystal_system": row["crystal_system"],
                "dof_total": int(row["dof_total"]),
                "dof_pos": int(row["dof_pos"]),
                "base5_recovered": ref_row.get("base5_recovered"),
                "fixed_recovered": ref_row.get("fixed_recovered"),
                "relaxed_recovered": ref_row.get("relaxed_recovered"),
                "dense_cn_recovered": ref_row.get("cn_recovered"),
                "lf_cn_recovered": recovered,
                "lf_cn_ceiling": ceiling,
                "lf_cn_verdict": verdict,
                "lf_cn_e_kept": e_kept,
                "lf_cn_de_kept": de_kept,
                "base10_recovered": ref_row.get("base10_recovered"),
            }
        )

    df_results = pd.DataFrame(records)
    df_results.to_csv(tables_dir / "results_per_structure.csv", index=False)

    summary = {
        "cohort_size": len(df_results),
        "base5_recovery_rate": float(df_results["base5_recovered"].mean()),
        "fixed_recovery_rate": float(df_results["fixed_recovered"].mean()),
        "relaxed_oracle_recovery_rate": float(df_results["relaxed_recovered"].mean()),
        "dense_cn_recovery_rate": float(df_results["dense_cn_recovered"].mean()),
        "lf_cn_recovery_rate": float(df_results["lf_cn_recovered"].mean()),
        "lf_cn_sampling_ceiling": float(df_results["lf_cn_ceiling"].mean()),
        "lf_cn_verdicts": dict(verdicts),
        "base10_recovery_rate": float(df_results["base10_recovered"].mean()),
    }

    with open(tables_dir / "comparison_headline.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    logger.info("=" * 70)
    logger.info("LATTICE-FREE CN STUDY HEADLINE RESULTS")
    logger.info("=" * 70)
    logger.info("Cohort size: %d structures (DoF >= 6)", len(df_results))
    logger.info("Lattice-Free Baseline (5 trials):   %.1f%%", summary["base5_recovery_rate"] * 100)
    logger.info("Dense-Cell CN Oracle (5 trials):    %.1f%%", summary["dense_cn_recovery_rate"] * 100)
    logger.info("LATTICE-FREE CN GUIDED (5 trials):  %.1f%%", summary["lf_cn_recovery_rate"] * 100)
    logger.info("Lattice-Free CN Sampling Ceiling:   %.1f%%", summary["lf_cn_sampling_ceiling"] * 100)
    logger.info("Lattice-Free Reference (10 trials): %.1f%%", summary["base10_recovery_rate"] * 100)
    logger.info("=" * 70)

    return summary, df_results


def main():
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Lattice-Free CN Study")
    parser.add_argument("--output-dir", type=Path, default=Path("generated/cryspr_lattice_free_cn_study"))
    parser.add_argument("--devices", type=str, default="cuda:0,cuda:0,cuda:1,cuda:1")
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--n-candidates", type=int, default=10)
    parser.add_argument("--factor", type=float, default=1.3)
    parser.add_argument("--stage", type=str, choices=["run", "score", "all"], default="all")
    args = parser.parse_args()

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_cohort = prepare_cohort(n_samples=args.n_samples)

    if args.stage in ["run", "all"]:
        lf_cn_results = run_experiment(
            df_cohort=df_cohort,
            output_dir=out_dir,
            devices=devices,
            n_trials=args.n_trials,
            n_candidates=args.n_candidates,
            factor=args.factor,
        )
    else:
        with open(out_dir / "data" / "lattice_free_cn_results.pkl", "rb") as f:
            lf_cn_results = pickle.load(f)

    if args.stage in ["score", "all"]:
        score_and_compare(df_cohort, lf_cn_results, out_dir)


if __name__ == "__main__":
    main()
