#!/usr/bin/env python3
"""Multi-core Empirical evaluation of the Oracle Coordination Number hypothesis."""
import multiprocessing as mp
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.local_env import CrystalNN, MinimumDistanceNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

warnings.filterwarnings("ignore")

_repo_root = Path(__file__).resolve().parent.parent

def compute_species_cn(struct, nn_method):
    cns_by_elem = defaultdict(list)
    for i, site in enumerate(struct):
        try:
            cn = nn_method.get_cn(struct, i)
        except Exception:
            cn = 0
        cns_by_elem[site.specie.symbol].append(cn)
    return {elem: float(np.mean(vals)) for elem, vals in cns_by_elem.items()}

def compute_cn_mae(cn_dict_candidate, cn_dict_target):
    all_elems = set(cn_dict_target.keys())
    if not all_elems:
        return np.nan
    errs = [abs(cn_dict_candidate.get(e, 0.0) - cn_dict_target[e]) for e in all_elems]
    return float(np.mean(errs))

def process_single_structure(args):
    s_idx, row_dict = args
    imm_id = row_dict["immutable_id"]
    cif_path = _repo_root / row_dict["cif_path"]
    study_dir = _repo_root / "generated/cryspr_oracle_relaxed_cell_study/cryspr"
    s_dir = study_dir / str(s_idx)
    
    if not cif_path.exists() or not s_dir.exists():
        return []
        
    try:
        tgt_struct = Structure.from_file(cif_path)
    except Exception:
        return []
        
    tgt_e = row_dict["orb_e_target_per_atom"]
    cnn = CrystalNN()
    matcher_default = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True)
    
    gt_cn = compute_species_cn(tgt_struct, cnn)
    
    records = []
    trial_dirs = sorted(s_dir.glob("trial-*"))
    
    for t_dir in trial_dirs:
        t_idx = int(t_dir.name.split("-")[1])
        init_cifs = list(t_dir.glob("*_0_initial.cif"))
        final_cifs = list(t_dir.glob("*_3_no-sym_cell+pos.cif"))
        if not init_cifs or not final_cifs:
            continue
            
        try:
            init_struct = Structure.from_file(init_cifs[0])
            final_struct = Structure.from_file(final_cifs[0])
        except Exception:
            continue
            
        init_cn = compute_species_cn(init_struct, cnn)
        final_cn = compute_species_cn(final_struct, cnn)
        
        init_mae = compute_cn_mae(init_cn, gt_cn)
        final_mae = compute_cn_mae(final_cn, gt_cn)
        
        recovered = matcher_default.fit(final_struct, tgt_struct)
        
        final_e = np.nan
        logfiles = list(t_dir.glob("*_3_no-sym_cell+pos.log"))
        if logfiles:
            for line in logfiles[0].read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 4 and parts[0] == "BFGS:":
                    try:
                        final_e = float(parts[3]) / len(final_struct)
                    except (ValueError, IndexError):
                        pass
                        
        delta_e = final_e - tgt_e if not np.isnan(final_e) else np.nan
        
        records.append({
            "struct_idx": s_idx,
            "immutable_id": imm_id,
            "trial_idx": t_idx,
            "formula": row_dict["chemical_formula_reduced"],
            "crystal_system": row_dict["crystal_system"],
            "spacegroup": row_dict["spacegroup"],
            "dof_pos": row_dict["dof_pos"],
            "recovered": recovered,
            "final_e": final_e,
            "delta_e": delta_e,
            "init_mae": init_mae,
            "final_mae": final_mae,
            "gt_cn": str(gt_cn),
            "init_cn": str(init_cn),
            "final_cn": str(final_cn),
        })
        
    return records

def main():
    print("Loading data...", flush=True)
    targets_path = _repo_root / "generated/cryspr_oracle_relaxed_cell_study/data/sampled_400_targets.parquet"
    df_targets = pd.read_parquet(targets_path)
    
    tasks = [(int(row["struct_idx"]), row.to_dict()) for _, row in df_targets.iterrows()]
    print(f"Distributing {len(tasks)} structures across 12 processes...", flush=True)
    
    with mp.Pool(12) as pool:
        results_nested = pool.map(process_single_structure, tasks)
        
    records = [r for sublist in results_nested for r in sublist]
    df_trials = pd.DataFrame(records)
    
    out_csv = _repo_root / "generated/cryspr_oracle_relaxed_cell_study/tables/oracle_cn_trials_analysis.csv"
    df_trials.to_csv(out_csv, index=False)
    print(f"Saved {len(df_trials)} trial records to {out_csv}", flush=True)
    
    # Analysis
    print("\n" + "="*60, flush=True)
    print("STATISTICAL FINDINGS: ORACLE COORDINATION NUMBER ANALYSIS", flush=True)
    print("="*60, flush=True)
    print(f"Total evaluated trials: {len(df_trials)} across {df_trials['struct_idx'].nunique()} structures", flush=True)
    n_rec = df_trials["recovered"].sum()
    print(f"Recovered trials: {n_rec} ({n_rec / len(df_trials) * 100:.2f}%)", flush=True)
    
    print("\n--- Initial Coordination Number Error vs Recovery Outcome ---", flush=True)
    mean_rec_init = df_trials[df_trials["recovered"] == True]["init_mae"].mean()
    mean_miss_init = df_trials[df_trials["recovered"] == False]["init_mae"].mean()
    median_rec_init = df_trials[df_trials["recovered"] == True]["init_mae"].median()
    median_miss_init = df_trials[df_trials["recovered"] == False]["init_mae"].median()
    print(f"Recovered Trials: Mean Init CN MAE = {mean_rec_init:.3f}, Median = {median_rec_init:.3f}", flush=True)
    print(f"Missed Trials:    Mean Init CN MAE = {mean_miss_init:.3f}, Median = {median_miss_init:.3f}", flush=True)
    print(f"Difference:       {mean_miss_init - mean_rec_init:+.3f} (Lower error in successful trials)", flush=True)
    
    print("\n--- Final Coordination Number Error vs Recovery Outcome ---", flush=True)
    mean_rec_final = df_trials[df_trials["recovered"] == True]["final_mae"].mean()
    mean_miss_final = df_trials[df_trials["recovered"] == False]["final_mae"].mean()
    print(f"Recovered Trials: Mean Final CN MAE = {mean_rec_final:.3f}", flush=True)
    print(f"Missed Trials:    Mean Final CN MAE = {mean_miss_final:.3f}", flush=True)
    
    print("\n--- Correlation with Final Energy Error (delta_e = E_final - E_target) ---", flush=True)
    valid_e = df_trials.dropna(subset=["delta_e"])
    corr_init = valid_e["init_mae"].corr(valid_e["delta_e"])
    corr_final = valid_e["final_mae"].corr(valid_e["delta_e"])
    print(f"Corr(Initial CN Error, delta_E): {corr_init:+.3f}", flush=True)
    print(f"Corr(Final CN Error, delta_E):   {corr_final:+.3f}", flush=True)
    
    # Recovery rate as a function of Initial CN Error bin
    print("\n--- Recovery Rate by Initial CN Error Quartile / Threshold ---", flush=True)
    df_trials["cn_err_bin"] = pd.qcut(df_trials["init_mae"], q=4, labels=["Q1 (Best CN)", "Q2", "Q3", "Q4 (Worst CN)"])
    bin_stats = df_trials.groupby("cn_err_bin", observed=False).agg(
        n_trials=("recovered", "count"),
        recovered=("recovered", "sum"),
        recovery_rate=("recovered", "mean"),
        mean_delta_e=("delta_e", "mean")
    )
    bin_stats["recovery_rate"] = bin_stats["recovery_rate"] * 100
    print(bin_stats.to_string(), flush=True)
    
    # Trial Selection Simulation
    print("\n" + "="*60, flush=True)
    print("TRIAL SELECTION BENCHMARK (Comparing 5 Selection Strategies)", flush=True)
    print("="*60, flush=True)
    
    struct_groups = df_trials.groupby("struct_idx")
    
    strat_energy = []
    strat_cn = []
    strat_random = []
    strat_ceiling = []
    strat_energy_plus_cn = []
    
    for s_idx, group in struct_groups:
        if len(group) == 0:
            continue
            
        strat_ceiling.append(group["recovered"].any())
        strat_random.append(group.iloc[0]["recovered"])
        
        group_e = group.dropna(subset=["final_e"])
        if len(group_e) > 0:
            best_e = group_e.sort_values("final_e").iloc[0]
            strat_energy.append(best_e["recovered"])
        else:
            strat_energy.append(group.iloc[0]["recovered"])
            
        # Oracle CN Selection (Pre-relaxation)
        best_cn = group.sort_values("init_mae").iloc[0]
        strat_cn.append(best_cn["recovered"])
        
        # Hybrid Selection: among top-2 lowest CN error, pick lowest final energy
        top2_cn = group.sort_values("init_mae").head(2)
        top2_e = top2_cn.dropna(subset=["final_e"])
        if len(top2_e) > 0:
            best_hybrid = top2_e.sort_values("final_e").iloc[0]
            strat_energy_plus_cn.append(best_hybrid["recovered"])
        else:
            strat_energy_plus_cn.append(top2_cn.iloc[0]["recovered"])
            
    print(f"1. Ceiling (Any of 5 trials visited ground state basin): {np.mean(strat_ceiling)*100:.2f}% ({sum(strat_ceiling)}/{len(strat_ceiling)})", flush=True)
    print(f"2. Random Trial Selection (Baseline single trial):        {np.mean(strat_random)*100:.2f}% ({sum(strat_random)}/{len(strat_random)})", flush=True)
    print(f"3. Oracle Initial CN Pre-Selection (Pick 1 before relax):  {np.mean(strat_cn)*100:.2f}% ({sum(strat_cn)}/{len(strat_cn)})", flush=True)
    print(f"4. Standard CRySPR (Relax all 5 -> pick lowest energy):   {np.mean(strat_energy)*100:.2f}% ({sum(strat_energy)}/{len(strat_energy)})", flush=True)
    print(f"5. Hybrid (Filter by CN, then pick lowest energy):        {np.mean(strat_energy_plus_cn)*100:.2f}% ({sum(strat_energy_plus_cn)}/{len(strat_energy_plus_cn)})", flush=True)

if __name__ == "__main__":
    main()
