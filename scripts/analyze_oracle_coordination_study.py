#!/usr/bin/env python3
"""Comprehensive analysis and reporting script for the Oracle Coordination Number Study."""
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


def compute_species_cn(struct: Structure, cnn: CrystalNN):
    cns_by_elem = defaultdict(list)
    for i, site in enumerate(struct):
        try:
            cn = cnn.get_cn(struct, i)
        except Exception:
            cn = 0
        cns_by_elem[site.specie.symbol].append(cn)
    return {elem: float(np.mean(vals)) for elem, vals in cns_by_elem.items()}


def compute_cn_mae(cn_dict_candidate, cn_dict_target):
    all_elems = set(cn_dict_target.keys())
    if not all_elems:
        return 999.0
    errs = [abs(cn_dict_candidate.get(e, 0.0) - cn_dict_target[e]) for e in all_elems]
    return float(np.mean(errs))


def main():
    repo_root = Path(__file__).resolve().parent.parent
    study_dir = repo_root / "generated" / "cryspr_oracle_coordination_study"
    tables_dir = study_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.read_csv(tables_dir / "results_per_structure.csv")
    with open(study_dir / "data" / "oracle_reconstruction_results.pkl", "rb") as f:
        data_reconstruction = pickle.load(f)

    # 1. Overall comparison headline
    with open(tables_dir / "comparison_headline.json") as f:
        headline = json.load(f)

    print("Headline:", headline)

    # 2. Breakdown by Crystal System
    cs_rows = []
    for cs, group in df_results.groupby("crystal_system"):
        n = len(group)
        cs_rows.append({
            "crystal_system": cs,
            "n_structures": n,
            "base5_rec": group["base5_recovered"].mean() * 100,
            "fixed5_rec": group["fixed_recovered"].mean() * 100,
            "relaxed_oracle5_rec": group["relaxed_recovered"].mean() * 100,
            "cn_oracle5_rec": group["cn_recovered"].mean() * 100,
            "cn_ceiling": group["cn_ceiling"].mean() * 100,
            "base10_rec": group["base10_recovered"].mean() * 100,
        })
    df_cs = pd.DataFrame(cs_rows).sort_values("n_structures", ascending=False)
    df_cs.to_csv(tables_dir / "breakdown_crystal_system.csv", index=False)
    print("\n--- Breakdown by Crystal System ---")
    print(df_cs.to_string(index=False))

    # 3. Breakdown by Total DOF
    def dof_cat(dof):
        if dof <= 6:
            return "6"
        elif dof <= 8:
            return "7-8"
        elif dof <= 11:
            return "9-11"
        else:
            return "12+"

    df_results["dof_cat"] = df_results["dof_total"].apply(dof_cat)
    dof_rows = []
    for cat in ["6", "7-8", "9-11", "12+"]:
        group = df_results[df_results["dof_cat"] == cat]
        if len(group) == 0:
            continue
        dof_rows.append({
            "dof_total_range": cat,
            "n_structures": len(group),
            "base5_rec": group["base5_recovered"].mean() * 100,
            "fixed5_rec": group["fixed_recovered"].mean() * 100,
            "relaxed_oracle5_rec": group["relaxed_recovered"].mean() * 100,
            "cn_oracle5_rec": group["cn_recovered"].mean() * 100,
            "cn_ceiling": group["cn_ceiling"].mean() * 100,
            "base10_rec": group["base10_recovered"].mean() * 100,
        })
    df_dof = pd.DataFrame(dof_rows)
    df_dof.to_csv(tables_dir / "breakdown_dof_total.csv", index=False)
    print("\n--- Breakdown by Total DOF ---")
    print(df_dof.to_string(index=False))

    # 4. Breakdown by Positional DOF
    def pos_dof_cat(dof):
        if dof <= 3:
            return "0-3"
        elif dof <= 6:
            return "4-6"
        elif dof <= 9:
            return "7-9"
        else:
            return "10+"

    df_results["pos_dof_cat"] = df_results["dof_pos"].apply(pos_dof_cat)
    pos_dof_rows = []
    for cat in ["0-3", "4-6", "7-9", "10+"]:
        group = df_results[df_results["pos_dof_cat"] == cat]
        if len(group) == 0:
            continue
        pos_dof_rows.append({
            "dof_pos_range": cat,
            "n_structures": len(group),
            "base5_rec": group["base5_recovered"].mean() * 100,
            "fixed5_rec": group["fixed_recovered"].mean() * 100,
            "relaxed_oracle5_rec": group["relaxed_recovered"].mean() * 100,
            "cn_oracle5_rec": group["cn_recovered"].mean() * 100,
            "cn_ceiling": group["cn_ceiling"].mean() * 100,
            "base10_rec": group["base10_recovered"].mean() * 100,
        })
    df_pos_dof = pd.DataFrame(pos_dof_rows)
    df_pos_dof.to_csv(tables_dir / "breakdown_dof_pos.csv", index=False)
    print("\n--- Breakdown by Positional DOF ---")
    print(df_pos_dof.to_string(index=False))

    # 5. Energy Analysis
    valid_base5 = df_results["base5_de_kept"].dropna()
    valid_cn = df_results["cn_de_kept"].dropna()

    energy_stats = {
        "base5_median_de": float(valid_base5.median()),
        "base5_mean_de": float(valid_base5.mean()),
        "base5_pct_lt_10meV": float((valid_base5 < 0.01).mean() * 100),
        "base5_pct_lt_50meV": float((valid_base5 < 0.05).mean() * 100),
        "base5_pct_lt_100meV": float((valid_base5 < 0.10).mean() * 100),
        "cn_median_de": float(valid_cn.median()),
        "cn_mean_de": float(valid_cn.mean()),
        "cn_pct_lt_10meV": float((valid_cn < 0.01).mean() * 100),
        "cn_pct_lt_50meV": float((valid_cn < 0.05).mean() * 100),
        "cn_pct_lt_100meV": float((valid_cn < 0.10).mean() * 100),
    }
    with open(tables_dir / "energy_stats.json", "w") as f:
        json.dump(energy_stats, f, indent=2)
    print("\n--- Energy Comparison ---")
    print(json.dumps(energy_stats, indent=2))

    # 6. Detailed Trial-Level CN Analysis
    print("\nAnalyzing all 2000 individual trials for CN and relaxation behavior...")
    cnn = CrystalNN()
    trial_records = []
    
    df_cohort = pd.read_parquet(repo_root / "generated/cryspr_oracle_fixed_lattice_study/data/sampled_400_targets.parquet")
    gt_cns = {}
    gt_structs = {}
    for _, row in df_cohort.iterrows():
        imm = row["immutable_id"]
        s = Structure.from_file(row["cif_path"])
        gt_structs[imm] = s
        gt_cns[imm] = compute_species_cn(s, cnn)

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0, primitive_cell=True)

    for imm_id, entry in data_reconstruction.items():
        s_idx = entry["struct_idx"]
        gt_s = gt_structs[imm_id]
        gt_cn = gt_cns[imm_id]
        e_target = df_results.loc[df_results["immutable_id"] == imm_id, "e_target"].values[0]

        for t in entry["trials"]:
            t_idx = t["trial_idx"]
            init_cn_err = t.get("init_cn_err", 999.0)
            kept_e_per_atom = t.get("kept_e_per_atom", 999.0)
            de = kept_e_per_atom - e_target

            kept_atoms = t.get("kept_atoms")
            is_match = False
            final_cn_err = 999.0
            if kept_atoms is not None:
                try:
                    pmg_s = Structure(
                        lattice=kept_atoms.get_cell(),
                        species=kept_atoms.get_chemical_symbols(),
                        coords=kept_atoms.get_positions(),
                        coords_are_cartesian=True,
                    )
                    is_match = bool(matcher.fit(pmg_s, gt_s))
                    final_cn = compute_species_cn(pmg_s, cnn)
                    final_cn_err = compute_cn_mae(final_cn, gt_cn)
                except Exception:
                    pass

            trial_records.append({
                "struct_idx": s_idx,
                "immutable_id": imm_id,
                "trial_idx": t_idx,
                "init_cn_err": init_cn_err,
                "final_cn_err": final_cn_err,
                "de_per_atom": de,
                "is_match": is_match,
                "accepted_s4": t.get("accepted_s4", False),
            })

    df_trials = pd.DataFrame(trial_records)
    df_trials.to_csv(tables_dir / "all_trials_cn_analysis.csv", index=False)

    print("\n--- Trial Statistics ---")
    print(f"Total evaluated trials: {len(df_trials)}")
    print(f"Mean Initial CN MAE: {df_trials['init_cn_err'].mean():.3f}")
    print(f"Median Initial CN MAE: {df_trials['init_cn_err'].median():.3f}")
    print(f"Trial Match Rate: {df_trials['is_match'].mean() * 100:.2f}%")

    # Match rate by initial CN error quartiles
    df_trials["init_cn_quartile"] = pd.qcut(df_trials["init_cn_err"], q=4, labels=["Q1 (lowest err)", "Q2", "Q3", "Q4 (highest err)"])
    q_stats = df_trials.groupby("init_cn_quartile", observed=False).agg(
        n_trials=("is_match", "count"),
        match_rate=("is_match", lambda x: x.mean() * 100),
        mean_de=("de_per_atom", "mean"),
        median_de=("de_per_atom", "median"),
        mean_final_cn_err=("final_cn_err", "mean"),
    ).reset_index()
    q_stats.to_csv(tables_dir / "cn_quartile_analysis.csv", index=False)
    print("\n--- Match Rate by Initial CN Error Quartile ---")
    print(q_stats.to_string(index=False))

    # Match rate for perfect initial CN vs non-zero
    zero_err = df_trials[df_trials["init_cn_err"] < 1e-4]
    nonzero_err = df_trials[df_trials["init_cn_err"] >= 1e-4]
    print(f"\nExact Initial CN Match (MAE=0) Trials: N={len(zero_err)}, Match Rate={zero_err['is_match'].mean()*100:.2f}%, Median dE={zero_err['de_per_atom'].median():.4f} eV")
    print(f"Inexact Initial CN Match (MAE>0) Trials: N={len(nonzero_err)}, Match Rate={nonzero_err['is_match'].mean()*100:.2f}%, Median dE={nonzero_err['de_per_atom'].median():.4f} eV")

    # Match rate for perfect FINAL CN vs non-zero
    final_zero = df_trials[df_trials["final_cn_err"] < 1e-4]
    final_nonzero = df_trials[df_trials["final_cn_err"] >= 1e-4]
    print(f"Exact Final CN Match (MAE=0) Trials: N={len(final_zero)}, Match Rate={final_zero['is_match'].mean()*100:.2f}%")
    print(f"Inexact Final CN Match (MAE>0) Trials: N={len(final_nonzero)}, Match Rate={final_nonzero['is_match'].mean()*100:.2f}%")

    # Comparison with Relaxed Oracle without CN guidance
    df_rel_trials = pd.read_csv(repo_root / "generated/cryspr_oracle_relaxed_cell_study/tables/oracle_cn_trials_analysis.csv")
    print("\n--- Comparison: CN Guidance vs Unfiltered Relaxed Cell Oracle ---")
    print(f"Unfiltered Initial CN MAE: {df_rel_trials['init_mae'].mean():.3f} -> CN-Guided Initial CN MAE: {df_trials['init_cn_err'].mean():.3f} (Reduction: {df_rel_trials['init_mae'].mean() - df_trials['init_cn_err'].mean():.3f})")
    print(f"Unfiltered Match Rate: {df_rel_trials['recovered'].mean()*100:.2f}% -> CN-Guided Match Rate: {df_trials['is_match'].mean()*100:.2f}%")
    print(f"Unfiltered Median dE: {df_rel_trials['delta_e'].median():.4f} eV -> CN-Guided Median dE: {df_trials['de_per_atom'].median():.4f} eV")


    print("\nAll analysis tables written to:", tables_dir)


if __name__ == "__main__":
    main()
