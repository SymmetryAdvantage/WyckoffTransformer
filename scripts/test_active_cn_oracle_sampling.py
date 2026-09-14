#!/usr/bin/env python3
"""Active Coordination Number Oracle Sampling Experiment.

For structures that were MISSED by standard 5-trial CRySPR:
1. Compute Oracle Coordination Numbers from ground truth.
2. Rapidly sample N=30 PyXtal candidates (permissive f=1.0).
3. Score each candidate by initial CN error against the Oracle.
4. Relax the best CN-matching candidate with ORB.
5. Compare against:
   - Baseline CRySPR (which missed)
   - Random candidate relaxation
"""
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pyxtal import pyxtal
from pyxtal.lattice import Lattice as PxtLattice
from pyxtal.tolerance import Tol_matrix

warnings.filterwarnings("ignore")

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "scripts"))
from run_cryspr_reconstruction_study import build_patched_orb_calculator

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

def relax_candidate(atoms, calc, fmax=0.03, max_steps=300):
    atoms = atoms.copy()
    atoms.calc = calc
    filt = FrechetCellFilter(atoms)
    opt = BFGS(filt, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)
    return atoms, atoms.get_potential_energy() / len(atoms)

def main():
    print("Loading ORB model and target cohort...", flush=True)
    calc = build_patched_orb_calculator(device="cpu")
    cnn = CrystalNN()
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True)
    
    df_targets = pd.read_parquet(_repo_root / "generated/cryspr_oracle_relaxed_cell_study/data/sampled_400_targets.parquet")
    df_results = pd.read_csv(_repo_root / "generated/cryspr_oracle_relaxed_cell_study/tables/results_per_structure.csv")
    
    # Pick 5 missed structures with varying degrees of freedom and crystal systems
    missed = df_results[(df_results["oracle_verdict"] == "missed") & (df_results["base5_verdict"] == "missed")]
    test_indices = [6, 17, 24, 30, 42]  # Variety of spacegroups and stoichiometries
    
    print("\n" + "="*70, flush=True)
    print("ACTIVE ORACLE COORDINATION NUMBER GUIDED GENERATION EXPERIMENT", flush=True)
    print("="*70, flush=True)
    
    results = []
    
    for s_idx in test_indices:
        row = df_targets[df_targets["struct_idx"] == s_idx].iloc[0]
        imm_id = row["immutable_id"]
        tgt_struct = Structure.from_file(_repo_root / row["cif_path"])
        tgt_e = row["orb_e_target_per_atom"]
        gene = row["wyckoff_gene"]
        formula = row["chemical_formula_reduced"]
        spg = row["spacegroup"]
        
        gt_cn = compute_species_cn(tgt_struct, cnn)
        print(f"\nEvaluating Struct {s_idx} ({formula}, SG {spg}, {len(tgt_struct)} atoms):", flush=True)
        print(f"  Target ORB Energy: {tgt_e:.4f} eV/atom", flush=True)
        print(f"  Oracle Ground-Truth Coordination Numbers: {gt_cn}", flush=True)
        
        # Get PyXtal lattice from conventional standard structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(tgt_struct, symprec=0.1)
        std_struct = sga.get_conventional_standard_structure()
        pxt_lat = PxtLattice.from_para(
            std_struct.lattice.a, std_struct.lattice.b, std_struct.lattice.c,
            std_struct.lattice.alpha, std_struct.lattice.beta, std_struct.lattice.gamma,
            ltype=row["crystal_system"], force_symmetry=True
        )
        
        # Sample N=30 candidate structures rapidly using PyXtal
        t0 = time.time()
        candidates = []
        tol_mat = Tol_matrix(prototype="atomic", factor=1.0)
        
        for seed in range(30):
            np.random.seed(seed * 100 + s_idx)
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
                    max_count=5,
                )
                if cand.valid:
                    pmg_cand = cand.to_pymatgen()
                    cn_cand = compute_species_cn(pmg_cand, cnn)
                    err = compute_cn_mae(cn_cand, gt_cn)
                    candidates.append({
                        "seed": seed,
                        "atoms": cand.to_ase(),
                        "pmg": pmg_cand,
                        "cn": cn_cand,
                        "cn_error": err,
                    })
            except Exception:
                continue
                
        dt_sample = time.time() - t0
        print(f"  Generated {len(candidates)}/30 valid PyXtal candidates in {dt_sample:.2f}s", flush=True)
        
        if not candidates:
            print("  Failed to generate candidates!", flush=True)
            continue
            
        candidates.sort(key=lambda c: c["cn_error"])
        
        best_cand = candidates[0]
        worst_cand = candidates[-1]
        random_cand = candidates[len(candidates)//2]
        
        print(f"  Best Candidate CN Error:  {best_cand['cn_error']:.3f} -> CN={best_cand['cn']}", flush=True)
        print(f"  Random Candidate CN Error: {random_cand['cn_error']:.3f} -> CN={random_cand['cn']}", flush=True)
        print(f"  Worst Candidate CN Error: {worst_cand['cn_error']:.3f} -> CN={worst_cand['cn']}", flush=True)
        
        # Relax Best Candidate
        t0 = time.time()
        rel_best_atoms, e_best = relax_candidate(best_cand["atoms"], calc)
        rel_best_pmg = Structure(rel_best_atoms.cell.array, rel_best_atoms.get_chemical_symbols(), rel_best_atoms.positions, coords_are_cartesian=True)
        best_recovered = matcher.fit(rel_best_pmg, tgt_struct)
        t_best = time.time() - t0
        print(f"  [ORACLE CN BEST TRIAL]  Final E = {e_best:.4f} eV/atom (dE = {e_best - tgt_e:+.4f}) | Recovered = {best_recovered} ({t_best:.1f}s)", flush=True)
        
        # Relax Random Candidate
        t0 = time.time()
        rel_rand_atoms, e_rand = relax_candidate(random_cand["atoms"], calc)
        rel_rand_pmg = Structure(rel_rand_atoms.cell.array, rel_rand_atoms.get_chemical_symbols(), rel_rand_atoms.positions, coords_are_cartesian=True)
        rand_recovered = matcher.fit(rel_rand_pmg, tgt_struct)
        t_rand = time.time() - t0
        print(f"  [RANDOM BASELINE TRIAL] Final E = {e_rand:.4f} eV/atom (dE = {e_rand - tgt_e:+.4f}) | Recovered = {rand_recovered} ({t_rand:.1f}s)", flush=True)
        
        results.append({
            "struct_idx": s_idx,
            "formula": formula,
            "best_cn_error": best_cand["cn_error"],
            "rand_cn_error": random_cand["cn_error"],
            "e_best": e_best,
            "e_rand": e_rand,
            "e_tgt": tgt_e,
            "de_best": e_best - tgt_e,
            "de_rand": e_rand - tgt_e,
            "recovered_best": best_recovered,
            "recovered_rand": rand_recovered,
        })
        
    df_res = pd.DataFrame(results)
    print("\n" + "="*70, flush=True)
    print("SUMMARY COMPARISON ACROSS MISSED STRUCTURES", flush=True)
    print("="*70, flush=True)
    print(df_res[["struct_idx", "formula", "best_cn_error", "de_best", "recovered_best", "rand_cn_error", "de_rand", "recovered_rand"]].to_string(), flush=True)

if __name__ == "__main__":
    main()
