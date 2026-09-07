#!/usr/bin/env python3
"""Run the CrySPR reconstruction fidelity study across all available GPUs.

Protocol specification: docs/cryspr_reconstruction_study.md
"""
import argparse
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import os
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

import numpy as np
import pandas as pd
import spglib
from ase import Atoms
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import write as ase_write
from ase.optimize import BFGS
from huggingface_hub import hf_hub_download
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, Lattice, Structure
from pyxtal import pyxtal
from pyxtal.tolerance import Tol_matrix

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(processName)s) %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cryspr_study")

# Single-threaded BLAS per worker to prevent CPU contention across multi-GPU processes
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


def apply_runtime_patches():
    """Apply Kepler/Maxwell PyTorch & Warp workarounds for Driver 470."""
    import torch
    import warp as wp

    wp.init()

    # Disable JIT fusers so NVRTC runtime compiler is not invoked
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)

    # CPU linalg patches (fallback to NumPy for missing LAPACK on CPU)
    orig_solve = torch.linalg.solve

    def safe_solve(A, B, *args, **kwargs):
        if not A.is_cuda:
            res = np.linalg.solve(A.detach().numpy(), B.detach().numpy())
            return torch.from_numpy(res).to(dtype=B.dtype)
        return orig_solve(A, B, *args, **kwargs)

    torch.linalg.solve = safe_solve

    orig_inv = torch.linalg.inv

    def safe_inv(A, *args, **kwargs):
        if not A.is_cuda:
            res = np.linalg.inv(A.detach().numpy())
            return torch.from_numpy(res).to(dtype=A.dtype)
        return orig_inv(A, *args, **kwargs)

    torch.linalg.inv = safe_inv

    _orig_det = torch.linalg.det

    def safe_det(A, *args, **kwargs):
        if A.shape[-2:] == (3, 3):
            return (
                A[..., 0, 0]
                * (A[..., 1, 1] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 1])
                - A[..., 0, 1]
                * (A[..., 1, 0] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 0])
                + A[..., 0, 2]
                * (A[..., 1, 0] * A[..., 2, 1] - A[..., 1, 1] * A[..., 2, 0])
            )
        return _orig_det(A, *args, **kwargs)

    torch.linalg.det = safe_det


def build_patched_orb_calculator(device: str):
    """Build ORB calculator with CPU neighbor listing and GPU forward evaluation."""
    from wyckoff_transformer.evaluation.hull_mlips import build_hull_calculator

    calc = build_hull_calculator("orb_conserv_inf", device=device)

    def patched_calculate(atoms=None, properties=None, system_changes=None):
        from ase.calculators.calculator import Calculator

        Calculator.calculate(calc, atoms)
        # Warp neighbor search runs on CPU
        batch = calc.adapter.from_ase_atoms(
            atoms=atoms,
            max_num_neighbors=calc.max_num_neighbors,
            edge_method=calc.edge_method,
            half_supercell=calc.half_supercell,
            device="cpu",
        )
        batch = batch.to(calc.device)
        out = calc.model.predict(batch)
        calc._update_results(out)

    calc.calculate = patched_calculate
    return calc


def get_crystal_system_and_lattice_dof(spacegroup_number: int) -> Tuple[str, int]:
    """Map space group number (1..230) to crystal system and free lattice DOF."""
    if 1 <= spacegroup_number <= 2:
        return "triclinic", 6
    elif 3 <= spacegroup_number <= 15:
        return "monoclinic", 4
    elif 16 <= spacegroup_number <= 74:
        return "orthorhombic", 3
    elif 75 <= spacegroup_number <= 142:
        return "tetragonal", 2
    elif 143 <= spacegroup_number <= 167:
        return "trigonal", 2
    elif 168 <= spacegroup_number <= 194:
        return "hexagonal", 2
    elif 195 <= spacegroup_number <= 230:
        return "cubic", 1
    else:
        return "unknown", 0


def kick_pyxtal_until_it_works_with_attempt(
    structure: Structure,
    tol: float = 0.1,
    a_tol: float = 5.0,
    attempts: int = 30,
) -> Tuple[pyxtal, int]:
    """Retry PyXtal conversion across tolerance ladder, returning (pyxtal_obj, attempt_idx)."""
    n_down_multipliers = attempts // 2
    tolerances = np.empty(attempts)
    tolerances[::2] = np.logspace(0, 2, attempts - n_down_multipliers)
    tolerances[1::2] = np.logspace(-0.01, -6, n_down_multipliers)

    for attempt, tolerance in enumerate(tolerances):
        try:
            pxt = pyxtal()
            pxt.from_seed(structure, tol=tol * tolerance, a_tol=a_tol)
            if len(pxt.atom_sites) > 0:
                return pxt, attempt
        except Exception:
            continue
    raise RuntimeError("Failed to make PyXtal work after all attempts.")


# ---------------------------------------------------------------------------
# Step 1: Sample 1000 structures from LeMat-Bulk with e_above_hull <= 0.1
# ---------------------------------------------------------------------------
def step1_sample_targets(output_dir: Path, seed: int = 42, n_samples: int = 1000):
    output_dir.mkdir(parents=True, exist_ok=True)
    targets_file = output_dir / "data" / f"sampled_{n_samples}_targets_raw.parquet"
    if targets_file.exists():
        logger.info("Step 1: Found existing sampled targets: %s", targets_file)
        return pd.read_parquet(targets_file)

    logger.info("Step 1: Sampling %d structures from LeMat-Bulk MLIP hull (seed=%d)...", n_samples, seed)
    hull_p = hf_hub_download(
        repo_id="LeMaterial/LeMat-Bulk-MLIP-Hull",
        filename="data/orb_conserv_inf-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df_hull = pd.read_parquet(hull_p)
    eligible = df_hull[df_hull["e_above_hull"] <= 0.1]
    logger.info("Found %d eligible structures (e_above_hull <= 0.1)", len(eligible))
    sampled = eligible.sample(n=n_samples, random_state=seed)
    sampled_ids = set(sampled["immutable_id"])

    # Extract geometry across the 17 shards of compatible_pbe
    logger.info("Extracting atomic geometries from LeMat-Bulk shards...")
    found_rows = []
    for shard_idx in range(17):
        filename = f"compatible_pbe/train-{shard_idx:05d}-of-00017.parquet"
        shard_path = hf_hub_download(repo_id="LeMaterial/LeMat-Bulk", filename=filename, repo_type="dataset")
        cols = [
            "immutable_id",
            "lattice_vectors",
            "cartesian_site_positions",
            "species_at_sites",
            "nsites",
            "chemical_formula_reduced",
        ]
        df_shard = pd.read_parquet(shard_path, columns=cols)
        matched = df_shard[df_shard["immutable_id"].isin(sampled_ids)]
        found_rows.append(matched)
        logger.info("Shard %d: matched %d rows (total: %d/%d)", shard_idx, len(matched), sum(len(m) for m in found_rows), n_samples)
        if sum(len(m) for m in found_rows) == n_samples:
            break

    df_all_sampled = pd.concat(found_rows, ignore_index=True)
    # Merge hull metadata
    df_final = df_all_sampled.merge(
        sampled[["immutable_id", "e_above_hull", "energy", "orb_conserv_inf_energy"]],
        on="immutable_id",
        how="left",
    )

    # Compute raw space groups via spglib
    raw_sgs = []
    for _, row in df_final.iterrows():
        cell = (np.array(row["lattice_vectors"].tolist()), np.array(row["cartesian_site_positions"].tolist()))
        # Convert species strings to atomic numbers
        nums = [Element(sp).number for sp in row["species_at_sites"]]
        # Fractional positions
        inv_lat = np.linalg.inv(cell[0])
        frac_pos = np.dot(cell[1], inv_lat)
        dataset = spglib.get_symmetry_dataset((cell[0], frac_pos, nums), symprec=1e-3)
        sg_num = dataset.number if dataset is not None else 0
        raw_sgs.append(sg_num)
    df_final["spacegroup_raw"] = raw_sgs

    df_final.to_parquet(targets_file, index=False)
    logger.info("Saved %d sampled target structures to %s", len(df_final), targets_file)

    # Save distribution summary
    dist_summary = {
        "n_samples": len(df_final),
        "nsites_quantiles": df_final["nsites"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
        "nsites_mean": float(df_final["nsites"].mean()),
        "nsites_min": int(df_final["nsites"].min()),
        "nsites_max": int(df_final["nsites"].max()),
        "top_spacegroups": df_final["spacegroup_raw"].value_counts().head(20).to_dict(),
    }
    with open(output_dir / "data" / "sample_draw_distributions.json", "w") as f:
        json.dump(dist_summary, f, indent=2)
    logger.info("Recorded nsites and spacegroup distributions of the draw.")
    return df_final


# ---------------------------------------------------------------------------
# Step 2: Relax Targets with ORB across all available GPUs
# ---------------------------------------------------------------------------
def _target_relaxation_worker(
    gpu_id: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
):
    """Worker process relaxing targets on an assigned GPU."""
    try:
        apply_runtime_patches()
        calc = build_patched_orb_calculator(gpu_id)
        import torch

        while True:
            task = task_queue.get()
            if task is None:
                break
            idx, immutable_id, lat, coords, species, output_cif_path = task
            lat = np.array(lat)
            coords = np.array(coords)
            atoms = Atoms(symbols=species, positions=coords, cell=lat, pbc=True)
            atoms.calc = calc

            filt = FrechetCellFilter(atoms)
            opt = BFGS(filt, logfile=None)
            opt.run(fmax=0.02, steps=500)

            final_e = float(atoms.get_potential_energy())
            final_e_per_atom = final_e / len(atoms)

            ase_write(str(output_cif_path), atoms, format="cif")

            torch.cuda.empty_cache()
            result_queue.put(
                (
                    idx,
                    immutable_id,
                    final_e,
                    final_e_per_atom,
                    atoms.cell.array.tolist(),
                    atoms.positions.tolist(),
                    atoms.get_chemical_symbols(),
                )
            )
    except Exception as exc:
        logger.exception("Error in target relaxation worker on %s: %s", gpu_id, exc)
        result_queue.put(("ERROR", str(exc)))


def step2_relax_targets(df_targets: pd.DataFrame, output_dir: Path, devices: List[str]):
    """Relax all sampled target structures using ORB on all specified GPUs."""
    output_dir = Path(output_dir)
    relaxed_file = output_dir / "data" / "relaxed_targets.parquet"
    relaxed_file.parent.mkdir(parents=True, exist_ok=True)
    if relaxed_file.exists():
        logger.info("Step 2: Found existing relaxed targets: %s", relaxed_file)
        return pd.read_parquet(relaxed_file)

    logger.info("Step 2: Relaxing %d target structures on ORB PES across %s...", len(df_targets), devices)
    cif_dir = output_dir / "targets"
    cif_dir.mkdir(parents=True, exist_ok=True)

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    tasks = []
    for idx, row in df_targets.iterrows():
        out_cif = cif_dir / f"{row['immutable_id']}_orb_relaxed.cif"
        lat = np.array(row["lattice_vectors"].tolist())
        coords = np.array(row["cartesian_site_positions"].tolist())
        tasks.append((idx, row["immutable_id"], lat.tolist(), coords.tolist(), list(row["species_at_sites"]), out_cif))

    for t in tasks:
        task_queue.put(t)
    for _ in devices:
        task_queue.put(None)  # Termination sentinel

    processes = []
    for dev in devices:
        p = mp.Process(target=_target_relaxation_worker, args=(dev, task_queue, result_queue), name=f"Worker-{dev}")
        p.start()
        processes.append(p)

    results = []
    n_done = 0
    t0 = time.time()
    while n_done < len(tasks):
        item = result_queue.get()
        if item[0] == "ERROR":
            for p in processes:
                p.terminate()
            raise RuntimeError(f"Target relaxation failed: {item[1]}")
        results.append(item)
        n_done += 1
        if n_done % 50 == 0 or n_done == len(tasks):
            elapsed = time.time() - t0
            rate = n_done / elapsed
            logger.info("Relaxed %d/%d targets (%.2f targets/sec, elapsed: %.1fs)", n_done, len(tasks), rate, elapsed)

    for p in processes:
        p.join()

    res_dict = {r[0]: r for r in results}
    rows = []
    for idx, row in df_targets.iterrows():
        r = res_dict[idx]
        rows.append(
            {
                "immutable_id": row["immutable_id"],
                "nsites": row["nsites"],
                "chemical_formula_reduced": row["chemical_formula_reduced"],
                "e_above_hull": row["e_above_hull"],
                "orb_e_target": r[2],
                "orb_e_target_per_atom": r[3],
                "relaxed_lattice": r[4],
                "relaxed_positions": r[5],
                "relaxed_species": r[6],
                "cif_path": str(cif_dir / f"{row['immutable_id']}_orb_relaxed.cif"),
            }
        )

    df_relaxed = pd.DataFrame(rows)
    df_relaxed.to_parquet(relaxed_file, index=False)
    logger.info("Saved %d relaxed targets to %s", len(df_relaxed), relaxed_file)
    return df_relaxed


# ---------------------------------------------------------------------------
# Step 3: Extract Wyckoff Genes and Deduplicate
# ---------------------------------------------------------------------------
def step3_extract_and_deduplicate_genes(df_relaxed_targets: pd.DataFrame, output_dir: Path):
    """Extract Wyckoff genes with kick_pyxtal_until_it_works and deduplicate."""
    output_dir = Path(output_dir)
    genes_file = output_dir / "data" / "unique_wyckoff_genes.json.gz"
    genes_file.parent.mkdir(parents=True, exist_ok=True)
    if genes_file.exists():
        logger.info("Step 3: Found existing deduplicated genes: %s", genes_file)
        with gzip.open(genes_file, "rt") as f:
            return json.load(f)

    logger.info("Step 3: Extracting Wyckoff genes for %d relaxed structures...", len(df_relaxed_targets))

    records = []
    for idx, row in df_relaxed_targets.iterrows():
        try:
            struct = Structure.from_file(row["cif_path"])
        except Exception:
            lat = Lattice(np.array(row["relaxed_lattice"].tolist(), dtype=float))
            struct = Structure(
                lat,
                list(row["relaxed_species"]),
                np.array(row["relaxed_positions"].tolist(), dtype=float),
                coords_are_cartesian=True,
            )

        try:
            pxt, attempt = kick_pyxtal_until_it_works_with_attempt(struct, tol=0.1, a_tol=5.0)
        except Exception as exc:
            logger.warning("Gene extraction failed for %s: %s", row["immutable_id"], exc)
            continue

        sites_by_species: Dict[str, List[str]] = defaultdict(list)
        for site in pxt.atom_sites:
            sp = str(site.specie)
            sites_by_species[sp].append(f"{site.wp.multiplicity}{site.wp.letter}")

        species = sorted(list(sites_by_species))
        num_ions = [sum(int(s[:-1]) for s in sites_by_species[sp]) for sp in species]
        sites = [sorted(sites_by_species[sp]) for sp in species]
        spg = pxt.group.number
        crystal_system, lat_dof = get_crystal_system_and_lattice_dof(spg)
        dof_pos = sum(site.wp.get_dof() for site in pxt.atom_sites)
        dof_tot = dof_pos + lat_dof

        gene_dict = {
            "group": spg,
            "species": species,
            "numIons": num_ions,
            "sites": sites,
        }
        # Canonical tuple for deduplication
        canonical_key = (spg, tuple((sp, tuple(sites_by_species[sp])) for sp in species))

        records.append(
            {
                "immutable_id": row["immutable_id"],
                "target_energy_per_atom": row["orb_e_target_per_atom"],
                "relaxed_cif": row["cif_path"],
                "wyckoff_gene": gene_dict,
                "canonical_key": canonical_key,
                "attempt": attempt,
                "dof_positional": dof_pos,
                "dof_total": dof_tot,
                "crystal_system": crystal_system,
                "spacegroup": spg,
                "nsites": len(struct),
                "n_wyckoff_sites": len(pxt.atom_sites),
            }
        )

    # Deduplicate genes
    gene_groups: Dict[Any, List[Dict]] = defaultdict(list)
    for rec in records:
        gene_groups[rec["canonical_key"]].append(rec)

    unique_genes = []
    for gid, (key, member_list) in enumerate(gene_groups.items()):
        first = member_list[0]
        # Target set energies
        target_energies = [m["target_energy_per_atom"] for m in member_list]
        min_e_target = min(target_energies)
        target_cifs = [m["relaxed_cif"] for m in member_list]
        min_attempt = min(m["attempt"] for m in member_list)

        unique_genes.append(
            {
                "gene_id": gid,
                "gene": first["wyckoff_gene"],
                "n_targets": len(member_list),
                "target_ids": [m["immutable_id"] for m in member_list],
                "target_cifs": target_cifs,
                "e_target": min_e_target,
                "symmetry_tol_attempt": min_attempt,
                "dof_positional": first["dof_positional"],
                "dof_total": first["dof_total"],
                "crystal_system": first["crystal_system"],
                "spacegroup": first["spacegroup"],
                "nsites": first["nsites"],
                "n_wyckoff_sites": first["n_wyckoff_sites"],
            }
        )

    logger.info(
        "Step 3 complete: %d target structures reduced to %d unique Wyckoff genes (deduplication ratio: %.2f)",
        len(records),
        len(unique_genes),
        len(records) / max(1, len(unique_genes)),
    )

    with gzip.open(genes_file, "wt") as f:
        json.dump(unique_genes, f, indent=2, default=_json_default)
    return unique_genes


# ---------------------------------------------------------------------------
# Step 4: CrySPR 4-stage Reconstruction Worker & Orchestrator
# ---------------------------------------------------------------------------
def _deterministic_seed(gene_id: int, trial_idx: int) -> int:
    h = hashlib.sha256(f"{gene_id}_{trial_idx}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


class PyXtalTimeout(Exception):
    """Raised when PyXtal crystal generation exceeds allowed time limit."""
    pass


def _alarm_handler(signum, frame):
    raise PyXtalTimeout("PyXtal candidate generation timed out")


def _reconstruct_single_trial(
    gene: Dict,
    gene_id: int,
    trial_idx: int,
    trial_dir: Path,
    calc,
    fmax: float = 0.02,
    steps_limit: int = 500,
) -> Optional[Dict]:
    """Execute 4-stage reconstruction for one trial of a gene."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    tol_mat = Tol_matrix(prototype="atomic", factor=1.3)

    # Deterministic generation
    seed = _deterministic_seed(gene_id, trial_idx)
    np.random.seed(seed)

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(10)  # 10 second timeout for PyXtal candidate generation
    try:
        cand = pyxtal()
        cand.from_random(
            dim=3,
            group=gene["group"],
            species=gene["species"],
            numIons=gene["numIons"],
            sites=gene["sites"],
            tm=tol_mat,
            max_count=10,
        )
        atoms = cand.to_ase()
    except PyXtalTimeout:
        logger.warning(
            "Gene %d trial %d: PyXtal generation timed out (>10s), marking trial as failed",
            gene_id,
            trial_idx,
        )
        return None
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if atoms is None:
        return None

    formula = atoms.get_chemical_formula(mode="metal")
    ase_write(str(trial_dir / f"{formula}_0_initial.cif"), atoms, format="cif")
    atoms.calc = calc

    try:
        # Stage 1: Fix cell, symmetry constrained
        atoms_s1 = atoms.copy()
        atoms_s1.calc = calc
        atoms_s1.set_constraint([FixSymmetry(atoms_s1, symprec=1e-3)])
        opt1 = BFGS(atoms_s1, logfile=str(trial_dir / f"{formula}_1_fix-cell.log"))
        opt1.run(fmax=fmax, steps=steps_limit)
        ase_write(str(trial_dir / f"{formula}_1_fix-cell.cif"), atoms_s1, format="cif")

        # Stage 2: Cell + pos, symmetry constrained
        atoms_s2 = atoms_s1.copy()
        atoms_s2.calc = calc
        atoms_s2.set_constraint([FixSymmetry(atoms_s2, symprec=1e-3)])
        filt2 = FrechetCellFilter(atoms_s2)
        opt2 = BFGS(filt2, logfile=str(trial_dir / f"{formula}_2_sym_cell+pos.log"))
        opt2.run(fmax=fmax, steps=steps_limit)
        ase_write(str(trial_dir / f"{formula}_2_sym_cell+pos.cif"), atoms_s2, format="cif")

        # Stage 3: Cell + pos, symmetry RELEASED
        atoms_s3 = atoms_s2.copy()
        atoms_s3.calc = calc
        atoms_s3.set_constraint([])  # Release symmetry
        filt3 = FrechetCellFilter(atoms_s3)
        opt3 = BFGS(filt3, logfile=str(trial_dir / f"{formula}_3_no-sym_cell+pos.log"))
        opt3.run(fmax=fmax, steps=steps_limit)
        e_s3 = float(atoms_s3.get_potential_energy())
        cif_s3_path = trial_dir / f"{formula}_3_no-sym_cell+pos.cif"
        ase_write(str(cif_s3_path), atoms_s3, format="cif")

        # Stage 4: Rattle + strain + re-relax
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

        # Acceptance rule: keep stage 4 only if it lowers E/atom by > 1 meV/atom
        n_atoms = len(atoms)
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
        logger.warning(
            "Gene %d trial %d: CUDA OutOfMemoryError during relaxation, freeing cache and skipping trial",
            gene_id,
            trial_idx,
        )
        import torch
        torch.cuda.empty_cache()
        return None
    except Exception as exc:
        logger.warning("Gene %d trial %d: relaxation error (%s), skipping trial", gene_id, trial_idx, exc)
        return None


def _reconstruction_worker(
    gpu_id: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    output_dir: Path,
    n_trials: int = 10,
):
    """Worker handling CrySPR reconstruction across gene tasks on one GPU."""
    try:
        apply_runtime_patches()
        calc = build_patched_orb_calculator(gpu_id)
        import torch

        while True:
            task = task_queue.get()
            if task is None:
                break
            gene_info = task
            gid = gene_info["gene_id"]
            gene = gene_info["gene"]
            gene_dir = output_dir / "cryspr" / str(gid)
            gene_dir.mkdir(parents=True, exist_ok=True)
            n_atoms = sum(gene.get("numIons", []))

            logger.info(
                "[%s] Worker starting Gene %d (sg=%d, sites=%d, atoms=%d, dof=%d)",
                gpu_id,
                gid,
                gene.get("group", 0),
                len(gene.get("sites", [])),
                n_atoms,
                gene_info.get("dof_total", 0),
            )

            trial_results = []
            n_generated = 0

            for t_idx in range(n_trials):
                t_dir = gene_dir / f"trial-{t_idx}"
                try:
                    res = _reconstruct_single_trial(
                        gene=gene,
                        gene_id=gid,
                        trial_idx=t_idx,
                        trial_dir=t_dir,
                        calc=calc,
                        fmax=0.02,
                        steps_limit=500,
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning("[%s] Gene %d trial %d hit CUDA OOM, skipping trial", gpu_id, gid, t_idx)
                    torch.cuda.empty_cache()
                    res = None
                except Exception as exc:
                    logger.warning("[%s] Gene %d trial %d error: %s", gpu_id, gid, t_idx, exc)
                    res = None

                if res is not None:
                    n_generated += 1
                    # Sanity guard: reject trial if |E/atom| > 50 eV/atom
                    if abs(res["kept_e_per_atom"]) <= 50.0:
                        trial_results.append(res)
                torch.cuda.empty_cache()

            logger.info(
                "[%s] Worker finished Gene %d: %d/%d trials valid (%d generated)",
                gpu_id,
                gid,
                len(trial_results),
                n_trials,
                n_generated,
            )
            result_queue.put((gid, n_generated, trial_results))
    except Exception as exc:
        logger.exception("Error in reconstruction worker on %s: %s", gpu_id, exc)
        result_queue.put(("ERROR", str(exc)))


def step4_reconstruct_genes(
    unique_genes: List[Dict],
    output_dir: Path,
    devices: List[str],
    n_trials: int = 10,
):
    """Run CrySPR 4-stage reconstruction over all unique genes across all GPUs."""
    output_dir = Path(output_dir)
    recon_cache = output_dir / "data" / "reconstruction_results.pkl"
    recon_cache.parent.mkdir(parents=True, exist_ok=True)
    import pickle

    results_by_gid = {}
    if recon_cache.exists():
        with open(recon_cache, "rb") as f:
            try:
                results_by_gid = pickle.load(f)
                logger.info("Loaded %d previously reconstructed genes from cache", len(results_by_gid))
            except Exception as exc:
                logger.warning("Could not read existing recon cache: %s", exc)

    genes_to_run = [g for g in unique_genes if g["gene_id"] not in results_by_gid]
    if not genes_to_run:
        logger.info("All %d requested genes already completed in cache!", len(unique_genes))
        return results_by_gid

    logger.info(
        "Step 4: Reconstructing %d genes (%d already in cache, %d to run, %d trials/gene, 4 stages) across %s...",
        len(unique_genes),
        len(results_by_gid),
        len(genes_to_run),
        n_trials,
        devices,
    )

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    for g in genes_to_run:
        task_queue.put(g)
    for _ in devices:
        task_queue.put(None)

    processes = []
    for dev in devices:
        p = mp.Process(
            target=_reconstruction_worker,
            args=(dev, task_queue, result_queue, output_dir, n_trials),
            name=f"ReconWorker-{dev}",
        )
        p.start()
        processes.append(p)

    n_done = 0
    t0 = time.time()

    while n_done < len(genes_to_run):
        item = result_queue.get()
        if item[0] == "ERROR":
            for p in processes:
                p.terminate()
            raise RuntimeError(f"Reconstruction worker failed: {item[1]}")
        gid, n_gen, trials = item
        results_by_gid[gid] = {"n_trials_generated": n_gen, "trials": trials}
        n_done += 1

        # Save checkpoint after every gene
        with open(recon_cache, "wb") as f:
            pickle.dump(results_by_gid, f)
        elapsed = time.time() - t0
        rate = n_done / max(elapsed, 1e-5)
        rem = (len(genes_to_run) - n_done) / max(rate, 1e-5)
        logger.info(
            "Completed gene %d: %d/%d new done (total: %d/%d, %.3f genes/s, elapsed: %.1fs, eta: %.1fs)",
            gid,
            n_done,
            len(genes_to_run),
            len(results_by_gid),
            len(unique_genes),
            rate,
            elapsed,
            rem,
        )

    for p in processes:
        p.join()

    import pickle

    with open(recon_cache, "wb") as f:
        pickle.dump(results_by_gid, f)
    logger.info("Saved reconstruction results to %s", recon_cache)
    return results_by_gid


def _score_single_gene(task: Tuple[Dict, Dict]) -> Dict:
    g, res = task
    matcher_default = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True)
    matcher_loose = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True, scale=True)
    gid = g["gene_id"]
    trials = res.get("trials", [])
    n_generated = res.get("n_trials_generated", 0)
    n_relaxed = len(trials)

    # Load target structures
    target_structures = []
    for cpath in g["target_cifs"]:
        try:
            target_structures.append(Structure.from_file(cpath))
        except Exception:
            pass

    if not trials or not target_structures:
        return {
            "gene_id": gid,
            "n_targets": g["n_targets"],
            "spacegroup": g["spacegroup"],
            "crystal_system": g["crystal_system"],
            "nsites": g["nsites"],
            "n_wyckoff_sites": g["n_wyckoff_sites"],
            "dof_positional": g["dof_positional"],
            "dof_total": g["dof_total"],
            "symmetry_tol_attempt": g["symmetry_tol_attempt"],
            "n_trials_generated": n_generated,
            "n_trials_relaxed": n_relaxed,
            "n_matching_trials_s3": 0,
            "n_matching_trials_s4": 0,
            "n_matching_trials_loose": 0,
            "matched_kept": False,
            "matched_any": False,
            "matched_loose_kept": False,
            "matched_loose_any": False,
            "e_kept": np.nan,
            "e_target": g["e_target"],
            "de_kept": np.nan,
            "verdict": "generation_failed",
            "best_volume_ratio": np.nan,
        }

    # Sort trials by kept energy
    trials_sorted = sorted(trials, key=lambda t: t["kept_e_per_atom"])
    kept_trial = trials_sorted[0]
    e_kept = kept_trial["kept_e_per_atom"]
    de_kept = e_kept - g["e_target"]

    # Match evaluations per trial
    matching_s3_count = 0
    matching_s4_count = 0
    matching_loose_count = 0
    matched_any = False
    matched_loose_any = False
    vol_ratios = []

    for t in trials:
        s3_pmg = Structure(
            Lattice(t["atoms_s3"].cell.array),
            t["atoms_s3"].get_chemical_symbols(),
            t["atoms_s3"].positions,
            coords_are_cartesian=True,
        )
        s4_pmg = Structure(
            Lattice(t["atoms_s4"].cell.array),
            t["atoms_s4"].get_chemical_symbols(),
            t["atoms_s4"].positions,
            coords_are_cartesian=True,
        )
        kept_pmg = Structure(
            Lattice(t["kept_atoms"].cell.array),
            t["kept_atoms"].get_chemical_symbols(),
            t["kept_atoms"].positions,
            coords_are_cartesian=True,
        )

        # Match against targets
        m_s3 = any(matcher_default.fit(s3_pmg, tgt) for tgt in target_structures)
        m_s4 = any(matcher_default.fit(s4_pmg, tgt) for tgt in target_structures)
        m_kept = any(matcher_default.fit(kept_pmg, tgt) for tgt in target_structures)
        m_loose = any(matcher_loose.fit(kept_pmg, tgt) for tgt in target_structures)

        if m_s3:
            matching_s3_count += 1
        if m_s4:
            matching_s4_count += 1
        if m_loose:
            matching_loose_count += 1
        if m_kept:
            matched_any = True
            for tgt in target_structures:
                if matcher_default.fit(kept_pmg, tgt):
                    v_ratio = (kept_pmg.volume / len(kept_pmg)) / (tgt.volume / len(tgt))
                    vol_ratios.append(v_ratio)
        if m_loose:
            matched_loose_any = True

    # Matched kept trial
    kept_pmg_best = Structure(
        Lattice(kept_trial["kept_atoms"].cell.array),
        kept_trial["kept_atoms"].get_chemical_symbols(),
        kept_trial["kept_atoms"].positions,
        coords_are_cartesian=True,
    )
    matched_kept = any(matcher_default.fit(kept_pmg_best, tgt) for tgt in target_structures)
    matched_loose_kept = any(matcher_loose.fit(kept_pmg_best, tgt) for tgt in target_structures)

    if matched_kept:
        verdict = "recovered"
    elif matched_any:
        verdict = "sampled_not_selected"
    elif de_kept < -0.001:
        verdict = "lower_energy_alternative"
    else:
        verdict = "missed"

    return {
        "gene_id": gid,
        "n_targets": g["n_targets"],
        "spacegroup": g["spacegroup"],
        "crystal_system": g["crystal_system"],
        "nsites": g["nsites"],
        "n_wyckoff_sites": g["n_wyckoff_sites"],
        "dof_positional": g["dof_positional"],
        "dof_total": g["dof_total"],
        "symmetry_tol_attempt": g["symmetry_tol_attempt"],
        "n_trials_generated": n_generated,
        "n_trials_relaxed": n_relaxed,
        "n_matching_trials_s3": matching_s3_count,
        "n_matching_trials_s4": matching_s4_count,
        "n_matching_trials_loose": matching_loose_count,
        "matched_kept": matched_kept,
        "matched_any": matched_any,
        "matched_loose_kept": matched_loose_kept,
        "matched_loose_any": matched_loose_any,
        "e_kept": e_kept,
        "e_target": g["e_target"],
        "de_kept": de_kept,
        "verdict": verdict,
        "best_volume_ratio": float(np.mean(vol_ratios)) if vol_ratios else np.nan,
    }


# ---------------------------------------------------------------------------
# Step 5: Scoring, Breakdown Generation & Verdict Assignment
# ---------------------------------------------------------------------------
def step5_score_and_report(
    unique_genes: List[Dict],
    recon_results: Dict[int, Dict],
    output_dir: Path,
):
    """Score reconstruction results with StructureMatcher, assign verdicts, compute breakdowns."""
    logger.info("Step 5: Scoring structures with StructureMatcher and computing metrics...")

    tasks = [(g, recon_results.get(g["gene_id"], {"n_trials_generated": 0, "trials": []})) for g in unique_genes]
    num_procs = min(4, os.cpu_count() or 4)
    logger.info("Scoring %d genes across %d CPU workers...", len(tasks), num_procs)
    with mp.Pool(processes=num_procs) as pool:
        rows = pool.map(_score_single_gene, tasks)

    df_genes = pd.DataFrame(rows)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_genes.to_csv(tables_dir / "results_genes.csv", index=False)
    logger.info("Saved per-gene results to %s", tables_dir / "results_genes.csv")

    # Compute Headline Numbers
    total_genes = len(df_genes)
    vc = df_genes["verdict"].value_counts().to_dict()
    recovered_count = vc.get("recovered", 0)
    sampled_not_selected_count = vc.get("sampled_not_selected", 0)
    lower_energy_count = vc.get("lower_energy_alternative", 0)
    missed_count = vc.get("missed", 0)
    gen_failed_count = vc.get("generation_failed", 0)

    recovery_rate = recovered_count / total_genes
    sampling_ceiling = (recovered_count + sampled_not_selected_count) / total_genes
    ambiguity_rate = lower_energy_count / total_genes
    missed_rate = missed_count / total_genes
    failure_rate = gen_failed_count / total_genes

    loose_recovery_rate = df_genes["matched_loose_kept"].mean()
    loose_ceiling = df_genes["matched_loose_any"].mean()

    s3_matches = df_genes["n_matching_trials_s3"].sum()
    s4_matches = df_genes["n_matching_trials_s4"].sum()

    headline = {
        "total_genes": total_genes,
        "recovered_rate": recovery_rate,
        "sampling_ceiling": sampling_ceiling,
        "ambiguity_rate": ambiguity_rate,
        "missed_rate": missed_rate,
        "generation_failed_rate": failure_rate,
        "loose_recovery_rate": float(loose_recovery_rate),
        "loose_sampling_ceiling": float(loose_ceiling),
        "total_matching_trials_s3": int(s3_matches),
        "total_matching_trials_s4": int(s4_matches),
        "verdict_counts": vc,
    }
    with open(tables_dir / "headline_metrics.json", "w") as f:
        json.dump(headline, f, indent=2, default=_json_default)

    logger.info("=== HEADLINE RESULTS ===")
    logger.info("Total Genes: %d", total_genes)
    logger.info("Recovery Rate:            %.1f%% (%d/%d)", recovery_rate * 100, recovered_count, total_genes)
    logger.info("Sampling Ceiling:         %.1f%% (%d/%d)", sampling_ceiling * 100, recovered_count + sampled_not_selected_count, total_genes)
    logger.info("Ambiguity Rate:           %.1f%% (%d/%d)", ambiguity_rate * 100, lower_energy_count, total_genes)
    logger.info("Missed Rate:              %.1f%% (%d/%d)", missed_rate * 100, missed_count, total_genes)
    logger.info("Generation Failed Rate:   %.1f%% (%d/%d)", failure_rate * 100, gen_failed_count, total_genes)
    logger.info("CDVAE Loose Recovery:     %.1f%%", loose_recovery_rate * 100)
    logger.info("Stage 3 vs 4 Trial Matches: %d vs %d", s3_matches, s4_matches)

    # Breakdown Helper
    def create_binned_breakdown(column_name: str, bins: List[float], labels: List[str], file_suffix: str):
        df_genes["bin"] = pd.cut(df_genes[column_name], bins=bins, labels=labels, include_lowest=True, right=True)
        g = df_genes.groupby("bin", observed=False)
        breakdown = pd.DataFrame(
            {
                "count": g.size(),
                "recovered_rate": g["matched_kept"].mean(),
                "sampling_ceiling": g["matched_any"].mean(),
                "ambiguity_rate": (g["verdict"].apply(lambda s: (s == "lower_energy_alternative").mean())),
            }
        ).reset_index()
        breakdown.to_csv(tables_dir / f"breakdown_{file_suffix}.csv", index=False)
        return breakdown

    # Positional DOF
    dof_pos_df = create_binned_breakdown(
        "dof_positional",
        bins=[-0.5, 0.5, 2.5, 5.5, 10.5, 1000.0],
        labels=["0", "1–2", "3–5", "6–10", ">10"],
        file_suffix="dof_positional",
    )

    # Total DOF
    dof_tot_df = create_binned_breakdown(
        "dof_total",
        bins=[-0.5, 0.5, 2.5, 5.5, 10.5, 1000.0],
        labels=["0", "1–2", "3–5", "6–10", ">10"],
        file_suffix="dof_total",
    )

    # nsites
    nsites_df = create_binned_breakdown(
        "nsites",
        bins=[0, 10, 20, 40, 1000],
        labels=["≤10", "11–20", "21–40", ">40"],
        file_suffix="nsites",
    )

    # n_wyckoff_sites
    n_wyckoff_df = create_binned_breakdown(
        "n_wyckoff_sites",
        bins=[0, 1.5, 3.5, 6.5, 1000],
        labels=["1", "2–3", "4–6", ">6"],
        file_suffix="n_wyckoff_sites",
    )

    # Crystal System
    g_cs = df_genes.groupby("crystal_system", observed=False)
    cs_df = pd.DataFrame(
        {
            "crystal_system": list(g_cs.groups.keys()),
            "count": g_cs.size(),
            "recovered_rate": g_cs["matched_kept"].mean(),
            "sampling_ceiling": g_cs["matched_any"].mean(),
            "ambiguity_rate": g_cs["verdict"].apply(lambda s: (s == "lower_energy_alternative").mean()),
        }
    ).reset_index(drop=True)
    cs_df.to_csv(tables_dir / "breakdown_crystal_system.csv", index=False)

    # Spacegroup
    g_sg = df_genes.groupby("spacegroup", observed=False)
    sg_df = pd.DataFrame(
        {
            "spacegroup": list(g_sg.groups.keys()),
            "count": g_sg.size(),
            "recovered_rate": g_sg["matched_kept"].mean(),
            "sampling_ceiling": g_sg["matched_any"].mean(),
        }
    ).reset_index(drop=True)
    sg_df.to_csv(tables_dir / "breakdown_spacegroup.csv", index=False)

    # Symmetry Tol Attempt Stratum
    df_genes["attempt_stratum"] = df_genes["symmetry_tol_attempt"].apply(lambda a: "attempt_0" if a == 0 else "attempt_>0")
    g_att = df_genes.groupby("attempt_stratum", observed=False)
    att_df = pd.DataFrame(
        {
            "attempt_stratum": list(g_att.groups.keys()),
            "count": g_att.size(),
            "recovered_rate": g_att["matched_kept"].mean(),
            "sampling_ceiling": g_att["matched_any"].mean(),
        }
    ).reset_index(drop=True)
    att_df.to_csv(tables_dir / "breakdown_symmetry_attempt.csv", index=False)

    # Generate STUDY_REPORT.md
    report_lines = [
        "# CrySPR Reconstruction Fidelity Study Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ",
        f"**Dataset:** LeMat-Bulk MLIP Hull (`e_above_hull <= 0.1 eV/atom`, seed=42)  ",
        f"**Evaluated Unique Genes:** {total_genes}  ",
        f"**Trials per Gene:** 10 (4 stages: lattice-only, fixed-cell, full-unconstrained, rattle+strain)  ",
        "",
        "---",
        "",
        "## 1. Headline Metrics",
        "",
        "| Metric | Rate | Count | Description |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Recovery Rate** | **{recovery_rate * 100:.1f}%** | {recovered_count} / {total_genes} | Kept lowest-energy trial matches target |",
        f"| **Sampling Ceiling** | **{sampling_ceiling * 100:.1f}%** | {recovered_count + sampled_not_selected_count} / {total_genes} | Any of the 10 trials matches target |",
        f"| **Ambiguity Rate** | **{ambiguity_rate * 100:.1f}%** | {lower_energy_count} / {total_genes} | Lower-energy alternative polymorph found ($\\Delta E < -1$ meV/atom) |",
        f"| **Missed Rate** | **{missed_rate * 100:.1f}%** | {missed_count} / {total_genes} | Trials relaxed to higher-energy non-target basins |",
        f"| **Generation Failed Rate** | **{failure_rate * 100:.1f}%** | {gen_failed_count} / {total_genes} | PyXtal candidate generation timed out or failed |",
        f"| **CDVAE Loose Recovery** | **{loose_recovery_rate * 100:.1f}%** | - | Match with `ltol=0.3, stol=0.5, angle_tol=10` |",
        f"| **CDVAE Loose Ceiling** | **{float(loose_ceiling) * 100:.1f}%** | - | Any trial matching under loose tolerance |",
        f"| **Stage 3 vs Stage 4 Matches** | **{s3_matches} vs {s4_matches}** | - | Total matching trials before vs after Stage 4 rattle+strain |",
        "",
        "---",
        "",
        "## 2. Breakdown by Total Degrees of Freedom (`dof_total`)",
        "",
        "| DOF (Total) | Gene Count | Recovery Rate | Sampling Ceiling | Ambiguity Rate |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for _, r in dof_tot_df.iterrows():
        rec_str = f"{r['recovered_rate'] * 100:.1f}%" if pd.notna(r['recovered_rate']) else "N/A"
        ceil_str = f"{r['sampling_ceiling'] * 100:.1f}%" if pd.notna(r['sampling_ceiling']) else "N/A"
        amb_str = f"{r['ambiguity_rate'] * 100:.1f}%" if pd.notna(r['ambiguity_rate']) else "N/A"
        report_lines.append(f"| {r['bin']} | {int(r['count'])} | {rec_str} | {ceil_str} | {amb_str} |")

    report_lines.extend([
        "",
        "## 3. Breakdown by Positional Degrees of Freedom (`dof_positional`)",
        "",
        "| DOF (Positional) | Gene Count | Recovery Rate | Sampling Ceiling | Ambiguity Rate |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for _, r in dof_pos_df.iterrows():
        rec_str = f"{r['recovered_rate'] * 100:.1f}%" if pd.notna(r['recovered_rate']) else "N/A"
        ceil_str = f"{r['sampling_ceiling'] * 100:.1f}%" if pd.notna(r['sampling_ceiling']) else "N/A"
        amb_str = f"{r['ambiguity_rate'] * 100:.1f}%" if pd.notna(r['ambiguity_rate']) else "N/A"
        report_lines.append(f"| {r['bin']} | {int(r['count'])} | {rec_str} | {ceil_str} | {amb_str} |")

    report_lines.extend([
        "",
        "## 4. Breakdown by System Size (`nsites`)",
        "",
        "| Number of Sites | Gene Count | Recovery Rate | Sampling Ceiling | Ambiguity Rate |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for _, r in nsites_df.iterrows():
        rec_str = f"{r['recovered_rate'] * 100:.1f}%" if pd.notna(r['recovered_rate']) else "N/A"
        ceil_str = f"{r['sampling_ceiling'] * 100:.1f}%" if pd.notna(r['sampling_ceiling']) else "N/A"
        amb_str = f"{r['ambiguity_rate'] * 100:.1f}%" if pd.notna(r['ambiguity_rate']) else "N/A"
        report_lines.append(f"| {r['bin']} | {int(r['count'])} | {rec_str} | {ceil_str} | {amb_str} |")

    report_lines.extend([
        "",
        "## 5. Breakdown by Crystal System",
        "",
        "| Crystal System | Gene Count | Recovery Rate | Sampling Ceiling | Ambiguity Rate |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for _, r in cs_df.iterrows():
        rec_str = f"{r['recovered_rate'] * 100:.1f}%" if pd.notna(r['recovered_rate']) else "N/A"
        ceil_str = f"{r['sampling_ceiling'] * 100:.1f}%" if pd.notna(r['sampling_ceiling']) else "N/A"
        amb_str = f"{r['ambiguity_rate'] * 100:.1f}%" if pd.notna(r['ambiguity_rate']) else "N/A"
        report_lines.append(f"| {r['crystal_system']} | {int(r['count'])} | {rec_str} | {ceil_str} | {amb_str} |")

    report_lines.extend([
        "",
        "## 6. Symmetry Tolerance Stratum (`attempt_0` vs `attempt_>0`)",
        "",
        "| Stratum | Gene Count | Recovery Rate | Sampling Ceiling |",
        "| :--- | :---: | :---: | :---: |",
    ])
    for _, r in att_df.iterrows():
        rec_str = f"{r['recovered_rate'] * 100:.1f}%" if pd.notna(r['recovered_rate']) else "N/A"
        ceil_str = f"{r['sampling_ceiling'] * 100:.1f}%" if pd.notna(r['sampling_ceiling']) else "N/A"
        report_lines.append(f"| {r['attempt_stratum']} | {int(r['count'])} | {rec_str} | {ceil_str} |")

    report_path = output_dir / "STUDY_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    logger.info("Saved final report to %s", report_path)

    logger.info("All breakdown tables generated in %s", tables_dir)
    return headline


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def main():
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="CrySPR Reconstruction Fidelity Study")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated/cryspr_reconstruction_study"),
        help="Root output directory",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="cuda:0,cuda:0,cuda:1,cuda:1,cuda:2",
        help="Comma-separated list of CUDA devices to spread the workload over",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of structures to sample (default: 1000)",
    )
    parser.add_argument(
        "--limit-genes",
        type=int,
        default=None,
        help="Limit number of unique genes for pilot run (e.g. 100)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of PyXtal reconstruction trials per gene (default: 10)",
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
        choices=["sample", "relax_targets", "extract_genes", "reconstruct", "score", "all"],
        default="all",
        help="Study pipeline stage to run",
    )
    args = parser.parse_args()

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== CrySPR Reconstruction Fidelity Study ===")
    logger.info("Devices: %s", devices)
    logger.info("Output dir: %s", out_dir)
    logger.info("Stage: %s", args.stage)

    # Step 1
    if args.stage in ["sample", "all"]:
        df_targets = step1_sample_targets(out_dir, seed=args.seed, n_samples=args.n_samples)
        if args.stage == "sample":
            return
    else:
        targets_raw_path = out_dir / "data" / f"sampled_{args.n_samples}_targets_raw.parquet"
        df_targets = pd.read_parquet(targets_raw_path)

    # Step 2
    if args.stage in ["relax_targets", "all"]:
        df_relaxed = step2_relax_targets(df_targets, out_dir, devices=devices)
        if args.stage == "relax_targets":
            return
    else:
        df_relaxed = pd.read_parquet(out_dir / "data" / "relaxed_targets.parquet")

    # Step 3
    if args.stage in ["extract_genes", "all"]:
        unique_genes = step3_extract_and_deduplicate_genes(df_relaxed, out_dir)
        if args.stage == "extract_genes":
            return
    else:
        with gzip.open(out_dir / "data" / "unique_wyckoff_genes.json.gz", "rt") as f:
            unique_genes = json.load(f)

    # Limit genes if running a pilot
    if args.limit_genes is not None:
        logger.info("Applying pilot limit: %d unique genes (out of %d)", args.limit_genes, len(unique_genes))
        unique_genes = unique_genes[: args.limit_genes]

    # Step 4
    if args.stage in ["reconstruct", "all"]:
        recon_results = step4_reconstruct_genes(unique_genes, out_dir, devices=devices, n_trials=args.n_trials)
        if args.stage == "reconstruct":
            return
    else:
        import pickle

        with open(out_dir / "data" / "reconstruction_results.pkl", "rb") as f:
            recon_results = pickle.load(f)

    # Step 5
    if args.stage in ["score", "all"]:
        step5_score_and_report(unique_genes, recon_results, out_dir)

    logger.info("Study execution completed successfully!")


if __name__ == "__main__":
    main()
