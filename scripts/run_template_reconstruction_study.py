#!/usr/bin/env python3
"""Template-matching starts on the 400-structure, 5-trial oracle cohort.

The arm under test replaces PyXtal's random draw with the geometry of a
*training* structure that occupies the same Wyckoff orbits: the gene's
anonymous fingerprint is looked up in LeMat-Bulk, the candidate whose formula
is closest is chosen, and the relaxation starts from its lattice and
coordinates with the gene's elements written onto its orbits
(:mod:`wyckoff_transformer.cryspr.template`).

This is a *reconstruction* study, so the target is itself a LeMat-Bulk entry and
would be its own best template.  Two exclusions keep the arm honest:

- the target's own ``immutable_id``, and
- any candidate that ``StructureMatcher`` says *is* the target, which catches
  the same structure deposited under a second id (LeMat-Bulk merges several
  source databases, so this is not rare).

No random start is re-run.  The four- and five-trial random arms are read
verbatim from ``generated/cryspr_reconstruction_study`` -- the same trials the
`base5`/`base10` references in every other oracle report are computed from --
and only the one template start per structure is relaxed here.  The relaxation
is the baseline's own four-stage schedule at the same ``fmax``, so a template
trial and a random trial differ in nothing but where they started.

Stages::

    match   select a template per structure and rebuild the start   (CPU)
    relax   the four-stage CrySPR schedule on those starts          (GPU)
    score   the arms, the breakdowns and the report                 (CPU)
"""
import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

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
from ase.io import read as ase_read, write as ase_write
from ase.optimize import BFGS
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice as PMGLattice, Structure

from run_cryspr_reconstruction_study import (
    apply_runtime_patches,
    build_patched_orb_calculator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(processName)s) %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("template_study")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("SPGLIB_OLD_ERROR_HANDLING", "0")

#: The cohort every oracle arm is measured on: 400 LeMat-Bulk hull structures
#: with total DoF >= 6, sampled once and archived.
COHORT = Path("generated/cryspr_oracle_fixed_lattice_study/data/sampled_400_targets.parquet")

#: The random-start trials this study reuses rather than recomputing.
BASELINE = Path("generated/cryspr_reconstruction_study/data/reconstruction_results.pkl")

#: Where the reused random trials wrote their optimiser logs, so that the two
#: kinds of start can be compared on how much relaxation they cost.
BASELINE_CRYSPR = Path("generated/cryspr_reconstruction_study/cryspr")

#: Candidates carried out of the index per structure.  More than one because
#: the closest may turn out to be the target under another id, or to resist
#: symmetry detection.
N_CANDIDATES = 8

MATCHES_FILE = "template_matches.csv"
STARTS_FILE = "template_starts.extxyz"
RESULTS_FILE = "template_results.pkl"


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _template_seed(immutable_id: str) -> int:
    """Rattle seed for a template trial: stable, and independent of the random arm."""
    digest = hashlib.sha256(f"template_{immutable_id}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def clean_gene(gene: Dict) -> Dict:
    """The parquet's gene as plain Python, which PyXtal and the fingerprinter want."""
    return {
        "group": int(gene["group"]),
        "species": [str(s) for s in gene["species"]],
        "numIons": [int(n) for n in gene["numIons"]],
        "sites": [[str(s) for s in sites] for sites in gene["sites"]],
    }


# --------------------------------------------------------------------------- #
# Stage 1: match
# --------------------------------------------------------------------------- #
def stage_match(df_cohort: pd.DataFrame, output_dir: Path, index_path: Optional[Path]) -> pd.DataFrame:
    """Choose a template per structure and rebuild the start it implies.

    One pass over the LeMat-Bulk CIF export for every candidate of every
    structure, then the rebuild, which is pure symmetry bookkeeping.

    Returns:
        One row per cohort structure, whether or not it got a template.
    """
    from wyckoff_transformer.cryspr.template import (
        TemplateIndex,
        gene_query,
        load_template_structures,
        single_template,
    )
    from wyckoff_transformer.evaluation.protocol import GeneFingerprinter
    from diagnostics.pyxtal_generation_audit import floor_ratio

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    index = TemplateIndex.load(index_path)
    logger.info("Template index: %d entries, %d anonymous fingerprints",
                len(index), index.n_fingerprints)

    fingerprinter = GeneFingerprinter()
    selections: Dict[str, dict] = {}
    for _, row in df_cohort.iterrows():
        immutable_id = str(row["immutable_id"])
        gene = clean_gene(row["wyckoff_gene"])
        query = gene_query(gene, fingerprinter)
        selections[immutable_id] = {
            "gene": gene,
            "n_candidates": len(index.candidates(query)),
            "matches": index.select(query, exclude=[immutable_id], k=N_CANDIDATES),
        }
    logger.info(
        "%d/%d structures have at least one candidate template",
        sum(1 for s in selections.values() if s["matches"]), len(selections),
    )

    ids = sorted({m.immutable_id for s in selections.values() for m in s["matches"]})
    logger.info("Reading %d candidate templates from the LeMat-Bulk CIF export", len(ids))
    structures = load_template_structures(ids)
    logger.info("Read %d of them", len(structures))

    # The same matcher the study scores recovery with, so "is the target" means
    # here exactly what "recovered the target" means there.
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5,
                               primitive_cell=True, scale=True)

    starts_path = data_dir / STARTS_FILE
    if starts_path.exists():
        starts_path.unlink()

    rows = []
    for _, cohort_row in df_cohort.iterrows():
        immutable_id = str(cohort_row["immutable_id"])
        selection = selections[immutable_id]
        target = Structure.from_file(cohort_row["cif_path"])

        usable, n_duplicate = [], 0
        for match in selection["matches"]:
            structure = structures.get(match.immutable_id)
            if structure is None:
                continue
            if matcher.fit(structure, target):
                n_duplicate += 1
                continue
            usable.append(match)

        atoms, match, error = single_template(selection["gene"], usable, structures)
        if atoms is None and selection["matches"] and not usable:
            error = (
                f"all {len(selection['matches'])} candidates are the target itself"
                if n_duplicate else "no candidate structure could be read"
            )
        row = {
            "struct_idx": int(cohort_row["struct_idx"]),
            "immutable_id": immutable_id,
            "n_candidates": selection["n_candidates"],
            "n_duplicates_dropped": n_duplicate,
            "has_template": atoms is not None,
            "template_id": match.immutable_id if match else None,
            "template_composition": match.composition if match else None,
            "composition_distance": match.distance if match else np.nan,
            "template_e_above_hull": match.energy_above_hull if match else np.nan,
            "floor_ratio": np.nan,
            "volume_per_atom": np.nan,
            "error": error,
        }
        if atoms is not None:
            row["floor_ratio"] = float(floor_ratio(atoms))
            row["volume_per_atom"] = float(atoms.get_volume() / len(atoms))
            atoms.info = {
                "struct_idx": int(cohort_row["struct_idx"]),
                "immutable_id": immutable_id,
                "template_id": match.immutable_id,
            }
            ase_write(str(starts_path), atoms, format="extxyz", append=True)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(data_dir / MATCHES_FILE, index=False)
    logger.info(
        "%d/%d starts rebuilt (%d structures dropped %d duplicate candidates) -> %s",
        int(frame["has_template"].sum()), len(frame),
        int((frame["n_duplicates_dropped"] > 0).sum()),
        int(frame["n_duplicates_dropped"].sum()), starts_path,
    )
    return frame


# --------------------------------------------------------------------------- #
# Stage 2: relax
# --------------------------------------------------------------------------- #
def relax_from_start(
    atoms: Atoms,
    trial_dir: Path,
    calc,
    seed: int,
    fmax: float = 0.02,
    steps_limit: int = 500,
) -> Optional[Dict]:
    """The baseline's four-stage schedule, on a start it did not generate.

    Stages, constraints, optimiser, tolerance and the rattle acceptance rule are
    the ones in :func:`run_cryspr_reconstruction_study._reconstruct_single_trial`;
    only the starting structure differs, which is the whole comparison.
    """
    import torch

    trial_dir.mkdir(parents=True, exist_ok=True)
    formula = atoms.get_chemical_formula(mode="metal")
    n_atoms = len(atoms)
    ase_write(str(trial_dir / f"{formula}_0_initial.cif"), atoms, format="cif")

    try:
        atoms_s1 = atoms.copy()
        atoms_s1.calc = calc
        atoms_s1.set_constraint([FixSymmetry(atoms_s1, symprec=1e-3)])
        BFGS(atoms_s1, logfile=str(trial_dir / f"{formula}_1_fix-cell.log")).run(
            fmax=fmax, steps=steps_limit)
        ase_write(str(trial_dir / f"{formula}_1_fix-cell.cif"), atoms_s1, format="cif")

        atoms_s2 = atoms_s1.copy()
        atoms_s2.calc = calc
        atoms_s2.set_constraint([FixSymmetry(atoms_s2, symprec=1e-3)])
        BFGS(FrechetCellFilter(atoms_s2),
             logfile=str(trial_dir / f"{formula}_2_sym_cell+pos.log")).run(
            fmax=fmax, steps=steps_limit)
        ase_write(str(trial_dir / f"{formula}_2_sym_cell+pos.cif"), atoms_s2, format="cif")

        atoms_s3 = atoms_s2.copy()
        atoms_s3.calc = calc
        atoms_s3.set_constraint([])
        BFGS(FrechetCellFilter(atoms_s3),
             logfile=str(trial_dir / f"{formula}_3_no-sym_cell+pos.log")).run(
            fmax=fmax, steps=steps_limit)
        e_s3 = float(atoms_s3.get_potential_energy())
        cif_s3 = trial_dir / f"{formula}_3_no-sym_cell+pos.cif"
        ase_write(str(cif_s3), atoms_s3, format="cif")

        rng = np.random.default_rng(seed)
        strain_draw = rng.normal(0, 0.01, size=(3, 3))
        strain = 0.5 * (strain_draw + strain_draw.T)
        atoms_s4 = atoms_s3.copy()
        atoms_s4.calc = calc
        atoms_s4.set_constraint([])
        atoms_s4.set_cell(atoms_s4.cell @ (np.eye(3) + strain), scale_atoms=True)
        atoms_s4.rattle(stdev=0.05, seed=int(rng.integers(1, 2**31 - 1)))
        BFGS(FrechetCellFilter(atoms_s4),
             logfile=str(trial_dir / f"{formula}_4_rattle_no-sym.log")).run(
            fmax=fmax, steps=steps_limit)
        e_s4 = float(atoms_s4.get_potential_energy())
        cif_s4 = trial_dir / f"{formula}_4_rattle_no-sym.cif"
        ase_write(str(cif_s4), atoms_s4, format="cif")

        accepted_s4 = (e_s4 - e_s3) / n_atoms < -0.001
        atoms_s3.calc = None
        atoms_s4.calc = None
        kept_atoms = (atoms_s4 if accepted_s4 else atoms_s3).copy()
        kept_atoms.calc = None

        return {
            "trial_idx": 0,
            "e_s3": e_s3,
            "e_s3_per_atom": e_s3 / n_atoms,
            "e_s4": e_s4,
            "e_s4_per_atom": e_s4 / n_atoms,
            "accepted_s4": accepted_s4,
            "kept_stage": "s4" if accepted_s4 else "s3",
            "kept_e": e_s4 if accepted_s4 else e_s3,
            "kept_e_per_atom": (e_s4 if accepted_s4 else e_s3) / n_atoms,
            "kept_cif_path": str(cif_s4 if accepted_s4 else cif_s3),
            "kept_atoms": kept_atoms,
            "volume_ratio_final_initial": float(
                kept_atoms.get_volume() / atoms.get_volume()),
        }
    except torch.cuda.OutOfMemoryError:
        logger.warning("%s: CUDA OOM", formula)
        torch.cuda.empty_cache()
        return None
    except Exception as exc:  # noqa: BLE001 - one bad start must not stop the stage
        logger.warning("%s: relaxation error (%s)", formula, exc)
        return None


def _relax_worker(device: str, task_queue, result_queue, output_dir: Path, fmax: float):
    try:
        apply_runtime_patches()
        calc = build_patched_orb_calculator(device)
        import torch

        while True:
            task = task_queue.get()
            if task is None:
                break
            struct_idx, immutable_id, atoms = task
            trial_dir = output_dir / "cryspr" / str(struct_idx) / "template"
            started = time.time()
            try:
                result = relax_from_start(
                    atoms=atoms,
                    trial_dir=trial_dir,
                    calc=calc,
                    seed=_template_seed(immutable_id),
                    fmax=fmax,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] %s: %s", device, immutable_id, exc)
                result = None
            torch.cuda.empty_cache()
            if result is not None:
                result["seconds"] = round(time.time() - started, 1)
            result_queue.put((struct_idx, immutable_id, result))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Worker on %s failed: %s", device, exc)
        result_queue.put(("ERROR", str(exc), None))


def stage_relax(output_dir: Path, devices: List[str], fmax: float) -> Dict[str, Dict]:
    """Relax every rebuilt start, one per structure, resuming from the cache."""
    data_dir = output_dir / "data"
    cache_path = data_dir / RESULTS_FILE
    results: Dict[str, Dict] = {}
    if cache_path.exists():
        with open(cache_path, "rb") as handle:
            results = pickle.load(handle)
        logger.info("Loaded %d completed template relaxations", len(results))

    starts = ase_read(str(data_dir / STARTS_FILE), index=":", format="extxyz")
    tasks = [
        (int(atoms.info["struct_idx"]), str(atoms.info["immutable_id"]), atoms)
        for atoms in starts
        if str(atoms.info["immutable_id"]) not in results
    ]
    if not tasks:
        logger.info("All %d template starts already relaxed", len(starts))
        return results

    logger.info("Relaxing %d template starts on %s", len(tasks), devices)
    task_queue, result_queue = mp.Queue(), mp.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in devices:
        task_queue.put(None)

    processes = []
    for device in devices:
        process = mp.Process(
            target=_relax_worker,
            args=(device, task_queue, result_queue, output_dir, fmax),
            name=f"TemplateWorker-{device}",
        )
        process.start()
        processes.append(process)

    started = time.time()
    for done in range(1, len(tasks) + 1):
        item = result_queue.get()
        if item[0] == "ERROR":
            for process in processes:
                process.terminate()
            raise RuntimeError(f"Template worker failed: {item[1]}")
        struct_idx, immutable_id, result = item
        results[immutable_id] = {
            "struct_idx": struct_idx,
            "immutable_id": immutable_id,
            "trials": [result] if result is not None else [],
        }
        with open(cache_path, "wb") as handle:
            pickle.dump(results, handle)
        elapsed = time.time() - started
        logger.info(
            "Relaxed %d/%d (%.1f s/start, %.0f min left)",
            done, len(tasks), elapsed / done, elapsed / done * (len(tasks) - done) / 60,
        )

    for process in processes:
        process.join()
    with open(cache_path, "wb") as handle:
        pickle.dump(results, handle)
    return results


# --------------------------------------------------------------------------- #
# Stage 3: score
# --------------------------------------------------------------------------- #
def optimiser_cost(trial_dir: Path) -> tuple[Optional[int], Optional[float]]:
    """BFGS steps over a trial's four stage logs, and the force it started at.

    The step count is the hardware-independent price of a start: a structure
    that begins near a minimum is cheap to relax however fast the card is.  The
    initial force is read from the first stage's step 0, which is the start
    itself, before anything has moved.

    Returns:
        ``(steps, initial fmax in eV/A)``, either of which is ``None`` when the
        logs are not there.
    """
    trial_dir = Path(trial_dir)
    if not trial_dir.is_dir():
        return None, None
    steps, initial = 0, None
    for stage in ("1_fix-cell", "2_sym_cell+pos", "3_no-sym_cell+pos", "4_rattle_no-sym"):
        logs = sorted(trial_dir.glob(f"*_{stage}.log"))
        if not logs:
            continue
        lines = [line for line in logs[0].read_text().splitlines()
                 if line.startswith("BFGS:")]
        steps += max(len(lines) - 1, 0)
        if stage.startswith("1_") and lines:
            initial = float(lines[0].split()[-1])
    return steps, initial


def _as_pmg(atoms: Atoms) -> Structure:
    return Structure(
        PMGLattice(atoms.cell.array),
        atoms.get_chemical_symbols(),
        atoms.positions,
        coords_are_cartesian=True,
    )


def evaluate_arm(trials: List[Dict], target: Structure, target_e: float,
                 matcher: StructureMatcher, matcher_loose: StructureMatcher) -> Dict:
    """Recovery, ceiling and energy for one set of trials, as the oracle reports do."""
    if not trials:
        return {"recovered": False, "ceiling": False, "loose_rec": False,
                "e_kept": np.nan, "de_kept": np.nan, "verdict": "generation_failed",
                "n_rel": 0}
    best = min(trials, key=lambda trial: trial["kept_e_per_atom"])
    e_kept = best["kept_e_per_atom"]
    any_match = any(matcher.fit(_as_pmg(t["kept_atoms"]), target) for t in trials)
    best_pmg = _as_pmg(best["kept_atoms"])
    recovered = matcher.fit(best_pmg, target)
    if recovered:
        verdict = "recovered"
    elif any_match:
        verdict = "sampled_not_selected"
    elif e_kept - target_e < -0.001:
        verdict = "lower_energy_alternative"
    else:
        verdict = "missed"
    return {
        "recovered": recovered,
        "ceiling": any_match,
        "loose_rec": matcher_loose.fit(best_pmg, target),
        "e_kept": e_kept,
        "de_kept": e_kept - target_e,
        "verdict": verdict,
        "n_rel": len(trials),
    }


#: The arms, as ``name -> (template trials, random trials, random trials when the
#: gene has no template)``.  The random ones are always a prefix of the
#: baseline's ten and are never recomputed; only the template trial is new.
#:
#: ``tmpl1`` is the method as proposed -- one template start, and the ordinary
#: randomised start for the genes no training structure shares orbits with --
#: against ``base1``, one random start for everyone.  ``tmpl_strict`` drops the
#: fallback so the template's own contribution is visible.  ``tmpl_base4`` spends
#: a ``base5`` budget with one of its five trials moved to the template.
ARMS = {
    "base1": (0, 1, 1),
    "base5": (0, 5, 5),
    "base10": (0, 10, 10),
    "tmpl1": (1, 0, 1),
    "tmpl_strict": (1, 0, 0),
    "tmpl_base4": (1, 4, 5),
}


def stage_score(df_cohort: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Build the arms from the template trials and the reused random ones."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"

    with open(BASELINE, "rb") as handle:
        baseline = pickle.load(handle)
    with open(data_dir / RESULTS_FILE, "rb") as handle:
        template_results = pickle.load(handle)
    matches = pd.read_csv(data_dir / MATCHES_FILE).set_index("immutable_id")

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5,
                               primitive_cell=True, scale=True)
    matcher_loose = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10,
                                     primitive_cell=True, scale=True)

    rows = []
    for _, cohort_row in df_cohort.iterrows():
        immutable_id = str(cohort_row["immutable_id"])
        target = Structure.from_file(cohort_row["cif_path"])
        target_e = float(cohort_row["orb_e_target_per_atom"])
        random_trials = baseline.get(cohort_row["gene_id"], {}).get("trials", [])
        template_trials = template_results.get(immutable_id, {}).get("trials", [])

        row = {
            "struct_idx": int(cohort_row["struct_idx"]),
            "immutable_id": immutable_id,
            "gene_id": cohort_row["gene_id"],
            "formula": cohort_row["chemical_formula_reduced"],
            "spacegroup": cohort_row["spacegroup"],
            "crystal_system": cohort_row["crystal_system"],
            "nsites": cohort_row["nsites"],
            "dof_total": cohort_row["dof_total"],
            "dof_pos": cohort_row["dof_pos"],
            "e_target": target_e,
        }
        if immutable_id in matches.index:
            for column in ("has_template", "template_id", "composition_distance",
                           "n_candidates", "n_duplicates_dropped", "floor_ratio",
                           "volume_per_atom"):
                row[column] = matches.at[immutable_id, column]
        row["template_relaxed"] = bool(template_trials)
        if template_trials:
            row["template_volume_ratio"] = template_trials[0].get(
                "volume_ratio_final_initial", np.nan)
        row["template_steps"], row["template_start_fmax"] = optimiser_cost(
            output_dir / "cryspr" / str(cohort_row["struct_idx"]) / "template")
        row["base0_steps"], row["base0_start_fmax"] = optimiser_cost(
            BASELINE_CRYSPR / str(cohort_row["gene_id"]) / "trial-0")

        for name, (n_template, n_random, n_fallback) in ARMS.items():
            if template_trials[:n_template]:
                trials = template_trials[:n_template] + random_trials[:n_random]
            else:
                trials = random_trials[:n_fallback]
            evaluation = evaluate_arm(trials, target, target_e, matcher, matcher_loose)
            for key, value in evaluation.items():
                row[f"{name}_{key}"] = value
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(tables_dir / "results_per_structure.csv", index=False)

    summary = {"cohort_size": int(len(frame))}
    for name in ARMS:
        summary[name] = {
            "recovery": float(frame[f"{name}_recovered"].mean()),
            "ceiling": float(frame[f"{name}_ceiling"].mean()),
            "loose_recovery": float(frame[f"{name}_loose_rec"].mean()),
            "verdicts": frame[f"{name}_verdict"].value_counts().to_dict(),
            "median_de": float(frame[f"{name}_de_kept"].median(skipna=True)),
            "pct_de_lt_10meV": float((frame[f"{name}_de_kept"] < 0.01).mean()),
        }
    summary["cost"] = {
        "median_template_steps": float(frame["template_steps"].median(skipna=True)),
        "median_random_steps": float(frame["base0_steps"].median(skipna=True)),
        "median_template_start_fmax": float(frame["template_start_fmax"].median(skipna=True)),
        "median_random_start_fmax": float(frame["base0_start_fmax"].median(skipna=True)),
    }
    summary["template"] = {
        "with_template": int(frame["has_template"].sum()),
        "relaxed": int(frame["template_relaxed"].sum()),
        "exact_formula": int((frame["composition_distance"] == 0).sum()),
        "median_composition_distance": float(frame["composition_distance"].median()),
        "duplicates_dropped": int(frame["n_duplicates_dropped"].fillna(0).sum()),
        "median_candidates": float(frame["n_candidates"].median()),
        "floor_broken": float((frame["floor_ratio"] < 1.0).mean()),
        "median_volume_ratio": float(frame["template_volume_ratio"].median(skipna=True))
        if "template_volume_ratio" in frame else None,
    }
    with open(tables_dir / "comparison_headline.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)

    _write_breakdowns(frame, tables_dir)
    logger.info("Scored %d structures -> %s", len(frame), tables_dir)
    print(json.dumps(summary, indent=2, default=_json_default))
    return frame


def _write_breakdowns(frame: pd.DataFrame, tables_dir: Path) -> None:
    """Recovery per arm, sliced the way the other oracle reports slice it."""
    def breakdown(column, bins, labels, name):
        binned = frame.assign(bin=pd.cut(frame[column], bins=bins, labels=labels,
                                         include_lowest=True))
        grouped = binned.groupby("bin", observed=False)
        table = pd.DataFrame({"count": grouped.size()})
        for arm in ARMS:
            table[f"{arm}_recovery"] = grouped[f"{arm}_recovered"].mean()
        table["delta_tmpl1_base1_pts"] = (
            table["tmpl1_recovery"] - table["base1_recovery"]) * 100
        table["delta_tmplbase4_base5_pts"] = (
            table["tmpl_base4_recovery"] - table["base5_recovery"]) * 100
        table.reset_index().to_csv(tables_dir / f"breakdown_{name}.csv", index=False)
        return table.reset_index()

    breakdown("dof_pos", [-0.5, 2.5, 5.5, 10.5, 1e9], ["0-2", "3-5", "6-10", ">10"], "dof_pos")
    breakdown("dof_total", [5.5, 10.5, 1e9], ["6-10", ">10"], "dof_total")
    breakdown("nsites", [0, 10, 20, 40, 1e9], ["<=10", "11-20", "21-40", ">40"], "nsites")
    breakdown("composition_distance", [-1e-9, 1e-9, 0.005, 0.02, 1.0],
              ["exact", "<=0.005", "<=0.02", ">0.02"], "composition_distance")

    grouped = frame.groupby("crystal_system", observed=False)
    table = pd.DataFrame({"count": grouped.size()})
    for arm in ARMS:
        table[f"{arm}_recovery"] = grouped[f"{arm}_recovered"].mean()
    table.reset_index().to_csv(tables_dir / "breakdown_crystal_system.csv", index=False)


# --------------------------------------------------------------------------- #
def main() -> None:
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("generated/cryspr_template_study"))
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--index", type=Path, default=None,
                        help="Template index parquet; built from the Wyckoff cache if absent")
    parser.add_argument("--devices", type=str, default="cuda:0,cuda:0,cuda:1,cuda:1,cuda:2")
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--stage", choices=["match", "relax", "score", "all"], default="all")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df_cohort = pd.read_parquet(args.cohort)
    logger.info("Cohort: %d structures from %s", len(df_cohort), args.cohort)

    if args.stage in ("match", "all"):
        stage_match(df_cohort, args.output_dir, args.index)
    if args.stage in ("relax", "all"):
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
        stage_relax(args.output_dir, devices, args.fmax)
    if args.stage in ("score", "all"):
        stage_score(df_cohort, args.output_dir)


if __name__ == "__main__":
    main()
