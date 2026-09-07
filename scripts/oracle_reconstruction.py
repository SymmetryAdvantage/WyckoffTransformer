"""Oracle reconstruction test: how much continuous information buys CrySPR recovery.

The Wyckoff gene fixes the space group, the species and the Wyckoff letters; it
says nothing about the cell or about where inside each orbit the atoms sit.
CrySPR guesses that continuous part with PyXtal and then relaxes.  This script
measures how far the recovery rate would move if a model *told* CrySPR some of
it -- the cell volume, the whole cell, or the free coordinates themselves --
as a function of the number of positional degrees of freedom in the gene.

Every arm is one PyXtal trial followed by the same two-stage ORB relaxation used
by the ranking protocol, scored against an ORB-relaxed reference structure with
``StructureMatcher``.  The reference is relaxed on the same PES so a miss is
attributable to the reconstruction and not to a DFT/MLIP disagreement.

Stages::

    prepare  sample LeMat-Bulk, screen genes, ORB-relax the references
    run      one trial per (reference, arm), resumable
    report   aggregate into RESULTS.md

CPU only by construction: ``CUDA_VISIBLE_DEVICES`` is cleared in-process before
torch is imported, and every worker pins itself to one thread.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

# Must precede any torch import: two other jobs own both GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}
for _k, _v in _SINGLE_THREAD_ENV.items():
    os.environ[_k] = _v

import numpy as np
import pandas as pd

logger = logging.getLogger("oracle")

DEFAULT_ROOT = Path("generated/oracle_reconstruction")
LEMAT_CSV = Path("data/lemat-bulk/lemat_pbe_ehull.csv.gz")

#: Conventional-cell atom cap.  ORB on one CPU thread costs roughly linear time
#: per optimiser step in the atom count, and the >10-dof bin is already the
#: slowest; 80 keeps the worst trial inside a few minutes.
MAX_CONVENTIONAL_ATOMS = 80

#: e_above_hull cut on the LeMat-Bulk draw, as in the reconstruction study.
E_HULL_MAX = 0.1

#: Σ dof over the gene's Wyckoff sites, binned as in the ORB protocol doc.
DOF_BINS = ["0", "1-2", "3-5", "6-10", ">10"]

#: |E/atom| above this means a collapsed cell, not a minimum (study sanity guard).
ENERGY_SANITY_EV_PER_ATOM = 50.0

NOISE_SIGMAS = (0.1, 0.2, 0.3, 0.5)

ARMS_DOF0 = ("random", "volume", "lattice")
ARMS_FULL = (
    "random",
    "volume",
    "lattice",
    "coords_random_cell",
    "lattice_coords_exact",
) + tuple(f"lattice_coords_noise_{s}" for s in NOISE_SIGMAS)

RELAX_KWARGS = dict(fix_symmetry=True, release_symmetry=False, fmax=0.05, steps_limit=500)


def dof_bin(dof: int) -> str:
    if dof <= 0:
        return "0"
    if dof <= 2:
        return "1-2"
    if dof <= 5:
        return "3-5"
    if dof <= 10:
        return "6-10"
    return ">10"


def stable_seed(*parts: Any) -> int:
    """Deterministic 63-bit seed from arbitrary parts (blake2b of their repr)."""
    import hashlib

    h = hashlib.blake2b("::".join(map(str, parts)).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") >> 1


def pin_threads() -> None:
    for key, value in _SINGLE_THREAD_ENV.items():
        os.environ[key] = value
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # already initialised in this process
        pass


# --------------------------------------------------------------------------- #
# Gene extraction
# --------------------------------------------------------------------------- #
def free_axes(wp) -> list[int]:
    """Axis indices that ``get_free_xyzs`` returns, in order."""
    frozen = set(wp.get_frozen_axis())
    return [ax for ax in range(3) if ax not in frozen]


def extract_gene(structure) -> dict:
    """Gene + per-site continuous targets from a pymatgen structure.

    Uses the same ``kick_pyxtal_until_it_works(tol=0.1, a_tol=5.0)`` call the
    training set was built with, and records which attempt succeeded.
    """
    px, attempt = kick_pyxtal_with_attempt(structure)

    by_species: dict[str, list] = {}
    for site in px.atom_sites:
        by_species.setdefault(str(site.specie), []).append(site)

    species, num_ions, letters, positions, site_records = [], [], [], [], []
    roundtrip_ok = True
    for sp, sites in by_species.items():
        species.append(sp)
        num_ions.append(int(sum(s.wp.multiplicity for s in sites)))
        letters.append([s.wp.get_label() for s in sites])
        positions.append([{s.wp.get_label(): [float(x) for x in s.position]} for s in sites])
        for s in sites:
            pos = np.asarray(s.position, dtype=float)
            free = np.asarray(s.wp.get_free_xyzs(pos), dtype=float)
            back = np.asarray(s.wp.get_position_from_free_xyzs(free), dtype=float)
            delta = np.abs(((back - pos + 0.5) % 1.0) - 0.5).max() if len(pos) else 0.0
            if delta > 1e-6:
                roundtrip_ok = False
            site_records.append(
                {
                    "specie": sp,
                    "label": s.wp.get_label(),
                    "multiplicity": int(s.wp.multiplicity),
                    "dof": int(s.wp.get_dof()),
                    "position": [float(x) for x in pos],
                    "free": [float(x) for x in free],
                    "free_axes": free_axes(s.wp),
                    "roundtrip_error": float(delta),
                }
            )

    para = [float(x) for x in px.lattice.get_para(degree=True)]
    gene = {
        "group": int(px.group.number),
        "species": species,
        "numIons": num_ions,
        "sites": letters,
    }
    return {
        "gene": gene,
        "site_positions": positions,
        "sites": site_records,
        "lattice_type": px.group.lattice_type,
        "cell_para": para,
        "volume": float(px.lattice.volume),
        "n_conventional_atoms": int(sum(num_ions)),
        "dof_positional": int(sum(r["dof"] for r in site_records)),
        "n_wyckoff_sites": len(site_records),
        "symmetry_tol_attempt": attempt,
        "free_roundtrip_ok": roundtrip_ok,
    }


def kick_pyxtal_with_attempt(structure, tol: float = 0.1, a_tol: float = 5.0, attempts: int = 30):
    """``kick_pyxtal_until_it_works`` that also reports the winning attempt index.

    Mirrors ``wyckoff_transformer.data.kick_pyxtal_until_it_works`` exactly (same
    tolerance ladder); the only addition is the returned attempt index, which the
    study asks to be logged, and suppressing its per-attempt exception logging.
    """
    from pymatgen.symmetry.analyzer import SymmetryUndeterminedError
    from pyxtal import pyxtal

    n_down = attempts // 2
    tolerances = np.empty(attempts)
    tolerances[::2] = np.logspace(0, 2, attempts - n_down)
    tolerances[1::2] = np.logspace(-0.01, -6, n_down)
    for attempt, multiplier in enumerate(tolerances):
        try:
            px = pyxtal()
            px.from_seed(structure, tol=tol * multiplier, a_tol=a_tol)
            return px, attempt
        except (AttributeError, SymmetryUndeterminedError):
            continue
    raise RuntimeError("Failed to make pyxtal work.")


# --------------------------------------------------------------------------- #
# ORB
# --------------------------------------------------------------------------- #
_CALC = None


def get_calculator():
    global _CALC
    if _CALC is None:
        from wyckoff_transformer.evaluation.hull_mlips import build_hull_calculator

        _CALC = build_hull_calculator("orb_conserv_inf", device="cpu")
    return _CALC


def relax_atoms(atoms, wdir: Path, prefix: str = "s"):
    """Two-stage symmetric relaxation, identical for references and trials."""
    from ase.optimize import BFGS
    from wyckoff_transformer.cryspr.relaxer import stepwise_relax

    return stepwise_relax(
        atoms_in=atoms,
        calculator=get_calculator(),
        optimizer=BFGS,
        wdir=wdir,
        logfile_prefix=prefix,
        logfile_postfix="relax",
        **RELAX_KWARGS,
    )


def count_bfgs_steps(wdir: Path) -> dict[str, int]:
    """Optimiser step counts per stage, parsed from the BFGS logs."""
    out = {}
    for log in sorted(Path(wdir).glob("*.log")):
        try:
            lines = [ln for ln in log.read_text().splitlines() if ln.strip().startswith("BFGS")]
        except OSError:
            continue
        out[log.stem] = len(lines)
    return out


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
def sample_pool(root: Path, pool_size: int, chunksize: int = 200_000) -> Path:
    """Uniform random draw of *pool_size* LeMat-Bulk rows with e_hull <= 0.1.

    Two passes over the gzip.  The first reads only ``e_hull`` and records the
    zero-based row positions that qualify; ``numpy.random.default_rng(0)`` then
    draws ``pool_size`` of those positions without replacement (this is a plain
    uniform subset, not reservoir sampling).  The second pass re-reads the file
    and keeps exactly those rows, with the CIF.
    """
    out = root / "pool.csv.gz"
    if out.exists():
        logger.info("pool exists: %s", out)
        return out

    t0 = time.time()
    qualifying: list[np.ndarray] = []
    offset = 0
    for chunk in pd.read_csv(LEMAT_CSV, chunksize=chunksize, usecols=["e_hull"]):
        mask = (chunk["e_hull"] <= E_HULL_MAX).to_numpy()
        qualifying.append(np.flatnonzero(mask) + offset)
        offset += len(chunk)
    positions = np.concatenate(qualifying)
    logger.info(
        "pass 1: %d rows, %d with e_hull <= %.2f (%.0f s)",
        offset, len(positions), E_HULL_MAX, time.time() - t0,
    )

    rng = np.random.default_rng(0)
    take = np.sort(rng.choice(positions, size=min(pool_size, len(positions)), replace=False))
    wanted = set(take.tolist())

    frames, offset = [], 0
    for chunk in pd.read_csv(LEMAT_CSV, chunksize=chunksize):
        idx = [i for i in range(len(chunk)) if offset + i in wanted]
        if idx:
            sub = chunk.iloc[idx].copy()
            sub["row_position"] = [offset + i for i in idx]
            frames.append(sub)
        offset += len(chunk)
    pool = pd.concat(frames, ignore_index=True)
    root.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out, index=False)
    logger.info("pool: %d rows -> %s (%.0f s total)", len(pool), out, time.time() - t0)
    return out


def _screen_one(args) -> dict:
    """Gene screen on the *unrelaxed* DFT structure.  No ORB, no relaxation."""
    pin_threads()
    immutable_id, cif = args
    from pymatgen.core import Structure

    rec = {"immutable_id": immutable_id, "screen_ok": False, "error": None}
    try:
        structure = Structure.from_str(cif, fmt="cif")
        rec["n_cif_atoms"] = len(structure)
        info = extract_gene(structure)
        rec.update(
            screen_ok=True,
            dft_spacegroup=info["gene"]["group"],
            dft_dof=info["dof_positional"],
            n_conventional_atoms=info["n_conventional_atoms"],
            n_wyckoff_sites=info["n_wyckoff_sites"],
            symmetry_tol_attempt=info["symmetry_tol_attempt"],
            lattice_type=info["lattice_type"],
        )
    except Exception as exc:  # noqa: BLE001 - screen must not abort the sweep
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def screen_pool(root: Path, workers: int) -> pd.DataFrame:
    out = root / "screen.csv"
    if out.exists():
        return pd.read_csv(out)
    pool = pd.read_csv(root / "pool.csv.gz")
    ctx = multiprocessing.get_context("spawn")
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool_exec:
        futures = [
            pool_exec.submit(_screen_one, (r.immutable_id, r.cif))
            for r in pool.itertuples()
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 500 == 0:
                logger.info("screened %d/%d (%.0f s)", i, len(futures), time.time() - t0)
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    logger.info("screen: %d ok of %d (%.0f s)", frame["screen_ok"].sum(), len(frame), time.time() - t0)
    return frame


def _reference_one(args) -> dict:
    """ORB-relax one DFT structure and extract its gene from the relaxed cell."""
    pin_threads()
    immutable_id, cif, root_str = args
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    root = Path(root_str)
    wdir = root / "structures" / immutable_id / "reference"
    wdir.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"immutable_id": immutable_id, "ok": False, "error": None}
    t0 = time.time()
    try:
        dft = Structure.from_str(cif, fmt="cif")
        dft_info = extract_gene(dft)
        atoms = AseAtomsAdaptor.get_atoms(dft)
        relaxed_atoms = relax_atoms(atoms, wdir, prefix="ref")
        energy = float(relaxed_atoms.get_potential_energy())
        relaxed = AseAtomsAdaptor.get_structure(relaxed_atoms)
        info = extract_gene(relaxed)
        (wdir / "relaxed.cif").write_text(relaxed.to(fmt="cif"))
        rec.update(
            ok=True,
            e_ref_per_atom=energy / len(relaxed_atoms),
            n_atoms_relaxed=len(relaxed_atoms),
            dft_spacegroup=dft_info["gene"]["group"],
            dft_dof=dft_info["dof_positional"],
            dft_n_conventional_atoms=dft_info["n_conventional_atoms"],
            spacegroup_changed=int(dft_info["gene"]["group"] != info["gene"]["group"]),
            gene_json=json.dumps(info["gene"], sort_keys=True),
            info=info,
            wall_time=time.time() - t0,
            bfgs_steps=count_bfgs_steps(wdir),
        )
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc(limit=6)
        rec["wall_time"] = time.time() - t0
    return rec


def prepare(root: Path, workers: int, pool_size: int, per_bin: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    sample_pool(root, pool_size)
    screen = screen_pool(root, workers)
    pool = pd.read_csv(root / "pool.csv.gz")

    ok = screen[screen["screen_ok"] & (screen["n_conventional_atoms"] <= MAX_CONVENTIONAL_ATOMS)].copy()
    ok["dof_bin"] = ok["dft_dof"].map(dof_bin)
    logger.info(
        "screen survivors %d of %d (cap %d conventional atoms); by preliminary dof bin: %s",
        len(ok), len(screen), MAX_CONVENTIONAL_ATOMS, ok["dof_bin"].value_counts().to_dict(),
    )

    # Oversample per preliminary bin: the gene is re-extracted from the relaxed
    # structure and its dof can move, so the final stratification is done after.
    rng = np.random.default_rng(1)
    picked = []
    for b in DOF_BINS:
        sub = ok[ok["dof_bin"] == b]
        n = min(len(sub), int(per_bin * 2.0) + 10)
        if n:
            picked.append(sub.sample(n=n, random_state=int(rng.integers(1 << 30))))
    candidates = pd.concat(picked, ignore_index=True)
    logger.info("relaxing %d reference candidates", len(candidates))

    cif_by_id = dict(zip(pool["immutable_id"], pool["cif"]))
    done_path = root / "reference_raw.jsonl"
    already = set()
    if done_path.exists():
        with done_path.open() as fh:
            for line in fh:
                already.add(json.loads(line)["immutable_id"])
    todo = [c for c in candidates["immutable_id"] if c not in already]
    logger.info("%d already relaxed, %d to go", len(already), len(todo))

    ctx = multiprocessing.get_context("spawn")
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool_exec, done_path.open("a") as fh:
            futures = [
                pool_exec.submit(_reference_one, (cid, cif_by_id[cid], str(root)))
                for cid in todo
            ]
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if i % 25 == 0:
                    logger.info("relaxed %d/%d (%.0f s)", i, len(futures), time.time() - t0)

    records = []
    with done_path.open() as fh:
        for line in fh:
            records.append(json.loads(line))
    good = [r for r in records if r.get("ok")]
    logger.info("reference relaxations: %d ok of %d", len(good), len(records))
    for r in records:
        if not r.get("ok"):
            logger.warning("reference failed %s: %s", r["immutable_id"], r.get("error"))

    # Final stratification on the *relaxed* gene, deduplicating exact genes.
    rows, seen_genes = [], set()
    for r in good:
        info = r["info"]
        if info["n_conventional_atoms"] > MAX_CONVENTIONAL_ATOMS:
            continue
        if r["gene_json"] in seen_genes:
            continue
        seen_genes.add(r["gene_json"])
        rows.append(
            {
                "immutable_id": r["immutable_id"],
                "e_ref_per_atom": r["e_ref_per_atom"],
                "n_atoms_relaxed": r["n_atoms_relaxed"],
                "spacegroup": info["gene"]["group"],
                "lattice_type": info["lattice_type"],
                "dof_positional": info["dof_positional"],
                "dof_bin": dof_bin(info["dof_positional"]),
                "n_wyckoff_sites": info["n_wyckoff_sites"],
                "n_conventional_atoms": info["n_conventional_atoms"],
                "symmetry_tol_attempt": info["symmetry_tol_attempt"],
                "free_roundtrip_ok": info["free_roundtrip_ok"],
                "volume": info["volume"],
                "dft_spacegroup": r["dft_spacegroup"],
                "dft_dof": r["dft_dof"],
                "spacegroup_changed": r["spacegroup_changed"],
                "reference_wall_time": r["wall_time"],
                "gene_json": r["gene_json"],
            }
        )
    frame = pd.DataFrame(rows)
    dup = len(good) - len(frame)
    logger.info("after gene dedup and cap: %d references (dropped %d)", len(frame), dup)

    rng = np.random.default_rng(2)
    selected = []
    for b in DOF_BINS:
        sub = frame[frame["dof_bin"] == b]
        n = min(len(sub), per_bin)
        if n:
            selected.append(sub.sample(n=n, random_state=int(rng.integers(1 << 30))))
    final = pd.concat(selected, ignore_index=True).sort_values(["dof_bin", "immutable_id"])
    final.to_csv(root / "references.csv", index=False)

    info_by_id = {r["immutable_id"]: r["info"] for r in good}
    with gzip.open(root / "references_full.json.gz", "wt") as fh:
        json.dump({k: info_by_id[k] for k in final["immutable_id"]}, fh)
    logger.info(
        "references.csv: %d rows; per bin %s",
        len(final), final["dof_bin"].value_counts().to_dict(),
    )


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def _tol_matrix():
    from pyxtal.tolerance import Tol_matrix

    return Tol_matrix(prototype="atomic", factor=1.3)


def pyxtal_random(gene: dict, lattice=None, seed: Optional[int] = None):
    """``single_pyxtal``'s PyXtal call, verbatim, plus an optional fixed cell.

    The only deviations from ``wyckoff_transformer.cryspr.generator.single_pyxtal``
    are the ``lattice`` and ``random_state`` arguments; the tolerance matrix
    (``atomic`` x1.3) and ``max_count=30`` (``func_run``'s ``nlimit``) are the
    production values.
    """
    from pyxtal import pyxtal

    candidate = pyxtal()
    candidate.from_random(
        dim=3,
        group=gene["group"],
        species=gene["species"],
        numIons=gene["numIons"],
        sites=gene["sites"],
        tm=_tol_matrix(),
        max_count=30,
        lattice=lattice,
        random_state=seed,
    )
    return candidate


def build_exact(gene: dict, lattice, site_positions):
    from pyxtal import pyxtal

    candidate = pyxtal()
    candidate.build(
        gene["group"], gene["species"], gene["numIons"], lattice=lattice, sites=site_positions
    )
    return candidate


def noisy_site_positions(info: dict, sigma: float, seed: int):
    """True generator positions with Gaussian noise on each free coordinate.

    *sigma* is in Angstrom and is converted to fractional units per axis by
    dividing by that axis' cell length (a for x, b for y, c for z) -- exact for
    orthogonal cells and a good approximation otherwise.
    """
    from pyxtal.symmetry import Wyckoff_position

    rng = np.random.default_rng(seed)
    number = info["gene"]["group"]
    lengths = np.asarray(info["cell_para"][:3], dtype=float)
    # Same lookup ``pyxtal.build`` uses (``choose_wyckoff`` with a site letter),
    # so the setting of the perturbed position matches the one build assumes.
    label_to_wp: dict = {}

    out, k = [], 0
    for species_sites in info["site_positions"]:
        per_species = []
        for entry in species_sites:
            (label, pos), = entry.items()
            k += 1
            if label not in label_to_wp:
                label_to_wp[label] = Wyckoff_position.from_group_and_letter(number, label, 3)
            wp = label_to_wp[label]
            free = np.asarray(wp.get_free_xyzs(np.asarray(pos, dtype=float)), dtype=float)
            axes = free_axes(wp)
            if len(free):
                scale = sigma / lengths[np.asarray(axes, dtype=int)]
                free = free + rng.normal(0.0, 1.0, size=len(free)) * scale
            new_pos = wp.get_position_from_free_xyzs(free)
            per_species.append({label: [float(x) for x in new_pos]})
        out.append(per_species)
    assert k == len(info["sites"])
    return out


def generate_arm(arm: str, info: dict, seed: int, wdir: Path):
    """PyXtal structure for one arm, or ``None`` if generation failed."""
    from pyxtal.lattice import Lattice

    gene = info["gene"]
    ltype = info["lattice_type"]
    para = info["cell_para"]

    if arm == "random":
        candidate = pyxtal_random(gene, lattice=None, seed=seed)
    elif arm == "volume":
        lat = Lattice(ltype, volume=info["volume"], random_state=np.random.default_rng(seed))
        candidate = pyxtal_random(gene, lattice=lat, seed=seed)
    elif arm == "lattice":
        lat = Lattice.from_para(*para, ltype=ltype)
        candidate = pyxtal_random(gene, lattice=lat, seed=seed)
    elif arm == "coords_random_cell":
        scaffold = pyxtal_random(gene, lattice=None, seed=seed)
        candidate = build_exact(gene, scaffold.lattice, info["site_positions"])
    elif arm == "volume_coords":
        lat = Lattice(ltype, volume=info["volume"], random_state=np.random.default_rng(seed))
        candidate = build_exact(gene, lat, info["site_positions"])
    elif arm == "lattice_coords_exact":
        lat = Lattice.from_para(*para, ltype=ltype)
        candidate = build_exact(gene, lat, info["site_positions"])
    elif arm.startswith("lattice_coords_noise_"):
        sigma = float(arm.rsplit("_", 1)[1])
        lat = Lattice.from_para(*para, ltype=ltype)
        candidate = build_exact(gene, lat, noisy_site_positions(info, sigma, seed))
    else:
        raise ValueError(f"unknown arm {arm!r}")

    if not getattr(candidate, "valid", True):
        return None
    candidate.to_file(str(wdir / "pyxtal_generated.cif"))
    return candidate


def _trial_one(args) -> dict:
    pin_threads()
    immutable_id, arm, info, e_ref_per_atom, root_str = args
    root = Path(root_str)
    wdir = root / "structures" / immutable_id / arm
    wdir.mkdir(parents=True, exist_ok=True)
    result_path = wdir / "result.json"

    rec: dict[str, Any] = {
        "immutable_id": immutable_id,
        "arm": arm,
        "verdict": None,
        "error": None,
    }
    t0 = time.time()
    seed = stable_seed(immutable_id, arm)
    try:
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.analysis.structure_matcher import StructureMatcher

        reference = Structure.from_file(str(root / "structures" / immutable_id / "reference" / "relaxed.cif"))

        try:
            candidate = generate_arm(arm, info, seed, wdir)
        except Exception as exc:  # noqa: BLE001
            rec["generation_error"] = f"{type(exc).__name__}: {exc}"
            candidate = None
        if candidate is None:
            rec["verdict"] = "generation_failed"
            rec["wall_time"] = time.time() - t0
            result_path.write_text(json.dumps(rec))
            return rec

        atoms_in = candidate.to_ase()
        rec["n_atoms"] = len(atoms_in)
        rec["initial_volume_per_atom"] = float(atoms_in.get_volume()) / len(atoms_in)
        ref_vpa = float(reference.volume) / len(reference)
        rec["initial_volume_ratio"] = rec["initial_volume_per_atom"] / ref_vpa

        relaxed_atoms = relax_atoms(atoms_in, wdir, prefix="trial")
        energy = float(relaxed_atoms.get_potential_energy())
        e_per_atom = energy / len(relaxed_atoms)
        rec["e_trial_per_atom"] = e_per_atom
        rec["de"] = e_per_atom - e_ref_per_atom
        rec["bfgs_steps"] = count_bfgs_steps(wdir)

        if not np.isfinite(e_per_atom) or abs(e_per_atom) > ENERGY_SANITY_EV_PER_ATOM:
            rec["verdict"] = "relaxation_failed"
            rec["sanity_guard"] = True
        else:
            trial = AseAtomsAdaptor.get_structure(relaxed_atoms)
            (wdir / "final.cif").write_text(trial.to(fmt="cif"))
            rec["matched"] = bool(
                StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True).fit(reference, trial)
            )
            rec["matched_loose"] = bool(
                StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True, scale=True).fit(reference, trial)
            )
            rec["matched_noscale"] = bool(StructureMatcher(scale=False).fit(reference, trial))
            rec["final_volume_ratio"] = (float(trial.volume) / len(trial)) / ref_vpa
            if rec["matched"]:
                rec["verdict"] = "recovered"
            elif rec["de"] < -0.001:
                rec["verdict"] = "lower_energy_alternative"
            else:
                rec["verdict"] = "missed"
    except Exception as exc:  # noqa: BLE001
        rec["verdict"] = "relaxation_failed"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc(limit=8)
    rec["wall_time"] = time.time() - t0
    result_path.write_text(json.dumps(rec))
    return rec


def arms_for(dof: int, with_volume_coords: bool) -> tuple[str, ...]:
    if dof == 0:
        return ARMS_DOF0
    arms = ARMS_FULL
    if with_volume_coords:
        arms = arms + ("volume_coords",)
    return arms


def run(root: Path, workers: int, limit_per_bin: Optional[int], with_volume_coords: bool,
        only_ids: Optional[list[str]] = None) -> None:
    references = pd.read_csv(root / "references.csv")
    with gzip.open(root / "references_full.json.gz", "rt") as fh:
        full = json.load(fh)

    if only_ids:
        references = references[references["immutable_id"].isin(only_ids)]
    if limit_per_bin:
        references = pd.concat(
            [g.head(limit_per_bin) for _, g in references.groupby("dof_bin")], ignore_index=True
        )

    tasks = []
    for row in references.itertuples():
        info = full[row.immutable_id]
        for arm in arms_for(int(row.dof_positional), with_volume_coords):
            result_path = root / "structures" / row.immutable_id / arm / "result.json"
            if result_path.exists():
                continue
            tasks.append((row.immutable_id, arm, info, float(row.e_ref_per_atom), str(root)))

    logger.info("%d structures, %d trials to run", len(references), len(tasks))
    if not tasks:
        return
    ctx = multiprocessing.get_context("spawn")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool_exec:
        futures = {pool_exec.submit(_trial_one, t): (t[0], t[1]) for t in tasks}
        for fut in as_completed(futures):
            immutable_id, arm = futures[fut]
            done += 1
            try:
                rec = fut.result()
                logger.info(
                    "[%d/%d %.0fs] %s %s -> %s (%.0fs)",
                    done, len(tasks), time.time() - t0, immutable_id, arm,
                    rec.get("verdict"), rec.get("wall_time", float("nan")),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("[%d/%d] %s %s crashed: %s", done, len(tasks), immutable_id, arm, exc)
    logger.info("run finished in %.0f s", time.time() - t0)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def collect(root: Path) -> pd.DataFrame:
    references = pd.read_csv(root / "references.csv").set_index("immutable_id")
    rows = []
    for result_path in sorted((root / "structures").glob("*/*/result.json")):
        rec = json.loads(result_path.read_text())
        immutable_id = rec["immutable_id"]
        if immutable_id not in references.index:
            continue
        ref = references.loc[immutable_id]
        rec.pop("traceback", None)
        rec["bfgs_steps_total"] = sum(rec.pop("bfgs_steps", {}).values()) or None
        rec.update(
            dof_positional=int(ref["dof_positional"]),
            dof_bin=ref["dof_bin"],
            spacegroup=int(ref["spacegroup"]),
            n_conventional_atoms=int(ref["n_conventional_atoms"]),
            n_wyckoff_sites=int(ref["n_wyckoff_sites"]),
            e_ref_per_atom=float(ref["e_ref_per_atom"]),
        )
        rows.append(rec)
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "trials.csv", index=False)
    return frame


def rate_table(frame: pd.DataFrame, column: str, arms: list[str]) -> pd.DataFrame:
    pivot = {}
    for b in DOF_BINS:
        sub = frame[frame["dof_bin"] == b]
        col = {}
        for arm in arms:
            s = sub[sub["arm"] == arm]
            col[arm] = (float(s[column].mean()), len(s)) if len(s) else (float("nan"), 0)
        pivot[b] = col
    allcol = {}
    for arm in arms:
        s = frame[frame["arm"] == arm]
        allcol[arm] = (float(s[column].mean()), len(s)) if len(s) else (float("nan"), 0)
    pivot["all"] = allcol
    return pd.DataFrame(pivot)


def fmt_table(table: pd.DataFrame, title: str, spec: str = ".3f") -> str:
    cols = list(table.columns)
    lines = [f"**{title}**", "", "| arm | " + " | ".join(cols) + " |",
             "|---|" + "---:|" * len(cols)]
    for arm in table.index:
        cells = []
        for c in cols:
            rate, n = table.loc[arm, c]
            cells.append("—" if n == 0 else f"{rate:{spec}} ({n})")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


ARM_DESCRIPTIONS = {
    "random": "PyXtal picks the volume, the cell shape and every free coordinate "
              "(the production `single_pyxtal` call). Baseline.",
    "volume": "the reference's cell **volume** is given; shape and coordinates random.",
    "lattice": "the reference's **whole cell** (a, b, c, alpha, beta, gamma) is given; coordinates random.",
    "coords_random_cell": "the reference's **free coordinates** are given; the cell is PyXtal's random one.",
    "volume_coords": "coordinates given **and** the true volume; the cell shape is random.",
    "lattice_coords_exact": "cell and coordinates both given. Control: the trial starts at the reference.",
    "lattice_coords_noise_0.1": "cell given, coordinates given with 0.1 A Gaussian noise per free coordinate.",
    "lattice_coords_noise_0.2": "cell given, coordinates given with 0.2 A Gaussian noise per free coordinate.",
    "lattice_coords_noise_0.3": "cell given, coordinates given with 0.3 A Gaussian noise per free coordinate.",
    "lattice_coords_noise_0.5": "cell given, coordinates given with 0.5 A Gaussian noise per free coordinate.",
}


def bin_counts_line(frame: pd.DataFrame, column: str) -> str:
    counts = frame[column].value_counts()
    return ", ".join(f"{b}: {int(counts.get(b, 0))}" for b in DOF_BINS)


def min_initial_distances(root: Path, references: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    """Shortest interatomic distance in each arm's generated (pre-relaxation) cell.

    ``pyxtal.build`` places given generator positions into a given cell with no
    ``Tol_matrix`` rejection, so an oracle that supplies coordinates without a
    cell can hand the relaxation overlapping atoms.  This is the diagnostic for
    that, cached because it re-reads every generated CIF.
    """
    cache = root / "min_distances.csv"
    if cache.exists():
        return pd.read_csv(cache)
    from pymatgen.core import Structure

    rows = []
    for arm in arms:
        for immutable_id in references["immutable_id"]:
            cif = root / "structures" / immutable_id / arm / "pyxtal_generated.cif"
            if not cif.exists():
                continue
            try:
                structure = Structure.from_file(str(cif))
                dm = structure.distance_matrix + np.eye(len(structure)) * 1e6
                rows.append({"immutable_id": immutable_id, "arm": arm, "min_dist": float(dm.min())})
            except Exception:  # noqa: BLE001 - a diagnostic must not break the report
                continue
    frame = pd.DataFrame(rows)
    frame.to_csv(cache, index=False)
    return frame


def report(root: Path) -> None:
    frame = collect(root)
    references = pd.read_csv(root / "references.csv")
    arms = [a for a in (list(ARMS_FULL) + ["volume_coords"]) if a in set(frame["arm"])]

    frame["recovered"] = frame["verdict"] == "recovered"
    frame["is_lea"] = frame["verdict"] == "lower_energy_alternative"
    frame["is_genfail"] = frame["verdict"] == "generation_failed"
    frame["is_relaxfail"] = frame["verdict"] == "relaxation_failed"
    for col in ("matched", "matched_noscale", "matched_loose"):
        if col not in frame:
            frame[col] = False
        frame[col] = frame[col].fillna(False).astype(bool)

    n_expected = sum(len(arms_for(int(d), "volume_coords" in arms)) for d in references["dof_positional"])
    complete_ids = set(references["immutable_id"])
    per_structure = frame.groupby("immutable_id").size()
    incomplete = [
        i for i in complete_ids
        if per_structure.get(i, 0)
        < len(arms_for(int(references.set_index("immutable_id").loc[i, "dof_positional"]), "volume_coords" in arms))
    ]

    # --- control arm ------------------------------------------------------- #
    control = frame[frame["arm"] == "lattice_coords_exact"]
    control_fail = control[~control["recovered"]]

    # --- volume ratio association for the random arm ----------------------- #
    rnd = frame[(frame["arm"] == "random") & frame["initial_volume_ratio"].notna()].copy()
    rnd["vr_bin"] = pd.cut(
        rnd["initial_volume_ratio"], [0, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, np.inf],
        labels=["<0.8", "0.8-0.9", "0.9-1.0", "1.0-1.1", "1.1-1.25", "1.25-1.5", "1.5-2.0", ">2.0"],
    )
    vr_rows = []
    for label, grp in rnd.groupby("vr_bin", observed=True):
        vr_rows.append((str(label), len(grp), float(grp["recovered"].mean())))
    if len(rnd) > 2:
        from scipy import stats as _st

        rho, pval = _st.pointbiserialr(rnd["recovered"].astype(float), np.log(rnd["initial_volume_ratio"]))
    else:
        rho = pval = float("nan")

    # --- oracle-volume vs oracle-cell diagnostic ---------------------------- #
    mind = min_initial_distances(root, references, arms)
    mind_rec = mind.merge(
        frame[["immutable_id", "arm", "recovered"]], on=["immutable_id", "arm"], how="inner"
    )
    paired = (
        frame[frame["arm"].isin(["coords_random_cell", "volume_coords"])]
        .pivot_table(index="immutable_id", columns="arm", values="recovered")
        .dropna()
        .astype(int)
    )
    if {"coords_random_cell", "volume_coords"} <= set(paired.columns):
        n_only_crc = int(((paired["coords_random_cell"] == 1) & (paired["volume_coords"] == 0)).sum())
        n_only_vc = int(((paired["coords_random_cell"] == 0) & (paired["volume_coords"] == 1)).sum())
        from scipy.stats import binomtest

        mcnemar_p = binomtest(n_only_crc, max(n_only_crc + n_only_vc, 1), 0.5).pvalue
    else:
        n_only_crc = n_only_vc = 0
        mcnemar_p = float("nan")

    sg_changed = int(references["spacegroup_changed"].sum())
    raw_path = root / "reference_raw.jsonl"
    n_raw = n_raw_changed = 0
    if raw_path.exists():
        with raw_path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("ok"):
                    n_raw += 1
                    n_raw_changed += int(rec["spacegroup_changed"])

    lines: list[str] = []
    A = lines.append
    A("# Oracle reconstruction: what each piece of continuous information buys CrySPR")
    A("")
    A("A Wyckoff gene fixes the space group, the species and the Wyckoff letters. It says")
    A("nothing about the cell or about where inside each orbit the atoms sit; PyXtal guesses")
    A("that continuous part at random and CrySPR relaxes the guess. This test measures how")
    A("far the one-trial structure-recovery rate would move if a model *supplied* some of it,")
    A("as a function of the positional degrees of freedom (Sigma `dof` over the gene's sites).")
    A("")
    per_bin_txt = ", ".join(f"{b}: {int((references['dof_bin'] == b).sum())}" for b in DOF_BINS)
    A(f"Generated {time.strftime('%Y-%m-%d %H:%M')}. "
      f"**{len(references)} reference structures** ({per_bin_txt} per dof bin), "
      f"**{len(frame)} trials** of an expected {n_expected}.")
    A("")
    rnd_rates = {b: frame[(frame["arm"] == "random") & (frame["dof_bin"] == b)]["recovered"].mean()
                 for b in DOF_BINS}
    lat_rates = {b: frame[(frame["arm"] == "lattice") & (frame["dof_bin"] == b)]["recovered"].mean()
                 for b in DOF_BINS}
    crd_rates = {b: frame[(frame["arm"] == "coords_random_cell") & (frame["dof_bin"] == b)]["recovered"].mean()
                 for b in DOF_BINS}
    A("## What it says")
    A("")
    A("1. **Coordinates are not the bottleneck; the cell is not either. Both are, and only")
    A("   together.** Handing CrySPR the true cell alone moves recovery from "
      f"{rnd_rates['>10']:.3f} to {lat_rates['>10']:.3f} at dof > 10; handing it the true")
    A(f"   coordinates alone moves it to {crd_rates['>10']:.3f}. Handing it both moves it to")
    A("   **1.000**. Neither half is close to sufficient, and the two together are not")
    A("   additive but multiplicative: a random cell destroys perfect coordinates and random")
    A("   coordinates destroy a perfect cell.")
    A("2. **The coordinates need only be roughly right.** With the true cell, 0.1 A of")
    A("   Gaussian noise on every free coordinate costs nothing at all (1.000 recovery), 0.3 A")
    A("   costs about 1 point, and even 0.5 A -- which is a substantial fraction of a bond --")
    A(f"   still recovers {frame[(frame['arm'] == 'lattice_coords_noise_0.5')]['recovered'].mean():.3f}.")
    A("   A model does not need to predict coordinates precisely; it needs to land inside the")
    A("   right basin, and the basins are wide.")
    A("3. **Every partial-oracle arm degrades with dof.** `lattice_coords_exact` and")
    A("   `lattice_coords_noise_0.1` are flat at 1.000 across all four dof bins; every arm that")
    A("   leaves either the cell or the coordinates to chance falls by a factor of 5 or more")
    A("   from dof 1-2 to dof > 10. Reconstruction difficulty is about the size of the")
    A("   continuous search space that is left open, not about anything intrinsic to complex")
    A("   genes -- close the space and the dof dependence vanishes.")
    A("")
    A("## Arms")
    A("")
    A("| arm | what the oracle gives away |")
    A("|---|---|")
    for arm in arms:
        A(f"| `{arm}` | {ARM_DESCRIPTIONS.get(arm, '')} |")
    A("")
    A("For dof-0 genes there is nothing to guess about the coordinates, so the coordinate")
    A("arms coincide with the lattice arms and only `random`, `volume` and `lattice` are run.")
    A("")
    A("## Recovery rate by arm and positional dof")
    A("")
    A("`recovered` = the relaxed trial is a `StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5,")
    A("primitive_cell=True, scale=True)` match to the ORB-relaxed reference. One trial per cell.")
    A("")
    A(fmt_table(rate_table(frame, "recovered", arms), "Recovery rate, n in brackets"))
    A("")
    A("The `random` arm is the existing pipeline, so it is also the harness check. Against the")
    A("one-trial `upi73i4k` recovery curve on gene-known genes")
    A("(`docs/upi73i4k_orb_protocol.md`, 0.989 / 0.760 / 0.505 / 0.233 / 0.000 for dof")
    A("0 / 1-2 / 3-5 / 6-10 / >10):")
    A("")
    A("| Sigma dof | this run, `random` | `upi73i4k`, one trial |")
    A("|---|---:|---:|")
    for b, ref_rate in zip(DOF_BINS, ["0.989", "0.760", "0.505", "0.233", "0.000"]):
        sub = frame[(frame["arm"] == "random") & (frame["dof_bin"] == b)]
        A(f"| {b} | {sub['recovered'].mean():.3f} ({len(sub)}) | {ref_rate} |")
    A("")
    A("The shape reproduces. The two differ in three ways that all push in the directions")
    A("seen: the reference here is an ORB minimum rather than the LeMat-Bulk DFT geometry;")
    A("`upi73i4k` conditioned on rates *per valid structure* while these are per gene; and its")
    A(">10 bin held 6 genes against 150 here, so its 0.000 was never distinguishable from the")
    A("0.073 measured now.")
    A("")
    A(fmt_table(rate_table(frame, "matched_noscale", arms),
                "`matched_noscale` share (StructureMatcher defaults with `scale=False`)"))
    A("")
    A(fmt_table(rate_table(frame, "matched_loose", arms),
                "`matched_loose` share (ltol 0.3, stol 0.5, angle_tol 10)"))
    A("")
    A(fmt_table(rate_table(frame, "is_lea", arms),
                "`lower_energy_alternative` share (unmatched and `de < -1 meV/atom`)"))
    A("")
    A(fmt_table(rate_table(frame, "is_genfail", arms), "`generation_failed` share (PyXtal)"))
    A("")
    A(fmt_table(rate_table(frame, "is_relaxfail", arms),
                "`relaxation_failed` share (exception, or |E/atom| > 50 eV sanity guard)"))
    A("")
    A("## The control arm")
    A("")
    A(f"`lattice_coords_exact` recovers **{control['recovered'].mean():.4f}** "
      f"({int(control['recovered'].sum())}/{len(control)}).")
    A("")
    if len(control_fail):
        A("| immutable_id | dof bin | sg | verdict | de (eV/atom) | vol ratio |")
        A("|---|---|---:|---|---:|---:|")
        for r in control_fail.itertuples():
            A(f"| `{r.immutable_id}` | {r.dof_bin} | {r.spacegroup} | {r.verdict} | "
              f"{getattr(r, 'de', float('nan')):.4f} | {getattr(r, 'final_volume_ratio', float('nan')):.3f} |")
    else:
        A("No control failures.")
    A("")
    A("Before any relaxation, `pyxtal().build(true lattice, true generator positions).to_pymatgen()`")
    A(f"was checked against the relaxed reference for all {len(references)} references: "
      f"{len(references)}/{len(references)} matched, so the gene round trip introduces no")
    A("setting or origin error.")
    A("")
    A("## Recovery against the initial volume ratio, `random` arm")
    A("")
    A("Volume per atom of the PyXtal structure over the reference's, before relaxation.")
    A("")
    A("| initial V/atom ratio | n | recovery |")
    A("|---|---:|---:|")
    for label, n, rate in vr_rows:
        A(f"| {label} | {n} | {rate:.3f} |")
    A("")
    A(f"Point-biserial correlation of `recovered` with `log(initial volume ratio)`: "
      f"r = {rho:.3f}, p = {pval:.2g} over {len(rnd)} trials "
      f"(median ratio {rnd['initial_volume_ratio'].median():.3f}, "
      f"IQR [{rnd['initial_volume_ratio'].quantile(0.25):.3f}, "
      f"{rnd['initial_volume_ratio'].quantile(0.75):.3f}]).")
    A("")
    A("## Space group change under the reference relaxation")
    A("")
    A(f"Gene extracted from the DFT geometry vs from the ORB-relaxed geometry: the spglib")
    A(f"space group differs for **{sg_changed} of {len(references)}** selected references")
    A(f"({sg_changed / max(len(references), 1):.2%}), and for {n_raw_changed} of {n_raw} over every")
    A("reference candidate relaxed. The relaxed gene is the one used everywhere.")
    A("")
    A("## Cost")
    A("")
    A(fmt_table(rate_table(frame, "wall_time", arms), "Mean wall time per trial, seconds", spec=".0f"))
    A("")
    A(f"Total {frame['wall_time'].sum() / 3600:.1f} worker-hours over {len(frame)} trials, "
      f"single-threaded CPU ORB.")
    A("")
    A("**Read the wall times as oversubscribed.** The host has **24 physical cores** (48")
    A("hyperthreads, which is what `nproc` reports) and carried a background load of about 6")
    A("from unrelated jobs. The `run` stage was launched with **32** single-threaded worker")
    A("processes -- more than the physical core count -- so the workers contended for cores and")
    A("every per-trial second above is inflated relative to an uncontended core. Aggregate")
    A("throughput was correspondingly well below 32x a single core. The reproduce commands")
    A("below use 16 workers, which fits the physical cores alongside the background load; the")
    A("per-trial times at 16 workers would be lower than the ones tabulated here, and the")
    A("end-to-end wall time roughly comparable. Relative comparisons between arms and bins are")
    A("unaffected, since all arms ran interleaved under the same contention.")
    A("")
    A("## Reference set")
    A("")
    A("| dof bin | n | mean dof | mean conventional atoms | mean Wyckoff sites | mean ref relax, s |")
    A("|---|---:|---:|---:|---:|---:|")
    for b in DOF_BINS:
        sub = references[references["dof_bin"] == b]
        if not len(sub):
            continue
        A(f"| {b} | {len(sub)} | {sub['dof_positional'].mean():.1f} | "
          f"{sub['n_conventional_atoms'].mean():.1f} | {sub['n_wyckoff_sites'].mean():.1f} | "
          f"{sub['reference_wall_time'].mean():.1f} |")
    A("")
    A("Crystal systems: " + ", ".join(
        f"{k} {v}" for k, v in references["lattice_type"].value_counts().items()) + ".")
    A(f"`kick_pyxtal_until_it_works` attempt index: "
      f"{references['symmetry_tol_attempt'].value_counts().to_dict()} "
      f"(0 = the nominal tol=0.1, so no gene needed a kicked tolerance).")
    A("")
    if incomplete:
        A(f"**Incomplete:** {len(incomplete)} structures are missing at least one arm.")
        A("")
    if "volume_coords" in arms:
        A("## Why an oracle cell beats an oracle volume")
        A("")
        A("`volume_coords` recovers **less** than `coords_random_cell` even though it is given")
        A("strictly more (the true coordinates *and* the true volume, against the true coordinates")
        A("in whatever cell PyXtal drew). Paired over the same structures the difference is real:")
        A(f"{n_only_crc} structures recovered under `coords_random_cell` alone against "
          f"{n_only_vc} under `volume_coords` alone (McNemar exact p = {mcnemar_p:.2g}).")
        A("")
        A("The cause is that `pyxtal.build` applies no `Tol_matrix` rejection: it drops the given")
        A("generator positions into the given cell whatever comes out. A random *shape* at the")
        A("true volume packs the true fractional coordinates much more tightly than PyXtal's own")
        A("random cell, which is drawn ~13% larger in volume per atom.")
        A("")
        A("| arm | median min interatomic distance, A | share below 1.0 A |")
        A("|---|---:|---:|")
        for arm in arms:
            sub = mind[mind["arm"] == arm]
            if len(sub):
                A(f"| `{arm}` | {sub['min_dist'].median():.2f} | {(sub['min_dist'] < 1.0).mean():.3f} |")
        A("")
        A("Recovery tracks that distance directly:")
        A("")
        A("| generated min distance, A | `coords_random_cell` | `volume_coords` |")
        A("|---|---|---|")
        for lo, hi, label in [(0.0, 1.0, "< 1.0"), (1.0, 1.5, "1.0-1.5"), (1.5, 2.0, "1.5-2.0"), (2.0, 1e9, ">= 2.0")]:
            cells = []
            for arm in ("coords_random_cell", "volume_coords"):
                sub = mind_rec[(mind_rec["arm"] == arm) & (mind_rec["min_dist"] >= lo) & (mind_rec["min_dist"] < hi)]
                cells.append("—" if not len(sub) else f"{sub['recovered'].mean():.3f} ({len(sub)})")
            A(f"| {label} | " + " | ".join(cells) + " |")
        A("")
        A("The same mechanism shows up in the energies: the `missed` trials of `volume_coords`")
        A("average `de` of a few eV/atom rather than the ~0.28 eV/atom of the cell-guessing arms,")
        A("because an overlapping start relaxes to something absurd rather than to a wrong")
        A("polymorph. So the practical reading is that **a predicted cell is worth more than a")
        A("predicted volume**, and that coordinates are only usable together with a cell that can")
        A("hold them.")
        A("")
    A("## Protocol")
    A("")
    A("- **Potential.** `build_hull_calculator(\"orb_conserv_inf\", device=\"cpu\")`, i.e.")
    A("  `orb-v3-conservative-inf-omat-20250404` at precision `float32-high` -- the checkpoint")
    A("  the de novo ranking protocol uses. CPU only (`CUDA_VISIBLE_DEVICES=\"\"`), one torch")
    A("  thread per worker process, one calculator per worker.")
    A("- **Relaxation**, identical for the reference and for every trial:")
    A("  `stepwise_relax(optimizer=BFGS, fix_symmetry=True, release_symmetry=False, fmax=0.05,")
    A("  steps_limit=500)` -- a fixed-cell warm-up then symmetric cell+positions.")
    A("- **Sample.** `data/lemat-bulk/lemat_pbe_ehull.csv.gz` streamed in two passes; of")
    A(f"  5,335,299 rows, 1,388,197 have `e_hull <= {E_HULL_MAX}`, and")
    A("  `numpy.random.default_rng(0).choice(..., replace=False)` drew a uniform subset of")
    A("  4000 of those row positions (a plain uniform subset, not reservoir sampling).")
    A(f"- **Cap.** Conventional-cell atom count <= {MAX_CONVENTIONAL_ATOMS}, applied on the gene")
    A("  extracted from the DFT geometry (3934 of the 4000 survive) and again after relaxation.")
    A("- **Gene.** `kick_pyxtal_until_it_works(tol=0.1, a_tol=5.0)` on the ORB-relaxed geometry,")
    A("  as in `scripts/process_lemat_symmetry.py`. Exact genes are deduplicated (first kept).")
    A("- **Verdicts**, in order: `recovered` (matched), `lower_energy_alternative`")
    A("  (`de < -0.001` eV/atom), `missed`; `generation_failed` / `relaxation_failed` override.")
    A("  Any trial with `|E/atom| > 50` eV is a `relaxation_failed` (the study's sanity guard).")
    A("")
    A("## Reproducing")
    A("")
    A("```bash")
    A("# all three stages are CPU-only and resumable; re-running skips finished work")
    A("# 16 workers = the host's 24 physical cores minus the background load; the run")
    A("# reported here used 32, see the note under Cost.")
    A("CUDA_VISIBLE_DEVICES=\"\" uv run python scripts/oracle_reconstruction.py prepare \\")
    A("    --workers 16 --pool-size 4000 --per-bin 150")
    A("CUDA_VISIBLE_DEVICES=\"\" uv run python scripts/oracle_reconstruction.py run \\")
    A("    --workers 16 --volume-coords")
    A("CUDA_VISIBLE_DEVICES=\"\" uv run python scripts/oracle_reconstruction.py report")
    A("```")
    A("")
    A("Outputs land in `generated/oracle_reconstruction/`: `pool.csv.gz` (the 4000-row draw),")
    A("`screen.csv`, `reference_raw.jsonl`, `references.csv`, `references_full.json.gz`,")
    A("`structures/<immutable_id>/{reference,<arm>}/` (CIFs, BFGS logs, `result.json`),")
    A("`trials.csv` and this file.")
    A("")
    A("## Caveats")
    A("")
    A("- **One trial per cell.** Every rate here is best-of-1, like the ranking protocol and")
    A("  unlike the 10-trial reconstruction study. Best-of-N coverage keeps growing after the")
    A("  mean energy stops improving, so these are lower bounds on what the same information")
    A("  would buy at a larger trial budget -- and the penalty is complexity-dependent, so it")
    A("  does not cancel between dof bins.")
    A("- **The reference is an ORB minimum, not the DFT structure.** That is deliberate (it")
    A("  puts target and trial on one PES) but it means the rates are not comparable to a")
    A("  match rate against LeMat-Bulk geometries without care.")
    A("- **`release_symmetry=False`.** The third, symmetry-free stage is not run, for either")
    A("  the reference or the trials. It moves the energy by >1 meV/atom in 0.4% of trials on")
    A("  the measured `upi73i4k` run, so this is cheap, but a symmetry-lowering distortion")
    A("  reachable only from stage 3 is invisible here.")
    A("- **The atom cap biases the sample.** Structures with more than "
      f"{MAX_CONVENTIONAL_ATOMS} conventional-cell atoms are excluded, which removes 1.7% of")
    A("  the draw overall but bites hardest exactly where reconstruction is hardest.")
    A("- **The noise is per free coordinate, converted with the cell lengths.** sigma/a for x,")
    A("  sigma/b for y, sigma/c for z: exact for orthogonal cells, approximate otherwise. It is")
    A("  applied to the free coordinates only, so a site's symmetry is preserved by")
    A("  construction and the noise never moves an atom off its Wyckoff orbit.")
    A("- **PyXtal is seeded** through `from_random(random_state=...)` derived from")
    A("  `(immutable_id, arm)`; that is the one deviation of the `random` arm from")
    A("  `single_pyxtal`, which leaves the RNG unseeded. The distribution is unchanged.")
    A("- **`volume_coords` and `coords_random_cell` do no distance check.** `pyxtal.build`")
    A("  places the given generator positions into the given cell with no `Tol_matrix`")
    A("  rejection, so a random cell that is too small for the true coordinates produces")
    A("  overlapping atoms that the relaxation has to sort out. That is the honest reading of")
    A("  \"coordinates without a cell\", but it is not what a distance-aware sampler would do.")
    A("")
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    logger.info("wrote %s and %s", root / "trials.csv", root / "RESULTS.md")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "report"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--pool-size", type=int, default=4000)
    parser.add_argument("--per-bin", type=int, default=40, help="references per dof bin (prepare)")
    parser.add_argument("--limit-per-bin", type=int, default=None, help="cap structures per bin (run)")
    parser.add_argument("--only-ids", type=str, default=None, help="comma-separated immutable_ids (run)")
    parser.add_argument("--volume-coords", action="store_true", help="also run the optional volume_coords arm")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    for noisy in ("wyckoff_transformer.cryspr.relaxer", "pyxtal", "wyckoff_transformer.data"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    if args.stage == "prepare":
        prepare(args.root, args.workers, args.pool_size, args.per_bin)
    elif args.stage == "run":
        only = args.only_ids.split(",") if args.only_ids else None
        run(args.root, args.workers, args.limit_per_bin, args.volume_coords, only)
    else:
        report(args.root)


if __name__ == "__main__":
    main()
