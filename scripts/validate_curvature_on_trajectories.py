#!/usr/bin/env python3
"""Validate ORB curvature and the gradient-matched estimator against MP relaxation trajectories.

Reads ``<trajectories>/index.parquet`` and ``tasks/<task_id>.json.gz`` (written by the MP
trajectory extraction, rewritten incrementally) and runs the checks of
:mod:`wyckoff_transformer.cryspr.trajectory_validation` per task:

* ``pair``      T0 energy/gradient consistency and T1 curvature, every consecutive pair
                of every calculation;
* ``boundary``  T3, last step of one calculation against the first of the next;
* ``t2``        the estimator at steps of the last relaxation, ``--t2-offsets`` from its
                end plus its first step, with the DFT drop to the final step;
* ``t4``        DFT-only secant Hessian of the last relaxation, against ORB's at its final
                step.

One JSON line per record in ``<output-dir>/records.jsonl``, written by this process only;
a task is done once its ``task`` line is. ``--poll-seconds`` keeps re-reading the index
and processing new tasks until ``--idle-polls`` polls in a row find nothing new.

``--synthetic N`` instead relaxes N small pilot structures with ORB itself, writes those
trajectories in the task format under ``<output-dir>/synthetic/`` and runs the same
checks on them: with the "DFT" equal to ORB every check must pass.

CPU only (float64 finite-difference Hessians do not fit a shared GPU)::

    CUDA_VISIBLE_DEVICES="" python scripts/validate_curvature_on_trajectories.py --workers 16 --poll-seconds 300
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from wyckoff_transformer.paths import cache_root

logger = logging.getLogger("validate_curvature")
RELAX_IBRION = {1, 2, 3}

_STATE: dict = {}


def _orb_batch_evaluator(model, adapter, max_atoms_per_batch: int):
    import torch

    def evaluate(structures):
        results, start = [], 0
        while start < len(structures):
            chunk, n = [], 0
            while start < len(structures) and (not chunk or n + len(structures[start]) <= max_atoms_per_batch):
                chunk.append(structures[start]); n += len(structures[start]); start += 1
            with torch.enable_grad():
                out = model.predict(adapter.from_ase_atoms_list(chunk, device="cpu"), split=True)
            for i in range(len(chunk)):
                s = out[model.grad_stress_name][i].detach().numpy().reshape(6)
                results.append((float(out["energy"][i].item()), out[model.grad_forces_name][i].detach().numpy(),
                                np.array([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])))
        return results
    return evaluate


def init_worker(args) -> None:
    import torch
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s pid{os.getpid()} %(message)s")
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator
    model, adapter = getattr(pretrained, args.model)(device="cpu", precision="float64", compile=False)
    model = model.eval()
    _STATE.update(args=args, calc=ORBCalculator(model, adapter, device="cpu"),
                  evaluator=_orb_batch_evaluator(model, adapter, args.max_atoms_per_batch))


def _incar(calc: dict) -> dict:
    incar = calc.get("incar") or {}
    return {key: incar.get(key) for key in ("ISIF", "IBRION", "ISMEAR", "SIGMA", "EDIFF", "EDIFFG", "ENCUT", "NSW", "ISYM",
                                            "PREC", "KSPACING", "LREAL", "ADDGRID", "LASPH", "ENMAX", "ENAUG", "ISPIN", "LDAU")}


def process_task(path: str) -> list[dict]:
    from wyckoff_transformer.cryspr import trajectory_validation as tv

    args, calc, evaluator = _STATE["args"], _STATE["calc"], _STATE["evaluator"]
    t0 = time.time()
    task = tv.load_task(path)
    meta = {k: task.get(k) for k in ("task_id", "immutable_id", "group")}
    try:
        frames_by_calc = [tv.frames_of(c, task["energy_key"]) for c in task["calcs"]]
        records = []
        pair_records_by_calc = []
        for ci, (c, frames) in enumerate(zip(task["calcs"], frames_by_calc)):
            incar = _incar(c)
            pairs = tv.analyse_pairs(frames, args.symprec, evaluator, fd_step=args.fd_step)
            pair_records_by_calc.append(pairs)
            for r in pairs:
                records.append({"kind": "pair", **meta, "calc": ci, "calc_name": c.get("name"),
                                "n_calc_steps": len(frames), "n_calcs": len(task["calcs"]), **incar, **r})
        for b in tv.calc_boundaries(frames_by_calc, args.symprec):
            a, z = task["calcs"][b["boundary"]], task["calcs"][b["boundary"] + 1]
            records.append({"kind": "boundary", **meta, **b, "a_name": a.get("name"), "b_name": z.get("name"),
                            "kpoints_same": json.dumps(a.get("kpoints"), sort_keys=True) == json.dumps(z.get("kpoints"), sort_keys=True),
                            "incar_differs": sorted(k for k in set(a.get("incar") or {}) | set(z.get("incar") or {})
                                                    if (a.get("incar") or {}).get(k) != (z.get("incar") or {}).get(k)),
                            **{f"a_{k}": v for k, v in _incar(a).items()},
                            **{f"b_{k}": v for k, v in _incar(z).items()}})

        relax_calcs = [i for i, (c, f) in enumerate(zip(task["calcs"], frames_by_calc))
                       if len(f) >= 2 and (_incar(c).get("IBRION") in RELAX_IBRION or _incar(c).get("IBRION") is None)]
        if relax_calcs and len(frames_by_calc[relax_calcs[-1]][0].atoms) <= args.max_sites:
            ci = relax_calcs[-1]
            frames, pairs, incar = frames_by_calc[ci], pair_records_by_calc[ci], _incar(task["calcs"][ci])
            last = len(frames) - 1
            ks = sorted({last - o for o in args.t2_offsets if last - o >= 0} | {0}, reverse=True)
            n_atoms = len(frames[-1].atoms)
            newton_final = None
            for k in ks:
                estimate, newton, coords = tv.estimate_at(
                    frames[k], calc, evaluator, args.symprec, fd_step=args.fd_step,
                    soft_curvature=args.soft_curvature, relax=not args.no_relax,
                    relax_kwargs={"fmax": args.fmax, "steps": args.max_steps})
                if k == last:
                    newton_final = newton
                q_final, _ = tv.relative_coordinates(coords, frames[-1].atoms)
                to_final = coords.basis.T @ q_final
                p = coords.n_position_modes
                suffix = pairs[k:]
                records.append({
                    "kind": "t2", **meta, "calc": ci, "calc_name": task["calcs"][ci].get("name"), **incar,
                    "k": k, "steps_to_end": last - k, "n_atoms": n_atoms,
                    "dft_drop": frames[k].energy - frames[-1].energy,
                    # The DFT drop k -> final split by the trapezoid rule; sums to dft_drop
                    # up to the T0 residuals of these pairs.
                    "dft_drop_positions": float(-sum(r["dE_trapezoid_positions"] for r in suffix)),
                    "dft_drop_cell": float(-sum(r["dE_trapezoid_cell"] for r in suffix)),
                    "distance_to_final": float(np.linalg.norm(to_final)),
                    "final_strain_fraction": (float(np.sum(to_final[p:] ** 2) / np.sum(to_final ** 2))
                                              if np.sum(to_final ** 2) > 0 else np.nan),
                    "max_abs_force": float(np.abs(frames[k].forces).max()),
                    "max_abs_stress_kbar": float(np.abs(frames[k].stress_kbar_vasp).max()),
                    "hydrostatic_kbar": float(np.trace(frames[k].stress_kbar_vasp) / 3),
                    **estimate})
            t4 = tv.dft_secant_hessian(frames, args.symprec,
                                       mlip_hessian=newton_final.hessian if newton_final is not None else None)
            records.append({"kind": "t4", **meta, "calc": ci, **incar, "n_atoms": n_atoms,
                            "n_calc_steps": len(frames), **t4})
        records.append({"kind": "task", **meta, "seconds": time.time() - t0, "n_calcs": len(task["calcs"]),
                        "frames_per_calc": [len(f) for f in frames_by_calc]})
        return records
    except Exception as exc:  # one bad task must not stop the run
        return [{"kind": "task", **meta, "error": repr(exc), "traceback": traceback.format_exc(limit=8),
                 "seconds": time.time() - t0}]


def _done(out: Path) -> set[str]:
    done = set()
    if out.exists():
        with open(out) as handle:
            for line in handle:
                if line.strip():
                    r = json.loads(line)
                    if r.get("kind") == "task" and "error" not in r:
                        done.add(r["task_id"])
    return done


def _read_index(root: Path) -> pd.DataFrame | None:
    path = root / "index.parquet"
    for _ in range(5):
        try:
            return pd.read_parquet(path)
        except FileNotFoundError:
            return None
        except Exception:  # being rewritten
            time.sleep(2)
    return None


def _write_synthetic(args) -> Path:
    import torch
    from ase import Atoms
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator
    from wyckoff_transformer.cryspr import trajectory_validation as tv

    torch.set_num_threads(4)
    root = args.output_dir / "synthetic"
    (root / "tasks").mkdir(parents=True, exist_ok=True)
    sample = pd.read_parquet(args.pilot_sample)
    sample = sample[(sample.source == "mp") & (sample.nsites <= 12) & (sample.nsites >= 3)].head(args.synthetic)
    model, adapter = getattr(pretrained, args.model)(device="cpu", precision="float64", compile=False)
    calc = ORBCalculator(model.eval(), adapter, device="cpu")
    rows = []
    rng = np.random.default_rng(0)
    for row in sample.itertuples():
        matrix = lambda v: np.array([np.asarray(x, float) for x in v])
        atoms = Atoms(symbols=list(row.species_at_sites), positions=matrix(row.cartesian_site_positions),
                      cell=matrix(row.lattice_vectors), pbc=True)
        strain = np.diag(rng.uniform(-0.02, 0.02, 3))
        atoms.set_cell(atoms.cell.array @ (np.eye(3) + strain), scale_atoms=True)
        task = tv.synthetic_task(atoms, calc, args.symprec, relax1_steps=5, fmax=1e-3, max_steps=200,
                                 task_id=f"synthetic-{row.immutable_id}")
        path = root / "tasks" / f"{task['task_id']}.json.gz"
        with gzip.open(path, "wt") as handle:
            json.dump(task, handle)
        rows.append({"immutable_id": row.immutable_id, "task_id": task["task_id"], "group": "synthetic",
                     "nsites": row.nsites, "path": str(path), "status": "ok"})
        logger.info("synthetic %s: %s steps", task["task_id"], [len(c["steps"]) for c in task["calcs"]])
    pd.DataFrame(rows).to_parquet(root / "index.parquet", index=False)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectories", type=Path, default=cache_root() / "mp_trajectories")
    parser.add_argument("--output-dir", type=Path, default=cache_root() / "gradient_matched" / "trajectory_validation")
    parser.add_argument("--pilot-sample", type=Path, default=cache_root() / "gradient_matched" / "pilot" / "sample.parquet")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--model", default="orb_v3_conservative_inf_mpa")
    parser.add_argument("--symprec", type=float, default=1e-3)
    parser.add_argument("--fd-step", type=float, default=1e-3)
    parser.add_argument("--soft-curvature", type=float, default=0.1)
    parser.add_argument("--fmax", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-sites", type=int, default=40, help="T2/T4 only; T0/T1/T3 run on every task.")
    parser.add_argument("--t2-offsets", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--no-relax", action="store_true")
    parser.add_argument("--max-atoms-per-batch", type=int, default=240)
    parser.add_argument("--groups", nargs="+", default=["pilot", "extra", "recovered"])
    parser.add_argument("--status", nargs="+", default=["ok"],
                        help="Index statuses to process ('pending' rows may still be being written).")
    parser.add_argument("--poll-seconds", type=float, default=0)
    parser.add_argument("--idle-polls", type=int, default=6)
    parser.add_argument("--synthetic", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        args.trajectories = _write_synthetic(args)
        args.groups = ["synthetic"]
        args.output_dir = args.output_dir / "synthetic"
    out = args.output_dir / "records.jsonl"

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=init_worker, initargs=(args,)) as pool, open(out, "a") as sink:
        idle, submitted = 0, set()
        while True:
            index = _read_index(args.trajectories)
            todo = []
            if index is not None:
                rows = index[index["group"].isin(args.groups)]
                if args.status is not None:
                    rows = rows[rows["status"].isin(args.status)]
                done = _done(out)
                # The pilot group answers the negative-curvature question; small cells first.
                priority = {g: i for i, g in enumerate(["synthetic", "pilot", "recovered", "extra"])}
                rows = rows.assign(_p=rows["group"].map(priority).fillna(9)).sort_values(["_p", "nsites"])
                for row in rows.itertuples():
                    path = Path(row.path) if "path" in rows and isinstance(row.path, str) and row.path else \
                        args.trajectories / "tasks" / f"{row.task_id}.json.gz"
                    if not path.is_absolute():
                        path = args.trajectories / path
                    if row.task_id in done or row.task_id in submitted or not path.exists():
                        continue
                    todo.append(str(path)); submitted.add(row.task_id)
                if args.limit is not None:
                    todo = todo[:args.limit]
            logger.info("index rows %s; %d new tasks", None if index is None else len(index), len(todo))
            if todo:
                idle = 0
                for i, records in enumerate(pool.imap_unordered(process_task, todo)):
                    for r in records:
                        sink.write(json.dumps(r, default=float) + "\n")
                    sink.flush()
                    if (i + 1) % 10 == 0 or i + 1 == len(todo):
                        logger.info("%d/%d tasks", i + 1, len(todo))
            else:
                idle += 1
            if args.poll_seconds <= 0 or idle >= args.idle_polls or args.synthetic:
                break
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
