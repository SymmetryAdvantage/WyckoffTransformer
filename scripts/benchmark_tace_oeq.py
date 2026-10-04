#!/usr/bin/env python3
"""Compare a TACE checkpoint's e3nn and OEQ ASE paths on identical structures.

Run in the relaxation worker's environment, with its GPU otherwise idle.
Each backend runs in a fresh subprocess in baseline/OEQ/OEQ/baseline order.
Outputs are diagnostic working copies; publish them with the study's W&B
compatibility uploader before using the decision to change a worker.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def prepare_cases(args) -> dict:
    from ase import Atoms
    from ase.io import iread, read, write

    requested = [(886, 1), (887, 0), (890, 0)]
    draws = {}
    for atoms in iread(args.input / "pyxtal.extxyz", format="extxyz"):
        key = int(atoms.info["gene"]), int(atoms.info["trial"])
        if key in requested:
            draws[key] = atoms
    if set(draws) != set(requested):
        raise ValueError("Missing requested common raw draws")
    with (args.arm / "trials.csv").open(newline="") as stream:
        rows = {(int(row["gene"]), int(row["trial"])): row
                for row in csv.DictReader(stream)}
    cases = [("Si2", Atoms("Si2", scaled_positions=[[0, 0, 0], [.25, .25, .25]],
                           cell=[5.43] * 3, pbc=True))]
    for key in requested:
        row = rows[key]
        if row["status"] != "ok":
            raise ValueError(f"Benchmark trial {key} is not successful")
        label = f"gene{key[0]}_trial{key[1]}"
        cases.append((label + "_raw", draws[key]))
        cases.append((label + "_kept", read(args.arm / row["cif"])))
    cases_dir = args.output / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, atoms in cases:
        path = cases_dir / f"{name}.extxyz"
        atoms = atoms.copy()
        atoms.calc = None
        atoms.set_constraint()
        write(path, atoms, format="extxyz")
        records.append({"name": name, "path": str(path), "n_atoms": len(atoms),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = json.loads((args.arm / "study_manifest.json").read_text())
    payload = {"date_utc": datetime.now(timezone.utc).isoformat(),
               "wyformer_commit": subprocess.check_output(
                   ["git", "rev-parse", "HEAD"], text=True).strip(),
               "arm_manifest": manifest, "cases": records,
               "relax_case": "gene886_trial1_raw", "relax_seed_identity": [886, 1],
               "repetitions": args.repetitions, "warmups": 2,
               "order": ["baseline", "oeq", "oeq", "baseline"]}
    write_json(args.output / "benchmark_manifest.json", payload)
    return payload


def child(args) -> None:
    import numpy as np
    import torch
    from ase.io import read
    from wyckoff_transformer.cryspr.calculator import resolve_model_path
    from wyckoff_transformer.cryspr.generator import _trial_seed, relax_trial
    from wyckoff_transformer.cryspr.relaxer import _print_logm_warning_once
    from scripts.run_mlip_bias_relax import trial_timeout
    from tace.interface.ase.calculator import TACEAseCalc

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _print_logm_warning_once()
    manifest = json.loads((args.output / "benchmark_manifest.json").read_text())
    result = {"backend": args.backend, "pass": args.pass_index,
              "date_utc": datetime.now(timezone.utc).isoformat(),
              "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0),
              "threads": torch.get_num_threads(), "cases": []}
    output_path = args.output / f"pass_{args.pass_index}_{args.backend}.json"
    try:
        started = time.perf_counter()
        checkpoint = resolve_model_path(manifest["arm_manifest"]["checkpoint_id"])
        result["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        calc = TACEAseCalc(model=str(checkpoint), device="cuda", dtype="float32",
                           enable_oeq=args.backend == "oeq", enable_cue=False,
                           enable_eqt=False, enable_compile=False)
        torch.cuda.synchronize()
        result["setup_seconds"] = time.perf_counter() - started
        result["oeq_modules"] = sum("openequivariance" in type(m).__module__
                                     for m in calc.model.modules())
        result["oeq_wrapper_modules"] = sum("._oeq." in type(m).__module__
                                             for m in calc.model.modules())
        if args.backend == "oeq" and result["oeq_modules"] == 0:
            raise RuntimeError("OEQ enabled but checkpoint has no OEQ modules")
        if args.backend == "baseline" and result["oeq_modules"]:
            raise RuntimeError("Baseline unexpectedly contains OEQ modules")
        for case in manifest["cases"]:
            atoms = read(case["path"])
            atoms.calc = calc
            for _ in range(manifest["warmups"]):
                calc.reset()
                atoms.get_potential_energy()
                atoms.get_forces()
                atoms.get_stress()
                torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(manifest["repetitions"]):
                calc.reset()  # Every timing includes an actual model evaluation.
                torch.cuda.synchronize()
                started = time.perf_counter()
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                stress = atoms.get_stress()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
            if not np.isfinite(energy) or not np.isfinite(forces).all() or not np.isfinite(stress).all():
                raise ValueError(f"Non-finite output for {case['name']}")
            result["cases"].append({"name": case["name"], "n_atoms": len(atoms),
                                    "energy_ev": float(energy), "forces": forces.tolist(),
                                    "stress": stress.tolist(), "seconds": times,
                                    "median_seconds": statistics.median(times),
                                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            write_json(output_path, result)
            print(f"{args.backend} {case['name']}: {statistics.median(times):.4f} s", flush=True)
        case = next(c for c in manifest["cases"] if c["name"] == manifest["relax_case"])
        calc.reset()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with trial_timeout(300):
            kept, energy, (fixed, fixed_energy) = relax_trial(
                atoms_in=read(case["path"]), calculator=calc,
                trial_dir=args.output / f"relax_{args.pass_index}_{args.backend}",
                label=f"OEQ benchmark {args.backend}", release_symmetry=True,
                rattle=True, seed=_trial_seed(*manifest["relax_seed_identity"]),
                fmax=manifest["arm_manifest"]["fmax_ev_per_angstrom"])
        torch.cuda.synchronize()
        result["relaxation"] = {"seconds": time.perf_counter() - started,
                                 "energy_ev_per_atom": energy / len(kept),
                                 "fixed_energy_ev_per_atom": fixed_energy / len(fixed),
                                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                 "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        result["status"] = "ok"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    write_json(output_path, result)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}), flush=True)


def summarize(args, manifest, passes) -> dict:
    import numpy as np

    result = {"date_utc": datetime.now(timezone.utc).isoformat(),
              "manifest": manifest, "passes": passes,
              "tolerances": {"energy_ev_per_atom_atol": 1e-5,
                             "forces_atol": 1e-4, "forces_rtol": 1e-4,
                             "stress_atol": 1e-5, "stress_rtol": 1e-4,
                             "relaxed_energy_ev_per_atom_atol": 1e-3},
              "switch": False}
    if any(p.get("status") != "ok" for p in passes):
        result["reason"] = "At least one benchmark pass failed"
        return result
    baseline = [p for p in passes if p["backend"] == "baseline"]
    oeq = [p for p in passes if p["backend"] == "oeq"]
    if len({p["checkpoint_sha256"] for p in passes}) != 1:
        result["reason"] = "Checkpoint hashes differ"
        return result
    comparisons = []
    for base, accelerated in zip(baseline, oeq):
        for left, right in zip(base["cases"], accelerated["cases"]):
            if left["name"] != right["name"]:
                raise ValueError("Case order differs")
            energy_error = abs(left["energy_ev"] - right["energy_ev"]) / left["n_atoms"]
            forces_ok = np.allclose(left["forces"], right["forces"], atol=1e-4, rtol=1e-4)
            stress_ok = np.allclose(left["stress"], right["stress"], atol=1e-5, rtol=1e-4)
            comparisons.append({"name": left["name"], "baseline_pass": base["pass"],
                                "energy_error_ev_per_atom": energy_error,
                                "max_force_error_ev_per_angstrom": float(np.max(
                                    np.abs(np.array(left["forces"]) - right["forces"]))),
                                "max_stress_error_ev_per_angstrom3": float(np.max(
                                    np.abs(np.array(left["stress"]) - right["stress"]))),
                                "correct": bool(energy_error <= 1e-5 and forces_ok and stress_ok)})
    base_case_time = statistics.median(sum(c["median_seconds"] for c in p["cases"]) for p in baseline)
    oeq_case_time = statistics.median(sum(c["median_seconds"] for c in p["cases"]) for p in oeq)
    base_relax_time = statistics.median(p["relaxation"]["seconds"] for p in baseline)
    oeq_relax_time = statistics.median(p["relaxation"]["seconds"] for p in oeq)
    relaxed_errors = [max(abs(b["relaxation"][key] - o["relaxation"][key])
                          for key in ("energy_ev_per_atom", "fixed_energy_ev_per_atom"))
                      for b, o in zip(baseline, oeq)]
    correct = all(c["correct"] for c in comparisons) and max(relaxed_errors) <= 1e-3
    faster = base_case_time / oeq_case_time > 1.05 and base_relax_time / oeq_relax_time > 1.05
    result.update(comparisons=comparisons, correct=correct, faster=faster,
                  single_point_speedup=base_case_time / oeq_case_time,
                  relaxation_speedup=base_relax_time / oeq_relax_time,
                  baseline_relaxation_seconds=base_relax_time,
                  oeq_relaxation_seconds=oeq_relax_time,
                  relaxed_energy_errors_ev_per_atom=relaxed_errors,
                  switch=bool(correct and faster),
                  reason="Passed equivalence and speed checks" if correct and faster else
                         "Equivalence or >5% speed improvement was not established")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--backend", choices=("baseline", "oeq"))
    parser.add_argument("--pass-index", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.backend:
        child(args)
        return
    manifest = prepare_cases(args)
    passes = []
    for index, backend in enumerate(manifest["order"]):
        environment = os.environ.copy()
        environment.update(TACE_USE_OEQ="1" if backend == "oeq" else "0",
                           TACE_USE_CUE="0", TACE_USE_EQT="0", TACE_USE_COMPILE="0")
        command = [sys.executable, __file__, "--input", str(args.input),
                   "--arm", str(args.arm), "--output", str(args.output),
                   "--backend", backend, "--pass-index", str(index)]
        subprocess.run(command, env=environment, check=True)
        passes.append(json.loads((args.output / f"pass_{index}_{backend}.json").read_text()))
        if passes[-1].get("status") != "ok":
            break
    result = summarize(args, manifest, passes)
    write_json(args.output / "benchmark_result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("manifest", "passes", "comparisons")}), flush=True)


if __name__ == "__main__":
    main()
