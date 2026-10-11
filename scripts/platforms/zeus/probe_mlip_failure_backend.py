"""Compare a proposed recovery backend's energies, forces and stress with its baseline."""

import argparse
import csv
import hashlib
import json
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk
from ase.io import read

from scripts.run_mlip_bias_relax import build_calculator


def evaluate(calc, cases):
    results = []
    for label, original in cases:
        atoms = original.copy()
        atoms.calc = calc
        calc.reset()
        start = time.monotonic()
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        stress = atoms.get_stress()
        torch.cuda.synchronize()
        assert np.isfinite(energy) and np.isfinite(forces).all() and np.isfinite(stress).all()
        results.append({"label": label, "n_atoms": len(atoms), "energy_ev": energy,
                            "forces": forces.tolist(), "stress": stress.tolist(),
                            "seconds": time.monotonic() - start})
        print(label, "seconds", results[-1]["seconds"], "energy", energy, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.arm / "study_manifest.json").read_text())
    rows = list(csv.DictReader((args.arm / "trials.csv").open()))
    kept = min((r for r in rows if r["status"] == "ok" and r["cif"]), key=lambda r: int(r["n_atoms"]))
    medium = next(r for r in rows if r["status"] == "ok" and 20 <= int(r["n_atoms"]) <= 32)
    si = bulk("Si", "diamond", a=5.43)
    distorted = si.copy()
    distorted.positions[0, 0] += 0.03
    distorted.set_cell(distorted.cell * 1.01, scale_atoms=True)
    cases = [("Si2", si), ("Si2_distorted", distorted),
             ("retained_small", read(args.arm / kept["cif"])),
             ("retained_medium", read(args.arm / medium["cif"]))]
    report = {"utc": datetime.now(UTC).isoformat(), "factory": args.factory,
                  "baseline_factory": manifest["calculator_factory"], "checkpoint": manifest["checkpoint_id"],
                  "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                  "source_manifest_sha256": hashlib.sha256((args.arm / "study_manifest.json").read_bytes()).hexdigest(),
                  "passed": False}
    try:
        calc = build_calculator(manifest["mlip"], "cuda", manifest["calculator_factory"])
        report["baseline"] = evaluate(calc, cases)
        del calc
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        calc = build_calculator(manifest["mlip"], "cuda", args.factory)
        report["initialization_seconds"] = time.monotonic() - start
        report["proposed"] = evaluate(calc, cases)
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        errors = []
        for old, new in zip(report["baseline"], report["proposed"]):
            errors.append({"label": old["label"], "energy_error_ev_per_atom": abs(old["energy_ev"]-new["energy_ev"])/old["n_atoms"],
                               "max_force_error_ev_a": float(np.max(np.abs(np.array(old["forces"])-new["forces"]))),
                               "max_stress_error_ev_a3": float(np.max(np.abs(np.array(old["stress"])-new["stress"])))})
        report["errors"] = errors
        report["passed"] = all(e["energy_error_ev_per_atom"] < 1e-4 and
                               e["max_force_error_ev_a"] < 1e-3 and
                               e["max_stress_error_ev_a3"] < 1e-4 for e in errors)
    except Exception as exc:  # noqa: BLE001 - preserve backend diagnostics in the probe artifact
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("PROBE", args.factory, "passed", report["passed"], report.get("error", ""), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
