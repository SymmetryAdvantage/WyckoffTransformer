#!/usr/bin/env python3
"""Probe the study's retained Matbench models and ORB control.

This is a local diagnostic. A missing adapter or dependency is recorded as a
setup failure; it is not called a scientific failure of the MLIP itself.
"""

from __future__ import annotations

import argparse
import importlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from ase import Atoms

from wyckoff_transformer.cryspr.mlips import MLIP_REGISTRY, build_prerelax_calculator
from wyckoff_transformer.evaluation.hull_mlips import HULL_MLIPS

# Matbench Discovery commit 71633e8, default CPS (F1 .5, kappa .4, RMSD .1).
# Preserve original ranks; EquFlash was excluded in favor of EquFlashV2.
STUDY_MODELS = (
    (1, "Prophet-OAME-MBD", "prophet-oame-mbd", 0.9116400),
    (2, "TECE-OAM-RRA-1.0", "tece-oam-rra-1.0", 0.9076267),
    (3, "EquFlashV2", "equflashv2-45m-oam", 0.9072133),
    (4, "EquiformerV3+DeNS-OAM", "equiformer-v3-oam", 0.9022733),
    (5, "GRACE-3L-OAM-L", "grace-3l-oam-l", 0.8999467),
    (6, "PET-OAM-XL", "pet-oam-xl-1.0.0", 0.8984267),
    (7, "TACE-OAM-L", "tace-oam-l", 0.8894000),
    (8, "eSEN-30M-OAM", "esen-30m-oam", 0.8878867),
    (10, "Nequip-OAM-XL", "nequip-oam-xl-0.1", 0.8859600),
    (None, "orb_conserv_inf", "control", None),
)


def test_calculator(name: str, device: str, factory: str | None = None) -> dict:
    if factory is None and name not in MLIP_REGISTRY and name not in HULL_MLIPS:
        return {"status": "builder_missing", "reason": "No ASE adapter in WyFormer registry"}
    spec = MLIP_REGISTRY.get(name) or HULL_MLIPS.get(name)
    result = {"backend": factory or getattr(spec, "backend", getattr(spec, "builder", None)),
              "checkpoint": getattr(spec, "checkpoint", None)}
    try:
        if factory:
            module, function = factory.split(":", 1)
            calc = getattr(importlib.import_module(module), function)(device=device)
        else:
            calc = build_prerelax_calculator(name, device=device)
    except Exception as exc:
        result.update(status="calculator_failed", reason=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(limit=4))
        return result
    # A neutral, periodic Si probe distinguishes missing dependencies from
    # failures caused by uncommon chemistry in the generated cohort.
    atoms = Atoms("Si2", scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
                  cell=[5.43, 5.43, 5.43], pbc=True)
    atoms.calc = calc
    try:
        result.update(energy_ev=float(atoms.get_potential_energy()),
                      force_shape=list(atoms.get_forces().shape),
                      stress_shape=list(atoms.get_stress().shape), status="ok")
    except Exception as exc:
        result.update(status="single_point_failed", reason=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(limit=4))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", action="append",
                        help="Probe only these model names; repeat for several")
    parser.add_argument("--calculator-factory", help="MODULE:FUNCTION for an external ASE calculator")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    output = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "matbench_commit": "71633e8bdfdfd41d56d64b1d777e5686d9eda3ec",
        "ranking": "default CPS on active models, unique-prototype F1",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "models": [],
    }
    for rank, name, key, cps in STUDY_MODELS:
        if args.model and name not in args.model:
            continue
        row = {"rank": rank, "name": name, "matbench_key": key, "cps": cps}
        row.update(test_calculator(name, args.device, args.calculator_factory))
        output["models"].append(row)
        print(f"{rank or 'control'} {name}: {row['status']} {row.get('reason', '')}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
