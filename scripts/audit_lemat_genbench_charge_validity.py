"""Audit LeMat-GenBench ChargeNeutralityMetric on ICSD-backed structures.

Streams ICSD-backed structures (``icsd_backed == 1.0``) from a WyFormer dataset
cache and CSV store into a worker running ``lemat_genbench`` (optionally in a
separate Python environment via ``--lemat-python``), evaluates
``ChargeNeutralityMetric.compute_structure`` on every structure, and prints a
per-bucket breakdown of all rejected entries.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

CHARGE_TOLERANCE = 0.1
DILUTE_FRACTION_THRESHOLD = 0.85


def _eval_cif_worker(record: tuple[str, str, float | None]) -> dict[str, Any]:
    """Evaluate one structure with LeMat-GenBench's ChargeNeutralityMetric."""
    material_id, cif_text, e_hull = record
    warnings.filterwarnings("ignore")

    from pymatgen.analysis.bond_valence import BVAnalyzer
    from pymatgen.core import Composition, Structure
    from smact.metallicity import metallicity_score

    logging.disable(logging.CRITICAL)
    try:
        from lemat_genbench.metrics.validity_metrics import ChargeNeutralityMetric
    finally:
        logging.disable(logging.NOTSET)

    try:
        structure = Structure.from_str(cif_text, fmt="cif")
    except Exception as exc:
        return {
            "material_id": material_id,
            "status": "parse_error",
            "error": str(exc),
            "e_hull": e_hull,
        }

    try:
        charge_dev = float(
            ChargeNeutralityMetric.compute_structure(structure, bv_analyzer=BVAnalyzer())
        )
    except Exception as exc:
        return {
            "material_id": material_id,
            "status": "compute_error",
            "error": str(exc),
            "formula": structure.composition.reduced_formula,
            "e_hull": e_hull,
        }

    comp = structure.composition.reduced_composition
    metal_score = float(metallicity_score(Composition(structure.formula)))
    all_mm = all(el.is_metal or el.is_metalloid for el in comp.elements)
    fractions = {str(el): float(comp.get_atomic_fraction(el)) for el in comp.elements}
    dominant_element = max(fractions, key=fractions.get)
    max_atomic_fraction = fractions[dominant_element]

    return {
        "material_id": material_id,
        "status": "ok",
        "formula": comp.reduced_formula,
        "charge_deviation": charge_dev,
        "e_hull": e_hull,
        "metal_score": metal_score,
        "all_metal_or_metalloid": all_mm,
        "dominant_element": dominant_element,
        "max_atomic_fraction": max_atomic_fraction,
        "elements": sorted(fractions),
    }


def _sort_key_ehull(item: dict[str, Any]) -> tuple[float, str]:
    e_hull = item["e_hull"]
    return (float("inf") if e_hull is None else float(e_hull), item["material_id"])


def _format_entry(item: dict[str, Any]) -> str:
    ehull_str = "None" if item["e_hull"] is None else f"{item['e_hull']:.4f}"
    return (
        f"  {item['material_id']:14s} {item['formula']:16s} "
        f"dev={item['charge_deviation']:.1f} e_hull={ehull_str} "
        f"metal_score={item['metal_score']:.4f} "
        f"max_frac={item['max_atomic_fraction']:.3f}({item['dominant_element']}) "
        f"els={item['elements']}"
    )


def _run_worker(args: argparse.Namespace) -> int:
    warnings.filterwarnings("ignore")
    # Pre-import in main process so worker forks inherit modules cleanly.
    logging.disable(logging.CRITICAL)
    try:
        from lemat_genbench.metrics.validity_metrics import ChargeNeutralityMetric  # noqa: F401
    finally:
        logging.disable(logging.NOTSET)

    records = [tuple(json.loads(line)) for line in sys.stdin if line.strip()]
    if not records:
        print("No ICSD-backed structures received.", file=sys.stderr)
        return 1

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(_eval_cif_worker, records))

    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] != "ok"]
    rejected = [r for r in ok if r["charge_deviation"] > CHARGE_TOLERANCE]

    near_hull = [
        r for r in ok if r["e_hull"] is not None and r["e_hull"] < args.ehull_threshold
    ]
    near_hull_rejected = [r for r in near_hull if r["charge_deviation"] > CHARGE_TOLERANCE]

    all_mm_rej = sorted(
        [r for r in rejected if r["all_metal_or_metalloid"]], key=_sort_key_ehull
    )
    dilute_rej = sorted(
        [
            r
            for r in rejected
            if not r["all_metal_or_metalloid"]
            and r["max_atomic_fraction"] >= DILUTE_FRACTION_THRESHOLD
        ],
        key=_sort_key_ehull,
    )
    other_rej = sorted(
        [
            r
            for r in rejected
            if not r["all_metal_or_metalloid"]
            and r["max_atomic_fraction"] < DILUTE_FRACTION_THRESHOLD
        ],
        key=_sort_key_ehull,
    )
    near_cutoff_non_mm = [
        r
        for r in rejected
        if not r["all_metal_or_metalloid"] and 0.6 <= r["metal_score"] <= 0.7
    ]

    print(f"Evaluated ok: {len(ok)}, errors: {len(errors)}")
    if errors:
        for err in errors[:5]:
            print(f"  ERROR: {err}")
    print(
        f"Rejected (charge_deviation > {CHARGE_TOLERANCE}): "
        f"{len(rejected)} / {len(ok)} ({100.0 * len(rejected) / len(ok):.2f}%)"
    )
    if near_hull:
        print(
            f"Near-hull (e_hull < {args.ehull_threshold} eV/atom): "
            f"{len(near_hull_rejected)} / {len(near_hull)} "
            f"({100.0 * len(near_hull_rejected) / len(near_hull):.2f}%)"
        )
    print("Mutually exclusive rejection buckets:")
    print(
        f"  1. all_metal_or_metalloid: {len(all_mm_rej)} "
        f"({100.0 * len(all_mm_rej) / len(ok):.2f}% of ICSD)"
    )
    print(
        f"  2. non-all-metal/metalloid with max_atomic_fraction >= "
        f"{DILUTE_FRACTION_THRESHOLD}: {len(dilute_rej)}"
    )
    print(
        f"  3. other non-all-metal/metalloid: {len(other_rej)} "
        f"({100.0 * len(other_rej) / len(ok):.2f}% of ICSD)"
    )
    print(
        f"  (non-all-metal/metalloid with 0.6 <= metal_score <= 0.7: "
        f"{len(near_cutoff_non_mm)})"
    )

    print(f"\n[Bucket 1] All metal/metalloid rejections ({len(all_mm_rej)}):")
    for item in all_mm_rej:
        print(_format_entry(item))

    print(
        f"\n[Bucket 2] Non-all-metal/metalloid with max_frac >= "
        f"{DILUTE_FRACTION_THRESHOLD} ({len(dilute_rej)}):"
    )
    for item in dilute_rej:
        print(_format_entry(item))

    print(f"\n[Bucket 3] Other non-all-metal/metalloid rejections ({len(other_rej)}):")
    for item in other_rej:
        print(_format_entry(item))

    if args.output_json:
        payload = {
            "summary": {
                "evaluated_ok": len(ok),
                "errors": len(errors),
                "rejected": len(rejected),
                "rejected_pct": 100.0 * len(rejected) / len(ok),
                "ehull_threshold": args.ehull_threshold,
                "near_hull_total": len(near_hull),
                "near_hull_rejected": len(near_hull_rejected),
                "near_hull_rejected_pct": (
                    100.0 * len(near_hull_rejected) / len(near_hull) if near_hull else None
                ),
                "bucket_all_metal_or_metalloid": len(all_mm_rej),
                "bucket_dilute_non_all_mm": len(dilute_rej),
                "bucket_other_non_all_mm": len(other_rej),
                "non_all_mm_near_cutoff_06_07": len(near_cutoff_non_mm),
            },
            "rejected": sorted(rejected, key=_sort_key_ehull),
            "errors": errors,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote JSON report to {out_path}")

    return 0


def _run_parent(args: argparse.Namespace) -> int:
    import pandas as pd

    from wyckoff_transformer.dataset_cache import dataset_cache_dir, load_split
    from wyckoff_transformer.paths import data_path

    icsd_cache = dataset_cache_dir(args.icsd_cache)
    data_dir = data_path(args.dataset)

    lines: list[str] = []
    for split in args.splits:
        flags = load_split(icsd_cache, split, columns=["icsd_backed", "energy_above_hull"])
        icsd_ids = set(flags.index[flags["icsd_backed"] == 1.0])
        df = pd.read_csv(data_dir / f"{split}.csv.gz", usecols=["immutable_id", "cif"])
        df = df[df["immutable_id"].isin(icsd_ids)].copy()
        df["e_hull"] = flags["energy_above_hull"].reindex(df["immutable_id"]).values
        for row in df[["immutable_id", "cif", "e_hull"]].itertuples(index=False):
            e_hull = None if pd.isna(row[2]) else float(row[2])
            lines.append(json.dumps([row[0], row[1], e_hull]))

    print(
        f"Dataset={args.dataset}, icsd_cache={args.icsd_cache}, "
        f"splits={args.splits}: {len(lines)} ICSD-backed structures"
    )

    worker_cmd = [
        args.lemat_python or sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--workers",
        str(args.workers),
        "--ehull-threshold",
        str(args.ehull_threshold),
    ]
    if args.output_json:
        worker_cmd.extend(["--output-json", args.output_json])

    proc = subprocess.run(worker_cmd, input="\n".join(lines) + "\n", text=True)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="lemat_bulk_fixed_fmax1_stress")
    parser.add_argument("--icsd-cache", default="lemat_bulk_fixed_fmax1_stress_icsd")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument(
        "--lemat-python",
        default=None,
        help="Python interpreter with lemat_genbench installed (defaults to current Python).",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--ehull-threshold", type=float, default=0.05)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        return _run_worker(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
