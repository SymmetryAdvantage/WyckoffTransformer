"""CLI for the de novo ranking protocol.

Three stages, each resumable from the previous one's output::

    wyformer-protocol genes.json.gz --output-dir run/ --stage screen
    wyformer-protocol genes.json.gz --output-dir run/ --stage relax --cores 20
    wyformer-protocol genes.json.gz --output-dir run/ --stage score

``screen`` needs no potential and takes seconds.  ``relax`` is the only
expensive stage.  ``score`` is cheap but depends on the relaxed structures, so
it is kept separate to be re-runnable without repeating the relaxations.

All three run in this environment: validity, BAWL fingerprinting and the hull
energy are implemented in :mod:`wyckoff_transformer.evaluation`, pinned to
LeMat-GenBench's own implementations by tests rather than importing them.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import queue
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd

from wyckoff_transformer.evaluation.hull_mlips import (
    DEFAULT_HULL_MLIP,
    HULL_MLIPS,
    build_hull_calculator,
    resolve_hull_mlip,
)
from wyckoff_transformer.evaluation.protocol import (
    DEFAULT_REFERENCE_CACHE,
    DEFAULT_REFERENCE_SPLITS,
    GeneFingerprinter,
    funnel,
    load_genes,
    load_reference_fingerprints,
    read_screen,
    screen_genes,
    write_screen,
)

logger = logging.getLogger(__name__)

SCREEN_FILE = "screen.json"
STRUCTURES_FILE = "structures.csv"
FUNNEL_FILE = "funnel.json"
MANIFEST_FILE = "manifest.json"
CIF_DIR = "cifs"

_SINGLE_THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}

#: Set by :func:`_init_worker` in each pool process.
_WORKER_DEVICE: Optional[str] = None
_WORKER_CALCULATOR = None


def resolve_devices(
    cores: Optional[int],
    devices: Optional[str],
    workers_per_device: int,
) -> list[str]:
    """Map the CPU/GPU request onto one device string per worker slot.

    Args:
        cores: Number of CPU worker processes.
        devices: Comma-separated torch devices, e.g. ``"cuda:0,cuda:1"``.
        workers_per_device: Worker processes per GPU.  More than one shares a
            card between processes, which helps when a single relaxation cannot
            saturate it.

    Returns:
        A device string per worker slot; its length is the pool size.

    Raises:
        ValueError: If both or neither of *cores* and *devices* are given, or
            the values are not positive.
    """
    if cores is not None and devices is not None:
        raise ValueError("--cores and --devices are mutually exclusive")
    if devices is not None:
        names = [d.strip() for d in devices.split(",") if d.strip()]
        if not names:
            raise ValueError("--devices is empty")
        if workers_per_device < 1:
            raise ValueError("--workers-per-device must be >= 1")
        return [d for d in names for _ in range(workers_per_device)]
    n = 1 if cores is None else cores
    if n < 1:
        raise ValueError("--cores must be >= 1")
    return ["cpu"] * n


def _init_worker(device_queue, mlip: str) -> None:
    """Claim one device for this process and build its calculator once."""
    global _WORKER_DEVICE, _WORKER_CALCULATOR
    for key, value in _SINGLE_THREAD_ENV_VARS.items():
        os.environ[key] = value
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.set_num_threads(1)

    try:
        _WORKER_DEVICE = device_queue.get_nowait()
    except queue.Empty:  # pragma: no cover - pool size matches the queue
        _WORKER_DEVICE = "cpu"
    # Built once per process rather than per gene: loading an MLIP costs far
    # more than relaxing one structure.
    _WORKER_CALCULATOR = build_hull_calculator(mlip, device=_WORKER_DEVICE)


def _relax_one(
    index: int,
    gene: dict,
    output_dir: str,
    mlip: str,
    n_trials: int,
    fmax: float,
    release_symmetry: bool,
) -> dict:
    """Relax one gene in a pool worker.  Returns a row for the structures CSV."""
    from wyckoff_transformer.cryspr.generator import func_run

    started = time.time()
    atoms, formula, energy, energy_per_atom, cif = func_run(
        id_gene=index,
        wyckoffgene=gene,
        calculator=_WORKER_CALCULATOR,
        output_dir=Path(output_dir),
        model_name=mlip,
        n_trials=n_trials,
        release_symmetry=release_symmetry,
        fmax=fmax,
    )
    return {
        "index": index,
        "formula": formula,
        "energy": energy,
        "energy_per_atom": energy_per_atom,
        "n_atoms": len(atoms) if atoms is not None else None,
        "has_structure": atoms is not None,
        "cif": cif,
        "device": _WORKER_DEVICE,
        "seconds": round(time.time() - started, 2),
    }


def stage_screen(args) -> None:
    """Validity, uniqueness with counts, and gene novelty.  No potential."""
    genes = load_genes(args.input)
    reference = load_reference_fingerprints(
        args.reference_cache,
        tuple(s.strip() for s in args.reference_splits.split(",")),
        fingerprint_cache=args.reference_fingerprint_cache,
    )
    screen = screen_genes(genes, reference, GeneFingerprinter())
    write_screen(screen, args.output_dir / SCREEN_FILE)
    print(json.dumps(screen.summary(), indent=2))
    print(
        f"\n{len(screen.novel)} genes to relax "
        f"({len(screen.valid) - len(screen.novel)} valid genes skipped: "
        f"{screen.n_unique - len(screen.novel)} known, "
        f"{len(screen.valid) - screen.n_unique} duplicates)."
    )


def stage_relax(args) -> None:
    """PyXtal + CrySPR on the gene-novel representatives only."""
    genes = load_genes(args.input)
    screen = read_screen(args.output_dir / SCREEN_FILE)
    spec = resolve_hull_mlip(args.mlip)
    slots = resolve_devices(args.cores, args.devices, args.workers_per_device)

    todo = [i for i in screen.novel if args.limit is None or i < args.limit]
    logger.info(
        "Relaxing %d gene-novel representatives with %s on %d worker(s): %s",
        len(todo), args.mlip, len(slots), ", ".join(sorted(set(slots))),
    )

    cif_dir = args.output_dir / CIF_DIR
    cif_dir.mkdir(parents=True, exist_ok=True)

    ctx = multiprocessing.get_context("spawn")
    device_queue = ctx.Queue()
    for device in slots:
        device_queue.put(device)

    rows: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=len(slots),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(device_queue, args.mlip),
    ) as pool:
        futures = {
            pool.submit(
                _relax_one,
                index,
                genes[index],
                str(args.output_dir / "cryspr"),
                args.mlip,
                args.n_trials,
                args.fmax,
                args.release_symmetry,
            ): index
            for index in todo
        }
        for done, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                logger.warning("Gene %d failed: %s", index, exc)
                row = {"index": index, "has_structure": False, "error": str(exc)}
            cif = row.pop("cif", None)
            if cif:
                (cif_dir / f"{index}.cif").write_text(cif, encoding="utf-8")
            rows.append(row)
            if done % 50 == 0 or done == len(futures):
                logger.info("Relaxed %d/%d", done, len(futures))

    frame = pd.DataFrame(rows).set_index("index").sort_index()
    frame.to_csv(args.output_dir / STRUCTURES_FILE)
    _write_manifest(args, spec, slots, n_relaxed=len(frame))
    print(
        f"{int(frame['has_structure'].fillna(False).sum())}/{len(frame)} genes "
        f"produced a structure -> {args.output_dir / STRUCTURES_FILE}"
    )


def stage_score(args) -> None:
    """Structure validity, BAWL uniqueness and novelty, and e_above_hull.

    Everything here runs in-process: validity, fingerprinting and the hull
    energy are our own implementations, pinned to LeMat-GenBench's by
    ``tests/test_genbench_equivalence.py``.  The only thing still sourced from
    that project is the reference fingerprint parquet, which is data.
    """
    from pymatgen.core import Structure

    from wyckoff_transformer.evaluation.bawl import BawlFingerprinter, get_fingerprint
    from wyckoff_transformer.evaluation.bawl_reference import load_bawl_reference
    from wyckoff_transformer.evaluation.hull_energy import HullEnergyCalculator
    from wyckoff_transformer.evaluation.structure_validity import is_valid

    screen = read_screen(args.output_dir / SCREEN_FILE)
    frame = pd.read_csv(args.output_dir / STRUCTURES_FILE, index_col="index")
    cif_dir = args.output_dir / CIF_DIR

    hull = HullEnergyCalculator(args.mlip)
    fingerprinter = BawlFingerprinter(shorten=args.short_bawl)
    reference = load_bawl_reference(
        args.bawl_reference, cache=args.bawl_reference_cache
    )

    validity, fingerprints, hull_energies = {}, {}, {}
    for index in frame.index:
        cif_path = cif_dir / f"{index}.cif"
        if not cif_path.is_file():
            continue
        try:
            structure = Structure.from_file(cif_path)
        except Exception as exc:
            logger.warning("Gene %d: unreadable CIF (%s)", index, exc)
            continue

        validity[index] = is_valid(structure)
        fingerprints[index] = get_fingerprint(structure, fingerprinter)

        energy = frame.at[index, "energy"]
        if pd.notna(energy):
            try:
                # The energy came from the same potential that defines this
                # hull, which is why --mlip is restricted to published hulls.
                hull_energies[index] = hull.energy_above_hull(
                    float(energy), structure.composition
                )
            except ValueError as exc:
                logger.warning("Gene %d: e_above_hull failed (%s)", index, exc)

    frame["valid_structure"] = pd.Series(validity)
    frame["bawl_fingerprint"] = pd.Series(fingerprints)
    frame["e_above_hull"] = pd.Series(hull_energies)

    # Uniqueness within the generated batch: first occurrence of each BAWL hash
    # wins.  Rows without a fingerprint cannot be shown unique, so they are not.
    seen: set[str] = set()
    unique = {}
    for index, fingerprint in frame["bawl_fingerprint"].items():
        if not isinstance(fingerprint, str):
            unique[index] = False
            continue
        unique[index] = fingerprint not in seen
        seen.add(fingerprint)
    frame["bawl_unique"] = pd.Series(unique)

    frame["bawl_novel"] = pd.Series(
        {
            index: isinstance(fingerprint, str) and fingerprint not in reference
            for index, fingerprint in frame["bawl_fingerprint"].items()
        }
    )

    frame.to_csv(args.output_dir / STRUCTURES_FILE)
    report = funnel(screen, frame)
    (args.output_dir / FUNNEL_FILE).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def _write_manifest(args, spec, slots: list[str], n_relaxed: int) -> None:
    """Record exactly what produced these numbers.

    The MLIP identity matters more than usual here: e_above_hull is only
    meaningful against the hull built with the same potential, and at least one
    of the published hulls (mace_mp) cannot be tied to a checkpoint from source.
    """
    manifest = {
        "input": str(args.input),
        "mlip": args.mlip,
        "hull_type": spec.hull_type,
        "checkpoint": spec.checkpoint,
        "mlip_note": spec.note or None,
        "n_trials": args.n_trials,
        "release_symmetry": args.release_symmetry,
        "fmax": args.fmax,
        "n_relaxed": n_relaxed,
        "workers": len(slots),
        "devices": sorted(set(slots)),
        "reference_cache": str(args.reference_cache),
        "reference_splits": args.reference_splits,
    }
    (args.output_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wyformer-protocol",
        description=(
            "Rank a WyFormer variant by MetaSUN per generated gene. Filters that "
            "need no potential (validity, uniqueness, gene novelty) run first, so "
            "only gene-novel structures are relaxed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="JSON(.gz) list of Wyckoff genes.")
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for all stage outputs; stages read each other's files here.",
    )
    parser.add_argument(
        "--stage", choices=("screen", "relax", "score", "all"), default="all",
        help="Which stage to run. 'all' runs screen and relax, then score.",
    )
    parser.add_argument(
        "--mlip", type=str, default=DEFAULT_HULL_MLIP, choices=sorted(HULL_MLIPS),
        help=(
            "Potential to relax and evaluate with. Restricted to MLIPs that "
            "LeMat-Bulk publishes a convex hull for, because e_above_hull is only "
            "meaningful when energy and hull come from the same model."
        ),
    )

    hardware = parser.add_argument_group("hardware")
    hardware.add_argument(
        "--cores", type=int, default=None,
        help="Run on CPU with this many worker processes (each single-threaded).",
    )
    hardware.add_argument(
        "--devices", type=str, default=None,
        help="Run on GPU: comma-separated torch devices, e.g. 'cuda:0,cuda:1'.",
    )
    hardware.add_argument(
        "--workers-per-device", type=int, default=1,
        help="Worker processes per GPU. Ignored with --cores.",
    )

    relax = parser.add_argument_group("relaxation")
    relax.add_argument(
        "--n-trials", type=int, default=1,
        help="PyXtal trials per gene. The protocol's default is 1.",
    )
    relax.add_argument(
        "--fmax", type=float, default=0.05, help="Force convergence in eV/A.",
    )
    relax.add_argument(
        "--release-symmetry", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Run the final symmetry-free relaxation stage. Off by default, "
            "giving the 2-stage schedule the protocol calls for."
        ),
    )
    relax.add_argument(
        "--limit", type=int, default=None,
        help="Only relax genes with an index below this. For smoke tests.",
    )

    reference = parser.add_argument_group("references")
    reference.add_argument(
        "--reference-cache", type=Path, default=DEFAULT_REFERENCE_CACHE,
        help="Cached LeMat-Bulk in the Wyckoff representation, for gene novelty.",
    )
    reference.add_argument(
        "--reference-splits", type=str, default=",".join(DEFAULT_REFERENCE_SPLITS),
        help="Splits of that cache to treat as known.",
    )
    reference.add_argument(
        "--reference-fingerprint-cache", type=Path,
        default=Path("cache/lemat_bulk_ehull/gene_fingerprints.pkl.gz"),
        help=(
            "Where to persist the reference fingerprint set. Computing it from "
            "4M rows takes minutes; every variant evaluation reuses the same set, "
            "so it is written once and loaded thereafter."
        ),
    )
    reference.add_argument(
        "--bawl-reference", type=Path,
        default=Path(
            os.environ.get("BAWL_REFERENCE_PARQUET", "data/unique_fingerprints.parquet")
        ),
        help=(
            "LeMat-Bulk BAWL fingerprints, for structure novelty. This is the "
            "only file the score stage still takes from LeMat-GenBench, where it "
            "ships as data/augmented_fingerprints/unique_fingerprints.parquet; "
            "copy it somewhere stable. Defaults to $BAWL_REFERENCE_PARQUET."
        ),
    )
    reference.add_argument(
        "--bawl-reference-cache", type=Path,
        default=Path("cache/bawl_reference.pkl.gz"),
        help="Compact cache of that fingerprint set, written on first load.",
    )
    reference.add_argument(
        "--short-bawl", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Use the short BAWL hash, which omits the space-group label. On by "
            "default, matching LeMat-GenBench's leaderboard configuration."
        ),
    )

    parser.add_argument("--debug", action="store_true", help="DEBUG-level logging.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stages = ("screen", "relax", "score") if args.stage == "all" else (args.stage,)
    for stage in stages:
        logger.info("=== stage: %s ===", stage)
        {"screen": stage_screen, "relax": stage_relax, "score": stage_score}[stage](args)


if __name__ == "__main__":
    main()
