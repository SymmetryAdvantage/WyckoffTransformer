"""CLI for the de novo ranking protocol.

Three stages, each resumable from the previous one's output::

    wyformer-protocol genes.json.gz --output-dir run/ --stage screen
    wyformer-protocol genes.json.gz --output-dir run/ --stage relax --cores 20
    wyformer-protocol genes.json.gz --output-dir run/ --stage score

``screen`` needs no potential and takes seconds.  ``relax`` is the only
expensive stage.  ``score`` is cheap but depends on the relaxed structures, so
it is kept separate to be re-runnable without repeating the relaxations.

All three run in this environment: validity, novelty and the hull energy are
implemented in :mod:`wyckoff_transformer.evaluation` rather than imported from
LeMat-GenBench.
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

#: Set by :func:`_init_worker` in each pool process.  ``_WORKER_DEVICE`` is the
#: device as requested on the command line, kept for reporting;
#: ``_WORKER_TORCH_DEVICE`` is what torch is told, after the pinning below
#: renumbers the visible cards.
_WORKER_DEVICE: Optional[str] = None
_WORKER_TORCH_DEVICE: Optional[str] = None
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


def _pin_visible_device(device: str) -> str:
    """Hide every GPU but the claimed one, and return its new device string.

    Passing ``device="cuda:N"`` to the calculator is not enough to keep a worker
    off the other cards.  Anything that reaches for the *current* device instead
    of the given one -- ``torch.cuda.synchronize()``, a tensor built with
    ``device="cuda"``, a library's own default -- lands on ``cuda:0`` and leaves
    a CUDA primary context there, ~200 MB per worker.  With twenty workers that
    is 4 GB taken from a card someone else is training on.

    ``CUDA_VISIBLE_DEVICES`` makes it impossible rather than unlikely: the
    driver reads it at initialisation, which has not happened yet in a freshly
    spawned worker, and the claimed card is then the only one that exists, at
    index 0.

    Args:
        device: The device this worker claimed, e.g. ``"cuda:1"`` or ``"cpu"``.

    Returns:
        The device string to hand to torch afterwards.
    """
    if not device.startswith("cuda"):
        return device
    _, _, index = device.partition(":")
    if not index:  # bare "cuda": no card was named, so pin nothing.
        return device
    os.environ["CUDA_VISIBLE_DEVICES"] = index
    return "cuda:0"


def _init_worker(device_queue, mlip: str) -> None:
    """Claim one device for this process and build its calculator once."""
    global _WORKER_DEVICE, _WORKER_TORCH_DEVICE, _WORKER_CALCULATOR
    for key, value in _SINGLE_THREAD_ENV_VARS.items():
        os.environ[key] = value

    try:
        _WORKER_DEVICE = device_queue.get_nowait()
    except queue.Empty:  # pragma: no cover - pool size matches the queue
        _WORKER_DEVICE = "cpu"
    _WORKER_TORCH_DEVICE = _pin_visible_device(_WORKER_DEVICE)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.set_num_threads(1)

    # Built once per process rather than per gene: loading an MLIP costs far
    # more than relaxing one structure.
    _WORKER_CALCULATOR = build_hull_calculator(mlip, device=_WORKER_TORCH_DEVICE)


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
        f"\n{screen.n_unique} genes to relax "
        f"({len(screen.novel)} with no LeMat-Bulk entry sharing their "
        f"fingerprint, {len(screen.known)} whose relaxed structure the matcher "
        f"still has to rule on; "
        f"{len(screen.valid) - screen.n_unique} duplicates skipped)."
    )


def _already_relaxed(output_dir: Path) -> set[int]:
    """Representative indices that a previous relax run already wrote a row for.

    Read from the structures CSV rather than from the CIF directory: a gene that
    produced no structure has a row and no CIF, and re-relaxing it would only
    fail again.
    """
    path = output_dir / STRUCTURES_FILE
    if not path.is_file():
        return set()
    return set(pd.read_csv(path, usecols=["index"])["index"].astype(int))


def stage_relax(args) -> None:
    """PyXtal + CrySPR on every unique gene.

    Gene-known representatives are relaxed too.  Their fingerprint matching a
    LeMat-Bulk entry only makes them *candidates* for the matcher: the same
    space group with the same elements on the same Wyckoff orbits is not the
    same structure, so a known gene can still relax into a novel one, and
    dropping it here would decide novelty by fingerprint alone.
    """
    genes = load_genes(args.input)
    screen = read_screen(args.output_dir / SCREEN_FILE)
    spec = resolve_hull_mlip(args.mlip)
    slots = resolve_devices(args.cores, args.devices, args.workers_per_device)

    todo = sorted(
        i for i in screen.counts if args.limit is None or i < args.limit
    )
    already = _already_relaxed(args.output_dir) if args.resume else set()
    if already:
        todo = [i for i in todo if i not in already]
        logger.info("Resuming: %d representatives already have a row", len(already))
    logger.info(
        "Relaxing %d representatives with %s on %d worker(s): %s",
        len(todo), args.mlip, len(slots), ", ".join(sorted(set(slots))),
    )

    cif_dir = args.output_dir / CIF_DIR
    cif_dir.mkdir(parents=True, exist_ok=True)

    # Set here, in the parent, and not only in _init_worker: a spawned worker
    # imports pandas (and with it numpy) before the initialiser runs, and MKL
    # fixes its thread count when it is first loaded.  Setting them afterwards
    # leaves every worker multi-threaded, which oversubscribes the machine
    # without making any single relaxation faster.
    for key, value in _SINGLE_THREAD_ENV_VARS.items():
        os.environ.setdefault(key, value)

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

    frame = pd.DataFrame(rows).set_index("index")
    if already:
        # Rows from the earlier invocation, whose CIFs are still on disk.
        previous = pd.read_csv(args.output_dir / STRUCTURES_FILE, index_col="index")
        frame = pd.concat([previous, frame])
        frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.sort_index()
    frame.to_csv(args.output_dir / STRUCTURES_FILE)
    _write_manifest(args, spec, slots, n_relaxed=len(frame))
    print(
        f"{int(frame['has_structure'].fillna(False).sum())}/{len(frame)} genes "
        f"produced a structure -> {args.output_dir / STRUCTURES_FILE}"
    )


def stage_score(args) -> None:
    """Structure validity, uniqueness, novelty and e_above_hull.

    Everything here runs in-process.  Uniqueness and novelty are both the
    two-stage filter from :mod:`wyckoff_transformer.evaluation.novelty`: the
    augmented Wyckoff fingerprint narrows the comparison down to the handful of
    structures that could possibly match, then ``StructureMatcher`` decides.
    The reference for novelty is built per run, since only LeMat-Bulk entries
    whose fingerprint collides with a generated one ever reach the matcher.
    """
    from pymatgen.core import Structure

    from wyckoff_transformer.evaluation.hull_energy import HullEnergyCalculator
    from wyckoff_transformer.evaluation.novelty import (
        NoveltyFilter,
        filter_by_unique_structure,
    )
    from wyckoff_transformer.evaluation.structure_novelty import build_novelty_reference
    from wyckoff_transformer.evaluation.structure_validity import is_valid

    screen = read_screen(args.output_dir / SCREEN_FILE)
    frame = pd.read_csv(args.output_dir / STRUCTURES_FILE, index_col="index")
    cif_dir = args.output_dir / CIF_DIR

    genes = load_genes(args.input)
    # Not persisted by write_screen -- large, and cheap to recompute.
    fingerprinter = GeneFingerprinter()
    hull = HullEnergyCalculator(args.mlip)

    validity, fingerprints, structures, hull_energies = {}, {}, {}, {}
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
        structures[index] = structure
        try:
            fingerprints[index] = fingerprinter.fingerprint(genes[index])
        except Exception as exc:
            logger.warning("Gene %d: no fingerprint (%s)", index, exc)

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
    frame["e_above_hull"] = pd.Series(hull_energies)

    # Only structures that got this far can be unique or novel, and comparing
    # the ones that did not would just cost matcher calls.
    scored = pd.DataFrame(
        {
            "fingerprint": pd.Series(fingerprints),
            "structure": pd.Series(structures),
        }
    ).dropna()
    scored = scored.loc[
        [i for i in scored.index if bool(validity.get(i, False))]
    ]

    unique_index = filter_by_unique_structure(scored).index
    frame["unique_structure"] = pd.Series(
        {index: index in set(unique_index) for index in scored.index}
    )

    reference = build_novelty_reference(
        scored["fingerprint"],
        cache=args.reference_cache,
        splits=tuple(s.strip() for s in args.reference_splits.split(",")),
        lemat_cif_csv=args.lemat_cif_csv,
    )
    novelty_filter = NoveltyFilter(reference)
    frame["novel_structure"] = pd.Series(
        {index: novelty_filter.is_novel(row) for index, row in scored.iterrows()}
    )

    frame.drop(columns=["structure"], errors="ignore").to_csv(
        args.output_dir / STRUCTURES_FILE
    )
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
    relax.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Skip representatives the structures CSV already has a row for, and "
            "merge the new rows into it. For extending an interrupted run, or "
            "one made before the gene-known representatives were relaxed too."
        ),
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
        "--lemat-cif-csv", type=Path,
        default=Path("data/lemat-bulk/lemat_pbe.csv.gz"),
        help=(
            "LeMat-Bulk export with immutable_id and cif. Structure novelty "
            "needs the reference geometries: the Wyckoff cache carries no "
            "coordinates, and StructureMatcher cannot rule on a fingerprint "
            "collision without them."
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
