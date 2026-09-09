"""CLI for the de novo ranking protocol.

Four stages, each resumable from the previous one's output, each writing what
it learned to disk before the next one starts::

    wyformer-protocol genes.json.gz --output-dir run/ --stage screen
    wyformer-protocol genes.json.gz --output-dir run/ --stage generate --pyxtal-cores 12
    wyformer-protocol genes.json.gz --output-dir run/ --stage relax --devices cuda:0,cuda:1
    wyformer-protocol genes.json.gz --output-dir run/ --stage score

They are split by the resource they need, because they need different ones and
nothing is gained by making one wait for another's hardware:

``screen``
    Validity, uniqueness and gene novelty.  One core, no potential; it holds
    the 4M-fingerprint reference in RAM and nothing else.
``generate``
    PyXtal draws, one per (gene, trial).  Pure CPU and embarrassingly
    parallel, with a per-draw timeout: PyXtal's rejection sampling can spin for
    a long time on a gene it cannot satisfy, and one such gene must not hold a
    worker (or, as when this ran inside the relaxation, a GPU) forever.
``relax``
    CrySPR on each generated draw.  The only stage that needs the MLIP, and the
    only expensive one; on a GPU it should be doing nothing but forward passes,
    which is why the PyXtal draws are made before it starts.
``score``
    Structure validity, uniqueness, novelty and e_above_hull.  Fast, but it
    loads large references (the hull parquet, and the LeMat-Bulk geometry of
    every colliding fingerprint) into RAM.

Each stage appends its rows to disk as they complete, so an interrupted run
keeps its finished work and ``--resume`` picks up the rest per *trial*, not per
gene.

All four run in this environment: validity, novelty and the hull energy are
implemented in :mod:`wyckoff_transformer.evaluation` rather than imported from
LeMat-GenBench.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import logging
import multiprocessing
import os
import signal
import time
import warnings
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
    DEFAULT_TRIAL_SCHEDULE,
    GeneFingerprinter,
    funnel,
    load_genes,
    load_reference_fingerprints,
    parse_trial_schedule,
    positional_dof,
    read_screen,
    screen_genes,
    trials_for_dof,
    write_screen,
)

logger = logging.getLogger(__name__)

SCREEN_FILE = "screen.json"
PYXTAL_FILE = "pyxtal.extxyz"
PYXTAL_TRIALS_FILE = "pyxtal.csv"
RELAXATIONS_FILE = "relaxations.csv"
STRUCTURES_FILE = "structures.csv"
FUNNEL_FILE = "funnel.json"
MANIFEST_FILE = "manifest.json"
CIF_DIR = "cifs"
CRYSPR_DIR = "cryspr"

STAGES = ("screen", "generate", "relax", "score")

#: Columns of ``pyxtal.csv``: one row per attempted PyXtal draw.
PYXTAL_COLUMNS = (
    "index", "trial", "status", "formula", "n_atoms",
    "dof_positional", "n_trials", "seconds", "error",
)

#: Columns of ``relaxations.csv``: one row per relaxed draw.
RELAXATION_COLUMNS = (
    "index", "trial", "status", "formula", "energy", "energy_per_atom",
    "n_atoms", "device", "seconds", "cif", "error",
)

_SINGLE_THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}

#: Set by :func:`_init_relax_worker` in each pool process.  ``_WORKER_DEVICE``
#: is the device as requested on the command line, kept for reporting;
#: ``_WORKER_TORCH_DEVICE`` is what torch is told, after the pinning below
#: renumbers the visible cards.
_WORKER_DEVICE: Optional[str] = None
_WORKER_TORCH_DEVICE: Optional[str] = None
_WORKER_CALCULATOR = None


class Timeout(BaseException):
    """Raised in a worker when a task outran its time limit.

    Derived from :class:`BaseException`, not :class:`Exception`, so that the
    broad ``except Exception`` inside PyXtal's callers -- including
    :func:`~wyckoff_transformer.cryspr.generator.single_pyxtal` -- cannot
    swallow it and report a timeout as an ordinary generation failure.
    """


@contextlib.contextmanager
def time_limit(seconds: Optional[float]):
    """Raise :class:`Timeout` in this process if the block outlives *seconds*.

    ``SIGALRM`` rather than a watchdog process: PyXtal's rejection sampling and
    ASE's optimisers are pure Python, so the handler runs at the next bytecode
    boundary, which is never far away.  A worker killed from outside would take
    its pool slot's calculator -- seconds of model loading on a GPU -- with it.

    Args:
        seconds: The limit.  ``None`` or non-positive disables it.
    """
    if not seconds or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handle(signum, frame):  # noqa: ARG001 - signal handler signature
        raise Timeout(f"exceeded {seconds:g} s")

    previous = signal.signal(signal.SIGALRM, handle)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def read_rows(path: Path, columns=None) -> pd.DataFrame:
    """Read a stage log, dropping any row a kill left half-written.

    ``pandas`` alone is not enough: ``on_bad_lines`` drops a line with too many
    fields, but a line cut short by SIGKILL has too *few*, and those it pads
    with NaN instead -- so a trial whose result never finished being written
    would be read back as a trial that had been done.  Counting the fields with
    the ``csv`` module (which knows that a comma inside a quoted error message
    is not a separator) is what tells the two apart.

    Args:
        path: The CSV to read.
        columns: Columns to report for a log that does not exist yet.

    Returns:
        The complete rows, typed as ``pd.read_csv`` would type them.
    """
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(columns=list(columns or ()))
    complete = io.StringIO()
    writer = csv.writer(complete, lineterminator="\n")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return pd.DataFrame(columns=list(columns or ()))
        writer.writerow(header)
        for row in reader:
            if len(row) == len(header):
                writer.writerow(row)
    complete.seek(0)
    return pd.read_csv(complete)


class RowLog:
    """A CSV written row by row, so an interrupted stage keeps what it finished.

    Also the stage's memory of what it has already done: on ``--resume`` the
    existing rows are read back and their ``(index, trial)`` keys -- the first
    two columns of both stage logs -- are what the stage skips.
    """

    def __init__(self, path: Path, columns, resume: bool) -> None:
        self.path = Path(path)
        self.columns = list(columns)
        if self.columns[:2] != ["index", "trial"]:
            raise ValueError("a stage log is keyed by (index, trial), in that order")
        self.done: set[tuple[int, int]] = set()
        append = resume and self.path.is_file()
        if append:
            previous = read_rows(self.path, self.columns)
            self.done = {
                (int(index), int(trial))
                for index, trial in zip(previous["index"], previous["trial"])
            }
            self._end_partial_line()
        self._handle = self.path.open("a" if append else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=self.columns, extrasaction="ignore"
        )
        if not append:
            self._writer.writeheader()
            self._handle.flush()

    def _end_partial_line(self) -> None:
        """Terminate a half-written last line, so the next row starts a new one."""
        with self.path.open("rb+") as handle:
            handle.seek(0, os.SEEK_END)
            if not handle.tell():
                return
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def frame(self) -> pd.DataFrame:
        """Everything written to this log, this run's rows and any resumed ones."""
        if not self._handle.closed:
            self._handle.flush()
        return read_rows(self.path, self.columns)


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


def _init_worker_logging(debug: bool) -> None:
    """Give a spawned worker the parent's logging setup.

    Without this a worker's records go nowhere: ``spawn`` starts a fresh
    interpreter, and a relaxation that failed said so only into a logger with
    no handler.  Every failure is also recorded in the stage's CSV, but the log
    is what makes one visible while the stage is still running.
    """
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s",
    )


def _init_generate_worker(debug: bool) -> None:
    for key, value in _SINGLE_THREAD_ENV_VARS.items():
        os.environ[key] = value
    _init_worker_logging(debug)


def claim_device(counter, slots: list[str]) -> str:
    """Take the next unclaimed slot in *slots*, atomically across processes.

    A shared counter rather than a queue of devices.  ``Queue.put`` hands its
    items to a background feeder thread, so an item is not necessarily readable
    the moment it is queued: workers all spawn at once, and one calling
    ``get_nowait`` before the feeder has flushed gets ``Empty`` even though the
    queue was filled with exactly one device per slot.  That is not a
    hypothetical -- it silently demoted a GPU worker to CPU, which shows up only
    as one idle card and a handful of very slow trials.

    The modulo matters for a worker the pool replaces after a crash: it takes
    the slot round again rather than running out of devices.

    Args:
        counter: A :func:`multiprocessing.Value` of type ``i``, shared with
            every worker and starting at zero.
        slots: One device string per worker slot.

    Returns:
        The device this worker owns.
    """
    with counter.get_lock():
        index = counter.value
        counter.value = index + 1
    return slots[index % len(slots)]


#: Start of SciPy's ``logm`` accuracy warning, which carries its residual in the
#: message text.
_LOGM_WARNING = "logm result may be inaccurate"

#: Relative residual below which that warning is round-off rather than news.
#: SciPy warns above ``1000 * eps`` = 2.2e-13; this keeps four further orders of
#: magnitude in reserve and is still far below anything that could move a
#: relaxation converging at 0.05 eV/A.
LOGM_ROUNDOFF = 1e-9


def _quiet_logm_roundoff(threshold: float = LOGM_ROUNDOFF) -> None:
    """Drop ``logm`` warnings that report round-off; let real ones through.

    ASE's ``FrechetCellFilter`` takes the matrix logarithm of the deformation
    gradient once per optimiser step, and SciPy warns whenever ``logm``'s
    relative residual exceeds ``1000 * eps`` (2.2e-13).  A near-identity
    deformation gradient lands a few multiples above that routinely, so a
    normal relaxation emits thousands of warnings reporting a ~1e-12 relative
    error -- twelve orders of magnitude below the force tolerance, and enough
    noise to bury the per-trial failures that do matter.

    This inspects the residual rather than filtering by message, because
    ``warnings`` cannot do the job here: every one of these messages is a
    *different* string (the residual is interpolated into it), so the
    ``default`` and ``once`` actions -- which key their registries by message
    text -- treat each as new and print it.  That is exactly why they spam.
    ``ignore`` would silence them, but a residual of, say, 1e-6 means a
    near-singular deformation gradient, i.e. a collapsing cell, and that one
    should still be seen.
    """
    original = warnings.showwarning
    if getattr(original, "_drops_logm_roundoff", False):
        return

    def showwarning(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if text.startswith(_LOGM_WARNING):
            _, _, residual = text.rpartition("=")
            try:
                if float(residual) < threshold:
                    return
            except ValueError:  # a message shape we do not recognise: keep it
                pass
        original(message, category, filename, lineno, file, line)

    showwarning._drops_logm_roundoff = True
    warnings.showwarning = showwarning


def _init_relax_worker(counter, slots: list[str], mlip: str, debug: bool) -> None:
    """Claim one device for this process and build its calculator once."""
    global _WORKER_DEVICE, _WORKER_TORCH_DEVICE, _WORKER_CALCULATOR
    for key, value in _SINGLE_THREAD_ENV_VARS.items():
        os.environ[key] = value
    _init_worker_logging(debug)
    _quiet_logm_roundoff()

    _WORKER_DEVICE = claim_device(counter, slots)
    _WORKER_TORCH_DEVICE = _pin_visible_device(_WORKER_DEVICE)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.set_num_threads(1)

    # Built once per process rather than per trial: loading an MLIP costs far
    # more than relaxing one structure.
    _WORKER_CALCULATOR = build_hull_calculator(mlip, device=_WORKER_TORCH_DEVICE)
    logger.info("Worker ready on %s (torch device %s)", _WORKER_DEVICE, _WORKER_TORCH_DEVICE)


def _trial_dir(output_dir: Path, index: int, trial: int) -> Path:
    return Path(output_dir) / CRYSPR_DIR / str(index) / f"trial-{trial}"


# --------------------------------------------------------------------------- #
# Stage 1: screen
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Stage 2: generate
# --------------------------------------------------------------------------- #
def _budget(gene: dict, schedule) -> tuple[Optional[int], int]:
    """The gene's positional DoF and the trials the schedule allots it.

    A gene whose DoF cannot be determined gets the largest allotment in the
    schedule: an unreadable gene is not evidence that it is easy to reconstruct,
    and this path is rare enough that being generous costs nothing.
    """
    try:
        dof = positional_dof(gene)
    except Exception as exc:
        largest = max(trials for _, trials in schedule)
        logger.warning(
            "No positional DoF for gene %s (%s); giving it %d trial(s)",
            gene.get("group"), exc, largest,
        )
        return None, largest
    return dof, trials_for_dof(dof, schedule)


def _generate_one(
    index: int,
    trial: int,
    gene: dict,
    trial_dir: str,
    timeout: Optional[float],
):
    """Draw one PyXtal structure in a pool worker.

    Returns:
        ``(row, atoms)``.  *atoms* is ``None`` unless the draw succeeded, and
        the row's ``status`` says which of ``ok``/``failed``/``timeout`` it was.
    """
    from wyckoff_transformer.cryspr.generator import single_pyxtal

    started = time.time()
    row = {"index": index, "trial": trial, "status": "failed", "error": None}
    atoms = None
    try:
        with time_limit(timeout):
            atoms = single_pyxtal(wyckoffgene=gene, nlimit=30, wdir=Path(trial_dir))
    except Timeout as exc:
        row["status"] = "timeout"
        row["error"] = str(exc)
        logger.warning("Gene %d trial %d: PyXtal timed out (%s)", index, trial, exc)
    except Exception as exc:  # noqa: BLE001 - single_pyxtal swallows most of these
        row["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Gene %d trial %d: PyXtal failed (%s)", index, trial, exc)
    else:
        if atoms is None:
            row["error"] = "PyXtal returned no structure"
        else:
            row["status"] = "ok"
            row["formula"] = atoms.get_chemical_formula(mode="metal")
            row["n_atoms"] = len(atoms)
    row["seconds"] = round(time.time() - started, 2)
    return row, atoms


def _todo_representatives(screen, limit: Optional[int]) -> list[int]:
    return sorted(i for i in screen.counts if limit is None or i < limit)


def stage_generate(args) -> None:
    """PyXtal draws for every unique gene, on CPU, with a per-draw timeout.

    Gene-known representatives are drawn too.  Their fingerprint matching a
    LeMat-Bulk entry only makes them *candidates* for the matcher: the same
    space group with the same elements on the same Wyckoff orbits is not the
    same structure, so a known gene can still relax into a novel one, and
    dropping it here would decide novelty by fingerprint alone.

    Writes the structures to ``pyxtal.extxyz`` (one frame per successful draw,
    tagged with its gene and trial) and one row per attempt to ``pyxtal.csv``.
    """
    from ase.io import write as ase_write

    genes = load_genes(args.input)
    screen = read_screen(args.output_dir / SCREEN_FILE)
    schedule = parse_trial_schedule(args.n_trials)
    todo = _todo_representatives(screen, args.limit)
    budget = {index: _budget(genes[index], schedule) for index in todo}

    log = RowLog(args.output_dir / PYXTAL_TRIALS_FILE, PYXTAL_COLUMNS, args.resume)
    structures_path = args.output_dir / PYXTAL_FILE
    if not args.resume and structures_path.exists():
        structures_path.unlink()

    tasks = [
        (index, trial)
        for index in todo
        for trial in range(budget[index][1])
        if (index, trial) not in log.done
    ]
    trials = [n for _, n in budget.values()]
    logger.info(
        "Trial schedule %r: %d trials over %d genes, %.2f per gene (%s)",
        args.n_trials, sum(trials), len(trials),
        sum(trials) / len(trials) if trials else 0,
        ", ".join(f"{n} trial(s): {trials.count(n)}" for n in sorted(set(trials))),
    )
    if log.done:
        logger.info("Resuming: %d draws already recorded", len(log.done))
    cores = args.pyxtal_cores or os.cpu_count() or 1
    logger.info("Drawing %d structures on %d core(s)", len(tasks), cores)

    ctx = multiprocessing.get_context("spawn")
    n_ok = 0
    try:
        with ProcessPoolExecutor(
            max_workers=cores,
            mp_context=ctx,
            initializer=_init_generate_worker,
            initargs=(args.debug,),
        ) as pool:
            futures = {
                pool.submit(
                    _generate_one,
                    index,
                    trial,
                    genes[index],
                    str(_trial_dir(args.output_dir, index, trial)),
                    args.pyxtal_timeout,
                ): (index, trial)
                for index, trial in tasks
            }
            for done, future in enumerate(as_completed(futures), start=1):
                index, trial = futures[future]
                try:
                    row, atoms = future.result()
                except Exception as exc:  # noqa: BLE001 - a worker died
                    logger.warning("Gene %d trial %d: worker failed (%s)", index, trial, exc)
                    row, atoms = (
                        {"index": index, "trial": trial, "status": "failed",
                         "error": f"{type(exc).__name__}: {exc}", "seconds": None},
                        None,
                    )
                dof, n_trials = budget[index]
                row["dof_positional"] = dof
                row["n_trials"] = n_trials
                if atoms is not None:
                    # Tagged in the frame itself, so the relax stage needs
                    # nothing but this file to know what it is relaxing.
                    atoms.info = {"gene": int(index), "trial": int(trial)}
                    ase_write(str(structures_path), atoms, format="extxyz", append=True)
                    n_ok += 1
                log.write(row)
                if done % 100 == 0 or done == len(futures):
                    logger.info("Drew %d/%d", done, len(futures))
    finally:
        log.close()

    frame = log.frame()
    _update_manifest(args.output_dir / MANIFEST_FILE, {
        "input": str(args.input),
        "trial_schedule": args.n_trials,
        "trials_total": int(len(frame)),
        "trials_per_gene": round(sum(trials) / len(trials), 3) if trials else None,
        "pyxtal_cores": cores,
        "pyxtal_timeout": args.pyxtal_timeout,
        "pyxtal_ok": int((frame["status"] == "ok").sum()),
        "pyxtal_failed": int((frame["status"] == "failed").sum()),
        "pyxtal_timed_out": int((frame["status"] == "timeout").sum()),
    })
    print(
        f"{n_ok} new draws, {int((frame['status'] == 'ok').sum())}/{len(frame)} "
        f"trials with a structure -> {structures_path}"
    )


# --------------------------------------------------------------------------- #
# Stage 3: relax
# --------------------------------------------------------------------------- #
def _relax_one(
    index: int,
    trial: int,
    atoms,
    trial_dir: str,
    fmax: float,
    release_symmetry: bool,
    rattle: bool,
    timeout: Optional[float],
) -> dict:
    """Relax one generated draw in a pool worker.  Returns a row for the CSV."""
    from wyckoff_transformer.cryspr.generator import KEPT_CIF_SUFFIX, _trial_seed, relax_trial

    started = time.time()
    row = {
        "index": index, "trial": trial, "status": "failed",
        "device": _WORKER_DEVICE, "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(mode="metal"),
        "energy": None, "energy_per_atom": None, "cif": None, "error": None,
    }
    try:
        with time_limit(timeout):
            relaxed, energy = relax_trial(
                atoms_in=atoms,
                calculator=_WORKER_CALCULATOR,
                trial_dir=Path(trial_dir),
                label=f"gene {index} trial {trial}",
                release_symmetry=release_symmetry,
                rattle=rattle,
                seed=_trial_seed(index, trial),
                fmax=fmax,
            )
    except Timeout as exc:
        row["status"] = "timeout"
        row["error"] = str(exc)
        logger.warning("Gene %d trial %d: relaxation timed out (%s)", index, trial, exc)
    except Exception as exc:  # noqa: BLE001 - one bad trial must not stop the stage
        row["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Gene %d trial %d: relaxation failed (%s)", index, trial, exc)
    else:
        if relaxed is None:
            row["status"] = "clash"
            row["error"] = "relaxed structure has atomic clashes"
        else:
            row["status"] = "ok"
            row["energy"] = energy
            row["energy_per_atom"] = energy / len(relaxed)
            row["n_atoms"] = len(relaxed)
            row["formula"] = relaxed.get_chemical_formula(mode="metal")
            row["cif"] = str(Path(trial_dir) / f"{row['formula']}{KEPT_CIF_SUFFIX}")
    row["seconds"] = round(time.time() - started, 2)
    return row


def _read_draws(path: Path, limit: Optional[int]) -> list[tuple[int, int, object]]:
    """The generated structures, as ``(gene, trial, atoms)``.

    Raises:
        FileNotFoundError: If the generate stage has not run.
    """
    from ase.io import read as ase_read

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No generated structures at {path}. Run --stage generate first."
        )
    draws = []
    for atoms in ase_read(str(path), index=":", format="extxyz"):
        index = int(atoms.info["gene"])
        if limit is not None and index >= limit:
            continue
        draws.append((index, int(atoms.info["trial"]), atoms))
    return draws


def stage_relax(args) -> None:
    """CrySPR on every generated draw, one pool task per trial.

    The task is a trial and not a gene so that the pool stays balanced: trial
    budgets differ by a factor of three, and a worker that finished early would
    otherwise idle while another works through a three-trial gene.

    Reads only ``pyxtal.extxyz``: which genes those draws belong to and how
    many trials each was allotted is recorded in the frames and in
    ``pyxtal.csv``, so this stage needs neither the gene file nor the screen.
    """
    spec = resolve_hull_mlip(args.mlip)
    slots = resolve_devices(args.cores, args.devices, args.workers_per_device)
    draws = _read_draws(args.output_dir / PYXTAL_FILE, args.limit)

    log = RowLog(args.output_dir / RELAXATIONS_FILE, RELAXATION_COLUMNS, args.resume)
    todo = [(i, t, atoms) for i, t, atoms in draws if (i, t) not in log.done]
    if log.done:
        logger.info("Resuming: %d trials already relaxed", len(log.done))
    logger.info(
        "Relaxing %d draws with %s on %d worker(s): %s",
        len(todo), args.mlip, len(slots), ", ".join(sorted(set(slots))),
    )

    # Set here, in the parent, and not only in the initialiser: a spawned worker
    # imports pandas (and with it numpy) before the initialiser runs, and MKL
    # fixes its thread count when it is first loaded.  Setting them afterwards
    # leaves every worker multi-threaded, which oversubscribes the machine
    # without making any single relaxation faster.
    for key, value in _SINGLE_THREAD_ENV_VARS.items():
        os.environ.setdefault(key, value)

    ctx = multiprocessing.get_context("spawn")
    # Inherited by every worker the pool spawns, replacements included.
    claimed = ctx.Value("i", 0)

    started = time.time()
    try:
        with ProcessPoolExecutor(
            max_workers=len(slots),
            mp_context=ctx,
            initializer=_init_relax_worker,
            initargs=(claimed, slots, args.mlip, args.debug),
        ) as pool:
            futures = {
                pool.submit(
                    _relax_one,
                    index,
                    trial,
                    atoms,
                    str(_trial_dir(args.output_dir, index, trial)),
                    args.fmax,
                    args.release_symmetry,
                    args.rattle,
                    args.relax_timeout,
                ): (index, trial)
                for index, trial, atoms in todo
            }
            for done, future in enumerate(as_completed(futures), start=1):
                index, trial = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001 - a worker died
                    logger.warning("Gene %d trial %d: worker failed (%s)", index, trial, exc)
                    row = {
                        "index": index, "trial": trial, "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                log.write(row)
                if done % 25 == 0 or done == len(futures):
                    rate = (time.time() - started) / done
                    logger.info(
                        "Relaxed %d/%d (%.1f s/trial, %.0f min left)",
                        done, len(futures), rate, rate * (len(futures) - done) / 60,
                    )
    finally:
        log.close()

    frame = aggregate_structures(args.output_dir)
    relaxed = log.frame()
    _update_manifest(args.output_dir / MANIFEST_FILE, {
        "mlip": args.mlip,
        "hull_type": spec.hull_type,
        "checkpoint": spec.checkpoint,
        "mlip_note": spec.note or None,
        "release_symmetry": args.release_symmetry,
        "rattle": args.rattle,
        "fmax": args.fmax,
        "relax_timeout": args.relax_timeout,
        "workers": len(slots),
        "devices": sorted(set(slots)),
        # The slots themselves, not just the distinct cards: a device named
        # twice has two workers on it, which is how cards of different size
        # get different shares.
        "device_slots": slots,
        "trials_relaxed": int((relaxed["status"] == "ok").sum()),
        "trials_failed": int((relaxed["status"] != "ok").sum()),
        # What the trials actually ran on, not just what was asked for. A run
        # once recorded three GPUs while one of them never took a single trial
        # and a worker fell back to the CPU; only the per-trial rows said so.
        "trials_by_device": {
            str(device): int(count)
            for device, count in relaxed["device"].value_counts().items()
        },
        "genes_with_structure": int(frame["has_structure"].sum()),
        "genes_attempted": int(len(frame)),
    })
    print(
        f"{int(frame['has_structure'].sum())}/{len(frame)} genes produced a "
        f"structure -> {args.output_dir / STRUCTURES_FILE}"
    )


def aggregate_structures(output_dir: Path) -> pd.DataFrame:
    """Reduce the per-trial logs to one row per gene, and collect its CIF.

    The kept structure of a gene is the lowest-energy trial that produced one.
    Genes that produced none keep a row, with the reason in ``error``: a gene
    that nothing could be built for and one whose relaxations all crashed are
    different failures, and the funnel's ``has_structure`` alone does not say
    which happened.

    Returns:
        The frame written to ``structures.csv``, indexed by gene.
    """
    output_dir = Path(output_dir)
    draws = read_rows(output_dir / PYXTAL_TRIALS_FILE, PYXTAL_COLUMNS)
    relaxations = read_rows(output_dir / RELAXATIONS_FILE, RELAXATION_COLUMNS)
    cif_dir = output_dir / CIF_DIR
    cif_dir.mkdir(parents=True, exist_ok=True)

    by_gene = {index: group for index, group in relaxations.groupby("index")}
    rows = []
    for index, group in draws.groupby("index"):
        relaxed = by_gene.get(index, relaxations.iloc[:0])
        succeeded = relaxed[relaxed["status"] == "ok"]
        row = {
            "index": int(index),
            "dof_positional": group["dof_positional"].iloc[0],
            "n_trials": int(group["n_trials"].iloc[0]),
            "n_drawn": int((group["status"] == "ok").sum()),
            "n_relaxed": int(len(succeeded)),
            "pyxtal_seconds": round(float(group["seconds"].fillna(0).sum()), 2),
            "relax_seconds": round(float(relaxed["seconds"].fillna(0).sum()), 2)
            if len(relaxed) else 0.0,
            "has_structure": False,
            "formula": None, "energy": None, "energy_per_atom": None,
            "n_atoms": None, "best_trial": None, "device": None, "error": None,
        }
        if len(succeeded):
            best = succeeded.loc[succeeded["energy"].idxmin()]
            row.update(
                has_structure=True,
                formula=best["formula"],
                energy=float(best["energy"]),
                energy_per_atom=float(best["energy_per_atom"]),
                n_atoms=int(best["n_atoms"]),
                best_trial=int(best["trial"]),
                device=best["device"],
            )
            source = Path(str(best["cif"]))
            if source.is_file():
                (cif_dir / f"{index}.cif").write_text(
                    source.read_text(encoding="utf-8"), encoding="utf-8"
                )
            else:
                row["error"] = f"kept CIF missing at {source}"
        else:
            row["error"] = _failure_reason(group, relaxed)
        rows.append(row)

    frame = pd.DataFrame(rows).set_index("index").sort_index()
    frame.to_csv(output_dir / STRUCTURES_FILE)
    return frame


def _failure_reason(draws: pd.DataFrame, relaxations: pd.DataFrame) -> str:
    """Why this gene has no structure, in the words of the stage that failed."""
    if not len(relaxations):
        statuses = sorted(set(draws["status"]) - {"ok"})
        if "ok" in set(draws["status"]):
            return "generated but never relaxed"
        return f"no PyXtal structure ({', '.join(statuses) or 'unknown'})"
    errors = [str(e) for e in relaxations["error"] if isinstance(e, str) and e]
    return f"all {len(relaxations)} relaxation(s) failed: {errors[0] if errors else 'unknown'}"


# --------------------------------------------------------------------------- #
# Stage 4: score
# --------------------------------------------------------------------------- #
def stage_score(args) -> None:
    """Structure validity, uniqueness, novelty and e_above_hull.

    Everything here runs in-process.  Uniqueness and novelty are both the
    two-stage filter from :mod:`wyckoff_transformer.evaluation.novelty`: the
    augmented Wyckoff fingerprint narrows the comparison down to the handful of
    structures that could possibly match, then ``StructureMatcher`` decides.
    The reference for novelty is built per run, since only LeMat-Bulk entries
    whose fingerprint collides with a generated one ever reach the matcher.

    Each relaxed structure is also re-fingerprinted (PyXtal symmetry detection),
    and novelty is decided against LeMat-Bulk entries sharing *either* the
    sampled gene's fingerprint or the relaxed structure's -- relaxation can move
    a structure off the orbit set PyXtal placed it on.  ``structures.csv`` then
    carries ``gene_novel`` (the sampled gene), ``novel_by_sampled_gene`` (the
    old sampled-fingerprint-only verdict), ``novel_structure`` (the two-
    fingerprint verdict, which MetaSUN uses), and ``relaxed_fingerprint_*``.
    """
    from pymatgen.core import Structure

    from wyckoff_transformer.evaluation.hull_energy import HullEnergyCalculator
    from wyckoff_transformer.evaluation.novelty import (
        NoveltyFilter,
        filter_by_unique_structure,
    )
    from wyckoff_transformer.evaluation.structure_novelty import build_novelty_reference
    from wyckoff_transformer.evaluation.structure_validity import is_valid

    phase = _PhaseTimer()
    screen = read_screen(args.output_dir / SCREEN_FILE)
    frame = pd.read_csv(args.output_dir / STRUCTURES_FILE, index_col="index")
    cif_dir = args.output_dir / CIF_DIR

    genes = load_genes(args.input)
    # Not persisted by write_screen -- large, and cheap to recompute.
    fingerprinter = GeneFingerprinter()
    phase.done("load genes and screen")
    hull = HullEnergyCalculator(args.mlip)
    phase.done("load the hull")

    validity, fingerprints, relaxed_fingerprints = {}, {}, {}
    structures, hull_energies = {}, {}
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
        # The relaxed structure's own fingerprint, which relaxation -- the
        # rattle stage especially -- can move away from the sampled gene's.
        # Novelty is judged against both, so a gene that PyXtal placed on a
        # LeMat-Bulk-known orbit set but relaxed off it is still checked.
        try:
            relaxed_fingerprints[index] = fingerprinter.fingerprint_structure(structure)
        except Exception as exc:
            logger.warning("Gene %d: no relaxed fingerprint (%s)", index, exc)

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

    phase.done("read CIFs, validity, fingerprints and e_above_hull")
    frame["valid_structure"] = pd.Series(validity)
    frame["e_above_hull"] = pd.Series(hull_energies)
    novel_genes = set(screen.novel)
    frame["gene_novel"] = pd.Series(
        {index: index in novel_genes for index in frame.index}
    )
    frame["relaxed_fingerprint_resolved"] = pd.Series(
        {index: index in relaxed_fingerprints for index in frame.index}
    )
    frame["relaxed_fingerprint_changed"] = pd.Series(
        {
            index: relaxed_fingerprints[index] != fingerprints.get(index)
            for index in relaxed_fingerprints
        }
    )

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
    scored["relaxed_fingerprint"] = pd.Series(relaxed_fingerprints).reindex(scored.index)

    unique_index = filter_by_unique_structure(scored).index
    frame["unique_structure"] = pd.Series(
        {index: index in set(unique_index) for index in scored.index}
    )
    phase.done("uniqueness (StructureMatcher)")

    # The matcher needs a candidate for either fingerprint, so the reference is
    # built over the union of the sampled and the relaxed ones.
    all_fingerprints = pd.concat(
        [scored["fingerprint"], scored["relaxed_fingerprint"].dropna()]
    )
    reference = build_novelty_reference(
        all_fingerprints,
        cache=args.reference_cache,
        splits=tuple(s.strip() for s in args.reference_splits.split(",")),
        lemat_cif_csv=args.lemat_cif_csv,
    )
    phase.done("build the novelty reference")
    novelty_filter = NoveltyFilter(reference)

    def _relaxed_missing(value) -> bool:
        return value is None or (isinstance(value, float) and pd.isna(value))

    def _is_novel(row: pd.Series) -> bool:
        """Novel iff no LeMat-Bulk entry sharing *either* fingerprint matches."""
        if not novelty_filter.is_novel(row):
            return False
        relaxed = row.get("relaxed_fingerprint")
        if _relaxed_missing(relaxed):
            return True
        return novelty_filter.is_novel(
            pd.Series({"fingerprint": relaxed, "structure": row.structure})
        )

    frame["novel_by_sampled_gene"] = pd.Series(
        {index: novelty_filter.is_novel(row) for index, row in scored.iterrows()}
    )
    frame["novel_structure"] = pd.Series(
        {index: _is_novel(row) for index, row in scored.iterrows()}
    )
    phase.done("novelty (StructureMatcher)")

    frame.drop(columns=["structure"], errors="ignore").to_csv(
        args.output_dir / STRUCTURES_FILE
    )
    _update_manifest(
        args.output_dir / MANIFEST_FILE,
        {"hull": hull.provenance, "novelty": "sampled+relaxed fingerprint"},
    )
    report = funnel(screen, frame)
    (args.output_dir / FUNNEL_FILE).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    phase.summary()
    print(json.dumps(report, indent=2))


class _PhaseTimer:
    """Time the score stage's phases, because they are unlike one another.

    Which phase dominates decides whether parallelising this stage is worth
    anything: the per-gene work is embarrassingly parallel, while the scan that
    builds the novelty reference is a single streaming gzip read that no worker
    count improves.
    """

    def __init__(self) -> None:
        self.started = self.last = time.time()
        self.phases: list[tuple[str, float]] = []

    def done(self, label: str) -> None:
        now = time.time()
        self.phases.append((label, now - self.last))
        self.last = now
        logger.info("%s: %.1f s", label, self.phases[-1][1])

    def summary(self) -> None:
        total = time.time() - self.started
        logger.info(
            "score stage %.1f s: %s", total,
            ", ".join(f"{label} {seconds:.0f} s ({seconds / total:.0%})"
                      for label, seconds in self.phases),
        )


def _update_manifest(manifest_path: Path, entries: dict) -> None:
    """Merge *entries* into the manifest, keeping what earlier stages wrote.

    Every stage knows something about the numbers that the others do not and
    that is not recoverable from them afterwards: the trial budget, the MLIP
    checkpoint, and which hull the energies were referenced to.  Merging rather
    than overwriting is also what lets a stage be re-run on its own.
    """
    manifest_path = Path(manifest_path)
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Unreadable manifest at %s (%s)", manifest_path, exc)
    manifest.update(entries)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wyformer-protocol",
        description=(
            "Rank a WyFormer variant by MetaSUN per generated gene. Filters that "
            "need no potential (validity, uniqueness, gene novelty) run first; "
            "every unique gene is then relaxed, gene-known ones included, since a "
            "known fingerprint can still relax into a novel structure and its "
            "e_above_hull is needed to keep the energy distribution unbiased."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="JSON(.gz) list of Wyckoff genes.")
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for all stage outputs; stages read each other's files here.",
    )
    parser.add_argument(
        "--stage", choices=STAGES + ("all",), default="all",
        help="Which stage to run. 'all' runs screen, generate, relax and score.",
    )
    parser.add_argument(
        "--mlip", type=str, default=DEFAULT_HULL_MLIP, choices=sorted(HULL_MLIPS),
        help=(
            "Potential to relax and evaluate with. Restricted to MLIPs that "
            "LeMat-Bulk publishes a convex hull for, because e_above_hull is only "
            "meaningful when energy and hull come from the same model."
        ),
    )

    generation = parser.add_argument_group("generation (stage: generate)")
    generation.add_argument(
        "--pyxtal-cores", type=int, default=None,
        help="CPU processes drawing PyXtal structures. Defaults to every core.",
    )
    generation.add_argument(
        "--pyxtal-timeout", type=float, default=300.0,
        help=(
            "Seconds one PyXtal draw may take before it is abandoned as failed. "
            "PyXtal rejection-samples, so a gene it cannot satisfy does not fail: "
            "it spins. 0 disables the limit."
        ),
    )
    generation.add_argument(
        "--n-trials", type=str, default=DEFAULT_TRIAL_SCHEDULE,
        help=(
            "PyXtal trials per gene, as 'dof:trials' pairs over the gene's "
            "positional degrees of freedom, ending in a '*' bin. A bare integer "
            "gives every gene the same number. The default spends one trial on "
            "the fifth of genes with no free coordinates, where a second one "
            "provably changes nothing, and two on the rest."
        ),
    )

    hardware = parser.add_argument_group("relaxation hardware (stage: relax)")
    hardware.add_argument(
        "--cores", type=int, default=None,
        help="Relax on CPU with this many worker processes (each single-threaded).",
    )
    hardware.add_argument(
        "--devices", type=str, default=None,
        help="Relax on GPU: comma-separated torch devices, e.g. 'cuda:0,cuda:1'.",
    )
    hardware.add_argument(
        "--workers-per-device", type=int, default=1,
        help="Worker processes per GPU. Ignored with --cores.",
    )

    relax = parser.add_argument_group("relaxation (stage: relax)")
    relax.add_argument(
        "--fmax", type=float, default=0.05, help="Force convergence in eV/A.",
    )
    relax.add_argument(
        "--relax-timeout", type=float, default=1800.0,
        help="Seconds one trial's four-stage relaxation may take. 0 disables the limit.",
    )
    relax.add_argument(
        "--release-symmetry", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Run a symmetry-free relaxation stage before the rattle. On by "
            "default: it is what the rattle is perturbed away from and what its "
            "energy is compared against, so the acceptance test asks whether "
            "the perturbation found a better basin rather than whether it "
            "finished a relaxation the constrained stages had not. It is nearly "
            "free -- zero optimiser steps in 78%% of trials, because gradient "
            "descent cannot leave a symmetric stationary point."
        ),
    )
    relax.add_argument(
        "--rattle", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Run the rattle stage: perturb positions and cell, relax again "
            "unconstrained, and keep the result only if it wins 1 meV/atom. "
            "On by default -- it is the only stage that can break the symmetry "
            "PyXtal imposed, and it recovered 99 further ground states in the "
            "reconstruction study."
        ),
    )

    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process genes with an index below this. For smoke tests.",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Keep the trials the generate and relax logs already have a row for "
            "and do only the rest. On by default, since both logs are written "
            "row by row and an interrupted stage has nothing to gain from "
            "repeating itself; --no-resume starts both from scratch."
        ),
    )

    reference = parser.add_argument_group("references (stages: screen, score)")
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


def run_stage(stage: str, args) -> None:
    """Run one stage by name.

    The function is looked up when the stage runs rather than bound in a table
    at import time, so that a caller -- ``wyformer-protocol-wandb``, or a test
    -- can substitute one.
    """
    logger.info("=== stage: %s ===", stage)
    globals()[f"stage_{stage}"](args)


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for stage in (STAGES if args.stage == "all" else (args.stage,)):
        run_stage(stage, args)


if __name__ == "__main__":
    main()
