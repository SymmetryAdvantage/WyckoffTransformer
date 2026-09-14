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

A fifth stage, ``template``, is optional and not part of ``--stage all``.  It
adds one start per gene taken from a training structure on the same Wyckoff
orbits rather than drawn at random, to be relaxed and scored as an extra trial
alongside them; see :mod:`wyckoff_transformer.cryspr.template` and
``docs/cryspr_template_starts.md``.

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

from wyckoff_transformer.cryspr.basin_hopping import (
    BASINHOP_STDEV,
    BASINHOP_STRAIN_STDEV,
    DEFAULT_STEPS as BASINHOP_STEPS,
    DEFAULT_TEMPERATURE_EV_PER_ATOM as BASINHOP_TEMPERATURE,
)
from wyckoff_transformer.cryspr.generator import DEFAULT_PYXTAL_TOL_FACTOR, PRERELAX_FMAX
from wyckoff_transformer.cryspr.mlips import DEFAULT_PRERELAX_MLIP, prerelax_mlip_names
from wyckoff_transformer.cryspr.prescreen import DEDUP_ENERGY_TOL_EV_PER_ATOM
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
    rattle_effect,
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

#: Where the pre-rattle structure of each gene's best trial is collected.
PRERATTLE_CIF_DIR = "cifs_prerattle"
CRYSPR_DIR = "cryspr"

#: Every pre-relaxed draw, and the per-trial log of the pre-relaxation.
PRESCREEN_ALL_FILE = "prescreen_all.extxyz"
PRESCREEN_TRIALS_FILE = "prescreen.csv"

#: The draws the pre-screen *selected*, and why each draw was kept or dropped.
#: Written by a deterministic pass over the two files above, so this pair is
#: derived state and is rewritten in full whenever the stage runs.
PRESCREEN_FILE = "prescreen.extxyz"
PRESCREEN_SELECTION_FILE = "prescreen_selection.csv"

STAGES = ("screen", "generate", "relax", "score")

#: Stages that are not part of ``--stage all``.  Neither is a step of the
#: cascade; both are alternative *sources of starting structures* for ``relax``.
#:
#: ``template`` appends one template-matched draw per gene to whatever
#: ``generate`` produced, and the relax and score stages treat it as one more
#: trial.  ``prescreen`` narrows: it relaxes every draw on a cheap potential
#: under fixed symmetry, drops the ones that landed on the same structure, and
#: writes out the schedule's usual number of survivors for ``relax --relax-from
#: prescreen`` to start from.  ``basinhop`` searches instead of narrowing a
#: fixed set: from each draw it walks between symmetry-preserving minima on the
#: cheap potential and offers everything it found to the same selection.
#: Leaving all three out of ``all`` is what keeps ``--stage all`` the published
#: protocol.
OPTIONAL_STAGES = ("template", "prescreen", "basinhop")

#: Per-walk log and every minimum a walk visited.
BASINHOP_TRIALS_FILE = "basinhop.csv"
BASINHOP_ALL_FILE = "basinhop_all.extxyz"

#: The minima the walks selected, and why each was kept or dropped.
BASINHOP_FILE = "basinhop.extxyz"
BASINHOP_SELECTION_FILE = "basinhop_selection.csv"

#: Synthetic trial numbers for the minima a gene's walks find.
#:
#: A walk visits many minima, so it cannot be filed under the trial it started
#: from without making ``(index, trial)`` -- the key both stage logs and
#: ``--resume`` use -- ambiguous.  The key is therefore derived arithmetically
#: from the walk's own trial and the hop that found the minimum, which keeps it
#: unique *and* stable across a resume: a rerun that skips a finished walk must
#: not renumber the minima of the walks it does run.
#:
#: The base sits far above :data:`TEMPLATE_TRIAL` and above any multiplied trial
#: budget, and the stride above any plausible hop count.
BASINHOP_TRIAL_BASE = 100_000
BASINHOP_TRIAL_STRIDE = 1_000

#: Where ``relax`` may take its starting structures from.
RELAX_SOURCES = {
    "pyxtal": PYXTAL_FILE,
    "prescreen": PRESCREEN_FILE,
    "basinhop": BASINHOP_FILE,
}

#: Columns of ``basinhop.csv``: one row per walk, not per minimum.
#:
#: ``n_symmetry_lost`` should be zero by construction -- the perturbation is
#: projected onto the space group's subspace -- so it is written down rather
#: than assumed: a non-zero count means the premise of the arm is not holding.
BASINHOP_COLUMNS = (
    "index", "trial", "status", "formula", "n_atoms", "spacegroup",
    "n_hops", "n_accepted", "n_minima", "n_symmetry_lost",
    "n_symmetry_gained", "n_failed",
    "energy_start", "energy_best", "energy_per_atom_best", "backend",
    "device", "seconds", "error",
)

#: Trial index the template start is filed under.  Far above any schedule's
#: budget so it never collides with a random trial, and constant so ``--resume``
#: recognises a template draw that has already been made or relaxed.
TEMPLATE_TRIAL = 1000

#: Columns of ``pyxtal.csv``: one row per attempted PyXtal draw.
PYXTAL_COLUMNS = (
    "index", "trial", "status", "formula", "n_atoms",
    "dof_positional", "n_trials", "seconds", "error",
)

#: Columns of ``relaxations.csv``: one row per relaxed draw.
RELAXATION_COLUMNS = (
    "index", "trial", "status", "formula", "energy", "energy_per_atom",
    "n_atoms", "device", "seconds", "cif",
    # The rattle stage's two sides.  ``energy`` and ``cif`` are the kept
    # structure, which is what the protocol scores; these are the structure the
    # rattle was handed, which is what the pre-rattle metrics score.  Equal to
    # the kept pair when the rattle did not run or did not win.
    "energy_prerattle", "energy_per_atom_prerattle", "cif_prerattle",
    "error",
)

#: Columns of ``prescreen.csv``: one row per cheaply relaxed draw.
#:
#: ``backend`` says whether the pre-relaxation potential itself answered
#: (``nep89``) or its fallback did (``fallback``), which is the only per-trial
#: record that a gene's chemistry is outside NEP89's 89 elements.  ``spacegroup`` is spglib's verdict
#: on the pre-relaxed cell, so that "fixed symmetry" is a measured claim, and
#: ``volume_ratio`` is what the cheap potential did to the cell -- a
#: pre-relaxation is meant to contract a loose PyXtal draw, and one that
#: expands it is selecting on a geometry the scoring potential will disown.
PRESCREEN_COLUMNS = (
    "index", "trial", "status", "formula", "energy", "energy_per_atom",
    "n_atoms", "spacegroup", "volume_ratio", "backend", "device", "seconds",
    "error",
)

#: Columns of ``prescreen_selection.csv``: one row per candidate, kept or not.
PRESCREEN_SELECTION_COLUMNS = (
    "index", "trial", "energy_per_atom", "verdict", "duplicate_of",
    "n_candidates", "n_distinct", "budget", "dof_positional",
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

#: The cheap potential of a two-stage trial, or ``None`` for the single-stage
#: protocol.  Built once per process, like the scoring one.
_WORKER_PRERELAX_CALCULATOR = None


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

    #: Errors that say the *machine* failed, not the trial.
    #:
    #: A trial whose relaxation diverged has been answered and re-running it
    #: would answer the same way, so ``--resume`` is right to skip it.  A trial
    #: whose worker was killed under it has not been answered at all, and
    #: skipping it silently turns an infrastructure failure into a permanent
    #: hole in the cohort -- 1034 of 1800 trials in one run here, which read as
    #: a collapsed reconstruction rate rather than as a crash.
    RETRYABLE_ERRORS = ("BrokenProcessPool", "A process in the process pool")

    def __init__(
        self, path: Path, columns, resume: bool, retry_failed: bool = False
    ) -> None:
        self.path = Path(path)
        self.columns = list(columns)
        if self.columns[:2] != ["index", "trial"]:
            raise ValueError("a stage log is keyed by (index, trial), in that order")
        self.done: set[tuple[int, int]] = set()
        append = resume and self.path.is_file()
        if append:
            previous = read_rows(self.path, self.columns)
            keep = self._answered(previous, retry_failed)
            self.done = {
                (int(index), int(trial))
                for index, trial in zip(previous["index"][keep], previous["trial"][keep])
            }
            if self._migrate_header(previous):
                append = False
            else:
                self._end_partial_line()
        migrated = resume and self.path.is_file() and not append
        self._handle = self.path.open(
            "a" if (append or migrated) else "w", newline="", encoding="utf-8"
        )
        self._writer = csv.DictWriter(
            self._handle, fieldnames=self.columns, extrasaction="ignore"
        )
        if not append and not migrated:
            self._writer.writeheader()
            self._handle.flush()

    def _answered(self, previous: pd.DataFrame, retry_failed: bool):
        """Which existing rows count as done.

        Everything, normally.  With *retry_failed*, only the successful ones, so
        a stage re-runs what it failed at.  Either way a row whose error names a
        killed worker is *not* done: see :data:`RETRYABLE_ERRORS`.
        """
        keep = pd.Series(True, index=previous.index)
        if not len(previous):
            return keep
        if "status" in previous.columns and retry_failed:
            keep &= previous["status"].astype(str) == "ok"
        if "error" in previous.columns:
            text = previous["error"].astype(str)
            retryable = pd.Series(False, index=previous.index)
            for marker in self.RETRYABLE_ERRORS:
                retryable |= text.str.contains(marker, regex=False, na=False)
            if retryable.any():
                logger.info(
                    "%s: %d row(s) record a killed worker rather than a failed "
                    "trial; they will be re-run", self.path, int(retryable.sum()),
                )
            keep &= ~retryable
        return keep

    def _migrate_header(self, previous: pd.DataFrame) -> bool:
        """Rewrite the log when its header no longer matches these columns.

        Appending a row with today's fieldnames to a file written with
        yesterday's header silently misaligns every subsequent row: the writer
        emits values in one order and ``read_csv`` labels them in another, so a
        resumed run's energies would land in whatever column happens to share
        their position.  Nothing raises, and the numbers are simply wrong.

        Rewriting once is cheap -- these logs are thousands of rows -- and it
        keeps the finished work, which is the whole point of resuming.  A column
        the old file lacks is written empty, which is the truth about it.

        Returns:
            ``True`` if the file was rewritten, so the caller opens it for
            writing rather than appending.
        """
        existing = list(previous.columns)
        if existing == self.columns:
            return False
        logger.info(
            "Migrating %s: %d columns -> %d (added %s, dropped %s), keeping %d rows",
            self.path, len(existing), len(self.columns),
            [c for c in self.columns if c not in existing] or "none",
            [c for c in existing if c not in self.columns] or "none",
            len(previous),
        )
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=self.columns, extrasaction="ignore"
            )
            writer.writeheader()
            for row in previous.to_dict("records"):
                writer.writerow(
                    {k: ("" if pd.isna(v) else v) for k, v in row.items()}
                )
        return True

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


def _init_relax_worker(
    counter,
    slots: list[str],
    mlip: Optional[str],
    prerelax_mlip: Optional[str],
    debug: bool,
) -> None:
    """Claim one device for this process and build its calculator(s) once.

    Args:
        counter: Shared slot counter; see :func:`claim_device`.
        slots: One device string per worker slot.
        mlip: Scoring potential, built with
            :func:`~wyckoff_transformer.evaluation.hull_mlips.build_hull_calculator`
            so that its energies stay paired with its hull.  ``None`` builds
            none, which is what the pre-screen stage wants: it never scores.
        prerelax_mlip: Cheap potential for the first stage of a two-stage
            trial, built through
            :func:`~wyckoff_transformer.cryspr.mlips.build_prerelax_calculator`,
            which is *not* restricted to published hulls.  ``None`` leaves the
            trial single-stage.
        debug: DEBUG-level logging in this worker.
    """
    global _WORKER_DEVICE, _WORKER_TORCH_DEVICE, _WORKER_CALCULATOR
    global _WORKER_PRERELAX_CALCULATOR
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
    if mlip is not None:
        _WORKER_CALCULATOR = build_hull_calculator(mlip, device=_WORKER_TORCH_DEVICE)
    if prerelax_mlip is not None:
        from wyckoff_transformer.cryspr.mlips import build_prerelax_calculator

        _WORKER_PRERELAX_CALCULATOR = build_prerelax_calculator(
            prerelax_mlip, device=_WORKER_TORCH_DEVICE
        )
    logger.info(
        "Worker ready on %s (torch device %s): %s%s",
        _WORKER_DEVICE, _WORKER_TORCH_DEVICE, mlip or "no scoring potential",
        f", pre-relaxing with {prerelax_mlip}" if prerelax_mlip else "",
    )


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
def _budget(gene: dict, schedule, multiplier: int = 1) -> tuple[Optional[int], int]:
    """The gene's positional DoF and the trials the schedule allots it.

    A gene whose DoF cannot be determined gets the largest allotment in the
    schedule: an unreadable gene is not evidence that it is easy to reconstruct,
    and this path is rare enough that being generous costs nothing.

    Args:
        gene: A PyXtal-notation gene.
        schedule: Parsed trial schedule.
        multiplier: Scales the allotment.  The wide-then-narrow variant draws
            ``--trial-multiplier`` times as many starts as it will relax, and
            the pre-screen then selects the unmultiplied number back down; it
            multiplies rather than replacing the schedule so that the extra
            starts stay proportional to the free coordinates that make a start
            worth repeating.
    """
    if multiplier < 1:
        raise ValueError(f"--trial-multiplier must be >= 1, got {multiplier}")
    try:
        dof = positional_dof(gene)
    except Exception as exc:
        largest = max(trials for _, trials in schedule)
        logger.warning(
            "No positional DoF for gene %s (%s); giving it %d trial(s)",
            gene.get("group"), exc, largest * multiplier,
        )
        return None, largest * multiplier
    return dof, trials_for_dof(dof, schedule) * multiplier


def _generate_one(
    index: int,
    trial: int,
    gene: dict,
    trial_dir: str,
    timeout: Optional[float],
    tol_factor: float = DEFAULT_PYXTAL_TOL_FACTOR,
):
    """Draw one PyXtal structure in a pool worker.

    Returns:
        ``(row, atoms)``.  *atoms* is ``None`` unless the draw succeeded, and
        the row's ``status`` says which of ``ok``/``failed``/``timeout`` it was.
    """
    from wyckoff_transformer.cryspr.generator import pyxtal_tol_matrix, single_pyxtal

    started = time.time()
    row = {"index": index, "trial": trial, "status": "failed", "error": None}
    atoms = None
    try:
        with time_limit(timeout):
            atoms = single_pyxtal(
                wyckoffgene=gene,
                iadm=pyxtal_tol_matrix(tol_factor),
                nlimit=30,
                wdir=Path(trial_dir),
            )
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
    multiplier = getattr(args, "trial_multiplier", 1)
    budget = {index: _budget(genes[index], schedule, multiplier) for index in todo}

    log = RowLog(args.output_dir / PYXTAL_TRIALS_FILE, PYXTAL_COLUMNS, args.resume, getattr(args, "retry_failed", False))
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
        "Trial schedule %r x%d: %d trials over %d genes, %.2f per gene (%s)",
        args.n_trials, multiplier, sum(trials), len(trials),
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
                    args.pyxtal_tol_factor,
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
        "trial_multiplier": multiplier,
        "trials_total": int(len(frame)),
        "trials_per_gene": round(sum(trials) / len(trials), 3) if trials else None,
        "pyxtal_cores": cores,
        "pyxtal_timeout": args.pyxtal_timeout,
        # Recorded because it changes what the draws *are*, not just how long
        # they took: a run's structures cannot be compared with another's
        # without it, and nothing else in the output says which floor was used.
        "pyxtal_tol_factor": args.pyxtal_tol_factor,
        "pyxtal_ok": int((frame["status"] == "ok").sum()),
        "pyxtal_failed": int((frame["status"] == "failed").sum()),
        "pyxtal_timed_out": int((frame["status"] == "timeout").sum()),
    })
    print(
        f"{n_ok} new draws, {int((frame['status'] == 'ok').sum())}/{len(frame)} "
        f"trials with a structure -> {structures_path}"
    )


# --------------------------------------------------------------------------- #
# Optional stage: template
# --------------------------------------------------------------------------- #
def stage_template(args) -> None:
    """One template-matched start per gene, appended to the generate stage's output.

    Where ``generate`` asks PyXtal to *guess* the cell and every free Wyckoff
    coordinate, this takes them from a LeMat-Bulk structure that already
    occupies the gene's orbits -- the closest one by chemical formula -- and
    writes the gene's elements onto it
    (:mod:`wyckoff_transformer.cryspr.template`).  A gene with no such training
    structure, or whose candidates cannot be rebuilt, gets nothing here and
    keeps the random draws ``generate`` made for it.

    The start is filed as trial :data:`TEMPLATE_TRIAL` in the same
    ``pyxtal.extxyz`` and ``pyxtal.csv`` the random draws live in, so ``relax``
    picks it up as one more trial and ``--resume`` relaxes only the new ones.
    Running this stage on a finished run therefore costs one relaxation per
    gene and re-runs none of them.
    """
    from ase.io import write as ase_write

    from wyckoff_transformer.cryspr.template import (
        TemplateIndex,
        gene_query,
        load_template_structures,
        single_template,
    )

    genes = load_genes(args.input)
    screen = read_screen(args.output_dir / SCREEN_FILE)
    schedule = parse_trial_schedule(args.n_trials)
    todo = _todo_representatives(screen, args.limit)

    log = RowLog(args.output_dir / PYXTAL_TRIALS_FILE, PYXTAL_COLUMNS, resume=True)
    todo = [index for index in todo if (index, TEMPLATE_TRIAL) not in log.done]
    if not todo:
        log.close()
        print("Every gene already has a template draw.")
        return

    index_table = TemplateIndex.load(args.template_index)
    logger.info(
        "Template index: %d LeMat-Bulk entries over %d anonymous fingerprints",
        len(index_table), index_table.n_fingerprints,
    )
    fingerprinter = GeneFingerprinter()

    matches = {}
    for gene_index in todo:
        try:
            query = gene_query(genes[gene_index], fingerprinter)
        except Exception as exc:  # noqa: BLE001 - the screen already ruled on validity
            logger.warning("Gene %d: no template query (%s)", gene_index, exc)
            continue
        matches[gene_index] = index_table.select(query, k=args.template_candidates)
    logger.info(
        "%d/%d genes have a candidate template", sum(1 for m in matches.values() if m),
        len(todo),
    )

    # One pass over the ~1 GB CIF export for every candidate of every gene: the
    # index carries no geometry, and reading it per gene would be the whole cost
    # of the stage.
    ids = sorted({match.immutable_id for found in matches.values() for match in found})
    structures = load_template_structures(ids, lemat_cif_csv=args.lemat_cif_csv)
    logger.info("Read %d of %d candidate templates", len(structures), len(ids))

    structures_path = args.output_dir / PYXTAL_FILE
    n_ok = 0
    try:
        for gene_index in todo:
            started = time.time()
            dof, n_trials = _budget(
                genes[gene_index], schedule, getattr(args, "trial_multiplier", 1)
            )
            row = {
                "index": gene_index, "trial": TEMPLATE_TRIAL, "status": "failed",
                "dof_positional": dof, "n_trials": n_trials, "error": None,
            }
            atoms, match, error = single_template(
                genes[gene_index], matches.get(gene_index, ()), structures
            )
            if atoms is None:
                row["error"] = error
            else:
                trial_dir = _trial_dir(args.output_dir, gene_index, TEMPLATE_TRIAL)
                trial_dir.mkdir(parents=True, exist_ok=True)
                atoms.info = {
                    "gene": int(gene_index),
                    "trial": int(TEMPLATE_TRIAL),
                    "template": match.immutable_id,
                    "template_distance": round(match.distance, 6),
                }
                ase_write(str(structures_path), atoms, format="extxyz", append=True)
                row["status"] = "ok"
                row["formula"] = atoms.get_chemical_formula(mode="metal")
                row["n_atoms"] = len(atoms)
                n_ok += 1
            row["seconds"] = round(time.time() - started, 2)
            log.write(row)
    finally:
        log.close()

    _update_manifest(args.output_dir / MANIFEST_FILE, {
        "template_index": str(args.template_index or "default"),
        "template_candidates": args.template_candidates,
        "template_genes": len(todo),
        "template_drawn": n_ok,
    })
    print(f"{n_ok}/{len(todo)} genes got a template start -> {structures_path}")


# --------------------------------------------------------------------------- #
# Optional stage: prescreen
# --------------------------------------------------------------------------- #
def _prescreen_one(
    index: int,
    trial: int,
    atoms,
    trial_dir: str,
    fmax: float,
    timeout: Optional[float],
    release_symmetry: bool = False,
    rattle: bool = False,
) -> tuple[dict, Optional[object]]:
    """Relax one draw on the cheap potential.

    Returns:
        ``(row, atoms)``.  *atoms* is the pre-relaxed structure, or ``None``
        unless the row's ``status`` is ``ok``; the caller appends it to
        ``prescreen_all.extxyz``, where the selection reads it back from.

    Fixed symmetry is the default, and it is the point of the stage rather than
    an economy: what the wide-then-narrow variant selects between is *which
    PyXtal draw of this gene* to spend the scoring potential on, and a draw that
    broke its own space group on the cheap potential is no longer a draw of that
    gene.  The stages that may leave the symmetric stationary point then run on
    the potential whose energies are reported.

    ``release_symmetry`` and ``rattle`` turn that off, which is what the
    NEP89-first arm wants: there the cheap potential runs the *whole* protocol
    schedule and the scoring potential only refines the single winner, so the
    symmetry question is answered on NEP89 by design and the arm is a test of
    whether that is acceptable.
    """
    from wyckoff_transformer.cryspr.generator import _trial_seed, prerelax
    from wyckoff_transformer.cryspr.relaxer import _get_spacegroup_info

    started = time.time()
    row = {
        "index": index, "trial": trial, "status": "failed",
        "device": _WORKER_DEVICE, "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(mode="metal"),
        "energy": None, "energy_per_atom": None, "spacegroup": None,
        "volume_ratio": None, "backend": None, "error": None,
    }
    volume_before = float(atoms.get_volume()) if atoms.cell.rank == 3 else None
    relaxed = None
    try:
        with time_limit(timeout):
            relaxed = prerelax(
                atoms_in=atoms,
                calculator=_WORKER_PRERELAX_CALCULATOR,
                wdir=Path(trial_dir),
                label=f"gene {index} trial {trial} prescreen",
                fix_symmetry=True,
                release_symmetry=release_symmetry,
                rattle=rattle,
                seed=_trial_seed(index, trial),
                fmax=fmax,
                # A failed pre-relaxation is fatal *here*, unlike in a
                # two-stage trial: its energy is the only signal the selection
                # has, so an unrelaxed draw silently recorded as "ok" would be
                # ranked against relaxed ones on a meaningless number.
                strict=True,
            )
            relaxed.calc = _WORKER_PRERELAX_CALCULATOR
            energy = float(relaxed.get_potential_energy())
    except Timeout as exc:
        row["status"] = "timeout"
        row["error"] = str(exc)
        logger.warning("Gene %d trial %d: pre-relaxation timed out (%s)", index, trial, exc)
        relaxed = None
    except Exception as exc:  # noqa: BLE001 - one bad trial must not stop the stage
        row["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Gene %d trial %d: pre-relaxation failed (%s)", index, trial, exc)
        relaxed = None
    else:
        row["status"] = "ok"
        row["energy"] = energy
        row["energy_per_atom"] = energy / len(relaxed)
        row["n_atoms"] = len(relaxed)
        row["formula"] = relaxed.get_chemical_formula(mode="metal")
        row["spacegroup"] = _get_spacegroup_info(relaxed, symprec=1e-3)[1]
        if volume_before:
            row["volume_ratio"] = round(relaxed.get_volume() / volume_before, 4)
        row["backend"] = getattr(_WORKER_PRERELAX_CALCULATOR, "last_backend", None)
        relaxed.calc = None
    row["seconds"] = round(time.time() - started, 2)
    return row, relaxed


def stage_prescreen(args) -> None:
    """Relax every draw on the cheap potential, deduplicate, and select.

    The wide-then-narrow variant: ``generate --trial-multiplier 10`` draws ten
    times the schedule's starts, this stage relaxes all of them under fixed
    symmetry on ``--prescreen-mlip``, drops the ones that landed on the same
    structure, and writes the schedule's *usual* number of survivors -- the
    lowest-energy ones -- to ``prescreen.extxyz`` for
    ``relax --relax-from prescreen``.

    Two files are per-trial and resumable (``prescreen.csv`` and
    ``prescreen_all.extxyz``, one row and one frame per attempted
    pre-relaxation); two are derived and rewritten in full every time the stage
    runs (``prescreen_selection.csv`` and ``prescreen.extxyz``), because the
    selection is a global function of the first two and re-deciding it is cheap.

    Changes no default: this stage is not part of ``--stage all``, and ``relax``
    reads ``pyxtal.extxyz`` unless it is told otherwise.
    """
    from ase.io import write as ase_write

    genes = load_genes(args.input)
    schedule = parse_trial_schedule(args.n_trials)
    slots = resolve_devices(args.cores, args.devices, args.workers_per_device)
    prescreen_mlip = (
        getattr(args, "prescreen_mlip", None)
        or getattr(args, "prerelax_mlip", None)
        or DEFAULT_PRERELAX_MLIP
    )
    draws = _read_draws(args.output_dir / PYXTAL_FILE, args.limit)

    log = RowLog(args.output_dir / PRESCREEN_TRIALS_FILE, PRESCREEN_COLUMNS, args.resume, getattr(args, "retry_failed", False))
    relaxed_path = args.output_dir / PRESCREEN_ALL_FILE
    if not args.resume and relaxed_path.exists():
        relaxed_path.unlink()
    todo = [(i, t, atoms) for i, t, atoms in draws if (i, t) not in log.done]
    if log.done:
        logger.info("Resuming: %d draws already pre-relaxed", len(log.done))
    logger.info(
        "Pre-relaxing %d draws with %s on %d worker(s): %s",
        len(todo), prescreen_mlip, len(slots), ", ".join(sorted(set(slots))),
    )

    for key, value in _SINGLE_THREAD_ENV_VARS.items():
        os.environ.setdefault(key, value)
    ctx = multiprocessing.get_context("spawn")
    claimed = ctx.Value("i", 0)
    started = time.time()
    try:
        with ProcessPoolExecutor(
            max_workers=len(slots),
            mp_context=ctx,
            initializer=_init_relax_worker,
            # No scoring potential: this stage never computes a reported energy,
            # and loading ORB in every worker would cost more than the stage.
            initargs=(claimed, slots, None, prescreen_mlip, args.debug),
        ) as pool:
            futures = {
                pool.submit(
                    _prescreen_one,
                    index,
                    trial,
                    atoms,
                    str(_trial_dir(args.output_dir, index, trial) / "prescreen"),
                    args.prescreen_fmax,
                    args.relax_timeout,
                    args.prescreen_release_symmetry,
                    args.prescreen_rattle,
                ): (index, trial)
                for index, trial, atoms in todo
            }
            for done, future in enumerate(as_completed(futures), start=1):
                index, trial = futures[future]
                try:
                    row, relaxed = future.result()
                except Exception as exc:  # noqa: BLE001 - a worker died
                    logger.warning("Gene %d trial %d: worker failed (%s)", index, trial, exc)
                    row, relaxed = (
                        {"index": index, "trial": trial, "status": "failed",
                         "error": f"{type(exc).__name__}: {exc}"},
                        None,
                    )
                if relaxed is not None:
                    relaxed.info = {"gene": int(index), "trial": int(trial)}
                    ase_write(str(relaxed_path), relaxed, format="extxyz", append=True)
                log.write(row)
                if done % 50 == 0 or done == len(futures):
                    rate = (time.time() - started) / done
                    logger.info(
                        "Pre-relaxed %d/%d (%.1f s/trial, %.0f min left)",
                        done, len(futures), rate, rate * (len(futures) - done) / 60,
                    )
    finally:
        log.close()

    selection = _prescreen_select(args, genes, schedule)
    _update_manifest(args.output_dir / MANIFEST_FILE, {
        "prescreen_mlip": prescreen_mlip,
        "prescreen_fmax": args.prescreen_fmax,
        "prescreen_dedup": args.prescreen_dedup,
        "prescreen_energy_tol": args.prescreen_energy_tol,
        "prescreen_release_symmetry": args.prescreen_release_symmetry,
        "prescreen_rattle": args.prescreen_rattle,
        "prescreen_select": args.prescreen_select,
        **selection,
    })
    print(
        f"{selection['prescreen_selected']} of {selection['prescreen_candidates']} "
        f"pre-relaxed draws selected over {selection['prescreen_genes']} genes "
        f"({selection['prescreen_duplicates']} duplicates, "
        f"{selection['prescreen_rejected']} distinct but over budget) "
        f"-> {args.output_dir / PRESCREEN_FILE}"
    )


def _select_and_write(
    args,
    genes: list[dict],
    schedule,
    *,
    candidates_by_gene: dict[int, list],
    frames: dict,
    structures_path: Path,
    selection_path: Path,
    tag: str,
) -> dict:
    """Deduplicate each gene's candidates, keep its budget, and write both files.

    Shared by the two narrowing stages, which differ only in where their
    candidates came from -- independent PyXtal draws for ``prescreen``,
    a symmetry-constrained walk for ``basinhop`` -- and not at all in what
    "deduplicate and keep the best" means.  Both outputs are rewritten in full:
    they are a deterministic function of the per-candidate inputs, so there is
    nothing to resume and re-deciding costs a few matcher calls per gene.

    Returns:
        Candidate, duplicate, rejected and selected counts over every gene.
    """
    from ase.io import write as ase_write

    from wyckoff_transformer.cryspr.prescreen import select_candidates

    matcher = False if args.prescreen_dedup == "energy" else None
    if structures_path.exists():
        structures_path.unlink()
    selection_log = RowLog(selection_path, PRESCREEN_SELECTION_COLUMNS, resume=False)
    totals = {"candidates": 0, "duplicates": 0, "rejected": 0, "selected": 0, "genes": 0}
    try:
        for index in sorted(candidates_by_gene):
            candidates = candidates_by_gene[index]
            if not candidates:
                continue
            dof, budget = _prescreen_budget(genes[index], schedule, args.prescreen_select)
            chosen = select_candidates(
                candidates, budget=budget, matcher=matcher,
                energy_tol=args.prescreen_energy_tol,
            )
            totals["genes"] += 1
            totals["candidates"] += len(candidates)
            totals["duplicates"] += len(chosen.duplicate_of)
            totals["rejected"] += len(chosen.rejected)
            totals["selected"] += len(chosen.selected)

            by_trial = {c.trial: c for c in candidates}
            for trial in chosen.selected:
                atoms = frames[(index, trial)]
                atoms.info = {"gene": index, "trial": trial, tag: True}
                ase_write(str(structures_path), atoms, format="extxyz", append=True)
            for trial, verdict in (
                [(t, "selected") for t in chosen.selected]
                + [(t, "rejected") for t in chosen.rejected]
                + [(t, "duplicate") for t in chosen.duplicate_of]
            ):
                selection_log.write({
                    "index": index,
                    "trial": trial,
                    "energy_per_atom": by_trial[trial].energy_per_atom,
                    "verdict": verdict,
                    "duplicate_of": chosen.duplicate_of.get(trial),
                    "n_candidates": len(candidates),
                    "n_distinct": chosen.n_distinct,
                    "budget": budget,
                    "dof_positional": dof,
                })
    finally:
        selection_log.close()
    return totals


def _prescreen_budget(gene: dict, schedule, select: str) -> tuple[Optional[int], int]:
    """How many pre-screened structures this gene hands the scoring potential.

    Args:
        gene: A PyXtal-notation gene.
        schedule: Parsed trial schedule.
        select: ``"dof"`` keeps the schedule's own per-DoF allotment, which is
            what makes the wide-then-narrow arm cost the same as the baseline.
            An integer overrides it -- ``"1"`` is the NEP89-first arm, where the
            cheap potential has already run the whole protocol and only its
            single winner is worth refining.

    Returns:
        ``(dof, budget)``.
    """
    dof, budget = _budget(gene, schedule)
    if select == "dof":
        return dof, budget
    fixed = int(select)
    if fixed < 1:
        raise ValueError(f"--prescreen-select must be 'dof' or >= 1, got {select!r}")
    return dof, fixed


def _prescreen_select(args, genes: list[dict], schedule) -> dict:
    """Deduplicate and select, from the per-trial pre-relaxation outputs.

    Rewrites ``prescreen.extxyz`` and ``prescreen_selection.csv`` in full: both
    are a deterministic function of ``prescreen.csv`` and
    ``prescreen_all.extxyz``, so there is nothing to resume and re-deciding
    costs a few matcher calls per gene.

    Returns:
        Manifest entries: the candidate, duplicate, rejected and selected
        counts, and how much of the pre-relaxation fell back to Lennard-Jones.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    from wyckoff_transformer.cryspr.prescreen import Candidate

    rows = read_rows(args.output_dir / PRESCREEN_TRIALS_FILE, PRESCREEN_COLUMNS)
    relaxed_path = args.output_dir / PRESCREEN_ALL_FILE
    if not relaxed_path.is_file():
        # Every pre-relaxation failed.  A stage that got this far and then
        # raised would look like a crash rather than a cohort the cheap
        # potential could not handle, so report it as the latter.
        logger.error(
            "No pre-relaxed structures at %s; %d draws, all failed. "
            "Nothing to select from.", relaxed_path, len(rows),
        )
        return {
            "prescreen_genes": 0, "prescreen_candidates": 0,
            "prescreen_duplicates": 0, "prescreen_rejected": 0,
            "prescreen_selected": 0, "prescreen_failed": int(len(rows)),
            "prescreen_by_backend": {},
        }
    frames = {
        (int(atoms.info["gene"]), int(atoms.info["trial"])): atoms
        for _, _, atoms in _read_draws(
            relaxed_path, args.limit, produced_by="prescreen"
        )
    }
    ok = rows[rows["status"] == "ok"]
    candidates_by_gene: dict[int, list] = {}
    adaptor = AseAtomsAdaptor()
    for index, group in ok.groupby("index"):
        index = int(index)
        for row in group.itertuples():
            atoms = frames.get((index, int(row.trial)))
            if atoms is None:
                logger.warning(
                    "Gene %d trial %d: pre-relaxed row with no frame; skipping",
                    index, int(row.trial),
                )
                continue
            candidates_by_gene.setdefault(index, []).append(
                Candidate(
                    trial=int(row.trial),
                    energy_per_atom=float(row.energy_per_atom),
                    structure=adaptor.get_structure(atoms),
                )
            )

    totals = _select_and_write(
        args, genes, schedule,
        candidates_by_gene=candidates_by_gene,
        frames=frames,
        structures_path=args.output_dir / PRESCREEN_FILE,
        selection_path=args.output_dir / PRESCREEN_SELECTION_FILE,
        tag="prescreened",
    )

    backends = ok["backend"].value_counts() if "backend" in ok else {}
    return {
        "prescreen_genes": totals["genes"],
        "prescreen_candidates": totals["candidates"],
        "prescreen_duplicates": totals["duplicates"],
        "prescreen_rejected": totals["rejected"],
        "prescreen_selected": totals["selected"],
        "prescreen_failed": int((rows["status"] != "ok").sum()),
        # How much of the pre-relaxation was the potential and how much the
        # fallback: the latter is a geometry regulariser with no fitted
        # chemistry, so a gene ranked through it was effectively picked at
        # random even though its geometry is sane.
        "prescreen_by_backend": {
            str(name): int(count) for name, count in dict(backends).items()
        },
    }


# --------------------------------------------------------------------------- #
# Optional stage: basinhop
# --------------------------------------------------------------------------- #
def _basinhop_one(
    index: int,
    trial: int,
    atoms,
    trial_dir: str,
    n_hops: int,
    temperature: float,
    stdev: float,
    strain_stdev: float,
    fmax: float,
    timeout: Optional[float],
):
    """Walk between symmetry-preserving minima from one draw, in a pool worker.

    Returns:
        ``(row, minima)`` where *minima* is a list of ``(trial_key, atoms,
        energy_per_atom)``.  The trial key is synthetic and stable across
        reruns; see :data:`BASINHOP_TRIAL_BASE`.
    """
    from wyckoff_transformer.cryspr.basin_hopping import symmetric_basin_hop
    from wyckoff_transformer.cryspr.generator import _trial_seed
    from wyckoff_transformer.cryspr.relaxer import _get_spacegroup_info

    started = time.time()
    row = {
        "index": index, "trial": trial, "status": "failed",
        "device": _WORKER_DEVICE, "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(mode="metal"),
        "spacegroup": _get_spacegroup_info(atoms, symprec=1e-3)[1],
        "n_hops": n_hops, "n_accepted": None, "n_minima": None,
        "n_symmetry_lost": None, "n_symmetry_gained": None, "n_failed": None,
        "energy_start": None, "energy_best": None,
        "energy_per_atom_best": None, "backend": None, "error": None,
    }
    minima = []
    try:
        with time_limit(timeout):
            result = symmetric_basin_hop(
                atoms,
                calculator=_WORKER_PRERELAX_CALCULATOR,
                n_steps=n_hops,
                temperature=temperature,
                rattle_stdev=stdev,
                strain_stdev=strain_stdev,
                fmax=fmax,
                wdir=Path(trial_dir),
                seed=_trial_seed(index, trial),
            )
    except Timeout as exc:
        row["status"] = "timeout"
        row["error"] = str(exc)
        logger.warning("Gene %d trial %d: basin hop timed out (%s)", index, trial, exc)
    except Exception as exc:  # noqa: BLE001 - one bad walk must not stop the stage
        row["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Gene %d trial %d: basin hop failed (%s)", index, trial, exc)
    else:
        row.update(
            n_accepted=result.n_accepted,
            n_minima=len(result.minima),
            n_symmetry_lost=result.n_symmetry_lost,
            n_symmetry_gained=result.n_symmetry_gained,
            n_failed=result.n_failed,
            backend=getattr(_WORKER_PRERELAX_CALCULATOR, "last_backend", None),
        )
        if result.minima:
            row["status"] = "ok"
            row["energy_best"] = result.best.energy
            row["energy_per_atom_best"] = result.best.energy_per_atom
            row["energy_start"] = next(
                (m.energy for m in result.minima if m.step == 0), None
            )
            for minimum in result.minima:
                key = BASINHOP_TRIAL_BASE + trial * BASINHOP_TRIAL_STRIDE + minimum.step
                minima.append((key, minimum.atoms, minimum.energy_per_atom))
        else:
            row["error"] = "the walk found no minimum"
    row["seconds"] = round(time.time() - started, 2)
    return row, minima


def stage_basinhop(args) -> None:
    """Search each draw's symmetry-preserving basins, then select as usual.

    Where ``prescreen`` relaxes a *fixed* set of independent draws and keeps the
    best, this searches: from each draw it walks between adjacent minima on the
    cheap potential, with every step projected onto the gene's own space group
    (:func:`~wyckoff_transformer.cryspr.relaxer.symmetric_perturb`), and offers
    every distinct minimum it found to the same deduplicate-and-select step.
    ``relax --relax-from basinhop`` then spends the schedule's usual number of
    scoring relaxations on the survivors.

    The two arms therefore differ in *how* the candidate pool is built and not
    in what happens to it, which is the comparison worth making: independent
    draws sample the whole space badly, a walk samples a neighbourhood well.

    Changes no default: not part of ``--stage all``, and ``relax`` reads
    ``pyxtal.extxyz`` unless told otherwise.
    """
    from ase.io import write as ase_write

    genes = load_genes(args.input)
    schedule = parse_trial_schedule(args.n_trials)
    slots = resolve_devices(args.cores, args.devices, args.workers_per_device)
    mlip = (
        getattr(args, "basinhop_mlip", None)
        or getattr(args, "prescreen_mlip", None)
        or DEFAULT_PRERELAX_MLIP
    )
    draws = _read_draws(args.output_dir / PYXTAL_FILE, args.limit)

    log = RowLog(args.output_dir / BASINHOP_TRIALS_FILE, BASINHOP_COLUMNS, args.resume, getattr(args, "retry_failed", False))
    minima_path = args.output_dir / BASINHOP_ALL_FILE
    if not args.resume and minima_path.exists():
        minima_path.unlink()
    todo = [(i, t, atoms) for i, t, atoms in draws if (i, t) not in log.done]
    if log.done:
        logger.info("Resuming: %d walks already run", len(log.done))
    logger.info(
        "Basin hopping from %d draws with %s, %d hops each, on %d worker(s): %s",
        len(todo), mlip, args.basinhop_steps, len(slots), ", ".join(sorted(set(slots))),
    )

    for key, value in _SINGLE_THREAD_ENV_VARS.items():
        os.environ.setdefault(key, value)
    ctx = multiprocessing.get_context("spawn")
    claimed = ctx.Value("i", 0)
    started = time.time()
    try:
        with ProcessPoolExecutor(
            max_workers=len(slots),
            mp_context=ctx,
            initializer=_init_relax_worker,
            initargs=(claimed, slots, None, mlip, args.debug),
        ) as pool:
            futures = {
                pool.submit(
                    _basinhop_one,
                    index,
                    trial,
                    atoms,
                    str(_trial_dir(args.output_dir, index, trial) / "basinhop"),
                    args.basinhop_steps,
                    args.basinhop_temperature,
                    args.basinhop_stdev,
                    args.basinhop_strain_stdev,
                    args.prescreen_fmax,
                    args.relax_timeout,
                ): (index, trial)
                for index, trial, atoms in todo
            }
            for done, future in enumerate(as_completed(futures), start=1):
                index, trial = futures[future]
                try:
                    row, minima = future.result()
                except Exception as exc:  # noqa: BLE001 - a worker died
                    logger.warning("Gene %d trial %d: worker failed (%s)", index, trial, exc)
                    row, minima = (
                        {"index": index, "trial": trial, "status": "failed",
                         "error": f"{type(exc).__name__}: {exc}"},
                        [],
                    )
                for key, atoms, energy_per_atom in minima:
                    atoms.info = {
                        "gene": int(index), "trial": int(key),
                        "walk_trial": int(trial),
                        "energy_per_atom": float(energy_per_atom),
                    }
                    ase_write(str(minima_path), atoms, format="extxyz", append=True)
                log.write(row)
                if done % 25 == 0 or done == len(futures):
                    rate = (time.time() - started) / done
                    logger.info(
                        "Walked %d/%d (%.1f s/walk, %.0f min left)",
                        done, len(futures), rate, rate * (len(futures) - done) / 60,
                    )
    finally:
        log.close()

    selection = _basinhop_select(args, genes, schedule)
    rows = log.frame()
    _update_manifest(args.output_dir / MANIFEST_FILE, {
        "basinhop_mlip": mlip,
        "basinhop_steps": args.basinhop_steps,
        "basinhop_temperature": args.basinhop_temperature,
        "basinhop_stdev": args.basinhop_stdev,
        "basinhop_strain_stdev": args.basinhop_strain_stdev,
        "basinhop_walks": int(len(rows)),
        "basinhop_walks_ok": int((rows["status"] == "ok").sum()),
        "basinhop_accepted_per_walk": (
            round(float(rows["n_accepted"].mean(skipna=True)), 2) if len(rows) else None
        ),
        # Zero by construction; recorded so that "symmetry-constrained" is a
        # measured claim about this run rather than a property of the code.
        "basinhop_symmetry_lost": int(rows["n_symmetry_lost"].fillna(0).sum()),
        # Kept, not rejected: a supergroup still has the gene's own operations.
        "basinhop_symmetry_gained": int(rows["n_symmetry_gained"].fillna(0).sum()),
        **selection,
    })
    print(
        f"{selection['basinhop_selected']} of {selection['basinhop_candidates']} "
        f"minima selected over {selection['basinhop_genes']} genes "
        f"({selection['basinhop_duplicates']} duplicates, "
        f"{selection['basinhop_rejected']} distinct but over budget) "
        f"-> {args.output_dir / BASINHOP_FILE}"
    )


def _basinhop_select(args, genes: list[dict], schedule) -> dict:
    """Pool every walk's minima per gene, deduplicate, and keep the budget.

    The pooling is across a gene's *walks*, not within one: three draws each
    walking twenty hops is one search of sixty minima for that gene, and two
    walks that converged on the same basin should cost one scoring relaxation
    and not two.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    from wyckoff_transformer.cryspr.prescreen import Candidate

    minima_path = args.output_dir / BASINHOP_ALL_FILE
    if not minima_path.is_file():
        logger.error("No minima at %s; nothing to select from.", minima_path)
        return {
            "basinhop_genes": 0, "basinhop_candidates": 0, "basinhop_duplicates": 0,
            "basinhop_rejected": 0, "basinhop_selected": 0,
        }

    adaptor = AseAtomsAdaptor()
    frames, candidates_by_gene = {}, {}
    for index, key, atoms in _read_draws(minima_path, args.limit, produced_by="basinhop"):
        frames[(index, key)] = atoms
        candidates_by_gene.setdefault(index, []).append(
            Candidate(
                trial=key,
                energy_per_atom=float(atoms.info["energy_per_atom"]),
                structure=adaptor.get_structure(atoms),
            )
        )

    totals = _select_and_write(
        args, genes, schedule,
        candidates_by_gene=candidates_by_gene,
        frames=frames,
        structures_path=args.output_dir / BASINHOP_FILE,
        selection_path=args.output_dir / BASINHOP_SELECTION_FILE,
        tag="basinhopped",
    )
    return {f"basinhop_{key}": value for key, value in totals.items()}


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
    prerelax_fmax: float = 0.1,
    prerelax_max_expansion: Optional[float] = None,
) -> dict:
    """Relax one generated draw in a pool worker.  Returns a row for the CSV.

    Two-stage when the worker built a pre-relaxation calculator: the cheap
    potential runs the symmetry-constrained schedule first, into the trial's
    ``prerelax/`` sub-directory, and the scoring potential starts from its
    output.  The timeout bounds the pair, since it is the trial that must not
    hold a worker forever.
    """
    from wyckoff_transformer.cryspr.generator import (
        KEPT_CIF_SUFFIX,
        PRERATTLE_CIF_SUFFIX,
        _trial_seed,
        relax_trial,
    )

    started = time.time()
    row = {
        "index": index, "trial": trial, "status": "failed",
        "device": _WORKER_DEVICE, "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(mode="metal"),
        "energy": None, "energy_per_atom": None, "cif": None,
        "energy_prerattle": None, "energy_per_atom_prerattle": None,
        "cif_prerattle": None, "error": None,
    }
    try:
        with time_limit(timeout):
            relaxed, energy, prerattle = relax_trial(
                atoms_in=atoms,
                calculator=_WORKER_CALCULATOR,
                trial_dir=Path(trial_dir),
                label=f"gene {index} trial {trial}",
                release_symmetry=release_symmetry,
                rattle=rattle,
                seed=_trial_seed(index, trial),
                fmax=fmax,
                prerelax_calculator=_WORKER_PRERELAX_CALCULATOR,
                prerelax_fmax=prerelax_fmax,
                prerelax_max_expansion=prerelax_max_expansion,
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
            prerattle_atoms, prerattle_energy = prerattle
            row["energy_prerattle"] = prerattle_energy
            row["energy_per_atom_prerattle"] = prerattle_energy / len(prerattle_atoms)
            row["cif_prerattle"] = str(
                Path(trial_dir) / f"{row['formula']}{PRERATTLE_CIF_SUFFIX}"
            )
    row["seconds"] = round(time.time() - started, 2)
    return row


def _read_draws(
    path: Path, limit: Optional[int], produced_by: str = "generate"
) -> list[tuple[int, int, object]]:
    """The generated structures, as ``(gene, trial, atoms)``.

    Args:
        path: An extxyz written by ``generate``, ``template`` or ``prescreen``,
            one frame per draw, tagged with its gene and trial.
        limit: Ignore genes at or above this index.
        produced_by: Stage named in the error when the file is missing.

    Raises:
        FileNotFoundError: If that stage has not run.
    """
    from ase.io import read as ase_read

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No structures at {path}. Run --stage {produced_by} first."
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
    source = getattr(args, "relax_from", "pyxtal")
    if source not in RELAX_SOURCES:
        raise ValueError(f"--relax-from must be one of {sorted(RELAX_SOURCES)}")
    draws = _read_draws(
        args.output_dir / RELAX_SOURCES[source],
        args.limit,
        produced_by="generate" if source == "pyxtal" else source,
    )
    prerelax_mlip = getattr(args, "prerelax_mlip", None)
    if source == "prescreen" and prerelax_mlip:
        # Not refused: pre-relaxing again on a *different* potential is a
        # coherent thing to ask for.  But the wide-then-narrow variant's draws
        # are already pre-relaxed, so the same potential twice is pure waste and
        # worth saying out loud.
        logger.warning(
            "--relax-from prescreen draws are already pre-relaxed; "
            "--prerelax-mlip %s will relax them again before %s",
            prerelax_mlip, args.mlip,
        )

    log = RowLog(args.output_dir / RELAXATIONS_FILE, RELAXATION_COLUMNS, args.resume, getattr(args, "retry_failed", False))
    todo = [(i, t, atoms) for i, t, atoms in draws if (i, t) not in log.done]
    if log.done:
        logger.info("Resuming: %d trials already relaxed", len(log.done))
    logger.info(
        "Relaxing %d %s draws with %s%s on %d worker(s): %s",
        len(todo), source, args.mlip,
        f" (pre-relaxed with {prerelax_mlip})" if prerelax_mlip else "",
        len(slots), ", ".join(sorted(set(slots))),
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
            initargs=(claimed, slots, args.mlip, prerelax_mlip, args.debug),
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
                    getattr(args, "prerelax_fmax", 0.1),
                    getattr(args, "prerelax_max_expansion", None),
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
        "relax_from": source,
        # None, not omitted: "this run was single-stage" and "this run predates
        # the two-stage option" are different facts about a manifest.
        "prerelax_mlip": prerelax_mlip,
        "prerelax_fmax": getattr(args, "prerelax_fmax", None) if prerelax_mlip else None,
        "prerelax_max_expansion": (
            getattr(args, "prerelax_max_expansion", None) if prerelax_mlip else None
        ),
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
    # Only present in a wide-then-narrow run.  Its per-gene counts and seconds
    # are what make the two arms comparable on cost: without them a run that
    # drew ten times as many starts looks identical to one that drew the usual
    # number, since `relax` sees the same number of trials either way.
    prescreened = read_rows(output_dir / PRESCREEN_TRIALS_FILE, PRESCREEN_COLUMNS)
    by_gene_prescreen = {
        index: group for index, group in prescreened.groupby("index")
    } if len(prescreened) else {}
    cif_dir = output_dir / CIF_DIR
    cif_dir.mkdir(parents=True, exist_ok=True)
    prerattle_dir = output_dir / PRERATTLE_CIF_DIR
    prerattle_dir.mkdir(parents=True, exist_ok=True)

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
            # The pre-rattle readout is a separate selection, not a column of
            # the same trial: the lowest-energy trial before the rattle need not
            # be the lowest-energy one after it, and reporting the winner's
            # pre-rattle energy instead would answer neither question.
            "has_prerattle": False, "energy_prerattle": None,
            "energy_per_atom_prerattle": None, "best_trial_prerattle": None,
        }
        if by_gene_prescreen:
            pre = by_gene_prescreen.get(index, prescreened.iloc[:0])
            row["n_prescreened"] = int((pre["status"] == "ok").sum())
            row["prescreen_seconds"] = round(float(pre["seconds"].fillna(0).sum()), 2)
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

            prerattled = succeeded[succeeded["energy_prerattle"].notna()] \
                if "energy_prerattle" in succeeded.columns else succeeded.iloc[:0]
            if len(prerattled):
                best_pre = prerattled.loc[prerattled["energy_prerattle"].idxmin()]
                row.update(
                    has_prerattle=True,
                    energy_prerattle=float(best_pre["energy_prerattle"]),
                    energy_per_atom_prerattle=float(best_pre["energy_per_atom_prerattle"]),
                    best_trial_prerattle=int(best_pre["trial"]),
                )
                pre_source = Path(str(best_pre["cif_prerattle"]))
                if pre_source.is_file():
                    (prerattle_dir / f"{index}.cif").write_text(
                        pre_source.read_text(encoding="utf-8"), encoding="utf-8"
                    )
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

    gene_fingerprints = {}
    for index in frame.index:
        try:
            gene_fingerprints[index] = fingerprinter.fingerprint(genes[index])
        except Exception as exc:
            logger.warning("Gene %d: no fingerprint (%s)", index, exc)

    def read_variant(directory: Path, energy_column: str) -> dict:
        """Structures, validity, relaxed fingerprints and hull energies for one readout."""
        out = {"validity": {}, "structures": {}, "relaxed": {}, "hull": {}}
        if not directory.is_dir():
            return out
        for index in frame.index:
            cif_path = directory / f"{index}.cif"
            if not cif_path.is_file():
                continue
            try:
                structure = Structure.from_file(cif_path)
            except Exception as exc:
                logger.warning("Gene %d: unreadable CIF at %s (%s)", index, cif_path, exc)
                continue
            out["validity"][index] = is_valid(structure)
            out["structures"][index] = structure
            # The relaxed structure's own fingerprint, which relaxation -- the
            # rattle stage especially -- can move away from the sampled gene's.
            # Novelty is judged against both, so a gene that PyXtal placed on a
            # LeMat-Bulk-known orbit set but relaxed off it is still checked.
            try:
                out["relaxed"][index] = fingerprinter.fingerprint_structure(structure)
            except Exception as exc:
                logger.warning("Gene %d: no relaxed fingerprint (%s)", index, exc)
            energy = frame.at[index, energy_column] if energy_column in frame else None
            if energy is not None and pd.notna(energy):
                try:
                    # The energy came from the same potential that defines this
                    # hull, which is why --mlip is restricted to published hulls.
                    out["hull"][index] = hull.energy_above_hull(
                        float(energy), structure.composition
                    )
                except ValueError as exc:
                    logger.warning("Gene %d: e_above_hull failed (%s)", index, exc)
        return out

    kept = read_variant(cif_dir, "energy")
    phase.done("read CIFs, validity, fingerprints and e_above_hull")

    # The same measurements on the structure the rattle stage was handed.  A
    # separate pass rather than a column, because it is a different structure
    # and every downstream verdict -- validity, uniqueness, novelty, the hull --
    # has to be recomputed on it rather than inherited.
    want_prerattle = getattr(args, "prerattle_metrics", True)
    prerattle_dir = args.output_dir / PRERATTLE_CIF_DIR
    prerattle = (
        read_variant(prerattle_dir, "energy_prerattle")
        if want_prerattle else {"validity": {}, "structures": {}, "relaxed": {}, "hull": {}}
    )
    if want_prerattle and not prerattle["structures"]:
        logger.info(
            "No pre-rattle structures at %s; the pre-rattle metrics will be "
            "reported as null. A run relaxed before they were recorded has none.",
            prerattle_dir,
        )
    elif want_prerattle:
        phase.done("read pre-rattle CIFs, validity, fingerprints and e_above_hull")
    def attach(variant: dict, suffix: str) -> pd.DataFrame:
        """Write a variant's per-gene verdicts onto *frame*, and return its scored rows."""
        frame[f"valid_structure{suffix}"] = pd.Series(variant["validity"])
        frame[f"e_above_hull{suffix}"] = pd.Series(variant["hull"])
        frame[f"relaxed_fingerprint_resolved{suffix}"] = pd.Series(
            {index: index in variant["relaxed"] for index in frame.index}
        )
        frame[f"relaxed_fingerprint_changed{suffix}"] = pd.Series(
            {
                index: variant["relaxed"][index] != gene_fingerprints.get(index)
                for index in variant["relaxed"]
            }
        )
        # Only structures that got this far can be unique or novel, and
        # comparing the ones that did not would just cost matcher calls.
        scored = pd.DataFrame(
            {
                "fingerprint": pd.Series(gene_fingerprints),
                "structure": pd.Series(variant["structures"]),
            }
        ).dropna()
        scored = scored.loc[
            [i for i in scored.index if bool(variant["validity"].get(i, False))]
        ]
        scored["relaxed_fingerprint"] = pd.Series(variant["relaxed"]).reindex(scored.index)
        return scored

    novel_genes = set(screen.novel)
    frame["gene_novel"] = pd.Series(
        {index: index in novel_genes for index in frame.index}
    )
    scored_kept = attach(kept, "")
    scored_pre = attach(prerattle, "_prerattle") if prerattle["structures"] else None

    for scored, suffix in ((scored_kept, ""), (scored_pre, "_prerattle")):
        if scored is None:
            continue
        unique_index = set(filter_by_unique_structure(scored).index)
        frame[f"unique_structure{suffix}"] = pd.Series(
            {index: index in unique_index for index in scored.index}
        )
    phase.done("uniqueness (StructureMatcher)")

    # The matcher needs a candidate for either fingerprint, and one reference
    # serves both readouts: built over the union of all four fingerprint sets,
    # so the streaming pass over the ~1 GB CIF export happens once rather than
    # twice.  That pass is the score stage's dominant cost.
    fingerprint_sets = [scored_kept["fingerprint"], scored_kept["relaxed_fingerprint"].dropna()]
    if scored_pre is not None:
        fingerprint_sets += [
            scored_pre["fingerprint"], scored_pre["relaxed_fingerprint"].dropna()
        ]
    reference = build_novelty_reference(
        pd.concat(fingerprint_sets),
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

    for scored, suffix in ((scored_kept, ""), (scored_pre, "_prerattle")):
        if scored is None:
            continue
        frame[f"novel_by_sampled_gene{suffix}"] = pd.Series(
            {index: novelty_filter.is_novel(row) for index, row in scored.iterrows()}
        )
        frame[f"novel_structure{suffix}"] = pd.Series(
            {index: _is_novel(row) for index, row in scored.iterrows()}
        )
    phase.done("novelty (StructureMatcher)")

    # What the rattle did, per gene, on the three axes it can move: the gene's
    # own orbit set, novelty, and the energy thresholds.
    if scored_pre is not None:
        frame["rattle_moved_off_gene"] = pd.Series(
            {
                index: (
                    kept["relaxed"].get(index) != gene_fingerprints.get(index)
                    and prerattle["relaxed"].get(index) == gene_fingerprints.get(index)
                )
                for index in frame.index
                if index in kept["relaxed"] and index in prerattle["relaxed"]
            }
        )
        frame["rattle_lowered_energy"] = pd.Series(
            {
                index: bool(
                    pd.notna(frame.at[index, "energy"])
                    and pd.notna(frame.at[index, "energy_prerattle"])
                    and frame.at[index, "energy_per_atom"]
                    < frame.at[index, "energy_per_atom_prerattle"] - 1e-9
                )
                for index in frame.index
            }
        )

    frame.drop(columns=["structure"], errors="ignore").to_csv(
        args.output_dir / STRUCTURES_FILE
    )
    _update_manifest(
        args.output_dir / MANIFEST_FILE,
        {
            "hull": hull.provenance,
            "novelty": "sampled+relaxed fingerprint",
            "prerattle_metrics": scored_pre is not None,
        },
    )
    report = funnel(screen, frame)
    if scored_pre is not None:
        report.update(funnel(screen, frame, prefix="prerattle_"))
        report.update(rattle_effect(screen, frame))
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
        "--stage", choices=STAGES + OPTIONAL_STAGES + ("all",), default="all",
        help=(
            "Which stage to run. 'all' runs screen, generate, relax and score. "
            "'template' and 'prescreen' are not part of 'all': the first adds one "
            "template-matched start per gene to the draws 'generate' made, to be "
            "relaxed alongside them; the second relaxes every draw on a cheap "
            "potential under fixed symmetry, drops duplicates, and writes the "
            "schedule's usual number of survivors for --relax-from prescreen."
        ),
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
        "--pyxtal-tol-factor", type=float, default=DEFAULT_PYXTAL_TOL_FACTOR,
        help=(
            "Scale on PyXtal's inter-atomic distance floor, which is "
            "0.5*(r_a+r_b) -- the covalent-radius mean -- times this. Lower is "
            "more permissive. PyXtal rejection-samples orbit by orbit, so the "
            "factor biases the joint draw towards loose configurations rather "
            "than bounding it: 45%% of returned draws are already below the "
            "nominal floor. Kept at 1.3, which every published number was "
            "measured with; see docs/pyxtal_tolerance_sweep.md for the sweep "
            "that decided against lowering it."
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

    generation.add_argument(
        "--trial-multiplier", type=int, default=1,
        help=(
            "Draw this multiple of the schedule's trials. 1, the default, is the "
            "published protocol. The wide-then-narrow variant sets 10 and pairs it "
            "with --stage prescreen, which selects the unmultiplied number back "
            "down after relaxing every draw on a cheap potential -- so the "
            "expensive relaxation count is unchanged and only the choice of start "
            "improves. It multiplies rather than replacing the schedule so the "
            "extra starts stay proportional to the free coordinates that make a "
            "start worth repeating."
        ),
    )

    template = parser.add_argument_group("template starts (stage: template)")
    template.add_argument(
        "--template-index", type=Path, default=None,
        help=(
            "Parquet of LeMat-Bulk keyed by anonymous Wyckoff fingerprint. Built "
            "from --reference-cache on first use and cached beside it."
        ),
    )
    template.add_argument(
        "--template-candidates", type=int, default=4,
        help=(
            "Templates carried out of the index per gene, closest formula first. "
            "More than one because the closest can resist symmetry detection, and "
            "reading its geometry costs a pass over the CIF export either way."
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
        "--relax-from", type=str, default="pyxtal", choices=sorted(RELAX_SOURCES),
        help=(
            "Where the starting structures come from: 'pyxtal' is what 'generate' "
            "(and 'template') wrote, 'prescreen' is what the pre-screen selected. "
            "The default is the published protocol."
        ),
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

    prerelax = parser.add_argument_group("two-stage relaxation (stage: relax)")
    prerelax.add_argument(
        "--prerelax-mlip", type=str, default=None, choices=prerelax_mlip_names(),
        help=(
            "Relax every trial on this cheap potential first, then hand its "
            "geometry to --mlip. Off by default, which is the published protocol. "
            "'nep89' is the 89-element neuroevolution potential of "
            "arXiv:2504.21286, ~1 ms per force call against ORB's tens to "
            "hundreds, with a ZBL-cored fallback for the five elements below Pu "
            "it omits (Po, At, Rn, Fr, Ra) and everything above. "
            "Its energies are never scored: this option changes which geometry "
            "--mlip starts from and nothing else. Not restricted to published "
            "hulls, for exactly that reason."
        ),
    )
    prerelax.add_argument(
        "--prerelax-fmax", type=float, default=PRERELAX_FMAX,
        help=(
            "Force convergence of the pre-relaxation, eV/A. Looser than --fmax on "
            "purpose: the stage delivers a starting geometry, and converging it "
            "tightly on a potential whose minimum is not the one being scored "
            "just reaches the wrong stationary point more precisely."
        ),
    )

    prerelax.add_argument(
        "--prerelax-max-expansion", type=float, default=None,
        help=(
            "Fall back to the raw PyXtal draw if the pre-relaxation grew the "
            "cell by more than this factor. Unset by default, which keeps "
            "whatever the cheap potential produced. The guard exists because a "
            "PyXtal draw is deliberately loose (tolerance factor 1.3, median "
            "1.68x the target volume) and the protocol relies on compressive "
            "relaxation: a pre-relaxation that *inflates* the cell hands the "
            "scoring potential a symmetric stationary point in an expanded "
            "cell, and gradient descent under a symmetry constraint cannot "
            "leave one -- so only the rattle recovers it. Set it from the "
            "volume_ratio distribution the run records in each trial's "
            "prerelax.json rather than guessing."
        ),
    )

    prescreen = parser.add_argument_group("wide-then-narrow (stage: prescreen)")
    prescreen.add_argument(
        "--prescreen-mlip", type=str, default=None, choices=prerelax_mlip_names(),
        help=(
            "Potential the pre-screen relaxes and ranks with. Defaults to "
            "--prerelax-mlip if that is set, otherwise to %s." % DEFAULT_PRERELAX_MLIP
        ),
    )
    prescreen.add_argument(
        "--prescreen-fmax", type=float, default=PRERELAX_FMAX,
        help="Force convergence of the pre-screen relaxation, eV/A.",
    )
    prescreen.add_argument(
        "--prescreen-dedup", type=str, default="matcher", choices=("matcher", "energy"),
        help=(
            "How two pre-relaxed draws of one gene are judged the same structure. "
            "'matcher' is StructureMatcher at pymatgen's defaults, the same "
            "tolerances the protocol's uniqueness and novelty filters use, so a "
            "pair this stage calls distinct is one the funnel would too. 'energy' "
            "uses --prescreen-energy-tol alone, which is faster and merges "
            "distinct structures that happen to be degenerate."
        ),
    )
    prescreen.add_argument(
        "--prescreen-release-symmetry", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the unconstrained stage in the pre-screen too. Off by default, "
            "because the wide-then-narrow arm is choosing between draws *of one "
            "gene* and a draw that left its space group on the cheap potential is "
            "no longer a draw of that gene. Turning it on, with --prescreen-rattle "
            "and --prescreen-select 1, is the NEP89-first arm: the cheap potential "
            "runs the whole schedule and the scoring one refines its single winner."
        ),
    )
    prescreen.add_argument(
        "--prescreen-rattle", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Run the rattle stage in the pre-screen. Off by default; see "
            "--prescreen-release-symmetry."
        ),
    )
    prescreen.add_argument(
        "--prescreen-select", type=str, default="dof",
        help=(
            "How many surviving structures per gene to hand the scoring "
            "potential. 'dof' keeps the trial schedule's own per-DoF allotment, "
            "which is what makes the wide-then-narrow arm cost the same as the "
            "baseline. An integer overrides it: '1' is the NEP89-first arm."
        ),
    )
    prescreen.add_argument(
        "--prescreen-energy-tol", type=float, default=DEDUP_ENERGY_TOL_EV_PER_ATOM,
        help=(
            "Energy gap, eV/atom, above which two pre-relaxed draws cannot be the "
            "same structure, so the matcher is not called. Two relaxations of one "
            "minimum on one potential agree far more closely than this; the error "
            "direction is safe, since too tight a gate keeps a duplicate (one "
            "extra relaxation) where too loose a one would merge two structures."
        ),
    )

    basinhop = parser.add_argument_group("basin hopping (stage: basinhop)")
    basinhop.add_argument(
        "--basinhop-mlip", type=str, default=None, choices=prerelax_mlip_names(),
        help="Potential the walk relaxes and ranks with. Defaults to "
             "--prescreen-mlip, else %s." % DEFAULT_PRERELAX_MLIP,
    )
    basinhop.add_argument(
        "--basinhop-steps", type=int, default=BASINHOP_STEPS,
        help=(
            "Hops per starting draw. Each costs one local relaxation on the "
            "cheap potential, so this is seconds per draw where a scoring "
            "relaxation is seconds to minutes."
        ),
    )
    basinhop.add_argument(
        "--basinhop-temperature", type=float, default=BASINHOP_TEMPERATURE,
        help=(
            "Metropolis temperature, eV/atom. 0 accepts only downhill moves, "
            "which makes the walk a greedy descent that stalls in the first "
            "basin; the default is roughly room temperature."
        ),
    )
    basinhop.add_argument(
        "--basinhop-stdev", type=float, default=BASINHOP_STDEV,
        help=(
            "Displacement drawn per hop, A, before projection onto the space "
            "group's subspace shrinks it to about 0.68 of this. Six times the "
            "rattle stage's 0.05, which was calibrated to break symmetry rather "
            "than to cross a barrier: at 0.05 a 15-hop walk revisits one basin, "
            "and the distinct-minimum count saturates from 0.15 upwards."
        ),
    )
    basinhop.add_argument(
        "--basinhop-strain-stdev", type=float, default=BASINHOP_STRAIN_STDEV,
        help="Cell strain drawn per hop, dimensionless, before projection.",
    )

    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process genes with an index below this. For smoke tests.",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help=(
            "On --resume, re-run the trials that failed as well as the ones "
            "never attempted. Off by default, because a trial whose relaxation "
            "diverged has been answered and would answer the same way. A row "
            "whose error names a killed worker is re-run regardless: that is an "
            "infrastructure failure, and skipping it leaves a permanent hole in "
            "the cohort that reads as a collapsed success rate."
        ),
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

    score = parser.add_argument_group("scoring (stage: score)")
    score.add_argument(
        "--prerattle-metrics", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Report every metric a second time on the structure the rattle stage "
            "was handed, under a 'prerattle_' prefix, plus what the rattle "
            "changed ('rattle_moved_off_gene', 'rattle_novel_became_known', "
            "'rattle_metasun_lost' and their counterparts). On by default: the "
            "rattle lowers energy but can discard the Wyckoff orbits WyFormer "
            "predicted and can relax a novel structure onto a known one, and "
            "neither is visible from the kept structure alone. It costs a second "
            "pass of the matcher and the hull, not a second relaxation, and it "
            "changes nothing about which structure the protocol keeps. "
            "--no-prerattle-metrics reports the kept readout only."
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
