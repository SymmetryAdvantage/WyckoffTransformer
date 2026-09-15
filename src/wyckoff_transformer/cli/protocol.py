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
import hashlib
import io
import json
import logging
import multiprocessing
import os
import signal
import time
import uuid
import warnings
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from wyckoff_transformer.cli.worker_pool import (
    RETRYABLE_ERRORS,
    describe,
    is_retryable_error,
    is_technical_error,
    mark_worker_faulty,
    run_supervised,
    set_worker_device,
)
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
    default_fingerprint_cache,
    funnel,
    load_genes,
    load_reference_fingerprints,
    parse_trial_schedule,
    positional_dof,
    read_screen,
    reference_identity,
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
#: Per gene, the lowest-energy fixed-symmetry structure, scored like the kept one.
STRUCTURES_FIXED_FILE = "structures_fixed_symmetry.csv"
FUNNEL_FILE = "funnel.json"
MANIFEST_FILE = "manifest.json"
CIF_DIR = "cifs"
CIF_FIXED_DIR = "cifs_fixed_symmetry"

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
    "n_atoms", "device", "seconds", "cif", "error",
    # The fixed-symmetry readout of the same trial: the output of the
    # symmetry-constrained stages, before the release and the rattle.  Its own
    # status, because the clash guard judges the two structures separately.
    "status_fixed", "energy_fixed", "energy_per_atom_fixed", "cif_fixed",
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
_WORKER_BUDGET: Optional[DeviceMemoryBudget] = None

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
    #: whose worker was killed under it, or whose GPU failed, has not been
    #: answered at all, and skipping it silently turns an infrastructure failure
    #: into a permanent hole in the cohort -- 1034 of 1800 trials in one run
    #: here, which read as a collapsed reconstruction rate rather than as a
    #: crash.  See :mod:`wyckoff_transformer.cli.worker_pool`.
    RETRYABLE_ERRORS = RETRYABLE_ERRORS

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
        killed worker or a failed device is *not* done: see
        :data:`RETRYABLE_ERRORS`.
        """
        keep = pd.Series(True, index=previous.index)
        if not len(previous):
            return keep
        if "status" in previous.columns and retry_failed:
            keep &= previous["status"].astype(str) == "ok"
        if "error" in previous.columns:
            retryable = previous["error"].map(is_retryable_error).astype(bool)
            if retryable.any():
                logger.info(
                    "%s: %d row(s) record a killed worker or a failed device "
                    "rather than a failed trial; they will be re-run",
                    self.path, int(retryable.sum()),
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


def latest_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The last row of each ``(index, trial)``.

    A resumed stage appends its retry of a trial below the row that recorded
    the failure, so a log can hold several rows for one trial; the last is the
    trial's answer.
    """
    if not len(frame):
        return frame
    return frame.drop_duplicates(["index", "trial"], keep="last")


class IncompleteStageError(RuntimeError):
    """A stage log still has trials that a technical failure left unanswered."""


def unanswered_trials(
    output_dir: Path, files, limit: Optional[int] = None
) -> dict[str, int]:
    """Trials per log whose latest row says the machine, not the trial, failed.

    Args:
        output_dir: The protocol run directory.
        files: Stage log file names to check; missing ones are skipped.
        limit: Only count genes with an index below this, as ``--limit`` does.

    Returns:
        ``{file name: count}`` for the logs with any such trial.
    """
    counts = {}
    for name in files:
        frame = latest_rows(read_rows(Path(output_dir) / name))
        if not len(frame) or "error" not in frame.columns:
            continue
        if limit is not None:
            frame = frame[frame["index"] < limit]
        n = int(frame["error"].map(is_retryable_error).sum())
        if n:
            counts[name] = n
    return counts


#: Every per-trial log whose holes change what the score stage counts.
TRIAL_LOGS = (
    PYXTAL_TRIALS_FILE, PRESCREEN_TRIALS_FILE, BASINHOP_TRIALS_FILE, RELAXATIONS_FILE,
)


def require_complete(args, files) -> None:
    """Refuse to go on from logs that a technical failure left holes in.

    A trial lost to a failed GPU or a killed worker would otherwise be counted
    as a gene without a structure, and the funnel -- MSUN included -- would
    report an infrastructure failure as a property of the model.

    Raises:
        IncompleteStageError: Unless ``args.allow_incomplete`` is set.
    """
    counts = unanswered_trials(args.output_dir, files, getattr(args, "limit", None))
    if not counts:
        return
    summary = ", ".join(f"{n} in {name}" for name, n in counts.items())
    if getattr(args, "allow_incomplete", False):
        logger.warning("Going on despite unanswered trials (%s)", summary)
        return
    raise IncompleteStageError(
        f"Trials left unanswered by technical failures: {summary}. Re-run the "
        "stage that wrote them with --resume, which retries exactly those "
        "trials, or pass --allow-incomplete to score what there is."
    )


#: Manifest key recording what each resumable output was built from.
LINEAGE_KEY = "lineage"

#: The per-trial log each ``--relax-from`` source is the selection of.
RELAX_SOURCE_LOGS = {
    "pyxtal": PYXTAL_TRIALS_FILE,
    "prescreen": PRESCREEN_TRIALS_FILE,
    "basinhop": BASINHOP_TRIALS_FILE,
}


class StaleOutputError(RuntimeError):
    """An output directory holds rows built from inputs that have since changed."""


def genes_digest(genes: list[dict]) -> str:
    """An identity for a gene cohort that survives re-compressing its file.

    The gzip header carries a timestamp, so the file's own bytes differ between
    two writes of the same genes; the canonical JSON does not.
    """
    payload = json.dumps(genes, sort_keys=True, separators=(",", ":"))
    return "genes:sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_manifest(output_dir: Path) -> dict:
    path = Path(output_dir) / MANIFEST_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_lineage(output_dir: Path, name: str, record: dict) -> None:
    lineage = dict(_read_manifest(output_dir).get(LINEAGE_KEY) or {})
    lineage[name] = record
    _update_manifest(Path(output_dir) / MANIFEST_FILE, {LINEAGE_KEY: lineage})


def _new_lineage_id() -> str:
    return uuid.uuid4().hex


def claim_lineage(
    output_dir: Path,
    name: str,
    parent: str,
    resume: bool,
    verify: Optional[Callable[[], Optional[str]]] = None,
) -> str:
    """Tie the output *name* to what it is built from, before a stage opens it.

    A stage log is keyed by ``(gene index, trial)`` and nothing else, so on
    ``--resume`` a row is taken as done whatever it was computed from.  That is
    how ``protocol_ehull5x-20260904-213346:v1`` came to be scored on draws of
    one gene cohort filed under the indices of another: the wrapper sampled a
    fresh gene file into a directory whose PyXtal draws and relaxations it then
    resumed, and 789 of its 998 structures belonged to a gene other than the
    one they were scored as.

    Each output therefore records, under ``lineage`` in ``manifest.json``, an
    ``id`` of its own and the ``parent`` it was built from: the gene file's
    :func:`genes_digest` for ``screen.json`` and ``pyxtal.csv``, and the ``id``
    of the log a stage reads its structures from for everything downstream.  A
    fresh (not resumed) output gets a new id, which is what makes the outputs
    built from its predecessor detectably stale.

    Args:
        output_dir: The protocol run directory.
        name: The output's file name.
        parent: What it is being built from now.
        resume: Whether the stage keeps the rows already in the file.
        verify: For a file written before lineage was recorded: returns why
            its rows do not belong to *parent*, or ``None`` if they do.

    Returns:
        The output's id, to be passed as the *parent* of what is built from it.

    Raises:
        StaleOutputError: If resuming would keep rows built from something else.
    """
    record = (_read_manifest(output_dir).get(LINEAGE_KEY) or {}).get(name)
    path = Path(output_dir) / name
    if resume and path.is_file() and len(read_rows(path)):
        if record is not None and record.get("parent") == parent:
            return record["id"]
        if record is not None and record.get("parent") is not None:
            raise StaleOutputError(
                f"{path} was built from {record['parent']}, not from {parent}, "
                "so resuming it would mix the two. Resume with the inputs it was "
                "built from, or pass --no-resume to start the stage over."
            )
        # Written before lineage was recorded, or adopted unverified by a later
        # stage: its rows can only be checked against their content.
        problem = verify() if verify is not None else None
        if problem:
            raise StaleOutputError(
                f"{path} does not belong to {parent}: {problem}. Resume with the "
                "inputs it was built from, or pass --no-resume to start the "
                "stage over."
            )
        if verify is None:
            logger.warning("%s has no recorded lineage; adopting it unchecked", path)
        record = {"id": record["id"] if record else _new_lineage_id(), "parent": parent}
    else:
        record = {"id": _new_lineage_id(), "parent": parent}
    _write_lineage(output_dir, name, record)
    return record["id"]


def lineage_id(output_dir: Path, name: str) -> str:
    """The id of an existing output, recording one if it predates lineage.

    Such an output is adopted with an unknown parent: nothing downstream can be
    checked against where it came from, but everything built from it from now
    on can be checked against it.
    """
    record = (_read_manifest(output_dir).get(LINEAGE_KEY) or {}).get(name)
    if record is not None:
        return record["id"]
    record = {"id": _new_lineage_id(), "parent": None}
    _write_lineage(output_dir, name, record)
    return record["id"]


def _reduced_formula(formula) -> Optional[str]:
    from pymatgen.core import Composition

    if not isinstance(formula, str) or not formula:
        return None
    return Composition(formula).reduced_formula


def _gene_formula(gene: dict) -> Optional[str]:
    from pymatgen.core import Composition

    try:
        counts: dict[str, float] = {}
        for species, n in zip(gene["species"], gene["numIons"]):
            counts[species] = counts.get(species, 0) + n
        return Composition(counts).reduced_formula
    except Exception:  # noqa: BLE001 - an unreadable gene is checked elsewhere
        return None


def _composition_mismatches(rows: pd.DataFrame, expected: dict) -> Optional[str]:
    """Why the rows' compositions are not those *expected* per ``(index, trial)``.

    Rows without a formula (a failed draw) and keys *expected* does not cover
    are not evidence either way and are skipped.
    """
    if "formula" not in rows.columns:
        return None
    checked = mismatched = 0
    for index, trial, formula in zip(rows["index"], rows["trial"], rows["formula"]):
        want = expected.get((int(index), int(trial)))
        got = _reduced_formula(formula)
        if want is None or got is None:
            continue
        checked += 1
        mismatched += got != want
    if mismatched:
        return f"{mismatched} of {checked} rows have another composition"
    return None


def _verify_draws_log(output_dir: Path, name: str, genes: list[dict]):
    """A *verify* for ``pyxtal.csv``: every draw has its gene's composition."""
    def verify() -> Optional[str]:
        rows = read_rows(Path(output_dir) / name)
        formulas = [_gene_formula(gene) for gene in genes]
        expected = {
            (int(index), int(trial)): formulas[int(index)]
            if 0 <= int(index) < len(formulas) else "<no such gene>"
            for index, trial in zip(rows["index"], rows["trial"])
        }
        return _composition_mismatches(rows, expected)
    return verify


def _verify_against_draws(output_dir: Path, name: str, draws):
    """A *verify* for a log of work on *draws*: each row has its draw's composition."""
    def verify() -> Optional[str]:
        expected = {
            (index, trial): _reduced_formula(atoms.get_chemical_formula(mode="metal"))
            for index, trial, atoms in draws
        }
        return _composition_mismatches(read_rows(Path(output_dir) / name), expected)
    return verify


def require_consistent_lineage(output_dir: Path, genes: Optional[list[dict]]) -> None:
    """Refuse to score outputs that were not built from one another.

    Only recorded lineage is checked; a run from before it was recorded passes.

    Raises:
        StaleOutputError: Naming every output whose parent is not the current one.
    """
    lineage = _read_manifest(output_dir).get(LINEAGE_KEY) or {}
    if not lineage:
        return
    problems = []
    digest = genes_digest(genes) if genes is not None else None
    for name in (SCREEN_FILE, PYXTAL_TRIALS_FILE):
        parent = (lineage.get(name) or {}).get("parent")
        if digest is not None and parent is not None and parent != digest:
            problems.append(f"{name} was built from another gene file")
    draws_id = (lineage.get(PYXTAL_TRIALS_FILE) or {}).get("id")
    for name in (PRESCREEN_TRIALS_FILE, BASINHOP_TRIALS_FILE):
        parent = (lineage.get(name) or {}).get("parent")
        if parent is not None and draws_id is not None and parent != draws_id \
                and (Path(output_dir) / name).is_file():
            problems.append(f"{name} was built from earlier PyXtal draws")
    relax_parent = (lineage.get(RELAXATIONS_FILE) or {}).get("parent")
    sources = {
        record["id"] for name, record in lineage.items()
        if name in RELAX_SOURCE_LOGS.values()
    }
    if relax_parent is not None and relax_parent not in sources:
        problems.append(f"{RELAXATIONS_FILE} relaxed structures that have since been replaced")
    if problems:
        raise StaleOutputError(
            f"Outputs in {output_dir} do not belong together: {'; '.join(problems)}. "
            "Re-run the stages after the first stale one with --no-resume."
        )


def require_screen_reference(output_dir: Path, cache: Path, splits) -> None:
    """Refuse to score against a reference other than the one the screen used.

    The funnel's gene section -- ``gene_novelty_rate``, and the
    ``gene_known_became_novel`` crossings measured against it -- comes from
    ``screen.json``, while structure novelty is judged in ``score`` against
    ``--reference-cache``.  Scoring a run screened against one reference with
    another would publish a funnel whose two halves disagree about what is
    known.  A screen that records no reference predates the record
    (2026-09-15), when the default was ``lemat_bulk_ehull``.

    Raises:
        StaleOutputError: Unless the screen's recorded reference is *cache* over
            *splits*.
    """
    record = (_read_manifest(output_dir).get(LINEAGE_KEY) or {}).get(SCREEN_FILE) or {}
    expected = reference_identity(cache, splits)
    recorded = record.get("reference")
    if recorded == expected:
        return
    found = (
        "records no reference, so it predates 2026-09-15, when gene novelty was "
        "judged against lemat_bulk_ehull by default"
        if recorded is None
        else f"judged gene novelty against {recorded['cache']} "
             f"({','.join(recorded['splits'])})"
    )
    raise StaleOutputError(
        f"{Path(output_dir) / SCREEN_FILE} {found}, but structure novelty would be "
        f"judged against {expected['cache']} ({','.join(expected['splits'])}). "
        "Re-run the screen stage with the same --reference-cache and "
        "--reference-splits first (--stage screen, or --stages screen,score in "
        "wyformer-protocol-wandb); it leaves the draws and relaxations alone."
    )


def _failed_row(key, status: str, error: str) -> dict:
    """The log row for a trial the worker pool could not answer."""
    index, trial = key
    logger.warning("Gene %d trial %d: %s (%s)", index, trial, status, error)
    return {"index": index, "trial": trial, "status": status, "error": error}


def _progress_logger(verb: str, every: int, unit: str = "trial"):
    """A ``progress`` callback for :func:`run_supervised` logging rate and ETA."""
    started = time.time()

    def report(done: int, total: int) -> None:
        if done % every and done != total:
            return
        rate = (time.time() - started) / done
        logger.info(
            "%s %d/%d (%.1f s/%s, %.0f min left)",
            verb, done, total, rate, unit, rate * (total - done) / 60,
        )

    return report


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
    _quiet_cif_parser_warnings()


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
    the slot round again rather than running out of devices.  A pool rebuilt
    after a technical failure gets a fresh counter, and only the slots still in
    service.

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


def estimate_relaxation_memory(atoms) -> int:
    """Projected peak GPU memory in MB for relaxing a structure with *atoms*.

    Derived empirically on Tesla K20c (Kepler sm_35) with ORB-v3:
      - Baseline context + model weights: ~350 MB.
      - Graph neighbor tensors & autograd backward pass for forces + stress:
        scales with N atoms and edge coordination density.
      - Small/medium (N <= 48): <= 650 MB total.
      - Dense metal / large cells (N >= 120): up to ~2300-3000 MB total.
    """
    n = len(atoms) if atoms is not None else 0
    return max(600, int(400 + 15 * n))


class DeviceMemoryBudget:
    """Atomic counting memory budget semaphore for devices shared by multiple workers."""

    def __init__(self, device: str, total_mb: int, ctx=None):
        ctx = ctx or multiprocessing.get_context("spawn")
        self.device = device
        self.total_mb = int(total_mb)
        self._lock = ctx.Lock()
        self._cond = ctx.Condition(self._lock)
        self._available_mb = ctx.Value("i", self.total_mb)

    @property
    def available_mb(self) -> int:
        with self._lock:
            return self._available_mb.value

    @contextlib.contextmanager
    def reserve(self, required_mb: int, label: str = ""):
        claim = min(int(required_mb), self.total_mb)
        waited = False
        with self._cond:
            while self._available_mb.value < claim:
                if not waited:
                    logger.info(
                        "%sWaiting for %d MB memory budget on %s (available: %d MB / %d MB)",
                        f"[{label}] " if label else "",
                        claim, self.device, self._available_mb.value, self.total_mb,
                    )
                    waited = True
                self._cond.wait(timeout=2.0)
            self._available_mb.value -= claim
            if waited:
                logger.info(
                    "%sAcquired %d MB memory budget on %s (remaining: %d MB)",
                    f"[{label}] " if label else "",
                    claim, self.device, self._available_mb.value,
                )
        try:
            yield
        finally:
            with self._cond:
                self._available_mb.value += claim
                self._cond.notify_all()


def resolve_device_budgets(
    slots: list[str],
    ctx=None,
    device_budget_overrides: Optional[dict[str, int]] = None,
) -> dict[str, DeviceMemoryBudget]:
    """Create a memory budget semaphore for each physical device in *slots*."""
    ctx = ctx or multiprocessing.get_context("spawn")
    budgets = {}
    overrides = device_budget_overrides or {}

    for device in sorted(set(slots)):
        if device in overrides:
            total_mb = overrides[device]
        elif device.startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    _, _, idx_str = device.partition(":")
                    idx = int(idx_str) if idx_str else 0
                    total_bytes = torch.cuda.get_device_properties(idx).total_memory
                    total_mb = int(total_bytes * 0.85 / (1024 ** 2))
                else:
                    total_mb = 4000
            except Exception:
                total_mb = 4000
        else:
            total_mb = 16000
        budgets[device] = DeviceMemoryBudget(device, total_mb, ctx=ctx)
    return budgets



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


_CIF_PARSER_WARNING = "Issues encountered while parsing CIF"


def _quiet_cif_parser_warnings() -> None:
    """Print pymatgen's CIF coordinate-rounding warning at most once per process."""
    original = warnings.showwarning
    if getattr(original, "_cif_parser_filter", False):
        return

    shown = False

    def showwarning(message, category, filename, lineno, file=None, line=None):
        nonlocal shown
        text = str(message)
        if _CIF_PARSER_WARNING in text:
            if shown:
                return
            shown = True
        original(message, category, filename, lineno, file, line)

    showwarning._cif_parser_filter = True
    warnings.showwarning = showwarning


_quiet_cif_parser_warnings()


def _init_relax_worker(
    counter,
    slots: list[str],
    mlip: Optional[str],
    prerelax_mlip: Optional[str],
    debug: bool,
    budgets: Optional[dict[str, DeviceMemoryBudget]] = None,
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
        budgets: Per-device memory budgets shared by every worker on a
            card; see :class:`DeviceMemoryBudget`.  ``None`` runs trials
            without reserving memory.
    """
    global _WORKER_DEVICE, _WORKER_TORCH_DEVICE, _WORKER_CALCULATOR
    global _WORKER_PRERELAX_CALCULATOR, _WORKER_BUDGET
    for key, value in _SINGLE_THREAD_ENV_VARS.items():
        os.environ[key] = value
    _init_worker_logging(debug)
    _quiet_logm_roundoff()
    _quiet_cif_parser_warnings()

    _WORKER_DEVICE = claim_device(counter, slots)
    _WORKER_TORCH_DEVICE = _pin_visible_device(_WORKER_DEVICE)
    _WORKER_BUDGET = budgets.get(_WORKER_DEVICE) if budgets else None
    set_worker_device(_WORKER_DEVICE)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.set_num_threads(1)

    # Built once per process rather than per trial: loading an MLIP costs far
    # more than relaxing one structure.
    try:
        if mlip is not None:
            _WORKER_CALCULATOR = build_hull_calculator(mlip, device=_WORKER_TORCH_DEVICE)
        if prerelax_mlip is not None:
            from wyckoff_transformer.cryspr.mlips import build_prerelax_calculator

            _WORKER_PRERELAX_CALCULATOR = build_prerelax_calculator(
                prerelax_mlip, device=_WORKER_TORCH_DEVICE
            )
    except Exception as exc:
        # A broken card is reported through the worker's tasks, which say which
        # device it was.  Raised from here, it would break the whole pool and
        # say nothing about where.  Anything else -- a misspelt potential, a
        # missing checkpoint -- is a configuration error and still raises.
        if not is_technical_error(describe(exc)):
            raise
        mark_worker_faulty(describe(exc))
        return
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
def _reference_splits(args) -> tuple[str, ...]:
    return tuple(s.strip() for s in args.reference_splits.split(","))


def stage_screen(args) -> None:
    """Validity, uniqueness with counts, and gene novelty.  No potential.

    The reference gene novelty was judged against is recorded with the screen's
    lineage, so that ``score`` can refuse to put it in one funnel with structure
    novelty judged against another.  Re-running this stage on a relaxed run is
    safe: validity, uniqueness and the representatives depend on the genes
    alone, and nothing downstream is built from the screen's id.
    """
    genes = load_genes(args.input)
    splits = _reference_splits(args)
    reference = load_reference_fingerprints(
        args.reference_cache,
        splits,
        fingerprint_cache=getattr(args, "reference_fingerprint_cache", None)
        or default_fingerprint_cache(args.reference_cache, splits),
    )
    screen = screen_genes(genes, reference, GeneFingerprinter())
    write_screen(screen, args.output_dir / SCREEN_FILE)
    _write_lineage(args.output_dir, SCREEN_FILE, {
        "id": _new_lineage_id(), "parent": genes_digest(genes),
        "reference": reference_identity(args.reference_cache, splits),
        "reference_fingerprints": len(reference),
    })
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


def _claim_draws_log(args, genes: list[dict], resume: bool) -> str:
    """Check that the screen and the draws log belong to *genes*; see :func:`claim_lineage`."""
    digest = genes_digest(genes)
    screened = (_read_manifest(args.output_dir).get(LINEAGE_KEY) or {}).get(SCREEN_FILE)
    if screened is not None and screened.get("parent") not in (None, digest):
        raise StaleOutputError(
            f"{args.output_dir / SCREEN_FILE} was computed from another gene file "
            f"than {args.input}. Run --stage screen on this one first."
        )
    return claim_lineage(
        args.output_dir, PYXTAL_TRIALS_FILE, digest, resume,
        verify=_verify_draws_log(args.output_dir, PYXTAL_TRIALS_FILE, genes),
    )


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

    _claim_draws_log(args, genes, args.resume)
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

    n_ok = 0

    def record(key, result) -> None:
        nonlocal n_ok
        index, trial = key
        row, atoms = result
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

    try:
        pool_report = run_supervised(
            (
                ((index, trial), (
                    index,
                    trial,
                    genes[index],
                    str(_trial_dir(args.output_dir, index, trial)),
                    args.pyxtal_timeout,
                    args.pyxtal_tol_factor,
                ))
                for index, trial in tasks
            ),
            _generate_one,
            slots=["cpu"] * cores,
            initializer=_init_generate_worker,
            initargs=lambda counter, live: (args.debug,),
            on_result=record,
            on_failure=lambda key, status, error: record(
                key, (_failed_row(key, status, error), None)
            ),
            task_timeout=args.pyxtal_timeout,
            progress=_progress_logger("Drew", 100),
        )
    finally:
        log.close()

    frame = log.frame()
    pyxtal_ok = int((frame["status"] == "ok").sum())
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
        "pyxtal_ok": pyxtal_ok,
        "pyxtal_failed": int((frame["status"] == "failed").sum()),
        "pyxtal_timed_out": int((frame["status"] == "timeout").sum()),
        **pool_report.manifest("pyxtal"),
    })
    print(
        f"{n_ok} new draws, {pyxtal_ok}/{len(frame)} "
        f"trials with a structure -> {structures_path}"
    )
    require_complete(args, (PYXTAL_TRIALS_FILE,))


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
        INDEX_FILE_NAME,
        TemplateIndex,
        gene_query,
        load_template_structures,
        single_template,
    )

    genes = load_genes(args.input)
    screen = read_screen(args.output_dir / SCREEN_FILE)
    schedule = parse_trial_schedule(args.n_trials)
    todo = _todo_representatives(screen, args.limit)

    _claim_draws_log(args, genes, resume=True)
    log = RowLog(args.output_dir / PYXTAL_TRIALS_FILE, PYXTAL_COLUMNS, resume=True)
    todo = [index for index in todo if (index, TEMPLATE_TRIAL) not in log.done]
    if not todo:
        log.close()
        print("Every gene already has a template draw.")
        return

    # Built from --reference-cache and kept beside it, as --template-index says;
    # the index of one cache must not be filed under another.
    index_table = TemplateIndex.load(
        args.template_index or Path(args.reference_cache).parent / INDEX_FILE_NAME,
        cache=args.reference_cache,
        splits=_reference_splits(args),
    )
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

    claim_lineage(
        args.output_dir, PRESCREEN_TRIALS_FILE,
        lineage_id(args.output_dir, PYXTAL_TRIALS_FILE), args.resume,
        verify=_verify_against_draws(args.output_dir, PRESCREEN_TRIALS_FILE, draws),
    )
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

    def record(key, result) -> None:
        index, trial = key
        row, relaxed = result
        if relaxed is not None:
            relaxed.info = {"gene": int(index), "trial": int(trial)}
            ase_write(str(relaxed_path), relaxed, format="extxyz", append=True)
        log.write(row)

    try:
        pool_report = run_supervised(
            (
                ((index, trial), (
                    index,
                    trial,
                    atoms,
                    str(_trial_dir(args.output_dir, index, trial) / "prescreen"),
                    args.prescreen_fmax,
                    args.relax_timeout,
                    args.prescreen_release_symmetry,
                    args.prescreen_rattle,
                ))
                for index, trial, atoms in todo
            ),
            _prescreen_one,
            slots=slots,
            initializer=_init_relax_worker,
            # No scoring potential: this stage never computes a reported energy,
            # and loading ORB in every worker would cost more than the stage.
            initargs=lambda counter, live: (counter, live, None, prescreen_mlip, args.debug),
            on_result=record,
            on_failure=lambda key, status, error: record(
                key, (_failed_row(key, status, error), None)
            ),
            task_timeout=args.relax_timeout,
            progress=_progress_logger("Pre-relaxed", 50),
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
        **pool_report.manifest("prescreen"),
    })
    print(
        f"{selection['prescreen_selected']} of {selection['prescreen_candidates']} "
        f"pre-relaxed draws selected over {selection['prescreen_genes']} genes "
        f"({selection['prescreen_duplicates']} duplicates, "
        f"{selection['prescreen_rejected']} distinct but over budget) "
        f"-> {args.output_dir / PRESCREEN_FILE}"
    )
    require_complete(args, (PRESCREEN_TRIALS_FILE,))


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

    claim_lineage(
        args.output_dir, BASINHOP_TRIALS_FILE,
        lineage_id(args.output_dir, PYXTAL_TRIALS_FILE), args.resume,
        verify=_verify_against_draws(args.output_dir, BASINHOP_TRIALS_FILE, draws),
    )
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

    def record(walk, result) -> None:
        index, trial = walk
        row, minima = result
        for key, atoms, energy_per_atom in minima:
            atoms.info = {
                "gene": int(index), "trial": int(key),
                "walk_trial": int(trial),
                "energy_per_atom": float(energy_per_atom),
            }
            ase_write(str(minima_path), atoms, format="extxyz", append=True)
        log.write(row)

    try:
        pool_report = run_supervised(
            (
                ((index, trial), (
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
                ))
                for index, trial, atoms in todo
            ),
            _basinhop_one,
            slots=slots,
            initializer=_init_relax_worker,
            initargs=lambda counter, live: (counter, live, None, mlip, args.debug),
            on_result=record,
            on_failure=lambda key, status, error: record(
                key, (_failed_row(key, status, error), [])
            ),
            task_timeout=args.relax_timeout,
            progress=_progress_logger("Walked", 25, unit="walk"),
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
        **pool_report.manifest("basinhop"),
    })
    print(
        f"{selection['basinhop_selected']} of {selection['basinhop_candidates']} "
        f"minima selected over {selection['basinhop_genes']} genes "
        f"({selection['basinhop_duplicates']} duplicates, "
        f"{selection['basinhop_rejected']} distinct but over budget) "
        f"-> {args.output_dir / BASINHOP_FILE}"
    )
    require_complete(args, (BASINHOP_TRIALS_FILE,))


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
        KEPT_FIXED_CIF_SUFFIX,
        _trial_seed,
        relax_trial,
    )

    started = time.time()
    # relax_trial names its CIFs after the draw's formula.
    formula_in = atoms.get_chemical_formula(mode="metal")
    row = {
        "index": index, "trial": trial, "status": "failed",
        "device": _WORKER_DEVICE, "n_atoms": len(atoms),
        "formula": formula_in,
        "energy": None, "energy_per_atom": None, "cif": None, "error": None,
        "status_fixed": "failed", "energy_fixed": None,
        "energy_per_atom_fixed": None, "cif_fixed": None,
    }
    req_mb = estimate_relaxation_memory(atoms)
    budget = _WORKER_BUDGET
    cm = (
        budget.reserve(req_mb, label=f"gene {index} trial {trial}")
        if budget is not None
        else contextlib.nullcontext()
    )
    try:
        with cm:
            with time_limit(timeout):
                relaxed, energy, (relaxed_fixed, energy_fixed) = relax_trial(
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
        row["status"] = row["status_fixed"] = "timeout"
        row["error"] = str(exc)
        logger.warning("Gene %d trial %d: relaxation timed out (%s)", index, trial, exc)
    except Exception as exc:  # noqa: BLE001 - one bad trial must not stop the stage
        row["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Gene %d trial %d: relaxation failed (%s)", index, trial, exc)
    else:
        if relaxed_fixed is None:
            row["status_fixed"] = "clash"
        else:
            row["status_fixed"] = "ok"
            row["energy_fixed"] = energy_fixed
            row["energy_per_atom_fixed"] = energy_fixed / len(relaxed_fixed)
            row["cif_fixed"] = str(Path(trial_dir) / f"{formula_in}{KEPT_FIXED_CIF_SUFFIX}")
        if relaxed is None:
            row["status"] = "clash"
            row["error"] = "relaxed structure has atomic clashes"
        else:
            row["status"] = "ok"
            row["energy"] = energy
            row["energy_per_atom"] = energy / len(relaxed)
            row["n_atoms"] = len(relaxed)
            row["formula"] = relaxed.get_chemical_formula(mode="metal")
            row["cif"] = str(Path(trial_dir) / f"{formula_in}{KEPT_CIF_SUFFIX}")
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


def _warn_about_idle_slots(relaxed: pd.DataFrame, slots: list[str]) -> None:
    """Say so when a card that was asked for took no trials.

    A worker that never finishes its initialiser costs its whole share of the
    throughput and says nothing: the run simply takes two and a half times as
    long as it should, which is easy to blame on the cohort.  The per-trial
    device column is the only place it shows.
    """
    used = set(relaxed["device"].dropna().unique())
    idle = [device for device in sorted(set(slots)) if device not in used]
    if idle:
        logger.warning(
            "%s took no trials of %d: its worker(s) never became ready, so this "
            "stage ran on %s alone",
            ", ".join(idle), len(relaxed), ", ".join(sorted(used)) or "nothing",
        )


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

    # Parented on the log of the stage that wrote the draws, not on the extxyz:
    # a draw re-made by `generate --no-resume` keeps its (gene, trial) key, and
    # PyXtal is not seeded, so only the new log id says the structure changed.
    claim_lineage(
        args.output_dir, RELAXATIONS_FILE,
        lineage_id(args.output_dir, RELAX_SOURCE_LOGS[source]), args.resume,
        verify=_verify_against_draws(args.output_dir, RELAXATIONS_FILE, draws),
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

    budgets: dict = {}

    def relax_initargs(counter, live):
        # Fresh budgets for every pool: a worker killed while holding a
        # reservation would otherwise leak it into the next one.
        budgets.clear()
        budgets.update(resolve_device_budgets(live))
        return (counter, live, args.mlip, prerelax_mlip, args.debug, dict(budgets))

    try:
        pool_report = run_supervised(
            (
                ((index, trial), (
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
                ))
                for index, trial, atoms in todo
            ),
            _relax_one,
            slots=slots,
            initializer=_init_relax_worker,
            initargs=relax_initargs,
            on_result=lambda key, row: log.write(row),
            on_failure=lambda key, status, error: log.write(
                _failed_row(key, status, error)
            ),
            task_timeout=args.relax_timeout,
            progress=_progress_logger("Relaxed", 25),
        )
    finally:
        log.close()

    frame = aggregate_structures(args.output_dir)
    relaxed = latest_rows(log.frame())
    _warn_about_idle_slots(relaxed, slots)
    trials_relaxed = int((relaxed["status"] == "ok").sum())
    trials_failed = int((relaxed["status"] != "ok").sum())
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
        "device_memory_budgets": {dev: b.total_mb for dev, b in budgets.items()},
        # The slots themselves, not just the distinct cards: a device named
        # twice has two workers on it, which is how cards of different size
        # get different shares.
        "device_slots": slots,
        "trials_relaxed": trials_relaxed,
        "trials_failed": trials_failed,
        # What the trials actually ran on, not just what was asked for. A run
        # once recorded three GPUs while one of them never took a single trial
        # and a worker fell back to the CPU; only the per-trial rows said so.
        "trials_by_device": {
            str(device): int(count)
            for device, count in relaxed["device"].value_counts().items()
        },
        "genes_with_structure": int(frame["has_structure"].sum()),
        "genes_attempted": int(len(frame)),
        **pool_report.manifest("relax"),
    })
    print(
        f"{int(frame['has_structure'].sum())}/{len(frame)} genes produced a "
        f"structure -> {args.output_dir / STRUCTURES_FILE}"
    )
    require_complete(args, (RELAXATIONS_FILE,))


def _backfill_fixed_symmetry_from_cryspr(relaxations: pd.DataFrame) -> pd.DataFrame:
    """Recover the fixed-symmetry CIF and energy from ``cryspr/`` for older runs.

    A run relaxed before the fixed-symmetry readout was recorded still has every
    stage's CIF and optimiser log in its trial directories: the
    ``*_2_sym_cell+pos.cif`` and the last energy of the
    ``*_sym_cell+positions_relax.log`` are that readout.
    """
    for column in ("status_fixed", "energy_fixed", "energy_per_atom_fixed", "cif_fixed"):
        if column not in relaxations.columns:
            relaxations[column] = None
    relaxations["status_fixed"] = relaxations["status_fixed"].astype(object)
    relaxations["cif_fixed"] = relaxations["cif_fixed"].astype(object)
    for idx, row in relaxations[relaxations["status"] == "ok"].iterrows():
        cif_str = str(row.get("cif", ""))
        if not cif_str or cif_str == "nan":
            continue
        cif_path = Path(cif_str)
        parent = cif_path.parent
        if not parent.is_dir():
            continue
        prefix = cif_path.name.replace("_kept.cif", "")
        sym_cifs = (
            list(parent.glob(f"{prefix}_*2_sym_cell+pos.cif"))
            or list(parent.glob(f"*_{prefix}_2_sym_cell+pos.cif"))
            or list(parent.glob("*_2_sym_cell+pos.cif"))
        )
        if not sym_cifs:
            continue
        sym_logs = (
            list(parent.glob(f"{prefix}*_sym_cell+positions_relax.log"))
            or list(parent.glob("*_sym_cell+positions_relax.log"))
        )
        last_energy = None
        if sym_logs and sym_logs[0].is_file():
            with open(sym_logs[0], encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) >= 4 and parts[0] == "BFGS:":
                        try:
                            last_energy = float(parts[3])
                        except ValueError:
                            pass
        if last_energy is None:
            continue
        relaxations.at[idx, "status_fixed"] = "ok"
        relaxations.at[idx, "energy_fixed"] = last_energy
        relaxations.at[idx, "cif_fixed"] = str(sym_cifs[0])
        n_atoms = row.get("n_atoms")
        if pd.notna(n_atoms) and int(n_atoms) > 0:
            relaxations.at[idx, "energy_per_atom_fixed"] = last_energy / int(n_atoms)
    return relaxations


#: Columns of ``structures.csv`` and ``structures_fixed_symmetry.csv`` before scoring.
STRUCTURE_COLUMNS = (
    "index", "dof_positional", "n_trials", "n_drawn", "n_relaxed",
    "pyxtal_seconds", "relax_seconds", "has_structure", "formula",
    "energy", "energy_per_atom", "n_atoms", "best_trial", "device", "error",
)


def aggregate_structures(output_dir: Path) -> pd.DataFrame:
    """Reduce the per-trial logs to one row per gene, for both readouts.

    The kept structure of a gene is the lowest-energy trial that produced one,
    written to ``structures.csv`` and ``cifs/``.  The fixed-symmetry structure
    is a separate selection -- the lowest *fixed-symmetry* energy, which need
    not be the same trial -- written to ``structures_fixed_symmetry.csv`` and
    ``cifs_fixed_symmetry/``.  Genes that produced none keep a row, with the
    reason in ``error``: a gene that nothing could be built for and one whose
    relaxations all crashed are different failures, and the funnel's
    ``has_structure`` alone does not say which happened.

    Returns:
        The frame written to ``structures.csv``, indexed by gene.
    """
    output_dir = Path(output_dir)
    draws = latest_rows(read_rows(output_dir / PYXTAL_TRIALS_FILE, PYXTAL_COLUMNS))
    relaxations = read_rows(output_dir / RELAXATIONS_FILE, RELAXATION_COLUMNS)
    if (relaxations["status"] == "ok").any() and not (
        "status_fixed" in relaxations.columns
        and (relaxations["status_fixed"] == "ok").any()
    ):
        relaxations = _backfill_fixed_symmetry_from_cryspr(relaxations)
        if (relaxations["status_fixed"] == "ok").any():
            relaxations.to_csv(output_dir / RELAXATIONS_FILE, index=False)
    relaxations = latest_rows(relaxations)
    # Only present in a wide-then-narrow run.  Its per-gene counts and seconds
    # are what make the two arms comparable on cost: without them a run that
    # drew ten times as many starts looks identical to one that drew the usual
    # number, since `relax` sees the same number of trials either way.
    prescreened = read_rows(output_dir / PRESCREEN_TRIALS_FILE, PRESCREEN_COLUMNS)
    by_gene_prescreen = {
        index: group for index, group in prescreened.groupby("index")
    } if len(prescreened) else {}
    readouts = (
        # (output, CIF dir, status, energy, energy per atom, CIF column)
        (STRUCTURES_FILE, CIF_DIR, "status", "energy", "energy_per_atom", "cif"),
        (STRUCTURES_FIXED_FILE, CIF_FIXED_DIR, "status_fixed", "energy_fixed",
         "energy_per_atom_fixed", "cif_fixed"),
    )
    for _, directory, *_ in readouts:
        (output_dir / directory).mkdir(parents=True, exist_ok=True)

    by_gene = {index: group for index, group in relaxations.groupby("index")}
    rows = {name: [] for name, *_ in readouts}
    for index, group in draws.groupby("index"):
        relaxed = by_gene.get(index, relaxations.iloc[:0])
        for name, directory, status, energy, energy_per_atom, cif in readouts:
            succeeded = relaxed[
                (relaxed[status] == "ok") & relaxed[energy].notna()
            ] if status in relaxed.columns else relaxed.iloc[:0]
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
            if by_gene_prescreen:
                pre = by_gene_prescreen.get(index, prescreened.iloc[:0])
                row["n_prescreened"] = int((pre["status"] == "ok").sum())
                row["prescreen_seconds"] = round(float(pre["seconds"].fillna(0).sum()), 2)
            if len(succeeded):
                best = succeeded.loc[pd.to_numeric(succeeded[energy]).idxmin()]
                row.update(
                    has_structure=True,
                    formula=best["formula"],
                    energy=float(best[energy]),
                    energy_per_atom=float(best[energy_per_atom]),
                    n_atoms=int(best["n_atoms"]),
                    best_trial=int(best["trial"]),
                    device=best["device"],
                )
                source = Path(str(best[cif]))
                if source.is_file():
                    (output_dir / directory / f"{index}.cif").write_text(
                        source.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                else:
                    row["error"] = f"kept CIF missing at {source}"
            else:
                row["error"] = _failure_reason(group, relaxed)
            rows[name].append(row)

    frames = {}
    for name, *_ in readouts:
        frame = (
            pd.DataFrame(rows[name]).set_index("index").sort_index()
            if rows[name]
            else pd.DataFrame(columns=list(STRUCTURE_COLUMNS)).set_index("index")
        )
        frame.to_csv(output_dir / name)
        frames[name] = frame
    return frames[STRUCTURES_FILE]


def _failure_reason(draws: pd.DataFrame, relaxations: pd.DataFrame) -> str:
    """Why this gene has no structure, in the words of the stage that failed."""
    if not len(relaxations):
        statuses = sorted(set(draws["status"]) - {"ok"})
        if "ok" in set(draws["status"]):
            return "generated but never relaxed"
        return f"no PyXtal structure ({', '.join(statuses) or 'unknown'})"
    latest = (
        relaxations.drop_duplicates(subset=["trial"], keep="last")
        if "trial" in relaxations.columns
        else relaxations
    )
    errors = [str(e) for e in latest["error"] if isinstance(e, str) and e]
    return f"all {len(latest)} relaxation(s) failed: {errors[0] if errors else 'unknown'}"


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

    Two readouts are scored the same way: the kept structures (``cifs/``,
    ``structures.csv``, the funnel's ``free`` section) and the fixed-symmetry
    ones (``cifs_fixed_symmetry/``, ``structures_fixed_symmetry.csv``,
    ``fixed_symmetry``).  Each is a different structure, so every verdict --
    validity, uniqueness, novelty, the hull -- is recomputed on it rather than
    inherited.
    """
    from pymatgen.core import Structure

    from wyckoff_transformer.evaluation.hull_energy import HullEnergyCalculator
    from wyckoff_transformer.evaluation.novelty import (
        NoveltyFilter,
        filter_by_unique_structure,
    )
    from wyckoff_transformer.evaluation.structure_novelty import build_novelty_reference
    from wyckoff_transformer.evaluation.structure_validity import is_valid

    # Before anything is loaded: a funnel with holes in it would be published
    # as the model's numbers.
    require_complete(args, TRIAL_LOGS)
    # And one built from a single cohort: a structure scored against another
    # gene's fingerprint makes every novelty verdict meaningless.
    require_consistent_lineage(
        args.output_dir,
        load_genes(args.input) if Path(args.input).is_file() else None,
    )
    # And one reference: gene novelty comes from the screen, structure novelty
    # from here.
    splits = _reference_splits(args)
    require_screen_reference(args.output_dir, args.reference_cache, splits)
    phase = _PhaseTimer()
    screen = read_screen(args.output_dir / SCREEN_FILE)
    free_file = args.output_dir / STRUCTURES_FILE
    fixed_file = args.output_dir / STRUCTURES_FIXED_FILE
    # A run relaxed before the fixed-symmetry readout existed has no
    # structures_fixed_symmetry.csv; aggregating again backfills it from the
    # trial directories, where they are still on disk.
    if not fixed_file.is_file() and (args.output_dir / PYXTAL_TRIALS_FILE).is_file() \
            and (args.output_dir / RELAXATIONS_FILE).is_file():
        aggregate_structures(args.output_dir)
    frame = pd.read_csv(free_file, index_col="index")
    frame_fixed = (
        pd.read_csv(fixed_file, index_col="index") if fixed_file.is_file() else None
    )
    if frame["has_structure"].sum() == 0:
        relax_path = args.output_dir / RELAXATIONS_FILE
        if relax_path.is_file():
            relax_df = read_rows(relax_path, RELAXATION_COLUMNS)
            if len(relax_df) > 0 and (relax_df["status"] == "ok").sum() == 0:
                worker_errors = relax_df["error"].fillna("").str.startswith(
                    ("BrokenProcessPool", "worker failed")
                )
                if worker_errors.all():
                    raise RuntimeError(
                        f"Cannot score: all {len(relax_df)} relaxation trial(s) failed with worker errors"
                    )

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

    def read_variant(target: pd.DataFrame, directory: Path) -> dict:
        """Structures, validity, relaxed fingerprints and hull energies for one readout."""
        out = {"validity": {}, "structures": {}, "relaxed": {}, "hull": {}}
        if not directory.is_dir():
            return out
        for index in target.index:
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
            energy = target.at[index, "energy"]
            if pd.notna(energy):
                try:
                    # The energy came from the same potential that defines this
                    # hull, which is why --mlip is restricted to published hulls.
                    out["hull"][index] = hull.energy_above_hull(
                        float(energy), structure.composition
                    )
                except ValueError as exc:
                    logger.warning("Gene %d: e_above_hull failed (%s)", index, exc)
        return out

    def attach(target: pd.DataFrame, variant: dict) -> pd.DataFrame:
        """Write a readout's per-gene verdicts onto *target*, and return its scored rows."""
        novel_genes = set(screen.novel)
        target["valid_structure"] = pd.Series(variant["validity"], dtype=object)
        target["e_above_hull"] = pd.Series(variant["hull"], dtype=float)
        target["gene_novel"] = pd.Series(
            {index: index in novel_genes for index in target.index}
        )
        target["relaxed_fingerprint_resolved"] = pd.Series(
            {index: index in variant["relaxed"] for index in target.index}
        )
        target["relaxed_fingerprint_changed"] = pd.Series(
            {
                index: variant["relaxed"][index] != gene_fingerprints.get(index)
                for index in variant["relaxed"]
            },
            dtype=object,
        )
        # Only structures that got this far can be unique or novel, and
        # comparing the ones that did not would just cost matcher calls.
        scored = pd.DataFrame(
            {
                "fingerprint": pd.Series(gene_fingerprints),
                "structure": pd.Series(variant["structures"], dtype=object),
            }
        ).dropna()
        scored = scored.loc[
            [i for i in scored.index if bool(variant["validity"].get(i, False))]
        ]
        scored["relaxed_fingerprint"] = pd.Series(variant["relaxed"], dtype=object).reindex(
            scored.index
        )
        return scored

    readouts = [("free", frame, args.output_dir / CIF_DIR, free_file)]
    if frame_fixed is not None:
        readouts.append(
            ("fixed_symmetry", frame_fixed, args.output_dir / CIF_FIXED_DIR, fixed_file)
        )
    scored = {}
    for label, target, directory, _ in readouts:
        scored[label] = attach(target, read_variant(target, directory))
    phase.done("read CIFs, validity, fingerprints and e_above_hull")

    for label, target, _, _ in readouts:
        unique_index = (
            set(filter_by_unique_structure(scored[label]).index)
            if len(scored[label]) else set()
        )
        target["unique_structure"] = pd.Series(
            {index: index in unique_index for index in scored[label].index}, dtype=object
        )
    phase.done("uniqueness (StructureMatcher)")

    # The matcher needs a candidate for either fingerprint, and one reference
    # serves both readouts: built over the union of their fingerprint sets, so
    # the streaming pass over the ~1 GB CIF export happens once rather than
    # twice.  That pass is the score stage's dominant cost.
    fingerprint_sets = []
    for label in scored:
        fingerprint_sets += [
            scored[label]["fingerprint"], scored[label]["relaxed_fingerprint"].dropna()
        ]
    reference = build_novelty_reference(
        pd.concat(fingerprint_sets),
        cache=args.reference_cache,
        splits=splits,
        lemat_cif_csv=args.lemat_cif_csv,
    )
    phase.done("build the novelty reference")
    # Which reference a number was judged against is not recoverable from the
    # number, and it has changed once already.
    novelty_reference = {
        "reference_cache": str(args.reference_cache),
        "reference_splits": ",".join(splits),
        "lemat_cif_csv": str(args.lemat_cif_csv),
        "novelty_reference": {
            **reference_identity(args.reference_cache, splits),
            "colliding_fingerprints": reference["fingerprint"].nunique()
            if len(reference) else 0,
            "candidate_entries": len(reference),
        },
    }
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

    for label, target, _, path in readouts:
        rows = scored[label]
        target["novel_by_sampled_gene"] = pd.Series(
            {index: novelty_filter.is_novel(row) for index, row in rows.iterrows()},
            dtype=object,
        )
        target["novel_structure"] = pd.Series(
            {index: _is_novel(row) for index, row in rows.iterrows()}, dtype=object
        )
        target.drop(columns=["structure"], errors="ignore").to_csv(path)
    phase.done("novelty (StructureMatcher)")

    _update_manifest(
        args.output_dir / MANIFEST_FILE,
        {"hull": hull.provenance, "novelty": "sampled+relaxed fingerprint",
         **novelty_reference},
    )
    report = funnel(screen, frame, frame_fixed)
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


#: Public alias. ``wyformer-protocol-wandb`` records the sampling temperature it
#: generated the cohort at, which no stage of this module can know.
update_manifest = _update_manifest


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
        "--relax-timeout", type=float, default=300.0,
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
            "whose error names a killed worker or a failed GPU is re-run "
            "regardless: that is an infrastructure failure, and skipping it "
            "leaves a permanent hole in the cohort that reads as a collapsed "
            "success rate."
        ),
    )
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help=(
            "Go on, and score, although some trials are still unanswered because "
            "a GPU failed or a worker was killed. Off by default: each stage "
            "retries such trials on a fresh pool and retires a device that keeps "
            "failing, and whatever is left over is refused rather than counted "
            "as a gene without a structure. --resume retries exactly those trials."
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
        "--reference-fingerprint-cache", type=Path, default=None,
        help=(
            "Where to persist the reference fingerprint set. Computing it from "
            "5M rows takes minutes; every variant evaluation reuses the same set, "
            "so it is written once and loaded thereafter -- whatever the file "
            "holds, so it must have been built from --reference-cache and "
            "--reference-splits. Defaults to gene_fingerprints.pkl.gz beside "
            "--reference-cache (with the splits in the name when not all of them)."
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
