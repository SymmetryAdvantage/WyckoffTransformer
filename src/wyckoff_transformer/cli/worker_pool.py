"""A process pool that tells a broken worker from a failed trial.

The protocol stages hand one task per trial to a pool of spawned workers, each
holding a potential on one device.  :class:`~concurrent.futures.ProcessPoolExecutor`
alone treats every outcome alike, and three kinds of *technical* failure then
turn into holes in the cohort that read as properties of the model:

* **A poisoned device.**  A CUDA error such as ``unspecified launch failure`` is
  sticky: the process's context is unusable afterwards, so every later trial on
  that worker fails in milliseconds.  A fast-failing worker also drains the
  queue fastest.  One card on iapetus did this to 750 of 2324 trials, and the
  run's MSUN dropped by a third with nothing in its funnel saying why.
* **A dead worker.**  A segfault or the OOM killer breaks the whole executor,
  and every unfinished future raises ``BrokenProcessPool``.
* **A hung worker.**  A trial stuck inside C code is out of reach of the
  in-task ``SIGALRM`` limit, and holds its slot forever.

:func:`run_supervised` handles all three.  Tasks are submitted no faster than
there are workers, so the tasks in flight at any moment are exactly the ones
running.  A worker that reports a technical error takes no further work: the
round stops submitting, lets the other workers finish what they are running,
and starts a fresh pool, in which the faulted trial is retried.  A crash that
names no culprit makes every trial it took down a suspect, and suspects then
run one at a time, so that a trial is charged an attempt only for a crash that
can be pinned on it.  A device that
faults in several consecutive rounds, with no success on it in between, is
dropped from the pool.  A trial the stage could not answer this way is handed
to the stage with an error that :data:`RETRYABLE_ERRORS` matches, so that
``--resume`` re-runs it rather than counting it as done, and the scoring stage
can refuse to publish a funnel with holes in it.
"""
from __future__ import annotations

import contextlib
import logging
import multiprocessing
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

#: Error text that says the device failed, not the trial.
#:
#: Matched against ``"<ExceptionType>: <message>"``, which is what the stages
#: write to their logs, so that a log written before this module existed is
#: classified the same way as a live exception.  Out-of-memory is here although
#: it does not poison the context: on a shared card it says how busy the card
#: was, not what the trial was, and a fresh worker later usually succeeds.
TECHNICAL_ERRORS = (
    "AcceleratorError",
    "CUDA error",
    "CUDA out of memory",
    "OutOfMemoryError",
    "CUBLAS_STATUS_",
    "CUDNN_STATUS_",
    "cuDNN error",
    "device-side assert",
)

#: Prefix of the error given to a task the supervisor never got to run.
NOT_RUN_ERROR = "NotRun"

#: Prefix of the error given to a task whose worker outlived the hang deadline.
WORKER_HUNG_ERROR = "WorkerHung"

#: Errors that say a trial was never answered, so ``--resume`` must re-run it.
#:
#: A trial whose relaxation diverged has been answered and re-running it would
#: answer the same way.  A trial whose worker was killed under it, or whose
#: device failed, has not been answered at all, and skipping it silently turns
#: an infrastructure failure into a permanent hole in the cohort.
#: :data:`WORKER_HUNG_ERROR` is deliberately absent: a trial that hung a fresh
#: worker every time it ran is recorded as a timeout, as the in-task limit
#: would have recorded it.
RETRYABLE_ERRORS = (
    "BrokenProcessPool",
    "A process in the process pool",
    "worker failed",
    NOT_RUN_ERROR,
) + TECHNICAL_ERRORS

#: Attempts a trial gets before its technical failure is written down.
DEFAULT_MAX_ATTEMPTS = 3

#: Consecutive faulted rounds, with no success in between, that retire a device.
DEFAULT_MAX_DEVICE_FAULTS = 3

#: Consecutive rounds that answer no trial at all before the supervisor gives up.
DEFAULT_MAX_IDLE_ROUNDS = 3

#: Seconds past twice a task's own time limit before its worker counts as hung.
#: Twice, because a submitted task can wait for one other task to finish before
#: a worker takes it -- a worker that never finished initialising takes none --
#: and wide on purpose, to cover a first task that also waits for the worker to
#: load its potential.
DEFAULT_HANG_GRACE = 600.0

#: Seconds a pool's workers get to exit on their own before they are terminated.
SHUTDOWN_GRACE = 120.0


def is_technical_error(text: Any) -> bool:
    """Whether an error says the device failed rather than the trial."""
    return isinstance(text, str) and any(marker in text for marker in TECHNICAL_ERRORS)


def is_retryable_error(text: Any) -> bool:
    """Whether an error says the trial was never answered."""
    return isinstance(text, str) and any(marker in text for marker in RETRYABLE_ERRORS)


def describe(exc: BaseException) -> str:
    """An exception as the stage logs write it."""
    return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Worker side
# --------------------------------------------------------------------------- #
#: The device this process relaxes on, for reporting a fault; set by the
#: stage's initialiser through :func:`set_worker_device`.
_DEVICE: Optional[str] = None

#: The technical error that broke this process, once one has.
_FAULT: Optional[str] = None


def set_worker_device(device: str) -> None:
    global _DEVICE
    _DEVICE = device


def mark_worker_faulty(error: str) -> None:
    """Refuse every later task in this process, reporting *error* instead.

    For an initialiser whose potential could not be built for a technical
    reason: raising there would break the whole pool and hide which device
    failed.
    """
    global _FAULT
    _FAULT = error
    logger.error("Worker on %s is out of service: %s", _DEVICE, error)


@dataclass(frozen=True)
class WorkerFault:
    """Returned in place of a task's result when the worker failed, not the trial.

    Attributes:
        device: The device of the worker that failed.
        error: The technical error, as the stage logs would write it.
        attempted: ``False`` when the worker was already broken and did not
            run the task, so the task itself is not to blame.
    """

    device: str
    error: str
    attempted: bool


def _row_of(result: Any) -> Optional[dict]:
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple) and result and isinstance(result[0], dict):
        return result[0]
    return None


def run_task(fn: Callable, task_args: tuple) -> Any:
    """Run one task in a worker, or report that this worker cannot.

    A task function records its own failures in the row it returns, so a
    technical error arrives either as that row's ``error`` or, if it escaped the
    task, as an exception.  Both put the worker out of service.
    """
    device = _DEVICE or "cpu"
    if _FAULT is not None:
        return WorkerFault(device, _FAULT, attempted=False)
    try:
        result = fn(*task_args)
    except Exception as exc:
        error = describe(exc)
        if not is_technical_error(error):
            raise
    else:
        row = _row_of(result)
        error = row.get("error") if row is not None else None
        if not is_technical_error(error):
            return result
    mark_worker_faulty(error)
    return WorkerFault(device, error, attempted=True)


# --------------------------------------------------------------------------- #
# Parent side
# --------------------------------------------------------------------------- #
@dataclass
class PoolReport:
    """What the supervisor had to do, for the stage's manifest."""

    rounds: int = 0
    worker_faults: int = 0
    pool_breaks: int = 0
    hung_workers: int = 0
    retried: int = 0
    unanswered: int = 0
    retired_devices: list[str] = field(default_factory=list)

    def manifest(self, prefix: str) -> dict:
        return {
            f"{prefix}_pool_rounds": self.rounds,
            f"{prefix}_worker_faults": self.worker_faults,
            f"{prefix}_pool_breaks": self.pool_breaks,
            f"{prefix}_hung_workers": self.hung_workers,
            f"{prefix}_trials_retried": self.retried,
            f"{prefix}_trials_unanswered": self.unanswered,
            f"{prefix}_retired_devices": sorted(self.retired_devices),
        }


def shutdown_pool(pool: ProcessPoolExecutor, grace: float = SHUTDOWN_GRACE) -> None:
    """Stop the workers without waiting on one that will never stop.

    By the time this runs every future has resolved and every row is on disk,
    so a worker still alive has nothing left to do -- but
    ``ProcessPoolExecutor.__exit__`` joins them unconditionally, and a worker
    that wedged inside its initialiser never returns.  That has happened on
    iapetus: a worker whose CUDA context never finished coming up spun at 100%
    CPU for the whole run, took no trial, and then deadlocked the stage *after*
    all 2369 relaxations were complete -- an eight-hour run with nothing to show
    for it, because the score stage never started.

    So: ask nicely, wait *grace* seconds, then terminate and move on.
    """
    # `_processes` is None until the pool spawns its first worker, and
    # shutdown() clears it, so it is read first.
    processes = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    deadline = time.monotonic() + grace
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
        if not process.is_alive():
            continue
        logger.warning("Worker %s did not exit; terminating it", process.pid)
        process.terminate()
        process.join(10)
        if process.is_alive():
            logger.warning("Worker %s ignored SIGTERM; killing it", process.pid)
            process.kill()
            process.join(10)


@contextlib.contextmanager
def _executor(max_workers: int, ctx, initializer: Callable, initargs: tuple):
    """A process pool that :func:`shutdown_pool` stops, however it ends."""
    pool = ProcessPoolExecutor(
        max_workers=max_workers, mp_context=ctx,
        initializer=initializer, initargs=initargs,
    )
    try:
        yield pool
    finally:
        shutdown_pool(pool)


def _kill_workers(pool: ProcessPoolExecutor) -> None:
    """Kill every process of *pool*, which breaks it and releases its futures.

    ``_processes`` is private, but the executor offers no way to reclaim a
    worker stuck in C code, and without this a hung trial holds the stage
    until someone notices.
    """
    for process in list((getattr(pool, "_processes", None) or {}).values()):
        try:
            process.kill()
        except Exception:  # noqa: BLE001 - it may have exited already
            pass


def run_supervised(
    tasks: Iterable[tuple[Hashable, tuple]],
    fn: Callable,
    *,
    slots: Sequence[str],
    initializer: Callable,
    initargs: Callable[[Any, list[str]], tuple],
    on_result: Callable[[Hashable, Any], None],
    on_failure: Callable[[Hashable, str, str], None],
    task_timeout: Optional[float] = None,
    hang_grace: float = DEFAULT_HANG_GRACE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_device_faults: int = DEFAULT_MAX_DEVICE_FAULTS,
    max_idle_rounds: int = DEFAULT_MAX_IDLE_ROUNDS,
    progress: Optional[Callable[[int, int], None]] = None,
) -> PoolReport:
    """Run *fn* on every task in a pool that survives technical worker failures.

    Args:
        tasks: ``(key, task_args)`` pairs; *fn* is called as ``fn(*task_args)``
            in a worker, and every key is reported back exactly once.
        fn: A module-level task function returning a row dict, or a tuple
            whose first element is one, with any failure in the row's ``error``.
        slots: One device string per worker, as from ``resolve_devices``.
        initializer: The stage's worker initialiser.
        initargs: Builds the initialiser's arguments from a fresh shared slot
            counter and the slots still in service, once per pool.
        on_result: Called in the parent with each answered task's result.
        on_failure: Called with ``(key, status, error)`` for a task the pool
            could not answer.  The error either names a crash in the task
            itself, as before, or matches :data:`RETRYABLE_ERRORS`.
        task_timeout: The task's own time limit.  A task still unanswered
            *hang_grace* seconds past twice this counts as a hung worker; see
            :data:`DEFAULT_HANG_GRACE`.  ``None`` or non-positive disables the
            check.
        hang_grace: See *task_timeout*.
        max_attempts: Attempts per task before its technical failure is final.
        max_device_faults: Consecutive faulted rounds that retire a device.
        max_idle_rounds: Consecutive rounds answering nothing before giving up.
        progress: Called with ``(answered, total)`` after each answer.

    Returns:
        What happened, for the manifest.
    """
    pending = deque(tasks)
    total = len(pending)
    report = PoolReport()
    if not pending:
        return report

    ctx = multiprocessing.get_context("spawn")
    attempts: Counter = Counter()
    strikes: Counter = Counter()
    # Tasks that were running when a pool broke without saying which one broke
    # it.  At most one of them runs at a time, so the next break is pinned on
    # the right one: charging every bystander instead lets a single segfaulting
    # trial use up the attempts of all the trials that were retried beside it.
    suspects: set = set()
    suspect_queue: deque = deque()
    retired: set[str] = set()
    answered = 0
    idle_rounds = 0
    last_error = "no worker could be started"
    deadline_after = (
        2 * task_timeout + hang_grace if task_timeout and task_timeout > 0 else None
    )

    def retry_or_fail(key, task_args, error: str, charge: bool, status: str = "failed"):
        nonlocal answered, last_error
        last_error = error
        if charge:
            attempts[key] += 1
        if attempts[key] >= max_attempts:
            logger.error("%s: giving up after %d attempts: %s", key, attempts[key], error)
            on_failure(key, status, error)
            report.unanswered += int(is_retryable_error(error))
            answered += 1
            if progress:
                progress(answered, total)
        else:
            report.retried += 1
            # To the front: a trial that breaks its worker should find out
            # quickly whether it does so every time.
            (suspect_queue if key in suspects else pending).appendleft((key, task_args))

    while pending or suspect_queue:
        live = [slot for slot in slots if slot not in retired]
        if not live:
            reason = f"every device was retired ({', '.join(sorted(retired))})"
            break
        if idle_rounds >= max_idle_rounds:
            reason = f"{idle_rounds} consecutive pool rounds answered no trial"
            break
        report.rounds += 1
        answered_before = answered
        faulted: set[str] = set()
        broken = killed = False
        blame: Optional[set] = None
        counter = ctx.Value("i", 0)
        with _executor(len(live), ctx, initializer, initargs(counter, live)) as pool:
            in_flight: dict = {}

            def fill() -> None:
                while len(in_flight) < len(live) and not (faulted or broken):
                    suspect_running = any(k in suspects for k, _, _ in in_flight.values())
                    if suspect_queue and not suspect_running:
                        key, task_args = suspect_queue.popleft()
                    elif pending:
                        key, task_args = pending.popleft()
                    else:
                        return
                    future = pool.submit(run_task, fn, task_args)
                    in_flight[future] = (key, task_args, time.monotonic())

            def assign_blame(key) -> set:
                """Which of the tasks lost to this break it is fair to charge."""
                wait(list(in_flight))  # a broken pool releases them all at once
                crashed = {key} | {
                    k for f, (k, _, _) in in_flight.items()
                    if isinstance(f.exception(), BrokenProcessPool)
                }
                if killed:
                    return set()
                if len(crashed) == 1:
                    blamed = crashed
                else:
                    prior = crashed & suspects
                    blamed = prior if len(prior) == 1 else set()
                suspects.update(blamed or crashed)
                return blamed

            fill()
            while in_flight:
                timeout = None
                if deadline_after is not None and not killed:
                    oldest = min(started for _, _, started in in_flight.values())
                    timeout = max(0.0, oldest + deadline_after - time.monotonic())
                done, _ = wait(in_flight, timeout=timeout, return_when=FIRST_COMPLETED)
                if not done:
                    now = time.monotonic()
                    for future, (key, task_args, started) in list(in_flight.items()):
                        if now - started >= deadline_after:
                            logger.error(
                                "%s: worker still busy after %.0f s; killing the pool",
                                key, now - started,
                            )
                            del in_flight[future]
                            report.hung_workers += 1
                            retry_or_fail(
                                key, task_args,
                                f"{WORKER_HUNG_ERROR}: exceeded {deadline_after:g} s",
                                charge=True, status="timeout",
                            )
                    killed = broken = True
                    _kill_workers(pool)
                    continue
                for future in done:
                    key, task_args, _ = in_flight.pop(future)
                    try:
                        result = future.result()
                    except BrokenProcessPool as exc:
                        broken = True
                        if blame is None:
                            blame = assign_blame(key)
                            if not killed:
                                logger.error(
                                    "Worker pool broke (%s); charging %s",
                                    exc, sorted(map(str, blame)) or "no trial yet",
                                )
                        retry_or_fail(key, task_args, describe(exc), charge=key in blame)
                    except Exception as exc:  # noqa: BLE001 - the task's own crash
                        on_failure(key, "failed", describe(exc))
                        answered += 1
                        if progress:
                            progress(answered, total)
                    else:
                        if isinstance(result, WorkerFault):
                            if result.device not in faulted:
                                logger.error(
                                    "Worker on %s failed (%s); retrying its trial "
                                    "on a fresh pool", result.device, result.error,
                                )
                            faulted.add(result.device)
                            report.worker_faults += int(result.attempted)
                            retry_or_fail(
                                key, task_args, result.error, charge=result.attempted
                            )
                            continue
                        suspects.discard(key)
                        row = _row_of(result)
                        if row is not None and row.get("device"):
                            strikes[row["device"]] = 0
                        on_result(key, result)
                        answered += 1
                        if progress:
                            progress(answered, total)
                fill()
        report.pool_breaks += int(broken and not killed)
        for device in faulted:
            strikes[device] += 1
            if strikes[device] >= max_device_faults:
                logger.error(
                    "Retiring %s: it faulted in %d consecutive rounds (%s)",
                    device, strikes[device], last_error,
                )
                retired.add(device)
        idle_rounds = 0 if answered > answered_before else idle_rounds + 1
    else:
        reason = None

    report.retired_devices = sorted(retired)
    pending.extend(suspect_queue)
    if pending:
        error = f"{NOT_RUN_ERROR}: {reason}; last error: {last_error}"
        logger.error("%d trial(s) not run: %s", len(pending), error)
        for key, _ in pending:
            on_failure(key, "failed", error)
            report.unanswered += 1
    return report
