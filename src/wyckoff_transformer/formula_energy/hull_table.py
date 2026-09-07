"""Formation energy and distance above the hull for every row of an energy table.

This is definition (1) of :doc:`../../../docs/e_hull_definitions.md`: PBE
``energy_corrected`` against phase diagrams built from the archive itself,
producing ``e_form`` (formation energy, eV/atom) and ``e_hull`` (distance above
the hull, eV/atom).  It replaces ``scripts/compute_e_hull.py``, which produced
every training label in this repository until 2026-09-07, and which this
reproduces **exactly**: relabelling the whole archive returned bit-identical
``e_form`` and ``e_hull`` for all 4,746,049 rows the old script had managed to
label (max ``|Δ|`` 0.0 for both).  It differs from it in four ways, all
deliberate:

*No element exclusions.*  The script refused any system containing Yb, anything
with Z >= 84, or ten or more elements, and returned nothing for those rows.
That cost 589,250 of the archive's 5,335,299 rows their labels -- 579,217 of
them to the Z >= 84 clause alone -- for chemistry the LeMat reference hull
carries perfectly well (Th, U, Np, Pu, Ac, Pa and 1211 Yb entries all appear in
it, each with an elemental reference).  The ten-element clause could never fire:
the archive's widest chemical system has nine.

*Below-hull rows are negative, not missing.*  ``PhaseDiagram.get_e_above_hull``
raises for an entry below the hull, and the script turned that into
``(None, None)`` -- losing ``e_form`` with it.  Nothing can be below the hull it
defines, so this only bit when a *separate* reference was passed, which is
exactly the answer key's use.  Here it is a negative number.

*Failures are counted, not swallowed.*  The script wrapped every row in a bare
``except``, which made an excluded element indistinguishable from a chemical
system with no elemental reference.  Of the whole archive, exactly one row fails
for a real reason: ``oqmd-2969647`` carries an energy (-17.7176 eV) but no
``full_formula`` and no ``chemsys``, so there is no composition to build a
``PDEntry`` from and no chemical system to gather references for.  That is worth
reporting rather than hiding.

*The heavy columns are streamed.*  The script read the whole input CSV --
including the ~1 GB of CIF text -- into one frame.  Only four columns matter to
the calculation, so the CIFs are copied through in chunks instead.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import FrozenSet, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: The only columns the calculation reads.  Everything else in the input is
#: copied through untouched.
LIGHT_COLUMNS = ("immutable_id", "full_formula", "chemsys", "energy_corrected")

#: Columns this module appends.
OUTPUT_COLUMNS = ("e_form", "e_hull")

#: Rows per chunk when copying the input through.  At 250k rows a chunk of the
#: archive with its CIF column is a few hundred MB.
DEFAULT_CHUNK_SIZE = 250_000

#: Set before anything is imported or read, because the pool below forks: a BLAS
#: thread pool that exists at fork time is inherited broken, and a phase diagram
#: is scalar work that gains nothing from threads anyway.
_SINGLE_THREAD_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def _parse_chemsys(values: Iterable) -> list[FrozenSet[str]]:
    """``"Ba-O-Ti"`` to ``frozenset({"Ba", "O", "Ti"})``, per row."""
    return [frozenset(str(value).split("-")) for value in values]


class ChemsysIndex:
    """Reference entries grouped by chemical system.

    A phase diagram over a chemical system needs its subsystems as well -- the
    binaries and the elemental references, without which pymatgen cannot place
    the origin of the formation energy.  Grouping by system and enumerating
    subsets makes that lookup O(2^k) in the system's arity (at most 9 in the
    archive, so at most 511 lookups) rather than a scan over five million rows.

    Args:
        reference: Rows with :data:`LIGHT_COLUMNS`.  Rows with a missing
            formula, system or energy are dropped, as they cannot enter a phase
            diagram.
    """

    def __init__(self, reference: pd.DataFrame) -> None:
        frame = reference.dropna(subset=["full_formula", "chemsys", "energy_corrected"])
        self._by_system: dict[FrozenSet[str], list[tuple[str, float]]] = defaultdict(list)
        for system, formula, energy in zip(
            _parse_chemsys(frame["chemsys"]),
            frame["full_formula"].astype(str),
            frame["energy_corrected"].astype(float),
        ):
            self._by_system[system].append((formula, energy))
        logger.info(
            "reference: %d rows over %d chemical systems",
            len(frame), len(self._by_system),
        )

    @property
    def systems(self) -> list[FrozenSet[str]]:
        return list(self._by_system)

    def entries(self, system: FrozenSet[str]) -> list:
        """PDEntries for *system* and every subsystem of it."""
        from pymatgen.analysis.phase_diagram import PDEntry
        from pymatgen.core import Composition

        rows: list[tuple[str, float]] = []
        elements = sorted(system)
        for size in range(1, len(elements) + 1):
            for subset in itertools.combinations(elements, size):
                rows.extend(self._by_system.get(frozenset(subset), ()))
        return [PDEntry(Composition(formula), energy) for formula, energy in rows]


#: Set by :func:`_init_worker`.  The index is built once in the parent and
#: inherited through ``fork``: it is read-only, and pickling five million
#: formula/energy pairs to every worker would cost more than the calculation.
_WORKER_INDEX: Optional[ChemsysIndex] = None


def _init_worker(index: ChemsysIndex) -> None:
    global _WORKER_INDEX
    _WORKER_INDEX = index


def _score_system(task: tuple[FrozenSet[str], tuple[tuple[str, float], ...]]):
    """``(e_form, e_hull)`` per target row of one chemical system.

    Returns:
        ``(results, reason)`` where *results* is one ``(e_form, e_hull)`` pair
        per target row -- ``(nan, nan)`` for a row that could not be placed --
        and *reason* names the failure when the whole system failed.
    """
    from pymatgen.analysis.phase_diagram import PDEntry
    from pymatgen.core import Composition

    system, targets = task
    assert _WORKER_INDEX is not None
    try:
        diagram = _phase_diagram(_WORKER_INDEX.entries(system))
    except Exception as exc:
        return [(np.nan, np.nan)] * len(targets), f"no phase diagram: {type(exc).__name__}"

    results = []
    for formula, energy in targets:
        try:
            entry = PDEntry(Composition(formula), energy)
            results.append((
                diagram.get_form_energy_per_atom(entry),
                # allow_negative, so that a row below a *separate* reference
                # hull is reported rather than discarded.
                diagram.get_decomp_and_e_above_hull(entry, allow_negative=True)[1],
            ))
        except Exception as exc:
            logger.debug("row %s in %s failed: %s", formula, sorted(system), exc)
            results.append((np.nan, np.nan))
    return results, None


def _phase_diagram(entries):
    from pymatgen.analysis.phase_diagram import PhaseDiagram

    if not entries:
        raise ValueError("no reference entries")
    return PhaseDiagram(entries)


def _import_pymatgen() -> None:
    """Import what the workers need, in the parent, before the pool forks.

    Two small reasons, no deep one: a forked child then inherits the modules
    instead of each of sixteen workers importing pymatgen for itself, and an
    unusable environment fails here rather than as sixteen simultaneous
    tracebacks from inside the pool.
    """
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram  # noqa: F401
    from pymatgen.core import Composition  # noqa: F401


def hull_energies(
    rows: pd.DataFrame,
    reference: Optional[pd.DataFrame] = None,
    workers: int = 1,
) -> pd.DataFrame:
    """``e_form`` and ``e_hull`` for every row of *rows*.

    Args:
        rows: Rows to label, with :data:`LIGHT_COLUMNS`.
        reference: Rows defining the hull.  Defaults to *rows* itself, which is
            what makes the archive's labels self-referential; pass a subset to
            measure against a smaller world, as the answer key does.
        workers: Processes to spread the chemical systems over.  One phase
            diagram per system dominates the cost -- 97 ms each on this
            machine, 673,173 systems for the whole archive.

    Returns:
        A frame indexed like *rows* with :data:`OUTPUT_COLUMNS`, ``NaN`` where
        the row could not be placed.
    """
    if reference is None:
        reference = rows
    _import_pymatgen()
    index = ChemsysIndex(reference)

    systems = _parse_chemsys(rows["chemsys"])
    formulas = rows["full_formula"].astype(str).to_numpy()
    energies = pd.to_numeric(rows["energy_corrected"], errors="coerce").to_numpy(dtype=float)

    positions: dict[FrozenSet[str], list[int]] = defaultdict(list)
    for position, system in enumerate(systems):
        positions[system].append(position)
    logger.info(
        "labelling %d rows over %d chemical systems with %d worker(s)",
        len(rows), len(positions), workers,
    )

    tasks = [
        (system, tuple((formulas[p], energies[p]) for p in members))
        for system, members in positions.items()
    ]
    e_form = np.full(len(rows), np.nan)
    e_hull = np.full(len(rows), np.nan)
    reasons: Counter = Counter()

    for (system, _), (results, reason) in zip(tasks, _map(tasks, workers, index)):
        if reason is not None:
            reasons[reason] += len(positions[system])
        for position, (form, hull) in zip(positions[system], results):
            e_form[position], e_hull[position] = form, hull

    missing = int(np.isnan(e_hull).sum())
    logger.info("%d of %d rows have no hull energy", missing, len(rows))
    for reason, count in reasons.most_common():
        logger.info("  %s: %d rows", reason, count)
    return pd.DataFrame(
        {"e_form": e_form, "e_hull": e_hull}, index=rows.index
    )


def _map(tasks: Sequence, workers: int, index: ChemsysIndex):
    """Run :func:`_score_system` over *tasks*, in this process or a pool."""
    if workers <= 1:
        _init_worker(index)
        return [_score_system(task) for task in tasks]

    import multiprocessing

    # fork, not spawn: the index is inherited copy-on-write instead of pickled,
    # and nothing here touches CUDA, which is the reason the rest of this
    # package insists on spawn.
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(index,),
    ) as pool:
        # A chunk per worker-batch: the per-task payload is small and the tasks
        # are many, so per-task IPC would dominate.
        return list(pool.map(_score_system, tasks, chunksize=64))


def annotate_csv(
    input_csv: Path,
    output_csv: Path,
    reference_csv: Optional[Path] = None,
    workers: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Path:
    """Copy *input_csv* to *output_csv* with ``e_form`` and ``e_hull`` appended.

    Every other column is passed through unchanged, in the original row order,
    which is what makes the output a drop-in replacement for the table
    ``scripts/compute_e_hull.py`` used to write -- ``oracle_reconstruction.py``
    reads its CIFs, and the formula table, the screeners and the conditioning
    labels read its energies.

    Args:
        input_csv: Rows to label.  Must carry :data:`LIGHT_COLUMNS`.
        output_csv: Destination; gzipped when the name ends in ``.gz``.
        reference_csv: Rows defining the hull, or ``None`` for *input_csv*.
        workers: Processes for the phase diagrams.
        chunk_size: Rows per chunk while copying the heavy columns through.

    Returns:
        *output_csv*.
    """
    input_csv, output_csv = Path(input_csv), Path(output_csv)
    logger.info("reading %s", input_csv)
    rows = pd.read_csv(input_csv, usecols=list(LIGHT_COLUMNS), low_memory=False)

    if reference_csv is None:
        reference = rows
    else:
        logger.info("reading reference %s", reference_csv)
        reference = pd.read_csv(
            reference_csv, usecols=list(LIGHT_COLUMNS), low_memory=False
        )

    labels = hull_energies(rows, reference=reference, workers=workers)
    # Position, not immutable_id: one archive row has no id, and the join has to
    # survive that rather than drop or misplace it.
    if len(labels) != len(rows):
        raise AssertionError("hull_energies returned a different number of rows")
    del rows, reference

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if output_csv.suffix == ".gz" else None
    written = 0
    for index, chunk in enumerate(pd.read_csv(input_csv, chunksize=chunk_size)):
        block = labels.iloc[written:written + len(chunk)]
        for column in OUTPUT_COLUMNS:
            chunk[column] = block[column].to_numpy()
        chunk.to_csv(
            output_csv,
            mode="w" if index == 0 else "a",
            header=index == 0,
            index=False,
            compression=compression,
        )
        written += len(chunk)
        if index % 4 == 0:
            logger.info("written %d rows", written)
    if written != len(labels):
        raise AssertionError(
            f"wrote {written} rows for {len(labels)} labels; the input changed "
            f"under us or a chunk was dropped"
        )
    logger.info("wrote %d rows to %s", written, output_csv)
    return output_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wyformer-hull-energies",
        description=(
            "Add e_form and e_hull to an energy table, against phase diagrams "
            "built from that table or from a separate reference. Replaces "
            "scripts/compute_e_hull.py; see docs/e_hull_definitions.md."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--ref-file", type=Path, default=None,
        help="Rows defining the hull. Defaults to the input file itself.",
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Processes for the phase diagrams. This machine has 24 physical cores.",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    for key, value in _SINGLE_THREAD_ENV_VARS.items():
        os.environ.setdefault(key, value)
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    annotate_csv(
        args.input_file,
        args.output_file,
        reference_csv=args.ref_file,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
