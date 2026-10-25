#!/usr/bin/env python3
"""Build an MP band-gap dataset whose gap *and* structure come from one functional.

``--functional gga`` (the default, and the only choice before 2026-10-22) keeps
GGA/GGA+U band gaps; ``--functional r2scan`` keeps r2SCAN ones. Everything below is
written for GGA; the r2SCAN build is the same with ``r2SCAN``/``r2SCAN+U`` gap tasks
and energies from the ``R2SCAN`` thermo document, MP's r2SCAN phase diagram, whose
energies carry no MP2020 correction.

Materials Project mixes functionals inside a single ``summary`` document. In the
2026.04.13 snapshot the blessed structure comes from r2SCAN for 37,129 of the
154,377 non-deprecated materials while the blessed band gap for 29,217 of those
same materials comes from a GGA calculation, so the obvious pull --
``summary.structure`` next to ``summary.band_gap`` -- pairs a meta-GGA geometry
with a semilocal gap in roughly a fifth of the rows. A model trained on that is
being asked to learn a map whose input and output were computed at different
levels of theory.

This script takes the structure from *the task that produced the gap*, so the
pairing is exact by construction: same task, same functional, same geometry.
``dft_run_type`` records which of GGA / GGA+U that task was. MP applies the
Hubbard U per species and anion chemistry rather than per material, so the two
run types are two different Hamiltonians sharing one table; the column is there
to be fed to the model, not to be ignored.

Energies come from the ``GGA_GGA+U`` thermo document, which is MP's GGA/GGA+U
phase diagram -- not the blessed mixed-functional one that ``summary`` exposes.
They are recorded verbatim under an ``mp_dft_`` prefix, all eV/atom:

``mp_dft_uncorrected_energy_per_atom``
    The raw VASP energy, no corrections applied.
``mp_dft_energy_per_atom``
    The same energy after MP's anion and +U corrections -- what the hull is
    built from.
``mp_dft_formation_energy_per_atom``
``mp_dft_energy_above_hull``

``formation_energy_per_atom``, ``energy_above_hull`` and ``band_gap`` are
duplicated under their bare names because ``LEGACY_SCALAR_COLUMNS`` carries
those into the cache automatically and the tokeniser configs condition on
``energy_above_hull`` by that name.

Forces and stress are requested but **cannot be filled from this snapshot**.
``output.structure`` in MP's parsed task collection has a nullable per-site
``forces`` property and no stress tensor anywhere in its schema; a count over
the whole collection returns 0 of 2,015,403 tasks with forces populated, and
``stress`` does not appear in any of the 29 task fields. ``max_force`` and
``stress`` are therefore written as NaN with ``max_force_missing`` and
``stress_missing`` set, following the convention ``build_lemat_bulk_fmax.py``
uses for LeMat-Bulk's own force-less MP rows. LeMat-Bulk does carry forces for
its MP subset; they come from an older snapshot parsed from the raw VASP output
rather than from this API, so joining them in here would silently mix snapshots.

``mp-api`` comes with the ``mp`` extra. MP release 2026.09.28 needs mp-api >= 0.47.0,
newer than the 0.46.5 the lock pins (2026-10-22); install it into the venv by hand
until the lock moves. Then cache and tokenise as usual.
"""
import argparse
import gzip
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from wyckoff_transformer.paths import data_path

logger = logging.getLogger("build_mp_gga_band_gap")

REPO = Path(__file__).resolve().parent.parent

#: The snapshot this was written against. MP re-blesses tasks between releases,
#: so the structure/gap pairing above is a statement about this version only.
EXPECTED_DB_VERSION = "2026.04.13"

#: Per functional: the run types accepted for both the gap and the structure, and the
#: thermo document -- MP's phase diagram at that functional -- energies come from.
FUNCTIONALS = {
    "gga": {"run_types": ("GGA", "GGA+U"), "thermo_type": "GGA_GGA+U"},
    "r2scan": {"run_types": ("r2SCAN", "r2SCAN+U"), "thermo_type": "R2SCAN"},
}

#: Columns written to the split CSVs. ``read_MP`` takes column 0 as the index and
#: needs ``cif``; ``compute_symmetry_sites`` carries the rest through to the cache.
OUTPUT_COLUMNS = [
    "material_id",
    "cif",
    "band_gap",
    "dft_run_type",
    "license",
    "mp_dft_uncorrected_energy_per_atom",
    "mp_dft_energy_per_atom",
    "mp_dft_formation_energy_per_atom",
    "mp_dft_energy_above_hull",
    "max_force",
    "max_force_missing",
    "stress",
    "stress_missing",
    "formation_energy_per_atom",
    "energy_above_hull",
]


def _rester(api_key: str | None):
    from mp_api.client import MPRester

    if api_key is None:
        env = REPO / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("MP_API_KEY"):
                    api_key = line.split("=", 1)[1].strip().strip("'\"")
        api_key = api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        raise SystemExit("No MP_API_KEY: pass --api-key, export it, or put it in .env")
    return MPRester(api_key, mute_progress_bars=True)


def _chunks(seq, size):
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


_LOCAL = threading.local()


def _map_chunks(query, ids: list[str], chunk_size: int, workers: int, api_key: str | None,
                label: str) -> list:
    """Every document *query(rester, chunk)* returns, over *ids*, in chunk order.

    A chunk is one REST call whose query string carries the ids, so *chunk_size* is
    bounded by the server's URL limit (300 fits; 1,000 does not). The calls are
    latency-bound -- about 2.5 s for 300 tasks on 2026-10-22 -- so *workers* of them run
    at once, each thread with its own MPRester.
    """
    def run(batch):
        if not hasattr(_LOCAL, "rester"):
            _LOCAL.rester = _rester(api_key)
        return list(query(_LOCAL.rester, batch))

    batches = list(_chunks(ids, chunk_size))
    docs = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, chunk in enumerate(pool.map(run, batches)):
            docs.extend(chunk)
            if index % 100 == 0:
                logger.info("  %s: %d/%d chunks, %d documents", label, index + 1,
                            len(batches), len(docs))
    return docs


def fetch_summary(mpr, cache: Path) -> pd.DataFrame:
    """material_id, band_gap and the task ids MP blessed, for every live material."""
    if cache.exists():
        logger.info("reading cached %s", cache)
        return pd.read_parquet(cache)
    logger.info("fetching summary documents")
    docs = mpr.materials.summary.search(
        deprecated=False, fields=["material_id", "band_gap", "origins"])
    rows = []
    for doc in docs:
        origins = {origin.name: str(origin.task_id) for origin in (doc.origins or [])}
        rows.append((str(doc.material_id), doc.band_gap,
                     origins.get("electronic_structure"), origins.get("structure")))
    frame = pd.DataFrame(rows, columns=["material_id", "band_gap", "gap_task", "structure_task"])
    frame.to_parquet(cache)
    logger.info("summary: %d materials", len(frame))
    return frame


def fetch_run_types(task_ids: list[str], cache: Path, chunk_size: int,
                    workers: int = 1, api_key: str | None = None) -> pd.DataFrame:
    """run_type of each blessed band-gap task -- the thing that says GGA vs GGA+U."""
    if cache.exists():
        logger.info("reading cached %s", cache)
        return pd.read_parquet(cache)
    logger.info("fetching run types for %d gap tasks", len(task_ids))
    docs = _map_chunks(
        lambda mpr, batch: mpr.materials.tasks.search(
            task_ids=batch, fields=["task_id", "run_type"]),
        task_ids, chunk_size, workers, api_key, "run types")
    rows = [(str(doc.task_id), str(doc.run_type)) for doc in docs]
    frame = pd.DataFrame(rows, columns=["gap_task", "dft_run_type"])
    frame.to_parquet(cache)
    return frame


def fetch_structures(task_ids: list[str], cache: Path, chunk_size: int,
                     workers: int = 1, api_key: str | None = None) -> pd.DataFrame:
    """The geometry each gap was computed on, as a CIF.

    Written in P1: ``read_MP`` reparses these and the cache re-detects symmetry with
    pyxtal at its own tolerance, so symmetrising here would only add a second, silent
    symmetry finder to disagree with the first.
    """
    if cache.exists():
        logger.info("reading cached %s", cache)
        return pd.read_parquet(cache)
    from pymatgen.io.cif import CifWriter

    logger.info("fetching structures for %d gap tasks", len(task_ids))
    docs = _map_chunks(
        lambda mpr, batch: mpr.materials.tasks.search(
            task_ids=batch, fields=["task_id", "output.structure"]),
        task_ids, chunk_size, workers, api_key, "structures")
    rows, forces_seen = [], 0
    for doc in docs:
        structure = doc.output.structure if doc.output else None
        if structure is None:
            continue
        forces = structure.site_properties.get("forces")
        max_force = np.nan
        if forces:
            forces_seen += 1
            max_force = float(np.linalg.norm(np.asarray(forces), axis=1).max())
        rows.append((str(doc.task_id), str(CifWriter(structure)), max_force))
    # Guard rather than assume: if a future snapshot starts populating forces, this
    # says so instead of writing an all-NaN column and calling it missing data.
    logger.info("tasks reporting forces: %d of %d", forces_seen, len(rows))
    frame = pd.DataFrame(rows, columns=["gap_task", "cif", "max_force"])
    frame.to_parquet(cache)
    return frame


def fetch_licenses(material_ids: list[str], cache: Path, chunk_size: int,
                   workers: int = 1, api_key: str | None = None) -> pd.DataFrame:
    """Each material's data licence, ``builder_meta.license``: ``BY-C`` or ``BY-NC``.

    MP serves most of its data under CC BY 4.0 but the GNoME structures and their
    r2SCAN statics (batch ``gnome_r2scan_statics``) under CC BY-NC 4.0, and they are
    only visible to an API key whose owner accepted that licence. In 2026.09.28 that
    is ~93% of the materials whose blessed band gap is r2SCAN and ~0.3% of the GGA+U
    ones. A model trained on BY-NC rows inherits the non-commercial term.
    """
    if cache.exists():
        logger.info("reading cached %s", cache)
        return pd.read_parquet(cache)
    logger.info("fetching licences for %d materials", len(material_ids))
    docs = _map_chunks(
        lambda mpr, batch: mpr.materials.summary.search(
            material_ids=batch, fields=["material_id", "builder_meta"]),
        material_ids, chunk_size, workers, api_key, "licences")
    rows = [(str(doc.material_id),
             str(getattr(doc.builder_meta, "license", None) or "unknown")) for doc in docs]
    frame = pd.DataFrame(rows, columns=["material_id", "license"])
    frame.to_parquet(cache)
    return frame


def fetch_energies(material_ids: list[str], cache: Path, chunk_size: int,
                   thermo_type: str = "GGA_GGA+U", workers: int = 1,
                   api_key: str | None = None) -> pd.DataFrame:
    """MP's phase diagram at one functional, not the blessed mixed-functional one."""
    if cache.exists():
        logger.info("reading cached %s", cache)
        return pd.read_parquet(cache)
    logger.info("fetching %s thermo for %d materials", thermo_type, len(material_ids))
    fields = ["material_id", "uncorrected_energy_per_atom", "energy_per_atom",
              "formation_energy_per_atom", "energy_above_hull", "entries"]
    docs = _map_chunks(
        lambda mpr, batch: mpr.materials.thermo.search(
            material_ids=batch, thermo_types=[thermo_type], fields=fields),
        material_ids, chunk_size, workers, api_key, "thermo")
    rows = [(str(doc.material_id), doc.uncorrected_energy_per_atom, doc.energy_per_atom,
             doc.formation_energy_per_atom, doc.energy_above_hull,
             "+".join(sorted(doc.entries or ()))) for doc in docs]
    frame = pd.DataFrame(rows, columns=[
        "material_id", "mp_dft_uncorrected_energy_per_atom", "mp_dft_energy_per_atom",
        "mp_dft_formation_energy_per_atom", "mp_dft_energy_above_hull", "energy_run_types"])
    frame.to_parquet(cache)
    return frame


def assemble(summary: pd.DataFrame, run_types: pd.DataFrame,
             structures: pd.DataFrame, energies: pd.DataFrame,
             accepted_run_types: tuple[str, ...] = FUNCTIONALS["gga"]["run_types"],
             licenses: pd.DataFrame | None = None,
             exclude_licenses: tuple[str, ...] = ()) -> pd.DataFrame:
    """Join the four pulls and drop what cannot carry a usable label."""
    frame = summary.merge(run_types, on="gap_task", how="left")
    logger.info("materials                       : %9d", len(frame))
    frame = frame[frame["band_gap"].notna()]
    logger.info("  with a band gap               : %9d", len(frame))
    frame = frame[frame["dft_run_type"].isin(accepted_run_types)]
    logger.info("  gap from %-21s: %9d", "/".join(accepted_run_types), len(frame))
    if licenses is not None:
        frame = frame.merge(licenses, on="material_id", how="left")
        frame["license"] = frame["license"].fillna("unknown")
        logger.info("  licences                      : %s",
                    frame["license"].value_counts().to_dict())
        frame = frame[~frame["license"].isin(exclude_licenses)]
        logger.info("  & licence not in %-13s: %9d", ",".join(exclude_licenses) or "-",
                    len(frame))

    frame = frame.merge(structures, on="gap_task", how="left")
    frame = frame[frame["cif"].notna()]
    logger.info("  & the gap task has a structure: %9d", len(frame))

    frame = frame.merge(energies, on="material_id", how="left")
    # The tokeniser configs use energy_above_hull as an AdaLN conditioning feature, so
    # a NaN there is not a missing covariate but a broken row. Materials outside MP's
    # GGA/GGA+U phase diagram have no such energy at all.
    frame = frame[frame["mp_dft_energy_above_hull"].notna()]
    logger.info("  & in the functional's hull    : %9d", len(frame))

    # MP blesses the gap task and the thermo entry independently, so a handful of rows
    # take their gap from GGA and their energies from GGA+U or vice versa. The gap is
    # the label, so dft_run_type follows it; this says when the energies disagree.
    mismatch = frame["energy_run_types"] != frame["dft_run_type"]
    logger.info("  energies from another run type: %9d", int(mismatch.sum()))

    frame["max_force_missing"] = frame["max_force"].isna().astype(float)
    # Nothing to impute from: no task in the snapshot reports a stress tensor.
    frame["stress"] = np.nan
    frame["stress_missing"] = 1.0
    frame["formation_energy_per_atom"] = frame["mp_dft_formation_energy_per_atom"]
    frame["energy_above_hull"] = frame["mp_dft_energy_above_hull"]
    return frame.set_index("material_id")


def write_splits(frame: pd.DataFrame, name: str, seed: int,
                 val_fraction: float, test_fraction: float, compresslevel: int) -> dict:
    rng = np.random.default_rng(seed)
    shuffled = frame.index.to_numpy().copy()
    rng.shuffle(shuffled)
    n_val = int(round(val_fraction * len(shuffled)))
    n_test = int(round(test_fraction * len(shuffled)))
    split_of = pd.Series("train", index=frame.index)
    split_of[shuffled[:n_val]] = "val"
    split_of[shuffled[n_val:n_val + n_test]] = "test"

    out_dir = data_path(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for split in ("train", "val", "test"):
        rows = frame[split_of == split].reset_index()[OUTPUT_COLUMNS]
        path = out_dir / f"{split}.csv.gz"
        with gzip.open(path, "wt", newline="", compresslevel=compresslevel) as handle:
            rows.to_csv(handle, index=False)
        written[split] = len(rows)
        logger.info("wrote %s: %d rows", path, len(rows))
    return written


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="mp_2026_gga_gap",
                        help="Dataset name; splits are written to data/<name>/.")
    parser.add_argument("--functional", choices=sorted(FUNCTIONALS), default="gga",
                        help="Which band gaps to keep: GGA/GGA+U or r2SCAN.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Where the per-stage pulls are cached so a re-run is cheap. "
                             "Defaults to data/<name>/.raw/.")
    parser.add_argument("--chunk-size", type=int, default=300,
                        help="Ids per REST call. 300 keeps the query string under the "
                             "server's URL limit.")
    parser.add_argument("--exclude-license", action="append", default=[],
                        choices=("BY-NC", "unknown"),
                        help="Drop materials MP serves under this licence; repeatable. "
                             "BY-NC is GNoME's (see fetch_licenses).")
    parser.add_argument("--workers", type=int, default=1,
                        help="REST calls in flight at once; each is latency-bound.")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--compresslevel", type=int, default=1)
    parser.add_argument("--allow-other-snapshot", action="store_true",
                        help="Proceed even if the live MP release is not "
                             f"{EXPECTED_DB_VERSION}.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    work_dir = args.work_dir or data_path(args.name, ".raw")
    work_dir.mkdir(parents=True, exist_ok=True)
    mpr = _rester(args.api_key)
    version = mpr.db_version
    logger.info("MP database version: %s", version)
    if version != EXPECTED_DB_VERSION and not args.allow_other_snapshot:
        raise SystemExit(
            f"Live MP release is {version}, not {EXPECTED_DB_VERSION}. The blessing of "
            "tasks changes between releases; re-read the module docstring, then pass "
            "--allow-other-snapshot.")

    summary = fetch_summary(mpr, work_dir / "summary.parquet")
    gap_tasks = sorted(set(summary["gap_task"].dropna()))
    run_types = fetch_run_types(gap_tasks, work_dir / "run_types.parquet", args.chunk_size,
                                args.workers, args.api_key)

    wanted = summary.merge(run_types, on="gap_task", how="left")
    functional = FUNCTIONALS[args.functional]
    wanted = wanted[wanted["band_gap"].notna()
                    & wanted["dft_run_type"].isin(functional["run_types"])]
    structures = fetch_structures(
        sorted(set(wanted["gap_task"])), work_dir / "structures.parquet", args.chunk_size,
        args.workers, args.api_key)
    energies = fetch_energies(
        sorted(set(wanted["material_id"])), work_dir / "energies.parquet", args.chunk_size,
        functional["thermo_type"], args.workers, args.api_key)

    licenses = fetch_licenses(
        sorted(set(wanted["material_id"])), work_dir / "licenses.parquet", args.chunk_size,
        args.workers, args.api_key)
    frame = assemble(summary, run_types, structures, energies, functional["run_types"],
                     licenses, tuple(args.exclude_license))
    logger.info("run types : %s", frame["dft_run_type"].value_counts().to_dict())
    logger.info("gap > 0   : %d (%.1f%%)", int((frame["band_gap"] > 0).sum()),
                100 * float((frame["band_gap"] > 0).mean()))
    written = write_splits(frame, args.name, args.seed,
                           args.val_fraction, args.test_fraction, args.compresslevel)
    logger.info("split sizes: %s", written)


if __name__ == "__main__":
    main()
