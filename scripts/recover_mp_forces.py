#!/usr/bin/env python3
"""Recover the forces and stress LeMat-Bulk lacks for 30,679 Materials Project rows.

LeMat-Bulk's MP rows carry the energy, geometry, forces and stress of one MP *task*
(a VASP calculation), keyed by the *material* id. For 22% of them ``forces`` and
``stress_tensor`` are empty lists. The public MP API cannot fill them: its task
endpoint serves a slimmed document without forces or stress. The full task documents
are public on AWS Open Data (``s3://materialsproject-parsed``, no credentials), and
there the relaxation's final ionic step still carries both.

Stages, each resumable and cached under ``cache/mp_forces_recovery/``:

``select``
    The rows to recover (every MP row with empty forces) plus a control sample of MP
    rows that *do* have forces in LeMat, used to prove that the procedure picks the
    calculation LeMat used.
``identify``
    MP API: material -> its GGA/GGA+U structure-optimisation and static tasks -> the
    task whose final energy equals LeMat's ``energy``.
``locate``
    Task id -> the S3 JSONL file holding its full document, from the bucket manifests.
``fetch``
    Stream each needed file once and keep only the needed task documents' outputs.
``match``
    Strict identity (species per site, lattice and positions to 1e-4 A, energy to
    1 meV total) and the recovered forces and stress.

Needs ``MP_API_KEY`` in ``.env``; it is read in-process and never echoed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from wyckoff_transformer.paths import cache_root, data_root

logger = logging.getLogger("recover_mp_forces")

REPO = Path(__file__).resolve().parent.parent
RAW = data_root() / "lemat-bulk" / "raw" / "data.parquet"
PROVENANCE = data_root() / "mp_provenance.csv.gz"
WORK = cache_root() / "mp_forces_recovery"
API = "https://api.materialsproject.org"

#: LeMat's energy is a PBE(+U) number; anything else cannot be the calculation behind it.
PBE_CALC_PREFIXES = ("GGA ", "GGA+U ")
CANDIDATE_TASK_TYPES = ("Structure Optimization", "Static")
ENERGY_TOL = 1e-3  # eV, total


# --------------------------------------------------------------------------- API client
class Rest:
    """Cached GET against the MP API. The key never leaves this object."""

    def __init__(self, cache: Path):
        import requests

        key = None
        env = REPO / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("MP_API_KEY"):
                    key = line.split("=", 1)[1].strip().strip("'\"")
        if not key:
            raise SystemExit("MP_API_KEY is not in .env")
        self.session = requests.Session()
        self.session.headers["X-API-KEY"] = key
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)

    def get(self, endpoint: str, params: dict) -> dict:
        tag = hashlib.sha1(json.dumps([endpoint, params], sort_keys=True).encode()).hexdigest()
        path = self.cache / f"{tag}.json"
        if path.exists():
            return json.loads(path.read_text())
        for attempt in range(6):
            response = self.session.get(f"{API}{endpoint}", params=params, timeout=300)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.rename(path)
            return payload
        response.raise_for_status()
        raise RuntimeError(f"{endpoint} kept failing")


def _legacy_id(identifier: str) -> str:
    """``aaaaadle`` / ``mp-aaaaadle`` -> ``mp-2318``.

    ``id_format=legacy`` rewrites id *values* (``material_id``, ``task_ids``) but leaves
    the *keys* of ``task_types`` / ``calc_types`` in MP's stored AlphaID form, which is
    the integer in base 26 with ``a`` = 0.
    """
    prefix, _, body = identifier.rpartition("-")
    if not re.fullmatch(r"[a-z]+", body):
        return identifier
    number = 0
    for letter in body:
        number = number * 26 + ord(letter) - ord("a")
    return f"{prefix or 'mp'}-{number}"


def _chunks(seq, size):
    for start in range(0, len(seq), size):
        yield seq[start:start + size]


# --------------------------------------------------------------------------- select
def select(n_control: int, limit_missing: int | None, seed: int) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    table = ds.dataset(RAW).to_table(
        columns=["immutable_id", "energy", "nsites", "last_modified", "chemical_formula_reduced",
                 "forces"],
        filter=pc.starts_with(ds.field("immutable_id"), "mp-"))
    n_forces = pc.list_value_length(table.column("forces")).fill_null(0).to_numpy()
    frame = table.drop(["forces"]).to_pandas()
    frame["group"] = np.where(n_forces == 0, "missing", "control_pool")
    missing = frame[frame["group"] == "missing"]
    if limit_missing is not None:
        missing = missing.sample(min(limit_missing, len(missing)), random_state=seed)
    control = frame[frame["group"] == "control_pool"].sample(n_control, random_state=seed)
    control = control.assign(group="control")
    targets = pd.concat([missing, control], ignore_index=True)
    logger.info("targets: %d missing, %d control", len(missing), len(control))
    return targets


# --------------------------------------------------------------------------- identify
def identify(rest: Rest, targets: pd.DataFrame, material_batch: int, task_batch: int) -> pd.DataFrame:
    """The PBE(+U) task whose final energy is LeMat's energy, per material."""
    materials = {}
    ids = targets["immutable_id"].tolist()
    fields = "material_id,task_ids,task_types,calc_types,deprecated_tasks,deprecated"
    for deprecated in ("false", "true"):
        todo = [i for i in ids if i not in materials]
        for index, batch in enumerate(_chunks(todo, material_batch)):
            payload = rest.get("/materials/core/", {
                "material_ids": ",".join(batch), "deprecated": deprecated,
                "id_format": "legacy", "_fields": fields, "_limit": len(batch)})
            for doc in payload["data"]:
                materials[doc["material_id"]] = doc
            if index % 20 == 0:
                logger.info("materials (deprecated=%s): %d/%d", deprecated,
                            min((index + 1) * material_batch, len(todo)), len(todo))
    logger.info("materials resolved: %d of %d", len(materials), len(ids))

    candidates = {}
    for mid, doc in materials.items():
        for key, task_type in (doc.get("task_types") or {}).items():
            calc_type = (doc.get("calc_types") or {}).get(key, "")
            if task_type in CANDIDATE_TASK_TYPES and calc_type.startswith(PBE_CALC_PREFIXES):
                candidates.setdefault(mid, []).append(_legacy_id(key))
    all_tasks = sorted({t for ts in candidates.values() for t in ts})
    logger.info("candidate tasks: %d", len(all_tasks))

    tasks = {}
    for index, batch in enumerate(_chunks(all_tasks, task_batch)):
        payload = rest.get("/materials/tasks/", {
            "task_ids": ",".join(batch), "id_format": "legacy",
            "_fields": "task_id,task_type,calc_type,completed_at,last_updated,nsites,output.energy",
            "_limit": len(batch)})
        for doc in payload["data"]:
            tasks[doc["task_id"]] = doc
        if index % 20 == 0:
            logger.info("tasks: %d/%d", min((index + 1) * task_batch, len(all_tasks)), len(all_tasks))

    rows = []
    for row in targets.itertuples(index=False):
        doc = materials.get(row.immutable_id)
        record = {"immutable_id": row.immutable_id, "group": row.group,
                  "material_found": doc is not None,
                  "material_deprecated": None if doc is None else bool(doc.get("deprecated")),
                  "n_candidates": 0, "n_energy_matches": 0, "task_id": None,
                  "identified_by": None, "time_match": None, "best_task_id": None,
                  "task_type": None, "calc_type": None, "task_deprecated": None,
                  "task_completed_at": None, "task_nsites": None, "best_dE": np.nan}
        if doc is not None:
            deprecated_tasks = {_legacy_id(t) for t in doc.get("deprecated_tasks") or []}
            lemat_time = pd.Timestamp(row.last_modified)
            scored = []
            for tid in candidates.get(row.immutable_id, []):
                task = tasks.get(tid)
                energy = ((task or {}).get("output") or {}).get("energy")
                if energy is None:
                    continue
                dE = energy - row.energy
                updated = task.get("last_updated")
                # LeMat's last_modified is the task's last_updated, to the second, for 98% of
                # the rows whose energy matches: the tie-breaker when several tasks share it.
                time_match = updated is not None and abs(
                    (pd.Timestamp(updated) - lemat_time).total_seconds()) <= 1
                scored.append((abs(dE) > ENERGY_TOL, not time_match, abs(dE), tid, dE, task, time_match))
            record["n_candidates"] = len(scored)
            if scored:
                scored.sort(key=lambda s: s[:3])
                best = scored[0]
                record.update({
                    "n_energy_matches": sum(not s[0] for s in scored),
                    "best_task_id": best[3], "best_dE": best[4], "time_match": best[6],
                    "task_type": best[5].get("task_type"), "calc_type": best[5].get("calc_type"),
                    "task_deprecated": best[3] in deprecated_tasks,
                    "task_completed_at": str(best[5].get("completed_at")),
                    "task_nsites": best[5].get("nsites")})
                # An energy within 1 meV but neither to the micro-eV nor at LeMat's timestamp
                # can be a different structure that happens to land there (seen in the pilot:
                # 0.5-0.8 meV apart, lattices 10 A apart). Let the timestamp decide first.
                if not best[0] and (best[6] or abs(best[4]) <= 1e-5):
                    record.update({"task_id": best[3], "identified_by": "api_energy"})
        rows.append(record)
    frame = pd.DataFrame(rows)
    frame = _timestamp_fallback(frame, targets)
    weak = frame["task_id"].isna() & (frame["best_dE"].abs() <= ENERGY_TOL)
    frame.loc[weak, "task_id"] = frame.loc[weak, "best_task_id"]
    frame.loc[weak, "identified_by"] = "api_energy_weak"
    logger.info("identified: %s", frame["identified_by"].fillna("none").value_counts().to_dict())
    return frame


def _timestamp_fallback(frame: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Rows the API cannot match: find the task by LeMat's last_modified and composition.

    The material documents no longer list some of the tasks LeMat took its energy from --
    in the pilot, mostly ``mvc-`` (Multivalent Consortium) tasks -- but the full task
    documents are still in the S3 buckets, whose manifest carries ``last_updated`` and
    ``formula_pretty``. A hit here is only a candidate: `match` verifies it by energy and
    geometry against the full document.
    """
    from pymatgen.core import Composition

    unmatched = frame["task_id"].isna()
    if not unmatched.any():
        return frame
    info = targets.set_index("immutable_id")
    if "chemical_formula_reduced" not in info:
        import pyarrow.compute as pc
        import pyarrow.dataset as ds

        formulas = ds.dataset(RAW).to_table(
            columns=["immutable_id", "chemical_formula_reduced"],
            filter=pc.field("immutable_id").isin(info.index.tolist())).to_pandas()
        info = info.join(formulas.set_index("immutable_id"))
    manifests()  # downloads them if absent
    manifest = pd.read_parquet(WORK / "manifest_atomate2.parquet",
                               columns=["task_id", "formula_pretty", "last_updated"])
    manifest["second"] = pd.to_datetime(manifest["last_updated"]).dt.floor("s")
    wanted = frame.loc[unmatched, "immutable_id"]
    times = pd.to_datetime(info.loc[wanted, "last_modified"]).dt.floor("s")
    window = pd.concat([times + pd.Timedelta(seconds=s) for s in (-1, 0, 1)])
    manifest = manifest[manifest["second"].isin(set(window))]
    manifest["reduced"] = [Composition(f).reduced_formula for f in manifest["formula_pretty"]]

    n_hits = []
    for index in frame.index[unmatched]:
        mid = frame.at[index, "immutable_id"]
        second = times.loc[mid]
        target = Composition(info.at[mid, "chemical_formula_reduced"]).reduced_formula
        hits = manifest[((manifest["second"] - second).abs() <= pd.Timedelta(seconds=1))
                        & (manifest["reduced"] == target)]
        n_hits.append((index, len(hits)))
        if len(hits) == 1:
            frame.at[index, "task_id"] = hits["task_id"].iloc[0]
            frame.at[index, "identified_by"] = "manifest_timestamp"
    counts = pd.Series(dict(n_hits))
    frame["n_timestamp_hits"] = counts.reindex(frame.index)
    logger.info("timestamp fallback: %d unmatched rows, %d with exactly one hit, %d with several",
                int(unmatched.sum()), int((counts == 1).sum()), int((counts > 1).sum()))
    return frame


# --------------------------------------------------------------------------- locate
S3 = "https://materialsproject-parsed.s3.amazonaws.com/"
LEGACY_MANIFEST = "tasks-legacy/tasks/manifest.jsonl"
ATOMATE2_MANIFEST = "tasks_atomate2/format=jsonl/manifest.parquet"
EXTRACTED = WORK / "extracted"
LAT_TOL = 1e-4  # A
POS_TOL = 1e-4  # A


def _download(url: str, path: Path) -> None:
    import requests

    if path.exists():
        return
    logger.info("downloading %s", url)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in response.iter_content(1 << 20):
                fh.write(chunk)
    tmp.rename(path)


def manifests() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Task id -> file, for both public task buckets. Legacy is converted to parquet once."""
    WORK.mkdir(parents=True, exist_ok=True)
    legacy_parquet = WORK / "manifest_legacy.parquet"
    if not legacy_parquet.exists():
        jsonl = WORK / "manifest_legacy.jsonl"
        _download(S3 + LEGACY_MANIFEST, jsonl)
        rows = []
        with open(jsonl) as fh:
            for line in fh:
                doc = json.loads(line)
                rows.append((doc["task_id"], doc["nelements"], doc["output_spacegroup_number"],
                             doc["dt"]))
        pd.DataFrame(rows, columns=["task_id", "nelements", "spacegroup", "dt"]).to_parquet(
            legacy_parquet)
    atomate2 = WORK / "manifest_atomate2.parquet"
    _download(S3 + ATOMATE2_MANIFEST, atomate2)
    return (pd.read_parquet(legacy_parquet, columns=["task_id", "nelements", "spacegroup", "dt"]),
            pd.read_parquet(atomate2, columns=["task_id", "sourced_from_path"]))


def locate(identified: pd.DataFrame) -> pd.DataFrame:
    legacy, atomate2 = manifests()
    tasks = identified.loc[identified["task_id"].notna(), ["task_id"]].drop_duplicates()
    legacy = legacy[legacy["task_id"].isin(tasks["task_id"])]
    legacy_keys = pd.DataFrame({
        "task_id": legacy["task_id"].to_numpy(),
        "legacy_key": ("tasks-legacy/tasks/nelements=" + legacy["nelements"].astype(str)
                       + "/output_spacegroup_number=" + legacy["spacegroup"].astype(str)
                       + "/dt=" + legacy["dt"].astype(str) + ".jsonl.gz").to_numpy()})
    atomate2 = atomate2[atomate2["task_id"].isin(tasks["task_id"])]
    atomate2_keys = pd.DataFrame({
        "task_id": atomate2["task_id"].to_numpy(),
        "atomate2_key": atomate2["sourced_from_path"].str.replace(
            "s3://materialsproject-parsed/", "", regex=False).to_numpy()})
    frame = tasks.merge(legacy_keys, on="task_id", how="left").merge(
        atomate2_keys, on="task_id", how="left")
    logger.info("tasks %d: in legacy %d, in atomate2 %d, in neither %d; legacy files %d",
                len(frame), frame["legacy_key"].notna().sum(), frame["atomate2_key"].notna().sum(),
                (frame["legacy_key"].isna() & frame["atomate2_key"].isna()).sum(),
                frame["legacy_key"].nunique())
    return frame


# --------------------------------------------------------------------------- fetch
#: ``mp-`` and ``mvc-`` (Multivalent Consortium) task ids both occur among LeMat's sources.
TASK_ID_RE = re.compile(rb'"task_id":\s*"([a-z]+-\d+)"')


#: Bumped whenever `_extract` gains a field, so older extractions are redone rather than
#: silently missing it.
EXTRACT_SCHEMA = 2


def _task_path(task_id: str) -> Path:
    return EXTRACTED / f"{task_id}.json.gz"


def _extracted(task_id: str) -> bool:
    path = _task_path(task_id)
    if not path.exists():
        return False
    try:
        return _read_json(path).get("schema", 1) >= EXTRACT_SCHEMA
    except (OSError, ValueError, EOFError):
        return False


def _absent_path(task_id: str, bucket: str) -> Path:
    return EXTRACTED / f"{task_id}.absent.{bucket}"


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt") as fh:
        json.dump(obj, fh)
    tmp.rename(path)


def _read_json(path: Path):
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def _structure(struct: dict | None) -> dict | None:
    if not struct:
        return None
    lattice = struct["lattice"]["matrix"]
    return {
        "lattice": lattice,
        "species": [site["species"][0]["element"] for site in struct["sites"]],
        "xyz": [site["xyz"] if "xyz" in site else np.dot(site["abc"], lattice).tolist()
                for site in struct["sites"]],
    }


def _extract(doc: dict, key: str) -> dict:
    """What `match` needs from a full task document, and nothing heavier."""
    output = doc.get("output") or {}
    calcs = doc.get("calcs_reversed") or []
    last_calc = calcs[0] if calcs else {}
    steps = (last_calc.get("output") or {}).get("ionic_steps") or []
    last = steps[-1] if steps else {}
    calc_input = last_calc.get("input") or {}
    incar = calc_input.get("incar") or {}
    parameters = calc_input.get("parameters") or {}
    task = last_calc.get("task")
    electronic = last.get("electronic_steps") or []
    final_electronic = electronic[-1] if electronic else {}
    return {
        "schema": EXTRACT_SCHEMA,
        # Which energy term of the final ionic step is the task's reported energy: the
        # evidence that forces, stress and energy come out of one and the same SCF.
        "step_e_wo_entrp": last.get("e_wo_entrp"),
        "step_e_0_energy": last.get("e_0_energy"),
        "step_final_electronic": {k: final_electronic.get(k)
                                  for k in ("e_fr_energy", "e_wo_entrp", "e_0_energy")},
        "n_electronic_last_step": len(electronic),
        "NELM": incar.get("NELM", parameters.get("NELM")),
        "task_id": doc["task_id"], "key": key,
        "task_type": doc.get("calc_type") or doc.get("task_type"),
        "output_energy": output.get("energy"),
        "output_structure": _structure(output.get("structure")),
        "output_forces": output.get("forces") or None,
        "output_stress": output.get("stress") or None,
        "n_calcs": len(calcs),
        "last_calc_name": task.get("name") if isinstance(task, dict) else last_calc.get("task_name"),
        "last_calc_energy": (last_calc.get("output") or {}).get("energy"),
        "n_ionic_steps": len(steps),
        "step_forces": last.get("forces") or None,
        "step_stress": last.get("stress") or None,
        "step_structure": _structure(last.get("structure")),
        "step_e_fr_energy": last.get("e_fr_energy"),
        **{name: incar.get(name, parameters.get(name))
           for name in ("NSW", "IBRION", "ISIF", "EDIFFG", "EDIFF", "ISMEAR")},
    }


def _scan_file(key: str, needed: list[str], bucket: str) -> tuple[str, int, int, bool]:
    """Stream one gzipped JSONL file and extract the needed task documents from it."""
    import requests

    wanted = set(needed)
    found = 0
    for attempt in range(3):
        try:
            with requests.get(S3 + key, stream=True, timeout=600) as response:
                response.raise_for_status()
                with gzip.GzipFile(fileobj=response.raw) as fh:
                    for line in fh:
                        ids = {m.decode() for m in TASK_ID_RE.findall(line)}
                        if not ids & wanted:
                            continue
                        doc = json.loads(line)
                        task_id = doc.get("task_id")
                        if task_id in wanted:
                            _write_json(_task_path(task_id), _extract(doc, key))
                            wanted.discard(task_id)
                            found += 1
                            if not wanted:
                                break
            for task_id in wanted:
                _absent_path(task_id, bucket).touch()
            return key, found, len(wanted), True
        except (requests.RequestException, OSError, EOFError, ValueError) as error:
            logger.warning("%s attempt %d failed: %s", key, attempt + 1, error)
            time.sleep(5 * (attempt + 1))
    return key, found, len(wanted), False


def fetch(locations: pd.DataFrame, workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    EXTRACTED.mkdir(parents=True, exist_ok=True)
    for column, bucket in (("legacy_key", "legacy"), ("atomate2_key", "atomate2")):
        pending = locations[locations[column].notna()]
        keep = [not _extracted(t) and not _absent_path(t, bucket).exists()
                for t in pending["task_id"]]
        pending = pending[keep]
        by_key = pending.groupby(column)["task_id"].apply(list)
        logger.info("%s pass: %d tasks to extract from %d files", bucket, len(pending), len(by_key))
        if by_key.empty:
            continue
        failed = 0
        with ProcessPoolExecutor(workers) as pool:
            futures = [pool.submit(_scan_file, key, ids, bucket) for key, ids in by_key.items()]
            for index, future in enumerate(as_completed(futures), 1):
                key, found, left, complete = future.result()
                failed += not complete
                if index % 25 == 0 or index == len(futures) or left:
                    logger.info("%s files %d/%d (%s: found %d, not in file %d%s)", bucket, index,
                                len(futures), key.rsplit("/", 1)[-1], found, left,
                                "" if complete else ", DOWNLOAD FAILED")
        if failed:
            logger.warning("%s: %d files failed; rerun fetch to retry them", bucket, failed)


# --------------------------------------------------------------------------- match
def _min_image_dev(lattice: np.ndarray, reference: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    frac = (np.asarray(xyz) - np.asarray(reference)) @ np.linalg.inv(lattice)
    frac -= np.round(frac)
    return np.linalg.norm(frac @ lattice, axis=-1)


def _align(ref_species, ref_xyz, lattice, species, xyz):
    """Permutation taking task sites to LeMat's site order, and the largest deviation."""
    ref_species, species = np.asarray(ref_species), np.asarray(species)
    ref_xyz, xyz = np.asarray(ref_xyz, dtype=float), np.asarray(xyz, dtype=float)
    if len(ref_species) != len(species) or sorted(ref_species) != sorted(species):
        return None, np.inf
    identity = np.arange(len(species))
    if np.array_equal(ref_species, species):
        dev = _min_image_dev(lattice, ref_xyz, xyz)
        if dev.max() <= POS_TOL:
            return identity, float(dev.max())
    from scipy.optimize import linear_sum_assignment

    perm = np.empty(len(species), dtype=int)
    inverse = np.linalg.inv(lattice)
    for element in np.unique(ref_species):
        i = np.flatnonzero(ref_species == element)
        j = np.flatnonzero(species == element)
        frac = (xyz[j][None, :, :] - ref_xyz[i][:, None, :]) @ inverse
        frac -= np.round(frac)
        rows, cols = linear_sum_assignment(np.linalg.norm(frac @ lattice, axis=2))
        perm[i[rows]] = j[cols]
    return perm, float(_min_image_dev(lattice, ref_xyz, xyz[perm]).max())


def _max_abs_diff(a, b) -> float:
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return np.nan
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.abs(a - b).max()) if a.shape == b.shape else np.inf


def match(identified: pd.DataFrame) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    ids = identified["immutable_id"].tolist()
    table = ds.dataset(RAW).to_table(
        columns=["immutable_id", "energy", "lattice_vectors", "cartesian_site_positions",
                 "species_at_sites", "forces", "stress_tensor"],
        filter=pc.field("immutable_id").isin(ids))
    lemat = {row["immutable_id"]: row for row in table.to_pylist()}

    records = []
    for row in identified.itertuples(index=False):
        rec = {"immutable_id": row.immutable_id, "group": row.group, "task_id": row.task_id,
               "identified_by": row.identified_by, "time_match": row.time_match,
               "best_task_id": row.best_task_id,
               "calc_type": row.calc_type, "task_deprecated": row.task_deprecated,
               "material_deprecated": row.material_deprecated, "n_candidates": row.n_candidates,
               "n_energy_matches": row.n_energy_matches, "dE_api": row.best_dE}
        ref = lemat[row.immutable_id]
        status = None
        if pd.isna(row.task_id):
            status = ("id_not_found" if not row.material_found
                      else "no_pbe_task_with_energy" if row.n_candidates == 0
                      else "near_miss_energy")
        elif not _task_path(row.task_id).exists():
            status = "task_doc_unavailable"
        if status is None:
            doc = _read_json(_task_path(row.task_id))
            rec.update({k: doc[k] for k in ("key", "n_calcs", "last_calc_name", "n_ionic_steps",
                                             "NSW", "IBRION", "ISIF", "EDIFFG", "EDIFF", "ISMEAR")})
            out = doc["output_structure"]
            lattice = np.asarray(ref["lattice_vectors"], dtype=float)
            rec["dE_s3"] = (np.nan if doc["output_energy"] is None
                            else doc["output_energy"] - ref["energy"])
            rec["dlat"] = float(np.abs(np.asarray(out["lattice"]) - lattice).max())
            perm, rec["dpos"] = _align(ref["species_at_sites"], ref["cartesian_site_positions"],
                                       lattice, out["species"], out["xyz"])
            rec["site_order_same"] = perm is not None and bool(np.array_equal(perm, np.arange(len(perm))))
            rec["task_type"] = doc["task_type"]
            if perm is None:
                status = "near_miss_species"
            elif doc["output_energy"] is None or abs(rec["dE_s3"]) > ENERGY_TOL:
                status = "near_miss_energy"
            elif rec["dlat"] > LAT_TOL or rec["dpos"] > POS_TOL:
                status = "near_miss_geometry"
        if status is None:
            step = doc["step_structure"]
            if step is not None and step["species"] == out["species"]:
                step_lattice = np.asarray(out["lattice"], dtype=float)
                rec["step_dlat"] = float(np.abs(np.asarray(step["lattice"]) - step_lattice).max())
                rec["step_dpos"] = float(_min_image_dev(step_lattice, out["xyz"], step["xyz"]).max())
            step_ok = rec.get("step_dlat", np.inf) <= LAT_TOL and rec.get("step_dpos", np.inf) <= POS_TOL
            forces, forces_source = doc["output_forces"], "output"
            if not forces:
                forces, forces_source = (doc["step_forces"], "last_ionic_step") if doc["step_forces"] else (None, None)
            stress, stress_source = doc["output_stress"], "output"
            if not stress:
                stress, stress_source = (doc["step_stress"], "last_ionic_step") if doc["step_stress"] else (None, None)
            if "last_ionic_step" in (forces_source, stress_source) and not step_ok:
                status = "near_miss_step_geometry"
            else:
                if forces is not None:
                    forces = np.asarray(forces, dtype=float)[perm]
                    rec["max_abs_force"] = float(np.abs(forces).max())
                    rec["forces"] = forces.tolist()
                if stress is not None:
                    rec["stress"] = np.asarray(stress, dtype=float).tolist()
                    rec["max_abs_stress"] = float(np.abs(rec["stress"]).max())
                rec["forces_source"], rec["stress_source"] = forces_source, stress_source
                energy_terms = {
                    "last_calc_energy": doc.get("last_calc_energy"),
                    "step_e_fr_energy": doc.get("step_e_fr_energy"),
                    "step_e_wo_entrp": doc.get("step_e_wo_entrp"),
                    "step_e_0_energy": doc.get("step_e_0_energy"),
                    **{f"electronic_{k}": v
                       for k, v in (doc.get("step_final_electronic") or {}).items()}}
                gaps = {k: abs(v - doc["output_energy"]) for k, v in energy_terms.items()
                        if isinstance(v, (int, float)) and k != "last_calc_energy"}
                if gaps:
                    closest = min(gaps, key=gaps.get)
                    rec["step_energy_term"], rec["dE_output_vs_last_step"] = closest, gaps[closest]
                if isinstance(energy_terms["last_calc_energy"], (int, float)):
                    rec["dE_output_vs_last_calc"] = abs(
                        energy_terms["last_calc_energy"] - doc["output_energy"])
                nsw = pd.to_numeric(doc.get("NSW"), errors="coerce")
                rec["hit_nsw"] = bool(np.isfinite(nsw) and doc["n_ionic_steps"] >= nsw and nsw > 1)
                nelm = pd.to_numeric(doc.get("NELM"), errors="coerce")
                rec["hit_nelm"] = (bool(doc.get("n_electronic_last_step", 0) >= nelm)
                                   if np.isfinite(nelm) else None)
                status = ("exact_forces_stress" if forces is not None and stress is not None
                          else "exact_forces_only" if forces is not None
                          else "exact_stress_only" if stress is not None else "exact_no_forces")
                if row.group == "control":
                    rec["control_dF"] = _max_abs_diff(rec.get("forces"), ref["forces"])
                    rec["control_dS"] = _max_abs_diff(rec.get("stress"), ref["stress_tensor"])
                    if doc["step_forces"] and step_ok:
                        rec["control_dF_step"] = _max_abs_diff(
                            np.asarray(doc["step_forces"], dtype=float)[perm], ref["forces"])
                        rec["control_dS_step"] = _max_abs_diff(doc["step_stress"], ref["stress_tensor"])
        rec["status"] = status
        records.append(rec)
    frame = pd.DataFrame(records)
    if PROVENANCE.exists():
        provenance = pd.read_csv(PROVENANCE, usecols=["material_id", "theoretical"])
        frame = frame.merge(provenance.rename(columns={"material_id": "immutable_id"}),
                            on="immutable_id", how="left")
    return frame


def summarise(frame: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\nstatus x group\n", pd.crosstab(frame["status"], frame["group"], margins=True))
    print("\nstatus x identified_by\n", pd.crosstab(
        frame["status"], frame["identified_by"].fillna("unidentified"), margins=True))
    missing = frame[frame["group"] == "missing"]
    if "theoretical" in frame:
        print("\nmissing rows: status x theoretical\n",
              pd.crosstab(missing["status"], missing["theoretical"].map(
                  {True: "theoretical", False: "icsd"}).fillna("unknown"), margins=True))
    exact = frame["status"].str.startswith("exact", na=False)
    print("\nforces_source x group (exact)\n",
          pd.crosstab(frame.loc[exact, "forces_source"].fillna("none"), frame.loc[exact, "group"]))
    control = frame[(frame["group"] == "control") & exact]
    for column in ("control_dF", "control_dS", "control_dF_step", "control_dS_step"):
        if column in control:
            values = control[column]
            print(f"{column}: n={values.notna().sum()} max={values.max():.3g} "
                  f"== 0: {(values == 0).sum()} <= 1e-6: {(values <= 1e-6).sum()}")
    near = frame[frame["status"].str.startswith("near_miss", na=False)]
    for column in ("dE_api", "dlat", "dpos", "step_dlat", "step_dpos"):
        if column in near and near[column].notna().any():
            print(f"near-miss |{column}| quantiles:",
                  near[column].abs().quantile([0, 0.5, 0.9, 1]).round(6).to_dict())
    if "max_abs_force" in missing:
        print("\nrecovered max |F| component on missing rows:",
              missing["max_abs_force"].describe().round(4).to_dict())


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["select", "identify", "locate", "fetch", "match", "all"])
    parser.add_argument("--run", default="full",
                        help="Name of the run directory under cache/mp_forces_recovery/runs")
    parser.add_argument("--limit-missing", type=int, default=None)
    parser.add_argument("--n-control", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--material-batch", type=int, default=100)
    parser.add_argument("--task-batch", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6,
                        help="Processes streaming S3 files; each is mostly gzip + JSON parsing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run = WORK / "runs" / args.run
    run.mkdir(parents=True, exist_ok=True)
    stages = ["select", "identify", "locate", "fetch", "match"] if args.stage == "all" else [args.stage]

    if "select" in stages:
        select(args.n_control, args.limit_missing, args.seed).to_parquet(run / "targets.parquet")
    if "identify" in stages:
        rest = Rest(WORK / "api_cache")
        targets = pd.read_parquet(run / "targets.parquet")
        identify(rest, targets, args.material_batch, args.task_batch).to_parquet(run / "identify.parquet")
    if "locate" in stages:
        locate(pd.read_parquet(run / "identify.parquet")).to_parquet(run / "locations.parquet")
    if "fetch" in stages:
        fetch(pd.read_parquet(run / "locations.parquet"), args.workers)
    if "match" in stages:
        frame = match(pd.read_parquet(run / "identify.parquet"))
        frame.to_parquet(run / "results.parquet")
        summarise(frame)
        logger.info("wrote %s", run / "results.parquet")


if __name__ == "__main__":
    main()
