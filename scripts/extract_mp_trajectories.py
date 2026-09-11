#!/usr/bin/env python3
"""Full DFT relaxation trajectories for Materials Project rows of LeMat-Bulk.

Built on ``recover_mp_forces.py``: the same MP-API identification of the task
behind each LeMat row and the same public full task documents on
``s3://materialsproject-parsed``, but instead of the final ionic step this keeps
*every* calculation of the task (``relax1``, ``relax2``, ...) in chronological order
and *every* ionic step of each, so that a force-field estimate of how far a row sits
from its minimum can be checked against what DFT actually did next.

Targets (MP only, at most 40 sites, forces known):

``pilot``
    Every MP row of ``cache/gradient_matched/pilot/sample.parquet``.
``extra``
    Rows of ``lemat_bulk_fmax1`` train, stratified by max |stress| (kB) and, within a
    stratum, balanced across ISMEAR (read from the MP API before any download).
``recovered``
    Rows whose forces ``recover_mp_forces.py`` recovered from the final ionic step.

Output, under ``cache_root() / "mp_trajectories"``:

``tasks/<task_id>.json.gz``
    ``immutable_id, task_id, group, energy_key, lemat_site_order, calcs`` with each calc
    ``name, incar, kpoints, steps`` and each step ``lattice, species, frac, e_fr_energy,
    e_wo_entrp, e_0_energy, forces, stress, n_electronic``. Task site order, VASP units
    and signs as stored (A, eV, eV/A, kB). ``incar`` holds the INCAR value where the
    INCAR sets it, else vasprun's effective parameter.
``index.parquet``
    One row per target, rewritten as tasks land so consumers can start early.
``validation.parquet``
    The last step of the last calc against LeMat (or the recovered values), and the
    geometric continuity between consecutive calcs.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recover_mp_forces as rmf  # noqa: E402

from wyckoff_transformer.paths import cache_root  # noqa: E402

logger = logging.getLogger("extract_mp_trajectories")

PILOT_SAMPLE = Path("gradient_matched") / "pilot" / "sample.parquet"
STRESS_BINS = [-1e-9, 1, 5, 20, np.inf]
STRESS_LABELS = ["[0,1]", "(1,5]", "(5,20]", ">20"]
#: Final extra rows per stress bin; >20 kB is oversampled on purpose.
EXTRA_QUOTA = {"[0,1]": 120, "(1,5]": 120, "(5,20]": 160, ">20": 200}
#: Candidates identified per bin before ISMEAR balancing picks the quota.
EXTRA_OVERSAMPLE = 2.5
RECOVERED_PER_BIN = 25
MAX_SITES = 40
INCAR_KEYS = ("ISIF", "IBRION", "ISMEAR", "SIGMA", "ENCUT", "EDIFF", "EDIFFG", "NSW", "POTIM",
              "ISYM", "SYMPREC", "PREC", "LREAL", "ISPIN", "LDAU", "LDAUTYPE", "LDAUU", "LDAUL",
              "LDAUJ", "MAGMOM", "NELM", "NELMIN", "ALGO", "LASPH", "GGA", "METAGGA", "ADDGRID",
              "KSPACING", "LMAXMIX", "ISTART", "ICHARG")
FORCE_TOL = 1e-6
INDEX_COLUMNS = ["immutable_id", "task_id", "group", "stress_bin", "nsites", "n_calcs",
                 "calc_names", "steps_per_calc", "ISMEAR", "ISIF", "IBRION", "energy_key",
                 "path", "status", "chemical_formula_reduced", "elements"]
#: Settings whose change across a restart would move the energy surface itself (basis
#: size, FFT grids, k-mesh), as opposed to only restarting the plane-wave basis.
BASIS_KEYS = ("ENCUT", "PREC")
KPOINT_FIELDS = ("generation_style", "kpoints", "usershift")


def _out() -> Path:
    return cache_root() / "mp_trajectories"


def _stress_bin(values: pd.Series) -> pd.Series:
    return pd.cut(values, STRESS_BINS, labels=STRESS_LABELS).astype(str)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.rename(path)


# --------------------------------------------------------------------------- select
def _raw_mp_table() -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    table = ds.dataset(rmf.RAW).to_table(
        columns=["immutable_id", "energy", "nsites", "last_modified", "chemical_formula_reduced",
                 "forces", "stress_tensor"],
        filter=pc.starts_with(ds.field("immutable_id"), "mp-"))
    n_forces = pc.list_value_length(table.column("forces")).fill_null(0).to_numpy()
    frame = table.drop(["forces", "stress_tensor"]).to_pandas()
    stress = table.column("stress_tensor").to_pylist()
    frame["max_abs_stress"] = [np.abs(np.asarray(s, dtype=float)).max() if s else np.nan
                               for s in stress]
    frame["has_forces"] = n_forces > 0
    return frame


def select(seed: int, fmax1_split_dir: Path) -> pd.DataFrame:
    raw = _raw_mp_table()
    raw["stress_bin"] = _stress_bin(raw["max_abs_stress"])
    keep = ["immutable_id", "energy", "nsites", "last_modified", "chemical_formula_reduced",
            "stress_bin"]

    sample = pd.read_parquet(cache_root() / PILOT_SAMPLE, columns=["immutable_id", "source",
                                                                     "nsites", "stress_bin"])
    pilot_ids = sample.loc[(sample["source"] == "mp") & (sample["nsites"] <= MAX_SITES),
                           ["immutable_id", "stress_bin"]]
    pilot = raw[keep[:-1]].merge(pilot_ids, on="immutable_id").assign(group="pilot")

    recovered_results = pd.read_parquet(
        rmf.WORK / "runs" / "full" / "results.parquet",
        columns=["immutable_id", "group", "status", "max_abs_stress"])
    recovered_pool = recovered_results[(recovered_results["group"] == "missing")
                                       & (recovered_results["status"] == "exact_forces_stress")]
    recovered_pool = raw[keep[:-1]].merge(recovered_pool[["immutable_id", "max_abs_stress"]],
                                          on="immutable_id")
    recovered_pool = recovered_pool[recovered_pool["nsites"] <= MAX_SITES]
    recovered_pool["stress_bin"] = _stress_bin(recovered_pool["max_abs_stress"])
    recovered = (recovered_pool.groupby("stress_bin", group_keys=False)
                 .apply(lambda g: g.sample(min(RECOVERED_PER_BIN, len(g)), random_state=seed))
                 [keep].assign(group="recovered"))

    train_ids = set()
    for split_file in sorted(fmax1_split_dir.glob("train.csv.gz")):
        train_ids |= set(pd.read_csv(split_file, usecols=["immutable_id"])["immutable_id"])
    if not train_ids:
        raise SystemExit(f"no train.csv.gz under {fmax1_split_dir}")
    pool = raw[raw["has_forces"] & (raw["nsites"] <= MAX_SITES)
               & raw["immutable_id"].isin(train_ids)
               & ~raw["immutable_id"].isin(pilot["immutable_id"])]
    extra = []
    for label, quota in EXTRA_QUOTA.items():
        stratum = pool[pool["stress_bin"] == label]
        n = min(len(stratum), int(quota * EXTRA_OVERSAMPLE))
        extra.append(stratum.sample(n, random_state=seed))
    extra = pd.concat(extra)[keep].assign(group="extra_candidate")

    targets = pd.concat([pilot, extra, recovered], ignore_index=True)
    logger.info("targets: %s", targets.groupby(["group", "stress_bin"]).size().to_dict())
    return targets


# --------------------------------------------------------------------------- identify
def identify(targets: pd.DataFrame, seed: int) -> pd.DataFrame:
    rest = rmf.Rest(rmf.WORK / "api_cache")
    known = pd.read_parquet(rmf.WORK / "runs" / "full" / "identify.parquet")
    recovered = targets[targets["group"] == "recovered"]
    recovered_ids = recovered[["immutable_id"]].merge(
        known[["immutable_id", "task_id", "identified_by"]], on="immutable_id", how="left")

    others = targets[targets["group"] != "recovered"].copy()
    identified = rmf.identify(rest, others.assign(group=others["group"]), 200, 200)
    identified = identified[["immutable_id", "task_id", "identified_by"]]
    ids = pd.concat([identified, recovered_ids], ignore_index=True)
    frame = targets.merge(ids, on="immutable_id", how="left")

    # ISMEAR of each extra candidate's task, from the API, to balance before downloading.
    candidates = frame[(frame["group"] == "extra_candidate") & frame["task_id"].notna()]
    ismear = {}
    for batch in rmf._chunks(sorted(set(candidates["task_id"])), 200):
        payload = rest.get("/materials/tasks/", {
            "task_ids": ",".join(batch), "id_format": "legacy",
            "_fields": "task_id,input.incar.ISMEAR,input.parameters.ISMEAR", "_limit": len(batch)})
        for doc in payload["data"]:
            inp = doc.get("input") or {}
            value = (inp.get("incar") or {}).get("ISMEAR", (inp.get("parameters") or {}).get("ISMEAR"))
            ismear[doc["task_id"]] = value
    frame["api_ismear"] = frame["task_id"].map(ismear)

    chosen = []
    rng = np.random.default_rng(seed)
    for label, quota in EXTRA_QUOTA.items():
        stratum = frame[(frame["group"] == "extra_candidate") & (frame["stress_bin"] == label)
                        & frame["task_id"].notna()]
        category = stratum["api_ismear"].map(
            lambda v: "unknown" if pd.isna(v) else "tetrahedron" if v == -5
            else "gaussian" if v == 0 else "methfessel_paxton" if v > 0 else "other")
        pools = {c: list(rng.permutation(stratum.index[category == c]))
                 for c in ("tetrahedron", "gaussian", "methfessel_paxton", "other")}
        picked = []
        while len(picked) < quota and any(pools.values()):
            for c in list(pools):
                if pools[c] and len(picked) < quota:
                    picked.append(pools[c].pop())
        if len(picked) < quota:  # top up from rows whose ISMEAR the API did not say
            unknown = list(rng.permutation(stratum.index[category == "unknown"]))
            picked += unknown[:quota - len(picked)]
        chosen += picked
    frame.loc[chosen, "group"] = "extra"
    frame = frame[frame["group"] != "extra_candidate"].reset_index(drop=True)
    logger.info("final targets: %s", frame.groupby(["group", "stress_bin"]).size().to_dict())
    logger.info("extra ISMEAR (API): %s",
                frame.loc[frame["group"] == "extra", "api_ismear"].value_counts(dropna=False).to_dict())
    logger.info("unidentified: %d", int(frame["task_id"].isna().sum()))
    return frame


# --------------------------------------------------------------------------- fetch
def _structure_step(step: dict) -> dict:
    structure = step["structure"]
    electronic = step.get("electronic_steps")
    return {
        "lattice": structure["lattice"]["matrix"],
        "species": [site["species"][0]["element"] for site in structure["sites"]],
        "frac": [site["abc"] for site in structure["sites"]],
        "e_fr_energy": step.get("e_fr_energy"),
        "e_wo_entrp": step.get("e_wo_entrp"),
        "e_0_energy": step.get("e_0_energy"),
        "forces": step.get("forces"),
        "stress": step.get("stress"),
        "n_electronic": len(electronic) if electronic is not None else None,
    }


def _trajectory(doc: dict) -> list[dict]:
    calcs = []
    for calc in reversed(doc.get("calcs_reversed") or []):  # calcs_reversed is newest first
        task = calc.get("task")
        calc_input = calc.get("input") or {}
        incar = calc_input.get("incar") or {}
        parameters = calc_input.get("parameters") or {}
        calcs.append({
            "name": task.get("name") if isinstance(task, dict) else calc.get("task_name"),
            "incar": {k: incar[k] if k in incar else parameters[k]
                      for k in INCAR_KEYS if k in incar or k in parameters},
            "kpoints": calc_input.get("kpoints"),
            "steps": [_structure_step(s) for s in (calc.get("output") or {}).get("ionic_steps") or []],
        })
    return calcs


def _scan_file(key: str, needed: dict[str, dict], bucket: str, tasks_dir: str):
    """Stream one S3 JSONL file; write the trajectory of every needed task found in it."""
    import gzip
    import json

    import requests

    wanted = dict(needed)
    written = []
    for attempt in range(3):
        try:
            with requests.get(rmf.S3 + key, stream=True, timeout=600) as response:
                response.raise_for_status()
                with gzip.GzipFile(fileobj=response.raw) as fh:
                    for line in fh:
                        ids = {m.decode() for m in rmf.TASK_ID_RE.findall(line)}
                        if not ids & wanted.keys():
                            continue
                        doc = json.loads(line)
                        task_id = doc.get("task_id")
                        if task_id not in wanted:
                            continue
                        meta = wanted.pop(task_id)
                        record = {"immutable_id": meta["immutable_id"], "task_id": task_id,
                                  "group": meta["group"], "energy_key": None,
                                  "lemat_site_order": None, "calcs": _trajectory(doc),
                                  "source_key": key}
                        rmf._write_json(Path(tasks_dir) / f"{task_id}.json.gz", record)
                        written.append(task_id)
                        if not wanted:
                            break
            return key, written, sorted(wanted), True
        except (requests.RequestException, OSError, EOFError, ValueError) as error:
            logger.warning("%s attempt %d failed: %s", key, attempt + 1, error)
            time.sleep(5 * (attempt + 1))
    return key, written, sorted(wanted), False


# --------------------------------------------------------------------------- validate
def _references(targets: pd.DataFrame) -> dict[str, dict]:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    table = ds.dataset(rmf.RAW).to_table(
        columns=["immutable_id", "energy", "lattice_vectors", "cartesian_site_positions",
                 "species_at_sites", "forces", "stress_tensor"],
        filter=pc.field("immutable_id").isin(targets["immutable_id"].tolist()))
    refs = {row["immutable_id"]: row for row in table.to_pylist()}
    recovered = pd.read_parquet(rmf.WORK / "runs" / "full" / "results.parquet",
                                columns=["immutable_id", "forces", "stress"])
    recovered = recovered[recovered["immutable_id"].isin(
        targets.loc[targets["group"] == "recovered", "immutable_id"])]
    for row in recovered.itertuples(index=False):
        refs[row.immutable_id]["forces"] = [list(f) for f in row.forces]
        refs[row.immutable_id]["stress_tensor"] = [list(s) for s in row.stress]
    return refs


def _finalise(path: Path, ref: dict) -> tuple[dict, dict]:
    """Fill energy_key and lemat_site_order in the task file, and validate it."""
    record = rmf._read_json(path)
    calcs = record["calcs"]
    # The S3 file a task came from is provenance, not part of the task schema: it lives in
    # validation.parquet only. (``get`` rather than ``pop`` keeps a re-settle idempotent.)
    source_key = record.pop("source_key", None)
    val = {"immutable_id": record["immutable_id"], "task_id": record["task_id"],
           "group": record["group"], "source_key": source_key}
    status = "ok"
    last_calc = calcs[-1] if calcs else None
    last = last_calc["steps"][-1] if last_calc and last_calc["steps"] else None
    if last is None:
        return record, {**val, "status": "no_ionic_steps"}

    lattice = np.asarray(last["lattice"], dtype=float)
    xyz = np.asarray(last["frac"], dtype=float) @ lattice
    ref_lattice = np.asarray(ref["lattice_vectors"], dtype=float)
    perm, dpos = rmf._align(ref["species_at_sites"], ref["cartesian_site_positions"], ref_lattice,
                            last["species"], xyz)
    val["dlat"] = float(np.abs(lattice - ref_lattice).max())
    val["dpos"] = dpos
    gaps = {k: abs(last[k] - ref["energy"]) for k in ("e_fr_energy", "e_wo_entrp")
            if isinstance(last.get(k), (int, float))}
    energy_key = min(gaps, key=gaps.get) if gaps else None
    val["dE"] = gaps.get(energy_key, np.nan) if energy_key else np.nan
    if perm is not None:
        record["lemat_site_order"] = [int(p) for p in perm]
        forces = np.asarray(last["forces"], dtype=float)[perm] if last["forces"] else None
        val["dF"] = rmf._max_abs_diff(forces, ref["forces"])
    else:
        val["dF"] = np.nan
    val["dS"] = rmf._max_abs_diff(last["stress"], ref["stress_tensor"])
    record["energy_key"] = energy_key
    reasons = [name for name, failed in (
        ("site_alignment", perm is None), ("lattice", val["dlat"] > rmf.LAT_TOL),
        ("positions", dpos > rmf.POS_TOL),
        ("energy_absent" if energy_key is None else "energy", not val["dE"] <= rmf.ENERGY_TOL),
        ("forces", not val["dF"] <= FORCE_TOL), ("stress", not val["dS"] <= FORCE_TOL)) if failed]
    val["mismatch_reason"] = ",".join(reasons) or None
    if reasons:
        status = "validation_mismatch"

    # Continuity between consecutive calcs: last step of calc k against first step of k+1.
    transitions = []
    changed = {"encut": [], "prec": [], "kpoints": [], "incar_keys": []}
    for before, after in zip(calcs[:-1], calcs[1:]):
        inc_a, inc_b = before["incar"], after["incar"]
        changed["encut"].append(inc_a.get("ENCUT") != inc_b.get("ENCUT"))
        changed["prec"].append(str(inc_a.get("PREC")).lower() != str(inc_b.get("PREC")).lower())
        kp_a, kp_b = before.get("kpoints") or {}, after.get("kpoints") or {}
        changed["kpoints"].append(any(kp_a.get(f) != kp_b.get(f) for f in KPOINT_FIELDS))
        changed["incar_keys"].append(",".join(sorted(
            k for k in set(inc_a) | set(inc_b) if k != "MAGMOM" and inc_a.get(k) != inc_b.get(k))))
        if not before["steps"] or not after["steps"]:
            transitions.append((before["name"], after["name"], np.nan, np.nan, np.nan))
            continue
        a, b = before["steps"][-1], after["steps"][0]
        la, lb = np.asarray(a["lattice"], dtype=float), np.asarray(b["lattice"], dtype=float)
        if a["species"] != b["species"]:
            transitions.append((before["name"], after["name"], np.inf, np.inf, np.nan))
            continue
        frac = np.asarray(b["frac"], dtype=float) - np.asarray(a["frac"], dtype=float)
        frac -= np.round(frac)
        de = (b["e_fr_energy"] - a["e_fr_energy"]) if (
            isinstance(a.get("e_fr_energy"), (int, float))
            and isinstance(b.get("e_fr_energy"), (int, float))) else np.nan
        transitions.append((before["name"], after["name"], float(np.abs(la - lb).max()),
                            float(np.linalg.norm(frac @ la, axis=1).max()), de))
    val["transitions"] = [f"{t[0]}->{t[1]}" for t in transitions]
    val["transition_dlat"] = [t[2] for t in transitions]
    val["transition_dpos"] = [t[3] for t in transitions]
    val["transition_dE_fr"] = [t[4] for t in transitions]
    val["transition_encut_changed"] = changed["encut"]
    val["transition_prec_changed"] = changed["prec"]
    val["transition_kpoints_changed"] = changed["kpoints"]
    val["transition_incar_changed_keys"] = changed["incar_keys"]
    val["nsites"] = len(last["species"])
    val["n_steps_total"] = sum(len(c["steps"]) for c in calcs)
    val["status"] = status
    rmf._write_json(path, record)
    return record, val


def _index_row(target, record: dict | None, path: Path | None, status: str) -> dict:
    row = {"immutable_id": target.immutable_id, "task_id": target.task_id, "group": target.group,
           "stress_bin": target.stress_bin, "nsites": int(target.nsites), "n_calcs": None,
           "calc_names": None, "steps_per_calc": None, "ISMEAR": None, "ISIF": None,
           "IBRION": None, "energy_key": None, "path": None if path is None else str(path),
           "status": status, "chemical_formula_reduced": target.chemical_formula_reduced,
           "elements": None}
    if record is not None:
        calcs = record["calcs"]
        species = calcs[-1]["steps"][-1]["species"] if calcs and calcs[-1]["steps"] else []
        row["elements"] = sorted(set(species))
        incar = calcs[-1]["incar"] if calcs else {}
        row.update({"n_calcs": len(calcs), "calc_names": [c["name"] for c in calcs],
                    "steps_per_calc": [len(c["steps"]) for c in calcs],
                    "ISMEAR": incar.get("ISMEAR"), "ISIF": incar.get("ISIF"),
                    "IBRION": incar.get("IBRION"), "energy_key": record["energy_key"]})
    return row


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ("ISMEAR", "ISIF", "IBRION", "n_calcs"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def fetch(targets: pd.DataFrame, locations: pd.DataFrame, workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out = _out()
    tasks_dir = out / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    refs = _references(targets)
    targets = targets.merge(locations, on="task_id", how="left")
    by_task = {row.task_id: row for row in targets.itertuples(index=False) if pd.notna(row.task_id)}
    index = {row.immutable_id: _index_row(row, None, None,
                                          "unidentified" if pd.isna(row.task_id) else "pending")
             for row in targets.itertuples(index=False)}
    validation = {}
    last_write = [0.0]

    def settle(task_id: str) -> None:
        target = by_task[task_id]
        path = tasks_dir / f"{task_id}.json.gz"
        record, val = _finalise(path, refs[target.immutable_id])
        validation[target.immutable_id] = val
        index[target.immutable_id] = _index_row(target, record, path, val["status"])

    def flush(force: bool = False) -> None:
        if not force and time.time() - last_write[0] < 20:
            return
        _write_parquet(_numeric(pd.DataFrame(list(index.values()), columns=INDEX_COLUMNS)),
                       out / "index.parquet")
        if validation:
            _write_parquet(pd.DataFrame(list(validation.values())), out / "validation.parquet")
        last_write[0] = time.time()

    for task_id in by_task:  # resume: tasks already on disk
        if (tasks_dir / f"{task_id}.json.gz").exists():
            settle(task_id)
    flush(force=True)

    for column, bucket in (("legacy_key", "legacy"), ("atomate2_key", "atomate2")):
        pending = [t for t in by_task.values()
                   if index[t.immutable_id]["status"] == "pending" and pd.notna(getattr(t, column))]
        by_key: dict[str, dict] = {}
        for t in pending:
            by_key.setdefault(getattr(t, column), {})[t.task_id] = {
                "immutable_id": t.immutable_id, "group": t.group}
        logger.info("%s pass: %d tasks in %d files", bucket, len(pending), len(by_key))
        if not by_key:
            continue
        with ProcessPoolExecutor(workers) as pool:
            futures = [pool.submit(_scan_file, key, needed, bucket, str(tasks_dir))
                       for key, needed in by_key.items()]
            for n, future in enumerate(as_completed(futures), 1):
                key, written, absent, complete = future.result()
                for task_id in written:
                    settle(task_id)
                if not complete:
                    logger.warning("%s: download failed, %d tasks left pending", key, len(absent))
                if n % 20 == 0 or n == len(futures):
                    logger.info("%s files %d/%d", bucket, n, len(futures))
                flush()
    for row in index.values():
        if row["status"] == "pending":
            row["status"] = "doc_not_found"
    flush(force=True)


def summarise() -> None:
    out = _out()
    pd.set_option("display.width", 220)
    index = pd.read_parquet(out / "index.parquet")
    print("\nstatus x group\n", pd.crosstab(index["status"], index["group"], margins=True))
    print("\ngroup x stress_bin (ok)\n",
          pd.crosstab(index.loc[index.status == "ok", "group"],
                      index.loc[index.status == "ok", "stress_bin"], margins=True))
    ok = index[index["status"] == "ok"]
    print("\nISMEAR x group (ok)\n", pd.crosstab(ok["ISMEAR"], ok["group"], margins=True))
    print("calc name sequences:", ok["calc_names"].map(lambda v: "->".join(map(str, v))).value_counts().head(8).to_dict())
    total = ok["steps_per_calc"].map(sum)
    print("total ionic steps per task:", total.describe().round(1).to_dict())
    for sequence, g in ok.groupby(ok["calc_names"].map(lambda v: "->".join(map(str, v)))):
        per_calc = np.array([list(s) for s in g["steps_per_calc"]], dtype=float)
        print(f"steps per calc for {sequence} (n={len(g)}): median {np.median(per_calc, axis=0).tolist()}, "
              f"p90 {np.percentile(per_calc, 90, axis=0).tolist()}, max {per_calc.max(axis=0).tolist()}")
    print("total steps by group (median, p90, max):",
          {k: (float(v.median()), float(v.quantile(0.9)), int(v.max()))
           for k, v in ok.assign(total=total).groupby("group")["total"]})
    print("energy_key:", ok["energy_key"].value_counts().to_dict())
    val = pd.read_parquet(out / "validation.parquet")
    for column in ("dlat", "dpos", "dE", "dF", "dS"):
        print(f"validation max {column}: {val[column].max():.3g}")
    print("mismatch reasons:", val["mismatch_reason"].value_counts().to_dict())
    columns = ["transitions", "transition_dlat", "transition_dpos", "transition_dE_fr",
               "transition_encut_changed", "transition_prec_changed", "transition_kpoints_changed",
               "transition_incar_changed_keys"]
    flat = val.explode(columns)
    flat = flat[flat["transitions"].notna()].merge(
        index[["immutable_id", "elements"]], on="immutable_id", how="left")
    for name, g in flat.groupby("transitions"):
        dl, dp = g["transition_dlat"].astype(float), g["transition_dpos"].astype(float)
        print(f"continuity {name}: n={len(g)} identical (0) {(np.maximum(dl, dp) == 0).sum()} "
              f"<=1e-6 {(np.maximum(dl, dp) <= 1e-6).sum()} max dlat {dl.max():.3g} max dpos {dp.max():.3g} "
              f"median |dE_fr| {g['transition_dE_fr'].astype(float).abs().median():.4g}")
        for flag in ("encut", "prec", "kpoints"):
            print(f"  {flag} changed: {int(g[f'transition_{flag}_changed'].astype(bool).sum())}/{len(g)}")
        print("  other INCAR keys changed:",
              g["transition_incar_changed_keys"].replace("", "none").value_counts().head(8).to_dict())
        per_atom = g["transition_dE_fr"].astype(float) / g["nsites"].astype(float)
        chemistry = g["elements"].map(lambda e: "F" if "F" in e else "O/N" if ({"O", "N"} & set(e))
                                      else "no F/O/N")
        print("  restart dE_fr per atom (eV) by chemistry [median, p10, p90, frac>0]:",
              {k: [round(float(v.median()), 5), round(float(v.quantile(.1)), 5),
                   round(float(v.quantile(.9)), 5), round(float((v > 0).mean()), 2)]
               for k, v in per_atom.groupby(chemistry)})


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["select", "identify", "locate", "fetch", "summary", "all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--fmax1-split-dir", type=Path, default=Path("data/lemat_bulk_fmax1"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = _out()
    out.mkdir(parents=True, exist_ok=True)
    stages = (["select", "identify", "locate", "fetch", "summary"] if args.stage == "all"
              else [args.stage])
    if "select" in stages:
        from wyckoff_transformer.paths import resolve_store_path

        _write_parquet(select(args.seed, resolve_store_path(args.fmax1_split_dir)),
                       out / "candidates.parquet")
    if "identify" in stages:
        _write_parquet(identify(pd.read_parquet(out / "candidates.parquet"), args.seed),
                       out / "targets.parquet")
    if "locate" in stages:
        _write_parquet(rmf.locate(pd.read_parquet(out / "targets.parquet")),
                       out / "locations.parquet")
    if "fetch" in stages:
        fetch(pd.read_parquet(out / "targets.parquet"), pd.read_parquet(out / "locations.parquet"),
              args.workers)
    if "summary" in stages:
        summarise()


if __name__ == "__main__":
    main()
