#!/usr/bin/env python3
"""Alexandria PBE geometry-optimisation paths, in the ``mp_trajectories`` schema.

Source: ``https://alexandria.icams.rub.de/data/geo_opt_paths/2025.07.02/pbe/pbe_XXXX.json.bz2``,
454 shards of one JSON object ``{agm_id: [run, ...]}`` in no id or chemistry order (there is
no index; the first keys of shards 0000/0001/0150/0300/0452 are unrelated ids). Each run is
``{"kpoints": [n1, n2, n3], "PREC", "ENMAX", "ENAUG", "steps": [{"structure", "energy",
"forces", "stress"}]}``; runs are chronological (the last geometry of run k is the first of
run k+1).

LeMat-Bulk's Alexandria rows are *not* read from these paths: its energy, forces and stress
equal, bit for bit, the entries of Alexandria's complete database
(``data/pbe/2025.07.02/alexandria_*.json.bz2``: ``ComputedStructureEntry.energy``,
site ``forces``, ``data.stress``), which carry no ENMAX/PREC/k-points. `validate` asks which
path step, if any, reproduces that entry, and what separates the path's final step from it.

Stages (each writes under ``cache_root() / "alexandria_trajectories"``):

``keys``      stream every downloaded shard once, record its ids (``shard_keys.parquet``)
``select``    every pilot id found, plus eligible ids stratified by stress bin x hardness
``extract``   stream the shards again, write ``tasks/<agm_id>.json.gz`` for selected ids
``validate``  ``index.parquet`` and ``validation.parquet``
``summary``   print the answers

Task files follow ``mp_trajectories``: ``calcs[i] = {name: "run<i>", incar: {ENMAX, ENAUG,
PREC, ...}, kpoints: [n1, n2, n3], steps: [...]}``. The dump has one energy per step; it is
stored as ``energy`` (with ``energy_key: "energy"``) and ``e_fr_energy``/``e_wo_entrp``/
``e_0_energy``/``n_electronic`` are null. Units as stored: A, eV, eV/A, kB.
"""
from __future__ import annotations

import argparse
import bz2
import json
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recover_mp_forces as rmf  # noqa: E402

from wyckoff_transformer.paths import cache_root  # noqa: E402

logger = logging.getLogger("extract_alexandria_trajectories")

KEY_RE = re.compile(r'"(agm\d+)": \[\{')
CHUNK = 1 << 24
TARGET_TOTAL = 500
STRESS_LABELS = ["[0,1]", "(1,5]", "(5,20]", ">20"]
HARDNESS = ["F", "O/N", "neither"]
MATCH_TOL = {"energy": 1e-4, "forces": 1e-6, "stress": 1e-6}
INDEX_COLUMNS = ["immutable_id", "task_id", "group", "stress_bin", "nsites", "n_calcs",
                 "calc_names", "steps_per_calc", "ISMEAR", "ISIF", "IBRION", "energy_key",
                 "path", "status", "chemical_formula_reduced", "elements", "hardness",
                 "ENMAX", "ENAUG", "PREC", "kpoints"]


def _out() -> Path:
    return cache_root() / "alexandria_trajectories"


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.rename(path)


def _iter_text(path: Path):
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                return
            yield chunk


# --------------------------------------------------------------------------- keys
def _shard_keys(path: str) -> tuple[str, list[str]]:
    keys, tail = [], ""
    for chunk in _iter_text(Path(path)):
        text = tail + chunk
        keys += KEY_RE.findall(text)
        tail = text[-40:]  # a key split across chunks is re-found here; dedupe below
    return Path(path).name, list(dict.fromkeys(keys))


def keys_stage(workers: int) -> pd.DataFrame:
    shards = sorted((_out() / "shards").glob("pbe_*.json.bz2"))
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(_shard_keys, str(p)) for p in shards]):
            name, keys = future.result()
            logger.info("%s: %d ids", name, len(keys))
            rows += [(name, k) for k in keys]
    return pd.DataFrame(rows, columns=["shard", "immutable_id"])


# --------------------------------------------------------------------------- select
def select_stage(keys: pd.DataFrame, seed: int) -> pd.DataFrame:
    eligible = pd.read_parquet(_out() / "eligible.parquet")
    pool = eligible.merge(keys.drop_duplicates("immutable_id"), on="immutable_id")
    logger.info("ids in shards: %d; eligible: %d; pilot: %d", keys["immutable_id"].nunique(),
                len(pool), int(pool["pilot"].sum()))
    chosen = pool[pool["pilot"]].assign(group="pilot")
    rest = pool[~pool["pilot"]]
    budget = TARGET_TOTAL - len(chosen)
    cells = {(s, h): rest[(rest["stress_bin"] == s) & (rest["hardness"] == h)]
             for s in STRESS_LABELS for h in HARDNESS}
    quota = {c: 0 for c in cells}
    while budget > 0:  # round-robin: equal cells, spill over from the ones that run dry
        progressed = False
        for c, frame in cells.items():
            if budget and quota[c] < len(frame):
                quota[c] += 1
                budget -= 1
                progressed = True
        if not progressed:
            break
    picked = [frame.sample(quota[c], random_state=seed) for c, frame in cells.items() if quota[c]]
    chosen = pd.concat([chosen, *[p.assign(group="stratified") for p in picked]], ignore_index=True)
    logger.info("selected:\n%s", pd.crosstab([chosen["group"], chosen["stress_bin"]],
                                             chosen["hardness"], margins=True))
    return chosen


# --------------------------------------------------------------------------- extract
def _task(agm: str, runs: list[dict], group: str) -> dict:
    calcs = []
    for i, run in enumerate(runs):
        steps = []
        for step in run.get("steps") or []:
            structure = step["structure"]
            steps.append({
                "lattice": structure["lattice"]["matrix"],
                "species": [site["species"][0]["element"] for site in structure["sites"]],
                "frac": [site["abc"] for site in structure["sites"]],
                "energy": step.get("energy"),
                "e_fr_energy": None, "e_wo_entrp": None, "e_0_energy": None,
                "forces": step.get("forces"),
                "stress": step.get("stress"),
                "n_electronic": None,
            })
        calcs.append({"name": f"run{i}",
                      "incar": {k: v for k, v in run.items() if k not in ("steps", "kpoints")},
                      "kpoints": run.get("kpoints"), "steps": steps})
    return {"immutable_id": agm, "task_id": agm, "group": group, "energy_key": "energy",
            "lemat_site_order": None, "calcs": calcs}


def _extract_shard(path: str, wanted: dict[str, str], tasks_dir: str) -> tuple[str, list[str]]:
    decoder = json.JSONDecoder()
    written, buf, pos = [], "", 0

    def emit(key: str, start: int) -> None:
        value, _ = decoder.raw_decode(buf, start)
        rmf._write_json(Path(tasks_dir) / f"{key}.json.gz", _task(key, value, wanted[key]))
        written.append(key)

    for chunk in _iter_text(Path(path)):
        buf = buf[pos:] + chunk
        pos = 0
        while True:
            match = KEY_RE.search(buf, pos)
            if match is None:
                pos = max(pos, len(buf) - 40)
                break
            following = KEY_RE.search(buf, match.end())
            if following is None:  # this value may not be complete yet
                pos = match.start()
                break
            if match.group(1) in wanted:
                emit(match.group(1), match.end() - 2)
            pos = following.start()
    match = KEY_RE.search(buf, pos)
    if match is not None and match.group(1) in wanted:  # the last value in the file
        emit(match.group(1), match.end() - 2)
    return Path(path).name, written


def extract_stage(targets: pd.DataFrame, workers: int) -> None:
    tasks_dir = _out() / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    todo = targets[[not (tasks_dir / f"{i}.json.gz").exists() for i in targets["immutable_id"]]]
    by_shard = {shard: dict(zip(g["immutable_id"], g["group"])) for shard, g in todo.groupby("shard")}
    logger.info("extracting %d ids from %d shards", len(todo), len(by_shard))
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(_extract_shard, str(_out() / "shards" / shard), wanted, str(tasks_dir))
                   for shard, wanted in by_shard.items()]
        for future in as_completed(futures):
            name, written = future.result()
            logger.info("%s: wrote %d", name, len(written))


# --------------------------------------------------------------------------- validate
def _cart(step: dict) -> np.ndarray:
    return np.asarray(step["frac"], dtype=float) @ np.asarray(step["lattice"], dtype=float)


def _hydro(stress) -> float:
    return float(np.trace(np.asarray(stress, dtype=float)) / 3) if stress is not None else np.nan


def _volume(step: dict) -> float:
    return float(abs(np.linalg.det(np.asarray(step["lattice"], dtype=float))))


def _compare(step: dict, ref: dict) -> dict:
    lattice = np.asarray(ref["lattice_vectors"], dtype=float)
    perm, dpos = rmf._align(ref["species_at_sites"], ref["cartesian_site_positions"], lattice,
                            step["species"], _cart(step))
    out = {"dlat": float(np.abs(np.asarray(step["lattice"]) - lattice).max()), "dpos": dpos,
           "dE": abs(step["energy"] - ref["energy"]) if step.get("energy") is not None else np.nan,
           "dS": rmf._max_abs_diff(step["stress"], ref["stress_tensor"]),
           "dS_flipped": rmf._max_abs_diff(-np.asarray(step["stress"], dtype=float), ref["stress_tensor"])
           if step["stress"] is not None else np.nan,
           "dF": rmf._max_abs_diff(np.asarray(step["forces"], dtype=float)[perm], ref["forces"])
           if perm is not None and step["forces"] is not None else np.nan}
    return out, perm


def validate_stage(targets: pd.DataFrame) -> None:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    out = _out()
    refs = {r["immutable_id"]: r for r in ds.dataset(rmf.RAW).to_table(
        columns=["immutable_id", "energy", "lattice_vectors", "cartesian_site_positions",
                 "species_at_sites", "forces", "stress_tensor", "nsites"],
        filter=pc.field("immutable_id").isin(targets["immutable_id"].tolist())).to_pylist()}
    index, validation = [], []
    for t in targets.itertuples(index=False):
        path = out / "tasks" / f"{t.immutable_id}.json.gz"
        row = {c: None for c in INDEX_COLUMNS}
        row.update({"immutable_id": t.immutable_id, "task_id": t.immutable_id, "group": t.group,
                    "stress_bin": t.stress_bin, "nsites": int(t.nsites), "energy_key": "energy",
                    "chemical_formula_reduced": t.chemical_formula_reduced,
                    "elements": list(t.elements), "hardness": t.hardness})
        if not path.exists():
            row["status"] = "not_extracted"
            index.append(row)
            continue
        record = rmf._read_json(path)
        calcs, ref = record["calcs"], refs[t.immutable_id]
        steps = [(i, j, s) for i, c in enumerate(calcs) for j, s in enumerate(c["steps"])]
        last_i, last_j, last = steps[-1]
        val = {"immutable_id": t.immutable_id, "group": t.group, "hardness": t.hardness,
               "stress_bin": t.stress_bin, "nsites": int(t.nsites), "shard": t.shard,
               "n_runs": len(calcs), "n_steps_total": len(steps)}

        # Which path step, if any, is LeMat's calculation?
        best = min(steps, key=lambda s: abs(s[2]["energy"] - ref["energy"])
                   if s[2].get("energy") is not None else np.inf)
        cmp_best, perm_best = _compare(best[2], ref)
        val.update({f"best_{k}": v for k, v in cmp_best.items()})
        val["best_run"], val["best_step"] = best[0], best[1]
        val["best_is_last_step_of_last_run"] = (best[0], best[1]) == (last_i, last_j)
        val["lemat_step_match"] = bool(cmp_best["dE"] <= MATCH_TOL["energy"]
                                       and cmp_best["dF"] <= MATCH_TOL["forces"]
                                       and cmp_best["dS"] <= MATCH_TOL["stress"]
                                       and cmp_best["dlat"] <= rmf.LAT_TOL and cmp_best["dpos"] <= rmf.POS_TOL)

        # LeMat against where the path ended.
        cmp_last, perm_last = _compare(last, ref)
        val.update({f"final_{k}": v for k, v in cmp_last.items()})
        val["P_lemat"] = _hydro(ref["stress_tensor"])
        val["P_final"] = _hydro(last["stress"])
        val["P_jump_lemat_minus_final"] = val["P_lemat"] - val["P_final"]
        val["E_lemat_minus_final_per_atom"] = (ref["energy"] - last["energy"]) / len(last["species"])
        val["V_lemat_over_final"] = abs(np.linalg.det(np.asarray(ref["lattice_vectors"], dtype=float))) / _volume(last)
        first = steps[0][2]
        val["V_final_over_initial"] = _volume(last) / _volume(first)
        record["lemat_site_order"] = [int(p) for p in (perm_last if perm_last is not None else [])] or None

        # Settings per run and the restart jumps inside the path itself.
        settings = [(c["incar"].get("ENMAX"), c["incar"].get("ENAUG"), c["incar"].get("PREC"),
                     tuple(c["kpoints"] or ())) for c in calcs]
        val["settings_identical_across_runs"] = len(set(settings)) == 1
        val["run_setting_keys"] = sorted({k for c in calcs for k in c["incar"]})
        jumps = []
        for a, b in zip(calcs[:-1], calcs[1:]):
            if not a["steps"] or not b["steps"]:
                continue
            sa, sb = a["steps"][-1], b["steps"][0]
            same = sa["species"] == sb["species"]
            dlat = float(np.abs(np.asarray(sa["lattice"]) - np.asarray(sb["lattice"])).max())
            frac = np.asarray(sb["frac"], dtype=float) - np.asarray(sa["frac"], dtype=float)
            frac -= np.round(frac)
            dpos = float(np.linalg.norm(frac @ np.asarray(sa["lattice"], dtype=float), axis=1).max()) if same else np.inf
            jumps.append({"transition": f"{a['name']}->{b['name']}", "dlat": dlat, "dpos": dpos,
                          "dP": _hydro(sb["stress"]) - _hydro(sa["stress"]),
                          "dE_per_atom": (sb["energy"] - sa["energy"]) / len(sb["species"]),
                          "V_ratio_run": _volume(sa) / _volume(a["steps"][0]),
                          "settings_changed": settings[calcs.index(a)] != settings[calcs.index(b)]})
        val["restart_jumps"] = json.dumps(jumps)

        # Stress sign convention: does the hydrostatic stress at a step predict the next volume change?
        agree = total = 0
        for c in calcs:
            for sa, sb in zip(c["steps"][:-1], c["steps"][1:]):
                p, dv = _hydro(sa["stress"]), _volume(sb) - _volume(sa)
                if abs(p) > 2 and abs(dv) > 1e-3:
                    total += 1
                    agree += np.sign(p) == np.sign(dv)
        val["sign_test_agree"], val["sign_test_total"] = int(agree), int(total)

        val["status"] = "lemat_step_match" if val["lemat_step_match"] else "no_step_matches_lemat"
        rmf._write_json(path, record)
        last_calc = calcs[-1]
        row.update({"n_calcs": len(calcs), "calc_names": [c["name"] for c in calcs],
                    "steps_per_calc": [len(c["steps"]) for c in calcs], "path": str(path),
                    "status": val["status"], "ENMAX": last_calc["incar"].get("ENMAX"),
                    "ENAUG": last_calc["incar"].get("ENAUG"), "PREC": last_calc["incar"].get("PREC"),
                    "kpoints": last_calc["kpoints"]})
        index.append(row)
        validation.append(val)
    _write_parquet(pd.DataFrame(index, columns=INDEX_COLUMNS), out / "index.parquet")
    _write_parquet(pd.DataFrame(validation), out / "validation.parquet")


# --------------------------------------------------------------------------- summary
def summary_stage() -> None:
    out = _out()
    pd.set_option("display.width", 220)
    index = pd.read_parquet(out / "index.parquet")
    val = pd.read_parquet(out / "validation.parquet")
    print("\nstatus x group\n", pd.crosstab(index["status"], index["group"], margins=True))
    print("\nstress_bin x hardness (extracted)\n",
          pd.crosstab(val["stress_bin"], val["hardness"], margins=True))
    print("runs:", val["n_runs"].value_counts().to_dict(), "| total steps:",
          val["n_steps_total"].describe()[["50%", "mean", "max"]].round(1).to_dict())
    print("steps per run position:", {i: pd.Series([s[i] for s in index["steps_per_calc"].dropna() if len(s) > i]).median() for i in range(4)})
    print("run setting keys:", val["run_setting_keys"].map(tuple).value_counts().to_dict())
    print("settings identical across runs:", val["settings_identical_across_runs"].value_counts().to_dict())
    print("PREC:", index["PREC"].value_counts().to_dict())
    print(f"sign test (P>0 -> volume grows next step): {val['sign_test_agree'].sum()}/{val['sign_test_total'].sum()}")
    print("\nbest-matching step: lemat_step_match", val["lemat_step_match"].sum(), "of", len(val),
          "| best is last step of last run:", val["best_is_last_step_of_last_run"].sum())
    for column in ("best_dE", "best_dF", "best_dS", "best_dlat", "best_dpos"):
        print(f"  {column}: median {val[column].median():.3g}, max {val[column].max():.3g}")
    print("final step vs LeMat: median |dE| %.4g eV, dlat %.3g, dpos %.3g, dF %.3g, dS %.3g, dS_flipped %.3g" % tuple(
        val[c].median() for c in ("final_dE", "final_dlat", "final_dpos", "final_dF", "final_dS", "final_dS_flipped")))
    print("  geometry identical to final step (dlat, dpos <= 1e-4):",
          int(((val["final_dlat"] <= 1e-4) & (val["final_dpos"] <= 1e-4)).sum()), "of", len(val))
    print("  V_lemat / V_final: median %.6f, p10 %.6f, p90 %.6f" % tuple(val["V_lemat_over_final"].quantile([.5, .1, .9])))
    g = val.groupby("hardness")
    print("\nLeMat minus path-final hydrostatic stress (kB) by hardness [median, p10, p90, frac>0, n]:")
    print({h: [round(v.median(), 2), round(v.quantile(.1), 2), round(v.quantile(.9), 2), round((v > 0).mean(), 2), len(v)]
           for h, v in g["P_jump_lemat_minus_final"]})
    print("P_lemat by hardness [median, frac>0]:", {h: [round(v.median(), 2), round((v > 0).mean(), 2)] for h, v in g["P_lemat"]})
    print("P_final by hardness [median, frac>0]:", {h: [round(v.median(), 2), round((v > 0).mean(), 2)] for h, v in g["P_final"]})
    print("E_lemat - E_final per atom (meV) by hardness [median, p10, p90]:",
          {h: [round(1e3 * v.median(), 3), round(1e3 * v.quantile(.1), 3), round(1e3 * v.quantile(.9), 3)] for h, v in g["E_lemat_minus_final_per_atom"]})
    bins = pd.cut(val["V_final_over_initial"], [0, 0.95, 0.99, 1.01, 1.05, np.inf])
    print("P jump by relaxation volume change V_final/V_initial [median, n]:",
          {str(b): [round(v.median(), 2), len(v)] for b, v in val.groupby(bins, observed=True)["P_jump_lemat_minus_final"]})
    print("corr(P jump, log V_final/V_initial):",
          round(float(np.corrcoef(val["P_jump_lemat_minus_final"], np.log(val["V_final_over_initial"]))[0, 1]), 3))
    jumps = pd.DataFrame([dict(j, hardness=h) for s, h in zip(val["restart_jumps"], val["hardness"]) for j in json.loads(s)])
    if not jumps.empty:
        jumps["continuous"] = (jumps["dlat"] <= 1e-6) & (jumps["dpos"] <= 1e-6)
        print("\nrestart jumps inside the path:", jumps.groupby("transition").size().to_dict(),
              "| geometry continuous:", int(jumps["continuous"].sum()), "of", len(jumps),
              "| settings changed:", int(jumps["settings_changed"].sum()))
        cont = jumps[jumps["continuous"]]
        print("dP at restart (kB) by transition x hardness [median, p10, p90, frac>0, n]:")
        for (tr, h), v in cont.groupby(["transition", "hardness"])["dP"]:
            print(f"  {tr} {h}: [{v.median():.2f}, {v.quantile(.1):.2f}, {v.quantile(.9):.2f}, {(v > 0).mean():.2f}, {len(v)}]")


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["keys", "select", "extract", "validate", "summary", "all"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = _out()
    stages = ["keys", "select", "extract", "validate", "summary"] if args.stage == "all" else [args.stage]
    if "keys" in stages:
        _write_parquet(keys_stage(args.workers), out / "shard_keys.parquet")
    if "select" in stages:
        _write_parquet(select_stage(pd.read_parquet(out / "shard_keys.parquet"), args.seed),
                       out / "targets.parquet")
    if "extract" in stages:
        extract_stage(pd.read_parquet(out / "targets.parquet"), args.workers)
    if "validate" in stages:
        validate_stage(pd.read_parquet(out / "targets.parquet"))
    if "summary" in stages:
        summary_stage()


if __name__ == "__main__":
    main()
