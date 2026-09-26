#!/usr/bin/env python3
"""Which MP energy-correction scheme do MP-20's energies carry?

MP-20 (CDVAE) ships ``formation_energy_per_atom`` and ``e_above_hull`` with no raw
energy and no word on corrections. This script fetches today's Materials Project
GGA_GGA+U thermo docs for every MP-20 ``material_id``, strips the corrections off
each doc's raw entry, and recomputes the formation energy under both
``MaterialsProject2020Compatibility`` and the legacy ``MaterialsProjectCompatibility``,
each against its own elemental references. Whichever scheme reproduces MP-20 is the
one it was built with. Findings: ``docs/archive/mp20_energy_corrections.md``.

A third stage asks the legacy REST API -- the one MP-20 was originally pulled from --
for the same ids and compares its values with MP-20 and with both recomputations.

    check_mp20_energy_corrections.py fetch   --mp20-dir data/mp_20 --work-dir DIR --env .env
    check_mp20_energy_corrections.py compare --mp20-dir data/mp_20 --work-dir DIR
    check_mp20_energy_corrections.py legacy  --mp20-dir data/mp_20 --work-dir DIR --env .env

``fetch`` needs ``MP_API_KEY``, ``legacy`` needs ``MP_LEGACY_API_KEY`` (the legacy API
rejects new-style keys), and ``legacy`` reads ``compare``'s output.

``mp-api`` is not in the WyFormer venv; run both stages from a throwaway venv with
``mp-api pymatgen python-dotenv`` installed. ``fetch`` takes about 7 minutes and
writes ~120 MB.
"""
from __future__ import annotations

import argparse
import pickle
import string
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
FIELDS = ["material_id", "formation_energy_per_atom", "energy_above_hull", "energy_per_atom",
          "uncorrected_energy_per_atom", "energy_type", "entries", "deprecated", "composition"]
CHUNK = 500
LEGACY_API = "https://legacy.materialsproject.org/rest/v2"


def load_mp20(mp20_dir: Path, columns: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(mp20_dir / f"{s}.csv", usecols=columns) for s in SPLITS])


def alpha_to_classic(mid: str) -> str:
    """MP's AlphaIDs are base-26 renderings of the classic integer: ``mp-aaaaabwr`` is ``mp-1265``."""
    prefix, body = mid.split("-", 1)
    if not body.isalpha():
        return mid
    n = 0
    for ch in body:
        n = n * 26 + string.ascii_lowercase.index(ch)
    return f"{prefix}-{n}"


def fetch(mp20_dir: Path, work_dir: Path, env: Path) -> None:
    from dotenv import dotenv_values
    from mp_api.client import MPRester

    key = dotenv_values(env)["MP_API_KEY"]
    ids = load_mp20(mp20_dir, ["material_id"]).material_id.tolist()
    docs = {}
    with MPRester(key, mute_progress_bars=True) as mpr:
        db_version = mpr.get_database_version()
        for i in range(0, len(ids), CHUNK):
            for d in mpr.materials.thermo.search(material_ids=ids[i:i + CHUNK],
                                                 thermo_types=["GGA_GGA+U"], fields=FIELDS):
                dd = d.model_dump()
                docs[alpha_to_classic(str(dd["material_id"]))] = dd
            print(f"{i + CHUNK:6d} ids queried, {len(docs)} docs", flush=True)
        elements = sorted({el for d in docs.values() for el in d["composition"]})
        refs = []
        for el in elements:
            refs += [d.model_dump() for d in mpr.materials.thermo.search(
                chemsys=el, thermo_types=["GGA_GGA+U"], fields=FIELDS)]
    missing = [i for i in ids if i not in docs]
    print(f"MP db {db_version}: {len(docs)} docs, {len(missing)} ids without a GGA_GGA+U doc, "
          f"{len(refs)} elemental docs for {len(elements)} elements")
    work_dir.mkdir(parents=True, exist_ok=True)
    with open(work_dir / "thermo.pkl", "wb") as f:
        pickle.dump({"db_version": db_version, "docs": docs, "refs": refs, "missing": missing}, f)


def compare(mp20_dir: Path, work_dir: Path) -> None:
    from pymatgen.entries.compatibility import (MaterialsProject2020Compatibility,
                                                MaterialsProjectCompatibility)
    from pymatgen.entries.computed_entries import ComputedStructureEntry

    warnings.filterwarnings("ignore")
    with open(work_dir / "thermo.pkl", "rb") as f:
        data = pickle.load(f)
    mp20 = load_mp20(mp20_dir, ["material_id", "pretty_formula", "formation_energy_per_atom",
                                "e_above_hull", "elements"]).set_index("material_id")

    def raw_entry(doc):
        e = dict(doc["entries"][doc["energy_type"]])
        e["energy_adjustments"] = []
        e["correction"] = 0.0
        return ComputedStructureEntry.from_dict(e)

    schemes = {"mp2020": MaterialsProject2020Compatibility(check_potcar=False),
               "legacy": MaterialsProjectCompatibility()}

    ref_entries = [raw_entry(d) for d in data["refs"] if d["energy_type"] in d["entries"]]
    mu = {}
    for name, compat in schemes.items():
        m = {}
        for e in compat.process_entries([e.copy() for e in ref_entries], clean=True,
                                        inplace=False, on_error="ignore"):
            el = e.composition.elements[0].symbol
            m[el] = min(m.get(el, np.inf), e.energy_per_atom)
        mu[name] = m

    rows = []
    for mid, doc in data["docs"].items():
        if doc["energy_type"] not in doc["entries"]:
            continue
        ent = raw_entry(doc)
        r = {"material_id": mid, "api_ef": doc["formation_energy_per_atom"],
             "api_ehull": doc["energy_above_hull"]}
        for name, compat in schemes.items():
            p = compat.process_entry(ent.copy(), on_error="ignore")
            if p is None:
                r[f"{name}_ef"] = np.nan
                continue
            ref = sum(amt * mu[name][el.symbol] for el, amt in p.composition.items())
            r[f"{name}_ef"] = (p.energy - ref) / p.composition.num_atoms
        rows.append(r)
    df = pd.DataFrame(rows).set_index("material_id").join(mp20, how="inner")
    df.to_pickle(work_dir / "compare.pkl")

    print(f"MP db {data['db_version']}; MP-20 rows {len(mp20)}, with a GGA_GGA+U doc {len(df)}, "
          f"without {len(data['missing'])}")
    for col in ("api_ef", "mp2020_ef", "legacy_ef"):
        d = (df.formation_energy_per_atom - df[col]).dropna()
        a = d.abs()
        print(f"{col:10s} n={len(d):6d}  median|Δ|={a.median():.5f}  <1meV: {np.mean(a < 1e-3):.3f}  "
              f"<10meV: {np.mean(a < 1e-2):.3f}  mean Δ={d.mean():+.4f}")
    a = (df.e_above_hull - df.api_ehull).abs()
    print(f"e_above_hull vs api: median|Δ|={a.median():.5f}  <1meV: {np.mean(a < 1e-3):.3f}  "
          f"<10meV: {np.mean(a < 1e-2):.3f}")

    dis = df[(df.mp2020_ef - df.legacy_ef).abs() > 5e-3].copy()
    dis["err_2020"] = (dis.formation_energy_per_atom - dis.mp2020_ef).abs()
    dis["err_legacy"] = (dis.formation_energy_per_atom - dis.legacy_ef).abs()
    print(f"\nschemes differ by >5 meV/atom: {len(dis)} rows; MP2020 closer in "
          f"{np.mean(dis.err_2020 < dis.err_legacy):.3f}")
    groups = {"oxide (O)": ["O"], "Fe": ["Fe"], "Br/I/Se/Si/Sb/Te": ["Br", "I", "Se", "Si", "Sb", "Te"]}
    for tag, els in groups.items():
        sub = dis[dis.elements.apply(lambda s: any(f"'{e}'" in s for e in els))]
        print(f"  {tag:18s} n={len(sub):5d}  MP2020 closer: {np.mean(sub.err_2020 < sub.err_legacy):.3f}  "
              f"median err MP2020={sub.err_2020.median():.4f}  legacy={sub.err_legacy.median():.4f}")
    print()
    for formula in ("MgO", "Fe2O3", "NiO", "ZnSe", "SiC", "CsI", "Sb2Te3"):
        s = df[df.pretty_formula == formula].sort_values("e_above_hull").head(1)
        print(s[["pretty_formula", "formation_energy_per_atom", "api_ef", "mp2020_ef",
                 "legacy_ef"]].round(4).to_string(header=False))


def legacy(mp20_dir: Path, work_dir: Path, env: Path) -> None:
    import json

    import requests
    from dotenv import dotenv_values

    key = dotenv_values(env)["MP_LEGACY_API_KEY"]
    mp20 = load_mp20(mp20_dir, ["material_id", "formation_energy_per_atom", "e_above_hull",
                                "band_gap"]).set_index("material_id")
    ids = mp20.index.tolist()
    rows = []
    for i in range(0, len(ids), CHUNK):
        r = requests.post(f"{LEGACY_API}/query", headers={"X-API-KEY": key}, timeout=300, data={
            "criteria": json.dumps({"task_id": {"$in": ids[i:i + CHUNK]}}),
            "properties": json.dumps(["material_id", "formation_energy_per_atom", "e_above_hull", "band_gap"])})
        r.raise_for_status()
        rows += r.json()["response"]
        db = r.json()["version"]["db"]
    leg = pd.DataFrame(rows).drop_duplicates("material_id").set_index("material_id").add_prefix("leg_")
    leg.to_pickle(work_dir / "legacy_api.pkl")
    df = mp20.join(leg, how="inner")
    print(f"legacy API db {db}: {len(leg)} of {len(mp20)} MP-20 ids")
    for col in ("formation_energy_per_atom", "e_above_hull", "band_gap"):
        a = (df[col] - df[f"leg_{col}"]).abs()
        print(f"MP-20 vs legacy API {col:26s} median|Δ|={a.median():.5f}  <1meV: {np.mean(a < 1e-3):.3f}  "
              f"<10meV: {np.mean(a < 1e-2):.3f}  exact: {np.mean(a < 1e-9):.3f}")
    both = pd.read_pickle(work_dir / "compare.pkl").join(leg, how="inner")
    for col in ("mp2020_ef", "legacy_ef"):
        a = (both.leg_formation_energy_per_atom - both[col]).abs()
        print(f"legacy API vs recomputed {col:10s} n={len(a)}  median|Δ|={a.median():.5f}  "
              f"<1meV: {np.mean(a < 1e-3):.3f}  <10meV: {np.mean(a < 1e-2):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("fetch", "compare", "legacy"))
    parser.add_argument("--mp20-dir", type=Path, required=True, help="directory holding MP-20 train/val/test.csv")
    parser.add_argument("--work-dir", type=Path, required=True, help="where thermo.pkl and compare.pkl go")
    parser.add_argument("--env", type=Path, help="dotenv file with MP_API_KEY / MP_LEGACY_API_KEY")
    args = parser.parse_args()
    if args.stage in ("fetch", "legacy") and args.env is None:
        parser.error(f"{args.stage} needs --env")
    if args.stage == "fetch":
        fetch(args.mp20_dir, args.work_dir, args.env)
    elif args.stage == "legacy":
        legacy(args.mp20_dir, args.work_dir, args.env)
    else:
        compare(args.mp20_dir, args.work_dir)


if __name__ == "__main__":
    main()
