"""M1: the structure-level coordination-motif screen on generated, relaxed pools.

For every protocol directory given (``structures.csv`` plus ``cifs/<index>.cif``), runs
:func:`wyckoff_transformer.evaluation.coordination_motifs.analyse` on each relaxed
structure and flags the motifs of docs/coordination_motif_screen.md with the same
definitions its database report uses. The result is a per-structure table with the
pool's own e_above_hull and novelty columns, and a summary.

    python scripts/screen_generated_pools.py --out $WYFORMER_RUNS/coordination_motifs/pools \\
        <protocol dir> [<protocol dir> ...]
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd

logger = logging.getLogger("screen_generated_pools")

D0 = {("Ti", 4), ("Zr", 4), ("Hf", 4), ("V", 5), ("Nb", 5), ("Ta", 5), ("Cr", 6), ("Mo", 6),
      ("W", 6), ("Mn", 7), ("Tc", 7), ("Re", 7)}
HARD_AN, SOFT_AN = {"O", "F"}, {"S", "Se", "Te", "I"}
FLAGS = ["edge_or_face_hv_tet", "face_hv_tet", "rule4_hetero", "fs_d0_oct_any",
         "fs_d0_oct_extended", "cr3_tet", "d8_tet_or_prism", "sq_d0_z4", "hsab_inversion",
         "perovskite_t_out", "sojt_strong_pinned"]


def _one(item):
    import warnings
    warnings.filterwarnings("ignore")
    from wyckoff_transformer.evaluation.coordination_motifs import analyse
    key, path = item
    try:
        return analyse(key, Path(path).read_text())
    except Exception as exc:
        return {"material_id": key, "status": f"error: {type(exc).__name__}"}, [], []


def flags_for(row: dict, sites: list, pairs: list) -> dict:
    from wyckoff_transformer.evaluation.coordination_motifs import hsab_cation
    f = {k: False for k in FLAGS}
    if row.get("status") != "ok":
        return f
    for p in pairs:
        both_of = p["of1"] and p["of2"]
        hv = p["ox1"] >= 4 and p["ox2"] >= 4
        shared = p["mode"] in ("edge", "face")
        if both_of and hv and p["geom1"] == p["geom2"] == "tetrahedron" and shared:
            f["edge_or_face_hv_tet"] = True
            f["face_hv_tet"] |= p["mode"] == "face"
        if both_of and hv and shared and p["el1"] != p["el2"] and p["cn1"] <= 4 and p["cn2"] <= 4:
            f["rule4_hetero"] = True
    dim = row.get("face_sharing_d0_oct_dim", -1)
    f["fs_d0_oct_any"] = dim is not None and dim >= 0
    f["fs_d0_oct_extended"] = dim is not None and dim >= 1
    hard_soft = soft_hard = False
    ligand_union = set()
    for s in sites:
        clean = s["gap"] >= 1.10 and s["angle_rms"] <= 10
        sp = (s["element"], s["ox"])
        ligs = set(re.findall(r"[A-Z][a-z]?", s["ligands"]))
        ligand_union |= ligs
        if s["ligands_of"] and clean:
            f["cr3_tet"] |= sp == ("Cr", 3) and s["geometry"] == "tetrahedron"
            f["d8_tet_or_prism"] |= sp in {("Pd", 2), ("Pt", 2), ("Au", 3)} and \
                s["geometry"] in ("tetrahedron", "trigonal_prism")
            f["sq_d0_z4"] |= sp in D0 and s["geometry"] == "square_planar"
        if s["ligands_of"] and sp in {("Mo", 6), ("V", 5)} and s["geometry"] == "octahedron" \
                and not s["polar_site"]:
            f["sojt_strong_pinned"] = True
        cls = hsab_cation(*sp)
        hard_soft |= cls == "hard" and bool(ligs) and ligs <= SOFT_AN
        soft_hard |= cls == "soft" and bool(ligs) and ligs <= HARD_AN
    mixed = bool(ligand_union & HARD_AN) and bool(ligand_union & SOFT_AN)
    f["hsab_inversion"] = mixed and hard_soft and soft_hard
    t = row.get("tolerance_t", np.nan)
    f["perovskite_t_out"] = bool(row.get("perovskite_corner_3d")) and (t < 0.7 or t > 1.15)
    return f


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pools", nargs="+", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(args.workers) as pool:
        for d in args.pools:
            meta = pd.read_csv(d / "structures.csv", index_col="index")
            items = [(f"{d.parent.name}/{d.name}:{i}", d / "cifs" / f"{i}.cif")
                     for i in meta.index if (d / "cifs" / f"{i}.cif").is_file()]
            for (key, _), (row, sites, pairs) in zip(items, pool.imap(_one, items, chunksize=16)):
                i = int(key.rsplit(":", 1)[1])
                r = {"pool": str(d), "index": i, "formula": meta.at[i, "formula"],
                     "status": row.get("status"),
                     "e_above_hull": meta.at[i, "e_above_hull"] if "e_above_hull" in meta else np.nan,
                     "novel_structure": meta.at[i, "novel_structure"] if "novel_structure" in meta else np.nan,
                     "valid_structure": meta.at[i, "valid_structure"] if "valid_structure" in meta else np.nan}
                r.update(flags_for(row, sites, pairs))
                rows.append(r)
            logger.info("%s: %d structures", d, len(items))
    df = pd.DataFrame(rows)
    df.to_parquet(args.out / "pool_structures.parquet")

    ok = df[df["status"] == "ok"]
    lines = ["# M1: coordination motifs in generated, relaxed pools", "",
             f"{len(df):,} relaxed structures in {df['pool'].nunique()} pools; {len(ok):,} in scope "
             "(ionic, single-valence oxidation states). *Novel & e_hull <= 0.1* uses each "
             "pool's own `novel_structure` and `e_above_hull` (ORB hull).", "",
             "| motif | structures | novel | novel & e_hull <= 0.1 | examples (novel, lowest e_hull) |",
             "|---|---|---|---|---|"]
    nov = ok["novel_structure"].astype("boolean").fillna(False).astype(bool)
    near = ok["e_above_hull"] <= 0.1
    for flag in FLAGS:
        m = ok[flag].astype(bool)
        ex = ok[m & nov].sort_values("e_above_hull").head(5)
        lines.append(f"| {flag} | {int(m.sum()):,} | {int((m & nov).sum()):,} | "
                     f"{int((m & nov & near).sum()):,} | " + ", ".join(
                         f"{r.formula} ({r.e_above_hull:.3f}, {Path(r.pool).name}:{r.index})"
                         for r in ex.itertuples()) + " |")
    lines += ["", f"In-scope structures that are novel: {int(nov.sum()):,}; novel and e_hull <= 0.1: "
              f"{int((nov & near).sum()):,}."]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
