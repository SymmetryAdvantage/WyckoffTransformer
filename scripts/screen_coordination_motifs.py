"""Screen a LeMat-Bulk structure dataset for the coordination-motif rules and
measure how rare each motif is.

The rules and their rationale are in
``docs/ideas/Theoretical & Crystallographic Rules Defining Unprecedented Coordination Motifs.md``;
the measurements are ``wyckoff_transformer.evaluation.coordination_motifs``.

Three stages, each resumable:

    # 1. filter to e_hull <= 0.1 and cut into parquet shards (one pass over the CSVs)
    python scripts/screen_coordination_motifs.py shard --out $WYFORMER_RUNS/coordination_motifs
    # 2. measure every shard not yet measured (any number of processes/nodes may run
    #    this on disjoint --shard-mod slices)
    python scripts/screen_coordination_motifs.py run --out ... --workers 16
    # 3. rarity tables
    python scripts/screen_coordination_motifs.py report --out ...
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from multiprocessing import Pool

# One BLAS thread per worker: 16 forked workers each spinning up a full OpenBLAS pool
# stalled the first full run at 100% CPU on a single shard.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("screen_coordination_motifs")

DEFAULT_DATA = Path(os.environ.get("WYFORMER_DATA", "/home/project/11001786/WyFormer/data")) \
    / "lemat_bulk_fmax1_stress"


# --- shard ------------------------------------------------------------------------------


def shard(args):
    out = Path(args.out) / "shards"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "_DONE").exists():
        logger.info("shards already written")
        return
    n_shard, buffer, total = 0, [], 0
    for split in ("train", "val", "test"):
        path = Path(args.data) / f"{split}.csv.gz"
        for chunk in pd.read_csv(path, usecols=["immutable_id", "cif", "energy_above_hull"],
                                 chunksize=100_000):
            chunk = chunk[chunk["energy_above_hull"] <= args.max_ehull].copy()
            chunk["split"] = split
            buffer.append(chunk)
            if sum(len(b) for b in buffer) >= args.shard_size:
                df = pd.concat(buffer, ignore_index=True)
                while len(df) >= args.shard_size:
                    df.iloc[:args.shard_size].to_parquet(out / f"{n_shard:05d}.parquet")
                    total += args.shard_size
                    n_shard += 1
                    df = df.iloc[args.shard_size:]
                buffer = [df]
            logger.info("%s: %d rows kept so far", split, total + sum(len(b) for b in buffer))
    df = pd.concat(buffer, ignore_index=True)
    if len(df):
        df.to_parquet(out / f"{n_shard:05d}.parquet")
        total += len(df)
        n_shard += 1
    (out / "_DONE").write_text(f"{total} rows in {n_shard} shards\n")
    logger.info("%d rows in %d shards", total, n_shard)


# --- run --------------------------------------------------------------------------------


PER_STRUCTURE_TIMEOUT = 20  # seconds; a few compositions blow up the oxidation-state search


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def _analyse_one(item):
    import signal
    import warnings
    warnings.filterwarnings("ignore")
    from wyckoff_transformer.evaluation.coordination_motifs import analyse
    mid, cif = item
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(PER_STRUCTURE_TIMEOUT)
    try:
        return analyse(mid, cif)
    except _Timeout:
        return {"material_id": mid, "status": "timeout"}, [], []
    except Exception as exc:  # one bad structure must not lose a shard
        return {"material_id": mid, "status": f"error: {type(exc).__name__}: {exc}"[:200]}, [], []
    finally:
        signal.alarm(0)


def run(args):
    shard_dir = Path(args.out) / "shards"
    res_dir = Path(args.out) / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(shard_dir.glob("*.parquet"))
    if args.limit_shards:
        shards = shards[:args.limit_shards]
    shards = [s for i, s in enumerate(shards) if i % args.shard_mod[1] == args.shard_mod[0]]
    deadline = time.time() + args.time_limit if args.time_limit else None
    with Pool(args.workers, maxtasksperchild=2000) as pool:
        for s in shards:
            done = res_dir / f"{s.stem}.structures.parquet"
            if done.exists():
                continue
            if deadline and time.time() > deadline:
                logger.info("time limit reached; stopping before %s", s.name)
                break
            t0 = time.time()
            df = pd.read_parquet(s)
            if args.limit_rows:
                df = df.iloc[:args.limit_rows]
            items = list(zip(df["immutable_id"], df["cif"]))
            structures, sites, pairs = [], [], []
            for row, site_rows, pair_rows in pool.imap(_analyse_one, items, chunksize=64):
                structures.append(row)
                sites.extend(site_rows)
                pairs.extend(pair_rows)
            meta = df.set_index("immutable_id")[["energy_above_hull", "split"]]
            st = pd.DataFrame(structures).join(meta, on="material_id")
            # Write the marker file last so a killed job leaves the shard to redo.
            pd.DataFrame(sites).to_parquet(res_dir / f"{s.stem}.sites.parquet")
            pd.DataFrame(pairs).to_parquet(res_dir / f"{s.stem}.pairs.parquet")
            st.to_parquet(done)
            logger.info("%s: %d structures in %.0f s (%.1f ms each)", s.name, len(df),
                        time.time() - t0, 1000 * (time.time() - t0) / max(len(df), 1))


# --- report -----------------------------------------------------------------------------


def _load(res_dir: Path, kind: str) -> pd.DataFrame:
    files = sorted(res_dir.glob(f"*.{kind}.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _frac(n, d):
    return f"{n:,} / {d:,} ({100 * n / d:.3g}%)" if d else f"{n:,} / 0"


def report(args):
    res_dir = Path(args.out) / "results"
    st = _load(res_dir, "structures")
    sites = _load(res_dir, "sites")
    pairs = _load(res_dir, "pairs")
    ok = st[st["status"] == "ok"]
    n_ok = len(ok)
    lines = ["# Coordination-motif rarity in lemat_bulk_fmax1_stress_ehull01", ""]
    lines.append(f"Structures screened: {len(st):,}. In scope (ionic, single-valence oxidation "
                 f"states): **{n_ok:,}**.")
    lines += ["", "| status | structures |", "|---|---|"]
    status = st["status"].where(~st["status"].str.startswith("error"), "error")
    for k, v in status.value_counts().items():
        lines.append(f"| {k} | {v:,} |")
    lines += ["", "Rates below are per in-scope structure unless stated. `clean` means the "
              "coordination number is well defined (next anion at least 10% farther than the "
              "last bonded one) and the shape fits its ideal polyhedron within 10 degrees RMS.", ""]

    sites = sites.merge(ok[["material_id"]], on="material_id")
    sites["clean"] = (sites["gap"] >= 1.10) & (sites["angle_rms"] <= 10)
    sites["species"] = sites["element"] + sites["ox"].map(lambda z: f"{z:+d}")

    def structures_with(mask_sites):
        return sites.loc[mask_sites, "material_id"].nunique()

    def examples(mask_sites, k=8):
        sel = sites.loc[mask_sites].merge(ok[["material_id", "reduced_formula", "energy_above_hull"]],
                                          on="material_id").drop_duplicates("material_id")
        sel = sel.sort_values("energy_above_hull").head(k)
        return ", ".join(f"{r.material_id} {r.reduced_formula} ({r.energy_above_hull:.3f})"
                         for r in sel.itertuples())

    # Rule 1
    lines += ["## 1A. Rule 1 (radius ratio) -- weak", ""]
    rr = sites.dropna(subset=["radius_ratio_cn"])
    band = lambda cn: np.argmin(np.abs(np.subtract.outer(np.asarray(cn), [2, 3, 4, 6, 8, 12])), axis=-1)
    off = np.abs(band(rr["cn"].to_numpy()) - band(rr["radius_ratio_cn"].to_numpy().astype(int)))
    lines.append(f"- Cation sites (single anion type, Shannon radii available): {len(rr):,}")
    lines.append(f"- Observed CN band equals the radius-ratio band: {_frac(int((off == 0).sum()), len(rr))}")
    lines.append(f"- Off by two or more bands: {_frac(int((off >= 2).sum()), len(rr))}")
    large_low = rr[(rr["radius_ratio_cn"] >= 8) & (rr["cn"] <= 4) & rr["clean"]]
    lines.append(f"- Ratio-predicted CN >= 8 but clean CN <= 4: {_frac(len(large_low), len(rr))} sites, "
                 f"{large_low['material_id'].nunique():,} structures")
    hyper = sites[sites["species"].isin(["Si+4", "Al+3"]) & (sites["cn"] >= 8)]
    lines.append(f"- Si4+/Al3+ with CN >= 8: {len(hyper):,} sites, {hyper['material_id'].nunique():,} structures. "
                 f"Examples: {examples(sites.index.isin(hyper.index))}")
    lines.append("")

    # Rule 2
    lines += ["## 1B. Rule 2 (electrostatic valence / bond valence)", ""]
    for col, name in (("max_anion_ebs_dev", "Pauling bond-strength sum"),
                      ("max_anion_bvs_dev", "Brown bond-valence sum")):
        v = ok[col].dropna()
        lines.append(f"- Worst anion deviation, {name}: median {v.median():.2f}; "
                     f"structures with some anion off by > 35% (O outside 1.3-2.7): {_frac(int((v > 0.35).sum()), len(v))}")
    g = ok["gii"].dropna()
    lines.append(f"- Global instability index: median {g.median():.2f} v.u.; > 0.2 v.u.: {_frac(int((g > 0.2).sum()), len(g))}")
    lines.append("")

    # Rules 3 & 4
    lines += ["## 1C. Rules 3 & 4 (polyhedral sharing)", ""]
    base = pairs.merge(ok[["material_id"]], on="material_id")
    base_tet = (base["geom1"] == "tetrahedron") & (base["geom2"] == "tetrahedron")
    base_of = base["of1"] & base["of2"]
    base_hv = (base["ox1"] >= 4) & (base["ox2"] >= 4)
    tt = base[base_tet & base_of & base_hv]
    n_links = tt.groupby("mode")["n_links"].sum()
    tot = int(n_links.sum())
    lines.append(f"Tetrahedron-tetrahedron links between z >= +4 cations with only O/F ligands: {tot:,}")
    for mode in ("corner", "edge", "face"):
        sel = tt[tt["mode"] == mode]
        lines.append(f"- {mode}: {_frac(int(n_links.get(mode, 0)), tot)} links, in "
                     f"{sel['material_id'].nunique():,} structures")
    for mode in ("edge", "face"):
        sel = tt[tt["mode"] == mode].merge(ok[["material_id", "reduced_formula", "energy_above_hull"]],
                                          on="material_id").sort_values("energy_above_hull")
        ex = sel.drop_duplicates("material_id").head(10)
        lines.append(f"  - {mode}-sharing examples: " + ", ".join(
            f"{r.material_id} {r.reduced_formula} {r.el1}-{r.el2} d={r.min_d_mm:.2f} ({r.energy_above_hull:.3f})"
            for r in ex.itertuples()))
    lines.append("")
    hetero = base[base_hv & base_of & (base["el1"] != base["el2"]) & (base["cn1"] <= 4)
                  & (base["cn2"] <= 4) & base["mode"].isin(["edge", "face"])]
    lines.append(f"- Rule 4 proper, different z >= +4 cations with CN <= 4 sharing an edge or face: "
                 f"{hetero['material_id'].nunique():,} structures "
                 f"({', '.join(f'{a}-{b}' for a, b in hetero.groupby(['el1', 'el2']).size().sort_values(ascending=False).head(8).index)})")
    oct_ = base[(base["geom1"] == "octahedron") & (base["geom2"] == "octahedron") & base_of]
    d0 = {("Ti", 4), ("Zr", 4), ("Hf", 4), ("V", 5), ("Nb", 5), ("Ta", 5), ("Cr", 6), ("Mo", 6),
          ("W", 6), ("Mn", 7), ("Tc", 7), ("Re", 7)}
    is_d0 = [((a, x) in d0) and ((b, y) in d0) for a, x, b, y in
             zip(oct_["el1"], oct_["ox1"], oct_["el2"], oct_["ox2"])]
    oo = oct_[is_d0]
    n_links = oo.groupby("mode")["n_links"].sum()
    tot = int(n_links.sum())
    lines.append(f"\nOctahedron-octahedron links between d0, z >= +4 cations with only O/F ligands: {tot:,}")
    for mode in ("corner", "edge", "face"):
        lines.append(f"- {mode}: {_frac(int(n_links.get(mode, 0)), tot)} links, in "
                     f"{oo.loc[oo['mode'] == mode, 'material_id'].nunique():,} structures")
    dim = ok["face_sharing_d0_oct_dim"]
    lines.append("- Face-sharing d0 octahedral networks by dimensionality: " + ", ".join(
        f"{'finite' if d == 0 else f'{d}D'}: {int((dim == d).sum()):,}" for d in (0, 1, 2, 3)))
    ext = ok[dim >= 1].sort_values("energy_above_hull").head(10)
    lines.append("  - extended examples: " + ", ".join(
        f"{r.material_id} {r.reduced_formula} {int(r.face_sharing_d0_oct_dim)}D ({r.energy_above_hull:.3f})"
        for r in ext.itertuples()))
    lines.append("")

    # Section 2A
    lines += ["## 2A. LFSE site inversion", ""]
    lines.append("*Gene level* = the site point group rules out the classical geometry "
                 "(octahedron for d3, square plane for d8, tetrahedron for square-planar d0), so "
                 "the gene alone forces the unusual case. *O/F only* excludes chalcogenides and "
                 "halides, where anion-anion bonds often make the formal oxidation state (and so "
                 "the d count) wrong.")
    lines.append("")
    d3 = sites["species"].isin(["Cr+3", "Mn+4"])
    d8 = sites["species"].isin(["Pd+2", "Pt+2", "Au+3"])
    d0_site = sites["d_count"].eq(0) & (sites["ox"] >= 3)
    for name, mask, targets, gene in (
            ("d3 (Cr3+, Mn4+)", d3, ["tetrahedron", "trigonal_prism"],
             ~sites["pg_allows_octahedron"]),
            ("low-spin d8 (Pd2+, Pt2+, Au3+)", d8, ["tetrahedron", "trigonal_prism"],
             ~sites["pg_allows_square_planar"]),
            ("d0, z >= +3", d0_site, ["square_planar"],
             sites["pg_allows_square_planar"] & ~sites["pg_allows_tetrahedron"])):
        sub = sites[mask]
        lines.append(f"**{name}**: {len(sub):,} sites in {sub['material_id'].nunique():,} structures "
                     f"({int((mask & sites['ligands_of']).sum()):,} sites with O/F only). "
                     "Geometry distribution (clean sites): " + ", ".join(
                         f"{k} {v:,}" for k, v in sub[sub['clean']]['geometry'].value_counts().head(6).items()))
        for t in targets:
            hit = mask & (sites["geometry"] == t) & sites["clean"]
            hit_of = hit & sites["ligands_of"]
            lines.append(f"- clean {t}: {_frac(int(hit.sum()), len(sub))} sites, "
                         f"{structures_with(hit):,} structures; O/F only: {int(hit_of.sum()):,} sites, "
                         f"{structures_with(hit_of):,} structures. Examples (O/F only): {examples(hit_of)}")
        g = mask & gene
        lines.append(f"- gene level, classical geometry excluded by site symmetry: "
                     f"{_frac(int(g.sum()), len(sub))} sites, {structures_with(g):,} structures; clean "
                     "geometries there: " + ", ".join(
                         f"{k} {v:,}" for k, v in sites[g & sites['clean']]['geometry'].value_counts().head(5).items()))
        lines.append("")

    # Section 2B
    lines += ["## 2B. SOJT suppression", ""]
    lines.append("Cation sites with only O/F ligands; d0 cations only when octahedral. *Pinned* = "
                 "non-polar site point group, so off-centring is forbidden by symmetry (gene "
                 "level). Pinned sites are centred to within the 0.1 A symmetry tolerance by "
                 "construction; *pinned, off > 0.05 A* counts those whose stored geometry is in "
                 "fact displaced, i.e. symmetric only within tolerance.")
    lines += ["", "| cation | class | sites | pinned | pinned, off > 0.05 A | structures pinned |",
              "|---|---|---|---|---|---|"]
    sojt = {"Mo+6": "strong d0", "V+5": "strong d0", "W+6": "intermediate d0",
            "Ti+4": "intermediate d0", "Nb+5": "intermediate d0", "Ta+5": "weak d0",
            "Zr+4": "weak d0", "Hf+4": "weak d0", "Sn+2": "ns2", "Sb+3": "ns2", "Te+4": "ns2",
            "Se+4": "ns2", "I+5": "ns2", "Bi+3": "ns2", "Pb+2": "ns2"}
    sojt_site = sites["ligands_of"] & (
        (sites["species"].isin([k for k, v in sojt.items() if v.endswith("d0")])
         & (sites["geometry"] == "octahedron"))
        | sites["species"].isin([k for k, v in sojt.items() if v == "ns2"]))
    for sp, cls in sojt.items():
        sub = sites[sojt_site & (sites["species"] == sp)]
        pinned = sub[~sub["polar_site"]]
        off = pinned[pinned["offcentre"] > 0.05]
        lines.append(f"| {sp} | {cls} | {len(sub):,} | {_frac(len(pinned), len(sub))} | "
                     f"{len(off):,} | {pinned['material_id'].nunique():,} |")
    strong = sojt_site & sites["species"].isin(["Mo+6", "V+5"]) & ~sites["polar_site"]
    lines.append(f"\nPinned strong-SOJT (Mo6+, V5+ octahedra) examples: {examples(strong)}")
    lp = sojt_site & sites["species"].isin(["Sn+2", "Sb+3", "Te+4", "Se+4", "I+5"]) & ~sites["polar_site"]
    lines.append(f"\nPinned light lone-pair (Sn2+, Sb3+, Te4+, Se4+, I5+) examples: {examples(lp)}")
    lines.append("")

    # Section 3. Recomputed here with hard anions = O, F only (N and Cl are borderline in
    # many tables), from the stored ligand lists.
    import re
    hard_an, soft_an = {"O", "F"}, {"S", "Se", "Te", "I"}
    from wyckoff_transformer.evaluation.coordination_motifs import hsab_cation
    lig_sets = sites["ligands"].map(lambda s: frozenset(re.findall(r"[A-Z][a-z]?", s)))
    cls = [hsab_cation(e, z) for e, z in zip(sites["element"], sites["ox"])]
    sites["hsab2"] = cls
    hard_in_soft = sites.loc[(sites["hsab2"] == "hard") & lig_sets.map(lambda s: bool(s) and s <= soft_an), "material_id"]
    soft_in_hard = sites.loc[(sites["hsab2"] == "soft") & lig_sets.map(lambda s: bool(s) and s <= hard_an), "material_id"]
    # "Mixed-anion" = both a hard and a soft element actually act as ligands somewhere.
    ligands_of_structure = pd.Series(lig_sets.values, index=sites["material_id"].values) \
        .groupby(level=0).agg(lambda sets: frozenset().union(*sets))
    mixed_lig = set(ligands_of_structure.index[ligands_of_structure.map(
        lambda e: bool(e & hard_an) and bool(e & soft_an))])
    mixed = ok[ok["material_id"].isin(mixed_lig)]
    mixed_ids = set(mixed["material_id"])
    his, sih = set(hard_in_soft) & mixed_ids, set(soft_in_hard) & mixed_ids
    hsab_inv = his & sih
    lines += ["## 3. HSAB inversion", ""]
    lines.append(f"- Structures where both hard (O, F) and soft (S, Se, Te, I) anions are ligands: {_frac(len(mixed), n_ok)}")
    for ids, name in ((his, "a hard cation coordinated only by soft anions"),
                      (sih, "a soft cation coordinated only by hard anions"),
                      (hsab_inv, "both at once (the doc's inversion)")):
        sel = mixed[mixed["material_id"].isin(ids)].sort_values("energy_above_hull")
        lines.append(f"- {name}: {_frac(len(sel), len(mixed))}. Examples: " + ", ".join(
            f"{r.material_id} {r.reduced_formula} ({r.energy_above_hull:.3f})" for r in sel.head(8).itertuples()))
    lines.append("")

    # Section 4
    lines += ["## 4. Goldschmidt tolerance", ""]
    abx3 = ok.dropna(subset=["tolerance_t"])
    corner = abx3[abx3["perovskite_corner_3d"].astype(bool)]
    lines.append(f"- ABX3 structures (X = O, F, Cl, Br, I): {len(abx3):,}; of which 3D corner-sharing "
                 f"BX6 perovskite networks: {len(corner):,}")
    bins = [0, 0.7, 0.8, 1.05, 1.15, 9]
    lines += ["", "| t | ABX3 | corner-sharing 3D perovskite |", "|---|---|---|"]
    for lo, hi in zip(bins[:-1], bins[1:]):
        a = abx3[(abx3["tolerance_t"] >= lo) & (abx3["tolerance_t"] < hi)]
        c = corner[(corner["tolerance_t"] >= lo) & (corner["tolerance_t"] < hi)]
        lines.append(f"| [{lo}, {hi}) | {len(a):,} | {_frac(len(c), len(a))} |")
    out_of = corner[(corner["tolerance_t"] < 0.7) | (corner["tolerance_t"] > 1.15)].sort_values("energy_above_hull")
    lines.append(f"\nCorner-sharing 3D perovskites with t < 0.70 or t > 1.15: {len(out_of):,}. Examples: " + ", ".join(
        f"{r.material_id} {r.reduced_formula} t={r.tolerance_t:.2f} ({r.energy_above_hull:.3f})"
        for r in out_of.head(10).itertuples()))
    lines.append("")

    # Rule 5
    lines += ["## Rule 5 (parsimony), gene level", ""]
    m = ok["max_orbits_per_element"]
    lines.append(f"- Median max Wyckoff orbits per element: {m.median():.0f}; >= 6: {_frac(int((m >= 6).sum()), n_ok)}; "
                 f">= 10: {_frac(int((m >= 10).sum()), n_ok)}")
    lines.append("")

    # By source
    lines += ["## Motif structures by source database", ""]
    flags = {
        "edge/face-sharing hv tetrahedra": set(tt.loc[tt["mode"] != "corner", "material_id"]),
        "extended face-sharing d0 octahedra": set(ok.loc[dim >= 1, "material_id"]),
        "tetrahedral/prismatic d3": set(sites.loc[d3 & sites["geometry"].isin(["tetrahedron", "trigonal_prism"]) & sites["clean"], "material_id"]),
        "tetrahedral/prismatic ls-d8": set(sites.loc[d8 & sites["geometry"].isin(["tetrahedron", "trigonal_prism"]) & sites["clean"], "material_id"]),
        "square-planar d0": set(sites.loc[d0_site & (sites["geometry"] == "square_planar") & sites["clean"], "material_id"]),
        "HSAB inversion": hsab_inv,
        "perovskite t out of range": set(out_of["material_id"]),
    }
    srcs = ok["source"].value_counts()
    lines.append("| motif | " + " | ".join(srcs.index) + " |")
    lines.append("|---|" + "---|" * len(srcs))
    lines.append("| in-scope structures | " + " | ".join(f"{v:,}" for v in srcs.values) + " |")
    src_of = ok.set_index("material_id")["source"]
    for name, ids in flags.items():
        vc = src_of.loc[list(ids)].value_counts() if ids else pd.Series(dtype=int)
        lines.append(f"| {name} | " + " | ".join(f"{int(vc.get(s, 0)):,}" for s in srcs.index) + " |")
    lines.append("")

    # Frequency table (the database-rarity definition of "unprecedented")
    freq = (sites[sites["clean"]].groupby(["species", "cn", "geometry"])
            .agg(sites=("material_id", "size"), structures=("material_id", "nunique")).reset_index())
    tot_sp = sites[sites["clean"]].groupby("species").size().rename("species_sites")
    freq = freq.join(tot_sp, on="species")
    freq["fraction"] = freq["sites"] / freq["species_sites"]
    freq.to_csv(Path(args.out) / "environment_frequency.csv", index=False)
    link_freq = (base.groupby(["el1", "ox1", "geom1", "el2", "ox2", "geom2", "mode"])
                 .agg(links=("n_links", "sum"), structures=("material_id", "nunique")).reset_index())
    link_freq.to_csv(Path(args.out) / "sharing_frequency.csv", index=False)
    lines.append("Full frequency tables: `environment_frequency.csv` (species, CN, geometry) and "
                 "`sharing_frequency.csv` (polyhedron pair, sharing mode).")
    (Path(args.out) / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("shard", "run", "report"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
    sub.choices["shard"].add_argument("--data", default=str(DEFAULT_DATA))
    sub.choices["shard"].add_argument("--max-ehull", type=float, default=0.1)
    sub.choices["shard"].add_argument("--shard-size", type=int, default=20_000)
    r = sub.choices["run"]
    r.add_argument("--workers", type=int, default=os.cpu_count())
    r.add_argument("--shard-mod", type=int, nargs=2, default=[0, 1], metavar=("I", "N"),
                   help="process shards with index %% N == I")
    r.add_argument("--time-limit", type=float, default=0, help="seconds; stop starting new shards after")
    r.add_argument("--limit-shards", type=int, default=0)
    r.add_argument("--limit-rows", type=int, default=0, help="rows per shard (testing)")
    args = p.parse_args(argv)
    {"shard": shard, "run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
