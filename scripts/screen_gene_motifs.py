"""Gene-level motif flags and prototype novelty for the archive and generated pools.

Measurements M3 (how often do gene-guaranteed motifs occur?) and M4 (how many
generated genes are a new structure type?) of docs/coordination_motif_screen.md.

    # keys + flags for every LeMat-Bulk gene (one-off)
    python scripts/screen_gene_motifs.py archive --out $WYFORMER_RUNS/coordination_motifs/gene_screen
    # keys + flags for a generated pool (a directory of wyformer-generate chunks)
    python scripts/screen_gene_motifs.py pool --out ... --name cfg_w3 --genes .../genes/cfg_w3_ehull005
    # the report
    python scripts/screen_gene_motifs.py report --out ...
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd

logger = logging.getLogger("screen_gene_motifs")

ARCHIVE = Path(os.environ.get("WYFORMER_CACHE", "/home/project/11001786/WyFormer/cache")) \
    / "lemat_bulk_fmax1_stress"
RECORD_COLUMNS = ["immutable_id", "spacegroup_number", "elements", "multiplicity",
                  "wyckoff_letters", "site_symmetries", "sites_enumeration",
                  "site_symmetries_augmented", "sites_enumeration_augmented",
                  "energy_above_hull"]


#: seconds per gene for the flags; a few generated compositions make the
#: combinatorial oxidation-state search run for many minutes.
FLAG_TIMEOUT = 10


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def _row_result(record: dict) -> dict:
    import signal
    from wyckoff_transformer.evaluation.gene_motifs import gene_flags, gene_key, prototype_key
    out = {"gene_key": gene_key(record), "prototype_key": prototype_key(record),
           "n_sites": len(record["elements"]),
           "n_elements": len({str(e) for e in record["elements"]})}
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(FLAG_TIMEOUT)
    try:
        out.update(gene_flags(record))
    except _Timeout:
        out["flag_error"] = "timeout"
    except Exception as exc:  # a flag failure must not lose the keys
        out["flag_error"] = f"{type(exc).__name__}: {exc}"[:120]
    finally:
        signal.alarm(0)
    return out


def _archive_chunk(df: pd.DataFrame) -> pd.DataFrame:
    import warnings
    warnings.filterwarnings("ignore")
    rows = []
    for rec in df.to_dict("records"):
        r = _row_result(rec)
        r["id"] = rec["immutable_id"]
        r["energy_above_hull"] = rec["energy_above_hull"]
        rows.append(r)
    return pd.DataFrame(rows)


def archive(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "archive.parquet"
    if target.exists():
        logger.info("%s exists", target)
        return
    frames = []
    with Pool(args.workers) as pool:
        for split in ("train", "val", "test"):
            df = pd.read_parquet(ARCHIVE / f"{split}.parquet", columns=RECORD_COLUMNS)
            if args.limit:
                df = df.iloc[:args.limit]
            t0 = time.time()
            chunks = [df.iloc[i:i + 5000] for i in range(0, len(df), 5000)]
            for i, res in enumerate(pool.imap(_archive_chunk, chunks)):
                res["split"] = split
                frames.append(res)
                if i % 100 == 0:
                    logger.info("%s: %d / %d chunks, %.0f s", split, i, len(chunks), time.time() - t0)
    pd.concat(frames, ignore_index=True).to_parquet(target)
    logger.info("wrote %s", target)


_FP = None


def _pool_chunk(genes: list) -> pd.DataFrame:
    import warnings
    warnings.filterwarnings("ignore")
    global _FP
    if _FP is None:
        from wyckoff_transformer.evaluation.protocol import GeneFingerprinter
        _FP = GeneFingerprinter()
    rows = []
    for g in genes:
        try:
            rec = _FP.record(g)
        except Exception:
            rows.append({"invalid": True})
            continue
        r = _row_result(rec)
        r["invalid"] = False
        rows.append(r)
    return pd.DataFrame(rows)


def pool(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"pool_{args.name}.parquet"
    files = sorted(Path(args.genes).glob("chunk_*.json.gz"))
    logger.info("%d chunk files in %s", len(files), args.genes)
    frames = []
    with Pool(args.workers) as p:
        for f in files:
            genes = json.load(gzip.open(f))
            parts = [genes[i:i + 2000] for i in range(0, len(genes), 2000)]
            df = pd.concat(p.map(_pool_chunk, parts), ignore_index=True)
            df["chunk"] = f.stem.split(".")[0]
            frames.append(df)
            logger.info("%s: %d genes", f.name, len(df))
    pd.concat(frames, ignore_index=True).to_parquet(target)
    logger.info("wrote %s", target)


def _pct(n, d):
    return f"{n:,} / {d:,} ({100 * n / d:.3g}%)" if d else "0 / 0"


def report(args):
    from wyckoff_transformer.evaluation.gene_motifs import FLAGS
    out = Path(args.out)
    arc = pd.read_parquet(out / "archive.parquet")
    arc_genes = set(arc["gene_key"])
    arc_protos = set(arc["prototype_key"])
    near = arc[arc["energy_above_hull"] <= 0.1]
    near_protos = set(near["prototype_key"])
    lines = ["# Gene-level motifs and prototype novelty (M3, M4)", ""]
    lines.append(f"Archive: LeMat-Bulk `lemat_bulk_fmax1_stress`, {len(arc):,} genes, "
                 f"{len(arc_genes):,} distinct genes, {len(arc_protos):,} distinct prototypes "
                 f"({len(near_protos):,} among the {len(near):,} with e_hull <= 0.1).")
    lines.append("")

    pools = {p.stem.removeprefix("pool_"): pd.read_parquet(p) for p in sorted(out.glob("pool_*.parquet"))}
    # M4
    lines += ["## M4. Prototype novelty", "",
              "Per arm, over valid genes. *Unique* = distinct gene keys within the arm. "
              "*Novel gene* = gene key absent from the archive. *Novel prototype* = structure "
              "type (elements anonymised) absent from the whole archive.", "",
              "| arm | valid genes | unique | novel genes (of unique) | novel prototypes (of unique) | distinct novel prototypes |",
              "|---|---|---|---|---|---|"]
    uniq = {}
    lines_err = []
    for name, df in pools.items():
        v = df[~df["invalid"].astype(bool)]
        u = v.drop_duplicates("gene_key").copy()
        u["novel_gene"] = ~u["gene_key"].isin(arc_genes)
        u["novel_proto"] = ~u["prototype_key"].isin(arc_protos)
        uniq[name] = u
        if "flag_error" in u:
            lines_err.append(f"- {name}: flags skipped for {int(u['flag_error'].notna().sum()):,} unique genes "
                             f"({int((u['flag_error'] == 'timeout').sum()):,} timeouts after {FLAG_TIMEOUT} s)")
        lines.append(f"| {name} | {len(v):,} | {len(u):,} | {_pct(int(u.novel_gene.sum()), len(u))} | "
                     f"{_pct(int(u.novel_proto.sum()), len(u))} | {u.loc[u.novel_proto, 'prototype_key'].nunique():,} |")
    lines.append("")
    lines += lines_err + ([""] if lines_err else [])
    for name, u in uniq.items():
        lines.append(f"Novel-prototype rate by number of Wyckoff sites, {name}:")
        bins = [0, 2, 4, 6, 10, 20, 100]
        u["site_bin"] = pd.cut(u["n_sites"], bins)
        tab = u.groupby("site_bin", observed=True).agg(genes=("novel_proto", "size"), novel=("novel_proto", "mean"))
        lines.append(", ".join(f"{b}: {100 * r.novel:.1f}% of {r.genes:,}" for b, r in tab.iterrows()))
        lines.append("")

    # M3
    lines += ["## M3. Gene-guaranteed motifs", "",
              "Rates per gene (archive rows / unique generated genes). Anions O and F only; "
              "oxidation states by the single-valence ICSD-prior guess.", ""]
    header = "| flag | archive, all | archive, e_hull <= 0.1 | " + " | ".join(
        f"{n} (novel genes / novel prototypes)" for n in uniq) + " |"
    lines += [header, "|---" * (3 + len(uniq)) + "|"]
    for flag in FLAGS:
        a_all = int(arc[flag].fillna(False).astype(bool).sum())
        a_near = int(near[flag].fillna(False).astype(bool).sum())
        cells = []
        for name, u in uniq.items():
            m = u[flag].fillna(False).astype(bool)
            cells.append(f"{int(m.sum()):,} ({int((m & u.novel_gene).sum()):,} / {int((m & u.novel_proto).sum()):,})")
        lines.append(f"| {flag} | {_pct(a_all, len(arc))} | {_pct(a_near, len(near))} | " + " | ".join(cells) + " |")
    lines.append("")
    for flag in ("fs_d0_chain_2h", "perovskite_t_out", "cr3_no_octahedron", "d8_no_square"):
        ex = arc[arc[flag].fillna(False).astype(bool)].sort_values("energy_above_hull").head(8)
        lines.append(f"- archive examples, {flag}: " + ", ".join(
            f"{r.id} ({r.energy_above_hull:.3f})" for r in ex.itertuples()))
    lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("archive", "pool", "report"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--workers", type=int, default=os.cpu_count())
    sub.choices["archive"].add_argument("--limit", type=int, default=0, help="rows per split (testing)")
    sub.choices["pool"].add_argument("--name", required=True)
    sub.choices["pool"].add_argument("--genes", required=True)
    args = p.parse_args(argv)
    {"archive": archive, "pool": pool, "report": report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
