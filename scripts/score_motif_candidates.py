"""Pool the built candidate genes of every arm, check and score them.

For each ``<arms dir>/<arm>.json.gz`` (a list of pyxtal-notation genes):
validity, deduplication across arms, gene-level motif flags and oxidation states
(:mod:`wyckoff_transformer.evaluation.gene_motifs`), gene and prototype novelty
against the LeMat-Bulk archive table, and the gene energy critic's predicted
formation energy against the fixed DFT hull (``wyformer-gene-screen``, run once
over all arms). Writes ``candidates.parquet`` and a ranked ``report.md``.

    python scripts/score_motif_candidates.py --arms <dir> --archive <archive.parquet> \\
        --regressor-path runs/min_energy_5x_adamw_wsd-20260921-125850 --out <dir>
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

logger = logging.getLogger("score_motif_candidates")

FLAG_COLUMNS = ["fs_d0_chain_2h", "cr3_no_octahedron", "d8_no_square_no_octahedron",
                "sojt_pinned_strong", "perovskite_t_out"]


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def _formula(gene: dict) -> str:
    c = Counter()
    for el, n in zip(gene["species"], gene["numIons"]):
        c[el] += n
    return "".join(f"{e}{n}" for e, n in sorted(c.items()))


def _sites(gene: dict) -> str:
    return " ".join(f"{el}:{','.join(s)}" for el, s in zip(gene["species"], gene["sites"]))


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--regressor-path", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    import warnings
    warnings.filterwarnings("ignore")
    from wyckoff_transformer.evaluation.gene_motifs import gene_flags, gene_key, prototype_key
    from wyckoff_transformer.evaluation.protocol import GeneFingerprinter

    signal.signal(signal.SIGALRM, _on_alarm)
    fp = GeneFingerprinter()
    rows, genes, seen = [], [], set()
    for f in sorted(args.arms.glob("*.json.gz")):
        arm = f.name.removesuffix(".json.gz")
        pool = json.load(gzip.open(f))
        n_valid = n_dup = 0
        for g in pool:
            try:
                rec = fp.record(g)
            except Exception:
                continue
            n_valid += 1
            key = gene_key(rec)
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            clean = {k: g[k] for k in ("group", "sites", "species", "numIons")}
            row = {"arm": arm, "formula": _formula(g), "space_group": g["group"], "sites": _sites(g),
                   "n_sites": len(rec["elements"]), "gene_key": key, "prototype_key": prototype_key(rec)}
            signal.alarm(10)       # the oxidation-state search can stall on odd compositions
            try:
                row.update(gene_flags(rec))
            except _Timeout:
                row["flag_error"] = "timeout"
            finally:
                signal.alarm(0)
            rows.append(row)
            genes.append(clean)
        logger.info("%s: %d genes, %d valid, %d duplicates", arm, len(pool), n_valid, n_dup)
    df = pd.DataFrame(rows)

    arc = pd.read_parquet(args.archive, columns=["gene_key", "prototype_key"])
    df["novel_gene"] = ~df["gene_key"].isin(set(arc["gene_key"]))
    df["novel_prototype"] = ~df["prototype_key"].isin(set(arc["prototype_key"]))

    pooled = args.out / "pooled_genes.json.gz"
    with gzip.open(pooled, "wt") as fh:
        json.dump(genes, fh)
    screen = args.out / "gene_screen.csv"
    if not screen.exists():
        subprocess.run(["wyformer-gene-screen", str(pooled), "--regressor-path", str(args.regressor_path),
                        "--out", str(screen), "--device", args.device], check=True)
    sc = pd.read_csv(screen, index_col="index")
    df = df.join(sc[["predicted_formation_energy", "hull_energy", "score", "reason"]]
                 .rename(columns={"score": "predicted_e_hull", "reason": "screen_reason"}))
    df.drop(columns=["gene_key", "prototype_key"]).to_parquet(args.out / "candidates.parquet")
    df.drop(columns=["gene_key", "prototype_key"]).to_csv(args.out / "candidates.csv", index=False)

    lines = ["# Built motif candidates, scored by the gene energy critic", "",
             f"Critic: `{args.regressor_path.name}` (predicted formation energy of the gene's best "
             "structure, minus the fixed LeMat-Bulk PBE hull at that composition; negative = "
             "predicted below the hull). Novelty is against all 5.3M LeMat-Bulk genes.", "",
             "| arm | unique valid | oxidation ok | motif flag | novel gene | novel prototype | pred. e_hull <= 0.05 | pred. <= 0 |",
             "|---|---|---|---|---|---|---|---|"]
    motif_of_arm = {}
    for arm, g in df.groupby("arm"):
        flags = g[FLAG_COLUMNS].fillna(False).astype(bool)
        motif = flags.any(axis=1)
        motif_of_arm[arm] = motif
        ok = g["predicted_e_hull"].notna()
        lines.append(f"| {arm} | {len(g):,} | {int(g['oxidation_ok'].fillna(False).sum()):,} | "
                     f"{int(motif.sum()):,} | {int(g['novel_gene'].sum()):,} | {int(g['novel_prototype'].sum()):,} | "
                     f"{int((ok & (g['predicted_e_hull'] <= 0.05)).sum()):,} | {int((ok & (g['predicted_e_hull'] <= 0)).sum()):,} |")
    lines += ["", "Best candidates per arm: motif flag set, novel gene, lowest predicted e_hull.", ""]
    for arm, g in df.groupby("arm"):
        sel = g[motif_of_arm[arm] & g["novel_gene"]].sort_values("predicted_e_hull").head(12)
        lines += [f"### {arm}", "", "| formula | sg | sites | flags | novel prototype | pred. E_f | pred. e_hull |",
                  "|---|---|---|---|---|---|---|"]
        for r in sel.itertuples():
            fl = ",".join(c for c in FLAG_COLUMNS if bool(getattr(r, c)) is True)
            lines.append(f"| {r.formula} | {r.space_group} | {r.sites} | {fl} | {r.novel_prototype} | "
                         f"{r.predicted_formation_energy:.3f} | {r.predicted_e_hull:.3f} |")
        lines.append("")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
