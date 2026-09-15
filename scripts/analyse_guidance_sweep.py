#!/usr/bin/env python3
"""Tabulate classifier-free guidance arms of the de novo ranking protocol.

An arm is one protocol output directory: a cohort drawn from one checkpoint at
one guidance scale and one condition (``wyformer-protocol-wandb
--guidance-scale W --arm NAME``), screened and possibly relaxed.  Arms may come
from different runs -- the point of the study is to put a model trained with
``condition_dropout`` beside the baseline trained without it -- so they are
named on the command line rather than discovered.

Besides the funnel, two things are read that the funnel does not carry:

*The archive e_hull of the genes the cohort reproduces.*  A gene whose
augmented Wyckoff fingerprint occurs in LeMat-Bulk has an energy above the
PBE hull there already -- the minimum over the archive's structures with that
fingerprint -- without relaxing anything.  Asked for e_hull = 0, a model that
follows its condition should reproduce low-e_hull genes rather than known genes
in general, and guidance should push that share up.  It only speaks for the
known part of the cohort, which is why it is reported beside the relaxed
readout and not instead of it.

*A test of every arm against a reference arm*, because the arms differ by a
few points and a 1000-gene cohort resolves about three: Fisher's exact test
on each per-sampled-gene rate, with the difference and its Newcombe interval.

Usage::

    scripts/analyse_guidance_sweep.py index
    scripts/analyse_guidance_sweep.py table \\
        --arm base=generated/<base-run>/protocol \\
        --arm w1=generated/<cfg-run>/protocol \\
        --arm w2=generated/<cfg-run>/guidance/w2 \\
        --reference base --output-dir generated/<cfg-run>/guidance/tables
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import pickle
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))
if str(_repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(_repo_root / "scripts"))

import numpy as np
import pandas as pd

from analyse_temperature_sweep import failure_breakdown, gene_shape, wilson
from wyckoff_transformer.cli.protocol import (
    FUNNEL_FILE,
    MANIFEST_FILE,
    RELAXATIONS_FILE,
    SCREEN_FILE,
    STRUCTURES_FILE,
)
from wyckoff_transformer.evaluation.protocol import (
    DEFAULT_REFERENCE_CACHE,
    DEFAULT_REFERENCE_SPLITS,
    GeneFingerprinter,
    _frame_fingerprints,
)
from wyckoff_transformer.paths import resolve_store_path

logger = logging.getLogger(__name__)

GENES_FILE = "wyckoff_genes.json.gz"
INDEX_NAME = "gene_ehull_index.pkl.gz"
EHULL_COLUMN = "energy_above_hull"

#: Archive e_hull thresholds, eV/atom, matching the protocol's stable and metastable.
STABLE, METASTABLE = 0.0, 0.1

#: (funnel section, key, label). Every key is a count; the table divides by `sampled`.
FUNNEL_COUNTS = (
    ("gene", "unique_gene", "unique gene"),
    ("gene", "sampled_novel", "novel gene"),
    ("free", "valid_structure", "valid structure"),
    ("free", "novel_structure", "novel structure"),
    ("free", "metastable", "metastable (ORB e_hull <= 0.1)"),
    ("free", "stable", "stable (ORB e_hull <= 0)"),
    ("free", "metastable_among_novel", "MetaSUN"),
    ("free", "stable_among_novel", "SUN"),
    ("fixed_symmetry", "metastable_among_novel", "MetaSUN, fixed symmetry"),
    ("fixed_symmetry", "stable_among_novel", "SUN, fixed symmetry"),
)


def index_path(reference_cache: Path) -> Path:
    """Where the fingerprint -> archive e_hull index of *reference_cache* is kept: beside it."""
    return Path(reference_cache).parent / INDEX_NAME


def build_index(reference_cache: Path, splits=DEFAULT_REFERENCE_SPLITS) -> dict:
    """Map every LeMat-Bulk fingerprint to (min archive e_hull, number of archive rows).

    The minimum, because a fingerprint fixes the orbits and not the coordinates:
    the archive can hold several structures on one gene, and the question is
    whether the gene *has* a stable realisation.
    """
    frames = pd.read_pickle(reference_cache)
    index: dict[tuple, list] = {}
    for split in splits:
        frame = frames[split]
        if EHULL_COLUMN not in frame:
            raise KeyError(f"{reference_cache}:{split} has no {EHULL_COLUMN} column")
        for fingerprint, e_hull in zip(_frame_fingerprints(frame), frame[EHULL_COLUMN].values):
            entry = index.get(fingerprint)
            if entry is None:
                index[fingerprint] = [float(e_hull), 1]
            else:
                entry[0] = min(entry[0], float(e_hull))
                entry[1] += 1
        logger.info("%s: %d rows, %d fingerprints so far", split, len(frame), len(index))
    return {fingerprint: tuple(entry) for fingerprint, entry in index.items()}


def load_index(reference_cache: Path) -> dict:
    path = index_path(reference_cache)
    if not path.is_file():
        raise SystemExit(f"No archive e_hull index at {path}; run `{Path(__file__).name} index`.")
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def archive_ehull(genes_path: Path, index: dict) -> pd.DataFrame:
    """Per sampled gene: whether LeMat-Bulk has its fingerprint, and the archive's min e_hull."""
    with gzip.open(genes_path, "rt", encoding="utf-8") as handle:
        genes = json.load(handle)
    fingerprinter = GeneFingerprinter()
    rows = []
    for gene in genes:
        try:
            entry = index.get(fingerprinter.fingerprint(gene))
        except Exception:  # noqa: BLE001 - an illegal gene is simply not in the archive
            entry = None
        rows.append({"known": entry is not None,
                     "archive_e_hull": np.nan if entry is None else entry[0]})
    return pd.DataFrame(rows)


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Newcombe's hybrid score interval for p1 - p2, from the two Wilson intervals."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    low = p1 - p2 - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    high = p1 - p2 + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return low, high


def read_arm(label: str, directory: Path, index: dict | None) -> dict:
    """Everything one arm contributes, however far it got."""
    screen_path = directory / SCREEN_FILE
    if not screen_path.is_file():
        raise SystemExit(f"{directory}: no {SCREEN_FILE}; screen the arm first")
    manifest_path = directory / MANIFEST_FILE
    funnel_path = directory / FUNNEL_FILE
    arm = {
        "label": label,
        "dir": directory,
        "screen": json.loads(screen_path.read_text()),
        "manifest": json.loads(manifest_path.read_text()) if manifest_path.is_file() else {},
        "funnel": json.loads(funnel_path.read_text()) if funnel_path.is_file() else None,
    }
    genes_path = directory / GENES_FILE
    arm["shape"] = gene_shape(genes_path)
    arm["shape"].pop("space_group_counts")
    counts = dict(arm["funnel"] or {"gene": arm["screen"]["summary"]})
    sampled = arm["screen"]["summary"]["sampled"]
    if index is not None:
        archive = archive_ehull(genes_path, index)
        if len(archive) != sampled:
            raise ValueError(f"{directory}: {len(archive)} genes against {sampled} sampled")
        known = archive[archive["known"]]
        counts["archive"] = {
            "known": int(len(known)),
            "archive_stable": int((known["archive_e_hull"] <= STABLE).sum()),
            "archive_metastable": int((known["archive_e_hull"] <= METASTABLE).sum()),
        }
        arm["archive_median_e_hull"] = (
            float(known["archive_e_hull"].median()) if len(known) else None)
        screened_known = arm["screen"]["summary"].get("sampled_known")
        if screened_known is not None and screened_known != len(known):
            # Both judge the same fingerprints against the same archive, so a mismatch
            # means the index was built from another reference than the screen's.
            logger.warning("%s: index finds %d known genes, the screen %d",
                           label, len(known), screened_known)
    arm["counts"] = counts
    arm["sampled"] = sampled
    structures_path = directory / STRUCTURES_FILE
    if structures_path.is_file():
        structures = pd.read_csv(structures_path, index_col="index")
        arm["failures"] = failure_breakdown(structures)
        unique = structures[structures["unique_structure"].astype(bool)]
        arm["median_orb_e_hull"] = float(unique["e_above_hull"].median(skipna=True))
    relaxations_path = directory / RELAXATIONS_FILE
    if relaxations_path.is_file():
        relaxations = pd.read_csv(relaxations_path)
        arm["trials"] = int(len(relaxations))
        arm["relax_hours"] = float(relaxations["seconds"].sum(skipna=True)) / 3600
    return arm


#: Archive rows: (key, label), counts under counts["archive"].
ARCHIVE_COUNTS = (
    ("known", "archive-known gene"),
    ("archive_stable", "known gene with archive e_hull <= 0"),
    ("archive_metastable", "known gene with archive e_hull <= 0.1"),
)


def count_rows(arm: dict):
    """(label, count) for every per-sampled-gene count the arm has."""
    for section, key, label in FUNNEL_COUNTS:
        value = arm["counts"].get(section, {}).get(key)
        if value is not None:
            yield label, int(value)
    for key, label in ARCHIVE_COUNTS:
        value = arm["counts"].get("archive", {}).get(key)
        if value is not None:
            yield label, int(value)


def stage_index(args) -> None:
    cache = resolve_store_path(args.reference_cache)
    path = index_path(cache)
    if path.is_file() and not args.force:
        raise SystemExit(f"{path} exists; pass --force to rebuild it")
    index = build_index(cache)
    with gzip.open(path, "wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Wrote %d fingerprints -> %s", len(index), path)


def stage_table(args) -> None:
    from scipy import stats  # noqa: PLC0415

    index = None if args.no_archive else load_index(resolve_store_path(args.reference_cache))
    arms = []
    for spec in args.arm:
        label, separator, directory = spec.partition("=")
        if not separator:
            raise SystemExit(f"--arm takes LABEL=DIR, got {spec!r}")
        arms.append(read_arm(label, Path(directory), index))
    labels = [arm["label"] for arm in arms]
    if args.reference not in labels:
        raise SystemExit(f"--reference {args.reference!r} is not one of the arms {labels}")
    reference = arms[labels.index(args.reference)]

    rows = []

    def add(metric, values):
        rows.append({"metric": metric, **dict(zip(labels, values))})

    add("guidance scale", [arm["manifest"].get("guidance_scale", 1.0) for arm in arms])
    add("condition", [json.dumps(arm["manifest"].get("generation_condition")) for arm in arms])
    add("formal validity of the draw", [arm["manifest"].get("formal_gene_validity") for arm in arms])
    add("sampled genes", [arm["sampled"] for arm in arms])
    for key in ("mean_orbits", "median_orbits", "mean_atoms", "mean_distinct_elements",
                "distinct_space_groups"):
        add(key.replace("_", " "), [round(arm["shape"][key], 2) for arm in arms])

    tests = {}
    reference_counts = dict(count_rows(reference))
    for label, _ in list(count_rows(reference)):
        cells, differences = [], []
        for arm in arms:
            counts = dict(count_rows(arm))
            if label not in counts:
                cells.append(None)
                differences.append(None)
                continue
            k, n = counts[label], arm["sampled"]
            low, high = wilson(k, n)
            cells.append(f"{k / n:.3f} [{low:.3f}, {high:.3f}]")
            if arm is reference:
                differences.append("reference")
                continue
            k0, n0 = reference_counts[label], reference["sampled"]
            d_low, d_high = newcombe(k, n, k0, n0)
            p_value = stats.fisher_exact([[k, n - k], [k0, n0 - k0]])[1]
            differences.append(f"{k / n - k0 / n0:+.3f} [{d_low:+.3f}, {d_high:+.3f}] p={p_value:.3g}")
            tests.setdefault(label, {})[arm["label"]] = {
                "difference": round(k / n - k0 / n0, 4),
                "newcombe_95": [round(d_low, 4), round(d_high, 4)],
                "fisher_p": float(p_value),
            }
        add(f"{label} per sampled gene [95% CI]", cells)
        add(f"  vs {args.reference} [95% CI]", differences)

    add("median archive e_hull of known genes",
        [None if arm.get("archive_median_e_hull") is None else round(arm["archive_median_e_hull"], 4)
         for arm in arms])
    add("median ORB e_hull, unique structures",
        [None if "median_orb_e_hull" not in arm else round(arm["median_orb_e_hull"], 4)
         for arm in arms])
    add("genes without a structure",
        [arm.get("failures", {}).get("genes_without_structure") for arm in arms])
    add("trials", [arm.get("trials") for arm in arms])
    add("relaxation worker-hours",
        [None if "relax_hours" not in arm else round(arm["relax_hours"], 1) for arm in arms])

    table = pd.DataFrame(rows)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "guidance_sweep.csv", index=False)
    (out_dir / "guidance_sweep.md").write_text(
        table.to_markdown(index=False) + "\n", encoding="utf-8")
    summary = {
        "reference": args.reference,
        "tests": tests,
        "arms": [{
            "label": arm["label"],
            "dir": str(arm["dir"]),
            "manifest": {key: arm["manifest"].get(key) for key in (
                "guidance_scale", "generation_condition", "sampling_temperature",
                "formal_gene_validity", "generation_attempted")},
            "counts": arm["counts"],
            "shape": arm["shape"],
            "archive_median_e_hull": arm.get("archive_median_e_hull"),
            "median_orb_e_hull": arm.get("median_orb_e_hull"),
            "failures": arm.get("failures"),
        } for arm in arms],
    }
    (out_dir / "guidance_sweep.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                 encoding="utf-8")
    with pd.option_context("display.max_columns", None, "display.width", 300,
                           "display.max_colwidth", 60):
        print(table.to_string(index=False))
    print(f"\n-> {out_dir}/guidance_sweep.csv, .md, .json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--reference-cache", type=Path, default=DEFAULT_REFERENCE_CACHE,
                        help="The LeMat-Bulk Wyckoff cache the archive e_hull is read from; "
                             "the protocol's novelty reference by default.")
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="Build the fingerprint -> archive e_hull index.")
    index.add_argument("--force", action="store_true")
    index.set_defaults(func=stage_index)
    table = sub.add_parser("table", help="Assemble the arms into one table.")
    table.add_argument("--arm", action="append", required=True, metavar="LABEL=DIR")
    table.add_argument("--reference", required=True,
                       help="Label of the arm every other arm is tested against.")
    table.add_argument("--output-dir", type=Path, required=True)
    table.add_argument("--no-archive", action="store_true",
                       help="Skip the archive e_hull rows (no index needed).")
    table.set_defaults(func=stage_table)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
