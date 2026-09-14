#!/usr/bin/env python3
"""Identify which MACE checkpoint produced LeMat-Bulk's published MLIP energies.

``LeMaterial/LeMat-Bulk-MLIP-Hull-All`` carries per-structure ``mace_mp_energy``
and ``mace_omat_energy`` for every LeMat-Bulk (compatible PBE) entry, and the
convex hulls LeMat-GenBench scores against are built from them.  Neither
dataset names the checkpoint.  This script recovers it the only way the data
allows: single-point energies on the LeMat-Bulk geometry under every MACE
foundation checkpoint, compared per structure with the published columns.

A published energy is a float32 total energy, so the right checkpoint agrees
to float32 rounding -- around 1e-6 eV/atom -- in either compute dtype, while a
wrong one misses by milli-eV per atom.  The gap is four orders of magnitude,
so a few hundred structures identify a checkpoint beyond doubt, and a uniform
sample over all shards bounds how much of the dataset could come from anything
else.

Result as of mace-torch 0.3.16 (Hull-All revision b7bfc3c6):
``mace_mp_energy`` is MACE-MP-0b3-medium, ``mace_omat_energy`` is
MACE-OMAT-0-medium.  Note that ``mace_mp()`` with no ``model`` -- what
LeMat-GenBench's calculator calls -- resolves to MACE-MPA-0-medium from
mace-torch 0.3.10 on, and to MACE-MP-0a-medium before; neither is the model
that built the hull.

Standalone: needs numpy, pandas, pyarrow, huggingface_hub, ase, torch and
mace-torch, and nothing from this repository.  Datasets download into the
HuggingFace cache and checkpoints into mace-torch's.

Examples::

    # Stratified by source from two shards that hold all three (~0.9 GB download)
    python scripts/verify_mace_hull_checkpoint.py --device cuda

    # Consistency sweep: 3000 structures uniform over the whole dataset
    python scripts/verify_mace_hull_checkpoint.py --shards all --per-source 0 \\
        --uniform 3000 --max-sites 100 --candidates medium-0b3,medium

    # Save per-structure energies for later analysis
    python scripts/verify_mace_hull_checkpoint.py --output mace_check.csv
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from ase import Atoms
from huggingface_hub import HfApi, hf_hub_download

HULL_REPO = "LeMaterial/LeMat-Bulk-MLIP-Hull-All"
BULK_REPO = "LeMaterial/LeMat-Bulk"
BULK_CONFIG = "compatible_pbe"
PUBLISHED_COLUMNS = ("mace_mp_energy", "mace_omat_energy")
SOURCES = ("agm", "oqmd", "mp")

#: Checkpoints not (or no longer) in mace-torch's alias registry.
EXTRA_CHECKPOINTS = {
    "large-2024-01-07": (
        "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/"
        "2024-01-07-mace-128-L2_epoch-199.model"
    ),
}

#: Multi-head models, evaluated once per head that plausibly targets PBE.
MULTIHEAD_HEADS = {
    "mh-0": ("mp_pbe_refit_add", "omat_pbe"),
    "mh-1": ("mp_pbe_refit_add", "omat_pbe"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--shards",
        default="2,4",
        help="LeMat-Bulk compatible_pbe shard indices to sample from, or 'all'. "
        "Shards 2 and 4 are the ones that hold Alexandria, OQMD and MP together.",
    )
    parser.add_argument("--per-source", type=int, default=100,
                        help="Structures sampled per source (agm, oqmd, mp).")
    parser.add_argument("--uniform", type=int, default=0,
                        help="Additional structures sampled uniformly.")
    parser.add_argument("--max-sites", type=int, default=40,
                        help="Skip structures with more sites, to bound runtime.")
    parser.add_argument("--candidates", default="all",
                        help="Comma-separated checkpoint names (mace-torch aliases, "
                        "EXTRA_CHECKPOINTS keys, or 'mh-1@omat_pbe'), or 'all'.")
    parser.add_argument("--dtypes", default="float64",
                        help="Comma-separated compute dtypes, e.g. float32,float64.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Max |delta| in eV/atom for a checkpoint to count as a match.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="Write per-structure energies to this CSV.")
    return parser.parse_args()


def candidate_checkpoints(spec: str) -> dict[str, tuple[str, str | None]]:
    """Map a display name to (checkpoint URL, head)."""
    from mace.calculators.foundations_models import mace_mp_urls

    registry: dict[str, tuple[str, str | None]] = {}
    for name, url in {**mace_mp_urls, **EXTRA_CHECKPOINTS}.items():
        heads = MULTIHEAD_HEADS.get(name)
        if heads:
            for head in heads:
                registry[f"{name}@{head}"] = (url, head)
        else:
            registry[name] = (url, None)
    if spec == "all":
        return registry
    unknown = [name for name in spec.split(",") if name not in registry]
    if unknown:
        sys.exit(f"Unknown candidates {unknown}; available: {', '.join(registry)}")
    return {name: registry[name] for name in spec.split(",")}


def bulk_shard_files(spec: str) -> list[str]:
    files = sorted(
        name for name in HfApi().list_repo_files(BULK_REPO, repo_type="dataset")
        if name.startswith(f"{BULK_CONFIG}/") and name.endswith(".parquet")
    )
    if spec == "all":
        return files
    return [files[int(index)] for index in spec.split(",")]


def sample_structures(args: argparse.Namespace) -> pd.DataFrame:
    """Draw the sample from LeMat-Bulk and attach the published energies."""
    rng = np.random.default_rng(args.seed)
    columns = ["immutable_id", "nsites", "lattice_vectors",
               "cartesian_site_positions", "species_at_sites", "energy"]
    frames = []
    for filename in bulk_shard_files(args.shards):
        print(f"Reading {BULK_REPO}/{filename}", flush=True)
        path = hf_hub_download(BULK_REPO, filename, repo_type="dataset")
        table = pq.read_table(path, columns=columns).to_pandas()
        frames.append(table[table.nsites <= args.max_sites])
    pool = pd.concat(frames, ignore_index=True)
    pool["source"] = pool.immutable_id.str.extract(r"^([a-z]+)", expand=False)

    picked = []
    for source in SOURCES:
        subset = pool.index[pool.source == source]
        count = min(args.per_source, len(subset))
        if count < args.per_source:
            print(f"Only {count} {source} structures available", flush=True)
        picked.extend(rng.choice(subset, count, replace=False))
    rest = pool.index.difference(picked)
    picked.extend(rng.choice(rest, min(args.uniform, len(rest)), replace=False))
    sample = pool.loc[picked].set_index("immutable_id")

    wanted = set(sample.index)
    energies = []
    for filename in sorted(
        name for name in HfApi().list_repo_files(HULL_REPO, repo_type="dataset")
        if name.endswith(".parquet")
    ):
        print(f"Reading {HULL_REPO}/{filename}", flush=True)
        path = hf_hub_download(HULL_REPO, filename, repo_type="dataset")
        table = pq.read_table(
            path, columns=["immutable_id", "true_energy", *PUBLISHED_COLUMNS]
        ).to_pandas()
        energies.append(table[table.immutable_id.isin(wanted)])
    published = pd.concat(energies).set_index("immutable_id")
    sample = sample.join(published, how="inner").dropna(subset=list(PUBLISHED_COLUMNS))

    # Guards the join: the Hull-All row must describe the same LeMat-Bulk entry.
    mismatch = (sample.energy - sample.true_energy).abs().max()
    if mismatch > 1e-6:
        sys.exit(f"Hull-All true_energy differs from LeMat-Bulk energy by {mismatch}")
    print(f"Sample: {len(sample)} structures, "
          f"{sample.source.value_counts().to_dict()}", flush=True)
    return sample


def to_atoms(row) -> Atoms:
    return Atoms(
        symbols=list(row.species_at_sites),
        positions=np.stack(row.cartesian_site_positions),
        cell=np.stack(row.lattice_vectors),
        pbc=True,
    )


def evaluate(sample: pd.DataFrame, url: str, head: str | None, dtype: str,
             device: str) -> np.ndarray:
    from mace.calculators import MACECalculator
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    kwargs = {"head": head} if head else {}
    calculator = MACECalculator(
        model_paths=download_mace_mp_checkpoint(url), device=device,
        default_dtype=dtype, **kwargs,
    )
    energies = np.full(len(sample), np.nan)
    for index, row in enumerate(sample.itertuples()):
        atoms = to_atoms(row)
        atoms.calc = calculator
        try:
            energies[index] = atoms.get_potential_energy()
        except Exception as exc:  # noqa: BLE001 -- one bad structure must not hide the rest
            print(f"  {row.Index}: {exc}", flush=True)
    del calculator
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return energies


def main() -> int:
    args = parse_args()
    candidates = candidate_checkpoints(args.candidates)
    sample = sample_structures(args)
    per_atom = sample.nsites.to_numpy()

    rows, computed = [], {}
    for name, (url, head) in candidates.items():
        for dtype in args.dtypes.split(","):
            label = f"{name}|{dtype}"
            try:
                energies = evaluate(sample, url, head, dtype, args.device)
            except Exception as exc:  # noqa: BLE001 -- a dead URL skips one candidate
                print(f"SKIP {label}: {exc}", flush=True)
                continue
            computed[label] = energies
            for column in PUBLISHED_COLUMNS:
                delta = np.abs(energies - sample[column].to_numpy()) / per_atom
                rows.append({
                    "candidate": name, "dtype": dtype, "published": column,
                    "median": np.nanmedian(delta),
                    "p99": np.nanquantile(delta, 0.99),
                    "max": np.nanmax(delta),
                    "frac_within_tol": np.nanmean(delta < args.tolerance),
                    "failed": int(np.isnan(energies).sum()),
                })
            best = min(rows[-2:], key=lambda r: r["median"])
            print(f"{label:34s} closest to {best['published']:17s} "
                  f"median |delta| {best['median']:.2e} eV/atom", flush=True)

    if args.output:
        out = sample[["source", "nsites", "true_energy", *PUBLISHED_COLUMNS]].copy()
        for label, energies in computed.items():
            out[label] = energies
        out.to_csv(args.output)
        print(f"Wrote {args.output}")

    summary = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_rows", None,
                           "display.float_format", "{:.2e}".format):
        for column in PUBLISHED_COLUMNS:
            print(f"\n|delta| per atom (eV) against {column}")
            print(summary[summary.published == column]
                  .drop(columns="published").sort_values("median")
                  .to_string(index=False))

    status = 0
    print()
    for column in PUBLISHED_COLUMNS:
        matches = summary[(summary.published == column)
                          & (summary["max"] < args.tolerance) & (summary.failed == 0)]
        names = sorted(set(matches.candidate))
        if names:
            print(f"{column}: every sampled structure within {args.tolerance:g} "
                  f"eV/atom under {', '.join(names)}")
        else:
            print(f"{column}: NO candidate matches every sampled structure")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
