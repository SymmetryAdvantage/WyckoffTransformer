"""Measure what PyXtal's ``random_crystal`` priors get wrong and how much a
Wyckoff gene determines the structure.

Three sub-commands, each printing the tables quoted in
``docs/pyxtal_dof_reduction_study.md``:

``priors``
    Parse a random sample of MP-20 training CIFs with ``pyxtal.from_seed`` and
    compare the empirical distributions of free coordinates, cell-shape
    parameters and cell volume with the distributions ``random_crystal`` draws
    from (``generate_point``, ``generate_cellpara``, ``set_volume``).

``prototype-transfer``
    For MP-20 test structures with positional freedom, check whether a training
    structure with the same element-anonymised prototype (space group, Wyckoff
    letters, and which sites share an element) is a ``StructureMatcher``
    framework match, i.e. whether copying its free parameters would land in the
    right basin.

``prototype-coverage``
    For a file of generated genes, the share whose exact augmented fingerprint,
    and whose element-anonymised prototype, occurs in the LeMat-Bulk reference
    fingerprint cache, by positional degrees of freedom.

Run with ``uv run python scripts/analyse_gene_priors.py <sub-command>``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import time
import warnings
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DOF_BINS = [-1, 0, 2, 5, 10, 10**6]
DOF_LABELS = ["0", "1-2", "3-5", "6-10", ">10"]

#: Polar space groups: those whose point group is polar (1, 2, m, mm2, 3, 3m,
#: 4, 4mm, 6, 6mm).  Their Euclidean normalizer contains continuous translations,
#: so the origin along the polar direction(s) is a gauge, not a coordinate.
POLAR_SPACE_GROUPS = frozenset(
    [1] + list(range(3, 10)) + list(range(25, 47)) + list(range(75, 81))
    + list(range(99, 111)) + list(range(143, 147)) + list(range(156, 162))
    + list(range(168, 174)) + list(range(183, 187))
)


def dof_bins(dof: pd.Series) -> pd.Series:
    return pd.cut(dof, DOF_BINS, labels=DOF_LABELS)


def load_cache(dataset: str) -> dict[str, pd.DataFrame]:
    with gzip.open(Path("cache") / dataset / "data.pkl.gz", "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- priors

def _from_seed_with_kick(structure):
    """``kick_pyxtal_until_it_works`` without the package import cost per worker."""
    from pyxtal import pyxtal

    for tol in (0.1, 0.2, 0.05, 0.3, 0.02):
        px = pyxtal()
        try:
            px.from_seed(structure, tol=tol, a_tol=5.0)
            return px
        except Exception:
            continue
    return None


def _priors_worker(cif: str) -> Optional[dict]:
    from pymatgen.core import Structure
    from pyxtal.database.element import Element as PxElement

    try:
        px = _from_seed_with_kick(Structure.from_str(cif, fmt="cif"))
        if px is None or len(px.atom_sites) == 0:
            return None
        free, dof_total, natoms, vol_est = [], 0, 0, 0.0
        for site in px.atom_sites:
            dof = site.wp.get_dof()
            dof_total += dof
            natoms += site.wp.multiplicity
            if dof > 0:
                free.extend(float(x) % 1.0 for x in site.wp.get_free_xyzs(site.position))
            # Mean of the uniform draw in random_crystal.set_volume, times factor 1.1.
            el = PxElement(site.specie)
            vmin, vmax = 4 / 3 * np.pi * el.covalent_radius**3, 4 / 3 * np.pi * el.vdw_radius**3
            vol_est += site.wp.multiplicity * 0.5 * (vmin + vmax)
        return dict(
            sg=px.group.number, ltype=px.group.lattice_type, dof=dof_total, natoms=natoms,
            free=free, para=list(px.lattice.get_para(degree=True)), volume=px.lattice.volume,
            vol_est=1.1 * vol_est,
        )
    except Exception:
        return None


def _near_fraction(vals: np.ndarray, denominators: Iterable[int], tol: float) -> float:
    near = np.zeros(len(vals), dtype=bool)
    for n in denominators:
        near |= np.abs(vals - np.round(vals * n) / n) < tol
    return float(near.mean())


def _uniform_measure(denominators: Iterable[int], tol: float) -> float:
    """Lebesgue measure of the union of tolerance intervals around all k/n on the circle."""
    points = np.array(sorted({round(k / n, 9) for n in denominators for k in range(n)}))
    lo, hi = points - tol, points + tol
    total, cur_lo, cur_hi = 0.0, lo[0], hi[0]
    for l, h in zip(lo[1:], hi[1:]):
        if l <= cur_hi:
            cur_hi = max(cur_hi, h)
        else:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = l, h
    return min(total + cur_hi - cur_lo, 1.0)


def cmd_priors(args: argparse.Namespace) -> None:
    df = pd.read_csv(Path("data") / args.dataset / "train.csv", index_col=0)
    rng = np.random.default_rng(args.seed)
    cifs = df["cif"].iloc[rng.choice(len(df), size=min(args.n, len(df)), replace=False)].tolist()
    with Pool(args.workers) as pool:
        res = [r for r in pool.map(_priors_worker, cifs, chunksize=20) if r is not None]
    print(f"parsed {len(res)} of {len(cifs)} structures")

    free = np.array([x for r in res for x in r["free"]])
    print(f"free coordinates: {len(free)} from {sum(r['dof'] > 0 for r in res)} structures with dof > 0")
    print("\nshare of free coordinates within tol of k/n, data vs uniform:")
    for denominators, label in (
        ((1, 2, 4), "n in {1,2,4}"),
        ((1, 2, 3, 4, 6, 8), "n <= 8"),
        ((1, 2, 3, 4, 5, 6, 8, 10, 12), "n <= 12"),
        (tuple(range(1, 25)), "n <= 24"),
    ):
        for tol in (0.005, 0.01, 0.02):
            print(f"  {label:14s} tol={tol:.3f}: data {_near_fraction(free, denominators, tol):.3f}"
                  f"  uniform {_uniform_measure(denominators, tol):.3f}")
    hist, _ = np.histogram(free, bins=24, range=(0, 1))
    print(f"\n24-bin histogram of free coordinates (uniform: {len(free) // 24} per bin):")
    print(hist.tolist())

    log_ca = [np.log(r["para"][2] / r["para"][0]) for r in res if r["ltype"] in ("tetragonal", "hexagonal", "trigonal")]
    print(f"\nlog(c/a), tetragonal/hexagonal/trigonal: n={len(log_ca)} mean={np.mean(log_ca):.3f} "
          f"sd={np.std(log_ca):.3f}   [PyXtal generate_cellpara: N(0, 0.35*sqrt(3)=0.61)]")
    beta = [r["para"][4] for r in res if r["ltype"] == "monoclinic"]
    print(f"monoclinic beta: n={len(beta)} mean={np.mean(beta):.1f} sd={np.std(beta):.1f} "
          f"p5={np.percentile(beta, 5):.1f} p95={np.percentile(beta, 95):.1f}   [PyXtal: N(90, 20) truncated to 30-150]")
    ortho = [np.std(np.log(r["para"][:3])) for r in res if r["ltype"] == "orthorhombic"]
    print(f"orthorhombic sd of log(a, b, c) within a cell: n={len(ortho)} median={np.median(ortho):.3f}   [PyXtal: 0.35]")

    ratio = np.array([r["volume"] / r["vol_est"] for r in res])
    print(f"\ntrue volume / PyXtal mean estimate: geo-mean={np.exp(np.mean(np.log(ratio))):.3f} "
          f"p25={np.percentile(ratio, 25):.3f} p75={np.percentile(ratio, 75):.3f} log-sd={np.std(np.log(ratio)):.3f}")
    vpa = np.array([r["volume"] / r["natoms"] for r in res])
    print(f"volume per atom: median={np.median(vpa):.2f} A^3, log-sd={np.std(np.log(vpa)):.3f}")

    dofs = np.array([r["dof"] for r in res])
    sgs = np.array([r["sg"] for r in res])
    polar = np.isin(sgs, list(POLAR_SPACE_GROUPS))
    print(f"\npositional dof: median={np.median(dofs):.0f} mean={dofs.mean():.2f} p90={np.percentile(dofs, 90):.0f} "
          f"share dof>=6: {(dofs >= 6).mean():.3f}")
    print(f"polar space groups: {polar.mean():.3f} of structures, {dofs[polar].sum() / dofs.sum():.3f} of positional dof")


# ------------------------------------------------------------- prototype transfer

def _record_gene_key(row: pd.Series) -> tuple:
    pairs = sorted(zip(row["wyckoff_letters"], (str(e) for e in row["elements"])))
    return (row["spacegroup_number"], tuple(pairs))


def _record_prototype_key(row: pd.Series) -> tuple:
    """Element-anonymised prototype: which sites share an element, but not which element."""
    pairs = sorted(zip(row["wyckoff_letters"], (str(e) for e in row["elements"])))
    relabel: dict[str, str] = {}
    out = []
    for letter, element in pairs:
        relabel.setdefault(element, f"E{len(relabel)}")
        out.append((letter, relabel[element]))
    return (row["spacegroup_number"], tuple(out))


_TRANSFER_STATE: dict = {}


def _transfer_init(train_cifs: pd.Series, test_cifs: pd.Series, by_proto: dict, max_candidates: int) -> None:
    from pymatgen.analysis.structure_matcher import FrameworkComparator, StructureMatcher

    _TRANSFER_STATE.update(
        train_cifs=train_cifs, test_cifs=test_cifs, by_proto=by_proto, max_candidates=max_candidates,
        strict=StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True,
                                comparator=FrameworkComparator()),
        loose=StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True, scale=True,
                               comparator=FrameworkComparator()),
    )


def _transfer_worker(task: tuple) -> tuple:
    from pymatgen.core import Structure

    test_index, proto, dof = task
    st = _TRANSFER_STATE
    candidates = st["by_proto"].get(proto, [])
    if not candidates:
        return dof, "no_prototype"
    try:
        target = Structure.from_str(st["test_cifs"].loc[test_index], fmt="cif")
    except Exception:
        return dof, "parse_failed"
    loose_hit = False
    for train_index in candidates[: st["max_candidates"]]:
        try:
            candidate = Structure.from_str(st["train_cifs"].loc[train_index], fmt="cif")
        except Exception:
            continue
        if st["strict"].fit(target, candidate):
            return dof, "framework_match"
        if st["loose"].fit(target, candidate):
            loose_hit = True
    return dof, "loose_match_only" if loose_hit else "prototype_only"


def cmd_prototype_transfer(args: argparse.Namespace) -> None:
    cache = load_cache(args.dataset)
    train, test = cache["train"], cache["test"]
    train_cifs = pd.read_csv(Path("data") / args.dataset / "train.csv", index_col=0)["cif"]
    test_cifs = pd.read_csv(Path("data") / args.dataset / "test.csv", index_col=0)["cif"]
    by_proto: dict[tuple, list] = defaultdict(list)
    for index, row in train.iterrows():
        by_proto[_record_prototype_key(row)].append(index)

    test = test.assign(dof_total=test["dof"].apply(sum), proto=test.apply(_record_prototype_key, axis=1))
    candidates = test[test.dof_total >= 1]
    rng = np.random.default_rng(args.seed)
    chosen = candidates.iloc[rng.choice(len(candidates), size=min(args.n, len(candidates)), replace=False)]
    tasks = [(index, row.proto, row.dof_total) for index, row in chosen.iterrows()]
    with Pool(args.workers, initializer=_transfer_init,
              initargs=(train_cifs, test_cifs, dict(by_proto), args.max_candidates)) as pool:
        results = pool.map(_transfer_worker, tasks, chunksize=10)
    df = pd.DataFrame(results, columns=["dof", "outcome"])
    bins = pd.cut(df.dof, [0, 2, 5, 10, 10**6], labels=["1-2", "3-5", "6-10", ">10"])
    table = pd.crosstab(bins, df.outcome, normalize="index").round(3)
    table["n"] = bins.value_counts().sort_index()
    print(f"{args.dataset} test structures with dof >= 1 against training structures with the same "
          f"element-anonymised prototype (up to {args.max_candidates} candidates each)")
    print(table)
    print("overall:", df.outcome.value_counts(normalize=True).round(3).to_dict(), "n =", len(df))


# ------------------------------------------------------------- prototype coverage

def _anonymise_fingerprint(fingerprint: tuple) -> tuple:
    """Drop element identity from an augmented fingerprint, keeping the site partition."""
    sg, variants = fingerprint
    out = set()
    for variant in variants:
        by_element: dict = defaultdict(list)
        for (element, site_symmetry, enumeration), count in variant:
            by_element[element].append(((site_symmetry, enumeration), count))
        out.add(tuple(sorted(tuple(sorted(v)) for v in by_element.values())))
    return (int(sg), frozenset(out))


def cmd_prototype_coverage(args: argparse.Namespace) -> None:
    from pyxtal.symmetry import Group

    from wyckoff_transformer.evaluation.protocol import GeneFingerprinter, load_genes

    t = time.time()
    with gzip.open(args.reference_cache, "rb") as f:
        reference = pickle.load(f)
    print(f"reference fingerprints: {len(reference)}, loaded in {time.time() - t:.0f} s")
    t = time.time()
    reference_prototypes = {_anonymise_fingerprint(fp) for fp in reference}
    print(f"distinct anonymised prototypes: {len(reference_prototypes)}, built in {time.time() - t:.0f} s")

    genes = load_genes(args.genes)
    fingerprinter = GeneFingerprinter()
    dof_of_letter: dict[tuple, int] = {}

    def gene_dof(gene: dict) -> int:
        group = Group(gene["group"])
        total = 0
        for sites in gene["sites"]:
            for site in sites:
                key = (gene["group"], site[-1])
                if key not in dof_of_letter:
                    dof_of_letter[key] = next(wp for wp in group if wp.letter == site[-1]).get_dof()
                total += dof_of_letter[key]
        return total

    rows = []
    for gene in genes:
        try:
            fingerprint = fingerprinter.fingerprint(gene)
        except Exception:
            continue
        rows.append((gene_dof(gene), fingerprint in reference,
                     _anonymise_fingerprint(fingerprint) in reference_prototypes))
    df = pd.DataFrame(rows, columns=["dof", "gene_known", "prototype_known"])
    bins = dof_bins(df.dof)
    table = df.groupby(bins, observed=True)[["gene_known", "prototype_known"]].mean().round(3)
    table["n"] = df.groupby(bins, observed=True).size()
    print(f"\n{args.genes}: exact gene known in the reference / element-anonymised prototype known")
    print(table)
    print("overall:", df[["gene_known", "prototype_known"]].mean().round(3).to_dict(), "n =", len(df))


# ----------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("priors", help="PyXtal priors against MP-20 free coordinates, cell shapes and volumes.")
    p.add_argument("--dataset", default="mp_20")
    p.add_argument("--n", type=int, default=6000, help="Training CIFs to sample.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    p.set_defaults(func=cmd_priors)

    p = sub.add_parser("prototype-transfer", help="Do same-prototype training structures share the framework?")
    p.add_argument("--dataset", default="mp_20")
    p.add_argument("--n", type=int, default=2000, help="Test structures with dof >= 1 to sample.")
    p.add_argument("--max-candidates", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    p.set_defaults(func=cmd_prototype_transfer)

    p = sub.add_parser("prototype-coverage", help="Prototype coverage of generated genes in LeMat-Bulk.")
    p.add_argument("--genes", type=Path, default=Path("generated/upi73i4k/wyckoff_genes_ehull0_n2500.json.gz"))
    p.add_argument("--reference-cache", type=Path, default=Path("cache/lemat_bulk_ehull/gene_fingerprints.pkl.gz"))
    p.set_defaults(func=cmd_prototype_coverage)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
