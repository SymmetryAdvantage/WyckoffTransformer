"""What PyXtal's random generation does with a Wyckoff gene, and what fixes it.

`single_pyxtal` hands a gene to `pyxtal.from_random`, which guesses the cell and
every free coordinate by rejection sampling. This script measures what comes
back and what the candidate fixes in
`docs/proposed_pyxtal_generation_fixes.md` change about it.

The quantity throughout is the **floor ratio**: `min(d_ij / tol_ij)` over pairs
of distinct atoms, periodic images included, where `tol_ij` is the
`Tol_matrix(prototype="atomic", factor=1.3)` entry for that pair. Below 1.0 the
structure violates the distance floor it was generated under. Self-image pairs
(`i == j`) are excluded: they are governed by the cell rather than by placement
and are not checked at all for a multiplicity-1 orbit -- proposal 5 in the doc.

Subcommands:

    audit    success rate and floor ratio by positional dof, over a gene file
    arms     the candidate fixes, cumulative, on the same genes
    rescue   which fix rescues which of the genes a run failed to generate
    solve    replace rejection sampling with a soft-sphere solve

The patches are applied by monkey-patching `pyxtal.crystal.random_crystal`, so
this runs against an unmodified PyXtal. `CFG` toggles them; all off reproduces
stock behaviour. The `pair_tol` arm is the one that is *not* a proposal -- it is
the bug fix, already on `kazeevn/PyXtal:fix/cross-species-distance-tolerance`
(`docs/pyxtal_pair_tolerance_bug.md`), and it is kept here so the proposals can
be measured on top of it.

Usage:

    uv run python scripts/diagnostics/pyxtal_generation_audit.py audit \
        generated/upi73i4k/wyckoff_genes_ehull0_n2500.json.gz --n 400
    uv run python scripts/diagnostics/pyxtal_generation_audit.py arms <genes> --n 200
    uv run python scripts/diagnostics/pyxtal_generation_audit.py solve <genes> --n 8
    uv run python scripts/diagnostics/pyxtal_generation_audit.py rescue <genes> \
        --genes-from generated/upi73i4k/protocol/structures.csv
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.optimize import minimize

from pyxtal import pyxtal
from pyxtal.crystal import random_crystal
from pyxtal.database.element import Element
from pyxtal.symmetry import Group, choose_wyckoff
from pyxtal.tolerance import Tol_matrix
from pyxtal.wyckoff_site import atom_site

#: The tolerance CrySPR generates under (`cryspr.generator._DEFAULT_IADM`).
TM = Tol_matrix(prototype="atomic", factor=1.3)

#: dof bins used by the reconstruction studies, so the tables line up.
BINS = [-0.5, 0.5, 2.5, 5.5, 10.5, 20.5, np.inf]
LABELS = ["0", "1-2", "3-5", "6-10", "11-20", ">20"]

_groups: dict[int, Group] = {}


def positional_dof(gene: dict) -> int:
    """Free internal coordinates PyXtal has to draw for this gene."""
    group = _groups.setdefault(int(gene["group"]), Group(int(gene["group"])))
    return sum(
        int(group[str(site)[-1]].get_dof())
        for species_sites in gene["sites"]
        for site in species_sites
    )


def floor_ratio(atoms: Atoms, tm: Tol_matrix = TM) -> float:
    """`min(d / tol(pair))` over pairs of distinct atoms; < 1 breaks the floor."""
    numbers = atoms.numbers
    unique = sorted({int(n) for n in numbers})
    cutoff = max(tm.get_tol(a, b) for a in unique for b in unique)
    first, second, dist = neighbor_list("ijd", atoms, cutoff)
    distinct = first != second
    first, second, dist = first[distinct], second[distinct], dist[distinct]
    if len(dist) == 0:
        return np.inf
    tols = np.array([tm.get_tol(int(numbers[a]), int(numbers[b]))
                     for a, b in zip(first, second)])
    return float(np.min(dist / tols))


def load_genes(path: str | Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# The candidate fixes, as patches over an unmodified PyXtal.
# ---------------------------------------------------------------------------

#: pair_tol  -- the bug fix (not a proposal): per-pair distance tolerances
#: per_site  -- proposal 3: attempt budget per site rather than shared
#: order     -- proposal 3: place the largest orbits and largest atoms first
#: escalate  -- proposal 1: grow the cell when placement keeps failing
CFG: dict = dict(pair_tol=False, per_site=None, order=False, escalate=None)

_orig_check_wp = random_crystal.check_wp
_orig_set_ion = random_crystal._set_ion_wyckoffs
_orig_set_coords = random_crystal._set_coords
_orig_set_crystal = random_crystal.set_crystal


def _check_wp(self, tmp, wyks, cell, new_site, tol):
    if new_site is None:
        return False
    if not CFG["pair_tol"]:
        return _orig_check_wp(self, tmp, wyks, cell, new_site, tol)
    for ws in tmp + wyks:
        pair_tol = self.tol_matrix.get_tol(new_site.specie, ws.specie)
        if not new_site.check_with_ws2(ws, cell, pair_tol if pair_tol else tol):
            return False
    return True


def _set_ion_wyckoffs(self, numIon, specie, cell, wyks):
    """`random_crystal._set_ion_wyckoffs` with a per-site budget and ordering.

    Stock PyXtal gives a species `max(2 * n_sites, 10)` attempts *shared*
    across its sites, so one unlucky site starves the rest and the whole
    coordinate attempt is discarded.
    """
    if CFG["per_site"] is None and not CFG["order"]:
        return _orig_set_ion(self, numIon, specie, cell, wyks)

    numIon_added = 0
    tol = self.tol_matrix.get_tol(specie, specie)
    placed: list = []
    sites_list = deepcopy(self.sites[specie])

    if sites_list is not None and CFG["order"]:
        def hardest_first(site):
            index = site[0] if type(site) is tuple else site
            wp = self.group[index]
            return (-wp.multiplicity, wp.get_dof())
        sites_list = sorted(sites_list, key=hardest_first)

    per_site = CFG["per_site"] or 2
    if sites_list is not None:
        budget = max(per_site * len(sites_list), 10)
    else:
        min_wyckoffs = int(numIon / len(self.group.wyckoffs_organized[0][0]))
        budget = max(per_site * min_wyckoffs, 10)

    cycle = 0
    while cycle < budget:
        site = sites_list[0] if sites_list else None
        new_site = None
        if type(site) is tuple:
            index, xyz = site
            new_site = atom_site(self.group[index], xyz, specie)
        else:
            if site is not None:
                wp = self.group[site]
                pt = self.lattice.generate_point()
                if len(wp.short_distances(pt, cell, tol)) > 0:
                    cycle += 1
                    continue
                pt = wp.project(pt, cell, self.PBC)
            else:
                wp = choose_wyckoff(self.group, numIon - numIon_added, site,
                                    self.dim, self.rng)
                if wp is not False:
                    pt = self.lattice.generate_point()
                    pt, wp, _ = wp.merge(pt, cell, tol, group=self.group)
            if wp is not False:
                new_site = atom_site(wp, pt, specie)

        if self.check_wp(placed, wyks, cell, new_site, tol):
            if sites_list:
                sites_list.pop(0)
            placed.append(new_site)
            numIon_added += new_site.multiplicity
            if numIon_added == numIon:
                return placed
        cycle += 1
        self.numattempts += 1
    return None


def _set_coords(self):
    """Place the species in order of decreasing covalent radius.

    The space hogs go in while there is still room. Stock PyXtal uses the
    order `species` was given in, which for a generated gene is the order the
    model happened to emit the elements.
    """
    if not CFG["order"]:
        return _orig_set_coords(self)
    wyks: list = []
    cell = self.lattice.matrix
    order = sorted(range(len(self.species)),
                   key=lambda i: -Element(self.species[i]).covalent_radius)
    for i in order:
        output = self._set_ion_wyckoffs(self.numIons[i], self.species[i], cell, wyks)
        if output is None:
            return None
        wyks.extend(output)
    self.valid = True
    return wyks


def _set_crystal(self):
    """`random_crystal.set_crystal` with a monotonically growing cell.

    Stock PyXtal redraws the volume i.i.d. from the composition prior on every
    failed lattice cycle, so a gene whose fixed special positions do not fit at
    the prior's typical volume never fits, however many times it retries.
    """
    if CFG["escalate"] is None or self.lattice0 is not None:
        return _orig_set_crystal(self)
    self.numattempts = 0
    self.lattice_attempts, self.coord_attempts = (40, 10) if self.has_freedom else (5, 5)
    inflate = 1.0
    for cycle1 in range(self.lattice_attempts):
        self.set_volume()
        self.volume *= inflate
        self.set_lattice(self.lattice0)
        self.cycle1 = cycle1
        for cycle2 in range(self.coord_attempts):
            self.cycle2 = cycle2
            output = self._set_coords()
            if output:
                self.atom_sites = output
                break
        if self.valid:
            return
        self.lattice.reset_matrix()
        inflate *= CFG["escalate"]
    return


random_crystal.check_wp = _check_wp
random_crystal._set_ion_wyckoffs = _set_ion_wyckoffs
random_crystal._set_coords = _set_coords
random_crystal.set_crystal = _set_crystal

ARMS: dict[str, dict] = {
    "base": {},
    "tol": dict(pair_tol=True),
    "budget": dict(pair_tol=True, per_site=20),
    "order": dict(pair_tol=True, per_site=20, order=True),
    "vol": dict(pair_tol=True, per_site=20, order=True, escalate=1.05),
}


def set_arm(cfg: dict) -> None:
    CFG.update(dict(pair_tol=False, per_site=None, order=False, escalate=None))
    CFG.update(cfg)


def generate(gene: dict, max_count: int = 30, factor: float = 1.1) -> Atoms:
    candidate = pyxtal()
    candidate.from_random(dim=3, group=gene["group"], species=gene["species"],
                          numIons=gene["numIons"], sites=gene["sites"],
                          tm=TM, factor=factor, max_count=max_count)
    return candidate.to_ase()


def measure(genes: list[dict], picks, max_count: int = 30) -> pd.DataFrame:
    rows = []
    for k in picks:
        gene = genes[int(k)]
        start = time.time()
        try:
            atoms = generate(gene, max_count=max_count)
            ok, ratio, vol = True, floor_ratio(atoms), atoms.get_volume() / len(atoms)
        except Exception:
            ok, ratio, vol = False, np.nan, np.nan
        rows.append((int(k), positional_dof(gene), ok, ratio, vol, time.time() - start))
    frame = pd.DataFrame(rows, columns=["gene", "dof_pos", "ok", "floor", "vol_pa", "sec"])
    frame["bin"] = pd.cut(frame.dof_pos, BINS, labels=LABELS)
    return frame


def report(frame: pd.DataFrame, title: str) -> None:
    kept = frame[frame.ok]
    print(f"\n=== {title}  success {frame.ok.mean():.3f}  "
          f"floor broken {(kept.floor < 1).mean() if len(kept) else float('nan'):.3f}  "
          f"total {frame.sec.sum():.0f}s")
    print("  dof     n  success  broken<1.0  broken<0.77  median floor  median V/atom")
    for label in LABELS:
        chunk = frame[frame.bin == label]
        if not len(chunk):
            continue
        kept = chunk[chunk.ok]
        if not len(kept):
            print(f"{label:>5} {len(chunk):5d}    {chunk.ok.mean():.3f}"
                  f"          --           --            --            --")
            continue
        print(f"{label:>5} {len(chunk):5d}    {chunk.ok.mean():.3f}       "
              f"{(kept.floor < 1).mean():.3f}        {(kept.floor < 0.77).mean():.3f}"
              f"        {kept.floor.median():6.3f}        {kept.vol_pa.median():6.1f}")


# ---------------------------------------------------------------------------
# Proposal 4: rejection sampling -> continuous solve.
# ---------------------------------------------------------------------------

def soft_sphere_solve(gene: dict, seed: int = 0, maxiter: int = 300,
                      relax_lattice: bool = False) -> tuple[float, float, float, int]:
    """Draw the gene unfiltered, then push the free parameters off the overlaps.

    The orbit positions are affine in the free Wyckoff parameters at fixed
    lattice, so `dr/dx` is a constant matrix: build it once by finite
    differences and the analytic gradient of the overlap penalty costs one
    neighbour list per iteration.

    Returns `(floor before, floor after, seconds, n free parameters)`.
    """
    start = time.time()
    candidate = pyxtal()
    # A tolerance of ~0 makes generation a formality: it never rejects.
    candidate.from_random(dim=3, group=gene["group"], species=gene["species"],
                          numIons=gene["numIons"], sites=gene["sites"],
                          tm=Tol_matrix(prototype="atomic", factor=0.01),
                          random_state=seed, max_count=5)

    x0 = np.asarray(candidate.get_1d_rep_x(), dtype=float)
    n_lattice = 0 if relax_lattice else candidate.lattice.dof
    x_free = x0[n_lattice:]

    atoms = candidate.to_ase(resort=False)
    numbers = atoms.numbers
    unique = sorted({int(n) for n in numbers})
    lut = {(a, b): TM.get_tol(a, b) for a in unique for b in unique}
    cutoff = max(lut.values())
    before = floor_ratio(atoms)
    if len(x_free) == 0:
        return before, before, time.time() - start, 0

    # dr/dx, constant at fixed lattice.
    base = atoms.get_positions().ravel()
    jacobian = np.empty((base.size, len(x0)))
    for i in range(len(x0)):
        bumped = x0.copy()
        bumped[i] += 1e-5
        candidate.update_from_1d_rep(bumped)
        jacobian[:, i] = (
            candidate.to_ase(resort=False).get_positions().ravel() - base) / 1e-5
    candidate.update_from_1d_rep(x0)
    jacobian = jacobian[:, n_lattice:]

    def objective(x):
        full = x0.copy()
        full[n_lattice:] = x
        candidate.update_from_1d_rep(full)
        current = candidate.to_ase(resort=False)
        first, second, dist, delta = neighbor_list("ijdD", current, cutoff)
        tols = np.array([lut[(int(numbers[a]), int(numbers[b]))]
                         for a, b in zip(first, second)])
        overlapping = dist < tols
        grad = np.zeros((len(numbers), 3))
        if not overlapping.any():
            return 0.0, jacobian.T @ grad.ravel()
        d, t, v = dist[overlapping], tols[overlapping], delta[overlapping]
        # Each pair is listed twice by the neighbour list, hence the halving.
        value = float(np.sum((t - d) ** 2)) / 2.0
        coefficient = (2.0 * (t - d) / d)[:, None] * v
        np.add.at(grad, first[overlapping], coefficient)
        np.add.at(grad, second[overlapping], -coefficient)
        return value, jacobian.T @ (grad / 2.0).ravel()

    result = minimize(objective, x_free, jac=True, method="L-BFGS-B",
                      options=dict(maxiter=maxiter, ftol=1e-14, gtol=1e-10))
    full = x0.copy()
    full[n_lattice:] = result.x
    candidate.update_from_1d_rep(full)
    return (before, floor_ratio(candidate.to_ase(resort=False)),
            time.time() - start, len(x_free))


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_audit(args) -> None:
    genes = load_genes(args.genes)
    picks = np.random.default_rng(args.seed).choice(
        len(genes), size=min(args.n, len(genes)), replace=False)
    frame = measure(genes, picks, max_count=args.max_count)
    report(frame, "stock pyxtal")
    if args.out:
        frame.to_csv(args.out, index=False)


def cmd_arms(args) -> None:
    genes = load_genes(args.genes)
    picks = np.random.default_rng(args.seed).choice(
        len(genes), size=min(args.n, len(genes)), replace=False)
    for name in (args.arms.split(",") if args.arms else list(ARMS)):
        set_arm(ARMS[name])
        report(measure(genes, picks, max_count=args.max_count), name)
    set_arm({})


def cmd_solve(args) -> None:
    genes = load_genes(args.genes)
    dof = np.array([positional_dof(g) for g in genes])
    picks = np.argsort(-dof)[:args.n] if args.hardest else np.random.default_rng(
        args.seed).choice(len(genes), size=min(args.n, len(genes)), replace=False)
    rows = []
    for k in picks:
        try:
            before, after, seconds, n_free = soft_sphere_solve(
                genes[int(k)], relax_lattice=args.relax_lattice)
            rows.append((int(k), int(dof[k]), n_free, before, after, seconds))
        except Exception as exc:  # noqa: BLE001 - a failed solve is a datum
            rows.append((int(k), int(dof[k]), np.nan, np.nan, np.nan, np.nan))
            print(f"  gene {k}: {type(exc).__name__}: {exc}")
    frame = pd.DataFrame(rows, columns=["gene", "dof_pos", "n_free",
                                        "before", "after", "sec"]).dropna()
    print(f"\n{len(frame)} genes solved, dof_pos "
          f"{int(frame.dof_pos.min())}..{int(frame.dof_pos.max())}\n")
    print(f"floor before: median {frame.before.median():.3f}, "
          f"share >= 1.0 {(frame.before >= 1).mean():.3f}")
    print(f"floor after : median {frame.after.median():.3f}, "
          f"share >= 1.0 {(frame.after >= 1).mean():.3f}, "
          f"share >= 0.95 {(frame.after >= 0.95).mean():.3f}")
    print(f"seconds: median {frame.sec.median():.2f}, p95 {frame.sec.quantile(.95):.2f}")
    print("\n" + frame.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))


def cmd_rescue(args) -> None:
    """Which fix rescues which of the genes a protocol run failed to generate."""
    genes = load_genes(args.genes)
    if args.genes_from:
        structures = pd.read_csv(args.genes_from, index_col="index")
        failed = structures.index[~structures.has_structure].tolist()
    else:
        failed = [int(g) for g in args.gene_ids.split(",")]
    print(f"{len(failed)} genes with no structure: {failed}\n")
    arms = {"tol": ARMS["tol"],
            "tol+vol": dict(pair_tol=True, per_site=20, order=True, escalate=1.15)}
    rows = []
    for i in failed:
        gene = genes[i]
        row = {"gene": i, "spg": gene["group"], "dof": positional_dof(gene),
               "sites": sum(len(s) for s in gene["sites"])}
        for name, cfg in arms.items():
            set_arm(cfg)
            start = time.time()
            try:
                generate(gene, max_count=args.max_count)
                outcome = "ok"
            except Exception:  # noqa: BLE001 - a failure is the measurement
                outcome = "--"
            row[name] = f"{outcome} {time.time() - start:5.1f}s"
        set_arm({})
        try:
            _, after, seconds, _ = soft_sphere_solve(gene)
            row["solve"] = f"ok {seconds:5.2f}s (floor {after:.3f})"
        except Exception as exc:  # noqa: BLE001
            row["solve"] = f"-- ({type(exc).__name__})"
        rows.append(row)
        print(f"{i:5d} spg{row['spg']:4d} dof{row['dof']:4d} sites{row['sites']:3d} | "
              + " | ".join(f"{k} {row[k]}" for k in list(arms) + ["solve"]), flush=True)
    frame = pd.DataFrame(rows)
    print("\nrescued (of %d):" % len(frame))
    for name in list(arms) + ["solve"]:
        print(f"  {name:<8} {sum('ok' in str(v) for v in frame[name])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("genes", help="gene file, .json or .json.gz")
        p.add_argument("--n", type=int, default=200)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--max-count", type=int, default=30,
                       help="from_random retries; the pipeline uses 30")
        return p

    audit = common(sub.add_parser("audit", help="stock pyxtal, by dof"))
    audit.add_argument("--out", type=Path, default=None)
    audit.set_defaults(func=cmd_audit)

    arms = common(sub.add_parser("arms", help="the candidate fixes, cumulative"))
    arms.add_argument("--arms", default=None, help=f"subset of {','.join(ARMS)}")
    arms.set_defaults(func=cmd_arms)

    solve = common(sub.add_parser("solve", help="soft-sphere solve instead of rejection"))
    solve.add_argument("--hardest", action="store_true", default=True)
    solve.add_argument("--relax-lattice", action="store_true",
                       help="solve the cell parameters too, not just the coordinates")
    solve.set_defaults(func=cmd_solve)

    rescue = common(sub.add_parser("rescue", help="which fix rescues which failure"))
    rescue.add_argument("--genes-from", type=Path, default=None,
                        help="protocol structures.csv; uses its has_structure column")
    rescue.add_argument("--gene-ids", default="", help="comma-separated gene indices")
    rescue.set_defaults(func=cmd_rescue)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
