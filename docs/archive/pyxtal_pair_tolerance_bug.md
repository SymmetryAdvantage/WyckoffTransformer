# PyXtal applies the wrong distance tolerance to pairs of unlike species

**Status:** fixed on [`kazeevn/PyXtal:fix/cross-species-distance-tolerance`](https://github.com/kazeevn/PyXtal/tree/fix/cross-species-distance-tolerance),
not yet sent upstream.
**Affects:** PyXtal 1.1.4 and `MaterSim/PyXtal@83fd465` (identical code), atomic
`from_random` / `random_crystal`. Molecular generation uses a different path
and is not affected.
**Severity:** silent. Generation succeeds and returns a structure that violates
the `Tol_matrix` it was given.

## Summary

`random_crystal` computes one distance tolerance per species — the like-like
one of the species it is about to place — and then applies it to *every* pair
that species forms, including pairs with the species already placed. The
`Tol_matrix` entry for the actual pair is never consulted.

With `prototype="atomic"` the tolerance for a pair is `f * (r_A + r_B)` where
`r` is the covalent radius, so what the generator applies, `f * 2 * r_A`, is
correct only when `r_A == r_B`. For unlike species it errs in whichever
direction the radii differ, and because species are placed one at a time, which
direction a given structure gets depends on the order `species` was passed in.

## Location

`pyxtal/crystal.py`, `random_crystal._set_ion_wyckoffs`:

```python
numIon_added = 0
tol = self.tol_matrix.get_tol(specie, specie)      # line 356: like-like only
...
    if self.check_wp(wyckoff_sites_tmp, wyks, cell, new_site, tol):   # line 418
```

and `random_crystal.check_wp`:

```python
def check_wp(self, wyckoff_sites_tmp, wyks, cell, new_site, tol):
    # Check current WP against existing WP's
    if new_site is None:
        return False

    return all(new_site.check_with_ws2(ws, cell, tol) for ws in wyckoff_sites_tmp + wyks)
```

`wyks` holds the sites of the species already finished, so the single `tol` is
applied across species. The other two consumers of `tol` in that function,
`wp.short_distances` and `wp.merge`, stay inside one orbit of one species, so
the like-like tolerance is the right one for them and they are not affected.

## What the generator applies vs. what `Tol_matrix` defines

`Tol_matrix(prototype="atomic", factor=1.3)`:

| A placed against an already-placed B | `Tol_matrix[A][B]` | applied, `[A][A]` | ratio |
|---|---:|---:|---:|
| Cs after O | 2.041 Å | 3.172 Å | 1.55 |
| O after Cs | 2.041 Å | 0.910 Å | 0.45 |
| Ba after O | 1.852 Å | 2.795 Å | 1.51 |
| O after Ba | 1.852 Å | 0.910 Å | 0.49 |
| Ti after H | 1.242 Å | 2.080 Å | 1.68 |
| H after Ti | 1.242 Å | 0.403 Å | 0.32 |

Two distinct consequences:

- **Too permissive** when the species being placed is the smaller one. Placing
  H against Ti admits contacts down to 0.40 Å, less than a third of the 1.24 Å
  the caller asked for. Nothing rejects the structure and nothing warns.
- **Too strict** when it is the larger one. Placing Cs against O demands
  3.17 Å where 2.04 Å was asked for. The sampler then rejects legal
  placements, retries, and — because `set_volume` redraws the cell volume on
  every failed lattice cycle — the structures that do survive are the ones
  drawn with an inflated cell. The tolerance factor thereby acts partly as a
  volume knob, which is not what it is documented to be.

## Reproducer

Needs only `pyxtal`, `numpy` and `ase`. It compares the current `check_wp`
against a corrected one on the same seeds.

```python
"""Reproducer: PyXtal checks unlike pairs against the wrong distance tolerance."""
import numpy as np
from ase.neighborlist import neighbor_list
from pyxtal import pyxtal
from pyxtal.crystal import random_crystal
from pyxtal.tolerance import Tol_matrix

TM = Tol_matrix(prototype="atomic", factor=1.3)


def worst_ratio(struc, tm=TM):
    """min(d / tol(pair)) over pairs of distinct atoms; < 1 breaks the floor."""
    atoms = struc.to_ase()
    num = atoms.numbers
    uniq = sorted({int(x) for x in num})
    cutoff = max(tm.get_tol(a, b) for a in uniq for b in uniq)
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    keep = i != j                      # self-images are a separate gap, see below
    i, j, d = i[keep], j[keep], d[keep]
    if len(d) == 0:
        return np.inf
    tol = np.array([tm.get_tol(int(num[a]), int(num[b])) for a, b in zip(i, j)])
    return float(np.min(d / tol))


def check_wp_current(self, tmp, wyks, cell, new_site, tol):
    """PyXtal 1.1.4, crystal.py:433 -- one tolerance for every pair."""
    if new_site is None:
        return False
    return all(new_site.check_with_ws2(ws, cell, tol) for ws in tmp + wyks)


def check_wp_fixed(self, tmp, wyks, cell, new_site, tol):
    """Each pair against its own Tol_matrix entry."""
    if new_site is None:
        return False
    for ws in tmp + wyks:
        t = self.tol_matrix.get_tol(new_site.specie, ws.specie)
        if not new_site.check_with_ws2(ws, cell, t if t is not None else tol):
            return False
    return True


CASES = [
    ("P1   Cs1 O3", 1, ["Cs", "O"], [1, 3], [["1a"], ["1a"] * 3]),
    ("P1   O3 Cs1", 1, ["O", "Cs"], [3, 1], [["1a"] * 3, ["1a"]]),
    ("Pnma Ba4 O4", 62, ["Ba", "O"], [4, 4], [["4c"], ["4c"]]),
    ("Pnma O4 Ba4", 62, ["O", "Ba"], [4, 4], [["4c"], ["4c"]]),
    ("P4mm Ti1 H4", 99, ["Ti", "H"], [1, 4], [["1a"], ["4d"]]),
    ("P4mm H4 Ti1", 99, ["H", "Ti"], [4, 1], [["4d"], ["1a"]]),
]

for label, impl in (("current", check_wp_current), ("fixed", check_wp_fixed)):
    random_crystal.check_wp = impl
    for name, spg, species, num_ions, sites in CASES:
        ratios = []
        for seed in range(60):
            s = pyxtal()
            s.from_random(3, spg, species, num_ions, sites=sites,
                          tm=TM, random_state=seed, max_count=10)
            ratios.append(worst_ratio(s))
        r = np.array(ratios)
        print(f"{label:>8} {name:<14} broken {(r < 1).mean():.3f}  "
              f"median {np.median(r):6.3f}  min {r.min():6.3f}")
```

Output (60 seeds per case; "broken" is the share of generated structures
containing a pair closer than `Tol_matrix` allows):

```
case           |  current: broken  median     min  |  fixed: broken  median     min
P1   Cs1 O3    |           0.567   0.961   0.458   |          0.000   1.161   1.000
P1   O3 Cs1    |           0.000   1.803   1.018   |          0.000   1.149   1.011
Pnma Ba4 O4    |           0.500   1.018   0.519   |          0.000   1.290   1.000
Pnma O4 Ba4    |           0.000     inf   2.503   |          0.000   1.333   1.008
P4mm Ti1 H4    |           0.383   1.100   0.464   |          0.000   1.446   1.004
P4mm H4 Ti1    |           0.000   2.389   1.012   |          0.000   1.347   1.012
```

The pairs of rows differ only in the order the two species are listed, and they
show the two faces of the bug:

- Large-then-small (`Cs1 O3`, `Ba4 O4`, `Ti1 H4`): 38–57% of structures come
  out with a pair below the requested floor, down to 0.46 of it.
- Small-then-large (`O3 Cs1`, `O4 Ba4`, `H4 Ti1`): no violations, but the
  structures are systematically over-spread — median worst ratio 1.80, 2.39,
  and for `Pnma O4 Ba4` a median of `inf`, meaning that for most seeds no two
  atoms are within even the largest tolerance of each other. Those cells are
  the survivors of the over-strict rejection.

With the fix, all six cases sit just above the floor (median 1.15–1.45) and
none breaks it, independently of the species order.

## Impact on a real workload

Measured on 400 Wyckoff site sets sampled by a generative model over space
groups 1–230 (arbitrary species order, `Tol_matrix(prototype="atomic",
factor=1.3)`, `max_count=30`), the share of *generated* structures containing a
pair closer than that floor, by total positional degrees of freedom:

| Σ dof | 0 | 1–2 | 3–5 | 6–10 | >10 |
|---|---:|---:|---:|---:|---:|
| below the f=1.3 floor | 0.032 | 0.064 | 0.234 | 0.254 | **0.327** |
| below even the f=1.0 floor | 0.021 | 0.013 | 0.019 | 0.042 | 0.000 |

Over a separate 200-structure sample the fix takes the overall rate from 0.200
to 0.005 while leaving generation success unchanged at 1.000, and halves total
generation time (69 s → 36 s), because the over-strict half of the error stops
making the sampler retry. It also rescues 4 of 10 site sets on which
`from_random` had previously exhausted `max_count=30` and raised.

## Fix

`check_wp` looks the tolerance up per pair, falling back to the passed-in value
for a pair with no tabulated radius (`get_tol` returns `None` there):

```python
for ws in wyckoff_sites_tmp + wyks:
    pair_tol = self.tol_matrix.get_tol(new_site.specie, ws.specie)
    if pair_tol is None:
        pair_tol = tol
    if not new_site.check_with_ws2(ws, cell, pair_tol):
        return False
return True
```

The signature is unchanged, so any external caller keeps working. Two
regression tests are added in `tests/test_crystal.py::TestDistanceTolerance`:
a deterministic unit test that a Cs–O contact of 1.5 Å (between the 0.91 Å O–O
tolerance and the 2.04 Å Cs–O one) is rejected, and a generation test that
asserts the pair floor holds over 10 seeds in both species orders. Both fail on
`master` and pass with the fix; `tests/test_crystal.py`, `test_wyckoff.py`,
`test_symmetry.py`, `test_group.py` and `test_lattice.py` are green (73 tests).

## A separate gap found alongside it, not fixed here

An atom is never checked against its own periodic images. For an orbit of
multiplicity 1, `wp.short_distances` has no intra-orbit pair to look at, so a
lattice vector shorter than the like-like tolerance survives generation. Real
example, `from_random(3, 1, ["Cs","O"], [1,3], sites=[["1a"],["1a"]*3],
tm=Tol_matrix("atomic", factor=1.3), random_state=8)`: cell
`a=3.116, b=10.153, c=5.488`, and the Cs–Cs contact along `a` is 3.116 Å
against a 3.172 Å tolerance. This is why the reproducer above excludes `i == j`
pairs. It is a cell-level check rather than a placement-level one and belongs in
`Lattice`; see proposal 5 of
[proposed_pyxtal_generation_fixes.md](proposed_pyxtal_generation_fixes.md),
which also carries the rest of the sampler-mechanics proposals this fix sits
under, and
[proposed_gene_search_improvements.md](proposed_gene_search_improvements.md) for
the search side.

## Environment

pyxtal 1.1.4 (and `MaterSim/PyXtal@83fd465`, byte-identical `crystal.py`),
numpy 2.5.3, ase 3.29.0, Python 3.12.3, Linux.
