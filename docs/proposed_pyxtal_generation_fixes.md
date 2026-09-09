# Proposed fixes to the gene → structure step

> **These are proposals.** Nothing here is implemented in the pipeline. Each
> item states what was measured, on what, and what is still unknown. The one
> change that is *not* a proposal — the pair-tolerance bug fix — is described in
> [pyxtal_pair_tolerance_bug.md](pyxtal_pair_tolerance_bug.md) and is on
> `kazeevn/PyXtal:fix/cross-species-distance-tolerance`; the proposals below are
> measured on top of it, because two of them only make sense once the distance
> floor is actually being enforced.

`single_pyxtal` hands a gene to `pyxtal.from_random`, which guesses the cell
volume, the cell shape and every free coordinate by rejection sampling.
[The dof reduction study](archive/pyxtal_dof_reduction_study.md) established
what the *guessing* costs and what a perfect predictor of each quantity would
buy. This document is about the *mechanics* of the sampler: what it does when
the guessing gets hard, why it sometimes returns nothing after 40 minutes, and
why the structures it does return are not always the ones it was asked for.

Everything below is reproducible with
`scripts/diagnostics/pyxtal_generation_audit.py`, which applies each candidate
as a monkey-patch over an unmodified PyXtal so the arms can be compared without
a fork. The metric throughout is the **floor ratio**, `min(d_ij / tol_ij)` over
pairs of distinct atoms with periodic images included, against the
`Tol_matrix(prototype="atomic", factor=1.3)` that `cryspr.generator` generates
under. Below 1.0 the structure violates the floor it was generated with.

## The failure taxonomy

Generation failure is rare but expensive. In the `upi73i4k` protocol run
(`generated/upi73i4k/protocol/structures.csv`), 10 of 2500 genes have
`has_structure == False`, and those 10 consumed a mean of 647 s each, up to
2567 s — about 1.8 CPU-hours returning nothing. `from_random(max_count=30)`
retries `random_crystal` 30 times, each of which runs 40 lattice cycles × 10
coordinate attempts, and every failure discards all placements made so far.
There is no backtracking and no escalation, so a doomed gene grinds to the end
of a 12,000-attempt budget before raising.

Running the candidate fixes against exactly those 10 genes separates them into
two causes that need different remedies:

```
 gene  spg  dof  sites |  tol           tol+vol       solve            real run
  280    1  186    62  |  ok   55.8s    ok   0.5s     ok 0.57s (1.000)  fail 2567s
 1808   60  173    62  |  ok    4.5s    ok   0.4s     ok 0.64s (1.008)  fail 2333s
 1856   36   50    25  |  --   37.7s    ok  16.1s     ok 0.62s (1.000)  fail  379s
 2245    8   70    35  |  --   36.4s    ok  12.7s     ok 1.48s (0.958)  fail  365s
 1871   12   58    30  |  --   31.0s    ok  13.1s     ok 0.94s (0.885)  fail  343s
 1430  225    1     3  |  ok    0.0s    ok   0.0s     ok 0.02s (1.071)  fail   92s
 2122  221    1     4  |  --    6.1s    ok   0.7s     ok 0.02s (0.855)  fail   73s
  889   47    6     8  |  --   12.0s    ok   5.3s     ok 0.05s (0.488)  fail  121s
 1106   47    7     9  |  --    8.0s    ok   2.9s     ok 0.04s (0.507)  fail   87s
 2085   47    4    10  |  ok    2.3s    ok   3.4s     ok 0.07s (0.441)  fail  115s
                          4/10 rescued  10/10        10/10 built
```

(`tol` = the bug fix alone; `tol+vol` = plus proposals 1 and 3; `solve` =
proposal 4, with the floor it reached in brackets. The arms ran at
`max_count=3`, a tenth of the pipeline's budget, so their timings are if
anything pessimistic against the "real run" column.)

- **Combinatorial** failures: 280, 1808, 1856, 2245, 1871, with 25–62 sites and
  50–186 positional dof. Sequential rejection cannot place that many sites
  within a budget of `2 × n_sites` shared attempts. The solve fixes these
  outright.
- **Volume-infeasible** failures: 1430, 2122, 889, 1106, 2085, with 1–7
  positional dof. Their sites are mostly fixed special positions, so the
  contacts are a function of the cell alone and no amount of coordinate
  redrawing helps. Note the bracketed floors — the solve *builds* a structure
  for these but cannot repair it (0.441, 0.488, 0.507), because there are no
  free coordinates to move. Only a larger cell helps.

The two remedies are therefore complementary, not alternatives.

## Proposal 1 — grow the cell when placement keeps failing

`random_crystal.set_crystal` (`pyxtal/crystal.py:299`) calls `set_volume()` at
the top of every lattice cycle, which redraws the volume i.i.d. from the
composition prior (uniform between the covalent-sphere and van-der-Waals-sphere
sums, × `factor`). Nothing escalates: the only `× 1.1` in the file fires on a
`VolumeError` from `Lattice` construction, not on a placement failure. A gene
whose fixed sites do not fit at the prior's typical volume therefore never
fits, however many times it retries.

Change: multiply the drawn volume by a growing factor on each failed lattice
cycle, guarded on `self.lattice0 is None` so a caller-supplied cell is never
silently rescaled.

```python
inflate = 1.0
for cycle1 in range(self.lattice_attempts):
    self.set_volume()
    self.volume *= inflate
    ...
    inflate *= 1.15        # or 1.05; both were tried
```

Measured: with the pair-tolerance fix, escalation at ×1.15 rescues all five
volume-infeasible genes, worst case 16 s. A direct check confirms the mechanism
— gene 1430 (`Fm-3m`, sites `24e/4b/4a`, one free coordinate) fails at
`factor=1.1` and `1.5` and succeeds instantly at `factor=2.0` and `3.0`.

**Unknown:** whether an escalated cell relaxes as well as a prior-drawn one of
the same size. The compressive-annealing result
([oracle relaxed cell report](archive/cryspr_oracle_relaxed_cell_report.md))
says loose cells are good for the relaxation, so the expectation is positive,
but nothing has been relaxed from an escalated cell.

## Proposal 2 — fail fast, and report why

Two changes, both cheap:

- Cap the wall-clock or attempt budget per gene and give up early. The current
  budget is `max_count × lattice_attempts × coord_attempts` deep and a gene
  that will fail spends all of it. In the protocol run the ten failures cost
  more than 30× what a *successful* gene costs end to end, relaxation included
  (median 1.1–22 s per gene by dof bin).
- `from_random` raises `RuntimeError("long time to generate structure, check
  inputs")`, and `single_pyxtal` logs it and returns `None`. Neither says which
  species or site exhausted its budget, nor how close the best attempt got. A
  structured failure — the species, the site, the best floor ratio reached —
  would let the caller choose between escalating the volume (proposal 1) and
  switching to the solve (proposal 4) instead of guessing.

**Unknown:** nothing to measure; this is instrumentation.

## Proposal 3 — per-site attempt budget and hardest-first ordering, for the tail only

`_set_ion_wyckoffs` (`pyxtal/crystal.py:360`) gives a species
`max(2 × n_sites, 10)` attempts **shared across all of its sites**. For a
species with 30 sites that is one rejection per site on average; the
probability of getting through degrades exponentially in the number of sites
even at a high per-site acceptance rate, and the placements already made are
discarded on failure. Separately, the sites are consumed in gene order and the
species in the order they were emitted (`wyckoff_processor.py:777-787`), so the
first species gets the whole cell and the last is squeezed into what is left.

Changes: budget per site rather than shared; sort sites by decreasing
multiplicity then increasing dof; place species in decreasing covalent radius.

**Measured — and this is the part that matters:** on 200 genes drawn at random
from the cohort, these are not worth applying globally.

| arm | floor broken | total time, 200 genes |
|---|---:|---:|
| `base` (stock) | 0.200 | 69 s |
| `tol` (bug fix only) | **0.005** | **36 s** |
| `+ per-site budget 20` | 0.000 | 140 s |
| `+ hardest-first ordering` | 0.005 | 156 s |
| `+ volume escalation ×1.05` | 0.000 | 78 s |

The bug fix alone does essentially all the work and halves the time. Raising
the budget costs 4× the time for a 0.005 improvement, because on an easy gene a
fast restart with a fresh cell beats persisting with a bad one. So: apply this
only after a first pass has failed, as part of a tail path together with
proposal 1 — which is how the `tol+vol` arm in the taxonomy table is
configured.

**Unknown:** the ordering heuristic was measured only as a bundle with the
budget change, so its own effect is not separated. On the failing genes it is
confounded with escalation for the same reason.

## Proposal 4 — replace rejection sampling with a continuous solve

Rejection sampling has the wrong asymptotics: acceptance decays exponentially
in the number of sites, which is why the combinatorial failures exist at all.
The alternative is to draw with *no* distance filter — always instant — and
then move the free Wyckoff parameters downhill on a soft-sphere overlap
penalty,

    S(x) = Σ_pairs max(0, tol_ij − d_ij)² ,

until every contact clears the floor. At fixed lattice the orbit positions are
an affine function of the free parameters, so `dr/dx` is a *constant* matrix:
build it once by finite differences and the analytic gradient costs one
neighbour list per iteration.

Measured on the 8 highest-dof genes in the cohort, lattice held fixed at
PyXtal's own unfiltered draw:

```
 gene  dof_pos  n_free   before    after      sec
  280      186     186    0.356    1.000    0.589
 1808      173     173    0.279    1.008    0.650
 1274      171     171    0.289    1.000    0.535
  467       96      96    0.226    1.000    1.549
  742       92      92    0.266    1.069    0.223
  638       90      90    0.184    1.027    0.169
  436       72      72    0.628    1.078    0.120
 2245       70      70    0.192    0.958    1.525
```

8 of 8 reach ≥ 0.95 of the floor and 6 of 8 clear it, at a median of 0.56 s and
a p95 of 1.54 s, against the 343–2567 s these genes cost as *failures*. Genes
280, 1808 and 2245 are three of the ten the protocol run could not generate.

Most of the machinery already exists in PyXtal and needs no new code:
`pyxtal.get_1d_rep_x()` / `update_from_1d_rep(x)` / `from_1d_rep(...)` give the
symmetry-reduced parameter vector, and `pyxtal.lego.util.calculate_S` /
`calculate_dSdx` plus `pyxtal.lego.basinhopping` are an objective, a jacobian
and a global search over exactly that vector. This is the solver half of
candidate 2 of the [dof reduction study](archive/pyxtal_dof_reduction_study.md),
which assumed it had to be written.

Two things follow for the learned version of that candidate. First, `x` begins
with `lattice.dof` entries, so the cell and the coordinates can be solved
*jointly* — which is what the oracle result that "fractional coordinates are
only meaningful in the cell they belong to" (`volume_coords` 0.420 against
`coords_random_cell` 0.500) demands, and which a per-site coordinate head
structurally cannot do. Second, the soft-sphere objective can be swapped for a
predicted distogram or descriptor target without changing anything else.

**Unknown, and this is the important caveat:** the solve fixes *feasibility*. It
has not been relaxed, so its effect on `e_above_hull` is unmeasured. Two
specific risks. (a) The penalty only pushes atoms apart; it does not spread
them, so a solved structure could be clash-free and still clumped. Starting
from a uniform draw and moving minimally should keep it close to uniform, but
that is an argument, not a measurement. (b) An unfiltered draw has none of the
survivorship inflation described in proposal 6, so its cell is the raw prior
(~1.2× the data) rather than the ~1.68× the pipeline currently gets. `factor=`
would have to be set explicitly to keep the loose cell the relaxation schedule
depends on.

## Proposal 5 — check a site against its own periodic images

An atom is never compared with its own periodic images. For an orbit of
multiplicity 1, `wp.short_distances` has no intra-orbit pair to look at, so a
lattice vector shorter than the like-like tolerance survives generation. Real
example:

```python
pyxtal().from_random(3, 1, ["Cs", "O"], [1, 3],
                     sites=[["1a"], ["1a"] * 3],
                     tm=Tol_matrix(prototype="atomic", factor=1.3),
                     random_state=8)
# cell a=3.116 b=10.153 c=5.488; Cs-Cs along a is 3.116 A against a 3.172 A tolerance
```

This is a cell-level constraint, not a placement-level one: the shortest
lattice vector must clear `max_A tol(A, A)` over the species present. It
belongs in `Lattice` generation, where it is a cheap rejection on the drawn
cell rather than a check repeated per placement. It is also the reason the
regression tests and the audit script exclude `i == j` pairs — otherwise this
gap would be attributed to the pair-tolerance bug.

**Unknown:** how often it bites. It was found while writing the pair-tolerance
tests, not by a survey; P1 and other low-multiplicity-only genes are the
exposed case, and no rate has been measured.

## Proposal 6 — decouple the distance floor from the cell volume

`Tol_matrix(factor=1.3)` is documented as a distance floor, but because
`set_volume` redraws the volume on every failed lattice cycle, the volume of a
structure that *survives* the filter is the prior conditioned on passing.
Raising the tolerance therefore inflates the realised cell. That is why
`cryspr.generator` observes a median 1.68× cell while passing `factor=1.1`, and
it is the mechanism behind experiment plan item 2, "fix volume inflation
(confusing — cryspr won't converge without factor=1.3)": 1.3 has been doing
volume work, not distance work.

Direct evidence, from the arms table above: at dof > 10 the median cell moves
from 20.4 to 24.7 Å³/atom when the floor is merely enforced *correctly*, with
no change to `factor`. The reproducer in the bug report shows the extreme —
`Pnma O4 Ba4` in stock PyXtal has a median floor ratio of `inf`, meaning that
for most seeds no two atoms are within even the largest tolerance of each
other, because only absurdly dilated cells got through.

Proposal: with the pair tolerance fixed, set the volume explicitly through
`factor=` (or from a predicted volume) and re-measure whether 1.3 is still
needed at all. This is a prerequisite for the lattice head — candidate 1 of the
dof reduction study — which cannot be evaluated while the tolerance is
covertly setting the volume.

**Unknown:** everything downstream. The trial schedule, the rattle acceptance
threshold and the recovery numbers the ranking protocol rests on were all
measured at `factor=1.3` with the buggy filter, so re-tuning the volume means
re-measuring those. That is the main cost of this proposal and the reason it is
listed last despite being conceptually the cleanest.

## What none of this addresses

Generation failure was never the main loss — 10 genes in 2500. The cost is
quality, and it degrades smoothly:

| Σ dof_pos | 0 | 1–2 | 3–5 | 6–10 | 11–20 | >20 |
|---|---:|---:|---:|---:|---:|---:|
| n | 493 | 565 | 706 | 446 | 213 | 77 |
| P(e_hull ≤ 0.1) | 0.408 | 0.300 | 0.214 | 0.144 | 0.113 | 0.167 |

Fixing the clashes should help — 33% of dof > 10 starts violate the floor, and
the `volume_coords` arm of the oracle test established that below-floor starts
relax to nonsense — but by how much is unmeasured, and none of it substitutes
for the lattice head. The oracle table already says what closes the curve: the
true cell and the true coordinates together recover 1.000 at every dof, and
each alone recovers 0.17 and 0.16 at dof > 10.

## Reproduce

```bash
# floor ratio and success by dof, stock pyxtal
uv run python scripts/diagnostics/pyxtal_generation_audit.py audit \
    generated/upi73i4k/wyckoff_genes_ehull0_n2500.json.gz --n 400

# the arms table (base / tol / budget / order / vol)
uv run python scripts/diagnostics/pyxtal_generation_audit.py arms \
    generated/upi73i4k/wyckoff_genes_ehull0_n2500.json.gz --n 200

# the soft-sphere solve on the highest-dof genes
uv run python scripts/diagnostics/pyxtal_generation_audit.py solve \
    generated/upi73i4k/wyckoff_genes_ehull0_n2500.json.gz --n 8

# the failure taxonomy, from a protocol run's has_structure column
uv run python scripts/diagnostics/pyxtal_generation_audit.py rescue \
    generated/upi73i4k/wyckoff_genes_ehull0_n2500.json.gz \
    --genes-from generated/upi73i4k/protocol/structures.csv --max-count 3
```

## See also

- [Finding the lowest-energy structure for a gene](proposed_gene_search_improvements.md)
  — the other half of the problem: these proposals decide whether a structure
  comes back at all, those decide whether it is the right one
- [The pair-tolerance bug](pyxtal_pair_tolerance_bug.md) — the one fix that is
  implemented, and the reason the arms here start from `tol`
- [Template-matched starts](cryspr_template_starts.md) — the one thing on this
  page's neighbouring list that is now implemented: a training structure's
  geometry instead of a random draw, which removes the guessing rather than
  making it cheaper. It wins the reconstruction benchmark outright and barely
  moves MetaSUN, because what it recovers is what novelty subtracts
- [What else should WyFormer learn to take the guesswork out of PyXtal?](archive/pyxtal_dof_reduction_study.md)
  — the lattice head, the distogram solve, prototype retrieval, and the oracle
  bounds on each
- [Oracle coordination number study](archive/cryspr_oracle_coordination_study_report.md)
  — why a dense oracle cell fails where a loose random one works
- [Lattice-free CN pilot](archive/cryspr_lattice_free_cn_pilot_report.md) — the
  adverse selection that CN preselection produces in an expanded cell
- [Experiments plan](human_experiments_plan.md) — item 2 of the CrySPR list is
  proposal 6 here
