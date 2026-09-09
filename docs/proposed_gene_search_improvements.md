# Proposed: finding the lowest-energy structure for a gene

> **These are proposals.** Nothing here is implemented. Each item says what it
> rests on — a measurement of ours, a measurement from the archive, or an
> argument — and every item ends with what is unknown about it. The companion
> document [proposed_pyxtal_generation_fixes.md](proposed_pyxtal_generation_fixes.md)
> covers the *mechanics* of the sampler: whether it returns a structure at all
> and whether that structure honours the distance floor. This one is about the
> harder half: given that a structure comes back, how often is it the lowest-
> energy structure the gene leads to.

## The framing

A gene fixes the space group, the Wyckoff assignment and the composition. What
is left free is exactly

    x = (lattice dof, free Wyckoff coordinates)

which for a generated gene averages 2.29 + 5.3 ≈ 8 dimensions
([dof reduction study](archive/pyxtal_dof_reduction_study.md)) and reaches ~190
in the tail. So the gene defines a **small, symmetry-reduced surface** of
candidate structures, and PyXtal exposes it directly as a vector:
`get_1d_rep_x()` / `update_from_1d_rep(x)`. Searching that surface is a global
optimisation problem in ~8 dimensions — which is small, and is not what the
pipeline currently does on it. (Whether the surface is the whole search space is
the subject of the next section; it is not.)

What the pipeline currently does is i.i.d. random multistart: `n_trials`
independent draws of `x`, one local relaxation each, keep the lowest. That is
the weakest global optimiser there is, and it is the right tool only when you
cannot perturb an incumbent and descend again. We demonstrably can — the rattle
stage already does exactly that once.

### The gene is a seed, not a constraint

An important qualification, because it decides what the search space actually
is. The rattle deliberately **abandons** strict symmetry: it clears
`FixSymmetry`, displaces every atom, and relaxes unconstrained, in order to
reach a lower energy than the symmetric stationary point allows. So the
pipeline does not optimise on the gene's manifold — it uses the gene as a
symmetric *starting surface* and then leaves it. Two different objectives
follow, and they are not the same problem:

- **(a) the lowest energy on the manifold** — the best structure that actually
  has the generated space group and Wyckoff assignment. Search space: `x`,
  dimension ~8.
- **(b) the lowest energy reachable from the manifold** — symmetry allowed to
  break. Search space: the full 3N + 6 coordinates, entered from a symmetric
  seed.

(b) is what the pipeline pursues and what the `e_hull` readouts reward, since
nothing downstream requires the relaxed structure to keep the generated space
group. It is also, as the rattle's own numbers show, a *different* search from
(a) rather than a continuation of it, and the two need different moves. A
symmetry-constrained global optimiser would be solving (a) and would give up
exactly what the rattle buys; a purely symmetry-breaking search cannot explore
the manifold at all, because descending off it is irreversible.

### The search is what binds, not the ranking

Three results, all from the archive, point the same way:

- **Coverage ≈ delivery.** Under the ORB protocol, coverage and delivered
  recovery agree within 1.4 points, so almost nothing correct is generated and
  then discarded. A better re-ranker has nothing to find. ([dof reduction
  study](archive/pyxtal_dof_reduction_study.md))
- **One perturbation is worth a lot — but mostly for energy, not for
  identity.** Over 9990 trials of the reconstruction study the unconstrained
  stage took *zero* optimiser steps in 78.2% of them, while the rattle lowered
  the energy in 33.1% by a median of 186 meV/atom. Relaxation alone leaves the
  structure where it started; a finite move does not. Note the asymmetry
  though: the same rattle recovered only **99 further ground-state matches**,
  1.0% of trials, against energy gains in a third of them. Breaking symmetry
  from a symmetric seed descends into the nearest *distortion* of whatever
  structure you started on — it cannot find a different structure. Energy
  progress and progress towards the right structure are therefore largely
  disjoint here, which is the sharpest evidence that (a) and (b) above are
  different searches.
- **More distinct starts buy recovery.** On the 400-structure dof ≥ 6 cohort,
  the sampling ceiling goes from 51.8% at 5 trials to 63.8% at 10.
  ([oracle coordination study](archive/cryspr_oracle_coordination_study_report.md))

And the cost is concentrated exactly where the search space is large. From
`generated/upi73i4k/protocol/structures.csv`:

| Σ dof_pos | 0 | 1–2 | 3–5 | 6–10 | 11–20 | >20 |
|---|---:|---:|---:|---:|---:|---:|
| n | 493 | 565 | 706 | 446 | 213 | 77 |
| P(e_hull ≤ 0.1) | 0.408 | 0.300 | 0.214 | 0.144 | 0.113 | 0.167 |

## Proposal 1 — a two-level search: hop on the manifold, then break symmetry

Because descending off the manifold is irreversible, the search has to be
structured as two nested levels rather than one walk:

**Inner level, on the manifold.** Basin-hop in `x` with `FixSymmetry` attached:
propose a move in the free Wyckoff parameters and the lattice dof, relax under
the constraint, accept by a Metropolis test. This is a walk in ~8 dimensions
and it is the *only* level that can change which structure you are on — which
polyhedra connect to which — because every point it visits still has the
generated symmetry.

**Outer level, off the manifold.** For the incumbents the inner level produces,
run the existing symmetry-breaking descent: clear the constraint, rattle, relax
unconstrained, keep it if it wins. This is a polish, not an exploration: it
finds the nearest distortion of the structure handed to it.

The present pipeline is the degenerate case of this — a single random point on
the manifold, no inner moves, one outer step, greedy acceptance with a
1 meV/atom margin (`RATTLE_ACCEPT_EV_PER_ATOM`). The proposal keeps the outer
level exactly as it is and spends new budget on the inner level, which
currently gets none.

The ordering matters and is the whole point: symmetry-break too early and the
manifold is no longer reachable, so the ~8-dimensional search that could have
found the right basin never happens. Conversely, symmetry-breaking every
incumbent is wasteful when it improves only a third of them, so the outer level
should run on the best few structures the inner level finds rather than on all
of them.

A budget-neutral mapping, using the accounting from
[analyse_trials_by_dof](../scripts/analyse_trials_by_dof.py) (3.2
relaxation-stage-equivalents per trial, since the unconstrained stage takes zero
steps in 78.2% of trials): three trials ≈ 9.6 stages, and one inner
perturb-and-relax is ≈ 1 stage. So **replace the 3-trial arm with one full
trial, ~5 inner hops, and the symmetry-breaking descent on the best two
incumbents.** The schedule `0:1,2:2,*:3` maps onto "0 dof: unchanged" — a gene
with no free coordinates has no manifold to search, and only the outer level
applies — "1–2 dof: 1 trial + 2 hops", "≥3 dof: 1 trial + 5 hops".

`pyxtal.lego.basinhopping` already operates on this vector and carries an
adaptive step size and a Metropolis test, so the inner driver largely exists.

**Unknown:** the temperature and step size, which are the whole game in basin
hopping and have no defaults that transfer from molecular work; whether
`FixSymmetry` survives repeated detach/reattach across steps without drift; how
to split budget between the two levels, since the inner level's value scales
with dof and the outer level's does not; and whether the walk's serial
dependence costs more wall-clock than it buys, since independent trials
parallelise perfectly and a walk does not. That last point is a real cost on a
GPU-batched relaxation and may argue for several short walks rather than one
long one.

**A cheaper inner level worth trying first.** Inner moves do not have to be
scored by the MLIP. The soft-sphere objective of
[proposal 4 of the generation fixes](proposed_pyxtal_generation_fixes.md) runs
in ~0.5 s at dof 186 and milliseconds below that, against 44–64 s for an MLIP
relaxation. An inner level that explores the manifold under a cheap geometric
objective and hands only its distinct minima to the MLIP would cost almost
nothing — at the price of exploring the wrong surface, since packing is not
energy. That trade, cheap geometric candidates against expensive MLIP
evaluations, is the same economics as proposal 5.

## Proposal 2 — the inner move set, which is not the rattle

The rattle should be left alone. `perturb` draws an i.i.d. Gaussian of
σ = 0.05 Å on every atom plus a 0.01 symmetric cell strain, and that is the
right design for its job: to break symmetry you want a dense generic direction,
and 0.05 Å suffices because the only requirement is to leave a stationary
point. It is simply not a move for the inner level — it changes symmetry, which
is exactly what the inner level must not do, and 0.05 Å is far below the scale
at which a structure changes basin.

The inner moves are different animals, and all of them keep the structure on
the manifold by construction, since they act on `x`:

- **per-site moves**: perturb the free coordinates of one site (or a few) at a
  time, with a σ on the order of 0.3–0.5 Å, which the noise arms say is the
  scale at which basins are still recovered (0.5 Å recovers 0.955 overall,
  0.867 at dof > 10). Single-site moves matter more the larger the dof: in a
  186-dof gene a dense move of any magnitude points almost surely in a useless
  direction, while a single site's displacement is interpretable and reversible.
- **lattice-only moves**: perturb `x[:lattice.dof]` on its own. The oracle test
  found the cell and the coordinates jointly necessary — the true coordinates in
  a wrong-shaped cell (`volume_coords`, 0.420) are worse than random
  coordinates in PyXtal's own cell (0.500) — so a coordinate-only move set can
  never repair a bad shape.
- **whole-`x` restarts** when the walk stagnates, which is the current
  behaviour and remains the right move at that point.

**Unknown:** the mix, and the σ. Nothing here is measured; the 0.3–0.5 Å range
is borrowed from the noise arms, which perturbed *true* coordinates and so
measure how wide a basin is, not how far a good move should travel.

## Proposal 3 — seed the search from prototype retrieval

Of the candidates in the dof reduction study this is the cheapest by a wide
margin: no training, and the generated gene's element-anonymised prototype is
already present in LeMat-Bulk for 66.8% of the `upi73i4k` genes — 39.0% even at
dof > 10, the bin where reconstruction currently never succeeds. Where a
prototype exists, its free parameters transfer the framework nine times in ten,
and the noise arms say ~0.3 Å is close enough.

As a **seed for the search** it is strictly better than as a standalone method.
Where a prototype exists the walk starts in the right basin; where it does not
you fall back to a random draw and lose nothing. It also makes the "any learned
method must beat retrieval" baseline concrete and cheap to maintain.

**Unknown:** the retrieved parameters have to be rescaled to a predicted or
estimated volume, and `volume_coords` (true coordinates at the true volume in a
random *shape*, 0.420) came out *worse* than random coordinates in PyXtal's own
cell (0.500). So retrieval must bring the cell shape with it, not just the
coordinates. Whether the prototype's shape transfers across chemistry as well
as its framework does is not measured.

## Proposal 4 — a few deterministic snapped starts

Free coordinates in real structures sit at simple fractions `k/n` more often
than uniform, by 6–10 percentage points. That was correctly rejected as a basis
for a *learned marginal* — dropped into `generate_point` it would reproduce
`random_state.random(3)` almost exactly. But as a handful of extra
**deterministic starting points** in a multistart set it costs nothing: enumerate
a few candidates with each free coordinate snapped to `k/n` for small `n`, relax
them alongside the random draws.

This is the cheap shadow of candidate 4 (aristotype plus distortion
amplitudes): high-symmetry parent structures are exactly the ones whose free
parameters are simple fractions, and a snapped start lands on the parent for
free without any group–subgroup machinery.

**Unknown:** the combinatorics. With `d` free coordinates and `m` candidate
fractions each there are `m^d` snapped points, so this only works as a small
enumeration at low dof or as a per-site snap of an otherwise random draw. No
measurement.

## Proposal 5 — turn cheap proposals into coverage

A clash-free candidate from the soft-sphere solve costs ~0.02–0.07 s at low dof
and a median 0.56 s at dof 70–186 (proposal 4 of the generation-fixes
document). An MLIP relaxation costs 44–64 s per trial. The ratio is two to three
orders of magnitude, so proposals are effectively free and relaxations are not.

Proposal: generate hundreds of candidates per gene, cluster them (by structure
fingerprint, or by distance in `x` after accounting for the normaliser), and
relax one representative per cluster. Nothing in the present pipeline prevents
two of its trials landing in the same basin, and nothing measures whether they
do — while the 5-vs-10-trial ceiling says *distinct* proposals are what buy
recovery.

**Unknown:** the clustering metric. `x` is not a metric space in the way one
would like — the Euclidean normaliser makes some coordinate differences pure
gauge, and along a polar direction one site's coordinate is arbitrary
altogether. A distance-based fingerprint sidesteps that, at the cost of being
slower than comparing vectors.

## Proposal 6 — rescue CN preselection by fixing its scale

The [lattice-free CN pilot](archive/cryspr_lattice_free_cn_pilot_report.md)
found that ranking candidates by CrystalNN error against oracle coordination
numbers *collapsed* recovery from 36.0% to 8.0%, by adverse selection: in a
1.7× cell a uniform candidate reads CN 2–4 against a target of 8–12, so the only
way to score well is to clump atoms together and leave voids.

That is a property of the estimator, not of the idea. The signal itself is
strong in the same study — trials whose *final* CN MAE is 0 match the ground
truth 45.4% of the time against 0.29% otherwise, and coordination is close to a
necessary condition for recovery.

Proposal: score at the right density and relax from the loose cell. Rescale the
candidate's fractional coordinates to the estimated equilibrium volume, compute
CrystalNN there, select, then hand the **unscaled** loose candidate to the
relaxation so compressive annealing is untouched. A few lines in
`run_lattice_free_cn_study.py`.

**Unknown:** whether CrystalNN at an estimated rather than true equilibrium
volume is accurate enough to rank, given that the volume estimate itself has a
log-sd of 0.28. And the encouraging number on the other side — initial CN
MAE = 0 giving 28.6% recovery against 5.35% — rests on **14 trials out of
1975**, so it is a hint, not a measurement.

## Proposal 7 — stop adaptively

The trial schedule is a fixed prior over dof bins. With a search whose
incumbent improves monotonically, the natural rule is instead: continue while
the best energy is still improving, stop after `m` non-improving steps. That
reallocates budget from genes that are already solved — dof 0 recovers 98.9% at
one trial, and 42.3% of genes in the `upi73i4k` cohort have dof_pos ≤ 2 — to
the tail, at no change in total cost.

**Unknown:** `m`, and the bias it introduces. Adaptive stopping makes the
per-gene budget data-dependent, which breaks the clean "trials per gene"
accounting the [ranking protocol](de_novo_ranking_protocol.md) power analysis
rests on; the arms would need to be compared at equal *total* relaxations
rather than equal trials per gene.

## How to read the results, if any of this works

A better search lowers the energy, so it improves `e_above_hull` monotonically.
It does **not** necessarily improve match rate against a reference structure —
it will also find more structures *lower* than the reference, which the
reconstruction studies score as `lower_energy_alternative` rather than as a
recovery. That category already sits at 2.2% of the 400-structure cohort at 5
trials and 2.5% at 10.

The rattle is the existing proof of this divergence, and the reason to expect
it: it improves the energy in 33.1% of trials but converts only 1.0% into
ground-state matches. A symmetry-breaking move is an `e_hull` instrument, not a
recovery instrument, and the two levels of proposal 1 are expected to load onto
the two metrics differently — the inner, manifold level should move recovery,
the outer level should move energy. **Report them separately**, or the two
effects will cancel in a single headline number and the experiment will read as
null.

So the instruments will diverge, and the divergence is the expected signature
of success rather than a regression:

- the [ranking protocol](de_novo_ranking_protocol.md) and MetaSUN, which
  threshold `e_hull`, should improve;
- the reconstruction and oracle studies, which score a `StructureMatcher` fit
  to a known reference, may not, and their `lower_energy_alt` column should
  grow.

Any experiment here should report both, and should treat a rise in
`lower_energy_alternatives` as a positive result.

There is a second consequence of the gene being a seed, worth stating because
it affects bookkeeping rather than metrics. Once symmetry breaks, the generated
gene no longer describes the structure that comes out: its space group and
Wyckoff assignment are those of the distorted minimum. Nothing downstream is
broken by this — the `bawl` fingerprint, uniqueness and novelty are all computed
on the final relaxed structure — but any analysis that joins a *gene* property
to a *structure* outcome (the conditioning audits, the gene-level critic, the
censored `min(E | gene)` target) is joining across that break, and the
proportion of genes that survive it is not currently recorded. A
`spacegroup_changed` column on `structures.csv` would make it visible and costs
one spglib call per structure.

## Suggested first experiment

The apples-to-apples benchmark already exists: the 400 LeMat-Bulk structures
with total dof ≥ 6 used by the oracle studies, where `base5` = 50.8% recovery
and `base10` = 60.8% at 5 and 10 trials. Run one arm — proposal 1 with the
proposal 2 move set, budget-matched to `base5` at ~16 relaxation stages per
structure — against those two numbers. It is the cheapest way to find out
whether the framing in this document is right, it needs no model training, and
the harness exists: [`run_oracle_coordination_study.py`](../scripts/run_oracle_coordination_study.py)
runs the arms and [`analyze_oracle_coordination_study.py`](../scripts/analyze_oracle_coordination_study.py)
already produces the `base5` / `base10` comparison table a new arm would slot
into.

Report the two levels separately — recovery and energy after the inner hops,
then again after the symmetry-breaking descent. The rattle's 33.1%-energy /
1.0%-recovery split says the levels load onto different metrics, so a single
headline number can hide a real inner-level gain behind an unchanged total.

If the inner level beats `base5` at equal budget, everything else here is worth
doing. If it does not, the framing is wrong: the loss is in the proposal
distribution rather than in the search of it, and the effort belongs back on the
lattice head, where the oracle bound is 1.000 at every dof.

## See also

- [Proposed fixes to the gene → structure step](proposed_pyxtal_generation_fixes.md)
  — the sampler mechanics: failure taxonomy, the continuous solve, the
  volume/tolerance entanglement
- [The pair-tolerance bug](pyxtal_pair_tolerance_bug.md) — why 20% of current
  proposals start below their own distance floor
- [What else should WyFormer learn to take the guesswork out of PyXtal?](archive/pyxtal_dof_reduction_study.md)
  — the oracle bounds this document's ceiling is set by: true cell + true
  coordinates recover 1.000 at every dof
- [Oracle coordination number study](archive/cryspr_oracle_coordination_study_report.md)
  and [lattice-free CN pilot](archive/cryspr_lattice_free_cn_pilot_report.md) —
  the CN evidence behind proposal 6
- [The de novo ranking protocol](de_novo_ranking_protocol.md) — the instrument
  proposals 1 and 7 would be measured on
