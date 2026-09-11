# Where the reconstruction ceiling comes from, and four ways to attack it

The de novo protocol recovers 55.6% of the oracle cohort's structures and 13.3%
of the >10-DoF ones ([the NEP89 variants](de_novo_ranking_protocol_nep89_variants.md)).
This note records what is known about *why*, and four proposals with their costs,
so that the next person does not re-derive the arithmetic. One of them —
a permissive PyXtal tolerance — is being implemented and measured separately;
the rest are written down here because the reasoning is worth more than the
ranking.

## What the ceiling is made of

The oracle study ([`oracle_reconstruction.py`](../scripts/oracle_reconstruction.py),
results in `generated/oracle_reconstruction/RESULTS.md`) hands CrySPR pieces of
the answer and measures what each is worth. One trial per cell, at >10
positional DoF:

| arm | what is given | recovery |
|---|---|---|
| `random` | nothing — the production pipeline | 0.073 |
| `volume` | the true cell volume | 0.060 |
| `lattice` | the whole true cell, coordinates random | **0.173** |
| `coords_random_cell` | the true free coordinates, cell random | **0.160** |
| `volume_coords` | true coordinates *and* volume, shape random | 0.140 |
| `lattice_coords_noise_0.5` | both, coordinates jittered by 0.5 A | 0.867 |
| `lattice_coords_exact` | both | 1.000 |

Four things follow, and they constrain every proposal below.

**The cell and the coordinates are each worth about 2.3x, and only together are
they worth everything.** 0.073 -> 0.173 for the cell, 0.073 -> 0.160 for the
coordinates, 1.000 for both. So the ceiling of *any* coordinate-sampling
improvement, with today's cell distribution, is 0.160 at >10 DoF. That is the
number to compare a proposal against — not 1.000.

**The funnels are wide.** Half an angstrom of noise on every free coordinate
still recovers 0.867. A proposal does not need precision; it needs to land in
the right basin, and the basins are large. This is good news for search and bad
news for any story in which a slightly distorted proposal distribution is the
cause of the ceiling.

**Volume is not the cell.** `volume_coords` (0.140) is no better than
`coords_random_cell` (0.160): the true volume with a random *shape* buys
nothing. Whatever is hard about the cell is its shape.

**The `wide` arm already beats the coordinate oracle.** Widening the draw 10x
and selecting with NEP89 reaches 0.240 at >10 DoF, above the 0.160 that a
*perfect* coordinate sampler would reach with a single random cell. Improving
the joint draw over cell and coordinates is already outperforming the ceiling of
coordinate-only work.

## The fixed-lattice paradox, and why it does not condemn cell prediction

[`cryspr_oracle_fixed_lattice_report.md`](archive/cryspr_oracle_fixed_lattice_report.md)
found that pinning the cell to the ground truth *lowered* 5-trial recovery from
50.7% to 16.2%, which reads as evidence that a predicted cell would hurt. It is
not, and the report's own diagnosis says why. Three causes, only one of them
about the cell:

1. **Exclusion-volume choking: 39.8% of structures failed to generate at all.**
   `Tol_matrix(factor=1.3)` inflates exclusion spheres 2.2x in volume; in a dense
   equilibrium cell, sequential rejection sampling without backtracking collapses.
   The report recommends an adaptive tolerance ladder to remove it.
2. **Axis and setting discrepancies.** Non-standard space-group settings and axis
   permutations produced incompatible coordinate frames and crippled monoclinic
   recovery to 2.3% (3 of 132) — a third of the cohort, and a defect rather than
   physics.
3. **Strain-displacement coupling locked out.** On the 241 structures that did
   generate, fixed-*cell* relaxation recovered 27.0% against 49.8% lattice-free.

Only (3) is a real property of the cell, and it is about *freezing* the cell
during relaxation rather than about *starting* from the right one. The
`oracle_reconstruction.py` `lattice` arm gives the true cell as a starting point
with the cell still free, and recovery *rises* — 0.591 against 0.488 overall,
better in every DoF bin. **Starting from the true cell helps; clamping it
throughout hurts.** A cell WyFormer predicts would be a starting point, so the
fixed-lattice result does not bear on it.

## Idea 1 — a permissive PyXtal tolerance: tested, and a null

**Done, and the hypothesis is not supported.** Five factors from 1.3 to 0.1 over
the 750-gene cohort: reconstruction 0.576 (control) / 0.564 / 0.567 / 0.545 /
0.565, no arm better, the only nominally significant cell a *degradation* that
does not survive correction for five comparisons. Full write-up in
[pyxtal_tolerance_sweep.md](pyxtal_tolerance_sweep.md).

The null is worth trusting: the manipulation was large (median closest contact
fell 47%, and at factor 0.1 the draw is rejection-free and therefore uniform),
and the cell distribution stayed flat to within 1% across arms, so the
experiment isolated coordinate sampling as intended. It does buy cost — an 18x
cheaper draw and a 32% cheaper pre-relaxation — which matters for the wide arm
where PyXtal was 39% of the bill, but that is a cost argument and not a quality
one.

**What it rules out and what survives.** Rejection-sampling bias is not what
caps reconstruction, so the remaining coordinate-side proposal is a better
*proposal distribution* rather than a less constrained one — which is idea 2's
affordable variant below. And with the coordinate half of the problem now known
not to be a sampler artefact, the oracle's verdict that the cell is worth as much
as the coordinates carries more weight, not less.

The original hypothesis was that PyXtal's sequential rejection sampling induces an
idiosyncratic distribution that under-samples the dense configurations real
crystals occupy: orbits are placed one at a time and rejected against what is
already down, so the *joint* distribution is distorted even where every marginal
is uniform — and the DoF-reduction study checked only the marginals.

Two measurements make the change safe, and one makes it a clean test:

- **The floor is not enforced anyway.** 45% of returned draws have a closest
  contact below the nominal 1.3 floor, worst 0.378 in tolerance units, and only
  2 of 272 sampled violations were an atom against its own periodic image.
- **NEP89 does not care.** 0 failures across 18,000 draws, including 227 below
  0.8 tolerance units, because its ZBL core is strictly repulsive and well
  conditioned to 0.05 A. The historical reason for a distance floor — relaxers
  collapsing on overlapped input — does not apply to it.
- **At factor -> 0 there are no rejections**, so the draw *is* uniform. If
  recovery does not move, rejection bias is not the explanation for the ceiling,
  and that is worth knowing for a day of compute.

The confound to watch is whether PyXtal's retry loop accepts *smaller cells*
under a permissive tolerance, which would conflate the cell and coordinate
effects and stop it being a clean test of sampling bias.

## Idea 2 — Monte-Carlo tree search over atom placement

Place orbits sequentially as tree levels, with the reward the energy of the
relaxed complete structure, and back that reward up to the early placements.
The diagnosis is right: PyXtal's placement *is* a sequential decision process
whose reward (a 0/1 distance check) is both crude and available only locally,
and MCTS is the standard remedy for exactly that.

**The cost does not work.** A rollout is one NEP89 relaxation, ~3 s. A thousand
rollouts per gene is 50 minutes per gene, ~35 CPU-days for a 1000-gene cohort,
against a current NEP89 budget of ~3 s per gene — three to four orders of
magnitude. And the structural problem is worse than the arithmetic: search is
needed *precisely* where it is most expensive, since the low-DoF genes already
recover at 90% and need no search at all.

### The affordable variant: rank with a pair potential, relax the survivors

Replace the rollout's relaxation with a **pair-potential score**.
[`ScreenedMorse`](../src/wyckoff_transformer/cryspr/nep89.py) is already built
and validated for this: ZBL plus Morse, analytic energy, forces and stress,
~0.5 ms per evaluation, and its entire purpose is to answer "is this geometry
physically sane". So draw 1e4 to 1e5 candidate placements, score all of them,
and hand NEP89 only the best ~30. That is a **1e3 to 1e4 times wider search for
the same NEP89 budget** as today's `wide` arm.

It needs no tree. MCTS buys credit assignment across sequential decisions, which
is only worth paying for when evaluating a complete placement is expensive; at
0.5 ms it is not. And it is a strictly better proposal than rejection sampling
for the reason idea 1 is about: a soft score *ranks* candidates where a
tolerance *excludes* them.

The obvious extension, given the section above: draw over the **cell and the
coordinates jointly**. The symmetry-allowed cell has 1 to 6 free parameters
(one cubic, two tetragonal or hexagonal, three orthorhombic, four monoclinic,
six triclinic) against up to 20 coordinates, so it is the cheaper half of the
space to cover — and the oracle says it is worth as much.

## Idea 3 — port relaxation intermediates back into the search

Record the relaxation trajectory and label every point on it with the energy the
trajectory *reached*, so the search learns the basin-of-attraction map: "starting
anywhere along this path gets you E". The trajectory is already computed, so the
data is free, and under `FixSymmetry` every point on it lies in the same
free-coordinate space the search covers, so the points are directly usable.

**The information is the wrong kind on its own.** A relaxation path is a downhill
path, so all of its points lie in the basin already found. The failure mode being
attacked is not finding *other* basins, and local catchment information does not
help with that. That is measured, not assumed: the symmetry-constrained
basin-hopping arm returned a flat null — 0.555 against the baseline's 0.556, 7
genes won and 8 lost, *p* = 1.0 — despite 36,000 hops, and it found 7.3 distinct
minima per gene at >10 DoF against 27.2 for the same budget spent on independent
draws.

It becomes worth having once something proposes *distant* starts, which the wide
pair-potential draw above does. Then the catchment map is what turns a wide draw
into a cheap one, by predicting which of 1e5 candidates lead somewhere already
seen.

## Suggested order

Revised now that idea 1 has returned a null:

1. **The pair-potential-scored wide draw over cell and coordinates jointly.**
   The only coordinate-side proposal still standing, and it attacks the half of
   the space the oracle says is underexploited. Fits the existing budget.
2. **Idea 3** once (1) exists to give it distant starts.
3. **MCTS with relaxation rollouts** last, and only if the cheaper proposals
   plateau.
4. ~~Idea 1~~ — done; see above. Retain the *option* (`--pyxtal-tol-factor`) for
   its cost benefit in the wide arm, not for reconstruction.

Independently of all four: the fixed-lattice study's monoclinic 2.3% is an
unfixed defect in how a supplied cell is handed over, and it will bite any
attempt to use a predicted cell.

## See also

- [The de novo ranking protocol](de_novo_ranking_protocol.md)
- [The NEP89 variants](de_novo_ranking_protocol_nep89_variants.md) — where the arms and their numbers come from
- [The PyXtal tolerance sweep](pyxtal_tolerance_sweep.md) — idea 1, tested
- [The oracle fixed-lattice report](archive/cryspr_oracle_fixed_lattice_report.md)
- `generated/oracle_reconstruction/RESULTS.md` — the oracle arm table quoted above
