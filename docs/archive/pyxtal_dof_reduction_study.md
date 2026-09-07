# What else should WyFormer learn to take the guesswork out of PyXtal?

WyFormer emits a Wyckoff gene and CrySPR turns it into a structure by asking
`pyxtal.from_random` to guess everything the gene does not fix: the cell volume,
the cell shape, and every free coordinate. Recovery of a known structure at one
trial falls from 98.9% at zero positional degrees of freedom to 0% above ten
([ORB protocol](upi73i4k_orb_protocol.md)), and the loss is entirely in that
proposal step, not in selection. The question here is which of the guessed
quantities the model should learn to predict instead, and what to learn for the
coordinates short of a full symmetry-constrained coordinate generator.

Two things were done. First, PyXtal's own priors were measured against real
structures, layer by layer, to see where a learned distribution would even
help. Second, an oracle reconstruction test fed CrySPR the *true* value of each
quantity in turn, on 750 LeMat-Bulk structures stratified by dof, which bounds
what a perfect predictor of it could buy before anything is trained. The short
version: the cell and the coordinates are each nearly useless without the other
and together recover everything; the coordinates only need to be right to about
0.3 Å; the cell shape matters far more than the volume. Measurements are from
`scripts/analyse_gene_priors.py` on MP-20 and on the `upi73i4k` genes; the
oracle test is `scripts/oracle_reconstruction.py`.

## Where the randomness is

For a fixed gene, `random_crystal` (PyXtal `crystal.py`) draws in three layers:

| layer | code | prior | dof it guesses |
|---|---|---|---|
| volume | `set_volume` | per element, uniform between the covalent-radius and van der Waals sphere volumes, summed, ×`factor` (1.1) | 1 |
| cell shape | `generate_cellpara` | log-normal axis ratios of width 0.35 per axis; monoclinic β ~ N(90°, 20°) truncated to 30–150°; triclinic from a random shear | 0 (cubic) to 5 (triclinic) |
| coordinates | `generate_point` + `_set_ion_wyckoffs` | uniform in the cell, projected onto the Wyckoff position, rejected if closer than `Tol_matrix` allows to any placed atom | Σ site dof, mean 5.3 for generated genes |

CrySPR passes `Tol_matrix(prototype="atomic", factor=1.3)`, i.e. a minimum
distance of 0.65 × the sum of covalent radii, and never passes `factor`, so the
volume gets PyXtal's default 1.1. Neither constant is exposed as a parameter.

## PyXtal's priors against the data

6000 MP-20 training structures, parsed with `pyxtal.from_seed` at the training
set's tolerance (`kick_pyxtal_until_it_works`, tol 0.1), 35,825 free
coordinates from the 4843 structures that have any.

### Volume and shape: badly centred, worth learning

| quantity | PyXtal draws | data | n |
|---|---|---|---|
| true volume / PyXtal's mean estimate | 1 by construction | geo-mean **0.822**, IQR 0.678–0.966, log-sd 0.28 | 6000 |
| volume per atom | — | median 18.2 Å³, log-sd 0.39 | 6000 |
| log(c/a), tetragonal/hexagonal/trigonal | N(0, 0.61) | mean **0.56**, sd 0.79 | 2331 |
| monoclinic β | N(90°, 20°) | mean **109°**, sd 12.6°, p5 90.2°, p95 130° | 868 |
| orthorhombic sd of log(a, b, c) in a cell | 0.35 | median 0.315 | 1164 |

The volume ratio is the same 18% inflation the ORB protocol measured on
generated genes from the other side (initial / reference 1.177, with only 25%
of trials starting within 10% of the right volume), and its log-sd of 0.28 means
the composition-based estimate is not merely biased but loose. The axial
ratio prior is centred on c/a = 1 while the data centre on c/a ≈ 1.75. The
monoclinic prior spends half its mass below 90°, where the standardised data
essentially never are. For orthorhombic cells the spread matches, but which
length goes to which axis is drawn at random and is exactly what the Wyckoff
letters fix.

### Coordinates: the marginals are uniform

Share of free coordinates within a tolerance of a simple fraction k/n, against
what a uniform distribution would give:

| denominators | tol 0.005 | tol 0.01 | tol 0.02 |
|---|---:|---:|---:|
| n ∈ {1, 2, 4} | 0.101 vs 0.040 | 0.153 vs 0.080 | 0.235 vs 0.160 |
| n ≤ 8 | 0.208 vs 0.120 | 0.337 vs 0.240 | 0.567 vs 0.480 |
| n ≤ 12 | 0.323 vs 0.240 | 0.546 vs 0.467 | 0.839 vs 0.780 |

There is an excess at simple fractions, but it is 6–10 percentage points, and
the 24-bin histogram of all free coordinates ranges from 926 to 2098 against a
uniform 1492. **A learned per-coordinate marginal dropped into
`generate_point` would reproduce almost exactly what `random_state.random(3)`
already does.** Whatever information there is about a coordinate sits in its
conditional on the other sites and the cell, so only a model that couples them
can help. This closes the most literal reading of "parametrise PyXtal's
distributions": it works for the lattice layer and not for the coordinate
layer.

### The polar gauge

68 of the 230 space groups are polar. In MP-20 they hold 15.8% of the
structures but **45.6% of the positional degrees of freedom**, because low
symmetry means both a polar direction and many free coordinates. Along a polar
direction the Euclidean normalizer contains a continuous translation, so the
origin is a gauge choice and the corresponding coordinate of any *one* site is
arbitrary. `pyxtal.from_seed` standardisation does not fix it: translating a
P4mm cell by z + 0.137 and re-standardising shifts every free z by 0.137,
whereas the same experiment in Pnma is undone exactly. PyXtal's
`wyckoff_sets.json`, which drives the enumeration augmentation, lists the
discrete cosets (half-translations, inversions) and not these continuous ones.
Any target expressed in coordinates must first choose a gauge; any target
expressed in distances need not.

### How much does the gene already determine?

For MP-20 test structures with at least one free coordinate, does a training
structure with the same *element-anonymised prototype* — same space group, same
Wyckoff letters, same partition of sites into species, elements dropped —
already have the right geometry? 2000 test structures, up to 10 candidates each,
`StructureMatcher` with `FrameworkComparator` (species ignored):

| Σ dof | framework match, default tol | match at CDVAE tol only | prototype exists, geometry differs | no prototype in train | n |
|---|---:|---:|---:|---:|---:|
| 1–2 | **0.933** | 0.011 | 0.014 | 0.042 | 712 |
| 3–5 | **0.770** | 0.021 | 0.014 | 0.196 | 439 |
| 6–10 | **0.678** | 0.022 | 0.086 | 0.214 | 453 |
| >10 | **0.487** | 0.018 | 0.104 | 0.391 | 396 |
| all | **0.751** | 0.017 | 0.048 | 0.184 | 2000 |

Where a prototype exists at all it transfers the framework nine times in ten
(0.751 of 0.816). Different elements, same gene, same structure: the
conditional distribution of the free parameters given the gene is close to a
point mass even across chemistry, which the ORB protocol saw within a single
composition (98.4% of known genes map to one structure). The failure to
transfer grows with dof, but slowly.

The same question for the 2500 `upi73i4k` genes against all of LeMat-Bulk
(3,959,797 augmented fingerprints, 33,265 distinct anonymised prototypes):

| Σ dof | exact gene known | anonymised prototype known | n |
|---|---:|---:|---:|
| 0 | 0.639 | 0.955 | 493 |
| 1–2 | 0.319 | 0.804 | 565 |
| 3–5 | 0.135 | 0.598 | 706 |
| 6–10 | 0.070 | 0.473 | 446 |
| >10 | 0.021 | 0.390 | 290 |
| all | 0.251 | **0.668** | 2500 |

The exact-gene column reproduces the protocol's 627 of 2500. Two thirds of what
the model generates has a known prototype; in the bin where reconstruction
currently never succeeds, four in ten do.

## Candidates

Ranked by how much of the guessing they remove per unit of new machinery.

### 1. A lattice head: volume and shape together

Predict `Lattice.encode()` — `[a]`, `[a, c]`, `[a, b, c]`, `[a, b, c, β]` or all
six by lattice system — plus log volume, from the complete gene, as a
heteroscedastic regression (mean and log-sigma) so that the trial budget can
sample from the predicted distribution rather than a point. This removes every
lattice degree of freedom, mean 2.29 per generated gene, and replaces the three
mis-centred priors above with conditional ones. The target is near-deterministic
given the gene, and the enumeration augmentation leaves it untouched because
PyXtal's cosets are translations and inversions. Practically it is the existing
`Scalar` target path with a vector target; the caches hold no lattices, so the
targets have to be added in `structure_to_sites` from the pyxtal object at
caching time. The volume alone is item 4 of the backbone plan; the shape comes
at no extra architectural cost and the data say it is at least as wrong.

### 2. Invariant pairwise targets and a cheap solve

For every pair of Wyckoff orbits (i, j), predict the shortest inter-orbit
distance and one or two further shells — a small distogram, N(N+1)/2 outputs
for N sites from pairwise attention features. Distances are invariant to the
polar gauge, to normalizer relabelling and to site order, so the target is
single-valued and a plain regression fits it; the transfer numbers above say it
is nearly a function of the gene. Generation then replaces rejection sampling
with optimisation: minimise the mismatch between the realised and predicted
distances over the dof_pos + dof_lat variables from thousands of random starts,
which costs no MLIP calls, and relax only the best few. That changes how
recovery scales with dof rather than paying for it with trials. It needs a
pairwise output head and a small solver, and it is naturally a separate
"realizer" model conditioned on the finished gene with bidirectional
attention, so the autoregressive backbone stays discrete and fast.

### 3. Prototype retrieval as the prior and the baseline

Zero training: look up the generated gene's anonymised prototype in LeMat-Bulk,
take the free parameters of the nearest entry, rescale to the predicted (or
composition-estimated) volume, relax. By the tables above it would start about
four in ten of the currently hopeless dof >10 genes in the right basin and most
of everything below. Any learned method has to beat this on the genes it does
not cover — a third overall, six in ten at dof >10 — and the retrieval itself
is the cheapest way to build training targets and diagnostics for the learned
one.

### 4. Aristotype plus distortion amplitudes

Describe each low-symmetry structure as a high-symmetry parent with atoms on
special positions plus small symmetry-adapted displacements, and learn the
parent Wyckoff assignment (categorical) and the amplitudes (small, unimodal).
PyXtal has the group–subgroup machinery in `supergroup.py`. Elegant, but the
search is expensive, not every structure is pseudo-symmetric, and it is a
research programme rather than a next step.

### Rejected

- **Learned coordinate marginals inside `generate_point`.** Dead by
  measurement: the marginals are uniform.
- **Coarse coordinate tokens as extra cascade fields** (16–32 bins per free
  coordinate, masked by the frozen axes). Discarded by decision. It handles
  multimodality and fits the cascade, but it needs the polar gauge fixed to a
  reference site for nearly half of all free coordinates, co-transformation of
  coordinates under the enumeration augmentation, and a variable number of
  outputs per site — it is the raw-coordinate generator with a coarser
  codebook, which is the thing this study set out to avoid.
- **A learned tolerance matrix.** It sharpens the rejection, but rejection
  sampling still fails exponentially in dof; the fix has to change the proposal
  or the search, not the filter.
- **A learned re-ranker over PyXtal trials.** Closed by the ORB protocol:
  coverage and delivered recovery agree within 1.4 points, so there is almost
  no correct answer being generated and then discarded.

## Oracle reconstruction test

Before building any of the above, measure what each would be worth if it were
perfect. PyXtal accepts a fixed cell (`lattice=`) and fixed generator positions
(`pyxtal.build`), so the true value of each quantity can be fed to the same
CrySPR relaxation and matched against the same reference. The design follows
the [reconstruction study](cryspr_reconstruction_study.md): reference structures
from LeMat-Bulk with `e_hull ≤ 0.1`, relaxed with ORB (`orb_conserv_inf`, on
CPU) under the two-stage symmetric schedule so target and trial share one PES;
gene, cell and free coordinates extracted from the relaxed reference; one trial
per arm; default `StructureMatcher`; the study's verdicts; stratified by
positional dof.

| arm | volume | shape | coordinates | what it bounds |
|---|---|---|---|---|
| `random` | PyXtal | PyXtal | PyXtal | current CrySPR, the baseline |
| `volume` | true | PyXtal | PyXtal | a volume head |
| `lattice` | true | true | PyXtal | the full lattice head |
| `coords_random_cell` | PyXtal | PyXtal | true | coordinates without the cell |
| `lattice_coords_exact` | true | true | true | control, must recover |
| `lattice_coords_noise_σ`, σ = 0.1, 0.2, 0.3, 0.5 Å | true | true | true + Gaussian noise | the precision a distogram solve or retrieval must reach |

Noise is applied to the free coordinates only, in Å converted by the
corresponding axis length, so the Wyckoff position is preserved. Dof-0
structures run only the first three arms. The driver is
`scripts/oracle_reconstruction.py`; results land in
`generated/oracle_reconstruction/` with a `RESULTS.md`.

### Results

Run 2026-09-05: 750 references, 150 per dof bin, drawn uniformly from the
1,388,197 LeMat-Bulk entries with `e_hull ≤ 0.1` and capped at 80 atoms in the
conventional cell; 6450 trials; ORB on CPU. Full tables, the protocol and the
timing are in `generated/oracle_reconstruction/RESULTS.md`, one row per trial
in `trials.csv`.

Recovery at one trial, by arm and Σ dof, 150 references per cell ("all" is over
750 for the first three arms and 600 for the rest):

| arm | 0 | 1–2 | 3–5 | 6–10 | >10 | all |
|---|---:|---:|---:|---:|---:|---:|
| `random` | 0.920 | 0.780 | 0.440 | 0.227 | 0.073 | 0.488 |
| `volume` | 0.947 | 0.780 | 0.527 | 0.220 | 0.060 | 0.507 |
| `lattice` | **1.000** | 0.880 | 0.527 | 0.373 | 0.173 | 0.591 |
| `coords_random_cell` | — | 0.820 | 0.653 | 0.367 | 0.160 | 0.500 |
| `volume_coords` | — | 0.767 | 0.467 | 0.307 | 0.140 | 0.420 |
| `lattice_coords_exact` (control) | — | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| `lattice_coords_noise_0.1` | — | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `lattice_coords_noise_0.2` | — | 0.993 | 0.987 | 0.993 | 0.993 | 0.992 |
| `lattice_coords_noise_0.3` | — | 1.000 | 0.993 | 0.993 | 0.973 | 0.990 |
| `lattice_coords_noise_0.5` | — | 1.000 | 0.980 | 0.973 | 0.867 | 0.955 |

Harness checks. The `random` arm reproduces the known-gene curve of the ORB
protocol (0.989 / 0.760 / 0.505 / 0.233 / 0.000) in shape, on a different
population and with 150 rather than 6 genes in the last bin. The control
recovers 600 of 600, and before any relaxation `pyxtal.build` with the true cell
and positions matched the reference for 750 of 750, so the gene round trip
introduces no setting or origin error. 10 of 6450 trials failed in PyXtal
generation, none in relaxation, none tripped the |E| > 50 eV/atom guard, and
the `scale=False` match differs from the default one in 5 trials. The space
group changed under the reference relaxation for 2 of 750.

What it says:

1. **The cell and the coordinates are jointly necessary and jointly
   sufficient.** At dof >10 the true cell alone gives 0.173, the true
   coordinates alone 0.160, both together 1.000. Neither half is "the
   bottleneck": a random cell wastes perfect coordinates and random coordinates
   waste a perfect cell. Every partial arm still falls about five-fold from dof
   1–2 to dof >10, while both complete arms are flat at 1.000, so the dof
   dependence is the size of the search space left open and nothing intrinsic
   to complex genes.
2. **The lattice head is worth about ten points on its own and closes dof 0
   completely.** From 0.920 to 1.000 at dof 0: the 8% of dof-0 references the
   current pipeline misses are cell-shape failures, since there is nothing else
   to guess. The volume alone is worth two points overall and nothing above dof
   2. That settles volume-versus-lattice in favour of the whole cell.
3. **Coordinates need to be roughly right, not precise.** 0.1 Å of Gaussian
   noise on every free coordinate costs nothing, 0.3 Å costs one point, and
   0.5 Å still recovers 0.955 overall and 0.867 at dof >10. The basins are
   wide. A realizer — a distogram solve or a retrieved prototype — has to land
   within about 0.3 Å, which is the scale at which a `StructureMatcher`
   framework match at default tolerances already puts a same-prototype
   structure (transfer table above).
4. **Fractional coordinates are only meaningful in the cell they belong to.**
   True coordinates at the true volume in a random *shape* (`volume_coords`,
   0.420) are worse than true coordinates in PyXtal's own random cell
   (`coords_random_cell`, 0.500; McNemar p = 3·10⁻⁴ paired over the same
   structures). `pyxtal.build` applies no distance rejection, and a wrong shape
   at the right volume packs the true fractional coordinates into overlaps
   (median minimum interatomic distance 1.77 Å, 12% below 1 Å, against 2.15 Å
   and 2.4% in PyXtal's 13%-larger cell) that relax to nonsense several eV/atom
   high. The cell and the coordinates therefore have to be predicted, or solved,
   together. A distance-based target does that by construction; a per-site
   coordinate head would not.
5. **Knowing the answer also makes the relaxation cheap.** Mean wall time per
   trial falls from 64 s for `random` to 44 s with the true cell, 24 s with the
   true cell and 0.5 Å coordinates, and 1–2 s with everything exact (timings at
   32 workers on 24 physical cores, so relative values only). A realizer pays
   for itself in relaxation budget as well as in recovery.
6. The association between recovery and the initial volume ratio is weak here
   (r = 0.10 with the log ratio; only the <0.8 and >2.0 tails are clearly bad,
   at 0.32 and 0.33 against about 0.5 elsewhere), weaker than the ORB protocol
   found on generated genes. The populations differ. Either way the volume by
   itself is not the lever.

Caveats: best-of-1 throughout, so the partial arms are lower bounds on what the
same information buys at a larger trial budget, and the penalty is
complexity-dependent; the reference is an ORB minimum, not the DFT geometry;
cells are capped at 80 conventional atoms, which excludes the largest generated
genes; and the noise arms perturb the free coordinates of the *true* Wyckoff
positions, so they measure precision, not the cost of choosing a wrong orbit.

### What this changes

The ranking above stands, with numbers attached. The lattice head (candidate 1)
is worth about ten points of one-trial recovery on its own and is a prerequisite
for anything done about the coordinates. For the coordinates (candidates 2 and
3), the target precision is about 0.3 Å and the cell has to come with them,
which is an argument for pairwise distances — invariant to the gauge and
constraining cell and coordinates at once — over any per-site coordinate output,
and it is the precision at which prototype retrieval already delivers where a
prototype exists. Predicting the volume alone, the original plan item, does not
move recovery and should be folded into the lattice head, as the experiments
plan now does.

## A note on preconditioned optimizers

ASE's `PreconLBFGS` with the `Exp` preconditioner (Packwood et al., J. Chem.
Phys. 144, 164109) is orthogonal to the problem above: an optimizer converges to
the minimum it starts near, and the loss is in where the start is. It is still
worth a benchmark for the tail, for two reasons. It reverts to plain LBFGS below
100 atoms, so it can only ever help the large cells — which are also the ones
that cost 50 s each and hit `steps_limit` — and cheaper high-dof relaxations
make the dof-aware trial budget cheaper. Its Armijo line search may also remove some of the collapsed-cell
failures. Two cautions: `variable_cell=True` wraps the atoms in `UnitCellFilter`,
not the `FrechetCellFilter` the relaxer uses, and its interaction with
`FixSymmetry` is untested — pass `precon=Exp(A=3)` and keep the existing filter
and constraint. A step-count comparison on the logged dof >10 trials of
`upi73i4k` is cheap.

## Reproduce

```bash
uv run python scripts/analyse_gene_priors.py priors --n 6000            # ~1 min on 16 cores
uv run python scripts/analyse_gene_priors.py prototype-transfer --n 2000 # ~3 min on 16 cores
uv run python scripts/analyse_gene_priors.py prototype-coverage          # ~3 min, loads the 4M-fingerprint cache

# Oracle test: CPU-only ORB, resumable; 16 workers fit the 24 physical cores beside
# the usual background load. The reported run took 72 worker-hours.
CUDA_VISIBLE_DEVICES="" uv run python scripts/oracle_reconstruction.py prepare --workers 16 --pool-size 4000 --per-bin 150
CUDA_VISIBLE_DEVICES="" uv run python scripts/oracle_reconstruction.py run --workers 16 --volume-coords
CUDA_VISIBLE_DEVICES="" uv run python scripts/oracle_reconstruction.py report
```

## See also

- [`upi73i4k` under the ORB ranking protocol](upi73i4k_orb_protocol.md) — recovery by dof, the volume inflation, the trial budget
- [How much does Wyckoff → structure cost us?](cryspr_reconstruction_study.md) — the reconstruction protocol the oracle test follows
- [CrySPR trial and stage spread](cryspr_trial_and_stage_spread.md) — why stage 3 is skipped here
- [Experiments plan](experiments_plan.md) — the lattice-prediction and Wyckoff-regressor items
