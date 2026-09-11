# What an unconverged relaxation costs, and whether to condition on it

Study run 2026-09-10/11. The question behind it: LeMat-Bulk rows differ in how converged
their relaxations were, and [`dirty_data_conditioning.md`](dirty_data_conditioning.md)
answers that by *telling* the model, through a `max_force` conditioning channel, rather
than by filtering. "What would this structure's energy be had its relaxation converged
fully?" is a legitimate question, and `max_force = 0` at generation time is a legitimate
way to ask it — provided `max_force` measures convergence and the energy it implies is
worth correcting.

Neither holds. The conclusions, in the order they were established:

1. **`max_force` is a provenance label, not a convergence label.** It separates source
   databases far more sharply than it separates converged from unconverged rows.
2. **The 30,679 rows with no forces were never missing at the source.** They are
   recoverable from Materials Project, and the imputation that stood in for them was
   wrong by a factor of two.
3. **Residual *stress*, not force, carries what an unfinished relaxation left behind** —
   about 98% of it, including on the 28% of rows whose force is identically zero.
4. **That energy is real, but negligible:** a population-weighted median of 0.08 meV/atom,
   against the 100 meV/atom metastability threshold (`evaluation/protocol.py`). It would
   move 0.06% of rows across that threshold.
5. **So: no correction, and no convergence channel.** Filtering on `max_force` would be
   worse than either, because it is the provenance filter this dataset exists to avoid.

## 1. Why a plain MLIP relaxation cannot answer the question

Relaxing an archived structure with an MLIP and taking the energy it releases measures the
MLIP's disagreement with DFT as much as the residual DFT forces: a DFT-relaxed structure is
not at the MLIP's minimum, and the MLIP would release energy from a perfectly converged row.
With MLIP force errors of tens of meV/Å against residual forces of a few meV/Å, the
artefact dominates the signal. Subtracting a converged control group does not fix it — the
disagreement is chemistry-dependent and correlates with the source database.

`wyckoff_transformer.cryspr.gradient_matched` removes the first-order part instead. The MLIP
surface is shifted by a term linear in displacement and strain, chosen per atom and per
component so its gradient at the archived geometry equals the archived DFT forces and stress:

```
E_corr(x) = E_MLIP(x) - dF . u(x) + V0 * dsigma : E_GL(x)
```

The MLIP's own minimum drops out. What remains is its *curvature*, so the estimate goes to
zero as the DFT gradient does, and the error becomes multiplicative rather than additive.
Three estimates come out of it: a symmetry-fixed relaxation, a Newton step `1/2 g'H^-1 g`
in the symmetry-allowed translation-free subspace, and the Cauchy-Schwarz lower bound
`1/2 |g|^4 / (g'Hg)`, which needs curvature along one direction only and whose curvature
source is swappable for DFT.

Units matter here and were established empirically: LeMat-Bulk `forces` are eV/Å, and
`stress_tensor` is **kBar in VASP's sign** (positive = compressed), the opposite of ASE's.

## 2. The forces that were never missing

30,679 rows — all Materials Project, 22.1% of it — carry an empty `forces` array *and* an
empty `stress_tensor`. No other row lacks either. They are not absent upstream: LeMaterial
read the task document's top-level `output.forces`, which MP's 2013–2017 legacy tasks leave
empty, while `calcs_reversed[0].output.ionic_steps[-1]` of the same calculation keeps both.

`scripts/recover_mp_forces.py` reads them from MP's public S3 task documents
(`materialsproject-parsed`, no credentials), matching on geometry (lattice within 4.9e-7 Å,
positions within 6.1e-6 Å) *and* total energy (within 8.5e-6 eV), because forces are only
valid at the geometry that produced them. It recovers 30,676 of 30,679; the remaining three
have two tasks sharing a timestamp and composition.

Two checks make this trustworthy:

- **Control.** On 800 MP rows that were never missing, the same extraction reproduces the
  archived forces and stress bit for bit.
- **Same calculation.** For every recovered row the forces come from the ionic step whose
  energy *is* LeMat's energy (`output.energy` equals the last step's `e_fr_energy` or
  `e_wo_entrp` exactly), at a geometry matching `output.structure` to 3e-14 Å, with no run
  hitting its ionic or electronic step limit. They are not a relaxation step later
  superseded by a differently configured static run.

The imputation they replace was wrong by 2×: the recovered median `max_force` is 0.088 eV/Å
against the 0.0417 imputed, and 356 rows exceed the dataset's own 1 eV/Å cut — they had been
training under a fabricated value that kept them in.

## 3. Stress, not force

The pilot (`scripts/gradient_matched_pilot.py`) ran 1,004 rows of `lemat_bulk_fmax1`,
stratified by source × force bin × stress bin, ≤40 sites, on conservative ORB
(`orb_v3_conservative_inf_mpa`, float64, CPU; 10.9 CPU-hours). 841 rows were well posed.

The cell block supplies a median **98%** of the energy released. δ rank-correlates 0.81
with residual stress against 0.31 with residual force. Splitting the gradient into force,
hydrostatic-stress and deviatoric-stress targets over the same ORB Hessian
(`cache/gradient_matched/stress_split/`): of the 110 rows above 5 meV/atom, hydrostatic
stress dominates 90, force 10, deviatoric 10, with negligible cross terms.

This also explains why `max_force = 0` is not a convergence request. 28.1% of the archive
has forces identically zero — every site on a Wyckoff position with no free coordinate, so
the site symmetry leaves no invariant vector — yet stress is *never* exactly zero, and
those rows still release a median 5.3 meV/atom above 20 kBar.

Residual stress is itself provenance-laden:

| hydrostatic stress, kBar (VASP sign) | Alexandria | MP | OQMD |
| --- | ---: | ---: | ---: |
| median | +1.4 | +0.02 | +7.5 |
| p99 | +12 | +14 | +75 |
| positive, of rows with \|p\| > 5 | 89% | 50% | 97% |

MP's is sign-symmetric, which is what under-convergence looks like. OQMD's is one-signed
compression, uncorrelated with residual force (Spearman 0.09), scaling with pseudopotential
hardness (median +62 kBar with F, +22 to +24 with O or N, +3 to +4 with K, Cs, I, Br).

## 4. What the DFT trajectories say

Two archives of full relaxation paths were extracted to settle whether ORB's curvature can
be trusted — in particular whether the ~10% of rows where ORB finds *negative* curvature
invalidate the method (`scripts/extract_mp_trajectories.py`,
`scripts/extract_alexandria_trajectories.py`; evaluated by
`wyckoff_transformer.cryspr.trajectory_validation`):

- **MP:** 1,040 tasks, every calc and every ionic step, 15,465 usable step pairs, 890 clean
  restart boundaries. 1,039 reproduce LeMat's final forces, stress and energy exactly.
- **Alexandria:** 500 structures from its own `geo_opt_paths` dumps, which record ENMAX,
  ENAUG, PREC and k-points per run; 33,498 step pairs.

**ORB's curvature is sound.** Against consecutive DFT steps within one calculation
(Alexandria, the cleaner reference):

| direction | ORB/DFT curvature, median | within ×2 |
| --- | ---: | ---: |
| positions | 0.91 | 98% |
| isotropic strain | 0.90 | 97% |
| deviatoric strain | 0.86 | 93% |

Stable across noise thresholds, chemistry and distance from the end of the path. End to
end — predicting the actual DFT drop from intermediate steps — the Newton estimate is
unbiased on MP (median ratio 0.99) and 11% high on Alexandria, as ~0.9 curvature predicts.
An interim "0.60 isotropic softening" measured on MP was an artefact of ISMEAR −5/0
reference data, where the reported stress and the energy change disagree by ~16%; such
trajectories are not a usable curvature reference.

**ORB's negative modes are numerical, not physical.** Where DFT actually moved along one:
MP 29 such modes, DFT curvature positive in 90–100%; Alexandria 6 of 6 positive. A
DFT-only secant Hessian finds a negative eigenvalue in 1% of tasks against ORB's 2%. The
fix is to clamp soft and negative modes and fall back to the lower bound, not to abort.

**The Pulay picture holds, and bounds the one unobservable term.** At a restart the plane-wave
basis is rebuilt for the current cell even when ENCUT is unchanged, so the energy jump at
identical geometry measures the staleness the previous run accumulated. It vanishes as the
preceding volume change does, takes the opposite sign (97–100% of expansions), and grows with
hardness. Fitting `E_jump ≈ -sigma_P * V * dlnV` gives an intrinsic Pulay stress of ~5 kBar
(F), ~3 (O/N), <1 (other) at MP settings, larger and less certain for Alexandria (medians
8.9 / 3.8 / 2.7, robust fits 2.5 / 2.1 / 1.0). It raises δ by 2–18%.

**The one-signed offsets are a change of settings, not an unfinished relaxation** — and the
energy they imply is still real. For Alexandria this is direct: LeMat's energy, forces and
stress match *no* step of the published path (0 of 500) but sit at exactly its final
geometry (500 of 500). They come from a separate calculation whose settings are not
published. LeMat's pressure tracks the jump between that calculation and the path
(Spearman 0.993) and not the path's own final pressure (0.04); the path itself ends below
0.23 kBar for 90% of rows, releasing ~0.001 meV/atom. For OQMD, reproducing its median
offsets from a stale basis at MP-measured slopes would need relaxation volume changes of
81% (F), 60% (O/N) and 3270% (other) — impossible, so the same explanation applies.

Either way the stress and the energy come from one calculation, so the stress is the
gradient of the labelled energy and the drop it implies is real *on the surface the label
lives on*. That is exactly the counterfactual the conditioning channel was meant to express.

## 5. Why it is not worth correcting

| | meV/atom |
| --- | ---: |
| population-weighted median δ | 0.08 |
| rows above 1 meV/atom | 14% (4.2% by the analytic estimate over all 5.3M rows) |
| rows above 5 meV/atom | 3.2% as computed; 4–6% calibrated for softening and `sigma_P` |
| `METASTABLE_THRESHOLD` | **100** |

Applying the analytic estimate `delta = V(p + sigma_P)^2 / 2B + deviatoric` to every row and
asking how many change side of a threshold:

| threshold | rows crossing into "stable" | share |
| --- | ---: | ---: |
| `e_hull <= 0.1` | 3,054 | 0.058% |
| `e_hull <= 0.05` | 3,846 | 0.073% |
| `e_hull <= 0` | 8,297 | 0.156% |

Even among rows within 20 meV *above* the 100 meV line, only 1.2% cross. The source
asymmetry survives only as a ratio (1.03% of OQMD rows cross zero against 0.04% of
Alexandria's) over numbers too small to matter. And the 100 meV threshold is applied to
*generated* structures against an MLIP hull, not to these labels; the labels feed
conditioning, where the model has not demonstrated resolution below ~25 meV
(`archive/upi73i4k_ehull_conditioning_audit.md`).

A full-archive run was costed at ~57,000 CPU-hours for exact Newton estimates, or ~3,000
tiered (analytic everywhere, ORB only above 0.5 meV). Neither buys anything the model can see.

## 6. Why not filter the bad tail either

A `max_force <= 0.1` cut is tempting and is the same trap as `<= 0.02`, in sharper form:

| | Alexandria | MP | OQMD |
| --- | ---: | ---: | ---: |
| rows above 0.1 eV/Å | 0.006% | **31.7%** | 3.1% |

It costs a third of Materials Project — the experimentally grounded rows — to remove 1.2%
of the archive.

And it does not find the rows that are actually damaged. What matters for WyFormer is not
the energy but the *gene*: a structure far from its minimum may carry a Wyckoff gene that
further relaxation would change, and that is an error in the target itself, which no energy
correction repairs. In the pilot, 11 of 900 relaxed rows changed space group at the
tolerance the dataset's genes use (symprec 0.1) — about 1% population-weighted, though on
11 events the uncertainty is roughly ±0.6% and two of the changes went to P1, which looks
more like the MLIP breaking symmetry than a finding. **Eight of the 11 had
`max_force <= 0.1`**, and the two largest were symmetry-locked rows with high stress:

| row | `max_force` | stress | space group | δ |
| --- | ---: | ---: | --- | ---: |
| `oqmd-6437972` | 0.002 | >20 kBar | 191 → 65 | 54 meV/atom |
| `agm002374661` | 0.005 | >20 kBar | 194 → 186 | 57 meV/atom |

Gene changes run at 1.5–2.3% in every force bin with no trend. If this ~1% label noise is
ever worth acting on, the operation is to *recompute* — relax and re-derive the gene for the
~2.3% of rows predicted to move more than 0.1 Å, which is predictable from force and stress
analytically and is provenance-neutral — not to delete a third of MP.

## 7. What this leaves

`lemat_bulk_fmax1_stress` (see [`dirty_data_conditioning.md`](dirty_data_conditioning.md))
carries the recovered forces and the stress invariants, keeps the 1 eV/Å cut, and drops the
356 rows that only survived it through a fabricated `max_force`. The recommendation for the
next conditioned run is **two channels, `energy_above_hull` and `delta_e_polymorph`, and no
convergence channel at all** — which also removes the original defect: asking for
`max_force = 0` at generation time selected against Materials Project and towards genes
with no free positional degrees of freedom.

## Reproducing

```bash
python scripts/recover_mp_forces.py all --run full
python scripts/build_lemat_bulk_fmax.py --name lemat_bulk_fmax1_stress

CUDA_VISIBLE_DEVICES="" python scripts/gradient_matched_pilot.py \
    --device cpu --workers 24 --max-atoms-per-batch 240

python scripts/extract_mp_trajectories.py all
python scripts/extract_alexandria_trajectories.py all
CUDA_VISIBLE_DEVICES="" python scripts/validate_curvature_on_trajectories.py \
    --trajectories cache/mp_trajectories --workers 12
```

ORB finite-difference Hessians in float64 cost ~26 MiB per atom and will exhaust a shared
GPU; run them on CPU. Results land under `cache/gradient_matched/`, `cache/mp_forces_recovery/`,
`cache/mp_trajectories/` and `cache/alexandria_trajectories/`, none of it in git.

Trajectory sources, for future work: MP's S3 task documents carry every ionic step;
Alexandria publishes `geo_opt_paths` with per-run cutoffs (~105 GB, shards in no id order);
HuggingFace `LeMaterial/LeMat-Traj` covers all three databases with LeMat-Bulk ids but is
thinned, carries no INCAR, and holds ~2 frames per OQMD trajectory. OQMD's own per-step
forces were not located — its `sigma_P` is the one number here still borrowed from MP.
