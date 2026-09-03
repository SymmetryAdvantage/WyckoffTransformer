# Improving De Novo Quality in WyFormer

## Scope

This plan is only about raising stability, SUN, and MetaSUN for unconditional
de novo generation. It does not propose new user-facing conditioning modes or
replace the WyFormer-plus-CrySPR design.

## Established evidence

The `upi73i4k` CrySPR experiment is the current reference point. Its analysis
is in [CrySPR trial and stage spread](cryspr_trial_and_stage_spread.md).

- CrySPR selects a gene's trial by its **final relaxed MLIP energy**, not its
  initial PyXtal energy.
- Three PyXtal trials capture most of the useful multi-start gain. More trials
  have poor marginal energy return.
- The symmetry-constrained cell-and-position relaxation is important. Releasing
  symmetry in the final stage usually is not.
- Re-relaxing submitted structures raises MetaSUN by about one fifth, so CrySPR
  robustness matters, but it cannot account for the full stability gap.
- Current production models are already conditioned on `energy_above_hull` and
  trained on LeMat-Bulk. A separate stability critic or MP-20 pretraining is
  therefore not the immediate priority.

## Priority order

1. **Complete the optimizer and capacity investigation.** LeMat-Bulk has about
   four million examples while the current base architecture is small. Compare
   the AdamW/WSD checkpoint with controlled depth, width, embedding, and head
   scaling. Hold tokenisation, e-hull condition, generation temperature, and
   CrySPR settings fixed.

2. **Audit e-hull conditioning against realized stability.** A condition passed
   to AdaLN is not evidence that it controls the e-hull after PyXtal and MLIP
   relaxation. Generate at a fixed grid of target values, under fixed seeds and
   a fixed space-group-start tensor, then measure the relaxed e-hull
   distribution and stable/MetaSUN yield. The audit runner is
   `scripts/audit_ehull_conditioning.py`.

3. **Improve the existing condition path only if the audit is weak near zero.**
   Test condition-bin balancing with extra resolution over 0--0.1 eV/atom,
   alternative scalar normalisation, and a binned-plus-continuous condition
   embedding. The criterion is lower realized e-hull at target zero, not lower
   validation NLL alone.

4. **Select checkpoints by a fixed low-e-hull generation probe.** Continue to
   log validation NLL, but periodically generate a fixed validation probe at
   target 0 and relax it with the standard three-trial CrySPR budget. Use its
   relaxed stable yield, subject to uniqueness/novelty guards, to choose among
   NLL-nearby checkpoints.

5. **Scale deterministic crystallographic inputs.** Add multiplicity, orbit
   degrees of freedom, and orbit-type descriptors as inputs. They enrich the
   existing representation rather than changing what the model generates.

6. **Harden CrySPR without spending more random-search budget.** Keep three
   trials; retain the symmetry-constrained cell relaxation; make the final
   symmetry-free stage optional; and reject/retry non-finite or collapsed-cell
   results.

## Audit protocol

The conditioning audit uses the best checkpoint from `runs/upi73i4k`, fixed
space-group starts, and the target grid `0, 0.025, 0.05, 0.1, 0.2` eV/atom. For
each target, it records the generated Wyckoff genes and then applies the same
CrySPR protocol used in the reference run: MACE-MP-0a-small, three trials, and
`fmax=0.05`.

Report all of the following against the number of initially sampled genes:

```text
sampled gene -> formally valid gene -> CrySPR structure -> valid benchmark structure
             -> e_hull <= 0.1 -> e_hull <= 0
```

The decisive plot is target e-hull versus the distribution of *relaxed* e-hull,
with bootstrap intervals. A monotone shift toward lower realized e-hull at lower
targets validates the current conditioning mechanism. If that shift is absent
or saturates near zero, condition-density balancing is the next experiment.
