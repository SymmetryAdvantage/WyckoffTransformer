# `upi73i4k` e-hull-conditioning audit

Run date: 2026-09-03/04. Checkpoint: `runs/upi73i4k/best_model_params.pt`.
This is the LeMat-Bulk e-hull-conditioned model, evaluated as a controlled de
novo-generation probe rather than as a design task.

> **Training-status qualification.** This checkpoint was selected while
> `upi73i4k` was still training, during its high-learning-rate phase. It is not
> a converged final model, so these results measure conditioning behaviour at
> that training point rather than the final attainable quality of this setup.

## Protocol

- Targets: 0, 0.025, 0.05, 0.1, and 0.2 eV/atom.
- 250 decoding starts per target, sampled from the same saved start-token
  distribution. The decoding RNG was independently seeded per target, so this
  is a matched-start distributional comparison, not a paired-sample test.
- CrySPR: MACE-MP-0a-small on CPU, three PyXtal trials, `fmax=0.05`; choose the
  lowest final relaxed MLIP energy for each gene.
- Evaluation: valid structures were MACE-MP re-relaxed at `fmax=0.02` for up to
  50 steps, then scored against the current public `mace_mp` convex hull.
  The installed GenBench code used a retired hull-data path, so the audit
  runner bridges it to the current MACE reference parquet and sparse
  composition matrix. All outputs are local under
  `generated/upi73i4k/ehull_conditioning_audit/stability/`.

## Primary result: unrelaxed MACE e-hull

| Requested target | Valid / finite | Mean | Median | 95% CI for mean | <= 0 | <= 0.05 | <= 0.1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 220 / 220 | 0.254 | 0.168 | [0.215, 0.293] | 3.2% | 16.8% | 34.1% |
| 0.025 | 229 / 228 | 0.227 | 0.142 | [0.190, 0.264] | 3.5% | 21.8% | 36.7% |
| 0.050 | 218 / 218 | 0.228 | 0.166 | [0.193, 0.262] | 2.3% | 17.4% | 35.3% |
| 0.100 | 215 / 215 | 0.246 | 0.173 | [0.211, 0.281] | 2.8% | 10.7% | 28.4% |
| 0.200 | 216 / 216 | 0.307 | 0.238 | [0.272, 0.342] | 1.9% | 5.1% | 13.9% |

All values are eV/atom. One 0.025-target valid structure had no finite hull
value because its MACE stability processing timed out.

## Interpretation

- The high target responds in the expected direction: 0.2 shifts the mean
  MACE e-hull about 0.08 eV/atom above the 0.025/0.05 cohorts and sharply
  reduces the fraction under 0.1 eV/atom.
- The four lower targets are not cleanly ordered. Their mean confidence
  intervals overlap, including zero versus 0.025/0.05. Thus the current audit
  supports weak coarse control at the upper end, not calibrated fine-grained
  control near the hull.
- It is not evidence that a stable-data generator “should” produce only stable
  materials. De novo decoders have broad support, and the target is a desired
  conditional distribution rather than a hard constraint.
- The model's training label and the MACE-MP oracle need not share an absolute
  scale. Therefore the absolute gap between a requested target and the MACE
  mean must not be called a calibration error without a label-aligned oracle.
  The target-to-target ordering and separation above are the defensible result
  of this audit.

## Relaxed reference

MACE re-relaxation lowers the mean e-hull in every cohort, but preserves the
same broad pattern. The per-structure unrelaxed and relaxed values are in
`ehull_<target>_energies.csv`; SUN JSON is retained only as a secondary
outcome, because the purpose of this experiment is e-hull conditioning.
