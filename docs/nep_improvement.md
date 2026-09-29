# Options for faster NEP89 prescreening

The agreed target is **NEP89 prescreening no slower than generation of the same
PyXtal starts**, while preserving downstream screening quality. The target
workload is general prescreening across supported chemistries; Cu–Ge–Te is a
stress case for large cells. Compare CPU and GPU solutions with their resource
costs reported separately.

This proposal follows the **2026-09-27** investigation at commit
**`b8191f769f7e1528ee934f7fcf7eef62c2411770`**. The
[overhead report](nep_overhead.md) contains measurements, input provenance and
limitations. No improvement arm below has a completed general quality study or
an associated W&B experiment from this investigation. External capabilities
were checked on the investigation date and need version pins before use.

## Recommended order

| Priority | Approach | Evidence and remaining question |
|---|---|---|
| 1 | Remove constraint copying inside calculator evaluation | A geometry-only input cut one 300-atom evaluation probe from 90 to 19 ms; validate complete constrained relaxations. |
| 1 | Incorporate the other agent's Cholesky BFGS | Preliminary full-relaxation speedups were 1.46–1.58× on two large cells; benchmark the completed patch more broadly. |
| 2 | Reduce optimizer work and force-call count | Test LBFGS/FIRE, warm-up budgets, staged tolerances and independent symmetry coordinates. |
| 3 | Batch NEP89 evaluation and relaxation on a GPU | Prototype ALCHEMI with TorchNEP, including explicit crystal symmetry support. |
| 4 | Move remaining expensive operations into native code | Choose C++ kernels or a native loop only after profiling the improved CPU path. |
| Experimental | Allocate relaxation effort progressively | Early ranking may discard good basins; require downstream quality evidence. |

These improvements interact. Do not multiply isolated speedups to predict a
combined result. Even with negligible Python overhead, hundreds of genuine
force calls can exceed the time for a fast PyXtal draw. Reaching the target may
require fewer evaluations, batching, or both.

## CPU path

### Calculator snapshots, reuse and scheduling

Keep constraints on the optimizer's atoms. Give the evaluator geometry snapshots
without copying `FixSymmetry` state. Preserve all calculator-relevant arrays
and invalidate cached results correctly for positions, cell, species and atom
count. Test constraint preservation on the caller and stale-state behavior
across consecutive structures. Do not remove the optimizer's symmetry handling.

Inspect repeated constraint construction between stages: `FixSymmetry` itself
stores an atoms snapshot, so constructing it from atoms that already carry
constraints can retain previous constraint state. Measure and avoid unnecessary
nesting while preserving the intended symmetry at each stage.

Group compatible trials to reuse loaded models and native buffers, while
retaining load balancing for large or slow cells. The wrapper currently rebuilds
on symbol-sequence changes; any relaxation of that rule must be justified by the
pinned backend's resizing and species-update behavior. Check actual worker BLAS
thread counts rather than relying solely on environment settings.

Test species-pruned models for recurring element sets. Calorine's
[model modification API](https://calorine.materialsmodeling.org/dev/get_started/nep_model_modification.html)
can retain selected species and remove unused parameters. This may reduce model
loading and storage; it does not imply that per-atom evaluation becomes 89/3
times faster for a ternary system. Verify energies, forces, stress and ZBL
behavior, and record both the parent checkpoint and derived model hashes.

After these changes, measure neighbor-list rebuilding, matrix logarithms,
Fréchet cell-force transformations, symmetry detection and file output.
Optimize their measured contribution; cached neighbor lists must remain valid
under both atomic motion and cell deformation.

### Optimizers and independent symmetry coordinates

Use the completed `/tmp/wyformer-nep89-speed` work as the Cholesky comparison
arm, recording its exact patch or commit. Retain ASE's eigensolver fallback for
indefinite Hessians and test non-finite steps and step limiting. Cholesky reduces
the constant factor but still uses dense storage and cubic factorization work.

Compare LBFGS and FIRE with this improved baseline. Their cheaper steps can be
offset by more force calls, so measure total relaxation cost and convergence.
Keep the existing force threshold and stage schedule for the first optimizer
comparison, then vary them separately.

Prototype optimization in the independent Wyckoff positional coordinates and
symmetry-allowed cell variables. Generate the full periodic structure for NEP
evaluation, and pull forces/stress back through the coordinate transformation.
Handle orbit multiplicities, coordinate scaling, periodic wrapping and cell
derivatives explicitly. Validate gradients with finite differences and check
that the requested Wyckoff assignments and symmetry survive every step. This
reduces optimization dimension; it does not automatically reduce the number of
atoms evaluated by NEP.

Test zero or 20-step fixed-cell warm-up against the current converged warm-up,
and preliminary force thresholds of 0.2 and 0.3 eV/Å against 0.1. These are
experimental arms, not proposed default values: overlapped starts and coupled
cell motion can change which basin is reached.

### What a C++ or Rust rewrite would mean

The evaluator is already [C++ NEP_CPU](https://github.com/brucefan1983/NEP_CPU).
Rewriting that same potential in another language has no demonstrated speed
benefit. The useful native targets are the remaining symmetry projections,
geometry transformations, buffer management and optimization loop.

Prefer a small C++ extension reusing the existing evaluator for the first native
prototype. Keep model parameters loaded, reuse work arrays, and cross the
Python boundary at trial or batch boundaries when that is measurably useful.
Rust remains possible through a C++ interface or a new evaluator, but adds
binding or parity work without an established performance advantage. Compare
both choices by the operations eliminated and kernel performance, not language
labels. No full evaluator rewrite is recommended before profiling the improved
pipeline.

## GPU paths

### ALCHEMI with a NEP89 evaluator

Use the open-source ALCHEMI Toolkit and Toolkit-Ops. The checked
[supported-model list](https://nvidia.github.io/nvalchemi-toolkit/models/index.html)
does not include NEP. Its [custom model interface](https://nvidia.github.io/nvalchemi-toolkit/userguide/models.html)
supports direct energy/force/stress outputs, and
[Toolkit-Ops](https://github.com/NVIDIA/nvalchemi-toolkit-ops) provides batched
neighbor lists, FIRE/FIRE2/LBFGS and coordinate/lattice relaxation. ALCHEMI is a
framework for this integration, not an existing NEP89 acceleration switch.

1. **Establish evaluator parity.** Start with TorchNEP and the pinned NEP89
   checkpoint, without retraining. Its [prediction interface](https://mushroomfire.github.io/torchnep/guide/prediction/)
   documents loading `nep.txt`, CUDA evaluation, per-atom virials, ZBL components
   and batched prediction. Verify the actual checkpoint and all descriptor/ZBL
   options; advertised NEP support alone is insufficient.
2. **Adapt tensor inputs and outputs.** Supply independent periodic cells and
   batched species/positions. Return total energies `[B, 1]`, forces `[N, 3]`
   and stresses `[B, 3, 3]`. Check virial ordering, stress sign and volume
   normalization against Calorine and strain finite differences.
3. **Keep the iterative path on the GPU.** Connect the tensor evaluator to
   GPU neighbor construction and optimizer state. An ASE calculator loop or
   CPU neighbor builder inside each batch step could erase the benefit.
4. **Preserve crystal symmetry explicitly.** Implement projection of forces,
   displacements and cell updates, including optimizer momentum/history where
   needed. Validate against ASE `FixSymmetry`. Built-in frozen-atom hooks and
   bond constraints do not establish Wyckoff-orbit preservation.
5. **Measure real batching.** Try batch sizes 1, 8, 32 and 128 where memory
   permits, grouped by structure size. Retire converged or failed structures
   individually and refill batches; retain stable trial IDs, per-system
   convergence and independent cells. Include transfers, compilation and
   model initialization in cold and amortized throughput reports.

Use pinned optional dependencies and the machine's documented environment
procedure. If TorchNEP's evaluation remains expensive, compare a persistent
CUDA evaluator reusing GPUMD kernels before writing NEP kernels from scratch.
Keep unsupported chemistry on the existing explicit fallback path.

### GPUMD and another batching framework

GPUMD already supports [internal FIRE minimization with variable cell](https://gpumd.org/gpumd/input_parameters/minimize.html).
Running a complete minimization inside it amortizes process and model setup over
many steps. Calorine's [GPUNEP calculator](https://calorine.materialsmodeling.org/get_started/ase_calculators.html)
invokes the executable; using that route for every individual force call should
not be assumed fast.

The documented GPUMD minimizer does not establish the crystallographic
constraints needed by the prescreen. Confirm or implement those before using it
as a replacement. Also reconcile stopping criteria: its documented maximum
Cartesian force component is not the same as ASE's filtered atomic/cell
convergence criterion. Independent periodic crystals cannot simply be combined
into one box without preserving their separate boundary conditions.

[TorchSim](https://github.com/TorchSim/torch-sim) is an alternative batching
framework with automatic batching and cell relaxation. It is a useful
comparison if ALCHEMI integration proves awkward; its published speedups on
other models do not predict NEP89 performance. It would also require a verified
NEP adapter and the appropriate symmetry constraints.

## Progressive screening and quality validation

As a separate algorithm experiment, briefly relax every draw, then allocate
more steps to candidates with promising energies and structural diversity.
For an initial pilot, use a 20-step preliminary budget, retain the best half
within each gene plus a seeded random 10% of the remainder, and fully relax
those survivors before the usual selection. Compare against full relaxation of
every draw and record the discarded candidates' eventual quality on the pilot.
Do not promote this policy using early NEP energy alone.

Cheap surrogate warm-ups, including the existing `ScreenedMorse`, are another
experimental arm. They change the trajectory and need the same quality checks.
Likewise, changing PyXtal tolerance alters both generation cost and starting
geometry; keep it fixed in the primary speed comparison. See the existing
[tolerance sweep](pyxtal_tolerance_sweep.md) before interpreting such an arm.

### Benchmark and acceptance proposal

- Freeze identical input starts and trial IDs. Use the existing 750-gene oracle
  reconstruction cohort described in [the variants study](de_novo_ranking_protocol_nep89_variants.md#evaluating-the-variants),
  plus 100 Cu–Ge–Te draws stratified by size and symmetry. Separate tuning and
  held-out genes. Fetch durable inputs where available; missing local files
  must not silently cause a different cohort to be used.
- Compare original ASE, the completed Cholesky patch, cumulative CPU overhead
  fixes, optimizer/schedule alternatives, progressive screening and the GPU
  prototype. Attribute gains with incremental comparisons.
- Record model setup, native evaluation, copies, symmetry, optimizer updates,
  cell filtering, output, force calls and stage convergence. Reaching a step
  cap must be distinguishable from convergence. Fix the preliminary benchmark's
  log-parser issue by reading counts directly from optimizers.
- Test numerical parity and finite-difference derivatives, high-symmetry and
  skewed cells, small cells requiring multiple periodic images, ZBL contacts,
  species/atom-count changes, mixed batches and unsupported-element fallback.
  Verify energy agreement per atom and force/stress agreement with explicit
  absolute and relative tolerances chosen for the evaluated precision.
- Evaluate final ORB reconstruction, selected-basin diversity, failures and ORB
  work at the same final selection budget. A suggested promotion gate is a
  paired 95% confidence interval excluding a reconstruction loss greater than
  **one percentage point**. This margin is a proposal, not an established
  project policy; an inconclusive result leaves an arm experimental.
- Report matched sums of trial elapsed times, median/p90/p99 latency, total
  stage wall time, CPU core-seconds and GPU-seconds. Compare a GPU against the
  complete CPU pool, account for resources it takes from ORB, and report
  success/failure denominators. Test the target ratio on held-out inputs;
  expensive PyXtal timeout tails must not conceal poor typical latency.
- Log inputs, derived models, patches, configurations, profiles and raw results
  as W&B artifacts, with date, commits, dependency versions and hardware.
  Keep execution/setup details in platform documentation.

Benchmark configurations should distinguish potential identity, evaluator
backend, optimizer and stopping schedule. Add production backend/optimizer
options only after validation, with explicit provenance and existing fallback
behavior preserved. NEP energies remain within-gene screening signals; they
must not be substituted for a compatible hull-scoring energy. Preserve the
project's energy-field definitions and compatibility checks.

The first implementation should combine calculator-copy removal with the
completed Cholesky patch. Use its profile to decide how much further CPU work
can buy; evaluate reduced-coordinate/fewer-step methods and a batched ALCHEMI
prototype against that improved baseline.
