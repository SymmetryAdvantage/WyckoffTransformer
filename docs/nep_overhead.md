# NEP89 relaxation overheads

NEP89's inexpensive force evaluation does not make a complete relaxation
inexpensive. The present pipeline also pays for optimizer linear algebra,
symmetry constraints, geometry copies, cell transformations, initialization and
output. On the sampled Cu–Ge–Te search, prescreening cost substantially more
than generating the same starts with PyXtal.

This document records observations. Proposed changes and their validation are in
[NEP89 improvement options](nep_improvement.md). The scientific purpose of the
prescreen is described in [the NEP89 protocol variants](de_novo_ranking_protocol_nep89_variants.md).

## Provenance and limits

- Investigation date: **2026-09-27**.
- Main checkout commit: **`b8191f769f7e1528ee934f7fcf7eef62c2411770`**.
- Host: **zeus**, shared with other work. Environment and execution instructions
  belong in [the zeus platform documentation](platforms/zeus/agent_brief.md).
- Installed versions: ASE **3.29.0**, Calorine **4.0**, NumPy **2.5.3**, SciPy
  **1.18.1**, PyTorch **2.14.0+cu133**.
- Model: the repository's pinned `nep89_20250409`, SHA-256
  `75168ece02e840e4a32644f982b78d43cba697f5b64b4c8134ab66c7a8c28be1`.
- Applicable W&B runs: **none identified for the timing inputs; these local
  diagnostics were not logged as a W&B run**. There is consequently no durable
  benchmark artifact to fetch. The tables below preserve the observed summary,
  not a claim that all raw inputs are available on another machine.
- The Cholesky results came from another agent's uncommitted work in
  `/tmp/wyformer-nep89-speed`, based on the same commit. Its implementation was
  still evolving; the base commit alone does not identify that experiment.

These are diagnostic observations, not a controlled general performance study.
GPU performance was not measured. No downstream quality comparison was performed
for the copying experiment or the preliminary optimizer benchmark.

## What the current path does

The default NEP89 prescreen calls `prerelax`, which runs a symmetry-constrained
fixed-cell warm-up followed by symmetry-constrained cell-and-position relaxation.
It uses ASE BFGS, `FixSymmetry`, and `FrechetCellFilter` for the second stage.
The prescreen force threshold is 0.1 eV/Å; the relaxation helper permits up to
500 optimizer steps **per stage**. Symmetry release and rattling are off for
this prescreen variant.

The calculator path is:

```text
ASE optimizer / cell filter / symmetry constraints
  -> Nep89WithFallback
    -> Calorine CPUNEP
      -> _nepy / native C++ NEP evaluation
```

The potential evaluator is already C++: upstream
[NEP_CPU](https://github.com/brucefan1983/NEP_CPU) supplies the engine used by
Calorine. The Python wrapper is not evaluating the neural network in Python.
See the repository's [calculator](../src/wyckoff_transformer/cryspr/nep89.py),
[pre-relaxation entry point](../src/wyckoff_transformer/cryspr/generator.py), and
[relaxation stages](../src/wyckoff_transformer/cryspr/relaxer.py).

At the inspected commit, a changed symbol sequence causes the wrapper to reset
the inner calculator, rebuilding its native object on the next evaluation.
Repeated draws with the same sequence can reuse it. The module's historical
estimate of a roughly 0.24 s model reparse is not a fresh measurement from this
investigation and is not a cost incurred on every optimizer step.

`run_ase_relaxer` currently returns the structure after `opt.run` without
recording its convergence result. A successful prescreen row therefore does
not by itself establish convergence rather than exhaustion of the step budget.

## End-to-end timing evidence

The Cu–Ge–Te hull search had 500,000 recorded PyXtal attempts. At the instant
sampled, `prescreen.csv` contained **13,167** attempts. Matching those rows to
`pyxtal.csv` by `(index, trial)` gave:

```text
sum(prescreen seconds) / sum(matching PyXtal seconds) = 12.7362
sum(prescreen seconds) = 199,265.75
```

The separate pilot had 250 matching attempts: 1,515.63 s of prescreening against
50.29 s of generation, a **30.14×** ratio.

These are sums of **per-trial elapsed seconds**, including recorded failures and
timeouts. They are neither process CPU time nor total stage wall time under a
worker pool. The active hull search was an incomplete cohort, ordered by the
work processed so far; it is not a random sample of all 500,000 draws.

### Size distribution of the sampled prescreen attempts

| Atoms per cell | Attempts | Median elapsed seconds | Summed elapsed seconds |
|---|---:|---:|---:|
| 1–40 | 2,671 | 0.79 | 2,813.8 |
| 41–100 | 4,000 | 2.56 | 12,343.1 |
| 101–200 | 4,008 | 7.11 | 33,119.3 |
| 201–400 | 2,062 | 22.35 | 67,659.1 |
| Above 400 | 426 | 197.61 | 83,330.4 |

Cells above 200 atoms were **18.9% of attempts but 75.8% of elapsed time**.
This motivates including large cells in a benchmark; it does not prove atom
count alone causes the cost. Symmetry, density, optimizer steps and timeouts
also vary. Across all 13,167 prescreen attempts, median/p90/p99 elapsed times
were 4.34/26.79/300.42 s; the upper tail includes timeouts.

## Symmetry copying at the calculator boundary

ASE `Atoms.copy()` deep-copies constraints. `FixSymmetry` carries rotations,
translations, atom mappings and an atoms snapshot. In the inspected versions,
geometry snapshots are copied in the outer calculator, the inner calculator,
and Calorine's native-state synchronization. Passing optimizer atoms into this
path also copies their constraint state.

A warm evaluation probe compared constrained atoms with fresh `Atoms` holding
only the same numbers, positions, cell and periodicity. It also timed the
native `get_potential_forces_and_virials()` call directly.

| Median of 10 calls | 172 atoms, 8 symmetry operations | 300 atoms, 192 symmetry operations |
|---|---:|---:|
| Wrapper, with `FixSymmetry` attached | 13.21 ms | 90.36 ms |
| Wrapper, geometry only | 10.33 ms | 18.57 ms |
| Native energy/forces/virials call | 9.65 ms | 19.02 ms |

For the 300-atom case, removing constraint state from the calculator input
reduced this evaluation probe by **4.86×**. That is not a full-relaxation
speedup. The optimizer still needs symmetry constraints, and the probe did not
implement or validate a new constrained relaxation path.

The direct native call repeated a fixed geometry; the wrapper calls perturbed
one coordinate to exercise synchronization. Small differences between the
geometry-only and native medians are timing variation, not evidence that the
wrapper accelerates the native kernel. The tiny successive perturbations also
mean this was not a numerical equivalence test on identical inputs.

## Optimizer and cell-filter costs

An eight-step BFGS profile on each initial structure, with variable cell and
symmetry enabled, found:

| Profile observation | 172 atoms | 300 atoms |
|---|---:|---:|
| Total profiled time, including constraint setup | 0.873 s | 11.544 s |
| `Atoms.copy` cumulative time | 0.152 s | 4.056 s |
| NumPy `eigh` cumulative time | 0.174 s | 1.018 s |

The profiles also exposed `FixSymmetry` setup, matrix logarithms and Fréchet
cell-force transformations. Cumulative timings overlap and must not be added.
Python profiling disproportionately slows copying and other Python-heavy work;
use the unprofiled evaluation probe above to estimate copying cost.

ASE BFGS diagonalizes a dense Hessian each step. For an atomic stage its
dimension is 3N, with nine additional cell coordinates under the filter.
Both dense storage and factorization remain costly even if only a few symmetry
degrees of freedom can actually move.

### Preliminary Cholesky BFGS experiment

The other agent replaced the eigensolve with a Cholesky solve for positive
definite Hessians, retaining ASE's eigenvalue treatment when factorization
fails. This preserves the BFGS step in exact arithmetic when the Hessian is
positive definite; floating-point trajectories still require comparison.

The saved benchmark timed both prescreen stages, including their normal output,
after warming the model. It ran ASE first and the modified optimizer second,
once per structure, using `fmax=0.1`, symmetry enabled, no release and no rattle.

| Input | ASE elapsed time | Cholesky elapsed time | Speedup | Max absolute Cartesian coordinate difference |
|---|---:|---:|---:|---:|
| Si16 | 0.304 s | 0.295 s | 1.03× | 3.02e-13 Å |
| CuGeTe172 | 10.319 s | 6.524 s | 1.58× | 2.89e-11 Å |
| CuGeTe300 | 192.046 s | 131.760 s | 1.46× | 3.72e-9 Å |

Both methods returned space groups 227, 15 and 225, respectively. Relative
volume differences were below 4.6e-10 and energy differences below 2.1e-11
eV/atom on these three inputs. This is encouraging numerical agreement on a
very small sample, not evidence of general downstream quality preservation.

**The modified optimizer's recorded step counts are invalid.** The benchmark
parser recognized only `BFGS:` log lines, whereas the initial subclass emitted
another class name, producing `steps=-2`. Do not infer iteration reductions
from that file. The baseline's 533 steps for the 300-atom input span two stages;
the saved summary does not establish that both stages converged.

Original local files were `/tmp/benchmark_nep89.py` and
`/tmp/nep89-benchmark.json`. These temporary files are evidence locations for
this session, not durable dependencies or guaranteed future inputs.

## Reproducing and extending the measurements

Use the current machine's documented environment from
[the platform index](platforms/README.md). Keep launch and installation details
in that platform's documentation. For the evaluation probe and eight-step
profiles, OpenBLAS, MKL and OpenMP were limited to one thread before imports;
`threadpoolctl` confirmed one thread in the loaded BLAS/OpenMP runtimes.

The original untracked input identifiers, relative to the checkout used for
the investigation, were:

| Input | Historical location |
|---|---|
| Hull timing tables | `artifacts/cu_ge_te_hull/reconstruction/{pyxtal,prescreen}.csv` |
| Pilot timing tables | `artifacts/cu_ge_te_pilot/reconstruction/{pyxtal,prescreen}.csv` |
| CuGeTe172 | `artifacts/cu_ge_te_pilot/reconstruction/cryspr/0/trial-1/prescreen/Cu9Ge10Te24_Cu36Ge40Te96_0_initial.cif` |
| CuGeTe300 | `artifacts/cu_ge_te_hull/reconstruction/cryspr/225/trial-44/prescreen/Cu30GeTe44_Cu120Ge4Te176_0_initial.cif` |

These locations describe the old inputs; new tooling should accept explicit
input paths and obtain shared data through the project's store helpers. Do not
assume these untracked directories exist on another host.

1. Freeze copies of the input tables and structures, recording hashes and row
   counts. Match trials by `(index, trial)` and report statuses separately.
   For the historical table, quantiles used sorted values at indices
   `int(q * n)` and summed times included all recorded statuses. A later
   snapshot of the active run will not reproduce these exact totals.
2. For each CIF, read it with ASE and attach `FixSymmetry(symprec=1e-3)`.
   Build and warm `build_nep89_calculator()`. Construct a separate `Atoms`
   from its numbers, positions, cell and PBC, without constraints.
3. For each wrapper input, time ten explicit `calculator.calculate(atoms)`
   calls with `perf_counter`, adding 1e-7 Å to the first atom's x coordinate
   before each call. Report the median. Then time ten direct native
   energy/forces/virials calls. Exclude imports, model loading and symmetry
   setup from these warm timings.
4. Separately profile constraint construction and eight BFGS steps on a fresh
   copy under `FrechetCellFilter`, with `logfile=None` and `fmax=0.1`. Do not
   compare profiled timings directly with unprofiled throughput.
5. Reproduce full relaxations using the frozen optimizer patch and the same
   initial inputs. Si16 was `bulk('Si', 'diamond', a=5.43, cubic=True)` repeated
   `(2, 1, 1)`. Record steps directly from optimizers, stage convergence, force
   calls, CPU time, elapsed time, final geometry and symmetry. Alternate method
   order and repeat rather than relying on the original single ordered pair.

Log future benchmark inputs, optimizer patches, raw measurements and profiles
as W&B artifacts with commit and dependency provenance. The next study should
measure the whole pool as well as individual workers, distinguish cold from
warm behavior, and quantify quality using the protocol's downstream scorer.
