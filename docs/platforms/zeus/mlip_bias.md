# EquFlashV2 and PET-OAM-XL on Zeus

On 2026-10-05, at WyFormer commit
`c3cb84e4dcfd6e08360e4ee7a2232786143c1a9a`, both published checkpoints passed
GPU 1 Si2 energy, forces and stress probes. EquFlashV2's first 24-atom raw
draw completed the full relaxation protocol in 41.197 seconds. Its ledger,
free and fixed-symmetry CIFs were uploaded to
[run 9b5e07e1](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/9b5e07e1).
An initial launcher count bug skipped EquFlashV2 and started PET. PET was
stopped after its seventh saved trial, and all seven successful rows/CIFs
were uploaded to [run a4bbbd2f](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/a4bbbd2f).
The corrected launcher validates counts and finishes EquFlashV2 before
resuming PET from those saved rows. Shell checks passed for sequential order,
completed-arm skipping, and the stop marker.
These are historical setup and initial-trial results. Both full relaxation
ledgers and recovery have since completed; current GenBench scoring and
provenance are documented below.

The user requested sequential execution: finish all 2,698 EquFlashV2 trials,
then run all 2,698 PET-OAM-XL trials. The detached worker uses GPU 1, four
EquFlashV2 calculator processes on physical CPU cores 16–23, then two PET
processes on cores 20–23, two BLAS/OpenMP threads per process, a 1,200-second per-trial timeout,
and the common raw `pyxtal.extxyz` from the handoff artifact. It processes
25-trial batches for resumable stopping, finishing every EquFlashV2 batch
before starting any PET relaxation. Snapshots are uploaded every 25 new
trials and at batch exit. Failed trials remain in the first-pass ledger.
One parent process writes the ledger, selects structures, and uploads W&B
snapshots. Spawned child processes each hold a calculator and relax uniquely
named trials; a per-trial timeout in one child leaves the others running.
`execution_settings.json` records worker/affinity changes without changing
the scientific manifest or W&B run ID, and is included in snapshots.

On 2026-10-05, four simultaneous EquFlashV2 validation trials all succeeded.
Their free/fixed-symmetry energies matched the serial results within
`6.36e-6 eV/atom`. GPU 1 was observed at 84% utilization and 6,048 MiB used
during the validation. The serial worker's six completed trials were uploaded
before resuming with four workers. PET retains its seven saved trials and
will resume after EquFlashV2 finishes. This validates sample correctness;
larger generated cells may require more memory. `EQUFLASH_WORKERS` and
`PET_WORKERS` override the launcher defaults.

```bash
scripts/platforms/zeus/run_mlip_bias_gpu1.sh
```

The active window is `0:4`, named `mlip-gpu1`; the original selected window
is unchanged. Output is appended to
`generated/mlip_bias_study/zeus_gpu1_20261005.log`. A lock rejects another
copy of this launcher. Before any direct runner invocation, inspect tmux
and processes to avoid a second writer for an arm.

```bash
tmux select-window -t 0:mlip-gpu1
tail -f generated/mlip_bias_study/zeus_gpu1_20261005.log
```

To stop after the active batch and its W&B upload, create
`generated/mlip_bias_study/zeus_gpu1.stop`. Move that marker aside before
resuming. Preserve arm working directories; if absent, the runner restores
their latest W&B snapshot. Outputs are in `generated/mlip_bias_study/equflashv2`
and `generated/mlip_bias_study/pet_oam_xl`.

## Environment

The base venv has local Torch `2.14.1+cu134`, Triton `3.8.0`, local
`torch-scatter` 2.1.2, `torch-sparse` 0.6.18, `metatomic-torch` 0.1.19 and
Warp 1.18.0. Install these explicit wheels from `/mnt/hdd/torch_wheels`
with `uv pip install --python .venv/bin/python --no-deps --no-index`.
The local lock now resolves Torch 2.14.1. Torch's NVSHMEM runtime fix is
documented in [environment.md](environment.md).

cuEquivariance's four packages were installed with dependency resolution
disabled:

```bash
uv pip install --python .venv/bin/python --no-deps \
    cuequivariance==0.12.0 cuequivariance-torch==0.12.0 \
    cuequivariance-ops-cu13==0.12.0 cuequivariance-ops-torch-cu13==0.12.0 \
    nvidia-ml-py==13.615.71
```

`cuequivariance-ops-cu13` declares `nvidia-cublas>=12.5.0`. The host already
provides cuBLAS 13 through system CUDA 13.4. Do not let a resolver install
that runtime package or replace the local Torch. `nvidia-ml-py` supplies the
Python NVML interface and does not bundle CUDA libraries. No NVIDIA CUDA
runtime distribution was installed. The EquFlashV2 probe and full first
relaxation passed with these actual compiled cuEquivariance ops; the old
`SegmentedPolynomialNaive.buffer_num_segments` failure did not recur.

Model-specific dependencies stay in separate ignored overlays:

| Arm | `PYTHONPATH` overlay | Key versions |
| --- | --- | --- |
| EquFlashV2 | `deps/GGNN:deps/fairchem` | GGNN commit `a25dfa4735582ade72b3e3a2dd33074c43d9fe73`; FairChem 1.10.0; NumPy 1.26.4; SciPy 1.15.3; Numba 0.61.2; llvmlite 0.44.0; e3nn 0.5.6; PyG 2.6.1; nvalchemi-toolkit-ops 0.3.1 |
| PET-OAM-XL | `deps/pet` | UPET 0.3.1; metatrain 2026.4.1; metatomic-ase 0.1.4; metatensor-torch 0.10.6; metatensor-core 0.2.3; metatensor-learn/operations 0.5.0; vesin 0.6.1; nvalchemi-toolkit-ops 0.4.0; python-hostlist 2.3.0 |

All overlay paths are below `generated/mlip_bias_study`. Create them with
`uv pip install --target <overlay> --no-deps` and the explicit package versions
in the environment record. Clone the official
[GGNN source](https://github.com/SamsungDS/GGNN), then check out the exact
commit above. Do not install its pinned requirements wholesale: they request
stock CUDA 12 Torch/extensions. FairChem's Torch pin is also intentionally
bypassed. PET's original metatensor-torch 0.10.0 lacked Torch 2.14 binaries;
0.10.6 imports successfully with the locally built metatomic library.
The local metatomic wheel has a stale build-time file URL for metatomic-ase,
so install its explicit dependencies rather than resolving that URL.
Both local-wheel packaging problems were reported to the build repository:
[Torch's NVSHMEM RUNPATH](https://github.com/kazeevn/pytorch-zeus/issues/1)
and [metatomic's file URL dependency](https://github.com/kazeevn/pytorch-zeus/issues/2).

Avoid syncing the base venv during a running job. A later bare `uv sync`
can remove the study's additional packages; `--inexact` preserves them.
The separate overlays are working copies, reproducible from the recorded
versions/source commit, not the relaxation result artifacts.

The dated setup artifact `mlip-bias-zeus-setup-20261005` in
[compatibility run mlipbias1](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/mlipbias1)
contains both successful probes, the earlier PET ABI failure, probe logs,
before/after package records, source/launcher snapshots, and SHA-256 hashes
of the local wheels and both published checkpoints. The nine existing
runner recovery/timeout tests passed during setup. The parallel extension
adds a real spawned-process timeout/resume check; all ten tests pass.
`scripts/__init__.py` now prevents MACE's top-level `scripts` package from
masking the checkout during normal test collection. The validation data,
GPU samples, parallel runner and test/launcher snapshots are in compatibility
artifact `mlip-bias-zeus-parallel-20261005` under the same run.

## Zeus timeout audit — 2026-10-05 22:21 Asia/Singapore

At HEAD `3f9ec13525b217dc5997401ef0c592ea5af0f735`, EquFlashV2 had
2,656 recorded trials (2,647 successful, two timeouts, four allocation
failures and three neighbor overflows). PET had 1,857 recorded trials
(1,851 successful and six timeouts). These are progress counts, not final
coverage metrics, in the same runs linked above.

All eight timeout rows have 304–464 atoms. The 600-second limit covers the
whole fixed-cell, symmetry-constrained-cell, released-cell and rattled-cell
protocol, rather than each stage. The optimizer logs show ongoing progress:
EquFlashV2 draw `(463, 2)`, with 464 atoms, spent about 501 seconds and 444
BFGS steps in the symmetry-constrained stage before timing out in the final
rattle stage. PET draw `(434, 1)` reached the 500-step symmetry-stage cap,
then continued for 352 released-cell steps before timing out. Other PET
failures timed out in symmetry, released-cell or rattle stages. Parallel
workers share GPU throughput and can increase individual wall times;
these cases have not been rerun serially, so the effect of contention is
not measured. The scientific timeout was still 600 seconds at this audit; the subsequent
user-requested increase is recorded below.

The launcher was counting physical CSV lines, so multiline exception
messages falsely marked EquFlashV2 complete with 42 draws still missing.
Its count now uses `csv.DictReader`; a mocked launcher regression check
verified multiline counting, completing EquFlashV2 before PET, and skipping
complete arms. PET was stopped after its active 25-trial batch and W&B
upload, at 1,857 rows. The detached launcher resumed EquFlashV2 at 22:21
with four workers and will resume PET only after all 2,698 EquFlashV2
records exist. This repair preserves saved rows and the existing run IDs.
Audit evidence is published as compatibility artifact
`mlip-bias-zeus-timeout-audit-20261005` in run `mlipbias1`.

## Timeout increased — 2026-10-05

At WyFormer HEAD `3f9ec13525b217dc5997401ef0c592ea5af0f735`, the user
raised Zeus's whole-trial timeout from 600 to **1,200 seconds** for both
EquFlashV2 and PET-OAM-XL. The launcher passes the new limit to each arm,
retains four EquFlashV2 workers and two PET workers, and completes the
EquFlashV2 ledger and its timeout retries before continuing PET. The active
600-second batch is saved and uploaded before restarting the launcher.

`--retry-timeouts` selectively requeues timeout failures, preserving
successful rows and unrelated allocation/neighbor failures. Each arm's
initial resumed batch uses that option if its ledger contains failures
from a shorter budget. Subsequent batches retain any new 1,200-second
failures, so the launcher cannot retry them indefinitely. The saved runtime
settings record the 600-to-1,200 change and trial boundary; scientific
manifests, source artifact and W&B run IDs stay the same. Earlier snapshots
preserve the original failed rows and optimizer logs.

All ten runner tests pass, including a real spawned-process test verifying
selective timeout retry, preserved successful/OOM rows, and recorded timeout
history. A mocked launcher test verifies the new timeout arguments, quoted
multiline CSV records, sequential execution, completed-arm skipping and no
repeated retries at the new budget. The evidence and updated source are
published in compatibility artifact `mlip-bias-zeus-timeout-1200-20261005`
under run `mlipbias1`. Both relaxation runs remain
[9b5e07e1](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/9b5e07e1)
and [a4bbbd2f](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/a4bbbd2f).

## Queued serial recovery — 2026-10-05

At WyFormer HEAD `3f9ec13525b217dc5997401ef0c592ea5af0f735`, the user
requested a recovery pass **after both normal arms conclude**: GPU 1,
**one calculator worker**, **3,600 seconds per whole trial**, retrying only
OOM/allocation and timeout failures. EquFlashV2's normal ledger had all
2,698 records; PET was still processing its 1,200-second normal pass.

`scripts/platforms/zeus/run_mlip_bias_gpu1_recovery.sh` is queued in detached
tmux window `0:5`, named `mlip-gpu1-recovery`. It waits on the same GPU 1
lock held by the normal launcher in `0:4`, then checks that both normal
ledgers contain 2,698 unique records. It saves the final eligible failed
rows to `generated/mlip_bias_study/zeus_gpu1_recovery_plan.json` and uploads
that plan to compatibility artifact `mlip-bias-zeus-recovery-plan-20261005`
before requeueing any trials. Recovery runs EquFlashV2 first, then PET,
without a batch/trial limit, and uploads each new trial and at arm exit.
Successful normal trials and neighbor-overflow/other failures are retained.
The GPU selection, model overlays, seeds, fmax and four-stage schedule are
unchanged. Two CPU threads per worker remain in effect.

The runner's `--retry-oom-timeouts` recognises Torch/TorchScript OOMs,
cuEquivariance `cudaErrorMemoryAllocation`, Warp `Failed to allocate`, and
`TimeoutError`. `recovery_settings.json` records original candidates,
budget, worker count and finished attempts and is included in the existing
arm W&B artifacts. Resuming the same recovery budget processes missing
attempts without repeating completed recovery failures. The serial pass
updates each arm's saved runtime timeout and worker history at its start.
Once an arm finishes, the outer plan records completion and is uploaded
again. Results remain under the existing runs `9b5e07e1` and `a4bbbd2f`.

To inspect the queued launcher:

```bash
tmux select-window -t 0:mlip-gpu1-recovery
tail -f generated/mlip_bias_study/zeus_gpu1_recovery_20261005.log
```

The shared `zeus_gpu1.stop` marker prevents recovery from starting and is
checked between recovery arms; during an arm the runner uploads every trial
and the launcher waits for that arm to finish. The complete candidate plan
and per-trial recovery state make a later invocation resumable. Do not
restart the normal 1,200-second launcher after recovery has begun: it would
lower the saved recovery budget. Resume with the recovery launcher instead.

All 18 runner tests passed, including real single-process OOM/timeout
recovery, preservation of unrelated failures, crash-boundary recovery state
and avoiding repeated failed retries. The mocked queue check verified lock
waiting, refusal of incomplete normal ledgers, model order, 1 worker/3,600
seconds, per-trial uploads and skipping finished arms. Queue/source/test
provenance is in compatibility artifact
`mlip-bias-zeus-recovery-queued-20261005` under run `mlipbias1`.

## Completed relaxations and scoring, 2026-10-06

WyFormer HEAD: `3f9ec13525b217dc5997401ef0c592ea5af0f735`. Both 2,698-row
normal ledgers and the one-worker, 3,600-second recovery pass completed.
The fresh-parent EquFlashV2 `(991, 2)` retry succeeded in 157.8 seconds,
with a transient recoverable OOM and 45,260 MiB peak usage. EquFlashV2 now
has 2,694 successful trials and PET has 2,696. Each selects 991 genes in
both tracks. Fresh diagnostics are in compatibility artifact
`mlip-bias-zeus-fresh-991-2-20261006` under run `mlipbias1`.

The scoring environment is an isolated `deps/genbench` overlay, installed
with `uv pip install --python .venv/bin/python --target ... --no-deps`.
It leaves custom Torch 2.14.1/CUDA 13.4 and its binary extensions in the
base venv. Important pins are fairchem-core 2.3.0, orb-models 0.5.4, e3nn
0.5.0, numpy 2.2.6, scipy 1.16.0 (the frozen lock version; current SciPy
removed the `sqrtm(disp=False)` API), numba 0.61.2, llvmlite 0.44.0,
setuptools 80.9.0 and material-hasher commit
`c9c6bd0bf694a7fb8b489e6a772e3e0f5f52b133`. The overlay's
structuregraph-helpers import aliases modern Pymatgen's
`BrunnerNNRelative` to its old `BrunnerNN_relative` spelling.

A detached LeMat worktree at `deps/lemat_genbench` is frozen at
`fbf1ba4855934acb8fe87135315504899064080d`; its source is unchanged.
The official LeMaterial/LeMat-Bulk-MLIP-Hull dataset revision
`70d505bb294c16658a9551e2b81ace69a28d9790` supplies full ORB/MACE/UMA
hulls. Local parquet symlinks and sparse composition matrices follow
the frozen LeMat 119-slot atomic-number encoding. Preparation scripts
and SHA256 records are in `generated/mlip_bias_study/setup/`.

ORB uses `orb_v3_conservative_inf_omat`; MACE uses `medium-0b3` for
energies and `medium-mpa-0` for reference-compatible embeddings; UMA
uses `uma-s-1`, task `omat`. All three pass GPU energy and embedding probes.
The scorer forces single-point evaluation and matches formation-energy
elemental references to each published MLIP hull, preserving the known
UMA reference-discrepancy warning. Per-model metadata is cleared before
ASE conversion while the original annotations and physical structure
are retained. Checkpoint saving copies structures before serialization
and restores nested NumPy arrays on reload.

Run the complete arms sequentially, with both tracks per arm:

```bash
scripts/platforms/zeus/run_mlip_bias_genbench_gpu1.sh
```

The launcher uses GPU 1, four scoring workers, two BLAS/OpenMP threads,
CPU affinity 16-23 and the same exclusive study lock as relaxation. It
processes EquFlashV2, PET, TECE and EquiformerV3 in that order.
Metrics and resumable score checkpoints are logged online to W&B.
Local working copies are under `generated/mlip_bias_study/metrics/`.
A `zeus_genbench_gpu1.stop` marker stops between tracks. Successful-gene
coverage differs between arms and must accompany any metric comparison.

Setup, complete dependency pins, source snapshots, model probes, hull
SHA256 records and the successful diagnostic metrics are preserved in
W&B compatibility artifact `mlip-bias-zeus-genbench-setup-20261006:v0`
under run `mlipbias1`. Production scoring runs are deterministic:

| Arm | Free | Fixed symmetry |
| --- | --- | --- |
| EquFlashV2 | `862c7545` | `affa83ae` |
| PET-OAM-XL | `66a1782e` | `cd151da3` |
| TECE | `ba9b6910` | `05e24948` |
| EquiformerV3 | `d4580d97` | `dc9ff602` |
| eSEN | `4704fb8e` | `f2b4b2e8` |
| TACE first pass | `740aba93` | `59a325bf` |

Run links use `https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/<id>`.
The launch log is `generated/mlip_bias_study/zeus_genbench_gpu1_20261006.log`;
per-track progress is in each output directory's `scoring.log`.

Fixed-symmetry scoring reuses completed free-track scores only for
byte-identical CIFs with matching source/configuration/reference identities.
The current track keeps its own validity and source-gene annotations.
Unused node embeddings and force arrays are omitted from score checkpoints;
all energy quantities and graph embeddings required by the requested metrics
are retained. The first production track resumed from its 100-structure
checkpoint after this checkpoint-size optimization.

On the next 2026-10-06 W&B refresh, eSEN's complete 2,698-row snapshot
`mlip-bias-e18262dd1ae93acd:v271` became available: 1,836 successful
trials and 718 selected genes in both tracks. It was downloaded and both
tracks queued after the first four arms, sharing the exclusive GPU 1 lock.
Other arms are still incomplete; all snapshots are pinned independently
of any later recovery on the originating machine.

The complete TACE first-pass snapshot `mlip-bias-8eb9d70c27c24afd:v108`
is queued after eSEN as `tace_oam_l_first_pass` (1,925 successful trials,
747 selected genes). Its current recovery ledger is incomplete. The report
labels this older snapshot explicitly and does not combine its scores with
the ongoing recovery.

Novelty matching uses the full LeMat-Bulk `compatible_pbe` reference pinned
at revision `0dc17eea904b860ad7288141e9870f67f8e6bb2c` (5,335,299 rows).
The adapter applies StructureMatcher's composition rejection before building
reference structures and stops on the first equivalent structure. The
matching predicate and tolerances are unchanged. Matching results are cached
by geometry, matcher settings and reference revision, including reuse for SUN
and identical fixed/free structures. The frozen reference composition matrix
is used explicitly; it was verified numerically identical to the older local
matrix despite different compressed-file hashes. All 5,335,299 rows also
match the pinned reference dataset's element lists exactly; see
`setup/genbench_reference_alignment_20261006.json`. Six tests pass, including
matcher equivalence for supercells, charged species and unmatched compositions.
Each completed benchmark family and partial novelty cache is also mirrored to
W&B, so restarting the CPU matching stage preserves earlier results.

Four spawned single-point scoring workers were validated on 2026-10-06.
Four concurrent 464-atom copies of valid gene 463 used at most 21,983 MiB
on GPU 1 (including the existing scorer) and reached 100% utilization.
Per-atom energy differences from serial scoring were at most 7.37e-6 eV
on that dense case; the small-structure probe differed by up to 1.71e-4 eV,
consistent with finite-precision GPU evaluation. Six tests also cover ordered
parallel result collection, a single parent checkpoint writer and partial
checkpoint resumption.

The Zeus launcher now defaults to four scoring workers, overridable with
`MLIP_BIAS_SCORING_WORKERS`. The generic runner still defaults to one.
EquFlashV2's two completed serial tracks are retained with their original
metrics and hashes. PET resumed from 150 serially scored structures and
continues with four workers; later tracks use four workers from the start.
Completed tracks are skipped only when their pinned artifact and selected-CIF
hashes still match. CPU matching remains serial.

Composition rejection is cached for immutable benchmark structures during
uniqueness/SUN pair comparisons. Subset and ignored-species matchers retain
their original paths. The accelerated uniqueness calculation reproduced
EquFlashV2/fixed (887 unique of 887) and PET/free (887 unique of 888) exactly,
in 0.67 seconds each; diagnostics are in
`setup/genbench_matching_validation_20261006.json`. Six tests also compare
the pairwise matching matrix with the original StructureMatcher.

The CPU report worker waits for all twelve source tracks, then recomputes
the seven benchmark families on the same valid, fully scored gene intersection across
all arms and both tracks. It uses cores 8-11, two threads and no visible
GPU, reusing saved single-point properties and graph embeddings. The
two-track EquFlashV2 smoke check completed all families on 887 common
valid genes. The comparison artifact contains full-cohort and common-cohort
metrics plus per-scorer paired/common-gene energy tables and provenance.

eSEN/free gene 722 (Cs16Rb4Hg2) has no neighbors inside UMA's 6 Å cutoff;
its minimum periodic distance is 6.0449 Å. A fresh CPU process reproduced
the rejection, ruling out GPU memory as its cause. ORB/MACE cover all 657
valid structures and UMA covers 656. The source full-cohort result retains
`final_results=false`, with its two-model Fréchet distance labelled as
incomplete. The common-cohort comparison excludes unsupported single points
and requires all three scorers for every included structure. eSEN/fixed has
complete ORB/MACE/UMA coverage for all 657 valid structures. The diagnostic
is `setup/genbench_esen_722_fresh_probe_20261006.json`.

## Finished GenBench comparison, 2026-10-06

All twelve source scoring attempts and all twelve common-cohort benchmarks
finished at WyFormer HEAD `3f9ec13525b217dc5997401ef0c592ea5af0f735`.
GPU 1 scoring jobs have exited. The shared cohort is 440 valid structures
with matching compositions and complete ORB/MACE/UMA scores in every arm
and both tracks. All seven families and all three Fréchet models pass the
consistency audit. Eleven source tracks have complete coverage; eSEN/free
retains the documented empty-graph UMA failure.

The [comparison run 6c532304](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/6c532304)
logs artifact `mlip-bias-genbench-comparison-6c53230488bb6d61:v1` and five
interactive tables: full-cohort metrics, common-cohort metrics, paired
free/fixed energy statistics, common-gene energy statistics and numerical
formation-reference residuals. Source snapshots and complete setup records
are in `mlip-bias-zeus-genbench-setup-20261006:v6` under `mlipbias1`.
Local working copies are in `generated/mlip_bias_study/metrics/report/`.

The report CPU worker uses cores 8-11 with no visible GPU. Its helper and
audit scripts are mirrored into the comparison/setup artifacts. The audit
checks all 12 source gene mappings and coverage counts, the one explicit
failure, all 12 matched benchmark results, three-model Fréchet coverage,
and all table row counts. UMA's independent formation-energy reevaluations
produce paired residuals up to 0.00163269 eV/atom; those outputs and
per-gene residuals are retained explicitly.

Updating the first launcher while it was running caused a shell parse
error after its eight tracks had saved and uploaded normally. A fresh
invocation verified their pinned input hashes, skipped them and exited
successfully. The eSEN/TACE queue and comparison report also finished.


## Two-machine migration — 2026-10-06

At Zeus HEAD `ebd0ad6`, with the dated source snapshots in W&B, the user
authorized both RTX 6000 Ada GPUs and delegated the existing iapetus Codex
session. The stopped iapetus round-robin parents were prevented from starting
another batch; TACE was stopped only after its active trial flushed a row.
The outgoing ledgers were verified against W&B by SHA-256 before any Zeus
writer resumed them. The final migration boundary was ORB
`mlip-bias-717bb362639c2c90:v353` (1,776 rows), Prophet
`mlip-bias-79114e403dd3fde7:v350` (1,744), TACE
`mlip-bias-8eb9d70c27c24afd:v134` (2,566), eSEN
`mlip-bias-e18262dd1ae93acd:v271` and EquiformerV3
`mlip-bias-756e18ed10a3f423:v268` (both 2,698).

As of 21:29 SGT, GPU 0 is running TACE's interrupted recovery ledger, then
will finish ORB and Prophet sequentially. GPU 1 is retrying TECE's 414 failed
first-pass trials, then EquiformerV3's 1,517 and eSEN's 862. Each GPU runs
one arm at a time. The first all-arms GPU 1 queue finishes its active
25-trial batch and uploads before a waiting replacement takes the same
lock with just TECE/EquiformerV3/eSEN. No GPU 1 writer for GPU 0's arms
is permitted by that replacement. Windows `0:11` and `0:12` contain the
partitioned queues; `0:10` contains the initial batch until its exit.
The user's active tmux window remains unchanged.

```bash
scripts/platforms/zeus/run_mlip_bias_migration_gpu0.sh
scripts/platforms/zeus/run_mlip_bias_migration_gpu1.sh tece_oam_rra_1_0 equiformerv3_dens_oam esen_30m_oam
```

The working copies are under `generated/mlip_bias_study/zeus_migration/`,
keeping the earlier scored snapshots intact. The per-GPU locks are
`zeus_gpu0.lock` and `zeus_gpu1.lock`; children inherit their lock, preventing
an orphaned writer from losing ownership. CPU affinity is 8–15 on GPU 0
and 16–23 on GPU 1, with two BLAS/OpenMP threads per process. Normal ORB/
Prophet passes use two calculator workers and a 1,200-second whole-trial
limit. Failure-recovery passes use one worker and 3,600 seconds.
TACE resumes missing trials without `--retry-failed`, then receives its
own persisted allocator/timeout recovery pass. ORB and Prophet receive
the same selective serial recovery after their complete first passes.

`zeus_failed_retry.json` preserves the original failed rows and a copied
original ledger before TECE/eSEN/EquiformerV3 requeue them exactly once.
A restart resumes missing rows without discarding new successes or
requeueing newly recorded failures. These originals and transition plans
are mirrored to compatibility artifacts. Once every planned pass finishes,
`zeus_relaxation_complete.json` pins the final relaxation artifact only
after its uploaded CSV hash matches the local CSV. A full 2,698-row ledger
alone does not declare a recovery queue finished.

All six adapters passed GPU energy/force/stress probes and reproduced a
saved kept structure's energy within `5.97e-5 eV/atom`. The environment
uses pinned TACE source `81f65a4c188bd09cec8d1419388f7afdcc1b6fd0`, Prophet
`dcfdc0b143978652d573630f1d2a6ce09e0c610a`, and EquiformerV3
`a7300c58df683dc99cb48027d5bfd4c887486c48`. The new `deps/migration`
overlay contains TACE 0.2.0, Prophet-MLIP 0.1.0, e3nn 0.6.0, Lightning/
PyTorch-Lightning 2.6.1, Hydra 1.3.7, TorchMetrics 1.9.0, Lightning-
Utilities 0.15.3 and the Zeus-local OpenEquivariance 0.7.0 wheel. All were
installed with `--no-deps`; Torch and CUDA were not replaced. TACE/Prophet
use this overlay before `deps/genbench`; eSEN/Equiformer retain the prepared
FairChem 1.10/NumPy 1.26 overlay. The gated eSEN checkpoint is accessible.
TECE retains OEQ; TACE retains e3nn. The adapter now honors an explicit
`TACE_USE_CUE=0`, avoiding a silent switch when cuEquivariance is installed.
Checkpoint SHA-256s, package versions and scientific manifests are included
in `mlip-bias-zeus-migration-setup-20261006` and per-arm migration artifacts.
Queue runs are `zeusmigrationgpu020261006` and
`zeusmigrationgpu120261006`; the original arm run IDs remain unchanged.

Iapetus now runs NequIP alone in window `0:1`, starting from 1,741 recorded
rows (1,670 successes), on its original run `4fb7a19a`. Its dedicated
launcher uses the tested copied runner, a 1,200-second normal limit, then
a one-worker 3,600-second allocator/timeout recovery. The runner/launcher
are mirrored in `mlip-bias-nequip-recovery-runner-20261006:v0`. Remote
coordination status is `generated/mlip_bias_study/iapetus_delegated_status_20261006.json`.
The earlier six-arm GenBench report remains a pinned historical snapshot;
updated scoring must use distinct snapshot identities and explicit final
completion markers, as described in [updated_genbench.md](updated_genbench.md).

At 21:44 SGT the final-snapshot monitor is running in `0:13`, the updated
GenBench watcher in `0:14` (GPU 0, four scorer workers), and the nine-arm
CPU reporter in `0:15`. Scoring waits for the same GPU lock until relaxation
releases it. The completion index is mirrored to
`mlip-bias-final-relaxation-index-20261006`, under run
`mlipstudycompletion20261006`. Iapetus's independent writer-exit observer
and final monitor require the 3,600-second recovery journal complete and
the uploaded CSV hash verified before publishing its marker; their source
is in `mlip-bias-nequip-iapetus-monitor-20261006:v0`, run `nqmon1006`.
The updated nine-arm metrics remain pending.

### Temporary GPU 0 loan, 2026-10-07

At approximately 00:45 SGT the user-approved Cu–Ge–Te study requested GPU 0
for up to roughly seven hours. `zeus_gpu0.stop` and
`zeus_genbench_gpu0.stop` stop both GPU 0 queues. The CPU-only controller
`generated/mlip_bias_study/setup/lend_gpu0_20261007.py` waits for TACE's
next saved trial and corresponding recovery-journal entry, sends SIGINT
to the identified runner, and allows its final ledger/CIF upload to finish.
It takes the released GPU 0 study lock, verifies no GPU 0 compute processes
remain, and verifies the uploaded TACE CSV hash before atomically writing
`GPU0 free` to `/tmp/claude-1000/gpu0_handoff.txt`.

The controller waits for the explicit `GPU0 returned` response and no
GPU 0 compute processes before removing the loan's stop markers and
restarting TACE/ORB/Prophet plus the scoring watcher with existing journals
and namespace. It never automatically resumes based on elapsed time.
GPU 1 recovery and iapetus NequIP continue. State/source are mirrored under
`mlip-bias-zeus-gpu0-loan-20261007`, run `zeusgpu0loan20261007`, at baseline
`ebd0ad64bd90e1a52f0afe2c697aab049962d99a` plus the uploaded source.

GPU 0 was returned at 05:43 SGT. Its return controller's `tmux new-window`
target was corrected from `0` (occupied window 0) to `0:` (the session).
TACE resumed from its unchanged 2,614-row ledger, pinned as
`mlip-bias-8eb9d70c27c24afd:v141` at the pause boundary. The relaxation
queue is now in `0:3`, followed by ORB and Prophet; the scoring watcher
is in `0:4`, waiting for the shared GPU lock with the same
`recovery-20261006` namespace. Both loan stop markers were removed.

At 14:40 SGT, TECE completed all 414 original failed-trial retries with
2,698 successes, final artifact `mlip-bias-55f04b6c21bc18d4:v128`. Its
updated free/fixed scoring is complete alongside EquFlashV2 and PET.
The other six arms and the final nine-arm comparison remain pending.
TACE reached 2,690 successes before its runner received SIGSEGV on
gene 755/trial 1, whose initial energy was approximately `1.03e21 eV`;
NumPy norm/outer-product overflow warnings preceded the native failure.
This is not an observed CUDA OOM. The fresh-process retry retains the
one-worker 3,600-second budget and enables `PYTHONFAULTHANDLER=1`.

EquiformerV3 reached 1,306 successes before a W&B shared-service busy
timeout surfaced through `AuthenticationError` during an artifact upload.
The final ledger upload succeeded, and the retry was restarted without
discarding rows. The queue now removes inherited `WANDB_SERVICE` from
each calculator child's environment so it initializes a separate service.
It reached 1,309 successes by 14:40 SGT. Both queue windows are restored,
GPU 0 in `0:3` and GPU 1 in `0:12`; the remaining ordering is unchanged.

The fresh retry reproduced TACE's native failure in
`spglib.get_symmetry_dataset` after its warm-up CIF had NaN fractional
positions. The shared relaxer's symmetry helper now rejects non-finite
cells/positions before native spglib; valid geometry passes the identical
inputs. The runner records the resulting ValueError as an ordinary failed
trial and can continue to the next draw. Five geometry-regression tests
plus the 24 existing relaxation/migration tests passed, and GPU 0 was
restarted with this guard. This preserves the trial protocol and labels
invalid geometry honestly instead of treating a native exit as an OOM.

### Scoring transfer to GPU 1, 2026-10-08

Completion on 2026-10-09 at 19:28 SGT: all nine arms and eighteen scoring
attempts finished, the common comparison has 874 fully scored valid genes,
and the report audit and W&B publication verification passed. Final
[run `d115c18c`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/d115c18c)
publishes `mlip-bias-genbench-comparison-d115c18c2610b276:v0` with source
baseline `ebd0ad6` plus recorded patches. The supervisor and campaign panes
exited; Zeus GPUs are idle. eSEN/free retains its documented unsupported
UMA score at gene 722; the shared cohort excludes it and is complete.

Update on 2026-10-09: all nine relaxation arms are final. Prophet's final
artifact is `mlip-bias-79114e403dd3fde7:v418` (2,679 successes, 19 OOM
failures). Its scoring was held after a float32 CSV/JSON precision mismatch
and resumed on GPU 1 at 18:57 SGT after validating the repair, passing ten
CPU tests, and explicitly reviewing/resetting the restart budget. The
previous eight scoring attempts remain unchanged. See
[the scoring notes](updated_genbench.md) for the precise validation rule.
Source/status are in W&B run `mlipcampaignstatus20261009`, baseline
`ebd0ad6` plus recorded patches. The final report remains pending.

At 22:58 SGT, eight relaxation arms are final. eSEN's recovery finished
with 2,698 successful draws, pinned as `mlip-bias-e18262dd1ae93acd:v306`.
NequIP on iapetus finished at 2,658 successes and 40 OOM failures after
all 91 serial recovery candidates were attempted, pinned as
`mlip-bias-4fb7a19aed6c26da:v479`. Prophet has 245 retry draws pending on
GPU 0 (2,452 successes and one failure recorded so far). GPU 1 and
iapetus are idle after completing their assigned relaxation queues.

At 23:00 SGT, the idle GPU 0 scoring watcher was stopped at its polling
boundary and transferred to GPU 1, retaining namespace `recovery-20261006`
and all checkpoints. Stop markers kept supervision from relaunching it
during the transfer. After its namespace lock was released, scoring restarted
with `MLIP_BIAS_GPU=1`; the CPU supervisor's scoring assignment also
uses GPU 1. Three arms have updated free/fixed metrics at this boundary;
the nine-arm report remains pending. Provenance is baseline `ebd0ad6`
plus the supervision source mirrored under `zeussupervision20261007`;
the final ledger pins are indexed by `mlipstudycompletion20261006`.

### Unattended queue supervision, 2026-10-07

By 15:30 SGT TACE finished at 2,697 successes, one NaN-geometry failure
`(755, 1)`, with all 100 recovery-journal candidates accounted for and final
pin `mlip-bias-8eb9d70c27c24afd:v146`. GPU 0 moved on to ORB, then Prophet.
GPU 1 continues EquiformerV3 before eSEN; NequIP continues on iapetus.

`scripts/platforms/zeus/supervise_mlip_bias_campaign.py` is a CPU-only
controller, launched in `mlip-supervisor` with `CUDA_VISIBLE_DEVICES=""`.
It watches the two relaxation queues, scoring, the completion monitor and
the report queue by their tmux window names. Only absent/dead jobs restart.
The GPU queues retain their lock, journal and arm order; foreign GPU
compute processes defer a new relaxation launch. Recorded failed trials
are preserved. GPU stop markers and the explicit GPU 0 loan state suppress
restarts. Scoring shares the relaxation lock and retains its namespace.

At most three consecutive restarts without durable progress are allowed;
the job then remains held with `review_required=true` in
`zeus_migration/supervision_state.json`. New saved trial/journal progress
resets that budget; changing a completion timestamp alone does not.
Source/state are mirrored to `mlip-bias-zeus-supervision-20261007` under
run `zeussupervision20261007`. The five CPU tests cover bounds, marker/loan
scope, true recovery completion and the nine-arm reporting gate.

Stop this controller between checks with
`generated/mlip_bias_study/zeus_migration/supervisor.stop`. It does not stop
an active calculator; use the documented per-GPU stop marker for that.
When all nine score exports and the common report are ready, the controller
requires the report's audit to pass, its W&B run to finish and its uploaded
source-index SHA to match before declaring the campaign complete. It saves
the pinned report identity and mirrors the publication proof to W&B.

### Failure pass requested 2026-10-09

At 20:39 SGT a separate `failure-recovery-20261009` pass is running on the
67 failures in the completed nine-arm ledgers. The original comparison
`d115c18c` is preserved. Baseline is commit `ebd0ad6` plus the dated source
snapshots; progress is in W&B run `mlipfailurepass20261009`, and per-arm
relaxation artifacts use fresh identities containing the namespace and
the pinned source artifact.

`retry_mlip_bias_failures.py` copies every retained CIF into an independent
working directory, keeps all 2,698 baseline rows, and replaces a failed row
only after a fresh worker returns a durable result. Each retry has one
worker and a 3,600-second relaxation budget, plus a separate 1,800-second
setup allowance. A native signal permits one further fresh process;
ordinary returned failures are attempted once. Journals, original rows,
new trial logs and CIFs are mirrored in W&B. Both queues share the existing
per-GPU study lock and respect `zeus_gpuN.stop` and the new pass's
`gpuN.stop` markers at trial boundaries.

GPU 0's sequential order is EquFlashV2 (four), PET (two), TACE (one),
EquiformerV3 (one), then NequIP (40). NequIP 0.19.1's pure Python overlay
was copied from iapetus into `deps/nequip`. Its cached model ZIP has SHA256
`99c3799b28026f1ecf66c413292038a27a0749d4d4d7cd0b3a642f2e68df9e9c`;
no Torch or CUDA packages were installed. A Zeus E/F/stress probe passed.

GPU 1 probes Prophet `use_kernel=True, use_compile=True` with the existing
Zeus OpenEquivariance 0.7.0 wheel. Kernel-only Prophet already matches the
baseline on four cases: identical energies, maximum force error
1.47e-5 eV/Å, maximum stress error 1.19e-7 eV/Å³. The combined compile
probe is CPU-bound after 20 minutes and will be bounded at 30 minutes.
The validated kernel-only factory is the fallback. The original eager
factory stays available, with its historical configuration unchanged.

At 21:00 SGT the combined probe was interrupted after its 30-minute
overall setup allowance. It had not returned its first proposed Si2
prediction. The traceback shows TorchDynamo creating graph-break resume
bytecode and deep-copying instructions; observed RSS was about 12 GiB,
GPU memory about 0.9 GiB. This combination is reported in
[pytorch-zeus issue 3](https://github.com/kazeevn/pytorch-zeus/issues/3),
without assigning an unisolated root cause to the wheel. GPU 1 now uses
the validated `use_kernel=True, use_compile=False` factory, run `202a06fb`.
Its first previously OOMing 104-atom draw completed with peak Torch
allocation 10,734,595,584 bytes. The worker's scalar NumPy-array JSON
handling was repaired; the first completed output is preserved as a
serialization diagnostic and the unfinished journal entry is resumed.

All eight EquFlashV2/PET/TACE/EquiformerV3 candidates are attempted:
EquFlashV2 recovered three, PET two, TACE and EquiformerV3 reproduced their
geometry failures. Final pins are `mlip-bias-7db8ec70d394b383:v4`,
`mlip-bias-62623eb6ef9dcee7:v2`, `mlip-bias-c423a109838f51f2:v1`, and
`mlip-bias-7de6bed52ea9fa87:v1`. W&B 0.30's `.name` can retain `:latest`;
completion explicitly combines the collection name with `.version`, and
downloaded ledger hashes were verified before repairing the first four
completion markers. GPU 0 now runs NequIP's 40 candidates, run `d65bf0bd`.

EquFlashV2/PET overflow recovery raises **scratch capacity**, with bounded
growth on `NeighborOverflowError`; cutoffs, checkpoint weights and modeled
edges are unchanged. Both backend probes passed. One EquFlashV2 case
`(474, 0)` completed after this change; `(398, 1)` passed the old capacity
limit then hit a Warp CUDA OOM during rattle as the model drove the cell
to extreme compression and energy. Numerical completion and physical
quality are separate, and new GenBench scoring uses the original frozen
rules. eSEN/free gene 722 remains the documented unsupported UMA score.

The new GenBench watcher uses `--completion-only` to avoid historical
markers overriding recovery pins. TECE/eSEN/ORB scoring is reused with
hash-verified sources. `monitor_mlip_failure_pass.py` publishes completion
evidence and runs the common-cohort report/audit after all source tracks
are ready. No old report, namespace or completed-run marker is rewritten.

### Failure pass completion and scoring resume, 2026-10-11

Relaxations finished on 2026-10-10 at 00:15 SGT with 64/67 failures
recovered. Prophet recovered 19/19 (kernel enabled, compile disabled),
final `mlip-bias-202a06fb20885902:v20`; NequIP recovered 40/40 on Zeus,
`mlip-bias-d65bf0bd70d518fd:v40`. EquFlashV2 recovered 3/4, PET 2/2;
TACE and EquiformerV3 each retain one geometry failure. The remaining
EquFlashV2 `(398, 1)` failure is a Warp CUDA OOM during cell collapse.

Scoring stopped after three restarts at a provenance mismatch in cache
reuse. On 2026-10-11 the scorer was repaired to distinguish relaxation
provenance from the frozen scoring contract. Exact CIF bytes and identical
scorer/reference configuration remain required; target checkpoint identity
continues to include the new artifact/ledger/namespace. Eight CPU scorer
tests passed, including recomputation of a changed CIF and rejection of
changed scoring contracts. The reviewed scoring budget reset is preserved
in `scoring_restart_review_20261011.json`. At 13:20 SGT the watcher resumed
on GPU 0, namespace `failure-recovery-20261009`; the CPU monitor will audit
and independently verify the new report publication. Source baseline is
`ebd0ad6` plus the dated W&B setup source snapshots.

The new report completed at 14:14 SGT on 2026-10-11:
[W&B run `1aaf8610`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/1aaf8610),
artifact `mlip-bias-genbench-comparison-1aaf8610d42b5413:v0`.
All 18 matched tracks and seven benchmark families passed on 884 common
genes, including every original 874-gene member plus ten additions.
The audit passed; the finished W&B run and downloaded source-index SHA256
`baa64670474328753070170fd0c2bf17b0effdc9e8c5bfeb11355d8c704b3957`
were verified. No new panel failures occurred; eSEN/free UMA gene 722
remains the sole unsupported full-cohort single point.

The independent relaxation audit verifies the 67 original retry keys and
unchanged prior successful rows. All 52 changed selected CIFs parse.
Prophet's 19 retries peaked at 9.31–20.41 GiB Torch allocation; NequIP's
40 peaked at 3.80–11.14 GiB. Reserved memory reached approximately
43.93 GiB, so these are allocation statistics, not total device use.
The historical eager Prophet runtime comparison does not measure this
optimized backend. Backend probes, ledgers, diagnostics and source patches
are preserved in the dated W&B setup/recovery artifacts. Campaign panes
exited automatically; interactive sessions remain open.
