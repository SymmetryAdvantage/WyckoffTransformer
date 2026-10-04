# MLIP relaxation bias study handoff — 2026-10-04

## Objective and provenance

Relax the 2,698 raw PyXtal draws from W&B artifact
`symmetry-advantage/WyckoffTransformer/protocol_ehull_adamw_wsd_5x_cfg-20260924-223159.cfg-grid-e0p05-w5:v0`
with the frozen Matbench Discovery top ten MLIPs plus `orb_conserv_inf`, then
compute LeMat-GenBench ORB, MACE and UMA single-point energies and all requested
metrics except relaxation RMSD. The raw draws must be common starting structures;
the input artifact's selected CIFs already carry ORB bias. The Matbench ranking
snapshot is commit `71633e8bdfdfd41d56d64b1d777e5686d9eda3ec` and the
LeMat-GenBench checkout is commit `fbf1ba4855934acb8fe87135315504899064080d`.
WyFormer HEAD at the earlier stop was `7638adc28976bdde8ae0b04a5cc0cc51dc4ed56d`;
at the latest resumption it is `6d17079e3181073dbbeda3371385a01a39e451b3`.
Study code and docs are committed together with this handoff; the dated
compatibility artifacts in
[W&B](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/mlipbias1)
include earlier snapshots of the key adapters and launchers.

User constraints: use iapetus GPUs, not iapetus CPU, for relaxations; log model
setup and GPU failures; keep outputs recoverable through W&B. Iapetus has six
physical CPU cores. The active workers were limited to separate pairs of two.
GRACE-3L-OAM-L is intentionally skipped because the user does not want the
TensorFlow dependency.

## Latest state — resumed 2026-10-04

At approximately 21:29 Asia/Singapore, the user requested continuing the study
and using the current tmux session. After confirming that all GPUs were idle
and no study containers were running, the three prepared launchers were
resumed in the existing default-socket tmux session `0`:

- Window `1:mlip-gpu0`: ORB, Prophet and NequIP round robin.
- Window `2:mlip-gpu1`: TECE.
- Window `3:mlip-gpu2`: eSEN and EquiformerV3 round robin.

The original window `0` remains selected. Output is visible in each worker
window and appended to `generated/mlip_bias_study/gpu{0,1,2}_resume_20261004.log`.
The worker windows retain their panes after exit. All three r3 containers were
running and all GPUs showed activity after startup. Existing W&B run IDs and
local ledgers are reused, with the same two-core-per-worker limits. Subsequent
logs confirmed successful new TECE trials and an ORB batch followed by Prophet.
No LeMat-GenBench metrics have been computed yet.

After the user's request to check OEQ, TECE was stopped immediately after its
2,395th trial row was saved, and its complete ledger/CIF snapshot was uploaded
to the same W&B run. The actual TECE checkpoint was benchmarked on GPU 1 in
baseline/OEQ/OEQ/baseline order with two CPU cores. All seven single-point
cases (Si2 and raw/kept structures from three study draws) passed energy,
force and stress equivalence checks. Median full relaxation time was
47.176 s with e3nn and 43.414 s with OEQ; warmed evaluations were 1.095x faster.
Peak reserved memory in the sample rose from 1.961 GB to 2.590 GB.

At approximately 21:56 Asia/Singapore, TECE resumed in the same window
`2:mlip-gpu1` with `TACE_USE_OEQ=1` and `TACE_USE_CUE=0` set inside the
container by its launcher. The other workers were left running. The first
trial assigned to OEQ is `(gene=891, trial=0)`. The arm's
`backend_transition.json` records the cutoff, previous ledger hash, checkpoint
hash and benchmark locator, and is now included in relaxation snapshots.
The benchmark and source snapshots are published under compatibility artifact
`mlip-bias-tece-oeq-benchmark-20261004` in W&B run `mlipbias1`.
The four runner checkpoint/resume tests passed. TACE-OAM-L remains a completed
first-pass arm; its separate checkpoint was not benchmarked or switched.
At 21:58 Asia/Singapore, the first resumed 40-atom trial `(891, 0)` completed
successfully and was saved as row 2,396; `(891, 1)` then started. The resumed
calculator reported four OEQ modules, and all three worker windows remained
active. Resume confirmation is also logged in compatibility artifact
`mlip-bias-tece-oeq-resume-20261004` under W&B run `mlipbias1`.

The user subsequently requested that SciPy's varying-residual `logm result
may be inaccurate` RuntimeWarning print only once. The MLIP relaxation runner
and benchmark now install a family-level display filter at startup, keeping
the first occurrence per process even when the residual changes. Unrelated
warnings retain their existing behavior. Eight warning-filter and
checkpoint/resume tests passed. TECE was stopped after its next trial row was
saved, checkpointed to W&B, and restarted in the same window to load this
change; OEQ stays enabled. Batch workers load it on subsequent invocations.
The filter source and verification record are saved in compatibility artifact
`mlip-bias-logm-warning-once-20261004` under W&B run `mlipbias1`.
The restarted TECE worker reported four OEQ modules and printed one logm
warning during its next trial's cell relaxation; repeated copies were absent.

Check `tmux list-windows -t 0` and Docker before starting any other worker;
the historical `tmux -L mlipbias` server is not used by this resumption.

The user subsequently doubled the per-trial relaxation timeout to 600 seconds.
All prepared study launchers now pass `--relax-timeout 600`; the runtime
override is persisted in each arm's `relaxation_settings.json` and included
in W&B snapshots, so an artifact-only resume also remembers it. The original
input manifest retains its 300-second value and SHA-256, and run IDs remain
unchanged. Before resumption, the worker windows were stopped, their completed
ledgers preserved, and all seven arms' current snapshots uploaded with the
new timeout history. Interrupted trials are replayed under the longer limit;
earlier timeout failures are retained until explicitly retried. TACE's full
first pass was performed with the earlier 300-second limit, although its
future resume setting is now 600 seconds.

At this change the ledger counts were ORB 1,246, Prophet 1,224, NequIP 1,221,
TECE 2,405, TACE 2,698, eSEN 490 and EquiformerV3 575. The dated transition
record and source snapshots are in W&B compatibility artifact
`mlip-bias-timeout-600-20261004` under run `mlipbias1`; WyFormer HEAD is
`6d17079e3181073dbbeda3371385a01a39e451b3`. Thirteen timeout, recovery and
warning-filter tests passed. The three normal worker windows continue to be
used, with TECE's OEQ option and all affinity limits retained.

## Historical state at stop request

The user requested stopping workers after their ongoing relaxation. GPU 0 and
GPU 2 round-robin sessions ended after their active batches and uploaded their
final batch artifacts. TECE on GPU 1 was stopped immediately after a completed
trial row, then a zero-trial invocation uploaded its final ledger and CIFs as
W&B artifact `mlip-bias-55f04b6c21bc18d4`. As of 2026-10-04 20:57
Asia/Singapore, `docker ps` shows no study containers and the `mlipbias` tmux
server has no sessions. The temporary stop marker and launcher guard were
removed. No LeMat-GenBench metrics have been computed yet.

| Arm | Local trials recorded | Successful | W&B run |
| --- | ---: | ---: | --- |
| ORB control | 1,221 | 1,215 | [717bb362](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/717bb362) |
| Prophet-OAME-MBD | 1,201 | 722 | [79114e40](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/79114e40) |
| TECE-OAM-RRA-1.0 | 2,389 | 2,004 | [55f04b6c](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/55f04b6c) |
| TACE-OAM-L | 2,698 | 1,925 | [8eb9d70c](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/8eb9d70c) |
| Nequip-OAM-XL | 1,201 | 1,149 | [4fb7a19a](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/4fb7a19a) |
| EquiformerV3+DeNS-OAM | 505 | 217 | [756e18ed](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/756e18ed) |
| eSEN-30M-OAM | 410 | 277 | [e18262dd](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/e18262dd) |

These are trial-ledger counts, not final comparable metrics. TACE completed its
first pass with 773 failed rows, mostly GPU 2 out-of-memory failures. Failed
rows are intentionally retained; `--retry-failed` requeues them on a larger GPU.
Selections use each relaxation MLIP's own energy only within that arm. The
runner saves free and fixed-symmetry tracks separately and logs a resumable
`mlip_bias_relax` artifact every batch/25 trials and at normal exit.

## Environment and model compatibility

Follow [the iapetus agent brief](docs/platforms/iapetus/agent_brief.md) and
[study launcher notes](docs/platforms/iapetus/mlip_bias.md). Run WyFormer Python
only through `scripts/platforms/iapetus/run.sh`; the host Python can damage the
project venv. The launcher now defaults to
`ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-cudnn8.7-iapetus-r3`, with custom Torch
2.14.0.post3/CUDA 11.8, cuDNN 8.7, `torch_scatter`, `torch_sparse`,
`openequivariance`, and `metatomic-torch`. Local diagnostic package overlays live under the ignored
`generated/mlip_bias_study/deps/`. Recreate those overlays on another machine
using the pinned source commits in the study notes; they are not W&B artifacts.

- Prophet, TECE, TACE, NequIP, ORB, EquiformerV3 and eSEN have passed GPU
  single-point probes. EquiformerV3 needs the official source at commit
  `a7300c58df683dc99cb48027d5bfd4c887486c48` before FairChem on
  `PYTHONPATH`. Its adapter registers that source model. NequIP runs eagerly
  without compilation.
- eSEN's official `facebook/OMAT24` checkpoint is gated. Access was granted
  on 2026-10-04; the iapetus container sees the Hugging Face token and
  `huggingface_hub` downloaded it into the mounted cache. Its probe and
  relaxations pass. A new host needs the same authorized access.
- EquFlashV2 and EquFlash pass `torch_sparse` import but fail in the upstream
  cuEquivariance convolution wrapper: the unpinned Python-only fallback is
  `SegmentedPolynomialNaive`, which lacks `buffer_num_segments`. Compatible
  cuEquivariance operations or a validated source change are required.
- PET-OAM-XL passes the metatomic ABI and architecture imports with
  `deps/pet_clean` plus `python-hostlist`, but GPU 2's 2 GiB memory is exhausted
  during model initialization, before a single point. Try a larger GPU.

The detailed dated log, model ranking, and scoring contract are in
[docs/mlip_relaxation_bias_study.md](docs/mlip_relaxation_bias_study.md).

## Resuming the relaxation arms later

First check `nvidia-smi`, `docker ps`, `tmux list-windows -t 0`, and any legacy
`tmux -L mlipbias list-sessions` so
there is never a second writer for the same arm. The working ledgers are under
`generated/mlip_bias_study/<arm>/trials.csv`. Preserve these local directories;
they can contain trials newer than the most recent W&B snapshot. If a local
ledger is absent, the same runner invocation downloads its latest W&B artifact
before continuing. All run IDs and checkpoint locators are deterministic from
the study manifest.

On iapetus, the prepared launchers are:

```bash
scripts/platforms/iapetus/run_mlip_bias_gpu0.sh
scripts/platforms/iapetus/run_mlip_bias_gpu1.sh
scripts/platforms/iapetus/run_mlip_bias_gpu2_roundrobin.sh
```

GPU 0 alternates ORB, Prophet and NequIP in five-trial batches. GPU 1 runs TECE.
GPU 2 alternates eSEN and EquiformerV3 in ten-trial batches. TACE already has
a full first-pass ledger. Each wrapper pins its GPU and two physical CPU cores.
The GPU 0 and GPU 2 scripts are foreground loops; use detached `tmux` sessions
if they should survive a client disconnect. On another host, adapt CUDA device
selection and package overlays. To recover out-of-memory rows on a larger GPU,
use `scripts/run_mlip_bias_relax.py --retry-failed` with the same arm output,
model, checkpoint and input; do not run it concurrently with the normal arm.

## Scoring still required

`scripts/run_mlip_bias_genbench.py` is prepared but has not been run or validated
against a complete LeMat-GenBench environment. It reads each arm's selected
free/fixed-symmetry CIFs, invokes the `comprehensive_multi_mlip_hull` protocol,
disables internal relaxation, records separate ORB/MACE/UMA single-point
energies, excludes relaxation RMSD, and uploads metrics and energies to W&B.
It refuses an incomplete 2,698-row trial ledger by default; `--allow-partial`
is diagnostic only. The LeMat checkout at `/home/kna/lemat-genbench` must be
mounted into the container, for example with
`WYFORMER_EXTRA_MOUNTS=/home/kna/lemat-genbench:/lemat-genbench:ro` and
`--lemat-root /lemat-genbench`. Its dependency environment and actual scoring
run are still pending. Report per-arm successful-gene coverage and scoring
failures; do not compare energies across different MLIP definitions by column
name.
