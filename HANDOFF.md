# MLIP relaxation bias study handoff — 2026-10-06

## Objective and provenance

Relax the 2,698 raw PyXtal draws from W&B artifact
`symmetry-advantage/WyckoffTransformer/protocol_ehull_adamw_wsd_5x_cfg-20260924-223159.cfg-grid-e0p05-w5:v0`
with eight retained MLIPs from the frozen Matbench Discovery top ten plus
`orb_conserv_inf`, then
compute LeMat-GenBench ORB, MACE and UMA single-point energies and all requested
metrics except relaxation RMSD. The raw draws must be common starting structures;
the input artifact's selected CIFs already carry ORB bias. The Matbench ranking
snapshot is commit `71633e8bdfdfd41d56d64b1d777e5686d9eda3ec` and the
LeMat-GenBench checkout is commit `fbf1ba4855934acb8fe87135315504899064080d`.
WyFormer HEAD at the earlier stop was `7638adc28976bdde8ae0b04a5cc0cc51dc4ed56d`;
at the latest resumption it is `6d17079e3181073dbbeda3371385a01a39e451b3`.
The last code commit before this handoff update is
`d54050bf38454e99429f916f7f5723d4228af8fb`.
Study code and docs are committed together with this handoff; the dated
compatibility artifacts in
[W&B](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/mlipbias1)
include earlier snapshots of the key adapters and launchers.

User constraints: use iapetus GPUs, not iapetus CPU, for relaxations; log model
setup and GPU failures; keep outputs recoverable through W&B. Iapetus has six
physical CPU cores. The active workers were limited to separate pairs of two.
GRACE-3L-OAM-L is intentionally skipped because the user does not want the
TensorFlow dependency. On 2026-10-05, the user also excluded EquFlash in favor
of EquFlashV2. EquFlashV2 remains planned; historical EquFlash probe records
are retained, but no further EquFlash setup, relaxation or scoring is planned.

On 2026-10-05, the user assigned **PET-OAM-XL and EquFlashV2 to the other,
larger-VRAM machine**. On 2026-10-06 that machine was identified as **zeus**,
which has two shared RTX 6000 Ada GPUs with 46,068 MiB each. These two arms
remain in the study and must use the same raw input artifact, published
checkpoints, relaxation protocol and 600-second timeout, with results and
environment provenance logged to W&B. Their arm run links are still pending.

## Migration to zeus — decision and boundary on 2026-10-06

The user requested that only one MLIP continue on iapetus and that the rest of
the study move to the proper GPUs on zeus. Keep **NequIP-OAM-XL** as the sole
iapetus MLIP: it has been reliable on the K20c, with 1,662 successful results
and 71 failures in the 1,733-row snapshot below. ORB is a control rather than
an MLIP. Do not restart the existing GPU 0 round-robin launcher after the
migration boundary because it also schedules ORB and Prophet.

This is the local-ledger snapshot observed on 2026-10-06 at approximately
21:00 Asia/Singapore. NequIP and TACE were still advancing when it was taken,
so later rows must be preserved before moving their writers.

| Arm | Rows | Successful | Failed | Assignment | W&B run |
| --- | ---: | ---: | ---: | --- | --- |
| ORB control | 1,766 | 1,756 | 10 | zeus: finish first pass | [717bb362](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/717bb362) |
| Prophet-OAME-MBD | 1,739 | 1,067 | 672 | zeus: finish first pass | [79114e40](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/79114e40) |
| NequIP-OAM-XL | 1,733 | 1,662 | 71 | **iapetus: sole remaining MLIP** | [4fb7a19a](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/4fb7a19a) |
| TECE-OAM-RRA-1.0 | 2,698 | 2,284 | 414 | zeus: retry failures | [55f04b6c](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/55f04b6c) |
| TACE-OAM-L | 2,563 | 2,464 | 99 | zeus: resume active retry pass | [8eb9d70c](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/8eb9d70c) |
| eSEN-30M-OAM | 2,698 | 1,836 | 862 | zeus: retry failures | [e18262dd](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/e18262dd) |
| EquiformerV3+DeNS-OAM | 2,698 | 1,181 | 1,517 | zeus: retry failures | [756e18ed](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/756e18ed) |
| PET-OAM-XL | 0 | 0 | 0 | zeus: first pass | pending |
| EquFlashV2 | 0 | 0 | 0 | zeus: first pass after compatibility fix | pending |

Create a clean single-writer boundary before launching anything on zeus:

1. On iapetus, let the active GPU 0 batch save and upload, stop the round
   robin, and resume only NequIP using the NequIP command inside
   `run_mlip_bias_gpu0.sh`. A dedicated NequIP-only launcher should replace
   the round robin for unattended use.
2. Let TACE save its active trial, stop its GPU 1 worker, and force a final
   W&B snapshot. Its original failed rows were already requeued. Resume this
   interrupted retry ledger on zeus **without** `--retry-failed`; using that
   option again would discard newly recorded retry failures and requeue them.
3. Confirm the latest ORB, Prophet and TACE ledgers and CIF outputs are present
   in their W&B artifacts before starting their zeus writers. Never run the
   same arm on both hosts concurrently.
4. ORB and Prophet have incomplete first passes. Resume missing trials without
   `--retry-failed`, then decide whether to retry their failures. TECE, eSEN
   and EquiformerV3 completed their first passes; invoke `--retry-failed` once
   on zeus, then omit it after any interruption.

Follow [the zeus agent brief](docs/platforms/zeus/agent_brief.md): use the host
`.venv/bin/python` or `uv run`, never iapetus's container or package overlays;
check `nvidia-smi` because both GPUs are shared; and select one GPU explicitly
with `CUDA_VISIBLE_DEVICES=0` or `1`. Recreate and probe each model's pinned
dependencies on zeus before its first relaxation. The current zeus environment
has not yet been verified for these adapters, PET initialization,
EquFlashV2/cuEquivariance compatibility, the gated eSEN checkpoint, or access
to the current W&B arm artifacts. Run at most one arm per free RTX 6000 Ada.
No zeus worker or new PET/EquFlashV2 W&B run had been verified when this
handoff section was written.

## Previous state — GPU 1 reassigned 2026-10-05

On 2026-10-05, the user requested a test of
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on GPU 2. The eSEN/
Equiformer round robin was stopped after Equiformer's current 10-trial batch
and W&B upload completed. A matched, fresh-process eSEN comparison used the
same six raw draws (8, 20, 28 and 69 atoms) and a diagnostic 120-second cap.
Baseline and expandable-segments modes both completed the same two draws and
OOMed on the same four; successful timings were 24.660/105.534 seconds versus
24.722/105.747 seconds. Expandable segments explicitly failed to map another
20 MiB when only 12--14 MiB was physically free. This sample shows capacity
failures rather than fragmentation recoverable by the option, so it remains
disabled. The original 600-second round robin resumed in tmux window `0:3`.
The full diagnostic is in compatibility artifact
`mlip-bias-gpu2-expandable-segments-20261005` under W&B run `mlipbias1`.

TECE completed its full 2,698-trial first pass normally at 2026-10-05 07:40
Asia/Singapore (23:40 UTC on October 4). It recorded 2,284 successful trials
and selected 896 genes in each free/fixed-symmetry track. The final snapshot
was uploaded to W&B run `55f04b6c` before the worker exited with status 0.

When the user reported idle GPU 1 on 2026-10-05, its free K20c was assigned
to retry TACE's 773 failed first-pass trials (766 OOM, seven timeouts), with
the saved 1,925 successes retained. Before requeuing, the full arm snapshot,
original ledger and dated transition were preserved in W&B run `8eb9d70c`;
the transition artifact is `mlip-bias-tace-gpu1-retry-20261005`.
The retry uses the original e3nn backend, the same checkpoint and seeds,
the 600-second timeout and CPU cores 2–3. TACE has not been switched to OEQ.

The existing tmux window `0:2` is reused for this TACE pass; its log is
`generated/mlip_bias_study/gpu1_tace_retry_20261005.log`. The launcher is
`scripts/platforms/iapetus/run_mlip_bias_gpu1_tace.sh --retry-failed` for the
initial requeue. Resume an interrupted pass **without** `--retry-failed`, so
newly recorded failures are not repeatedly requeued. GPU 0 and GPU 2 retain
their existing round-robin workers. PET and EquFlashV2 remain assigned to
the user's other machine. TECE's failed trials remain recorded.

## Resumption history — 2026-10-04

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
- EquFlashV2 passes `torch_sparse` import but fails in the upstream
  cuEquivariance convolution wrapper: the unpinned Python-only fallback is
  `SegmentedPolynomialNaive`, which lacks `buffer_num_segments`. Compatible
  cuEquivariance operations or a validated source change are required.
- PET-OAM-XL passes the metatomic ABI and architecture imports with
  `deps/pet_clean` plus `python-hostlist`, but GPU 2's 2 GiB memory is exhausted
  during model initialization, before a single point. The user will run this
  arm on the other, larger-VRAM machine, alongside EquFlashV2.

The detailed dated log, model ranking, and scoring contract are in
[docs/mlip_relaxation_bias_study.md](docs/mlip_relaxation_bias_study.md).

## Resuming the relaxation arms

First check `nvidia-smi`, `docker ps`, `tmux list-windows -t 0`, and any legacy
`tmux -L mlipbias list-sessions` so
there is never a second writer for the same arm. The working ledgers are under
`generated/mlip_bias_study/<arm>/trials.csv`. Preserve these local directories;
they can contain trials newer than the most recent W&B snapshot. If a local
ledger is absent, the same runner invocation downloads its latest W&B artifact
before continuing. All run IDs and checkpoint locators are deterministic from
the study manifest.

The following iapetus launchers describe the historical three-GPU schedule:

```bash
scripts/platforms/iapetus/run_mlip_bias_gpu0.sh
scripts/platforms/iapetus/run_mlip_bias_gpu1.sh
scripts/platforms/iapetus/run_mlip_bias_gpu2_roundrobin.sh
```

GPU 0 alternates ORB, Prophet and NequIP in five-trial batches; do not use that
round robin after the 2026-10-06 migration. The original GPU 1 launcher runs
TECE, while `run_mlip_bias_gpu1_tace.sh` runs TACE's active failed-trial pass.
GPU 2 alternates eSEN and EquiformerV3 in ten-trial batches; both first passes
are now complete. Each wrapper pins its GPU and two physical CPU cores. Keep a
NequIP-only worker on iapetus and follow the migration section for all other
arms. To recover out-of-memory rows on zeus, use
`scripts/run_mlip_bias_relax.py --retry-failed` with the same arm output,
model, checkpoint and input exactly once per newly initiated retry pass; do not
run it concurrently with another writer for that arm.

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
