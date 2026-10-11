# GenBench after the two-machine relaxation recovery

The additional failure pass completed its updated comparison on
**2026-10-11 at 14:14 SGT**, source baseline `ebd0ad6` plus recorded W&B
patches. Relaxation recovered 64 of the 67 remaining failures, including
all Prophet and NequIP OOM cases. All 18 matched tracks and seven benchmark
families passed on **884 common valid, fully scored genes**, retaining
the original 874 and adding ten. The report is
[run `1aaf8610`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/1aaf8610),
artifact `mlip-bias-genbench-comparison-1aaf8610d42b5413:v0`.
The audit passed and the independently downloaded publication index matches
SHA256 `baa64670474328753070170fd0c2bf17b0effdc9e8c5bfeb11355d8c704b3957`.
The sole unsupported full-cohort score remains eSEN/free UMA gene 722;
no new scoring failures occurred. The original comparison and its sources
below remain preserved. Both recovery worker queues, the scorer and the
CPU monitor have exited.

Completed on 2026-10-09 at 19:28 SGT. All nine final relaxation snapshots
and all eighteen scoring attempts are accounted for. The common comparison
contains 874 valid, fully scored genes across every track and ORB/MACE/UMA;
all requested benchmark families completed and the report audit passed.
The final publication is
[run `d115c18c`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/d115c18c),
artifact `mlip-bias-genbench-comparison-d115c18c2610b276:v0`. The supervisor
verified the finished W&B run and downloaded the published source index to
confirm SHA256 `e8e9cbeaba33e2a28d3fbe6cd7eaee7be5dbb0938191f1202ee6950e6e81e592`.
Its state is complete and all campaign panes exited automatically.

The eSEN/free full-cohort flag remains false for the documented UMA failure
at gene 722. Its Cs16Rb4Hg2 structure has no edges inside UMA's 6 Å cutoff
(minimum periodic distance 6.044885 Å); its missing UMA energy and embedding
exclude the gene from every shared track. The shared comparison is complete.
Source baseline remains `ebd0ad6` plus recorded patches, with final source
and scoring identities preserved in the report. Earlier dated states below
are historical.

Prepared on 2026-10-06 at WyFormer commit `ebd0ad64bd90e1a52f0afe2c697aab049962d99a`.
The source changes accompanying this page are uncommitted until the operator
records their next commit; the scoring coordinator uploads its source files
and pinned input index to W&B. The earlier six-arm comparison remains at
[run `6c532304`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/6c532304),
artifact `mlip-bias-genbench-comparison-6c53230488bb6d61:v1`.

At 21:44 SGT, the watcher is running on GPU 0 in tmux `0:14`, waiting for
the relaxation lock; the CPU reporter is waiting in `0:15`. The snapshot
monitor in `0:13` mirrors both machines' verified completion evidence to
W&B run `mlipstudycompletion20261006`. Updated nine-arm metrics are pending.

On 2026-10-07, the watcher resumed in `mlip-updated-genbench` (`0:4` at
15:30 SGT), alongside the updated reporter in `0:15`. EquFlashV2, PET and
TECE have complete scoring; TACE's final relaxation is now ready and waits
for GPU 0's relaxation queue to release its lock. CPU supervision in
`mlip-supervisor` restarts absent jobs within a three-restart no-progress
budget and verifies the final audited report actually reached W&B before
declaring the campaign complete.

On 2026-10-08 at 22:58 SGT, eSEN completed its recovery with 2,698/2,698
successes (`mlip-bias-e18262dd1ae93acd:v306`), leaving GPU 1 idle. Eight
relaxation arms are final; Prophet has 245 retry draws pending on GPU 0.
At 23:00 SGT, the scoring coordinator was transferred to GPU 1 after its
GPU 0 instance exited gracefully and released the namespace lock. Its existing
namespace and checkpoints are retained. The CPU supervisor now launches
scoring with `MLIP_BIAS_GPU=1`; Prophet continues independently on GPU 0.
EquFlashV2, PET and TECE have complete updated scoring at this boundary;
the final nine-arm comparison remains pending. Source baseline remains
`ebd0ad6` with the updated supervision source mirrored to W&B run
`zeussupervision20261007`.

`scripts/run_mlip_bias_genbench.py --snapshot-namespace recovery-20261006`
uses a new scoring identity containing the namespace, pinned relaxation
artifact, ledger hash, selected CIF hashes and scoring configuration. Both
the W&B run and scoring checkpoint artifact receive this new identity.
Invocations without the option keep the historical IDs and checkpoint format.
A local checkpoint with different input identity is rejected before W&B is
opened. Repeated invocations on the same snapshot resume its checkpoints.

## Completion evidence

On 2026-10-09, the final relaxation index reached all nine arms. Prophet
finished at 10:20 SGT with 2,679 successes and 19 OOM failures, pinned as
`mlip-bias-79114e403dd3fde7:v418`. Its scoring watcher stopped at validation
and reached the bounded three-restart hold. At 18:54 SGT the mismatch was
diagnosed: gene 973/trial 1's CSV energy `-1.8897468` and JSON energy
`-1.8897467851638794` encode the same float32 value. The validator accepts
unequal decimals only when the JSON number is exactly float32-representable
and both numbers round to that identical float32 value. Adjacent float32
values and changed float64 numbers are rejected. Selected trial/minimum,
ledger SHA and CIF hash checks remain required.

Ten CPU tests passed. Prophet's immutable snapshot validates with 986
selected genes per track; the eight recorded snapshot validations remain
identical. The held job was reviewed, its bounded restart count reset with
a recorded explanation, and scoring resumed on GPU 1 at 18:57 SGT. The
CPU reporter remains queued. Source/status are mirrored in W&B run
`mlipcampaignstatus20261009` against baseline `ebd0ad6` plus the recorded
patches. eSEN/free still reports the known unsupported UMA score for gene
722, which will be excluded from the common cohort.

The new Zeus coordinator consumes
`generated/mlip_bias_study/zeus_migration/study_relaxation_completion.json`:

```json
{
  "arms": {
    "nequip_oam_xl": {
      "complete": true,
      "artifact": "symmetry-advantage/WyckoffTransformer/mlip-bias-4fb7a19aEXAMPLE:v999",
      "ledger_sha256": "SHA256_OF_THE_FINAL_UPLOADED_TRIALS_CSV"
    }
  }
}
```

The artifact above is illustrative. Supply the actual final pinned name/version,
including the final persistent recovery completion evidence for NequIP on
iapetus. Each migrated Zeus arm can instead supply its own
`zeus_migration/<arm>/zeus_relaxation_complete.json` with the same fields.
Per-arm markers override the aggregate entry. `complete` must be the boolean
`true`; neither a W&B run's finished state nor a 2,698-row first-pass snapshot
alone certifies that recovery finished.

The migration coordinator writes these markers only after its assigned passes
finish and the final artifact's uploaded ledger hash matches its local ledger.
TECE, EquiformerV3 and eSEN require completion of their persisted failed-trial
retry. TACE, ORB and Prophet require completion of their persisted recovery.
The NequIP entry must be copied from iapetus's final completion evidence.
EquFlashV2 and PET already have final pinned versions `v116` and `v130`;
the scoring watcher reuses their unchanged earlier complete scoring outputs.

For each ready arm the watcher downloads the pinned artifact to an immutable
namespace snapshot directory. It checks exactly 2,698 distinct gene/trial rows,
the original raw-input hashes, track-specific selected trials and energies, selected CIF hashes,
the model identity and, when provided, the final uploaded ledger hash. When
the original EquFlashV2 ledger exists locally, its gene/trial identities must
also match. A pinned local snapshot cannot change on resume.
Selections follow the relaxation runner's minimum-energy rule independently
for each track. Fixed-symmetry outputs can remain usable when the later free
relaxation failed; the validator checks the relevant track's energy and trial
CIF bytes, without requiring the free row's status to be successful.

## Launch

Start one coordinator for the namespace, on the GPU whose relaxation queue
should release first:

```bash
MLIP_BIAS_GPU=1 scripts/platforms/zeus/run_updated_mlip_bias_genbench.sh \
  --namespace recovery-20261006
```

Use `MLIP_BIAS_GPU=0` to choose GPU 0. `--arms` can restrict the requested set;
the default is all nine arms. `--workers 4` is the tested scorer pool size.
The launcher sets the existing GenBench dependency overlay, online W&B logging,
CPU affinity and explicit visible GPU. It changes no environment packages.

The watcher polls every 30 seconds and acquires the same
`zeus_gpu<GPU>.lock` used by relaxation. It holds the lock only while preparing
and scoring a ready snapshot, retaining it in scoring subprocesses. Waiting
for incomplete arms holds no GPU lock. A namespace coordinator lock prevents
two watchers from writing the same scoring outputs. To stop between arms,
create `generated/mlip_bias_study/zeus_genbench_gpu<GPU>.stop`.

Free and fixed-symmetry tracks run consecutively for each ready arm. Fixed
scoring can reuse only byte-identical, fully scored free CIFs. LeMat-GenBench
must remain at `fbf1ba4855934acb8fe87135315504899064080d`; the comprehensive
protocol, all seven families, ORB/MACE/UMA single points, own compatible hull
references, frozen validity and novelty reference remain unchanged. Internal
relaxation is disabled and relaxation RMSD/RMSE outputs are excluded.

Results go under `generated/mlip_bias_study/metrics/recovery-20261006/`.
`scoring_source_index.json` maps each arm to its exact pinned snapshot,
selected CIF hashes, output paths and source metrics hashes. This index and
the scoring source are mirrored to a W&B `mlip_bias_compat` artifact after each
arm. Each scorer publishes its own normal scoring checkpoint and metrics
artifacts. The earlier directories and W&B runs remain intact.

A completed scoring attempt with a documented unsupported single point is
retained with `scoring_complete`/`final_results` false. The watcher advances to
the other arms rather than silently re-running the unsupported structure.
The later common comparison excludes the affected gene from every track.

## Updated nine-arm common comparison

Start the separate CPU reporter alongside the scoring watcher:

```bash
scripts/platforms/zeus/report_updated_mlip_bias_genbench.sh recovery-20261006
```

It waits until the source index contains all nine arms, then runs
`scripts/compute_mlip_bias_common_genbench.py` and
`scripts/report_mlip_bias_genbench.py`. A namespace report lock prevents
concurrent reporters. It requires no GPU and can wait while either machine
continues its relaxation queue. Each completed common track is checkpointed
to W&B; missing local common checkpoints can be downloaded on resume.

Once the source index contains all nine arms, use its output paths as the
authoritative input map for the report. Do not replace the old six-arm report
or its fixed 440-gene cohort. The updated cohort is the intersection of valid
genes with finite values for all three scorers' four exported energy quantities
and all three graph embeddings across all eighteen source tracks. Verify each
gene's fractional composition matches in every track.

The new helpers adapt the earlier frozen six-arm analysis to dynamic input
paths and cohort size, with `tace_oam_l` replacing the historical
`tace_oam_l_first_pass` snapshot. They write to the namespace's `report/`
directory and derive documented failures from the actual exports. Scoring
run IDs come from the explicit snapshot identity, with a historical-ID
fallback for unchanged scores. Physical energy comparisons preserve separate
ORB/MACE/UMA fields and compatibility checks; original UMA predictions and
their measured formation-energy cancellation residuals remain intact.

`scripts/verify_mlip_bias_genbench_report.py` checks source coverage and failure
flags, source hashes, the exact global energy/embedding mask, compositions,
all eighteen matched tracks, all seven families, three-model Fréchet coverage
and the expected dynamic table counts before publication. The reporter then
publishes a new comparison W&B run/artifact with five interactive tables,
the source index, original source hashes and current source/package provenance.
Full-cohort failure flags remain separate from matched-cohort completeness.

## Validation

The CPU tests cover historical-ID preservation, isolation when pinned source,
ledger/CIF hashes, configuration, namespace or track changes, safe checkpoint
resume, rejection before W&B when an output directory belongs to another
snapshot, rejection of `:latest`, complete/unique ledgers, original input hashes,
track-specific selections, fixed-only outputs after failed free relaxations,
final-completion markers and documented incomplete scorer coverage. The report
audit test derives a new failure gene and common mask without historical
hardcoding, and rejects a mismatched composition or mask. Twelve tests passed
on 2026-10-06; Ruff's import/undefined-name checks and launcher shell syntax also
passed. A CPU-only diagnostic reused existing EquFlashV2/PET scores across four
tracks and reproduced all seven families on 887 common genes, then verified
the report tables and provenance. The diagnostic used `--no-wandb` to avoid
publishing a second scientific comparison from the old subset. No GPU scoring
job was launched as part of preparing these files.
