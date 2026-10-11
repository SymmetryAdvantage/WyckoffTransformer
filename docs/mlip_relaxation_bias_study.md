# MLIP relaxation bias study

The additional failure pass completed relaxation on 2026-10-10: 64 of 67
draws recovered, including all 19 Prophet and all 40 NequIP OOM failures.
Prophet used the validated kernel backend with compilation disabled;
NequIP moved to the larger Zeus GPU. EquFlashV2 recovered three of four
and PET both; one EquFlashV2 CUDA OOM during cell collapse and two geometry
failures (TACE/EquiformerV3) remain. On 2026-10-11 the separate frozen
GenBench comparison resumed after repairing cache reuse across recovery
provenance: only byte-identical CIFs with identical scoring contracts reuse
scores. Eight scorer tests passed. The new comparison completed on
**2026-10-11 at 14:14 SGT**, with all seven benchmark families passing
across 18 tracks on **884 common valid, fully scored genes**. This retains
all 874 original common genes and adds ten. The publication is
[W&B run `1aaf8610`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/1aaf8610),
artifact `mlip-bias-genbench-comparison-1aaf8610d42b5413:v0`.
The audit and independent publication hash check passed. Only the original
eSEN/free UMA score at gene 722 remains unsupported in the full-cohort
exports; no new panel scoring failures occurred. The original report below
remains intact. Source baseline is `ebd0ad6` plus dated W&B patches in the
recovery setup/progress runs. Earlier dated states below are historical.

On 2026-10-09 the user requested another recovery pass at the 67 remaining
relaxation failures. It is in progress under a separate namespace and
[W&B progress run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/mlipfailurepass20261009).
The published comparison below is preserved. Recovery uses the same raw
draws, seeds, checkpoint weights and relaxation schedule; backend changes
must pass energy/force/stress checks. New selected geometries receive
frozen GenBench scoring, with byte-identical prior single points reusable
only after checkpoint/reference and CIF-hash validation. A new matched
comparison will follow completion. Source baseline remains `ebd0ad6` plus
dated W&B patches; operational details are in the platform notes.

The nine-arm campaign completed on 2026-10-09. The final audited comparison
is [W&B run `d115c18c`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/d115c18c),
artifact `mlip-bias-genbench-comparison-d115c18c2610b276:v0`, at source
baseline `ebd0ad64bd90e1a52f0afe2c697aab049962d99a` plus dated W&B patches.
All 2,698 common raw PyXtal draws were attempted by every MLIP, including
the planned recovery passes. The report compares free and fixed-symmetry
tracks across all nine arms on 874 common valid genes with finite ORB,
MACE and UMA energies and graph embeddings. All requested benchmark
families completed on this matched cohort, and the audit passed. Each
energy/scorer field and its reference provenance remain separate.

Full-cohort eSEN/free results retain one unsupported UMA single point:
gene 722, Cs16Rb4Hg2, has minimum periodic distance 6.044885 Å and no
neighbors inside UMA's 6 Å cutoff. ORB/MACE scored it, but UMA returned
no energy or embedding. The full-cohort completion flag remains false;
gene 722 is excluded from every matched track. Relaxation failures remain
recorded separately, including 19 Prophet OOM failures after serial Zeus
recovery and 40 NequIP OOM failures after serial iapetus recovery. The
earlier six-arm, 440-gene first-pass report below remains historical.

The current cohort contains eight MLIPs plus the ORB control. GRACE is skipped
to avoid its TensorFlow dependency. On 2026-10-05, at WyFormer commit
`35104ff`, the user excluded EquFlash in favor of EquFlashV2; V2 remains in
the cohort. Historical EquFlash probes below are retained as setup records,
but EquFlash is excluded from future setup, relaxation and scoring.

On 2026-10-05, the user assigned PET-OAM-XL and EquFlashV2 to the other,
larger-VRAM machine, Zeus. Both arms completed their normal and recovery
passes on 2026-10-06 and remain in the cohort. Their runs should use the same raw input artifact, checkpoints and
relaxation protocol with W&B logging; the Zeus timeout was increased from
600 to 1,200 seconds on 2026-10-05, with timeout failures requeued. The machine
is Zeus. Both GPU 1 energy/forces/stress probes passed on 2026-10-05, at
WyFormer HEAD `c3cb84e4dcfd6e08360e4ee7a2232786143c1a9a`, with local Torch
2.14.1/CUDA 13.4. The normal Zeus worker finished EquFlashV2 before continuing PET.
EquFlashV2's first successful full relaxation is in
[run 9b5e07e1](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/9b5e07e1);
seven successful PET trials started prematurely by an initial launcher bug
were preserved in [run a4bbbd2f](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/a4bbbd2f).
The count bug was corrected and sequential launch checks pass. The detached
worker, package/source versions and local-wheel issue reports are in
[the Zeus notes](platforms/zeus/mlip_bias.md). Setup provenance and source
snapshots are in W&B compatibility artifact `mlip-bias-zeus-setup-20261005`
under run `mlipbias1`. On 2026-10-06 the migration handoff at commit
`9e4a1de` assigns only NequIP to iapetus. ORB and Prophet move to Zeus
to finish their first passes, TACE to resume its interrupted retry ledger,
and TECE/eSEN/EquiformerV3 to retry failures once. The single-writer
boundary must be verified before starting these Zeus arms; see [HANDOFF.md](../HANDOFF.md).

On 2026-10-07, at baseline commit `ebd0ad64bd90e1a52f0afe2c697aab049962d99a`
plus the dated W&B source snapshots, TECE completed its 414 failed-trial
retries with 2,698 successes, final `mlip-bias-55f04b6c21bc18d4:v128`.
TACE completed at 2,697 successes and one non-finite-geometry failure,
final `mlip-bias-8eb9d70c27c24afd:v146`, with its 100-candidate recovery
journal complete. Both final ledgers were verified against their uploaded
CSV hashes. A reproduced TACE crash was traced to NaN optimized positions
entering native spglib; the symmetry helper now rejects non-finite geometry
while preserving its inputs for finite geometry. The remaining five arms
are still running or queued. The current [status/source record](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/mlipcampaignstatus20261007)
and [supervision record](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/zeussupervision20261007)
retain diagnostics and bounded-restart provenance. The updated comparison
uses namespace `recovery-20261006` and distinct scoring identities; the
six-arm first-pass comparison below remains a historical snapshot. Current
commands and queue state are in the platform notes and handoff.

On 2026-10-05, at WyFormer commit `44abb58`, TECE's completed first-pass
snapshot was confirmed in [W&B run 55f04b6c](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/55f04b6c):
2,698 recorded trials, 2,284 successes and 896 selected genes per track.
Its freed K20c GPU 1 was assigned to retry TACE's 773 failed first-pass
trials, retaining 1,925 successful rows, the original checkpoint and e3nn
backend, and the 600-second timeout. The prior ledger and transition are
preserved in artifact `mlip-bias-tace-gpu1-retry-20261005` under
[TACE run 8eb9d70c](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/8eb9d70c).
These historical counts describe relaxation coverage; scoring was pending
at that snapshot. Completed Zeus scoring is recorded below.

Also on 2026-10-05, at WyFormer commit `6e19309`, GPU 2's allocator was tested
with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Fresh baseline and
candidate processes relaxed the same six eSEN raw draws, ordered identically,
under a diagnostic 120-second cap. Both modes succeeded on the 8- and 20-atom
draws and OOMed on the same two 28-atom and two 69-atom draws. Successful
times were 24.660 and 105.534 seconds at baseline and 24.722 and 105.747
seconds with expandable segments. The candidate allocator reported memory
mapping failures when another 20 MiB was requested with only 12--14 MiB
physically free. It gave no coverage or speed benefit in this matched sample,
so GPU 2 continues without the option. Results are stored in W&B compatibility
artifact `mlip-bias-gpu2-expandable-segments-20261005`.

Initial status on 2026-10-01: GPU relaxations were running; final LeMat-GenBench metrics were **not yet available**. WyFormer commit: `c360bb35e28ca9fdebc1a892ccc8d3d8b51a68c6`. The leaderboard snapshot is Matbench Discovery commit `71633e8bdfdfd41d56d64b1d777e5686d9eda3ec`; LeMat-GenBench is commit `fbf1ba4855934acb8fe87135315504899064080d`.

Later on 2026-10-01 the user stopped all three GPU jobs. The iapetus image was then replaced with `iapetus/pytorch:2.14.0-cuda11.8-py312`. The WyFormer environment in that image now has `openequivariance` 0.7.0, `torch_scatter` 2.1.2, and `metatomic-torch` 0.1.18; all three import successfully. The compatibility outcomes below were measured before that image update. In particular, the EquFlash, EquiformerV3, PET, and eSEN failures need new model-level probes before they can be treated as current blockers. At 2026-10-01 19:57 Asia/Singapore, the user requested resumption and the GPU 0 ORB/Prophet/NequIP round robin, GPU 1 TECE, and GPU 2 TACE jobs restarted from their local trial ledgers on the new image. The new image needed `WANDB_DATA_DIR` set to the mounted W&B store to upload artifacts; the launcher now sets it. The user then identified CPU oversubscription, so the three jobs were capped at two distinct physical CPU cores each before resuming again.

On 2026-10-02 at 10:52 Asia/Singapore, GPU 0 was found idle after its NequIP batch had finished and uploaded normally about six hours earlier. Its outer round-robin shell had exited without a recorded model error. It was restarted from the local ledgers in a detached `tmux -L mlipbias` session named `gpu0`; GPU 1 and GPU 2 remained active. All three GPUs were occupied again at 10:54. The stopped worker's last ORB, Prophet, and NequIP batches had reached W&B, so no recorded trials were lost.

On 2026-10-03, TACE finished its 2,698-trial first pass and uploaded the ledger. Of those trials, 1,925 succeeded, 766 ran out of the GTX 750 Ti's 2 GiB GPU memory, and 7 timed out. The free GPU 2 was used for updated-image compatibility probes. EquFlashV2, EquFlash, and EquiformerV3 now pass the former `torch_scatter` import blocker but stop at missing `torch_sparse`. The first source-build attempt used an ineffective `FORCE_CUDA=0` flag; the package's actual CPU-only switch is `FORCE_ONLY_CPU=1`. A corrected isolated build started, but was stopped after discovering another four-worker `torch_sparse` build on the host, to avoid oversubscribing its six physical cores. No model probe has yet passed `torch_sparse` import. EquFlashV2 also needs the cuEquivariance Python packages directly imported by its source; unpinned `cuequivariance` and `cuequivariance-torch` 0.12.0 passed that import without CUDA ops packages. PET-OAM-XL passed the old metatomic ABI failure when UPET alone was overlaid on the image's new metatomic build, and its ASE adapter was corrected to provide the required model name. Its older `metatrain` stack still lacks compatible architecture dependencies; no PET GPU single point has passed. An eSEN-30M-OAM registry entry points to the checkpoint URL in the frozen Matbench metadata, but that URL returns HTTP 401 before the calculator can import FairChem. These are setup findings, not model performance results.

On 2026-10-04, the host image changed to `ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-iapetus-r2`. Its `torch_sparse` 0.6.18 imports alongside Torch 2.14.0.post2 and `torch_scatter` 2.1.2. EquFlashV2 and EquFlash then reach a cuEquivariance convolution wrapper that expects `buffer_num_segments` on the selected polynomial implementation; the unpinned Python-only cuEquivariance installation falls back to `SegmentedPolynomialNaive`, which has no such attribute. Their source requires compatible cuEquivariance operations or a tested implementation change, so neither GPU single point passes. The official EquiformerV3 source at `atomicarchitects/equiformer_v3` commit `a7300c58df683dc99cb48027d5bfd4c887486c48` registers the model absent from plain FairChem. With that source, its Si2 GPU energy, forces and stress pass. Its first 24-atom relaxation runs out of GPU 2 memory, while an 8-atom trial completes. CUDA cache cleanup after memory failures raised the sample's successful count to 5 of 21; the full arm started on GPU 2 in a detached session and checkpointed to its [W&B run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/756e18ed). PET passes the remaining architecture import after adding `python-hostlist` but runs out of GPU 2 memory while loading its 2.8 GiB checkpoint, before its Si2 single point. The [official OMat24 checkpoint repository](https://huggingface.co/facebook/OMAT24) requires access conditions; the frozen Matbench eSEN checkpoint URL returned HTTP 401. These are GPU 2 setup and capacity outcomes, not comparative MLIP results.

Later on 2026-10-04, access to the [official OMat24 repository](https://huggingface.co/facebook/OMAT24) was granted. The container sees the user's Hugging Face token, and `huggingface_hub.hf_hub_download` fetched `esen_30m_oam.pt` into the mounted cache. The eSEN adapter now uses that authenticated official URL instead of the unauthenticated frozen Matbench URL. Its GPU Si2 energy, forces and stress pass, and 7 of its first 10 generated relaxation trials completed on GPU 2; all ten trial rows and retained structures were uploaded to its [W&B run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/e18262dd). EquiformerV3 was paused after 105 locally recorded trials to free GPU 2 for this probe; its prior snapshots and local ledger remain resumable. Both arms are now alternating in ten-trial batches on GPU 2.

## Cohort and protocol

The input is W&B artifact `symmetry-advantage/WyckoffTransformer/protocol_ehull_adamw_wsd_5x_cfg-20260924-223159.cfg-grid-e0p05-w5:v0`. Its `pyxtal.extxyz` contains 2,698 successful raw draws from 1,000 genes. The study starts every relaxation from these raw draws. The input artifact's selected CIFs have already been selected and relaxed with ORB, so using them as common starts would retain the bias being measured. The frozen [Matbench Discovery snapshot](https://github.com/janosh/matbench-discovery/tree/71633e8bdfdfd41d56d64b1d777e5686d9eda3ec) defines the model list and published environments.

`scripts/run_mlip_bias_relax.py` applies the WyFormer protocol's fixed-cell, symmetry-preserving cell, released cell, and rattle stages at `fmax=0.05 eV/Å`. The input manifest supplies the original 300-second per-trial timeout. On 2026-10-04 the user doubled the study's timeout to **600 seconds** for resumed/future trials, using a recorded runtime override; the source artifact remains unchanged. Each arm's `relaxation_settings.json` records timeout changes and their completed-trial boundaries, is included in W&B snapshots, and is restored on resume even when `--relax-timeout` is omitted. Earlier timeout failures remain recorded, so coverage reflects both timeout budgets until those rows are explicitly retried. On 2026-10-05 the user raised the Zeus EquFlashV2/PET-OAM-XL limit to **1,200 seconds** and timeout failures from the shorter budget are selectively requeued using `--retry-timeouts`. Iapetus launchers remain at 600 seconds. The user also requested a later Zeus recovery pass, after both normal arms complete: GPU 1, one worker and 3,600 seconds per trial, selectively retrying OOM/timeouts with `--retry-oom-timeouts`; see the Zeus study notes. The runner selects the lowest-energy successful trial for each gene using that relaxation MLIP's own energy. Selection energies are **only** used within each model; they are not compared across models. Fixed-symmetry and free tracks are saved separately. Each arm logs a trial ledger, every retained trial CIF, and the current selected CIFs to W&B artifact type `mlip_bias_relax` every 25 new trials and at normal exit. If the local ledger is missing, rerunning the identical command restores the latest W&B artifact before resuming. The control is `orb_conserv_inf`.

Ranking is the Matbench Discovery site's default Composite Performance Score among active models with the requisite metrics: unique-prototype F1 (weight 0.5), phonon conductivity score (0.4), and geometry optimization RMSD (0.1). This is a frozen ranking, not a claim about which model is best for these generated structures.

| Rank | MLIP | CPS | iapetus GPU status on 2026-10-01 |
| ---: | --- | ---: | --- |
| 1 | Prophet-OAME-MBD | 0.911640 | Running in alternating batches with ORB on GPU 0. Python 3.12, a plain PyTorch single point, and a complete relaxation trial passed with `use_kernel=False`. [W&B run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/79114e40). |
| 2 | TECE-OAM-RRA-1.0 | 0.907627 | Running on K20c GPU 1; Si2 probe and a complete relaxation trial passed. [W&B run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/55f04b6c). |
| 3 | EquFlashV2 | 0.907213 | Retested GGNN source without version pins, cuEquivariance or `nvalchemi-toolkit-ops`. The import reaches missing `torch_scatter`; an unpinned build against Torch 2.14 fails because the CUDA 11.8 compiler rejects its C++20 flag. The V2 source imports `cuequivariance` and `cuequivariance_torch` directly in its convolution and interaction modules, so removing those packages also requires source changes. |
| 4 | EquiformerV3+DeNS-OAM | 0.902273 | Retested upstream source with Torch 2.14, relaxed Python pins and no explicit CUDA wheels. Its `fairchem.core` calculator import fails at missing `torch_scatter`; an unpinned extension build fails against CUDA 11.8 as described above. |
| 5 | GRACE-3L-OAM-L | 0.899947 | Intentionally skipped at user request to avoid carrying its TensorFlow dependency. |
| 6 | PET-OAM-XL | 0.898427 | Installed UPET in an isolated overlay; its `metatomic-torch` binary rejects Torch `2.14.0.post2` (published wheels through 2.13). |
| 7 | TACE-OAM-L | 0.889400 | Running on GPU 2; Si2 probe and completed relaxations passed. At least one larger trial hit the 2 GiB GPU memory limit and is recorded as failed for later `--retry-failed` on a larger GPU. [W&B run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/8eb9d70c). |
| 8 | eSEN-30M-OAM | 0.887887 | FairChem 1.10.0 overlay import fails at `ModuleNotFoundError: torch_scatter`; its published environment pins Torch 2.4 and CUDA 12.1 extensions. No compatible adapter yet. |
| 9 | EquFlash | 0.887707 | Excluded on 2026-10-05 in favor of EquFlashV2. Historical probe: same GGNN `torch_scatter` blocker; its source also imports cuEquivariance directly. |
| 10 | Nequip-OAM-XL | 0.885960 | Running in alternating batches on GPU 0. The unpinned NequIP 0.19.1 adapter and published OAM-XL package run eagerly under Torch 2.14 without compilation; GPU energy, force, stress, and a complete relaxation trial passed. [W&B run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/4fb7a19a). |
| control | `orb_conserv_inf` | — | Running in alternating batches with Prophet on GPU 0; Si2 and complete relaxation probes passed. [W&B run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/717bb362). |

These are **execution setup** outcomes on iapetus, not performance results or claims that a model fails on supported hardware. Probe JSONs and this failure log are saved in a [W&B compatibility run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/mlipbias1). TACE was installed into `generated/mlip_bias_study/deps/tace` without changing the project venv or its custom Torch. Prophet's pinned source and checkpoint run without the optional OpenEquivariance kernel; NequIP uses its eager package loader. The PET and FairChem probes also used isolated overlays. The active GPU 0 arms run in alternating batches because they exceed its memory together; GPU 2 has only 2 GiB.

## Scoring contract

For each completed arm, run LeMat-GenBench on its selected free and fixed-symmetry CIFs. Its `comprehensive_multi_mlip_hull` config uses **ORB, MACE and UMA single-point energies**, each compared only to its own compatible hull; it also requests validity, distribution, diversity, novelty, uniqueness, HHI, SUN and stability metrics. Force `MultiMLIPStabilityPreprocessor(relax_structures=False, calculate_formation_energy=True, calculate_energy_above_hull=True)` while scoring. The upstream `scripts/run_benchmarks.py` currently derives `relax_structures` from whether stability was requested and ignores the config's `relax_structures` setting, so calling it unchanged would relax the structures again. Exclude `relaxation_rmse` and relaxation RMSD metrics. Report per-arm coverage and scoring failures; do not silently compare arms with different successful-gene sets.

No final metric is reported until the applicable relaxation arm and single-point scoring finish. On 2026-10-06, a compatible isolated scoring environment was prepared and tested on Zeus against the frozen LeMat checkout; see [the Zeus scoring notes](platforms/zeus/mlip_bias.md). Results retain scorer-specific energy definitions and the frozen source's known UMA energy/hull discrepancy warning.

## Resume and scoring

Rerun an arm with the same input, output, model, checkpoint and settings. The runner restores its most recent W&B artifact if its local trial ledger is missing, then skips completed trials. Do not run two processes for the same arm concurrently. W&B snapshots are recoverable across machines, but the most recent local trials after a snapshot have not yet been uploaded. Preserve a local working directory if a process exits unexpectedly; restart from it to retain those trials.

`scripts/run_mlip_bias_genbench.py` is the scoring entry point. It refuses to score an incomplete trial ledger by default, can download the relaxation arm with `--arm-artifact`, uses the LeMat-GenBench `comprehensive_multi_mlip_hull` configuration, disables its internal relaxation, records ORB/MACE/UMA single-point energies separately, and uploads its metrics and energies to W&B. It reports per-model and per-quantity scoring coverage, failed gene IDs, and missing primary metrics. Validity, partial single-point scores every 25 structures, and complete single-point scores are mirrored as W&B checkpoints for resumption. Host-specific launch commands and installation notes live in `docs/platforms/iapetus/mlip_bias.md` and `docs/platforms/zeus/mlip_bias.md`.

Formation energies use elemental minima from each scorer's own published MLIP hull. Frozen upstream formation-energy methods subtract PBE elemental references from MLIP totals; the scoring adapter replaces these incompatible references and checks compatibility through `energy_fields.check_compatible`. This correction does not change single-point energies or energies above hull. The validity predicate is preserved, including the frozen source's known charge-validity limitations; its oxidation-state enumeration skips only combinations rejected by the original adjacency rule.

### Zeus relaxation completion, 2026-10-06

At WyFormer commit `3f9ec13525b217dc5997401ef0c592ea5af0f735`, both Zeus arms completed 2,698 trial rows, including a one-worker, 3,600-second OOM/timeout recovery pass. PET has 2,696 successful trials; EquFlashV2 has 2,694 after the requested fresh-parent retry of `(991, 2)` succeeded in 157.8 seconds. That retry peaked at 45,260 MiB GPU memory and recovered from a transient OOM. The remaining EquFlashV2 failures are one OOM `(991, 0)` and three neighbor overflows; PET retains two neighbor overflows. Each arm selects 991 genes in both tracks. Relaxation artifacts are `mlip-bias-9b5e07e1c31040ee:v116` and `mlip-bias-a4bbbd2f9825a3a3:v130`, respectively, under the [EquFlashV2 run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/9b5e07e1) and [PET run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/a4bbbd2f). Fresh-retry diagnostics are preserved in `mlip-bias-zeus-fresh-991-2-20261006` under the [compatibility run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/mlipbias1).

The scoring queue includes both tracks for EquFlashV2, PET, TECE (`mlip-bias-55f04b6c21bc18d4:v111`, 896 selected genes), and EquiformerV3 (`mlip-bias-756e18ed10a3f423:v268`, 480 selected genes). These are different successful-gene cohorts; their raw full-cohort metrics do not establish a controlled ranking. Other arms remain incomplete at this snapshot.

On the next 2026-10-06 W&B refresh, eSEN's complete 2,698-row snapshot
`mlip-bias-e18262dd1ae93acd:v271` became available: 1,836 successful
trials and 718 selected genes in both tracks. It was downloaded and both
tracks queued after the first four arms, sharing the exclusive GPU 1 lock.
Other arms are still incomplete; all snapshots are pinned independently
of any later recovery on the originating machine.

At this 2026-10-06 snapshot TACE's recovery ledger was incomplete,
but its complete first-pass snapshot `mlip-bias-8eb9d70c27c24afd:v108`
is retained. Both tracks of that snapshot are also queued, under the
explicit `tace_oam_l_first_pass` namespace: 1,925 successful trials and
747 selected genes. Its metrics describe the first pass, before recovery.

Novelty matching uses the full LeMat-Bulk `compatible_pbe` reference pinned
at revision `0dc17eea904b860ad7288141e9870f67f8e6bb2c` (5,335,299 rows).
The adapter applies StructureMatcher's composition rejection before building
reference structures and stops on the first equivalent structure. The
matching predicate and tolerances are unchanged. Matching results are cached
by geometry, matcher settings and reference revision, including reuse for SUN
and identical fixed/free structures. The frozen reference composition matrix
is used explicitly; it was verified numerically identical to the older local
matrix despite different compressed-file hashes.
The matrix was also checked against the pinned dataset's elements in all
5,335,299 rows; every row matches. Six tests pass, including
matcher equivalence for supercells, charged species and unmatched compositions.
Each completed benchmark family and partial novelty cache is also mirrored to
W&B, so restarting the CPU matching stage preserves earlier results.

The comparison report includes full-cohort metrics with explicit coverage,
per-scorer matched-gene energy statistics, and all seven benchmark families
recomputed on the intersection of valid, fully scored genes across every included arm and
both tracks. This last comparison reuses saved single-point properties and
embeddings; it runs no calculators or relaxation. Validity is not recomputed
on the already-valid intersection. A two-track EquFlashV2 check passed all
families on the same 887 genes before queuing the full comparison.

eSEN/free has one unsupported UMA single point, gene 722 (Cs16Rb4Hg2):
its minimum periodic distance is 6.0449 Å, beyond UMA's 6 Å cutoff.
A fresh CPU process reproduced the empty-neighbor-graph rejection, so this
is not a VRAM failure. ORB and MACE cover all 657 valid structures; UMA
covers 656. The full-cohort result remains incomplete, including a
Fréchet distance using only two models. This gene is excluded from the
fully scored common-cohort comparison. Diagnostics are in
`setup/genbench_esen_722_fresh_probe_20261006.json`.

The first complete scoring result is EquFlashV2/free, W&B run
`862c7545`: 888/991 valid genes, all 888 scored by ORB/MACE/UMA,
81.7568% novelty, 100% uniqueness and 56.4189% LeMat MetaSUN.
UMA's mean hull distance is strongly affected by several noble-gas
compositions (for example Al2HeNe5 and Ga2Ne7, around 62 and 57 eV/atom).
A fresh UMA-only GPU 1 process reproduced their very large predictions;
the diagnostic values and repeated-call differences are retained in
`setup/genbench_uma_outlier_probe_20261006.json`. These are retained in
the protocol results, alongside the frozen UMA reference warning; no
outlier filtering is applied.

### Completed comparison, 2026-10-06

At WyFormer HEAD `3f9ec13525b217dc5997401ef0c592ea5af0f735`, scoring
attempts finished for all six pinned snapshots in both tracks. The
uncommitted scoring/report adapters and their provenance are preserved in
`mlip-bias-zeus-genbench-setup-20261006:v6` under the compatibility run.
The frozen LeMat commit remains `fbf1ba4855934acb8fe87135315504899064080d`.
The [audited W&B comparison](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/6c532304)
contains artifact `mlip-bias-genbench-comparison-6c53230488bb6d61:v1`:
all seven benchmark families on **440 identical valid, fully scored genes**
in every arm and both tracks, full-cohort metrics with coverage flags,
scorer-specific energy tables and the passed consistency audit. Scoring
uses saved geometries without further relaxation; relaxation RMSE/RMSD
outputs are excluded.

| Relaxation arm | Selected genes | Valid free / fixed | MetaSUN free / fixed on common 440 (%) |
| --- | ---: | ---: | ---: |
| EquFlashV2 | 991 | 888 / 887 | 61.36 / 45.68 |
| PET-OAM-XL | 991 | 888 / 889 | 61.59 / 46.59 |
| TECE-OAM-RRA-1.0 | 896 | 815 / 815 | 61.14 / 46.14 |
| EquiformerV3+DeNS-OAM | 480 | 441 / 441 | 60.68 / 44.32 |
| eSEN-30M-OAM | 718 | 657 / 657 | 61.59 / 45.45 |
| TACE-OAM-L, first pass | 747 | 682 / 683 | 61.59 / 45.68 |

These shared-cohort rates are conditional on successful relaxation,
validity and complete single-point scoring in every track. They do not
estimate performance on all original draws. Full-cohort rates use different
gene sets and are retained separately. Eleven source tracks have complete
ORB/MACE/UMA coverage; eSEN/free retains its one unsupported UMA score and
`final_results=false`. Its full-cohort Fréchet distance uses two models and
is explicitly marked incomplete. Every common-cohort Fréchet distance
uses all three models.

UMA's frozen formation/hull helpers independently reevaluate energy.
Paired formation-minus-total-energy changes therefore have small numerical
residuals, up to 0.00163269 eV/atom; ORB/MACE agree to floating-point
precision. Compositions and compatible reference definitions match.
The reported values are retained, with per-gene residuals and an additional
audit table. The known UMA hull-reference discrepancy, noble-gas outliers
and frozen charge-validity limitations also remain explicit.

At the 17:28 SGT W&B refresh, ORB, Prophet and NequIP had 1,726, 1,694
and 1,691 trial rows out of 2,698. Their completed-arm metrics await
relaxation completion. TACE recovery had 2,475 rows; this report uses its
earlier completed first-pass snapshot. Scoring run IDs for all twelve
tracks, input hashes, model hashes and dataset revisions are in the
comparison artifact's `report_provenance.json` and environment records.
