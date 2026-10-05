# MLIP bias study on iapetus

EquFlash was excluded from the study on 2026-10-05 in favor of EquFlashV2.
The historical EquFlash probe artifacts are preserved. On the same date,
the user assigned PET-OAM-XL and EquFlashV2 to the other, larger-VRAM machine,
where they will run those arms themselves. Both remain in the study; future
setup and relaxation of these two arms belong on that machine. Iapetus
continues its existing workers.

On 2026-10-05, TECE finished its first pass with 2,698 recorded trials,
2,284 successes and 896 selected genes per track, and uploaded its final
snapshot to W&B run `55f04b6c`. GPU 1's freed K20c was then assigned to TACE
retries, retaining the original e3nn backend and 600-second timeout. The
773 first-pass failures and complete original ledger were preserved in W&B
before requeuing. Use `scripts/platforms/iapetus/run_mlip_bias_gpu1_tace.sh`
to resume this pass. Its initial invocation used `--retry-failed`; omit that
flag when resuming to avoid requeuing newly failed trials. The existing
tmux window `0:2` shows this worker and logs to
`generated/mlip_bias_study/gpu1_tace_retry_20261005.log`.

Run from the WyFormer checkout. Check W&B runs and local processes before starting a command; **never run two processes for the same arm concurrently**. Prophet, NequIP, and ORB together exceed GPU 0's memory, so alternate them with `scripts/platforms/iapetus/run_mlip_bias_gpu0.sh`. It runs five trials per arm per invocation and checkpoints each batch to W&B. Its local working copies and W&B artifacts carry the same trial identities across invocations.

```bash
scripts/platforms/iapetus/run_mlip_bias_gpu0.sh
scripts/platforms/iapetus/run_mlip_bias_gpu1.sh
scripts/platforms/iapetus/run_mlip_bias_gpu2.sh
```

The three wrappers assign GPU 0/1/2 distinct physical CPU pairs 0–1/2–3/4–5 and cap OpenMP, MKL, OpenBLAS, and NumExpr at two threads per job. `run.sh` forwards those limits into the container using Docker CPU affinity. This prevents the three concurrent jobs from oversubscribing iapetus's six physical cores.

TACE was installed into the isolated `generated/mlip_bias_study/deps/tace` overlay from source commit `81f65a4c188bd09cec8d1419388f7afdcc1b6fd0`, along with `lightning`, `hydra-core`, `matscipy`, `e3nn`, `configargparse`, `lmdb`, `torchmetrics`, `lightning-utilities`, `opt-einsum`, and `opt-einsum-fx`, all without replacing the container's custom Torch. This overlay is a local working copy. To resume on another machine, recreate it from that pinned TACE source or use a compatible TACE environment.

Prophet's pinned source commit is `dcfdc0b143978652d573630f1d2a6ce09e0c610a`, copied into `generated/mlip_bias_study/deps/prophet`; it uses `e3nn` and `matscipy` from the TACE overlay. Its factory disables the optional OpenEquivariance kernel. The exact checkpoint URL is recorded in the arm's `study_manifest.json` and in `scripts/mlip_bias_factories.py`. If a trial fails from GPU memory pressure, rerun that arm with `--retry-failed` on an otherwise free GPU; the round-robin script intentionally skips failed rows until they can be retried explicitly.

NequIP 0.19.1 is installed into `generated/mlip_bias_study/deps/nequip` without changing Torch. `scripts/mlip_bias_factories.py` loads the published `mir-group/NequIP-OAM-XL:0.1` package eagerly, without TorchScript or AOTInductor. The 260 MB package is cached under `wyckoff_transformer.paths.cache_root()/nequip` so every short GPU 0 batch reuses it. Its first full relaxation passed.

After the 2026-10-01 image update, the WyFormer environment has importable `openequivariance` 0.7.0, `torch_scatter` 2.1.2, and `metatomic-torch` 0.1.18. The prior EquFlash, EquiformerV3, PET, and eSEN failures describe the old image; updated probes are below. The full LeMat-GenBench scoring environment has not been validated. The LeMat-GenBench checkout is outside the WyFormer mount; expose it to the container with `WYFORMER_EXTRA_MOUNTS=/home/kna/lemat-genbench:/lemat-genbench:ro` when that environment is ready. Then call `scripts/run_mlip_bias_genbench.py` for each completed arm and each `--track` with `--lemat-root /lemat-genbench`; its default 2,698-trial guard prevents partial final metrics. If the scoring environment's CUDA wheels cannot target these GPUs, run scoring on a later compatible GPU host after downloading the relaxation artifacts from W&B. Do not substitute CPU relaxation for an unfinished GPU arm.

On 2026-10-03, GPU 2 became free after the TACE first pass. Updated-image probes use `generated/mlip_bias_study/deps/fairchem` with isolated NumPy 1.26.4, SciPy 1.15.3, Numba 0.61.2, and FairChem's missing Python dependencies. EquFlashV2 additionally uses `generated/mlip_bias_study/deps/equflash` with unpinned cuEquivariance Python packages. Both still stop at missing `torch_sparse`. PET's original overlay shadows the image's compatible metatomic build; use `generated/mlip_bias_study/deps/pet_clean` to expose only UPET and metatrain plus isolated `metatensor-learn`. PET then reaches missing metatrain architecture dependencies. These overlays are diagnostic working copies, not completed model environments. The dated results and probe files are described in [the study notes](../../mlip_relaxation_bias_study.md).

The 2026-10-04 replacement image `ghcr.io/kazeevn/pytorch:2.14.0-cuda11.8-cudnn8.7-iapetus-r3` includes cuDNN 8.7 and `torch_sparse` 0.6.18 and is now the launcher default. The EquiformerV3 arm additionally needs the official source at `generated/mlip_bias_study/deps/equiformer_v3_src` (commit `a7300c58df683dc99cb48027d5bfd4c887486c48`) before the FairChem overlay in `PYTHONPATH`; its model registration is absent from FairChem alone. PET uses the `pet_clean` overlay plus `python-hostlist`. Updated probe outcomes and the GPU 2 memory limit are recorded in [the study notes](../../mlip_relaxation_bias_study.md).

Resume EquiformerV3 on GPU 2 with `scripts/platforms/iapetus/run_mlip_bias_gpu2_equiformer.sh`. Its working copy is `generated/mlip_bias_study/equiformer_v3_dens_oam`; the runner restores the W&B artifact when the local ledger is absent. Do not start it while the `gpu2_equiformer` tmux session is still running.

After OMat24 access was granted on 2026-10-04, eSEN's authenticated checkpoint downloaded from `facebook/OMAT24` and its GPU probe passed. Its working copy is `generated/mlip_bias_study/esen_30m_oam`. Use `scripts/platforms/iapetus/run_mlip_bias_gpu2_roundrobin.sh` to alternate ten trials per invocation between eSEN and EquiformerV3 on GPU 2. Do not also run either single-arm GPU 2 wrapper while the round robin is active.

The 2026-10-04 21:29 resumption uses the user's existing default-socket tmux session `0`, with windows `1:mlip-gpu0`, `2:mlip-gpu1`, and `3:mlip-gpu2`. Check `tmux list-windows -t 0` as well as Docker before launching another worker; this resumption does not use the historical `mlipbias` socket. Each window shows live output and appends it to `generated/mlip_bias_study/gpu{0,1,2}_resume_20261004.log`, and retains its pane after exit. All three workers use the r3 launcher default and the existing W&B run IDs. See [HANDOFF.md](../../../HANDOFF.md) for the latest operational state.

On 2026-10-04 at 21:34–21:36 Asia/Singapore, with WyFormer HEAD `6d17079e3181073dbbeda3371385a01a39e451b3`, OEQ 0.7.0 reported both `BUILT_EXTENSION=True` and `USE_PRECOMPILED_EXTENSION=True` in the r3 image. A tiny fused convolution passed forward/backward execution on K20c GPU 1. The installed TACE OEQ scatter wrapper also matched an e3nn tensor-product-plus-scatter reference on scalar/vector test inputs: maximum output difference `2.38e-7`, maximum gradient difference `7.63e-6` (float32). These are kernel compatibility checks, not full checkpoint equivalence or speed measurements. Probe JSONs are recorded in the [compatibility W&B run](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/mlipbias1).

The small kernel probes did not change the TECE/TACE calculators. `run.sh` does not forward `TACE_USE_OEQ`; a host-shell export alone does not enable it in the container. Set it inside the container before constructing the calculator, or pass `enable_oeq=True` to `TACEAseCalc`. Keep `CUDA_VISIBLE_DEVICES` restricted to a single physical GPU because OEQ compiles for the first visible GPU architecture.

The subsequent actual-checkpoint benchmark on 2026-10-04 (WyFormer HEAD `6d17079e3181073dbbeda3371385a01a39e451b3`) used TECE's same GPU 1/two-core environment and baseline/OEQ/OEQ/baseline ordering. `scripts/benchmark_tace_oeq.py` tested Si2, three successful raw study draws (12, 16 and 24 atoms), their retained structures, and a complete 12-atom relaxation with the study's original rattle seed, fmax and timeout. Both backends used the identical checkpoint SHA-256 `9f36562582d931347c3904f763e820edcaf5f27c3f13beb6776e49fcc7de38bb`. The baseline had no OEQ modules; each accelerated pass had four. Across the seven single-point cases and repeat passes, maximum differences were `4.77e-7 eV/atom` in energy, `6.44e-6 eV/Å` in forces, and `8.94e-8 eV/Å³` in stress. Median full relaxation time fell from `47.176 s` to `43.414 s` (1.087x throughput); warmed single-point throughput improved by 1.095x. Final relaxed energies agreed within `6.36e-7 eV/atom`, and all four relaxations accepted the rattle. Peak reserved memory across the cases rose from `1.961 GB` to `2.590 GB`; this is a sample measurement, not a bound for larger cells.

Following the user's conditional authorization, `run_mlip_bias_gpu1.sh` now sets `TACE_USE_OEQ=1 TACE_USE_CUE=0` inside the container. TECE resumes in the same tmux window and W&B run after 2,395 recorded trials, starting with `(891, 0)`. Its `backend_transition.json` records the boundary, previous ledger hash and benchmark provenance; the relaxation runner includes it in W&B snapshots and reports the loaded OEQ module count. The benchmark directory and key source snapshots are in W&B compatibility artifact `mlip-bias-tece-oeq-benchmark-20261004`, under [run mlipbias1](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/mlipbias1). The four checkpoint/resume tests passed. The completed TACE-OAM-L first pass was not switched: its distinct checkpoint needs its own benchmark.

The resumed calculator reported four OEQ modules, and its first 40-atom trial `(891, 0)` completed successfully at 21:58 Asia/Singapore and was recorded as row 2,396. The next trial started normally. All three worker windows remained active. Resume confirmation and the updated notes are in compatibility artifact `mlip-bias-tece-oeq-resume-20261004` under the same W&B compatibility run.

### The AOTI warning

The installed TACE source emits `AOTI is not enabled for energy, forces, stress, or virials outputs` whenever those properties are requested without its compiled model path. It is a general acceleration notice, not a measured finding about this worker. OEQ accelerates selected convolutions independently of whole-model compilation, so the notice still appears with OEQ enabled.

On 2026-10-04, source inspection in the r3 image (Torch `2.14.0.post3`, WyFormer HEAD `6d17079e3181073dbbeda3371385a01a39e451b3`) confirmed that TACE's AOTI export calls `torch._inductor.aoti_compile_and_package`, and its in-process compilation uses the Inductor backend. Triton is absent as required by this platform's environment rules. The installed Torch Inductor scheduler rejects CUDA devices with compute-capability major version below 7 when Triton is unavailable; the K20c is 3.5 and the GTX 750 Ti is 5.0. Thus the standard CUDA AOTI/Inductor path is unavailable on iapetus. Setting `TACE_USE_COMPILE=1` is not a supported speed improvement here. Compiling on a newer GPU does not make its generated kernels compatible with these cards. See [environment.md](environment.md#triton-exclusion) and [TACE's acceleration documentation](https://tace.readthedocs.io/en/latest/guide/acceleration.html). No worker was stopped or changed for this source inspection.

### Relaxation timeout

On 2026-10-04, at WyFormer HEAD `6d17079e3181073dbbeda3371385a01a39e451b3`, the user doubled the per-trial limit from 300 to **600 seconds**. All five study launchers now pass `--relax-timeout 600`, including the single-arm TACE/Equiformer launchers for future use. The three active worker windows were stopped and resumed with their existing ledgers/run IDs; interrupted trials are replayed from their common raw starts. TECE retains OEQ and its warning filter. The original input manifest and scientific study identity remain unchanged.

Each arm persists its effective timeout and change boundaries in `relaxation_settings.json`; the runner includes that file in W&B snapshots, records the effective and source limits in the run summary, and uses the saved setting on subsequent resumes without an explicit flag. Earlier failed rows are retained; changing the limit does not requeue them. TACE's completed first pass remains a 300-second-budget result, with a 600-second setting prepared for any future retries. The transition record, tests and source snapshots are logged under compatibility artifact `mlip-bias-timeout-600-20261004` in [run mlipbias1](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/mlipbias1). Thirteen timeout, recovery and warning-filter tests passed, including recovery of the 600-second override and preservation of a 450-second successful row on resume.

Later on 2026-10-04, the user requested showing SciPy's `logm result may be inaccurate` RuntimeWarning only once. The relaxation runner and benchmark now install a display filter that deduplicates the warning family per process rather than its changing residual text. It preserves the first message and its source; unrelated warnings remain visible. TECE was checkpointed after a saved trial and restarted in its existing window with OEQ still enabled; batch workers load the filter on their next invocation. Eight warning-filter and checkpoint/resume tests passed. Source snapshots and the verification record are in compatibility artifact `mlip-bias-logm-warning-once-20261004` in [run mlipbias1](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/mlipbias1).

The restarted TECE log confirmed four OEQ modules and one logm warning during the next trial's cell relaxation, with no repeated copies while computation continued.
