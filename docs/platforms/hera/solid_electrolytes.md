# Liverpool solid-electrolyte follow-up on Hera

The [campaign record](../../solid_electrolyte_campaign.md#liverpool-ionics-extension-2026-11-03)
defines the 19 systems and source exclusions. Launch from this checkout with its
Python 3.14 environment:

```bash
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=online OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    .venv/bin/python -u scripts/platforms/hera/run_se_families_followup.py
```

Use a persistent session or detach the process for a full run. The driver uses
both GPUs for local ORB relaxation, two CPU lanes of 12 cores, and 12 report
workers. Generation and band-gap prediction use GPU 0. It does not call iapetus.
Check GPU load before restarting; these settings assume both GPUs are available.

The launch on 2026-11-03 used repository commit
`9c157a1f66a3e9a992f872b375b46fe99491ded9` plus the patch and source files
archived in the run's `-inputs` artifact. It was detached with PID 1128832.
W&B run: [se-families-hera-20261103](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/se-families-hera-20261103).

Inspect local status through the configured store:

```bash
.venv/bin/python - <<'PY'
from wyckoff_transformer.paths import runs_root
root = runs_root() / 'se-families-hera-20261103'
for name in ('driver.log', 'preflight.log', 'generate.log', 'bandgap.log',
             'select.log', 'relax.log', 'report.log'):
    path = root / name
    if path.exists():
        print(name, path.read_text()[-3000:])
PY
```

W&B's summary records the current stage and completion flags. Inputs and completed
stages are artifacts; logs are also W&B files. If a stage fails, inspect its log,
fix the cause, then re-run the same driver. Keep the source/configuration stable
while the driver runs. Do not remove the run directory to restart it.

## Recovery after the 2026-11-04 reboot

All 19 systems and all six protocol runs had completed before the reboot,
including scoring. The driver had stopped in reporting: when no candidates
needed exclusion, pandas returned a DataFrame from the empty row-wise `apply`,
which could not be assigned to `excluded_because`. Reporting now builds an
explicit Series, with regression tests for empty exclusions and multiple reasons.

Reporting was resumed on 2026-11-04 in the same W&B run, at commit
`9c157a1f66a3e9a992f872b375b46fe99491ded9` plus the newly archived working-tree
patch. Generation, selection and relaxation outputs were retained. The detached
driver PID for this attempt is 4739. To resume directly at a saved stage:

```bash
.venv/bin/python -u scripts/platforms/hera/run_se_families_followup.py --start-stage report
```

Use this only when the preceding stages' outputs are present; the default launch
still checks every stage. The run's `failed_stage` summary is cleared on success.

Reporting completed on 2026-11-04. A final W&B cleanup failure (`"failed_stage" in
run.summary` raised `KeyError: 0`) was fixed by using `run.summary.get`. The report
artifacts were already uploaded; the final tables and corrected summary were then
uploaded to the same run and it was marked complete. The completion fix is
archived in its `-completion-fix` artifact. The report contains 1,484 free-track
structures within 0.1 eV/atom, or 440 compositions (394 new to the LeMat reference).
