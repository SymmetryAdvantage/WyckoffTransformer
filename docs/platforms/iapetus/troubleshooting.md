# Troubleshooting WyFormer on iapetus

| Symptom | Cause and resolution |
| --- | --- |
| `ModuleNotFoundError: No module named 'torch'` from `.venv/bin/python` | The venv was created without `--system-site-packages`, or it was created on the host. Rebuild it with `scripts/platforms/iapetus/build_venv.sh`; see [environment.md](environment.md). |
| `torch.__file__` points into `.venv` | uv installed a second torch instead of inheriting the image's custom build. Rebuild with `scripts/platforms/iapetus/build_venv.sh`; do not run `uv sync` or bare `uv pip install` in the venv. |
| `torch.cuda.is_available()` is false | Ensure the command uses Docker's `--runtime=nvidia` and `NVIDIA_VISIBLE_DEVICES=all`, then check `nvidia-smi` inside the same container. An empty `CUDA_VISIBLE_DEVICES` deliberately hides GPUs. |
| `CUDA driver version is insufficient` or `no kernel image is available` | A stock PyPI CUDA torch has replaced the image build. Rebuild the venv; do not install torch from PyPI. |
| `CUDA initialization: CUDA unknown error` / `RuntimeError: CUDA unknown error` | After a host reboot, `/dev/nvidia-uvm` is not created until `nvidia-modprobe` runs. Without it, the NVIDIA container runtime cannot mount UVM into the container, breaking CUDA initialization. `scripts/platforms/iapetus/run.sh` runs `nvidia-modprobe -c 0 -u` automatically when `/dev/nvidia-uvm` is missing, or run `nvidia-modprobe -c 0 -u` manually on the host. |
| `IndexError: list index out of range` in `warp/_src/torch.py` while ORB uses CUDA | The CPU-only Warp override has no CUDA devices. Keep Warp on CPU for graph construction, then transfer the graph to the CUDA ORB model with `build_patched_orb_calculator(device="cuda")` from `scripts/run_cryspr_reconstruction_study.py`. |
| `Warp requires CUDA driver 12.0 or higher` | The inherited Warp 1.17 CUDA wheel cannot run on iapetus's 470 driver. Restore the CPU-only Warp wheel from [environment.md](environment.md); this does not prevent GPU ORB forward evaluation. |
| A worker on one card never becomes ready, and that card takes no trials | That card's CUDA context creation is hanging. Seen on **GPU 1** on 2026-09-10: `CUDA_VISIBLE_DEVICES=1` wedges before `torch.cuda.is_available()` returns, with 5 MiB allocated and 0% utilisation, while GPU 0 runs the same check in seconds. It is not ORB-specific -- a bare `torch` matmul hangs identically -- and not ECC: both K20c cards report the same two retired double-bit pages. Check each card with the health checks below before a long run, and drop the failing one from `--devices`. The container must be `docker kill`ed; the hung process ignores the client timeout. |
| CUDA out of memory | The cards have 2--5 GiB of memory. Reduce batch size/model size, choose CPU, or use a better-equipped platform. |
| `ModuleNotFoundError: No module named 'torch'` from a venv that has `container-base.pth` missing | The image keeps torch in `/opt/venv312`, which `--system-site-packages` does not reach; the build adds it with `container-base.pth`. A venv built by calling `build_singularity_venv.sh` without `BASE_PYTHON=/opt/venv312/bin/python` lacks it. Rebuild with `build_venv.sh`. |
| `ModuleNotFoundError` for a project dependency inside `run.sh bash -c '...'` | Only `run.sh`'s first word is resolved against the venv; inside `bash -c` a bare `python` is the image's. Call `/workspace/.venv/bin/python`. |
| `run.sh`: `error: ... exists, so it is authoritative, but does not set <KEY>` | `~/.config/wyformer/paths.env` lacks a key. Add it with the value from [environment.md](environment.md#data-store-cache-and-runs). |
| `fatal: not a git repository` inside the container, in a worktree | The worktree's `.git` names the main checkout's `.git` by host path, and that directory is not mounted. `run.sh` mounts it; a hand-written `docker run` must add `-v /home/kna/WyckoffTransformer/.git:/home/kna/WyckoffTransformer/.git:ro`. |
| `nvcc: command not found` on the host | Expected. The CUDA toolkit is part of the image; build and run extensions from inside the container. |
| `ModuleNotFoundError` for a pure-Python dep (`sklearn`, `omegaconf`, …) that is installed in `.venv`, and `sys.executable` is `/opt/venv312/bin/python` | `.venv/bin/` is gone, so `run.sh` cannot resolve `python` against the venv and falls through to the image interpreter. Almost always caused by running `uv` or `uv run` **on the host**: it sees the venv's interpreter symlink as broken, deletes `.venv/bin/`, then aborts on a permission error in `.venv/lib` (container-owned files), leaving `.venv` half-destroyed. Never run `uv*` on the host. `lib/site-packages` (deps, the editable `.pth`, and `container-base.pth` which wires in the image's torch) survives, so recreate only `bin/` inside the container, no rebuild: `run.sh bash -lc '/opt/uv-python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -m venv --system-site-packages --without-pip /workspace/.venv'` — `venv` without `--clear` re-uses the directory and leaves `lib/` untouched. Then run the health check above. |

## Health checks

Run this through the container after creating the venv:

```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --ipc=host \
    -v "$PWD:/workspace" -w /workspace \
    pytorch:2.14.0-cuda11.8-py312-universal \
    .venv/bin/python -c "
import torch, wyckoff_transformer
print(torch.__version__, torch.version.cuda)
print(torch.__file__)
print(torch.cuda.is_available(), torch.cuda.device_count())
"
```

Python must be 3.12.x and `torch.__file__` must be under
`/opt/venv312/lib/python3.12/site-packages`, not `.venv`.

## ORB CPU-neighbour/GPU-forward check

The following confirms the supported split execution: Warp builds the
periodic neighbour graph on CPU, and the ORB model evaluates it on GPU. It
uses the patched calculator already used by the CRySPR reconstruction study.

```bash
docker run --rm --entrypoint /bin/bash --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=0 --ipc=host \
    -v "$PWD:/workspace" -w /workspace \
    pytorch:2.14.0-cuda11.8-py312-universal -lc '
.venv/bin/python -c "
from ase.build import bulk
from scripts.run_cryspr_reconstruction_study import build_patched_orb_calculator
atoms = bulk(\"Cu\", \"fcc\", a=3.6)
atoms.calc = build_patched_orb_calculator(device=\"cuda\")
print(atoms.get_potential_energy())
"
'
```

The command must print an energy. The selected card will appear as `cuda:0`
inside the container, even when its host index is different.

## `matminer` fails to import (`_pt_data`, `sph_harm`)

```
ImportError: cannot import name 'sph_harm' from 'scipy.special'
ImportError: cannot import name '_pt_data' from 'pymatgen.core.periodic_table'
```

**Fixed** by the `matminer >=0.10.1` floor in `pyproject.toml`. A venv built
before that floor still carries `matminer` 0.8.0, which loses
`tests/test_trainer_cache.py` at collection and the five Magpie-baseline tests
in `formula_energy/tests/test_experiment.py` at setup. Move the two packages in
without a full rebuild, as in
[environment.md](environment.md#moving-a-single-pinned-dependency-into-the-venv):

```bash
scripts/platforms/iapetus/run.sh bash -lc '
    uv pip compile --no-annotate --no-header pyproject.toml -o /tmp/full.txt
    grep -E "^(matminer|pandas|pytz)==" /tmp/full.txt > /tmp/mm.txt
    uv pip install --python .venv/bin/python --no-deps -r /tmp/mm.txt'
```

`pytz` is in that grep on purpose. `matminer` 0.10.1 requires `pandas<3`, so
this downgrades pandas 3.0.5 to 2.3.3 — and pandas 2 imports `pytz`, which
pandas 3 dropped. `--no-deps` will not bring it in, so without the third name
every module that touches pandas dies at collection with `Unable to import
required dependencies: pytz`.

Afterwards the whole suite runs with nothing deselected:

```bash
run python -m pytest src/
```
