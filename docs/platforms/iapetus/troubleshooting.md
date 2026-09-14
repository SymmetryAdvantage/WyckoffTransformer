# Troubleshooting WyFormer on iapetus

| Symptom | Cause and resolution |
| --- | --- |
| `ModuleNotFoundError: No module named 'torch'` from `.venv/bin/python` | The venv was created without `--system-site-packages`, or it was created on the host. Rebuild it through the container command in [environment.md](environment.md). |
| `torch.__file__` points into `.venv` | uv installed a second torch instead of inheriting the image's custom build. Rebuild with `scripts/build_singularity_venv.sh`; do not run `uv sync` or bare `uv pip install` in the venv. |
| `torch.cuda.is_available()` is false | Ensure the command uses Docker's `--runtime=nvidia` and `NVIDIA_VISIBLE_DEVICES=all`, then check `nvidia-smi` inside the same container. An empty `CUDA_VISIBLE_DEVICES` deliberately hides GPUs. |
| `CUDA driver version is insufficient` or `no kernel image is available` | A stock PyPI CUDA torch has replaced the image build. Rebuild the venv; do not install torch from PyPI. |
| `IndexError: list index out of range` in `warp/_src/torch.py` while ORB uses CUDA | The CPU-only Warp override has no CUDA devices. Keep Warp on CPU for graph construction, then transfer the graph to the CUDA ORB model with `build_patched_orb_calculator(device="cuda")` from `scripts/run_cryspr_reconstruction_study.py`. |
| `Warp requires CUDA driver 12.0 or higher` | The inherited Warp 1.17 CUDA wheel cannot run on iapetus's 470 driver. Restore the CPU-only Warp wheel from [environment.md](environment.md); this does not prevent GPU ORB forward evaluation. |
| CUDA out of memory | The cards have 2--5 GiB of memory. Reduce batch size/model size, choose CPU, or use a better-equipped platform. |
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
