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

Two test modules error at collection in this venv:

```
ImportError: cannot import name '_pt_data' from 'pymatgen.core.periodic_table'
ImportError: cannot import name 'sph_harm' from 'scipy.special'
```

Both come from `matminer`, which reaches into private pymatgen internals and
into a SciPy function removed in SciPy 1.17. Nothing in WyFormer's own code is
involved, and neither the de novo ranking protocol nor training touches
`matminer`. Deselect the two modules until `matminer` is updated:

```bash
run python -m pytest src/ \
    --ignore=src/wyckoff_transformer/tests/test_trainer_cache.py \
    --ignore=src/wyckoff_transformer/formula_energy/tests/test_experiment.py
```
