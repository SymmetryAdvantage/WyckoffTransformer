# The iapetus environment

WyFormer runs inside
`pytorch:2.14.0-cuda11.8-py312-universal`, the custom image from
`/home/kna/pytorch-research/`. It contains Python 3.12, custom PyTorch
2.14.0.post2, CUDA 11.8, and kernels for both the K20c (`sm_35`) and GTX 750 Ti
(`sm_50`). It is matched to the host's NVIDIA 470.256.02 driver.

The project venv is created **inside that container**, at the bind-mounted
repository path. It inherits the container site-packages so torch remains the
image's custom build. Never create or activate this venv on the host: the host
default Python is 3.14 and the venv's interpreter and torch libraries belong to
the image.

## Creating or repairing the venv

`scripts/build_singularity_venv.sh` implements this workflow despite its
historic filename; it works when invoked from the Docker image as well. It
creates `.venv` with `--system-site-packages`, resolves all non-torch
dependencies, and explicitly excludes torch and CUDA distributions so uv
cannot replace the container's torch.

From the repository root:

```bash
docker run --rm -it \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    --ipc=host \
    -v "$PWD:/workspace" \
    -w /workspace \
    pytorch:2.14.0-cuda11.8-py312-universal \
    bash -lc 'REPO_DIR=/workspace UV="$(command -v uv)" bash scripts/build_singularity_venv.sh'
```

The build script compiles a fully pinned requirements file, removes torch and
its CUDA stack, then installs those requirements and the editable project with
`--no-deps`. `uv sync` is intentionally not used: it would install the lock's
ordinary torch selection over the custom container build.

### Triton exclusion

Do not install Triton in the iapetus-specific venv. Its K20c GPUs are compute
capability 3.5, while Triton's GPU compiler requires capability 7.0 or newer.
If Triton was added while testing, remove it from the venv:

```bash
docker run --rm --runtime=nvidia --ipc=host \
    -v "$PWD:/workspace" -w /workspace \
    pytorch:2.14.0-cuda11.8-py312-universal \
    /usr/local/bin/uv pip uninstall --python .venv/bin/python triton
```

### Moving a single pinned dependency into the venv

The venv is built from a compiled requirements file, so re-running
`build_singularity_venv.sh` to pick up one changed pin costs a full rebuild and
throws away the CPU-only Warp wheel below. For a single package — the pinned
PyXtal fork in `[tool.uv.sources]` is the standing example — compile just the
direct requirements and install that one line, so `pyproject.toml` stays the
only place the pin is written:

```bash
scripts/platforms/iapetus/run.sh bash -lc '
    uv pip compile --no-deps --no-annotate --no-header pyproject.toml \
        | grep "^pyxtal " > /tmp/pyxtal.txt
    uv pip install --python .venv/bin/python --no-deps -r /tmp/pyxtal.txt'
```

`--no-deps` on both commands is what keeps this safe: without it the resolver
reconsiders the whole closure and puts a PyPI torch over the container's custom
build. Note that a bare `uv pip install pyxtal` does **not** work here —
`tool.uv.sources` applies only to a requirement that comes from the project's
own dependency list, so the source has to reach uv through `pyproject.toml`.

### CPU-only Warp with GPU ORB inference

The base image currently carries the stock CUDA-enabled `warp-lang` wheel,
which requires a newer driver than iapetus has. Install the CPU-only
`warp-lang` wheel built for this host into `.venv`; it takes precedence over
the inherited package. This is intentional: Warp constructs ORB's neighbor
list on the CPU, while the ORB model and its forward evaluation still run on
the selected GPU:

```bash
docker run --rm --runtime=nvidia --ipc=host \
    -v "$PWD:/workspace" \
    -v /home/kna/pytorch-research:/opt/pytorch-research:ro \
    -w /workspace \
    pytorch:2.14.0-cuda11.8-py312-universal \
    /usr/local/bin/uv pip install --python .venv/bin/python --no-deps \
    --force-reinstall \
    /opt/pytorch-research/archive/dist-warp/warp_lang-1.17.0-py3-none-manylinux_2_28_x86_64.whl
```

The stock `orb-models` calculator passes its CUDA device to Warp and therefore
does not work with this CPU-only wheel. Use the CPU-neighbour/GPU-forward
calculator path in `scripts/run_cryspr_reconstruction_study.py`:
`build_patched_orb_calculator(device="cuda")`. It constructs the graph with
`device="cpu"` and transfers the resulting batch to the CUDA model. Do not
replace the CPU-only Warp wheel with the inherited CUDA wheel: Warp 1.17
requires a CUDA 12 driver, whereas iapetus is limited to driver 470 / CUDA
11.4.

The container entrypoint detects the owner of `/workspace` and drops to that
UID/GID, so files written to the mounted checkout remain host-owned.

## Do not use host-local torch setup

Do not copy `uv.toml.cpu` to `uv.toml`, install a PyPI CUDA torch wheel, or use
`uv.toml.local`. The first two bypass the custom Kepler/Maxwell-compatible
torch; the last points at zeus's private wheel index.
