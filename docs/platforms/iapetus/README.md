# WyFormer on iapetus

`iapetus` is running Ubuntu 26.04.1 LTS (kernel
7.0.0-31). It has an Intel Core i7-5930K (6 physical cores, 12 threads), 30 GiB
of RAM, and no scheduler.

Its NVIDIA GPUs are legacy hardware on driver **470.256.02**:

| `nvidia-smi` index | Model | Memory |
| --- | --- | --- |
| 0 | Tesla K20c | 4743 MiB |
| 1 | Tesla K20c | 4743 MiB |
| 2 | GeForce GTX 750 Ti | 1999 MiB |

WyFormer runs in the custom Docker image maintained at
`/home/kna/pytorch-research/`, with a project uv venv layered over the image's
PyTorch. The image supplies custom PyTorch 2.14.0.post2 for CUDA 11.8 and
`sm_35`/`sm_50`; this is what makes the legacy GPUs usable despite the lack of a
host CUDA toolkit. See [environment.md](environment.md).

## Which page do you want?

| I want to... | Read |
| --- | --- |
| create or repair the container venv | [environment.md](environment.md) |
| run training, generation, or tests | [usage.md](usage.md) |
| diagnose an environment or GPU problem | [troubleshooting.md](troubleshooting.md) |

## Current host facts

| Item | Value |
| --- | --- |
| Project checkout | `/home/kna/WyckoffTransformer` |
| Container image | `pytorch:2.14.0-cuda11.8-py312-universal` |
| Container Python | 3.12.14 |
| Container torch | 2.14.0.post2, CUDA 11.8, custom `sm_35` + `sm_50` build |
| Host Python / uv | 3.14.4 / 0.12.9; not used for WyFormer |
| Docker | 29.8.0 |
| Root filesystem | 233 GiB total, about 55 GiB free when recorded |

These figures were recorded on 2026-09-08. This is a shared interactive host:
check GPU and disk capacity before large runs and keep bulky results outside the
repository.
