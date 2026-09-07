#!/bin/bash
# Build the PyTorch CUDA container used for WyFormer on luna.
#
# Why cuda12.6 and not the cuda13.x tags of the same PyTorch release:
#   luna's NVIDIA driver is 550.163.01, which is a CUDA 12.4 driver. CUDA 12.x
#   minor-version compatibility lets a 12.6 build run on it; CUDA 13.x needs a
#   >=580 driver and fails at runtime. Check `nvidia-smi` before bumping this.
#
# Why the `devel` variant and not `runtime`:
#   it ships nvcc, headers and a compiler, so dependencies without a matching
#   wheel can still build. Disk is not scarce on /home.
#
# Why a sandbox directory and not a .sif:
#   apptainer 1.5.1 bundles mksquashfs 4.7.5, whose orderer crashes on an image
#   this large:
#       FATAL ERROR: Bug in orderer
#   The system mksquashfs is 4.6.1 and would probably work, but apptainer takes
#   its mksquashfs path from /etc/apptainer/apptainer.conf, which needs root to
#   change. A sandbox skips squashfs entirely. `apptainer exec` treats a sandbox
#   directory and a .sif the same way, so nothing downstream cares.
#
#   To retry a .sif once the image exists (the 4.7.5 bug is concurrency
#   related, so single-threaded may get through):
#       apptainer build --mksquashfs-args="-processors 1" \
#           "$WYFORMER_CONTAINER.sif" "$WYFORMER_CONTAINER"
#   then point WYFORMER_CONTAINER at the .sif.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

mkdir -p "$(dirname "$WYFORMER_CONTAINER")" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "Building $WYFORMER_IMAGE_TAG -> $WYFORMER_CONTAINER (sandbox)"
apptainer build --force --sandbox "$WYFORMER_CONTAINER" "docker://$WYFORMER_IMAGE_TAG"

echo
echo "==> Checking the container can see the GPUs"
# GPU 2 is excluded: it has uncorrectable ECC errors and a pending row remap.
apptainer exec --nv "$WYFORMER_CONTAINER" python -c "
import torch
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())
print('devices', torch.cuda.device_count())
"
