# Sourced, not run: puts the pinned Singularity on PATH, sets the default image and,
# when the node's driver is too old for that image's CUDA, points the container at the
# image's CUDA forward-compatibility driver.
#
#   source scripts/platforms/aspire2a/singularity_env.sh
#
# Every ASPIRE 2A script that runs singularity sources this instead of loading the
# module itself. `module load singularity` alone gives 3.10.0, and if any singularity
# is already loaded, loading another version fails with a conflict yet returns 0. So
# this unloads whatever is there first and checks the version it ends up with.
#
# SIF and WYFORMER_SINGULARITY_MODULE keep values the caller has set.

WYFORMER_SINGULARITY_MODULE=${WYFORMER_SINGULARITY_MODULE:-singularity/4.3.1}
SIF=${SIF:-/home/project/11001786/WyFormer/images/pytorch_2.14.1-cuda13.2-cudnn9-devel.sif}

# A batch shell may not have the module system initialised, and the nodes have no
# /etc/profile.d/modules.sh: ASPIRE 2A's is Cray's, which /etc/profile.d/zz-cray-pe.sh
# sets up together with the site's /app/apps/modulefiles. Its init script and the
# `module` function both read unset variables, so `set -u` is off around them.
case $- in *u*) _wyformer_nounset=1; set +u ;; *) _wyformer_nounset= ;; esac
if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/zz-cray-pe.sh /etc/profile.d/modules.sh; do
        [ -f "$init" ] && { source "$init"; break; }
    done
fi
module unload singularity apptainer >/dev/null 2>&1 || true
module load "$WYFORMER_SINGULARITY_MODULE" || true
[ -n "$_wyformer_nounset" ] && set -u
unset _wyformer_nounset
if ! singularity --version 2>/dev/null | grep -qF "${WYFORMER_SINGULARITY_MODULE#*/}"; then
    echo "error: module load $WYFORMER_SINGULARITY_MODULE did not put it on PATH" \
         "(got: $(singularity --version 2>/dev/null || echo 'no singularity'))" >&2
    return 1 2>/dev/null || exit 1
fi

# CUDA 13 needs driver >= 580; ASPIRE 2A's A100 nodes run 570 (CUDA 12.8). NVIDIA's
# forward-compatibility libcuda, which the image ships in /usr/local/cuda/compat, runs
# CUDA 13.2 on 570 on data-centre GPUs. It must come before the host driver that --nv
# binds, and must not be used on a driver newer than itself, hence only below 580.
# Singularity appends --nv's /.singularity.d/libs after this LD_LIBRARY_PATH.
WYFORMER_HOST_DRIVER=$(sed -nE 's/.*Kernel Module( for [^ ]+)? +([0-9]+)\..*/\2/p' \
                           /proc/driver/nvidia/version 2>/dev/null | head -1 || true)
if [ -n "$WYFORMER_HOST_DRIVER" ] && [ "$WYFORMER_HOST_DRIVER" -lt 580 ]; then
    case ":${SINGULARITYENV_LD_LIBRARY_PATH:-}:" in
        *:/usr/local/cuda/compat:*) ;;
        *) export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/cuda/compat${SINGULARITYENV_LD_LIBRARY_PATH:+:$SINGULARITYENV_LD_LIBRARY_PATH} ;;
    esac
fi
