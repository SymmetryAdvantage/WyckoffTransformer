#!/usr/bin/env bash
# Build the WyFormer-GeoCSP v2.3 Singularity image from committed sources only.
#
# Stages: WyFormer at WYFORMER_REF (git archive, no local state), GeoCSP (the
# DiffCSPNew repository) at GEOCSP_REF, the exact dependency pins of both uv.lock
# files, the four pinned WyFormer runs, the GeoCSP weights and the alex-mp-20 gene-key
# table. Writes $OUT_DIR/WyFormer-GeoCSP-$IMAGE_TAG.sif and runs its %test.
#
#   scripts/alex_bench/container/build.sh
#   singularity push $OUT_DIR/WyFormer-GeoCSP-v2.3.sif \
#       oras://ghcr.io/symmetryadvantage/wyckofftransformer:wyformer-geocsp-v2.3
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WYFORMER_REF="${WYFORMER_REF:-HEAD}"
GEOCSP_REPO="${GEOCSP_REPO:-/home/kna/DiffCSPNew}"
GEOCSP_REF="${GEOCSP_REF:-1d8b164a47db39f9c86f5e9e6d47b430f6ff1a0f}"
GEOCSP_CKPT="${GEOCSP_CKPT:-$GEOCSP_REPO/runs/alex_mp20_geov2/geov2_alex_mp20_150e.pt}"
STORE="${STORE:-/home/kna/.local/share/wyformer}"
IMAGE_TAG="${IMAGE_TAG:-v2.3}"
OUT_DIR="${OUT_DIR:-$STORE/containers}"
RUNS=(uncond_adamw_wsd_5x-20260929-143845 ehull_adamw_wsd_5x-20260929-143848
      ehull_adamw_wsd_5x_cfg-20260929-150414 gene_min_ehull_adamw_wsd-20260929-154926)

if [[ -n "$(git -C "$REPO" status --porcelain -uno)" && "$WYFORMER_REF" == HEAD ]]; then
    echo "The WyFormer checkout has uncommitted changes; the image is built from commits only." >&2
    exit 1
fi
wyformer_revision=$(git -C "$REPO" rev-parse "$WYFORMER_REF")
geocsp_revision=$(git -C "$GEOCSP_REPO" rev-parse "$GEOCSP_REF")

stage="$OUT_DIR/stage-$IMAGE_TAG"
rm -rf "$stage"
mkdir -p "$stage/wyformer" "$stage/geocsp" "$stage/okhotin/runs" "$stage/okhotin/refs" \
    "$stage/okhotin/geocsp" "$OUT_DIR/cache" "$OUT_DIR/tmp"

git -C "$REPO" archive --format=tar "$wyformer_revision" -- \
    pyproject.toml uv.lock README.md LICENSE src scripts yamls docs/archive/okhotin_submission.md |
    tar -x -C "$stage/wyformer"
git -C "$GEOCSP_REPO" archive --format=tar "$geocsp_revision" -- \
    pyproject.toml uv.lock README.md diffcsp bench |
    tar -x -C "$stage/geocsp"

# The exact versions of the study's environments, without torch (the base image's) and
# spglib (zeus builds its own; the image takes the same version from PyPI).
for pair in "wyformer:$REPO" "geocsp:$GEOCSP_REPO"; do
    name=${pair%%:*}; source=${pair#*:}
    (cd "$stage/$name" && uv export --frozen --no-dev --no-hashes --no-emit-project \
        --no-emit-package torch --no-emit-package triton --no-emit-package spglib \
        --no-header > "$stage/$name-requirements.txt")
done

for run in "${RUNS[@]}"; do
    cp -r "$STORE/runs/$run" "$stage/okhotin/runs/"
done
cp "$STORE/alex_bench/refs/alex_mp_20_labelled_train+val_keys.npz" "$stage/okhotin/refs/"
cp "$GEOCSP_CKPT" "$stage/okhotin/geocsp/"
cp "$(command -v uv)" "$stage/uv"
cp "$HERE/WyFormer-GeoCSP.def" "$stage/"

sif="$OUT_DIR/WyFormer-GeoCSP-$IMAGE_TAG.sif"
(
    cd "$stage"
    SINGULARITY_CACHEDIR="$OUT_DIR/cache" SINGULARITY_TMPDIR="$OUT_DIR/tmp" \
        singularity build --fakeroot --force \
            --build-arg "SOURCE_REVISION=$wyformer_revision" \
            --build-arg "GEOCSP_REVISION=$geocsp_revision" \
            --build-arg "IMAGE_VERSION=$IMAGE_TAG" \
            "$sif" WyFormer-GeoCSP.def
)
singularity test "$sif"
echo "built $sif (wyformer $wyformer_revision, geocsp $geocsp_revision)"
