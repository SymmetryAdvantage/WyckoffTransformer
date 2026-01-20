#!/bin/bash

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Error: This script must be executed, not sourced."
    echo "Usage: ./$(basename "${BASH_SOURCE[0]}") <output_dir>"
    return 1
fi

set -e

OUTPUT_DIR=$1

if [ -z "$OUTPUT_DIR" ]; then
  echo "Error: Missing arguments."
  echo "Usage: $0 <output_dir>"
  exit 1
fi

if [ -d "$OUTPUT_DIR" ]; then
  echo "Error: Output directory $OUTPUT_DIR already exists. Aborting to prevent overwrite."
  exit 1
else
  echo "Creating output directory: $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
fi

ABS_OUTPUT=$(realpath "$OUTPUT_DIR")


docker run --rm \
  --name wyckoff-transformer-training \
  -v "$ABS_OUTPUT":/home/appuser/workdir/run \
  --gpus all \
  wyckoff-transformer:latest \
  /bin/bash -c "cd /home/appuser/workdir/run && wandb offline && python \
    /opt/wyformer_app/WyckoffTransformer/scripts/train.py \
    /opt/wyformer_app/WyckoffTransformer/yamls/models/NextToken/v6/base_sg.yaml \
    mp_20 cuda --run-path ./runs"
