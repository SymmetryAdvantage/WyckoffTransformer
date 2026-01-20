#!/bin/bash

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Error: This script must be executed, not sourced."
    echo "Usage: ./$(basename "${BASH_SOURCE[0]}") <input_dir> <output_dir>"
    return 1
fi

set -e


INPUT_DATA_DIR=$1
OUTPUT_DIR=$2

if [ -z "$INPUT_DATA_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Error: Missing arguments."
  echo "Usage: $0 <input_data_dir> <output_dir>"
  exit 1
fi

if [ ! -d "$INPUT_DATA_DIR" ]; then
  echo "Error: Input directory $INPUT_DATA_DIR does not exist."
  exit 1
fi
if [ -d "$OUTPUT_DIR" ]; then
  echo "Error: Output directory $OUTPUT_DIR already exists. Aborting to prevent overwrite."
  exit 1
else
  echo "Creating output directory: $OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
fi

ABS_INPUT=$(realpath "$INPUT_DATA_DIR")
ABS_OUTPUT=$(realpath "$OUTPUT_DIR")

docker run --rm \
  --name wyckoff-transformer-inference \
  -v "$ABS_INPUT":/app/model_data:ro \
  -v "$ABS_OUTPUT":/app/outputs \
  wyckoff-transformer:latest \
  python /opt/wyformer_app/WyckoffTransformer/scripts/generate.py \
    --model-path /app/model_data \
    /app/outputs/generated.json.gz \
    --use-cached-tensors
