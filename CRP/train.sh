#!/bin/bash
export WANDB_DIR=/tmp/wandb
mkdir -p $WANDB_DIR
source CRP/env_setup.sh
poetry run python train.py ${MODEL_CONFIG} mp_20 cuda --run-path /output --torch-num-thread ${ROLOS_AVAILABLE_CPU%.*}