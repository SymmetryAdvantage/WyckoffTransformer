#!/bin/bash
export WANDB_DIR=/tmp/wandb
mkdir -p $WANDB_DIR
source CRP/env_setup.sh
uv run python train.py yamls/models/NextToken/v6/pilot.yaml mp_20 cpu --run-path /output --torch-num-thread ${ROLOS_AVAILABLE_CPU%.*}