#!/bin/bash
mkdir -p $WANDB_DIR
source CRP/env_setup.sh
uv run python train.py yamls/models/mp_20/letter_vs_ss/energy_letter_dropout.yaml mp_20 cuda --run-path /output --torch-num-thread ${ROLOS_AVAILABLE_CPU%.*}