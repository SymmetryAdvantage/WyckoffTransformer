#!/bin/bash
source CRP/env_setup.sh
mkdir -p $WANDB_DIR
uv run python train.py yamls/models/mp_20/formation_energy_per_site/not_augmented/harmonic_no_dropout.yaml mp_20 cuda --run-path /output --torch-num-thread ${ROLOS_AVAILABLE_CPU%.*}