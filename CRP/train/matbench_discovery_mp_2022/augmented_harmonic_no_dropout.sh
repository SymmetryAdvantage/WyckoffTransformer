#!/bin/bash
mkdir -p $WANDB_DIR
source CRP/env_setup.sh
poetry run python train.py yamls/models/matbench_discovery_mp_2022/augmented_harmonic_no_dropout.yaml matbench_discovery_mp_2022 cuda --run-path /output --torch-num-thread ${ROLOS_AVAILABLE_CPU%.*}