#!/bin/bash
export WANDB_DIR=/tmp/wandb
mkdir -p $WANDB_DIR
source CRP/env_setup.sh
poetry run python train.py yamls/models/matbench_discovery_mp_2022/fat_multiplicity_weight_sg_energy_batch.yaml matbench_discovery_mp_2022 cuda --run-path /output --torch-num-thread ${ROLOS_AVAILABLE_CPU%.*}