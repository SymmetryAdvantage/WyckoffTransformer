#!/bin/bash
mkdir -p $WANDB_DIR
source CRP/env_setup.sh
poetry run python train.py yamls/models/mp_20/NextToken/cluster_harmonic.yaml mp_20 cuda --run-path /output --torch-num-thread ${ROLOS_AVAILABLE_CPU%.*}