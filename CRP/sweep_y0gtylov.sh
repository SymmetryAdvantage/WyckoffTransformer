#!/bin/bash
export WANDB_DIR=/tmp/wandb
mkdir -p $WANDB_DIR
source CRP/env_setup.sh
poetry run python wandb_agent.py --project WyckoffTransformer y0gtylov cuda --count 50 --run-path /output