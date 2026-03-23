#!/bin/bash
export WANDB_DIR=/tmp/wandb
mkdir -p $WANDB_DIR
source CRP/env_setup.sh
uv run python wandb_agent.py --project WyckoffTransformer fdocos15 cuda --count 50 --run-path /output