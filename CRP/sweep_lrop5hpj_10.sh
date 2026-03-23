#!/bin/bash
export WANDB_DIR=/tmp/wandb
mkdir -p $WANDB_DIR
source CRP/env_setup.sh
uv run python wandb_agent.py --project WyckoffTransformer lrop5hpj cuda --count 10 --run-path /output