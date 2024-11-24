#!/bin/bash
export OMP_NUM_THREADS=${ROLOS_AVAILABLE_CPU%.*}
export MKL_NUM_THREADS=${ROLOS_AVAILABLE_CPU%.*}
export WANDB_DIR=/tmp/wandb