#!/bin/bash
scripts/platforms/iapetus/run.sh wyformer-protocol-wandb $1 --output-dir generated/$1/protocol --pyxtal-cores 6 --devices cuda:0,cuda:0,cuda:1,cuda:1,cuda:2 --skip-generate
