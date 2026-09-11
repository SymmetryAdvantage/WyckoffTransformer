#!/bin/bash
alias run=scripts/platforms/iapetus/run.sh
run wyformer-protocol-wandb upi73i4k --output-dir generated/upi73i4k/protocol --pyxtal-cores 6 --devices cuda:0,cuda:0,cuda:1,cuda:1,cuda:2 &> upi73i4k.log
run wyformer-protocol-wandb e9ywwsie --output-dir generated/e9ywwsie/protocol --pyxtal-cores 6 --devices cuda:0,cuda:0,cuda:1,cuda:1,cuda:2 &> e9ywwsie.log
run wyformer-protocol-wandb 19qbxo6l --output-dir generated/19qbxo6l/protocol --pyxtal-cores 6 --devices cuda:0,cuda:0,cuda:1,cuda:1,cuda:2 &> 19qbxo6l.log
run wyformer-protocol-wandb e_all_adamw_wsd-20260909-001225 --output-dir generated/e_all_adamw_wsd-20260909-001225/protocol --pyxtal-cores 6 --devices cuda:0,cuda:0,cuda:1,cuda:1,cuda:2 &> e_all_adamw_wsd-20260909-001225.log
run wyformer-protocol-wandb ehull-ssops-20260904-235534 --output-dir generated/ehull-ssops-20260904-235534/protocol --pyxtal-cores 6 --devices cuda:0,cuda:0,cuda:1,cuda:1,cuda:2 &> ehull-ssops-20260904-235534.log
