#!/bin/bash
# Resume TACE, then finish ORB and Prophet, while GPU 1 runs recovery arms.
set -euo pipefail
export MLIP_BIAS_GPU=0
exec scripts/platforms/zeus/run_mlip_bias_migration_gpu1.sh tace_oam_l orb_conserv_inf prophet_oame_mbd
