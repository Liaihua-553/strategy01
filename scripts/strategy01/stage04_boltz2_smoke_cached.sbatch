#!/bin/bash
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=stage04_boltz2_smoke_cached
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/logs/stage04_boltz2_smoke_cached_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/logs/stage04_boltz2_smoke_cached_%j.err

set -euo pipefail
REPO=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
BOLTZ=/data/kfliao/general_model/envs/boltz_cb04aec/bin/boltz
CACHE=/data/kfliao/general_model/boltz_cache
cd "$REPO"
mkdir -p logs reports/strategy01/probes/boltz_smoke_outputs_cached
hostname
nvidia-smi || true
"$BOLTZ" predict reports/strategy01/probes/boltz_smoke_inputs/stage04_tiny_ppi.yaml \
  --out_dir reports/strategy01/probes/boltz_smoke_outputs_cached \
  --cache "$CACHE" \
  --devices 1 \
  --accelerator gpu \
  --model boltz2 \
  --recycling_steps 1 \
  --sampling_steps 5 \
  --diffusion_samples 1 \
  --num_workers 0 \
  --preprocessing-threads 1 \
  --output_format pdb \
  --no_kernels \
  --override
