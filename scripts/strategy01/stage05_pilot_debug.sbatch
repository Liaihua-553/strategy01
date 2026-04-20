#!/bin/bash
#SBATCH -p new
#SBATCH --job-name=s05_pilot_dbg
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage05_pilot_debug_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage05_pilot_debug_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
mkdir -p reports/strategy01/probes
hostname
nvidia-smi || true
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python scripts/strategy01/stage05_predictor_pilot_debug.py \
  --dataset data/strategy01/stage05_predictor_multistate_smoke_template/stage05_predictor_pilot_samples_ae_latents.pt \
  --device cuda \
  --run-name stage05_predictor_smoke_template_ae \
  --overfit1-steps 300 \
  --mini-steps 100 \
  --eval-every 100 \
  --mini-batch-size 1
