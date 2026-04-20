#!/bin/bash
#SBATCH -p new
#SBATCH --job-name=s05_boltz_tpl
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage05_boltz_template_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage05_boltz_template_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
mkdir -p reports/strategy01/probes
hostname
nvidia-smi || true
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python scripts/strategy01/stage05_build_predictor_multistate_dataset.py \
  --run-boltz \
  --max-samples 1 \
  --train-count 1 \
  --val-count 0 \
  --states-per-sample 2 \
  --out-dir data/strategy01/stage05_predictor_multistate_smoke_template \
  --boltz-recycling-steps 1 \
  --boltz-sampling-steps 5 \
  --boltz-diffusion-samples 1 \
  --timeout-sec 0 \
  --allow-short-dataset
