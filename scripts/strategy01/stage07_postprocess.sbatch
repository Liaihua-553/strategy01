#!/bin/bash
#SBATCH --job-name=stage07_post
#SBATCH --partition=new
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_postprocess_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_postprocess_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
RAW=data/strategy01/stage07_production/stage06_predictor_pilot_samples.pt
AE=data/strategy01/stage07_production/stage06_predictor_pilot_samples_ae_latents.pt
$PY scripts/strategy01/stage05_extract_ae_latents.py \
  --dataset "$RAW" \
  --output "$AE" \
  --device cuda \
  --summary reports/strategy01/probes/stage07_ae_latent_extract_summary.json
$PY scripts/strategy01/stage07_sequence_consensus_training.py \
  --dataset "$AE" \
  --device cuda \
  --run-name stage07_prod_pilot \
  --batch-candidates 4,2,1 \
  --overfit1-steps 300 \
  --overfit4-steps 600 \
  --mini-steps 3000 \
  --eval-every 100
$PY scripts/strategy01/stage07_plot_loss_curves.py \
  --results reports/strategy01/probes/stage07_prod_pilot_results.json \
  --out-dir reports/strategy01/figures/stage07/prod_pilot