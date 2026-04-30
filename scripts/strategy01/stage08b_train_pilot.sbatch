#!/bin/bash
#SBATCH --job-name=stage08b_train_pilot
#SBATCH --partition=new
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage08b_train_pilot_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage08b_train_pilot_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python

$PY scripts/strategy01/stage07_sequence_consensus_training.py \
  --dataset data/strategy01/stage08b_merged_training/stage08b_merged_pilot_samples.pt \
  --device cuda \
  --run-name stage08b_merged_pilot \
  --batch-candidates 4,2,1 \
  --overfit1-steps 300 \
  --overfit4-steps 600 \
  --mini-steps 1500 \
  --eval-every 100

$PY scripts/strategy01/stage07_plot_loss_curves.py \
  --results reports/strategy01/probes/stage08b_merged_pilot_results.json \
  --out-dir reports/strategy01/figures/stage08b/merged_pilot
