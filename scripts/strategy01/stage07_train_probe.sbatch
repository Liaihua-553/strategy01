#!/bin/bash
#SBATCH --job-name=stage07_train_probe
#SBATCH --partition=new
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_train_probe_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_train_probe_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
$PY scripts/strategy01/stage07_sequence_consensus_training.py \
  --device cuda \
  --run-name stage07_gpu_probe_pilot8 \
  --batch-candidates 4,2,1 \
  --overfit1-steps 80 \
  --overfit4-steps 120 \
  --mini-steps 200 \
  --eval-every 40
$PY scripts/strategy01/stage07_plot_loss_curves.py \
  --results reports/strategy01/probes/stage07_gpu_probe_pilot8_results.json \
  --out-dir reports/strategy01/figures/stage07/gpu_probe_pilot8