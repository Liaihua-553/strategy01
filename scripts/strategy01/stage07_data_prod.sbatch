#!/bin/bash
#SBATCH --job-name=stage07_data_prod
#SBATCH --partition=new
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_data_prod_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_data_prod_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
$PY scripts/strategy01/stage06_build_boltz_production_dataset.py \
  --out-dir data/strategy01/stage07_production \
  --summary reports/strategy01/probes/stage07_data_production_summary.json \
  --max-entries 4000 \
  --max-base-seeds 512 \
  --train-count 512 \
  --val-count 128 \
  --min-states 2 \
  --max-states 3 \
  --timeout-sec 1200 \
  --allow-short-dataset