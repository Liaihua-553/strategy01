#!/bin/bash
#SBATCH --job-name=stage07_mat
#SBATCH --partition=new
#SBATCH --nodelist=gu02
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_materialize_v23_wave1_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_materialize_v23_wave1_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
export PYTHONUNBUFFERED=1
echo "[stage07_materialize] start $(date) node=$(hostname)"
$PY scripts/strategy01/stage07_materialize_predictions.py \
  --data-dir data/strategy01/stage07_production_v23_wave1 \
  --summary reports/strategy01/probes/stage07_v23_wave1_materialize_summary.json
echo "[stage07_materialize] done $(date)"
