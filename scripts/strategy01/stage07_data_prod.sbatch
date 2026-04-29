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
export PYTHONUNBUFFERED=1
WORK_DIR=/tmp/kfliao_strategy01_stage07_${SLURM_JOB_ID:-manual}
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
trap 'echo "[stage07_data_prod] keeping scratch work dir for debugging: $WORK_DIR"' EXIT
echo "[stage07_data_prod] start $(date) work_dir=$WORK_DIR"
echo "[stage07_data_prod] seed_manifest=data/strategy01/stage07_seed_pool/T_prod_seed_manifest.json"
$PY scripts/strategy01/stage06_build_boltz_production_dataset.py \
  --seed-manifest data/strategy01/stage07_seed_pool/T_prod_seed_manifest.json \
  --out-dir data/strategy01/stage07_production \
  --work-dir "$WORK_DIR" \
  --summary reports/strategy01/probes/stage07_data_production_summary.json \
  --max-entries 0 \
  --max-base-seeds 512 \
  --train-count 512 \
  --val-count 128 \
  --min-states 2 \
  --max-states 3 \
  --timeout-sec 1200 \
  --allow-short-dataset
echo "[stage07_data_prod] done $(date)"