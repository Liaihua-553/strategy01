#!/bin/bash
#SBATCH --job-name=stage07_v23_soft
#SBATCH --partition=new
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_data_prod_v23_soft_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_data_prod_v23_soft_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
export PYTHONUNBUFFERED=1
WORK_DIR=/tmp/kfliao_strategy01_stage07_v23_soft_${SLURM_JOB_ID:-manual}
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
trap 'echo "[stage07_v23_soft] keeping scratch work dir for debugging: $WORK_DIR"' EXIT
echo "[stage07_v23_soft] start $(date) work_dir=$WORK_DIR"
$PY scripts/strategy01/stage06_build_boltz_production_dataset.py \
  --seed-manifest data/strategy01/stage07_seed_pool/T_prod_seed_manifest_v22_ranked_top80.json \
  --out-dir data/strategy01/stage07_production_v23_soft_calib \
  --work-dir "$WORK_DIR" \
  --summary reports/strategy01/probes/stage07_data_production_v23_soft_summary.json \
  --max-entries 0 \
  --max-base-seeds 12 \
  --train-count 32 \
  --val-count 8 \
  --min-states 2 \
  --max-states 3 \
  --timeout-sec 1200 \
  --pass-strategy anchor_soft_first \
  --disable-msa-cache \
  --allow-short-dataset
echo "[stage07_v23_soft] done $(date)"
