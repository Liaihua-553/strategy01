#!/bin/bash
#SBATCH --job-name=stage07_seed_mine
#SBATCH --partition=new
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_seed_mine_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_seed_mine_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
export PYTHONUNBUFFERED=1
CACHE_DIR=/tmp/kfliao_strategy01_stage07_seed_cache_${SLURM_JOB_ID:-manual}
mkdir -p "$CACHE_DIR"
echo "[stage07_seed_mine] start $(date) cache=$CACHE_DIR"
$PY scripts/strategy01/stage07_mine_train_seeds.py \
  --cache-dir "$CACHE_DIR" \
  --out-dir data/strategy01/stage07_seed_pool \
  --output-manifest data/strategy01/stage07_seed_pool/T_prod_seed_manifest.json \
  --summary reports/strategy01/probes/stage07_seed_mining_summary.json \
  --max-entries 3000 \
  --target-seeds 256 \
  --min-states 2 \
  --max-states 3 \
  --progress-every 20 \
  --shuffle
echo "[stage07_seed_mine] done $(date)"