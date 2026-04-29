#!/bin/bash
#SBATCH --job-name=stage07_vexact_expand
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_vexact_expand_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_vexact_expand_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
export PYTHONUNBUFFERED=1
echo "[stage07_vexact_expand] start $(date)"
$PY scripts/strategy01/stage06_mine_multistate_validation.py \
  --out-dir data/strategy01/stage07_validation_expand \
  --cache-dir /tmp/kfliao_stage07_vexact_expand_${SLURM_JOB_ID:-manual} \
  --exact-manifest data/strategy01/stage07_validation_expand/V_exact_manifest.json \
  --hybrid-seed-manifest data/strategy01/stage07_validation_expand/V_hybrid_seed_manifest.json \
  --summary reports/strategy01/probes/stage07_vexact_expand_summary.json \
  --max-entries 4000 \
  --target-exact-count 64 \
  --min-exact-count 32 \
  --target-hybrid-count 16 \
  --min-states 2 \
  --max-states 3
echo "[stage07_vexact_expand] done $(date)"