#!/bin/bash
#SBATCH --job-name=stage07_v23_salv
#SBATCH --partition=new
#SBATCH --nodelist=gu02
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_v23_salvage_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_v23_salvage_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
echo HOST=$(hostname)
echo DATE=$(date)
du -sh /tmp/kfliao_strategy01_stage07_v23_soft_1908320 2>/dev/null || true
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python scripts/strategy01/stage07_salvage_partial.py \
  --scratch /tmp/kfliao_strategy01_stage07_v23_soft_1908320 \
  --out-dir data/strategy01/stage07_partial_v23_soft \
  --summary reports/strategy01/probes/stage07_v23_soft_salvage_summary.json \
  --max-examples 8
