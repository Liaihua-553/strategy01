#!/bin/bash
#SBATCH --job-name=stage12b_denovo
#SBATCH --partition=new
#SBATCH --nodelist=gu02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --output=reports/strategy01/probes/stage12b_denovo_%j.out
#SBATCH --error=reports/strategy01/probes/stage12b_denovo_%j.err

set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase

/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python \
  scripts/strategy01/stage12_de_novo_multistate_training.py \
  --device cuda \
  --run-name stage12b_denovo_pilot \
  --batch-size 4 \
  --max-train-samples 161 \
  --max-val-samples 36 \
  --overfit1-steps 80 \
  --overfit4-steps 120 \
  --pilot-steps 200 \
  --eval-every 40
