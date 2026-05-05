#!/bin/bash
#SBATCH -J stage10_full
#SBATCH -p new
#SBATCH -w gu02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -o /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage10_pose_init_full_%j.out
#SBATCH -e /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage10_pose_init_full_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python scripts/strategy01/stage10_pose_init_training.py \
  --device cuda \
  --run-name stage10_pose_init_exactaug_full \
  --batch-candidates 4,2,1 \
  --overfit1-steps 300 \
  --overfit4-steps 600 \
  --mini-steps 1500 \
  --eval-every 100
