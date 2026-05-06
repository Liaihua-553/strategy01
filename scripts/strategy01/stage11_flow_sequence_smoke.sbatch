#!/bin/bash
#SBATCH -J s11_flowseq
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH -o /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage11_flow_sequence_%j.out
#SBATCH -e /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage11_flow_sequence_%j.err

set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase

/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python scripts/strategy01/stage11_flow_sequence_training.py \
  --device cuda \
  --run-name stage11_flow_sequence_v2_smoke \
  --batch-size 2 \
  --max-train-samples 16 \
  --max-val-samples 8 \
  --overfit1-steps 80 \
  --overfit4-steps 120 \
  --mini-steps 180 \
  --eval-every 40 \
  --ae-ca-source init \
  --noise-repeats 2 \
  --mini-trainable-phase seq_latent \
  --lambda-ae-seq 1.5 \
  --lambda-seq-ae-consistency 0.15
