#!/usr/bin/env bash
#SBATCH -J stage09_guided_full
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o reports/strategy01/probes/stage09_guided_full_%j.out
#SBATCH -e reports/strategy01/probes/stage09_guided_full_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
$PY scripts/strategy01/stage09_guided_state_specific_sampling.py \
  --device cuda --seed 123 --max-samples 48 --nsteps 24 \
  --dataset data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt \
  --ckpt ckpts/stage07_sequence_consensus/runs/stage08b_merged_pilot/mini_final_lightning.ckpt \
  --out-dir results/strategy01/stage09_guided_state_sampling_full_best \
  --report reports/strategy01/probes/stage09_guided_state_sampling_full_best_summary.json \
  --guidance-start-fraction 0.85 --guidance-lr 0.20 --guidance-inner-steps 10 \
  --max-anchor-pairs 32 --clash-weight 80 --shell-weight 0.10 --tether-weight 0.002
$PY scripts/strategy01/stage08b_full_exact_benchmark.py \
  --b1-summary reports/strategy01/probes/stage09_guided_state_sampling_full_best_summary.json \
  --output reports/strategy01/probes/stage09_guided_state_sampling_full_best_exact_benchmark.json