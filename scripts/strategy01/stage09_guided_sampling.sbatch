#!/usr/bin/env bash
#SBATCH -J stage09_guided_sampling
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o reports/strategy01/probes/stage09_guided_sampling_%j.out
#SBATCH -e reports/strategy01/probes/stage09_guided_sampling_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
MAX_SAMPLES=${MAX_SAMPLES:-8}
OUT_SUFFIX=${OUT_SUFFIX:-smoke}
$PY scripts/strategy01/stage09_guided_state_specific_sampling.py \
  --device cuda \
  --dataset data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt \
  --ckpt ckpts/stage07_sequence_consensus/runs/stage08b_merged_pilot/mini_final_lightning.ckpt \
  --out-dir results/strategy01/stage09_guided_state_sampling_${OUT_SUFFIX} \
  --report reports/strategy01/probes/stage09_guided_state_sampling_${OUT_SUFFIX}_summary.json \
  --max-samples ${MAX_SAMPLES} \
  --nsteps ${NSTEPS:-24} \
  --guidance-start-fraction ${GUIDANCE_START:-0.35} \
  --guidance-lr ${GUIDANCE_LR:-0.015} \
  --clash-weight ${CLASH_WEIGHT:-30.0} \
  --shell-weight ${SHELL_WEIGHT:-0.10} \
  --tether-weight ${TETHER_WEIGHT:-0.002}
$PY scripts/strategy01/stage08b_full_exact_benchmark.py \
  --b1-summary reports/strategy01/probes/stage09_guided_state_sampling_${OUT_SUFFIX}_summary.json \
  --output reports/strategy01/probes/stage09_guided_state_sampling_${OUT_SUFFIX}_exact_benchmark.json