#!/usr/bin/env bash
#SBATCH -J stage08b_vexact_sampling
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o reports/strategy01/probes/stage08b_vexact_sampling_%j.out
#SBATCH -e reports/strategy01/probes/stage08b_vexact_sampling_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
$PY scripts/strategy01/stage08b_multistate_exact_sampling.py \
  --device cuda \
  --dataset data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt \
  --ckpt ckpts/stage07_sequence_consensus/runs/stage08b_merged_pilot/mini_final_lightning.ckpt \
  --out-dir results/strategy01/stage08b_vexact_sampling \
  --report reports/strategy01/probes/stage08b_vexact_sampling_summary.json \
  --max-samples 48 \
  --nsteps 24
