#!/usr/bin/env bash
#SBATCH -J stage09e_smoke
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o reports/strategy01/probes/stage09e_smoke_%j.out
#SBATCH -e reports/strategy01/probes/stage09e_smoke_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
$PY scripts/strategy01/stage09e_multiseed_candidate_select.py \
  --device cuda --seed 123 --max-samples 8 --num-candidates 8 --nsteps 24 \
  --dataset data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt \
  --ckpt ckpts/stage07_sequence_consensus/runs/stage08b_merged_pilot/mini_final_lightning.ckpt \
  --out-dir results/strategy01/stage09e_multiseed_smoke \
  --report reports/strategy01/probes/stage09e_multiseed_smoke_summary.json
$PY scripts/strategy01/stage08b_full_exact_benchmark.py \
  --b1-summary reports/strategy01/probes/stage09e_multiseed_smoke_summary.json \
  --output reports/strategy01/probes/stage09e_multiseed_smoke_exact_benchmark.json
