#!/bin/bash
#SBATCH -J st10_b1sm
#SBATCH -p new
#SBATCH -w gu02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -o /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage10_b1_new_smoke_%j.out
#SBATCH -e /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage10_b1_new_smoke_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python scripts/strategy01/stage09_guided_state_specific_sampling.py \
  --ckpt ckpts/stage10_pose_init_exactaug/runs/stage10_pose_init_exactaug_full_v2/mini_final_lightning.ckpt \
  --autoencoder-ckpt ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt \
  --dataset data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt \
  --out-dir results/strategy01/stage10_b1_new_smoke \
  --report reports/strategy01/probes/stage10_b1_new_smoke_summary.json \
  --split val \
  --max-samples 8 \
  --nsteps 24 \
  --device cuda \
  --seed 123 \
  --anchor-source source_state0 \
  --enable-init-pose \
  --guidance-start-fraction 0.85 \
  --guidance-inner-steps 10 \
  --guidance-lr 0.2 \
  --max-anchor-pairs 32 \
  --clash-weight 30.0 \
  --clash-loss-mode mean \
  --shell-weight 0.10 \
  --tether-weight 0.002 \
  --overcontact-weight 0.02 \
  --contact-temperature-nm 0.06 \
  --max-guidance-displacement-nm 0.25 \
  --safe-accept
