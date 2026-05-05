#!/bin/bash
#SBATCH -J st10c_lat
#SBATCH -p new
#SBATCH -w gu02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -o /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage10c_pose_fixed_latent_flow_%j.out
#SBATCH -e /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage10c_pose_fixed_latent_flow_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
CKPT=ckpts/stage10_pose_init_exactaug/runs/stage10_pose_init_exactaug_full_v2/mini_final_lightning.ckpt
AE=ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt
DATA=data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt
COMMON=(--ckpt "$CKPT" --autoencoder-ckpt "$AE" --dataset "$DATA" --split val --max-samples 0 --nsteps 24 --device cuda --seed 123 --anchor-source source_state0 --enable-init-pose --guidance-start-fraction 0.85 --guidance-inner-steps 10 --guidance-lr 0.2 --max-anchor-pairs 32 --clash-weight 30.0 --clash-loss-mode mean --shell-weight 0.10 --tether-weight 0.002 --overcontact-weight 0.02 --contact-temperature-nm 0.06 --max-guidance-displacement-nm 0.25 --safe-accept --bb-ca-flow-step-scale 0.0 --local-latents-flow-step-scale 1.0 --max-flow-displacement-nm 0.0)
$PY scripts/strategy01/stage09_guided_state_specific_sampling.py "${COMMON[@]}" --out-dir results/strategy01/stage10c_pose_fixed_latent_flow/no_ca --report reports/strategy01/probes/stage10c_pose_fixed_latent_flow_summary.json
$PY scripts/strategy01/stage08b_full_exact_benchmark.py --v-exact-dataset "$DATA" --b1-summary reports/strategy01/probes/stage10c_pose_fixed_latent_flow_summary.json --output reports/strategy01/probes/stage10c_pose_fixed_latent_flow_exact_benchmark.json
$PY scripts/strategy01/stage09_guided_state_specific_sampling.py "${COMMON[@]}" --enable-ca-feature --out-dir results/strategy01/stage10c_pose_fixed_latent_flow/ca --report reports/strategy01/probes/stage10c_pose_fixed_latent_flow_ca_summary.json
$PY scripts/strategy01/stage08b_full_exact_benchmark.py --v-exact-dataset "$DATA" --b1-summary reports/strategy01/probes/stage10c_pose_fixed_latent_flow_ca_summary.json --output reports/strategy01/probes/stage10c_pose_fixed_latent_flow_ca_exact_benchmark.json
