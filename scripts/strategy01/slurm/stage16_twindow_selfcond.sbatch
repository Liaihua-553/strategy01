#!/bin/bash
#SBATCH -J st16_tw_train
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/slurm/stage16_twindow_selfcond_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/slurm/stage16_twindow_selfcond_%j.err
set -euo pipefail
ROOT=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
export STRATEGY01_RUNTIME_ROOT=/data/kfliao/general_model/strategy_repos/.runtime/strategy01
export TMPDIR=$STRATEGY01_RUNTIME_ROOT/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export XDG_CACHE_HOME=$STRATEGY01_RUNTIME_ROOT/xdg_cache
export TORCH_HOME=$STRATEGY01_RUNTIME_ROOT/torch_home
export PYTORCH_KERNEL_CACHE_PATH=$STRATEGY01_RUNTIME_ROOT/torch_kernel_cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PYTORCH_KERNEL_CACHE_PATH"
cd "$ROOT"
$PY scripts/strategy01/stage12_de_novo_multistate_training.py \
  --device cuda \
  --run-name stage16_twindow_selfcond_from_stage15 \
  --init-ckpt ckpts/stage12_de_novo_multistate/runs/stage15_teacher_selfcond_from_warmup/stage12_pilot_final.pt \
  --stage13-native-state-path \
  --stage13-heads-only \
  --stage14-native-latent-warmup \
  --enable-stage16-t-window \
  --stage16-t-min 0.55 \
  --stage16-t-max 0.95 \
  --stage16-t-window-prob 1.0 \
  --enable-teacher-self-conditioning \
  --teacher-self-cond-prob 1.0 \
  --lambda-ae-seq 4.0 \
  --lambda-seq-ae-consistency 0.75 \
  --max-train-samples 161 \
  --max-val-samples 36 \
  --batch-size 4 \
  --overfit1-steps 60 \
  --overfit4-steps 100 \
  --pilot-steps 600 \
  --eval-every 120
CKPT=ckpts/stage12_de_novo_multistate/runs/stage16_twindow_selfcond_from_stage15/stage12_pilot_final.pt
$PY scripts/strategy01/stage13_single_state_de_novo_audit.py \
  --device cuda \
  --checkpoint "$CKPT" \
  --sampling-config configs/pipeline/model_sampling.yaml \
  --native-state-path \
  --split val \
  --states all \
  --max-source-samples 4 \
  --nsteps 16 \
  --report-json reports/strategy01/probes/stage16_single_state_audit_twindow_selfcond_val4_n16.json
$PY scripts/strategy01/stage13_single_state_de_novo_audit.py \
  --device cuda \
  --checkpoint "$CKPT" \
  --sampling-config configs/pipeline/model_sampling.yaml \
  --native-state-path \
  --split val \
  --states all \
  --max-source-samples 4 \
  --nsteps 32 \
  --report-json reports/strategy01/probes/stage16_single_state_audit_twindow_selfcond_val4_n32.json
