#!/bin/bash
#SBATCH -J st15_sc_train
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/slurm/stage15_teacher_selfcond_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/slurm/stage15_teacher_selfcond_%j.err
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
  --run-name stage15_teacher_selfcond_from_warmup \
  --init-ckpt ckpts/stage12_de_novo_multistate/runs/stage14_native_latent_warmup_161/stage12_pilot_final.pt \
  --stage13-native-state-path \
  --stage13-heads-only \
  --stage14-native-latent-warmup \
  --enable-teacher-self-conditioning \
  --teacher-self-cond-prob 1.0 \
  --lambda-ae-seq 3.0 \
  --lambda-seq-ae-consistency 0.50 \
  --max-train-samples 161 \
  --max-val-samples 36 \
  --batch-size 4 \
  --overfit1-steps 80 \
  --overfit4-steps 120 \
  --pilot-steps 800 \
  --eval-every 160
CKPT=ckpts/stage12_de_novo_multistate/runs/stage15_teacher_selfcond_from_warmup/stage12_pilot_final.pt
$PY scripts/strategy01/stage13_single_state_de_novo_audit.py \
  --device cuda \
  --checkpoint "$CKPT" \
  --sampling-config configs/pipeline/model_sampling.yaml \
  --native-state-path \
  --split val \
  --states all \
  --max-source-samples 4 \
  --nsteps 16 \
  --report-json reports/strategy01/probes/stage15_single_state_audit_teacher_selfcond_val4_n16.json
$PY scripts/strategy01/stage13_single_state_de_novo_audit.py \
  --device cuda \
  --checkpoint "$CKPT" \
  --sampling-config configs/pipeline/model_sampling.yaml \
  --native-state-path \
  --split val \
  --states all \
  --max-source-samples 4 \
  --nsteps 32 \
  --report-json reports/strategy01/probes/stage15_single_state_audit_teacher_selfcond_val4_n32.json
