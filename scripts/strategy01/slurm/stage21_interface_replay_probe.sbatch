#!/bin/bash
#SBATCH -J st21_iface
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/slurm/stage21_interface_replay_probe_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/slurm/stage21_interface_replay_probe_%j.err
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
RUN=stage21_interface_replay_probe
$PY scripts/strategy01/stage12_de_novo_multistate_training.py \
  --device cuda \
  --run-name "$RUN" \
  --init-ckpt ckpts/stage12_de_novo_multistate/runs/stage15_teacher_selfcond_from_warmup/stage12_pilot_final.pt \
  --batch-size 2 \
  --max-train-samples 64 \
  --max-val-samples 16 \
  --overfit1-steps 40 \
  --overfit4-steps 60 \
  --pilot-steps 160 \
  --eval-every 40 \
  --stage13-native-state-path \
  --stage14-enable-bounded-latent-repair \
  --stage14-latent-repair-max-norm 0.25 \
  --enable-teacher-self-conditioning \
  --teacher-self-cond-prob 1.0 \
  --enable-replay \
  --replay-mix-prob 0.50 \
  --replay-warmup-steps 20 \
  --replay-rollout-steps 8 \
  --replay-collect-fraction 0.60 \
  --replay-max-loss-t 0.75 \
  --replay-teacher-blend 0.35 \
  --enable-fixed-replay-cache \
  --replay-cache-batches 2 \
  --replay-cache-refresh-every 80 \
  --replay-base-loss-scale 0.50 \
  --replay-extra-ae-seq-weight 0.50 \
  --replay-extra-seq-ae-consistency-weight 0.20 \
  --replay-extra-contact-weight 1.00 \
  --replay-extra-distance-weight 0.50 \
  --replay-extra-clash-weight 1.00
rm -rf results/strategy01/stage21_interface_replay_probe_pdbs/val12
$PY scripts/strategy01/stage17_per_sample_multiseed_probe.py \
  --device cuda \
  --checkpoint ckpts/stage12_de_novo_multistate/runs/$RUN/stage12_pilot_final.pt \
  --sampling-config configs/pipeline/model_sampling.yaml \
  --split val \
  --max-samples 12 \
  --nsteps 16 \
  --stage13-native-state-path \
  --stage18-enable-clash-relief \
  --stage18-relief-iters 8 \
  --seeds 1207,1211,1217,1223,1231,1237,1249,1259 \
  --write-candidate-pdb-dir results/strategy01/stage21_interface_replay_probe_pdbs/val12 \
  --report-json reports/strategy01/probes/stage21_interface_replay_probe_val12_candidates.json
$PY scripts/strategy01/stage19_candidate_geometry_benchmark.py \
  --candidate-json reports/strategy01/probes/stage21_interface_replay_probe_val12_candidates.json \
  --output reports/strategy01/probes/stage21_interface_replay_probe_val12_geometry.json
