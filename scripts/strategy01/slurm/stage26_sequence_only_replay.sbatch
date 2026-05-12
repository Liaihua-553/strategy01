#!/usr/bin/env bash
#SBATCH -J stg26_seqonly
#SBATCH -p new
#SBATCH --nodelist=gu02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/slurm/stage26_sequence_only_replay_%j.out
#SBATCH -e /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/slurm/stage26_sequence_only_replay_%j.err
set -euo pipefail
REPO=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
export STRATEGY01_RUNTIME_ROOT=/data/kfliao/general_model/strategy_repos/.runtime/strategy01
export TMPDIR=$STRATEGY01_RUNTIME_ROOT/tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
export XDG_CACHE_HOME=$STRATEGY01_RUNTIME_ROOT/cache
export TORCH_HOME=$STRATEGY01_RUNTIME_ROOT/torch
export PYTORCH_KERNEL_CACHE_PATH=$STRATEGY01_RUNTIME_ROOT/torch_extensions
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PYTORCH_KERNEL_CACHE_PATH" "$REPO/reports/strategy01/probes/slurm" "$REPO/reports/strategy01/probes" "$REPO/results/strategy01/stage26_sequence_only_replay_pdbs"
cd "$REPO"
echo "START $(date) host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-none}"
$PY scripts/strategy01/stage12_de_novo_multistate_training.py \
  --device cuda \
  --run-name stage26_sequence_only_replay \
  --init-ckpt ckpts/stage12_de_novo_multistate/runs/stage25a_trainable_interface_field/stage12_pilot_final.pt \
  --batch-size 4 \
  --max-train-samples 64 \
  --max-val-samples 16 \
  --overfit1-steps 40 \
  --overfit4-steps 80 \
  --pilot-steps 480 \
  --eval-every 80 \
  --stage26-sequence-only-trainable \
  --stage13-native-state-path \
  --stage22-enable-native-flow-coupling \
  --stage24-enable-target-interface-field \
  --stage14-enable-bounded-latent-repair \
  --lambda-struct 0.0 \
  --lambda-seq 1.0 \
  --lambda-state-seq 0.75 \
  --lambda-seq-consensus 0.30 \
  --lambda-anchor-disagreement 0.10 \
  --lambda-ae-seq 2.0 \
  --lambda-seq-ae-consistency 0.50 \
  --lambda-flow-gate-reg 0.0 \
  --lambda-target-interface-site 0.0 \
  --lambda-target-interface-center 0.0 \
  --enable-teacher-self-conditioning \
  --teacher-self-cond-prob 1.0 \
  --enable-replay \
  --replay-mix-prob 0.50 \
  --replay-warmup-steps 30 \
  --replay-rollout-steps 8 \
  --replay-collect-fraction 0.60 \
  --replay-max-loss-t 0.75 \
  --replay-teacher-blend 0.65 \
  --enable-fixed-replay-cache \
  --replay-cache-batches 2 \
  --replay-cache-refresh-every 120 \
  --replay-cache-levels easy:0.65,medium:0.55 \
  --replay-curriculum-switches 0.60 \
  --replay-safe-max-bb-ca-distance-nm 0.55 \
  --replay-safe-max-local-latents-distance 0.75 \
  --replay-safe-fallback-max-blend 0.90 \
  --replay-base-loss-scale 0.50 \
  --replay-extra-ae-seq-weight 1.0 \
  --replay-extra-seq-ae-consistency-weight 0.5 \
  --replay-extra-contact-weight 0.0 \
  --replay-extra-distance-weight 0.0 \
  --replay-extra-clash-weight 0.0
CKPT=ckpts/stage12_de_novo_multistate/runs/stage26_sequence_only_replay/stage12_pilot_final.pt
$PY scripts/strategy01/stage17_per_sample_multiseed_probe.py \
  --device cuda \
  --checkpoint "$CKPT" \
  --split val \
  --sample-offset 0 \
  --max-samples 4 \
  --nsteps 16 \
  --stage13-native-state-path \
  --stage22-enable-native-flow-coupling \
  --stage24-enable-target-interface-field \
  --stage18-enable-clash-relief \
  --stage18-relief-iters 8 \
  --write-candidate-pdb-dir results/strategy01/stage26_sequence_only_replay_pdbs/val4 \
  --report-json reports/strategy01/probes/stage26_sequence_only_replay_val4_candidates.json
$PY scripts/strategy01/stage17_per_sample_multiseed_probe.py \
  --device cuda \
  --checkpoint "$CKPT" \
  --split val \
  --sample-offset 0 \
  --max-samples 12 \
  --nsteps 16 \
  --stage13-native-state-path \
  --stage22-enable-native-flow-coupling \
  --stage24-enable-target-interface-field \
  --stage18-enable-clash-relief \
  --stage18-relief-iters 8 \
  --write-candidate-pdb-dir results/strategy01/stage26_sequence_only_replay_pdbs/val12 \
  --report-json reports/strategy01/probes/stage26_sequence_only_replay_val12_candidates.json
$PY scripts/strategy01/stage19_candidate_geometry_benchmark.py \
  --dataset data/strategy01/stage10_exactaug_training/stage10_exactaug_trainval.pt \
  --candidate-json reports/strategy01/probes/stage26_sequence_only_replay_val12_candidates.json \
  --split val \
  --output reports/strategy01/probes/stage26_sequence_only_replay_val12_geometry.json
$PY scripts/strategy01/stage20_selection_sweep_from_pdbs.py \
  --dataset data/strategy01/stage10_exactaug_training/stage10_exactaug_trainval.pt \
  --candidate-json reports/strategy01/probes/stage26_sequence_only_replay_val12_candidates.json \
  --split val \
  --output reports/strategy01/probes/stage26_sequence_only_replay_val12_selection_sweep.json
echo "END $(date)"