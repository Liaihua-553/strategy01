#!/usr/bin/env bash
#SBATCH -J stg27_k1
#SBATCH -p new
#SBATCH --nodelist=gu02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/slurm/stage27_k1_curriculum_%j.out
#SBATCH -e /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/slurm/stage27_k1_curriculum_%j.err
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
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PYTORCH_KERNEL_CACHE_PATH" "$REPO/reports/strategy01/probes/slurm" "$REPO/results/strategy01/stage27_k1_curriculum_pdbs"
cd "$REPO"
echo "START $(date) host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-none}"
$PY scripts/strategy01/stage27_build_k1_curriculum_dataset.py
$PY scripts/strategy01/stage12_de_novo_multistate_training.py \
  --device cuda \
  --run-name stage27_k1_curriculum_native_warmup \
  --dataset data/strategy01/stage27_k1_curriculum/stage27_k1_trainval.pt \
  --init-ckpt ckpts/stage12_de_novo_multistate/runs/stage25a_trainable_interface_field/stage12_pilot_final.pt \
  --batch-size 4 \
  --max-train-samples 256 \
  --max-val-samples 64 \
  --overfit1-steps 60 \
  --overfit4-steps 100 \
  --pilot-steps 800 \
  --eval-every 100 \
  --stage13-native-state-path \
  --stage13-heads-only \
  --stage14-native-latent-warmup \
  --stage22-enable-native-flow-coupling \
  --stage24-enable-target-interface-field \
  --stage14-enable-bounded-latent-repair \
  --stage14-latent-repair-max-norm 0.25 \
  --lambda-struct 0.25 \
  --lambda-seq 0.75 \
  --lambda-state-seq 0.35 \
  --lambda-seq-consensus 0.15 \
  --lambda-anchor-disagreement 0.05 \
  --lambda-ae-seq 2.0 \
  --lambda-seq-ae-consistency 0.50 \
  --lambda-flow-gate-reg 0.02 \
  --lambda-target-interface-site 0.15 \
  --lambda-target-interface-center 0.03 \
  --target-interface-positive-weight 5.0 \
  --enable-teacher-self-conditioning \
  --teacher-self-cond-prob 1.0 \
  --enable-replay \
  --replay-mix-prob 0.45 \
  --replay-warmup-steps 40 \
  --replay-rollout-steps 8 \
  --replay-collect-fraction 0.60 \
  --replay-max-loss-t 0.75 \
  --replay-teacher-blend 0.55 \
  --enable-fixed-replay-cache \
  --replay-cache-batches 2 \
  --replay-cache-refresh-every 160 \
  --replay-cache-levels easy:0.65,medium:0.45,hard:0.25 \
  --replay-curriculum-switches 0.40,0.75 \
  --replay-safe-max-bb-ca-distance-nm 0.55 \
  --replay-safe-max-local-latents-distance 0.75 \
  --replay-safe-fallback-max-blend 0.90 \
  --replay-base-loss-scale 0.60 \
  --replay-extra-ae-seq-weight 1.0 \
  --replay-extra-seq-ae-consistency-weight 0.5 \
  --replay-extra-contact-weight 0.05 \
  --replay-extra-distance-weight 0.02 \
  --replay-extra-clash-weight 0.05
CKPT=ckpts/stage12_de_novo_multistate/runs/stage27_k1_curriculum_native_warmup/stage12_pilot_final.pt
$PY scripts/strategy01/stage17_per_sample_multiseed_probe.py \
  --device cuda \
  --dataset data/strategy01/stage27_k1_curriculum/stage27_k1_trainval.pt \
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
  --write-candidate-pdb-dir results/strategy01/stage27_k1_curriculum_pdbs/k1_val12 \
  --report-json reports/strategy01/probes/stage27_k1_curriculum_k1_val12_candidates.json
$PY scripts/strategy01/stage19_candidate_geometry_benchmark.py \
  --dataset data/strategy01/stage27_k1_curriculum/stage27_k1_trainval.pt \
  --candidate-json reports/strategy01/probes/stage27_k1_curriculum_k1_val12_candidates.json \
  --split val \
  --output reports/strategy01/probes/stage27_k1_curriculum_k1_val12_geometry.json
$PY scripts/strategy01/stage17_per_sample_multiseed_probe.py \
  --device cuda \
  --dataset data/strategy01/stage10_exactaug_training/stage10_exactaug_trainval.pt \
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
  --write-candidate-pdb-dir results/strategy01/stage27_k1_curriculum_pdbs/multistate_val12 \
  --report-json reports/strategy01/probes/stage27_k1_curriculum_multistate_val12_candidates.json
$PY scripts/strategy01/stage19_candidate_geometry_benchmark.py \
  --dataset data/strategy01/stage10_exactaug_training/stage10_exactaug_trainval.pt \
  --candidate-json reports/strategy01/probes/stage27_k1_curriculum_multistate_val12_candidates.json \
  --split val \
  --output reports/strategy01/probes/stage27_k1_curriculum_multistate_val12_geometry.json
echo "END $(date)"