#!/usr/bin/env bash
#SBATCH -J stg24_samp
#SBATCH -p new
#SBATCH --nodelist=gu02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/slurm/stage24_sampling_from_probe_%j.out
#SBATCH -e /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/slurm/stage24_sampling_from_probe_%j.err
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
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PYTORCH_KERNEL_CACHE_PATH" "$REPO/reports/strategy01/probes/slurm"
cd "$REPO"
echo "START $(date) host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-none}"
$PY scripts/strategy01/stage17_per_sample_multiseed_probe.py \
  --device cuda \
  --checkpoint ckpts/stage12_de_novo_multistate/runs/stage24_target_interface_field_probe/stage12_pilot_final.pt \
  --split val \
  --sample-offset 0 \
  --max-samples 12 \
  --nsteps 16 \
  --stage13-native-state-path \
  --stage22-enable-native-flow-coupling \
  --stage24-enable-target-interface-field \
  --stage24-enable-interface-guidance \
  --stage24-interface-guidance-scale 0.5 \
  --stage24-interface-guidance-max-shift-nm 0.10 \
  --lambda-target-interface-site 0.20 \
  --lambda-target-interface-center 0.10 \
  --stage18-enable-clash-relief \
  --stage18-relief-iters 8 \
  --write-candidate-pdb-dir results/strategy01/stage24_target_interface_field_pdbs/val12 \
  --report-json reports/strategy01/probes/stage24_target_interface_field_val12_candidates.json
$PY scripts/strategy01/stage19_candidate_geometry_benchmark.py \
  --dataset data/strategy01/stage10_exactaug_training/stage10_exactaug_trainval.pt \
  --candidate-json reports/strategy01/probes/stage24_target_interface_field_val12_candidates.json \
  --split val \
  --output reports/strategy01/probes/stage24_target_interface_field_val12_geometry.json
$PY scripts/strategy01/stage20_selection_sweep_from_pdbs.py \
  --dataset data/strategy01/stage10_exactaug_training/stage10_exactaug_trainval.pt \
  --candidate-json reports/strategy01/probes/stage24_target_interface_field_val12_candidates.json \
  --split val \
  --output reports/strategy01/probes/stage24_target_interface_field_val12_selection_sweep.json
echo "END $(date)"
