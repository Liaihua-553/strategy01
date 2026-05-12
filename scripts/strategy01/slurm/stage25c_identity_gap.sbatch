#!/usr/bin/env bash
#SBATCH -J stg25c_idgap
#SBATCH -p new
#SBATCH --nodelist=gu02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -o /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/slurm/stage25c_identity_gap_%j.out
#SBATCH -e /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/slurm/stage25c_identity_gap_%j.err
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
for CK in stage25a_trainable_interface_field stage25b_interface_shell; do
  $PY scripts/strategy01/stage17_per_sample_multiseed_probe.py \
    --device cuda \
    --checkpoint ckpts/stage12_de_novo_multistate/runs/${CK}/stage12_pilot_final.pt \
    --split val \
    --sample-offset 0 \
    --max-samples 4 \
    --seeds 1207,1211,1217,1223 \
    --nsteps 16 \
    --stage13-native-state-path \
    --stage22-enable-native-flow-coupling \
    --stage24-enable-target-interface-field \
    --stage24-enable-interface-guidance \
    --stage24-interface-guidance-scale 0.25 \
    --stage24-interface-guidance-max-shift-nm 0.08 \
    --stage18-enable-clash-relief \
    --stage18-relief-iters 8 \
    --report-json reports/strategy01/probes/stage25c_identity_gap_${CK}_val4.json
 done
$PY - <<'PY'
import json
for ck in ['stage25a_trainable_interface_field','stage25b_interface_shell']:
    p=f'reports/strategy01/probes/stage25c_identity_gap_{ck}_val4.json'
    d=json.load(open(p))
    print(ck, d['summary'])
PY
echo "END $(date)"