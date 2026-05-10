#!/bin/bash
#SBATCH -J st17_geo2ps
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/slurm/stage17_geo2_persample_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/slurm/stage17_geo2_persample_%j.err
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
$PY scripts/strategy01/stage17_per_sample_multiseed_probe.py \
  --device cuda \
  --checkpoint ckpts/stage12_de_novo_multistate/runs/stage15_teacher_selfcond_from_warmup/stage12_pilot_final.pt \
  --sampling-config configs/pipeline/model_sampling.yaml \
  --split val \
  --max-samples 12 \
  --nsteps 16 \
  --stage13-native-state-path \
  --seeds 1207,1211,1217,1223,1231,1237,1249,1259 \
  --report-json reports/strategy01/probes/stage17_per_sample_multiseed_geo2_stage15_val12_n16.json
