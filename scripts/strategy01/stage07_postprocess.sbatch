#!/bin/bash
#SBATCH --job-name=stage07_post
#SBATCH --partition=new
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_postprocess_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_postprocess_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
export PYTHONUNBUFFERED=1
DATA_DIR=${STAGE07_DATA_DIR:-data/strategy01/stage07_production_v23_wave1}
RUN_NAME=${STAGE07_RUN_NAME:-stage07_v23_wave1_pilot}
RAW=${STAGE07_RAW_DATASET:-$DATA_DIR/stage06_predictor_pilot_samples_materialized.pt}
AE=${STAGE07_AE_DATASET:-$DATA_DIR/stage06_predictor_pilot_samples_materialized_ae_latents.pt}
echo "[stage07_post] start $(date) raw=$RAW"
if [ ! -s "$RAW" ]; then
  echo "[stage07_post] missing raw dataset: $RAW" >&2
  exit 2
fi
if [ -s "$AE" ]; then
  echo "[stage07_post] step=ae_latents skip_existing=$AE"
else
  echo "[stage07_post] step=ae_latents"
  $PY scripts/strategy01/stage05_extract_ae_latents.py \
    --dataset "$RAW" \
    --output "$AE" \
    --device cuda \
    --summary reports/strategy01/probes/${RUN_NAME}_ae_latent_extract_summary.json
fi
echo "[stage07_post] step=train"
$PY scripts/strategy01/stage07_sequence_consensus_training.py \
  --dataset "$AE" \
  --device cuda \
  --run-name "$RUN_NAME" \
  --batch-candidates 4,2,1 \
  --overfit1-steps 300 \
  --overfit4-steps 600 \
  --mini-steps 3000 \
  --eval-every 100
echo "[stage07_post] step=plot"
$PY scripts/strategy01/stage07_plot_loss_curves.py \
  --results reports/strategy01/probes/${RUN_NAME}_results.json \
  --out-dir reports/strategy01/figures/stage07/${RUN_NAME}
echo "[stage07_post] done $(date)"