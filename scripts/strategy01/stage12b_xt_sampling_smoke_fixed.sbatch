#!/bin/bash
#SBATCH --job-name=stage12b_samp
#SBATCH --partition=new
#SBATCH --nodelist=gu02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --output=reports/strategy01/probes/stage12b_xt_sampling_%j.out
#SBATCH --error=reports/strategy01/probes/stage12b_xt_sampling_%j.err
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
CKPT=ckpts/stage12_de_novo_multistate/runs/stage12b_xt_denovo_pilot/stage12_pilot_final.pt
DATA=data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt
$PY scripts/strategy01/stage09_guided_state_specific_sampling.py \
  --ckpt "$CKPT" \
  --dataset "$DATA" \
  --out-dir results/strategy01/stage12b_xt_denovo_sampling_smoke \
  --report reports/strategy01/probes/stage12b_xt_denovo_sampling_smoke_summary.json \
  --split any \
  --max-samples 8 \
  --nsteps 24 \
  --device cuda \
  --sequence-source shared_head \
  --guidance-start-fraction 2.0 \
  --de-novo-multistate-mode \
  --pass-x-t-states
$PY scripts/strategy01/stage08b_full_exact_benchmark.py \
  --v-exact-dataset "$DATA" \
  --b1-summary reports/strategy01/probes/stage12b_xt_denovo_sampling_smoke_summary.json \
  --output reports/strategy01/probes/stage12b_xt_denovo_sampling_smoke_exact_benchmark.json
