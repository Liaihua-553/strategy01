#!/usr/bin/env bash
#SBATCH -J stage09_guided_sweep2
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o reports/strategy01/probes/stage09_guided_sweep2_%j.out
#SBATCH -e reports/strategy01/probes/stage09_guided_sweep2_%j.err
set -euo pipefail
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
run_one () {
  local suffix=$1; shift
  $PY scripts/strategy01/stage09_guided_state_specific_sampling.py \
    --device cuda --seed 123 --max-samples 8 --nsteps 24 \
    --dataset data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt \
    --ckpt ckpts/stage07_sequence_consensus/runs/stage08b_merged_pilot/mini_final_lightning.ckpt \
    --out-dir results/strategy01/stage09_guided_state_sampling_${suffix} \
    --report reports/strategy01/probes/stage09_guided_state_sampling_${suffix}_summary.json "$@"
  $PY scripts/strategy01/stage08b_full_exact_benchmark.py \
    --b1-summary reports/strategy01/probes/stage09_guided_state_sampling_${suffix}_summary.json \
    --output reports/strategy01/probes/stage09_guided_state_sampling_${suffix}_exact_benchmark.json
}
run_one guided_seed123_s075_lr015_i8_a32 --guidance-start-fraction 0.75 --guidance-lr 0.15 --guidance-inner-steps 8 --max-anchor-pairs 32 --clash-weight 60 --shell-weight 0.10 --tether-weight 0.002
run_one guided_seed123_s085_lr020_i10_a32 --guidance-start-fraction 0.85 --guidance-lr 0.20 --guidance-inner-steps 10 --max-anchor-pairs 32 --clash-weight 80 --shell-weight 0.10 --tether-weight 0.002
run_one guided_seed123_s070_lr012_i8_a64 --guidance-start-fraction 0.70 --guidance-lr 0.12 --guidance-inner-steps 8 --max-anchor-pairs 64 --clash-weight 60 --shell-weight 0.10 --tether-weight 0.002
$PY - <<'PY'
import json
from pathlib import Path
rows=[]
for p in sorted(Path('reports/strategy01/probes').glob('stage09_guided_state_sampling_*seed123*_exact_benchmark.json')):
    d=json.loads(p.read_text())
    b=d['B1_strategy01']
    rows.append({'file':p.name,'contact_mean':b['contact_f1']['mean'],'rmsd_mean':b['worst_interface_rmsd_A']['mean'],'clash_rate':b['clash_rate'],'state_count':b['state_count_ok']})
print(json.dumps(rows,indent=2))
Path('reports/strategy01/probes/stage09_guided_sampling_sweep2_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
PY