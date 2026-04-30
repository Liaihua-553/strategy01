#!/usr/bin/env bash
#SBATCH -J stage09_guided_pair
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o reports/strategy01/probes/stage09_guided_pair_%j.out
#SBATCH -e reports/strategy01/probes/stage09_guided_pair_%j.err
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
run_one control_seed123 --guidance-start-fraction 2.0
run_one guided_seed123_s065_lr005_i3 --guidance-start-fraction 0.65 --guidance-lr 0.05 --guidance-inner-steps 3 --clash-weight 30 --shell-weight 0.10 --tether-weight 0.002
run_one guided_seed123_s075_lr008_i5 --guidance-start-fraction 0.75 --guidance-lr 0.08 --guidance-inner-steps 5 --clash-weight 40 --shell-weight 0.10 --tether-weight 0.002
$PY - <<'PY'
import json
from pathlib import Path
rows=[]
for p in sorted(Path('reports/strategy01/probes').glob('stage09_guided_state_sampling_*seed123*_exact_benchmark.json')):
    d=json.loads(p.read_text())
    b=d['B1_strategy01']
    rows.append({'file':p.name,'contact_mean':b['contact_f1']['mean'],'rmsd_mean':b['worst_interface_rmsd_A']['mean'],'clash_rate':b['clash_rate'],'state_count':b['state_count_ok']})
print(json.dumps(rows,indent=2))
Path('reports/strategy01/probes/stage09_guided_sampling_pair_sweep_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
PY