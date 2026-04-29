#!/bin/bash
#SBATCH --job-name=stage07_refold
#SBATCH --partition=new
#SBATCH --nodelist=gu02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_b1_b2_refold_smoke_%j.out
#SBATCH --error=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_b1_b2_refold_smoke_%j.err
set -euo pipefail
REPO=/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
cd "$REPO"
$PY scripts/strategy01/stage07_b1_b2_refold_smoke.py
