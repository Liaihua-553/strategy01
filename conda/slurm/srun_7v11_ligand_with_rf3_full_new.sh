#!/bin/bash
set -euo pipefail

# CREATED 2026-04-05: launch the standard 7V11 ligand full workflow with RF3 on the SLURM new partition.

GPUS="${GPUS:-4}"
ENV_NAME="${ENV_NAME:-proteina-complexa}"
GEN_NJOBS="${GEN_NJOBS:-4}"
EVAL_NJOBS="${EVAL_NJOBS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NSAMPLES="${NSAMPLES:-8}"
NSTEPS="${NSTEPS:-400}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec srun -p new --gres="gpu:${GPUS}" bash -lc \
    "cd \"${CONDA_ROOT}/../..\" && ENV_NAME=\"${ENV_NAME}\" GEN_NJOBS=\"${GEN_NJOBS}\" EVAL_NJOBS=\"${EVAL_NJOBS}\" BATCH_SIZE=\"${BATCH_SIZE}\" NSAMPLES=\"${NSAMPLES}\" NSTEPS=\"${NSTEPS}\" bash env/conda/runs/ligand/run_7v11_ligand_with_rf3_full.sh"
