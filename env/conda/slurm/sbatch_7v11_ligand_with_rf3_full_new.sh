#!/bin/bash
set -euo pipefail

# CREATED 2026-04-05: submit the standard 7V11 ligand full workflow with RF3 to the SLURM new partition.

GPUS="${GPUS:-4}"
ENV_NAME="${ENV_NAME:-proteina-complexa}"
GEN_NJOBS="${GEN_NJOBS:-4}"
EVAL_NJOBS="${EVAL_NJOBS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NSAMPLES="${NSAMPLES:-8}"
NSTEPS="${NSTEPS:-400}"
JOB_NAME="${JOB_NAME:-7v11_ligand_with_rf3_full_remote_new}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

sbatch <<EOF
#!/bin/bash
#SBATCH -p new
#SBATCH --job-name=${JOB_NAME}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --output=${REPO_ROOT}/logs/${JOB_NAME}_%j.out
#SBATCH --error=${REPO_ROOT}/logs/${JOB_NAME}_%j.err
set -euo pipefail
cd "${REPO_ROOT}"
ENV_NAME="${ENV_NAME}" GEN_NJOBS="${GEN_NJOBS}" EVAL_NJOBS="${EVAL_NJOBS}" BATCH_SIZE="${BATCH_SIZE}" NSAMPLES="${NSAMPLES}" NSTEPS="${NSTEPS}" bash env/conda/runs/ligand/run_7v11_ligand_with_rf3_full.sh
EOF
