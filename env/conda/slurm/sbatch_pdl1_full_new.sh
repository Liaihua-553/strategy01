#!/bin/bash
set -euo pipefail

# CREATED 2026-04-05: submit the verified PDL1 full workflow to the SLURM new partition.

GPUS="${GPUS:-4}"
ENV_NAME="${ENV_NAME:-proteina-complexa}"
JOB_NAME="${JOB_NAME:-pdl1_full_remote_new}"
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
ENV_NAME="${ENV_NAME}" bash env/conda/run_pdl1_full.sh
EOF
