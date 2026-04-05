#!/bin/bash
set -euo pipefail

# CREATED 2026-04-05: launch the verified PDL1 full workflow on the SLURM new partition.
# The actual design command stays in env/conda/run_pdl1_full.sh so file edits and
# lightweight validation can remain on the login node.

GPUS="${GPUS:-4}"
ENV_NAME="${ENV_NAME:-proteina-complexa}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec srun -p new --gres="gpu:${GPUS}" bash -lc \
    "cd \"${CONDA_ROOT}/../..\" && ENV_NAME=\"${ENV_NAME}\" bash env/conda/run_pdl1_full.sh"
