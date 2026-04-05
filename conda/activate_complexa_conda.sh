#!/bin/bash
set -euo pipefail

ENV_NAME="${1:-complexa}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# MODIFIED 2026-04-05: broaden conda.sh discovery for remote user-managed installs.
# ORIGINAL:
# if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
#     source "${HOME}/miniconda3/etc/profile.d/conda.sh"
# elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
#     source "${HOME}/anaconda3/etc/profile.d/conda.sh"
# else
#     echo "Could not find conda.sh under \$HOME/miniconda3 or \$HOME/anaconda3." >&2
#     exit 1
# fi
CONDA_SH_CANDIDATES=(
    "${HOME}/miniconda3/etc/profile.d/conda.sh"
    "${HOME}/anaconda3/etc/profile.d/conda.sh"
    "${HOME}/data/anaconda3/etc/profile.d/conda.sh"
    "/home/kfliao/data/anaconda3/etc/profile.d/conda.sh"
)

CONDA_SH=""
for candidate in "${CONDA_SH_CANDIDATES[@]}"; do
    if [ -f "${candidate}" ]; then
        CONDA_SH="${candidate}"
        break
    fi
done

if [ -z "${CONDA_SH}" ] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
    fi
fi

if [ -z "${CONDA_SH}" ]; then
    echo "Could not find conda.sh in the expected user-managed locations." >&2
    exit 1
fi

source "${CONDA_SH}"

conda activate "${ENV_NAME}"
cd "${REPO_ROOT}"

if [ -f "${REPO_ROOT}/env.sh" ]; then
    source "${REPO_ROOT}/env.sh"
fi

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/community_models:${PYTHONPATH:-}"
export REPO_ROOT

echo "Activated ${ENV_NAME} for Proteina-Complexa at ${REPO_ROOT}"
