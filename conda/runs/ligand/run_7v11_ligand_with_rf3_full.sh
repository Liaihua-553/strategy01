#!/bin/bash
set -euo pipefail

# Purpose:
# - run the standard ligand-binder full workflow when RF3 is available
# - leaves official config semantics intact and only standardizes invocation

ENV_NAME="${ENV_NAME:-complexa}"
GEN_NJOBS="${GEN_NJOBS:-1}"
EVAL_NJOBS="${EVAL_NJOBS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NSAMPLES="${NSAMPLES:-8}"
NSTEPS="${NSTEPS:-400}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG_PATH="configs/search_ligand_binder_local_pipeline.yaml"

source "${CONDA_ROOT}/activate_complexa_conda.sh" "${ENV_NAME}"
cd "${REPO_ROOT}"

export CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/.cache}"
export COMPLEXA_ALLOW_TRUSTED_TORCH_BIN="${COMPLEXA_ALLOW_TRUSTED_TORCH_BIN:-1}"
if [ -d "${CACHE_DIR}/models--facebook--esmfold_v1" ]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

if [ -z "${RF3_CKPT_PATH:-}" ] || [ ! -f "${RF3_CKPT_PATH}" ]; then
    echo "RF3_CKPT_PATH is missing or not a file: ${RF3_CKPT_PATH:-<unset>}" >&2
    exit 1
fi

if [ -z "${RF3_EXEC_PATH:-}" ] || [ ! -x "${RF3_EXEC_PATH}" ]; then
    echo "RF3_EXEC_PATH is missing or not executable: ${RF3_EXEC_PATH:-<unset>}" >&2
    exit 1
fi

if [ ! -f "./ckpts/complexa_ligand.ckpt" ] || [ ! -f "./ckpts/complexa_ligand_ae.ckpt" ]; then
    echo "Ligand checkpoints are missing under ./ckpts" >&2
    exit 1
fi

COMMON_ARGS=(
    "++run_name=7v11_ligand_with_rf3_full"
    "++generation.task_name=39_7V11_LIGAND"
    "++gen_njobs=${GEN_NJOBS}"
    "++eval_njobs=${EVAL_NJOBS}"
    "++generation.dataloader.batch_size=${BATCH_SIZE}"
    "++generation.dataloader.dataset.nres.nsamples=${NSAMPLES}"
    "++generation.args.nsteps=${NSTEPS}"
)

python -m proteinfoundation.cli.cli_runner validate design "${CONFIG_PATH}"
python -m proteinfoundation.cli.cli_runner design "${CONFIG_PATH}" "${COMMON_ARGS[@]}" --verbose
