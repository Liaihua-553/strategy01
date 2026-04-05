#!/bin/bash
set -euo pipefail

# Purpose:
# - run the RF3-free ligand-conditioned full workflow
# - user can scale sample count and parallelism with environment variables

ENV_NAME="${ENV_NAME:-complexa}"
GEN_NJOBS="${GEN_NJOBS:-1}"
EVAL_NJOBS="${EVAL_NJOBS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NSAMPLES="${NSAMPLES:-8}"
NSTEPS="${NSTEPS:-400}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG_PATH="configs/variants/ligand/no_rf3/search_ligand_binder_no_rf3_pipeline.yaml"
CONFIG_STEM="search_ligand_binder_no_rf3_pipeline"
TARGET_TASK="${TARGET_TASK:-39_7V11_LIGAND}"
RUN_NAME="${RUN_NAME:-7v11_ligand_no_rf3_full}"
INFERENCE_DIR="./inference/${CONFIG_STEM}_${TARGET_TASK}_${RUN_NAME}"
EVAL_DIR="./evaluation_results/${CONFIG_STEM}_${TARGET_TASK}_${RUN_NAME}"

source "${CONDA_ROOT}/activate_complexa_conda.sh" "${ENV_NAME}"
cd "${REPO_ROOT}"

if [ ! -f "./ckpts/complexa_ligand.ckpt" ] || [ ! -f "./ckpts/complexa_ligand_ae.ckpt" ]; then
    echo "Ligand checkpoints are missing under ./ckpts" >&2
    exit 1
fi

export CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/.cache}"
export COMPLEXA_ALLOW_TRUSTED_TORCH_BIN="${COMPLEXA_ALLOW_TRUSTED_TORCH_BIN:-1}"
if [ -d "${CACHE_DIR}/models--facebook--esmfold_v1" ]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

COMMON_ARGS=(
    "++run_name=${RUN_NAME}"
    "++generation.task_name=${TARGET_TASK}"
    "++gen_njobs=${GEN_NJOBS}"
    "++eval_njobs=${EVAL_NJOBS}"
    "++generation.dataloader.batch_size=${BATCH_SIZE}"
    "++generation.dataloader.dataset.nres.nsamples=${NSAMPLES}"
    "++generation.args.nsteps=${NSTEPS}"
    # MODIFIED 2026-04-04 no-RF3 full defaults
    # ORIGINAL:
    # no explicit monomer metric override
    #
    # MODIFIED:
    # Match the verified dependency set used in the local CentOS7 run:
    # codesignability + ESMFold is enabled by default, while ProteinMPNN-
    # based designability stays off unless the user explicitly re-enables it.
    "++metric.compute_designability=false"
    "++metric.compute_codesignability=true"
    "++metric.codesignability_modes=[ca]"
    "++sample_storage_path=${INFERENCE_DIR}"
    "++output_dir=${EVAL_DIR}"
    "++results_dir=${EVAL_DIR}"
)

python -m proteinfoundation.cli.cli_runner validate design "${CONFIG_PATH}"
python -m proteinfoundation.cli.cli_runner design "${CONFIG_PATH}" "${COMMON_ARGS[@]}" --verbose
