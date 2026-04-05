#!/bin/bash
set -euo pipefail

# Purpose:
# - run the smallest RF3-free ligand-conditioned end-to-end smoke test
# - validate generate -> filter -> evaluate(monomer) -> analyze(monomer)

ENV_NAME="${ENV_NAME:-complexa}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG_PATH="configs/variants/ligand/no_rf3/search_ligand_binder_no_rf3_pipeline.yaml"
CONFIG_STEM="search_ligand_binder_no_rf3_pipeline"
TARGET_TASK="39_7V11_LIGAND"
RUN_NAME="7v11_ligand_no_rf3_smoke"
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
    "++generation.search.algorithm=single-pass"
    "++generation.dataloader.batch_size=1"
    "++generation.dataloader.dataset.nres.nsamples=1"
    "++generation.args.nsteps=50"
    "++generation.filter.filter_samples_limit=5"
    # MODIFIED 2026-04-04 no-RF3 smoke defaults
    # ORIGINAL:
    # "++metric.designability_num_seq=1"
    # "++metric.compute_codesignability=false"
    #
    # MODIFIED:
    # The requested download set provides LigandMPNN + ESMFold cache, but not
    # ProteinMPNN weights. Designability would therefore fail early and short-
    # circuit the actual folding stage. For a real smoke test, default to
    # codesignability-only monomer evaluation on the generated binder sequence.
    "++metric.compute_designability=false"
    "++metric.compute_codesignability=true"
    "++metric.codesignability_modes=[ca]"
)

EVAL_ARGS=(
    "${COMMON_ARGS[@]}"
    "++sample_storage_path=${INFERENCE_DIR}"
    "++output_dir=${EVAL_DIR}"
)

ANALYZE_ARGS=(
    "${COMMON_ARGS[@]}"
    "++results_dir=${EVAL_DIR}"
)

python -m proteinfoundation.cli.cli_runner validate design "${CONFIG_PATH}"
python -m proteinfoundation.cli.cli_runner generate "${CONFIG_PATH}" "${COMMON_ARGS[@]}" --verbose
python -m proteinfoundation.cli.cli_runner filter "${CONFIG_PATH}" "${COMMON_ARGS[@]}" --verbose
python -m proteinfoundation.cli.cli_runner evaluate "${CONFIG_PATH}" "${EVAL_ARGS[@]}" --verbose
python -m proteinfoundation.cli.cli_runner analyze "${CONFIG_PATH}" "${ANALYZE_ARGS[@]}" --verbose
