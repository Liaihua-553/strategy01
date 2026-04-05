#!/bin/bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-complexa}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/activate_complexa_conda.sh" "${ENV_NAME}"
cd "${REPO_ROOT}"

COMMON_ARGS=(
    "++run_name=pdl1_smoke"
    "++generation.task_name=02_PDL1"
    "++generation.search.algorithm=single-pass"
    "++generation.dataloader.batch_size=1"
    "++generation.dataloader.dataset.nres.nsamples=1"
    "++generation.args.nsteps=50"
    "++generation.reward_model.reward_models.af2folding.num_recycles=1"
    "++generation.filter.filter_samples_limit=5"
    "++metric.compute_esm_metrics=false"
    "++metric.compute_monomer_metrics=false"
    "++aggregation.analysis_modes=[binder]"
)

python -m proteinfoundation.cli.cli_runner generate configs/search_binder_local_pipeline.yaml "${COMMON_ARGS[@]}" --verbose
python -m proteinfoundation.cli.cli_runner filter configs/search_binder_local_pipeline.yaml "${COMMON_ARGS[@]}" --verbose
python -m proteinfoundation.cli.cli_runner evaluate configs/search_binder_local_pipeline.yaml "${COMMON_ARGS[@]}" --verbose
python -m proteinfoundation.cli.cli_runner analyze configs/search_binder_local_pipeline.yaml "${COMMON_ARGS[@]}" --verbose
