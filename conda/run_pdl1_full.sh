#!/bin/bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-complexa}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/activate_complexa_conda.sh" "${ENV_NAME}"
cd "${REPO_ROOT}"

python -m proteinfoundation.cli.cli_runner design configs/search_binder_local_pipeline.yaml \
    ++run_name=pdl1_full \
    ++generation.task_name=02_PDL1 \
    --verbose
