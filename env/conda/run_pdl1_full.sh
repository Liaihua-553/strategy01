#!/bin/bash
set -euo pipefail

# MODIFIED 2026-04-05: keep a single full-run entrypoint for remote PDL1 execution.
# ORIGINAL:
# the script only executed the design command directly.
# MODIFIED:
# retain the direct design command, but document that this is the canonical
# full-run entrypoint used by the remote SLURM wrappers.

ENV_NAME="${ENV_NAME:-complexa}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/activate_complexa_conda.sh" "${ENV_NAME}"
cd "${REPO_ROOT}"

python -m proteinfoundation.cli.cli_runner design configs/search_binder_local_pipeline.yaml \
    ++run_name=pdl1_full \
    ++generation.task_name=02_PDL1 \
    --verbose
