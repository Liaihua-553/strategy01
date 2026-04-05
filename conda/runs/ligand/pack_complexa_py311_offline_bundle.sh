#!/bin/bash
set -euo pipefail

# Purpose:
# - create an offline transfer bundle for a Python 3.11 CentOS7 deployment
# - keep all outputs under artifacts/offline_bundle/

ENV_NAME="${ENV_NAME:-complexa}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUNDLE_ROOT="${BUNDLE_ROOT:-${REPO_ROOT}/artifacts/offline_bundle}"
STAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${BUNDLE_ROOT}/${STAMP}"
ENV_DIR="${WORK_DIR}/env"
REPO_DIR="${WORK_DIR}/repo"
WEIGHTS_DIR="${WORK_DIR}/weights"
MANIFEST_DIR="${WORK_DIR}/manifests"

mkdir -p "${ENV_DIR}" "${REPO_DIR}" "${WEIGHTS_DIR}" "${MANIFEST_DIR}"

if ! command -v conda-pack >/dev/null 2>&1; then
    echo "conda-pack not found. Install it in the active environment first:" >&2
    echo "  pip install conda-pack" >&2
    exit 1
fi

if ! command -v tar >/dev/null 2>&1; then
    echo "tar not found in PATH." >&2
    exit 1
fi

cd "${REPO_ROOT}"

conda-pack -n "${ENV_NAME}" -o "${ENV_DIR}/complexa-py311.tar.gz"

tar \
  --exclude='.git' \
  --exclude='./artifacts/offline_bundle' \
  --exclude='./inference' \
  --exclude='./evaluation_results' \
  --exclude='./logs/hydra_outputs' \
  --exclude='./.cache' \
  --exclude='./.research' \
  -czf "${REPO_DIR}/Proteina-complexa.tar.gz" \
  .

WEIGHT_INPUTS=()

if [ -d "./ckpts" ]; then
  WEIGHT_INPUTS+=("./ckpts")
fi

if [ -d "./community_models" ]; then
  WEIGHT_INPUTS+=("./community_models")
fi

if [ -d "./assets/target_data" ]; then
  WEIGHT_INPUTS+=("./assets/target_data")
fi

if [ -d "./.cache" ]; then
  WEIGHT_INPUTS+=("./.cache")
fi

if [ "${#WEIGHT_INPUTS[@]}" -gt 0 ]; then
  tar -czf "${WEIGHTS_DIR}/complexa-weights-and-targets.tar.gz" "${WEIGHT_INPUTS[@]}"
fi

cp "./env/conda/complexa-conda-explicit.txt" "${MANIFEST_DIR}/"
cp "./env/conda/complexa-pip-freeze.txt" "${MANIFEST_DIR}/"

cat > "${MANIFEST_DIR}/README.txt" <<EOF
Offline bundle created: ${STAMP}

Contents:
- env/complexa-py311.tar.gz
- repo/Proteina-complexa.tar.gz
- weights/complexa-weights-and-targets.tar.gz
- manifests/complexa-conda-explicit.txt
- manifests/complexa-pip-freeze.txt

Recommended remote unpack order:
1. unpack repo
2. unpack weights under the repo root
3. unpack the conda environment
4. run conda-unpack
5. edit .env to match remote absolute paths
EOF

if command -v sha256sum >/dev/null 2>&1; then
  (
    cd "${WORK_DIR}"
    find . -type f -maxdepth 3 | sort | xargs sha256sum > "${MANIFEST_DIR}/SHA256SUMS.txt"
  )
fi

echo "Offline bundle created at: ${WORK_DIR}"
