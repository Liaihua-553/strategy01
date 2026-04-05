#!/bin/bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-complexa}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

configure_vault_repos() {
    if [ ! -f /etc/yum.repos.d/CentOS-Base.repo ]; then
        return
    fi
    python - <<'PY'
from pathlib import Path
path = Path("/etc/yum.repos.d/CentOS-Base.repo")
text = path.read_text()
d = "$"
text = text.replace(f"mirrorlist=http://mirrorlist.centos.org/?release={d}releasever&arch={d}basearch&repo=os&infra={d}infra", "# mirrorlist disabled")
text = text.replace(f"mirrorlist=http://mirrorlist.centos.org/?release={d}releasever&arch={d}basearch&repo=updates&infra={d}infra", "# mirrorlist disabled")
text = text.replace(f"mirrorlist=http://mirrorlist.centos.org/?release={d}releasever&arch={d}basearch&repo=extras&infra={d}infra", "# mirrorlist disabled")
text = text.replace(f"mirrorlist=http://mirrorlist.centos.org/?release={d}releasever&arch={d}basearch&repo=centosplus&infra={d}infra", "# mirrorlist disabled")
text = text.replace(f"#baseurl=http://mirror.centos.org/centos/{d}releasever/os/{d}basearch/", f"baseurl=http://vault.centos.org/7.9.2009/os/{d}basearch/")
text = text.replace(f"#baseurl=http://mirror.centos.org/centos/{d}releasever/updates/{d}basearch/", f"baseurl=http://vault.centos.org/7.9.2009/updates/{d}basearch/")
text = text.replace(f"#baseurl=http://mirror.centos.org/centos/{d}releasever/extras/{d}basearch/", f"baseurl=http://vault.centos.org/7.9.2009/extras/{d}basearch/")
text = text.replace(f"#baseurl=http://mirror.centos.org/centos/{d}releasever/centosplus/{d}basearch/", f"baseurl=http://vault.centos.org/7.9.2009/centosplus/{d}basearch/")
path.write_text(text)
PY
}

if [ "${CONFIGURE_VAULT_REPOS:-1}" = "1" ]; then
    configure_vault_repos
fi

yum clean all
yum makecache
yum install -y \
    git wget curl tar bzip2 which findutils patch ca-certificates \
    gcc gcc-c++ make

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "Conda is required before running this script." >&2
    exit 1
fi

conda create -n "${ENV_NAME}" -y python="${PYTHON_VERSION}" pip setuptools wheel conda-build
conda activate "${ENV_NAME}"

conda install -y -c conda-forge \
    hydra-core==1.3.1 \
    ml-collections==0.1.1 \
    python-dotenv==1.0.1 \
    einops==0.6.1 \
    dm-tree \
    loguru==0.7.2 \
    pandas==2.3.3 \
    scipy==1.12.0 \
    h5py \
    xarray \
    deepdiff \
    joblib==1.4.2 \
    rich==14.0.0 \
    rich-click \
    multipledispatch \
    plotly \
    seaborn \
    scikit-learn \
    jaxtyping \
    biopandas==0.5.1 \
    biopython \
    mdtraj==1.10.2 \
    prody==2.6.1 \
    openbabel \
    rdkit \
    wandb \
    transformers \
    numpy

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

pip install \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-geometric \
    -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

pip install 'lightning>=2.5,<2.6' loralib

pip install ml-dtypes==0.4.0
pip install --no-deps \
    jaxlib==0.4.29+cuda12.cudnn91 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jax==0.4.29
pip install flax==0.9.0 --no-deps
pip install nvidia-cuda-nvcc-cu12==12.4.131

pip install --no-deps \
    dm-haiku==0.0.12 \
    chex==0.1.86 \
    optax==0.2.2 \
    immutabledict==4.2.1 \
    colabdesign==1.1.1 \
    alphafold-colabfold==2.3.7

pip install --no-deps \
    biotite==1.4.0 \
    atomworks==2.2.0 \
    pyarrow==17.0.0 \
    py3dmol \
    pymol-remote \
    cytoolz==0.12.3 \
    hydride==1.2.3 \
    biotraj==1.2.2

pip install Cython==0.29.37
pip install cpdb-protein==0.2.0 --no-build-isolation #没装成功，pip缓存配额满了，删一下或移动到大空间里
pip install graphein==1.7.7 --no-deps
pip install Cython==3.1.4   #atomworks import成功还需要装toolz

pip install wget bioservices "modin[ray]" ray

#atomworks报：化学和PDB镜像路径环境变量没有设置，某些功能可能无法使用。要设置它们，您可以：
# Environment variable CCD_MIRROR_PATH not set. Will not be able to use function requiring this variable. To set it you may:
#   (1) add the line 'export VAR_NAME=path/to/variable' to your .bashrc or .zshrc file
#   (2) set it in your current shell with 'export VAR_NAME=path/to/variable'
#   (3) write it to a .env file in the root of the atomworks.io repository
# Environment variable PDB_MIRROR_PATH not set. Will not be able to use function requiring this variable. To set it you may:
#   (1) add the line 'export VAR_NAME=path/to/variable' to your .bashrc or .zshrc file
#   (2) set it in your current shell with 'export VAR_NAME=path/to/variable'
#   (3) write it to a .env file in the root of the atomworks.io repository





if [ "${INSTALL_OPTIONAL_TOOLS:-0}" = "1" ]; then
    conda install -y -c conda-forge -c bioconda \
        foldseek=10.941cd33 \
        mmseqs2=18.8cc5c \
        dssp=4.6.0  #4.5.3成功
fi

conda develop "${REPO_ROOT}/src"
conda develop "${REPO_ROOT}/community_models"

mkdir -p "${REPO_ROOT}/ckpts" "${REPO_ROOT}/community_models/ckpts"

if [ ! -f "${REPO_ROOT}/.env" ]; then
    cp "${REPO_ROOT}/env/conda/complexa.centos7.env.template" "${REPO_ROOT}/.env"
fi

python -m proteinfoundation.cli.cli_runner init uv --force

echo
echo "Environment creation finished."
echo "Next steps:"
echo "  1. Edit ${REPO_ROOT}/.env"
echo "  2. source ${REPO_ROOT}/env/conda/activate_complexa_conda.sh ${ENV_NAME}"
echo "  3. bash ${REPO_ROOT}/env/download_startup.sh --complexa --af2"
echo "  4. Optionally add --pmpnn --ligmpnn --esm2 for the remote full run"
echo "  5. Set INSTALL_OPTIONAL_TOOLS=1 before rerunning this script if you also want foldseek/mmseqs2/dssp"
