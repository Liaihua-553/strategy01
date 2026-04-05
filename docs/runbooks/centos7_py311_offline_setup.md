<!-- MODIFIED 2026-04-04: Rewritten with verified no-RF3 smoke workflow and offline ESMFold cache notes. -->
# CentOS7 Python 3.11 离线部署手册

## 目标

这份手册面向你的远程 `CentOS7` 普通用户环境，假设条件如下：

- 主环境固定 `Python 3.11`
- 远程机器不一定能联网到所有外部站点
- 没有 root 权限
- 可以使用 `conda/mamba`
- 有 `A100 40G` GPU 使用权限
- 主线先不依赖 `rf3`

主线目标是先部署并跑通：

```text
ligand-conditioned generation + monomer evaluation
```

也就是 no-RF3 路线。

## 本地准备

### 1. 创建并激活 CentOS7 Python 3.11 环境

推荐入口：

```bash
bash env/conda/install_complexa_centos7.sh
source env/conda/activate_complexa_conda.sh complexa
```

如果环境已经存在，只需要激活：

```bash
source env/conda/activate_complexa_conda.sh complexa
```

### 2. 配置 `.env`

如果仓库根目录还没有 `.env`：

```bash
cp env/conda/complexa.centos7.env.template .env
python -m proteinfoundation.cli.cli_runner init uv --force
```

重点变量至少要对：

```bash
LOCAL_CODE_PATH=/absolute/path/to/Proteina-complexa
LOCAL_DATA_PATH=${LOCAL_CODE_PATH}/assets
LOCAL_CHECKPOINT_PATH=${LOCAL_CODE_PATH}/ckpts
COMMUNITY_MODELS_PATH=${LOCAL_CODE_PATH}/community_models
AF2_DIR=${COMMUNITY_MODELS_PATH}/ckpts/AF2
ESM_DIR=${COMMUNITY_MODELS_PATH}/ckpts/ESM2
UV_VENV=/absolute/path/to/miniconda3/envs/complexa
```

### 3. 下载主线所需权重

no-RF3 主线至少需要：

```bash
bash env/download_startup.sh --complexa-ligand --ligmpnn --esm2
```

说明：

- 这套最小下载集不包含 `ProteinMPNN` 权重目录
- 因此当前 no-RF3 smoke/full 默认采用 `codesignability + ESMFold` 作为 monomer 评估主线
- 如果你后续自己补齐 `community_models/ProteinMPNN` 的权重目录，再额外开启 `designability`

如果后续还想切到标准 RF3 路线，再额外下载：

```bash
bash env/download_startup.sh --rf3
```

### 4. 如果 CentOS7 拉不到 HuggingFace

这是本次本地实测里确实遇到的问题。处理方式是：

1. 在一台能访问 HuggingFace 的机器上下载 `facebook/esmfold_v1`
2. 下载到仓库根目录的 `.cache/`
3. 把 `.cache/` 一起打包带到远程机

Windows 上的下载示例：

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/esmfold_v1', cache_dir=r'C:\LKF\Proteina-complexa\.cache')"
```

当前仓库里的 ligand 脚本已经会自动：

- 使用 `CACHE_DIR=$REPO_ROOT/.cache`
- 发现本地 ESMFold cache 后自动切到离线模式

### 5. 本地先验证一次

先做基本校验：

```bash
python -m proteinfoundation.cli.cli_runner validate env
python -m proteinfoundation.cli.cli_runner validate design configs/variants/ligand/no_rf3/search_ligand_binder_no_rf3_pipeline.yaml
```

再跑最小闭环：

```bash
bash env/conda/runs/ligand/run_7v11_ligand_no_rf3_smoke.sh
```

当前本地已经实测通过这一步。

## 生成离线包

统一入口：

```bash
bash env/conda/runs/ligand/pack_complexa_py311_offline_bundle.sh
```

输出目录固定在：

```text
artifacts/offline_bundle/<timestamp>/
```

目录结构：

```text
artifacts/offline_bundle/<timestamp>/
├─ env/
│  └─ complexa-py311.tar.gz
├─ repo/
│  └─ Proteina-complexa.tar.gz
├─ weights/
│  └─ complexa-weights-and-targets.tar.gz
└─ manifests/
   ├─ complexa-conda-explicit.txt
   ├─ complexa-pip-freeze.txt
   ├─ README.txt
   └─ SHA256SUMS.txt
```

说明：

- 现在离线包脚本也会把仓库根目录 `.cache/` 一起打进权重包
- 这样远程机器就能直接离线加载 ESMFold

## 远程服务器部署

### 1. 上传离线包

例如：

```bash
scp -r artifacts/offline_bundle/<timestamp> user@remote:/home/user/
```

### 2. 解压仓库

远程执行：

```bash
mkdir -p ~/work
cd ~/work
tar -xzf ~/offline_bundle/<timestamp>/repo/Proteina-complexa.tar.gz
```

### 3. 解压权重和缓存

```bash
cd ~/work/Proteina-complexa
tar -xzf ~/offline_bundle/<timestamp>/weights/complexa-weights-and-targets.tar.gz
```

### 4. 解压 conda 环境

```bash
mkdir -p ~/envs/complexa
tar -xzf ~/offline_bundle/<timestamp>/env/complexa-py311.tar.gz -C ~/envs/complexa
source ~/envs/complexa/bin/activate
conda-unpack
```

如果你的 `conda-pack` 实际解包结构略有不同，以实际目录为准执行 `conda-unpack`。

### 5. 改远程 `.env`

至少改这些变量：

```bash
LOCAL_CODE_PATH=/home/<user>/work/Proteina-complexa
LOCAL_DATA_PATH=${LOCAL_CODE_PATH}/assets
LOCAL_CHECKPOINT_PATH=${LOCAL_CODE_PATH}/ckpts
COMMUNITY_MODELS_PATH=${LOCAL_CODE_PATH}/community_models
AF2_DIR=${COMMUNITY_MODELS_PATH}/ckpts/AF2
ESM_DIR=${COMMUNITY_MODELS_PATH}/ckpts/ESM2
UV_VENV=/home/<user>/envs/complexa
```

no-RF3 主线下，`RF3_EXEC_PATH` 可以先不配置。

### 6. 激活并校验

```bash
source env/conda/activate_complexa_conda.sh complexa
python -m proteinfoundation.cli.cli_runner validate env
python -m proteinfoundation.cli.cli_runner validate design configs/variants/ligand/no_rf3/search_ligand_binder_no_rf3_pipeline.yaml
```

### 7. 先跑 smoke

```bash
bash env/conda/runs/ligand/run_7v11_ligand_no_rf3_smoke.sh
```

### 8. 再跑 full

```bash
GEN_NJOBS=4 \
EVAL_NJOBS=4 \
BATCH_SIZE=8 \
NSAMPLES=32 \
NSTEPS=400 \
bash env/conda/runs/ligand/run_7v11_ligand_no_rf3_full.sh
```

## 预期输出

至少要看到：

- `inference/search_ligand_binder_no_rf3_pipeline_39_7V11_LIGAND_*`
- `rewards_*.csv`
- `top_samples_*.csv`
- `evaluation_results/search_ligand_binder_no_rf3_pipeline_39_7V11_LIGAND_*`
- `monomer_results_*.csv`
- `RAW_monomer_results_*_combined.csv`

## 远程 A100 上的建议

你远程服务器是 `40G A100` 时，通常可以直接受益于：

- ESMFold 走 GPU，不用像本地低显存机器那样退回 CPU
- `gen_njobs` 与 `eval_njobs` 可以更积极地放大
- full 版本评估速度会明显好于本地

建议顺序仍然是：

1. 先 smoke
2. 再 full
3. 后续若补齐 `rf3`，再切到标准 ligand binder 流程
