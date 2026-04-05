<!-- MODIFIED 2026-04-04: Rewritten with verified no-RF3 smoke debugging history. -->
# CentOS7 排障与错误处理记录

## 使用方式

遇到问题时，建议按下面这个格式排：

1. 报错现象
2. 根因判断
3. 处理步骤
4. 复验方式
5. 当前状态

下面是这次本地 CentOS7 + Python 3.11 实测中已经确认过的高频问题。

## 1. `GLIBC_2.27` 相关错误

### 报错现象

- 使用仓库默认更高版本 CUDA / Torch 组合时，运行期提示缺更高版本 `glibc`

### 根因

- CentOS7 的用户态太老
- 新版 CUDA 依赖链要求更高版本系统库

### 处理步骤

不要走仓库默认的 `torch 2.7.0+cu126`，改用已经验证过的组合：

- `torch 2.5.1+cu124`
- `torchvision 0.20.1`
- `torchaudio 2.5.1`

### 复验方式

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

### 当前状态

- 已确认这是 CentOS7 主线可用组合

## 2. `ptxas` 缺失导致 JAX 初始化失败

### 报错现象

- `jax` / `jaxlib` 初始化 CUDA 失败
- 提示找不到 `ptxas`

### 根因

- 只装 `jaxlib` 不够
- 还需要 CUDA nvcc 工具链

### 处理步骤

```bash
pip install nvidia-cuda-nvcc-cu12==12.4.131
```

### 复验方式

```bash
python - <<'PY'
import jax
print(jax.devices())
PY
```

## 3. 老版本 `wget` 不支持 `--show-progress`

### 报错现象

- 在 CentOS7 上跑下载脚本时，因为 `wget` 参数不兼容而失败

### 根因

- 系统自带 `wget` 太旧

### 处理步骤

- 继续使用已经兼容处理过的 `env/download_startup.sh`
- 它会自动探测并降级下载参数

### 复验方式

```bash
bash env/download_startup.sh --status
```

## 4. `cpdb-protein` / `graphein` 构建失败

### 报错现象

- `cpdb-protein` 编译失败
- `graphein` 安装后导入异常

### 根因

- CentOS7 的编译环境偏老

### 处理步骤

```bash
yum install -y gcc gcc-c++ make
pip install Cython==0.29.37
pip install cpdb-protein==0.2.0 --no-build-isolation
pip install graphein==1.7.7 --no-deps
```

## 5. `RF3_EXEC_PATH` 缺失

### 报错现象

- `validate design configs/search_ligand_binder_local_pipeline.yaml` 失败

### 根因

- 标准 ligand binder 路线要求 `rf3` 可执行
- 当前主环境没有 `rf3`

### 处理步骤

二选一：

1. 继续走 no-RF3 主线  
   使用 `configs/variants/ligand/no_rf3/search_ligand_binder_no_rf3_pipeline.yaml`
2. 等拿到可用 `rf3` 后再切标准路线

### 复验方式

```bash
python -m proteinfoundation.cli.cli_runner validate design configs/search_ligand_binder_local_pipeline.yaml
```

## 6. 只有 RF3 checkpoint，没有 `rf3` 命令

### 报错现象

- `RF3_CKPT_PATH` 已存在
- 但 `RF3_EXEC_PATH` 指向空路径或不可执行文件

### 根因

- checkpoint 只提供权重
- 不能替代 `rf3 fold ...` 命令本身

### 处理步骤

- 不要把这种状态当成“RF3 已可用”
- 继续走 no-RF3 主线，或者采用双环境 sidecar 方案

## 7. no-RF3 结果被误写成 ligand binding 成功

### 报错现象

- 已经生成了 ligand 条件样本，也完成了 monomer analyze
- 但结果被当成 ligand binder 成功

### 根因

- no-RF3 主线只验证生成与 monomer 可设计性
- 不包含 RF3 标准 binding 证据

### 处理步骤

报告中必须明确写：

- `ligand-conditioned generation + monomer evaluation`

不要写：

- `RF3-backed ligand binder success`

## 8. `complexa_ligand.ckpt` / `complexa_ligand_ae.ckpt` 缺失

### 报错现象

- `validate design configs/variants/ligand/no_rf3/search_ligand_binder_no_rf3_pipeline.yaml` 失败

### 根因

- 当前工作区只有 protein binder 的 `complexa.ckpt` / `complexa_ae.ckpt`
- 还没有下载 ligand 专用 checkpoint

### 处理步骤

```bash
bash env/download_startup.sh --complexa-ligand
ls ckpts/complexa_ligand.ckpt
ls ckpts/complexa_ligand_ae.ckpt
```

### 复验方式

```bash
python -m proteinfoundation.cli.cli_runner validate design configs/variants/ligand/no_rf3/search_ligand_binder_no_rf3_pipeline.yaml
```

## 9. Windows 终端 `GBK` 编码导致 `validate` 崩掉

### 报错现象

- 在 Windows PowerShell 直接跑 `python -m proteinfoundation.cli.cli_runner validate ...`
- 报 `UnicodeEncodeError: 'gbk' codec can't encode character ...`

### 根因

- validation 报告里有 emoji 和 UTF-8 字符
- 当前终端输出编码是 `GBK`

### 处理步骤

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = 'utf-8'
python -m proteinfoundation.cli.cli_runner validate design configs/search_ligand_binder_local_pipeline.yaml
```

### 复验方式

- validation report 能正常打印，不再在 `print_report()` 阶段崩掉

## 10. CentOS7 访问不了 HuggingFace

### 报错现象

- `env/download_startup.sh --esm2` 或运行时下载 ESMFold/ESM 权重失败
- `requests` / `httpx` / `curl` 到 `https://huggingface.co` 报 `Network is unreachable`

### 根因

- 本次本地 CentOS7 环境能访问其他站点，但不能直接访问 HuggingFace

### 处理步骤

1. 在能访问 HuggingFace 的 Windows/Linux 机器下载：

```python
from huggingface_hub import snapshot_download
snapshot_download("facebook/esmfold_v1", cache_dir=".../.cache")
```

2. 把 `.cache/models--facebook--esmfold_v1` 搬到仓库根目录
3. 运行 ligand 脚本时让其自动启用：
   - `HF_HUB_OFFLINE=1`
   - `TRANSFORMERS_OFFLINE=1`

### 复验方式

- 运行时不再尝试联网
- ESMFold 可以直接从 `.cache/` 成功加载

### 当前状态

- 已在本地通过 Windows 预下载缓存的方式解决

## 11. `transformers` 拒绝加载本地 `.bin`，提示 `torch<2.6`

### 报错现象

- 加载 `facebook/esmfold_v1` 时被 `transformers` 安全检查阻止
- 栈里会落在 `check_torch_load_is_safe`

### 根因

- 当前主环境为了兼容 CentOS7 固定在 `torch 2.5.1`
- 新版 `transformers` 默认阻止在 `torch<2.6` 下加载 `.bin`

### 处理步骤

- 只对“本地可信缓存”开启兼容补丁
- 运行时设置：

```bash
export COMPLEXA_ALLOW_TRUSTED_TORCH_BIN=1
```

当前仓库中已经把这个兼容逻辑封装到：

- `src/proteinfoundation/utils/hf_load_compat.py`

并接入：

- `src/proteinfoundation/metrics/folding_models.py`
- `src/proteinfoundation/evaluation/esm_eval.py`

### 复验方式

- `facebook/esmfold_v1` 能从本地 `.cache` 成功加载

## 12. ESMFold 在低内存 WSL/CentOS7 上被 OOM Kill

### 报错现象

- 运行到 ESMFold 时进程直接被系统杀掉
- `dmesg` 里能看到 OOM kill

### 根因

- WSL 分配内存过小
- 同时本地 GPU 显存不够，ESMFold 退回 CPU，进一步放大内存压力

### 处理步骤

Windows 侧新增：

```ini
[wsl2]
memory=12GB
swap=8GB
```

然后执行：

```powershell
wsl --shutdown
```

### 复验方式

- `free -h` 能看到更大的内存与 swap
- ESMFold 不再在加载阶段被 OOM kill

### 当前状态

- 本地 smoke 已依赖这个修复跑通

## 13. no-RF3 时 `total_reward` 为空，filter 把样本全丢掉

### 报错现象

- 生成成功
- 但 `rewards_*.csv` 中 `total_reward` 为空
- `filter.py` 的 `dropna(subset=["total_reward"])` 直接把样本全部删掉

### 根因

- 原逻辑只在存在 reward model 时才为 final samples 计算 reward
- no-RF3 场景下 `reward_model=null`，导致 rewards 根本没写入

### 处理步骤

- 修改 `src/proteinfoundation/proteina.py`
- 即使 `reward_model is None`，也统一调用 `compute_reward_from_samples()`
- `reward_utils` 在这种情况下返回 `total_reward=0.0`

### 复验方式

- `rewards_*.csv` 里的 `total_reward` 变成 `0.0`
- filter 不再把样本全部删光

### 当前状态

- 已本地验证修复成功

## 14. no-RF3 嵌套配置下 evaluate/analyze 默认目录推断错误

### 报错现象

- generate 输出目录是  
  `./inference/search_ligand_binder_no_rf3_pipeline_39_7V11_LIGAND_<run_name>`
- 但 evaluate 默认去找  
  `./inference/search_ligand_binder_no_rf3_pipeline_<run_name>`
- 导致 `FileNotFoundError`

### 根因

- no-RF3 变体关闭了 binder 模式分析，evaluate/analyze 的默认推断没带上 `task_name`

### 处理步骤

在脚本里显式传固定路径：

- `++sample_storage_path=...`
- `++output_dir=...`
- `++results_dir=...`

对应脚本：

- `env/conda/runs/ligand/run_7v11_ligand_no_rf3_smoke.sh`
- `env/conda/runs/ligand/run_7v11_ligand_no_rf3_full.sh`

### 复验方式

- evaluate 能正确读到 inference 目录
- analyze 能正确读到 evaluation_results 目录

### 当前状态

- 已本地验证修复成功

## 15. 本地已验证成功状态

### 已验证命令

```bash
bash env/conda/runs/ligand/run_7v11_ligand_no_rf3_smoke.sh
```

### 已验证目标

- `39_7V11_LIGAND`

### 已验证结果

- generate 成功
- filter 成功
- evaluate(monomer) 成功
- analyze(monomer) 成功

### 已确认输出

- `rewards_search_ligand_binder_no_rf3_pipeline_0.csv`
- `monomer_results_search_ligand_binder_no_rf3_pipeline_0.csv`
- `RAW_monomer_results_search_ligand_binder_no_rf3_pipeline_combined.csv`
- `overall_monomer_pass_rates_search_ligand_binder_no_rf3_pipeline.csv`

### 已确认关键数值

- `_res_co_scRMSD_ca_esmfold = 187.9285430908203`
- `_res_codesignability_pass_rate_ca_esmfold = 0.0`

## 16. ProteinMPNN 权重缺失导致 designability 短路

### 报错现象

- no-RF3 评估阶段很快结束
- `monomer_results_*.csv` 存在，但 designability 相关列是空的
- 整个 evaluate 阶段只跑了十几秒，不像真的执行了 ESMFold

### 根因

- `compute_designability` 会调用 `community_models/ProteinMPNN/protein_mpnn_run.py`
- 当前最小下载集没有 `community_models/ProteinMPNN/{vanilla_model_weights,ca_model_weights,...}`
- ProteinMPNN 一失败，就会把当前样本的后续 monomer 指标一起短路成 NaN

### 处理步骤

对于当前 no-RF3 主线，默认改成：

- `metric.compute_designability=false`
- `metric.compute_codesignability=true`
- `metric.codesignability_modes=[ca]`

这样会直接用生成出来的 binder 序列做 ESMFold 回折并计算 monomer scRMSD，不再依赖 ProteinMPNN。

### 复验方式

- evaluate 时间显著增加到“真的跑了 ESMFold”的量级
- `monomer_results_*.csv` 中出现 `_res_co_scRMSD_ca_esmfold`

### 当前状态

- 已在本地用 codesignability-only 路线验证过真实 ESMFold 计算

## 推荐排障顺序

1. `validate env`
2. `validate design` 针对目标 pipeline
3. 先跑 smoke，不要直接跑 full
4. 先看路径和环境变量，再看 checkpoint，再看外部模型缓存
5. 有 RF3 再切标准 ligand binder，不要混淆验证边界
