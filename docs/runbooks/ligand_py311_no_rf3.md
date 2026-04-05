<!-- MODIFIED 2026-04-04: Rewritten after a verified local CentOS7 no-RF3 smoke run. -->
# Python 3.11 无 RF3 的 ligand 条件生成闭环

## 这条路线验证什么

这条路线用于下面这个场景：

- 主环境严格固定 `Python 3.11`
- 当前没有可用的 `rf3` 可执行
- 但希望先把“小分子条件生成 + monomer 级评估”真实跑通

这条路线验证的是：

```text
generate -> filter -> evaluate(monomer only) -> analyze(monomer only)
```

它验证的是“ligand-conditioned generation 能不能跑通，以及生成出的 binder 链能不能进入 monomer 评估/分析流程”，不是标准 RF3 ligand binder 成功判定。

## 这条路线不证明什么

这条路线不会输出也不会证明下面这些 RF3 依赖指标：

- `min_ipAE`
- `ligand_scRMSD`
- `ligand_scRMSD_aligned_allatom`
- RF3 refold 后的 ligand-binding 置信度

所以报告里必须写成：

- `ligand-conditioned generation + monomer evaluation`

不要写成：

- `ligand binder 已成功命中`

## 固定配置与脚本

固定配置文件：

```text
configs/variants/ligand/no_rf3/search_ligand_binder_no_rf3_pipeline.yaml
```

固定脚本：

```text
env/conda/runs/ligand/run_7v11_ligand_no_rf3_smoke.sh
env/conda/runs/ligand/run_7v11_ligand_no_rf3_full.sh
```

no-RF3 变体做了这些关键限制：

- 保留 `LigandFeatures`
- 保留 ligand 专用 checkpoint 和 LoRA
- 固定 `generation.search.algorithm=single-pass`
- 固定 `generation.reward_model=null`
- 关闭 `metric.compute_binder_metrics`
- 保留 `metric.compute_monomer_metrics`
- 把 `result_type` 和 `aggregation.analysis_modes` 切到 monomer
- 关闭 diversity 相关外部依赖

## 前置资源

至少需要下面这些文件：

- `ckpts/complexa_ligand.ckpt`
- `ckpts/complexa_ligand_ae.ckpt`
- `community_models/LigandMPNN/model_params/*`
- `.cache/models--facebook--esmfold_v1/*`

如果 CentOS7 本机拉不到 HuggingFace，可以先在能联网的 Windows/Linux 机器上把 `facebook/esmfold_v1` 下载到仓库根目录的 `.cache/`，再一起搬过来。当前脚本会自动优先使用本地 `.cache` 离线加载。

## 已本地验证的 smoke 结果

本地真实验证时间：

- 日期：`2026-04-04`
- 环境：`CentOS7-Complexa` WSL，`Python 3.11`
- 命令：

```bash
bash env/conda/runs/ligand/run_7v11_ligand_no_rf3_smoke.sh
```

本次实测固定参数：

- `target=39_7V11_LIGAND`
- `single-pass`
- `batch_size=1`
- `nsamples=1`
- `nsteps=50`
- `metric.compute_designability=false`
- `metric.compute_codesignability=true`
- `metric.codesignability_modes=[ca]`

说明：

- 当前这套下载集没有 `ProteinMPNN` 权重目录，所以 smoke 脚本默认关闭 `designability`，避免一进评估就被 ProteinMPNN 短路。
- smoke 脚本默认保留 `codesignability + ESMFold`，这样 monomer 指标是真实算出来的，不是空 CSV。
- full 脚本仍保留完整可扩展入口，适合后续在远程 A100 上放大。

本次实测输出目录：

- `inference/search_ligand_binder_no_rf3_pipeline_39_7V11_LIGAND_7v11_ligand_no_rf3_smoke`
- `evaluation_results/search_ligand_binder_no_rf3_pipeline_39_7V11_LIGAND_7v11_ligand_no_rf3_smoke`

本次实测耗时：

- generation：`282.44s`
- evaluation：`1305.13s`
- total：`1587.57s`

说明：

- 这是本地低显存环境下 ESMFold 自动回退 CPU 的结果
- 远程 `40G A100` 上通常会明显更快

本次实测至少确认存在这些文件：

- `rewards_search_ligand_binder_no_rf3_pipeline_0.csv`
- `all_rewards_search_ligand_binder_no_rf3_pipeline.csv`
- `top_samples_search_ligand_binder_no_rf3_pipeline.csv`
- `monomer_results_search_ligand_binder_no_rf3_pipeline_0.csv`
- `RAW_monomer_results_search_ligand_binder_no_rf3_pipeline_combined.csv`
- `overall_monomer_pass_rates_search_ligand_binder_no_rf3_pipeline.csv`

本次实测结论：

- 生成成功
- filter 成功
- monomer evaluate 成功
- analyze 成功
- 该次样本的 `_res_co_scRMSD_ca_esmfold` 为 `187.9285`
- 该次样本的 codesignability pass rate 为 `0.0`
- 这说明流程已经真实算出了 monomer 指标，但该样本本身不达标

## 推荐执行顺序

### 1. 先激活环境

```bash
source env/conda/activate_complexa_conda.sh complexa
```

### 2. 环境与配置校验

```bash
python -m proteinfoundation.cli.cli_runner validate env
python -m proteinfoundation.cli.cli_runner validate design configs/variants/ligand/no_rf3/search_ligand_binder_no_rf3_pipeline.yaml
```

### 3. 先跑 smoke

```bash
bash env/conda/runs/ligand/run_7v11_ligand_no_rf3_smoke.sh
```

### 4. 检查输出

至少要看到：

- 生成的 complex PDB
- `rewards_*.csv`
- `top_samples_*.csv`
- `monomer_results_*.csv`
- `RAW_monomer_results_*_combined.csv`

### 5. 再放大到 full

```bash
bash env/conda/runs/ligand/run_7v11_ligand_no_rf3_full.sh
```

例如在多卡机器上：

```bash
GEN_NJOBS=4 \
EVAL_NJOBS=4 \
BATCH_SIZE=8 \
NSAMPLES=32 \
NSTEPS=400 \
bash env/conda/runs/ligand/run_7v11_ligand_no_rf3_full.sh
```

## 对远程 A100 服务器的说明

你远程服务器有 `40G A100` 时，这条 no-RF3 路线通常会比本地快很多，原因是：

- ESMFold 会优先走 GPU
- monomer 评估阶段不需要再退回 CPU
- `gen_njobs` / `eval_njobs` 可以放大

建议远程机器的顺序仍然是：

1. 先跑 smoke
2. 确认输出齐全
3. 再放大到 full

## 当前脚本自动做的事情

这些脚本已经自动处理了几个容易踩坑的点：

- 自动把 `CACHE_DIR` 指到仓库根目录的 `.cache`
- 如果发现 `.cache/models--facebook--esmfold_v1`，自动启用 `HF_HUB_OFFLINE=1`
- 自动启用 `TRANSFORMERS_OFFLINE=1`
- 自动启用 `COMPLEXA_ALLOW_TRUSTED_TORCH_BIN=1`
- no-RF3 smoke/full 自动把 `sample_storage_path`、`output_dir`、`results_dir` 传成固定目录，避免 evaluate/analyze 走错默认路径
- no-RF3 smoke/full 默认走 `codesignability + ESMFold`，避开当前下载集中缺失的 `ProteinMPNN` 权重

## 结果解读边界

你应该把这条路线的成功描述为：

- `39_7V11_LIGAND` 在 `Python 3.11`、无 RF3 条件下已完成 ligand 条件生成与 monomer 闭环验证

你不应该把它写成：

- `39_7V11_LIGAND` 已完成标准 ligand binder 成功验证
