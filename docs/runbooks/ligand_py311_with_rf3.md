# Python 3.11 下有 RF3 时的标准 ligand binder 路线

## 使用前提

只有同时满足以下条件，才进入标准 ligand binder 路线：

- `RF3_CKPT_PATH` 已设置且文件存在
- `RF3_EXEC_PATH` 已设置且可以直接执行
- `ckpts/complexa_ligand.ckpt` 已存在
- `ckpts/complexa_ligand_ae.ckpt` 已存在

如果只有 checkpoint，没有 `rf3` 可执行命令，这条路线仍然不能进入 fullrun。

## 使用的官方配置

标准路线继续使用原始主配置：

```text
configs/search_ligand_binder_local_pipeline.yaml
```

这里不复制第二套 ligand 主配置，避免以后和官方配置漂移。

## 环境变量示例

```bash
export RF3_CKPT_PATH=/absolute/path/to/community_models/ckpts/RF3/rf3_foundry_01_24_latest_remapped.ckpt
export RF3_EXEC_PATH=/absolute/path/to/rf3
```

验证：

```bash
"${RF3_EXEC_PATH}" --help
python -m proteinfoundation.cli.cli_runner validate design configs/search_ligand_binder_local_pipeline.yaml
```

## Smoke 流程

统一入口：

```bash
bash env/conda/runs/ligand/run_7v11_ligand_with_rf3_smoke.sh
```

这个脚本会先检查：

- `RF3_CKPT_PATH`
- `RF3_EXEC_PATH`
- ligand checkpoint 是否齐全

然后执行标准 ligand pipeline 的缩小版 smoke。

## Full 流程

统一入口：

```bash
bash env/conda/runs/ligand/run_7v11_ligand_with_rf3_full.sh
```

可用环境变量放大规模：

```bash
GEN_NJOBS=4 \
EVAL_NJOBS=4 \
BATCH_SIZE=8 \
NSAMPLES=32 \
NSTEPS=400 \
bash env/conda/runs/ligand/run_7v11_ligand_with_rf3_full.sh
```

## 标准路线包含哪些阶段

标准 ligand binder 路线包含：

1. ligand 条件生成
2. RF3 作为 reward/refolding 核心组件
3. ligand-aware inverse folding
4. binder analyze
5. 如果配置保留，也会额外跑 monomer 指标

## 标准输出重点

当 RF3 路线跑通后，才重点看这些字段：

- `min_ipAE`
- `pLDDT`
- `binder_scRMSD_ca`
- `ligand_scRMSD`
- `ligand_scRMSD_aligned_allatom`

## 如果校验失败

最常见的几类失败：

- `RF3_EXEC_PATH not set`
- `RF3 executable not found`
- `RF3_CKPT_PATH not set`
- `complexa_ligand.ckpt` 或 `complexa_ligand_ae.ckpt` 缺失

这时不要继续 fullrun，先回到：

1. `validate env`
2. `validate design`
3. 修复路径
4. 重新跑 smoke
