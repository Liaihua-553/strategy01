# Python 3.11 主环境 + Python 3.12 RF3 Sidecar 方案

## 适用场景

这个方案是可选高级方案，适用于：

- 你坚持 Complexa 主环境继续用 `Python 3.11`
- 但又想使用官方更顺的 `rf3` / `rc-foundry` 安装路径
- 愿意单独维护一个 `Python 3.12` 辅助环境

这不是当前主验收路径，只是你后续可选的 unblock 方案。

## 核心思想

- `Python 3.11` 环境：
  - 跑 Complexa 主体
  - 跑 ligand 生成、filter、evaluate、analyze
- `Python 3.12` sidecar 环境：
  - 只负责提供 `rf3` 可执行命令

主环境通过：

```bash
RF3_EXEC_PATH=/path/to/py312_env/bin/rf3
```

把 sidecar 接回来。

## 目录建议

```text
~/envs/complexa311      # 主环境
~/envs/rf3py312         # sidecar 环境
~/work/Proteina-complexa
```

## 建 sidecar 环境

示例：

```bash
conda create -n rf3py312 -y python=3.12
conda activate rf3py312
pip install "rc-foundry[all]"
```

确认：

```bash
rf3 --help
```

## 主环境接入 sidecar

主环境激活后：

```bash
export RF3_EXEC_PATH=/home/<user>/envs/rf3py312/bin/rf3
export RF3_CKPT_PATH=/home/<user>/work/Proteina-complexa/community_models/ckpts/RF3/rf3_foundry_01_24_latest_remapped.ckpt
```

然后验证：

```bash
python -m proteinfoundation.cli.cli_runner validate design configs/search_ligand_binder_local_pipeline.yaml
```

## 共享内容

两个环境共享这些资源：

- 同一份仓库目录
- 同一份 `ckpts/`
- 同一份 `community_models/ckpts/RF3/`
- 同一份输出目录 `inference/` / `evaluation_results/`

## 回退方式

如果 dual-env sidecar 不稳定，直接回退到：

- `Python 3.11 无 RF3 主线`
- 继续跑 `search_ligand_binder_no_rf3_pipeline.yaml`

这样不会把整个项目卡死在 RF3 安装上。
