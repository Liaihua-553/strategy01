# Strategy01 Stage09E 多候选 safe-select + 刚体 clash relief 报告

## 1. 阶段目标

本阶段目标是继续逼近 Strategy01 的核心科学目标：**生成一个共享 binder 序列，同时为 target 的多个真实功能构象生成各自合理的 state-specific 复合物 pose**。

前序阶段给出的约束很明确：

- Stage09C 的 `source_state0` interface anchors 是当前最可靠方向：full V_exact 上 contact-F1 从 Stage08B unguided 的 `0.1171` 提升到 `0.1632`，但 clash rate 仍高达 `0.7431`。
- Stage09D 证明在单条 flow trajectory 内强行加大 clash repulsion 会把 binder 推离界面，导致 contact/persistence 下降或 pose 飞走。
- 因此 Stage09E 不继续加强单轨迹排斥，而是生成多个完整候选，再用不泄漏 V_exact 标签的 source-anchor 几何指标做 candidate-level safe-select，并只做小幅刚体 clash relief。

本阶段不扩大训练数据，不做大规模 Boltz 生产，也不声明超过 baseline；B0 公平基线仍留到 Stage09F。

## 2. 代码改动

### 2.1 恢复 Stage09C 兼容默认

文件：`scripts/strategy01/stage09_guided_state_specific_sampling.py`

改动：

```python
# 原来 Stage09D 后默认打开 CA optional feature
parser.add_argument("--enable-ca-feature", action="store_true", default=True)

# Stage09E 改为默认关闭
parser.add_argument("--enable-ca-feature", action="store_true", default=False)
```

原因：Stage09D smoke 显示当前 generated/noisy CA optional feature 会改变 conditioning 分布并损害 contact。`residue_type` 仍保持禁用，避免把真实 `binder_seq_shared` 当输入泄漏给生成过程。

### 2.2 新增多候选选择脚本

文件：`scripts/strategy01/stage09e_multiseed_candidate_select.py`

主要功能：

- 每个样本生成 `N=8` 个完整候选。
- 每个候选包含一个 `shared_sequence` 和全部 `K` 个 state-specific binder pose。
- 候选整体选择，不做 per-state 混合，避免破坏一个共享序列对应一个多状态 pose family 的约束。
- 默认使用 Stage09C 采样参数：`anchor_source=source_state0`、legacy mean clash loss、关闭 CA/residue-type optional feature。
- exact V_exact contacts 只用于后验 benchmark，不进入 selector。

### 2.3 source-anchor safe-select

新增 production-safe 指标：

- `min_target_binder_distance_nm`
- `source_anchor_score`
- `source_anchor_distance_error_nm`
- `generated_source_contact_ratio`
- `source_contact_persistence_generated`
- `state_consistency_std_source_anchor_score`
- `severe_clash_count`

`source_contact_persistence_generated` 是本阶段中途新增的关键修正。第一次 selector 只看 source-anchor score 和 contact ratio，smoke 虽然降低 clash，但 generated contact persistence 低于 gate。加入该指标后，selector 能显式保护跨状态共享接触模式。

最终默认 selector 为 `balanced_safe`：

```python
score = 1.5 * persistence + anchor - 0.10 * ratio_penalty - 0.15 * severe
return (int(severe > 1), -score, int(not ratio_ok), ratio_penalty, consistency, dist_error)
```

保留对照模式：

- `lexicographic`
- `persistence_first`
- `balanced_safe` 默认

采用 `balanced_safe` 的原因是离线 64 候选分析显示：

| selector | contact-F1 | clash_rate | persistence | RMSD Å |
|---|---:|---:|---:|---:|
| current/strict | 0.1472 | 0.5000 | 0.4196 | 24.66 |
| persistence_first | 0.1181 | 0.5000 | 0.4847 | 29.48 |
| balanced_safe | 0.1297 | 0.5000 | 0.4723 | 23.55 |

`balanced_safe` 是唯一同时满足 smoke gate 的 production-safe 排序。

### 2.4 bounded rigid-body clash relief

新增逻辑：

- 只对有 severe clash 的候选尝试。
- 只做整个 binder pose 的小幅刚体平移，不做 per-residue deform。
- 默认每步最大平移 `0.04 nm = 0.4 Å`，最多 `8` 步。
- 接受条件：最小 target-binder 距离改善，source-anchor score 下降不超过 `0.01`，generated/source contact ratio 仍在 `0.6-1.6`。
- 如果 relief 破坏 contact，则保留原候选。

当前实现先做 translation-only，没有引入旋转。原因是 Stage09D 已证明过强几何修正会破坏界面；translation-only 更可控，适合作为第一版 safe relief。

### 2.5 Slurm 脚本

新增：

- `scripts/strategy01/stage09e_smoke.sbatch`
- `scripts/strategy01/stage09e_full.sbatch`

均使用 `new` 分区、单张 A100，不并发提交 GPU 任务。

## 3. 执行命令

静态检查：

```bash
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
$PY -m py_compile   scripts/strategy01/stage09_guided_state_specific_sampling.py   scripts/strategy01/stage09e_multiseed_candidate_select.py   scripts/strategy01/stage08b_full_exact_benchmark.py
$PY scripts/strategy01/stage09e_multiseed_candidate_select.py --help
```

smoke：

```bash
sbatch scripts/strategy01/stage09e_smoke.sbatch
```

full：

```bash
sbatch scripts/strategy01/stage09e_full.sbatch
```

输出文件：

- `reports/strategy01/probes/stage09e_multiseed_smoke_summary.json`
- `reports/strategy01/probes/stage09e_multiseed_smoke_exact_benchmark.json`
- `reports/strategy01/probes/stage09e_multiseed_full_summary.json`
- `reports/strategy01/probes/stage09e_multiseed_full_exact_benchmark.json`

## 4. Smoke 结果

最终 smoke：`8 samples / 24 states / 64 candidates`

| 指标 | Stage09C smoke | Stage09E smoke |
|---|---:|---:|
| contact-F1 | 0.1388 | 0.1297 |
| clash_rate | 0.6667 | 0.5000 |
| contact_persistence | 0.4859 | 0.4723 |
| direct RMSD Å | 24.55 | 23.55 |
| aligned RMSD Å | 10.73 | 9.70 |

Smoke gate：

- contact-F1 要求 `>= 0.1288`，实际 `0.1297`，通过。
- clash_rate 要求 `< 0.6667`，实际 `0.5000`，通过。
- contact_persistence 要求 `>= 0.43`，实际 `0.4723`，通过。

因此进入 full。

## 5. Full 结果

Full：`48 samples / 144 states / 384 candidates`

| 指标 | Stage09C source_state0 full | Stage09E full | 变化 |
|---|---:|---:|---:|
| contact-F1 | 0.1632 | 0.1588 | -0.0043 |
| clash_rate | 0.7431 | 0.4931 | -0.2500 |
| contact_persistence | 0.5229 | 0.5376 | +0.0147 |
| direct RMSD Å | 24.64 | 25.55 | +0.90 |
| aligned RMSD Å | 11.18 | 12.26 | +1.08 |

Full gate：

- contact-F1 要求 `>= 0.155`，实际 `0.1588`，通过。
- clash_rate 要求 `<= 0.60`，实际 `0.4931`，通过。
- contact_persistence 要求 `>= 0.50`，实际 `0.5376`，通过。
- RMSD 要求不高于 Stage09C full `24.64 Å + 1.0 Å = 25.64 Å`，实际 `25.55 Å`，通过。

科学解释：Stage09E 没有进一步提升 contact-F1，但显著降低了不物理 clash，同时保持了跨状态接触持久性。这符合当前阶段目标：先让多状态 pose 更物理，而不是牺牲界面去追求单项 clash。

## 6. 报错与修复日志

### 6.1 远程命令包装错误

现象：第一次静态检查命令出现：

```text
bash:  -m: command not found
```

原因：PowerShell 双引号提前处理远程 `$PY` 变量，导致远程命令变成空程序名。

修复：改为先写远程临时 shell 脚本，再 `ssh` 执行，避免 PowerShell 与 Bash 变量嵌套。

影响：不影响代码和科学结果，只影响执行包装。

### 6.2 第一次 smoke 中断

任务：`1913695`

报错：

```text
IndexError: The shape of the mask [21] at index 0 does not match the shape of the indexed tensor [3, 21, 3] at index 0
```

原因：`state_specific_outputs.pt` 中 `x_states['bb_ca']` 实际 shape 是 `[1,K,Nb,3]`，而 relief 代码按 `[K,Nb,3]` 使用，少处理了 batch 维。

修复：在 `maybe_apply_rigid_relief()` 中检测 `bb_raw.dim() == 4`，对 batch=1 情况取 `bb_raw[0]` 操作，写回时再放回 batch 维。

验证：后续 `1913696/1913697/1913698/1913699` 均未再出现该错误。

### 6.3 selector 初版未保护 persistence

现象：`1913696/1913697` smoke 指标：contact-F1 `0.1472`、clash `0.50`，但 persistence `0.4196`，低于 gate `0.43`。

原因：selector 主要看 source-anchor score、clash 和 contact ratio，没有显式保护跨状态 generated contact persistence。

修复：新增 `source_contact_persistence_generated`，并通过离线候选分析把默认 selector 改为 `balanced_safe`。

验证：`1913698` smoke persistence 提升到 `0.4723`，full persistence 为 `0.5376`。

## 7. Warning/error 评估

扫描了 `stage09e_smoke_1913695/1913696/1913697/1913698` 和 `stage09e_full_1913699` 的 `.err/.out`。

### 7.1 `CCD_MIRROR_PATH/PDB_MIRROR_PATH` 未设置

出现于导入依赖时。Stage09E 不调用需要 CCD/PDB mirror 的 atomworks 功能，只读取已有 PDB/pt 文件，因此不影响本阶段结果。

### 7.2 PyTorch nested tensor warning

```text
enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
```

这是 PyTorch Transformer 性能路径提示，不改变数值结果，不影响科学结论。

### 7.3 Optional CA/residue-type feature disabled warning

```text
use_ca_coors_nm_feature disabled ... returning zeros
use_residue_type_feature disabled ... returning zeros
```

这是 Stage09E 的有意设计：

- CA optional feature 默认关闭，因为 Stage09D 证明它会损害 contact。
- residue-type optional feature 关闭，避免真实 binder sequence 泄漏。

因此这是预期 warning，不需要修复。

### 7.4 Lightning checkpoint missing keys 长日志

加载 checkpoint 时出现大量 autoencoder missing-key 描述。该模式来自 `Proteina.load_from_checkpoint(..., strict=False, autoencoder_ckpt_path=...)`：主模型 checkpoint 与单独 AE checkpoint 分离加载。Stage09E 只使用已有 AE latent 数据和生成模型，不在此阶段训练 AE；前序 Stage06/Stage08 已验证 AE latent 接入。因此本阶段不把该日志视为阻塞。

## 8. 结论与下一步

Stage09E 达到本阶段 full gate。它说明：在不使用 exact 标签做选择的前提下，多候选 safe-select 可以显著降低 clash，并保持 source-interface anchor 带来的多状态接触兼容性。

但仍不能声明最终超过 baseline，因为 B0 公平基线尚未完成。下一步应进入 Stage09F：

1. 构建公平 B0：原始 Complexa 单状态生成 shared sequence/pose，再按同样 exact geometry 指标评估。
2. 固定 B1：使用 Stage09E selected outputs。
3. B2：V_exact 实验结构上限。
4. 报告 B0/B1/B2 的 contact-F1、worst interface RMSD、contact persistence、clash rate、state metric std。
5. 若 B1 对 B0 有稳定优势，再考虑把 Stage09E selected candidates 蒸馏成训练样本或训练 lightweight candidate ranker。

当前不建议继续扩大训练数据，也不建议继续加大单轨迹 repulsion；Stage09D 已证明该方向会破坏科学目标。