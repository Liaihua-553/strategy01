# Strategy01 Stage09F 公平 B0/B1/B2 exact benchmark 报告

## 1. 阶段目标

Stage09E 已经让 B1 的多候选 safe-select 在 exact V_exact 上通过 gate：clash 明显降低，同时保持一定 contact 和 persistence。但 Stage09E 仍没有公平 B0，因此不能判断 Strategy01 是否真正优于“单状态静态设计跨状态泛化”。

Stage09F 的目标是补齐一个可运行、可复现、不会泄漏 future-state exact 标签的 B0，并用同一 `stage08b_full_exact_benchmark.py` 对 B0/B1/B2 做 exact geometry 评估。

## 2. B0 定义

本阶段实现的是：`B0_static_source_state_same_model`。

定义：

- 只给模型 source state0 的 target 和 source interface anchors。
- 每个样本生成 `N=8` 个 source-state 单状态候选。
- 候选选择只使用 source-state 可得指标，不使用其它 state 的 exact contacts。
- 选中一个 source-state binder pose 后，通过 target CA Kabsch 刚体对齐，把该静态 pose 转移到其它 target states。
- 用与 B1 相同的 exact geometry evaluator 评价所有 states。

这个 B0 不是原始 Complexa CLI native baseline。原因是原始 `ckpts/complexa.ckpt` 没有 `shared_seq_head/state_seq_head`，不能直接产出 Strategy01 exact benchmark 需要的 `shared_sequence + K state_specific binder_ca.pdb` schema。本阶段 B0 是同 checkpoint/同 refiner 的单状态消融，用来回答：**只设计 source state，能否自然泛化到其它 target 功能构象？**

这个定义是公平且严格的，因为 B0 不看 future states；但它比完全无约束原始 baseline 更强，因为它和 B1 一样允许使用 source-state interface anchors 和 same-refiner candidate selection。

## 3. 代码改动

### 3.1 新增 B0 static-source baseline 脚本

文件：`scripts/strategy01/stage09f_b0_static_source_baseline.py`

主要流程：

1. 从 `stage08b_vexact_samples_ae_latents.pt` 读取 V_exact tensor samples。
2. 对每个样本截取 source state0，构造只含一个 target state 的单状态输入。
3. 调用 Stage09C/09E 采样路径生成 `N=8` 个 source-state 候选。
4. 用 source-anchor safe-select 选择一个候选。
5. 对选中 binder pose 做 target CA alignment，刚体转移到其它 target states。
6. 输出与 B1 相同 summary schema，使 `stage08b_full_exact_benchmark.py` 可直接比较。

关键函数：

- `make_source_state_sample()`：裁剪样本，只保留 state0 输入。
- `kabsch_transform()`：用 target CA 对齐把 source binder pose 转移到其它 states。
- `transfer_source_pose_to_all_states()`：写出 `state00/state01/state02_binder_ca.pdb`。
- `sample_b0_one()`：多候选生成、source-safe 选择、pose transfer。

### 3.2 新增 Slurm 脚本

- `scripts/strategy01/stage09f_b0_smoke.sbatch`
- `scripts/strategy01/stage09f_b0_full.sbatch`

均使用：

- partition: `new`
- GPU: `--gres=gpu:1`
- 不并发提交其它 GPU 作业。

### 3.3 复用 exact benchmark

复用：`scripts/strategy01/stage08b_full_exact_benchmark.py`

smoke 输出：

- `reports/strategy01/probes/stage09f_b0_static_source_smoke_summary.json`
- `reports/strategy01/probes/stage09f_b0_b1_b2_smoke_exact_benchmark.json`

full 输出：

- `reports/strategy01/probes/stage09f_b0_static_source_full_summary.json`
- `reports/strategy01/probes/stage09f_b0_b1_b2_full_exact_benchmark.json`

## 4. 执行命令

静态检查：

```bash
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python
$PY -m py_compile scripts/strategy01/stage09f_b0_static_source_baseline.py
```

smoke：

```bash
sbatch scripts/strategy01/stage09f_b0_smoke.sbatch
```

full：

```bash
sbatch scripts/strategy01/stage09f_b0_full.sbatch
```

## 5. Smoke 结果

Smoke：`8 samples / 24 states / 64 B0 candidates`

| 方法 | contact-F1 | clash_rate | contact_persistence | direct RMSD Å |
|---|---:|---:|---:|---:|
| B0 static-source | 0.1763 | 0.6250 | 0.7632 | 24.33 |
| B1 Stage09E | 0.1297 | 0.5000 | 0.4723 | 23.55 |

Smoke 解释：B0 在 contact 和 persistence 上明显更强，但 clash 更高。B1 的优势主要是减少 clash 和略好 RMSD。

## 6. Full 结果

Full：`48 samples / 144 states / 384 B0 candidates`

| 方法 | contact-F1 | clash_rate | contact_persistence | direct RMSD Å | aligned RMSD Å |
|---|---:|---:|---:|---:|---:|
| B0 static-source | 0.2064 | 0.6667 | 0.7446 | 21.52 | 7.97 |
| B1 Stage09E | 0.1588 | 0.4931 | 0.5376 | 25.55 | 12.26 |

结论：

- B1 相比 B0 的唯一明确优势是 clash rate 更低：`0.4931` vs `0.6667`。
- B0 在 contact-F1、contact persistence、direct RMSD、aligned RMSD 上都明显优于 B1。
- 因此 Stage09F 结果不支持“当前 B1 已经超过静态单状态 B0”。
- 这也说明很多 V_exact case 的功能构象变化下，source-state 静态 pose 经过 target CA 对齐仍能保持较多接触；B1 当前随机 state-specific generation 反而破坏了这部分 interface continuity。

## 7. Warning/error 评估

扫描：

- `reports/strategy01/probes/stage09f_b0_smoke_1913987.err/out`
- `reports/strategy01/probes/stage09f_b0_full_1913988.err/out`

发现：

### 7.1 PyTorch nested tensor warning

```text
enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
```

这是性能路径提示，不改变结果。

### 7.2 Optional CA/residue-type feature disabled warning

```text
use_ca_coors_nm_feature disabled ... returning zeros
use_residue_type_feature disabled ... returning zeros
```

这是预期设计：Stage09D 已证明 noisy/generated CA feature 会损害 contact；residue-type feature 禁用是为了避免真实 binder sequence 泄漏。

未发现 traceback、OOM、NaN 或 failed run。上述 warning 不影响 Stage09F 科学结论。

## 8. 科学解释

Stage09F 的结果很有价值，因为它纠正了一个潜在错误方向：不能只看 B1 的 clash 下降就认为多状态设计成功。

当前 V_exact 上，B0 static-source 的高 contact/persistence 说明：

1. 很多目标状态间保守 interface anchors 很强。
2. 单状态 source pose 经过 target CA alignment 后，仍能解释相当多 exact contacts。
3. Strategy01 B1 的 state-specific generation 虽然降低 clash，但还没有充分继承 source interface continuity。

这和你的核心科学目标一致：真实 binder 兼容功能构象变化时，往往不是每个状态完全重新设计一个 interface，而是保留一组跨状态 anchors，并允许少量局部/刚体适应。当前 B1 还没有把这点做好。

## 9. 下一步建议：Stage10

不建议继续扩大训练或只强化 sampling repulsion。下一步应该把 B0 的强项作为 prior 接入 B1：

### Stage10：source-pose initialized state-specific refinement

核心思路：

- 不再从纯噪声生成每个 state-specific pose。
- 用 B0 static-source pose 经过 target alignment 后作为每个 state 的初始 pose。
- 在这个初始化上做小步 state-specific flow/refinement。
- loss/selection 同时约束：
  - 保持 source-anchor contact/persistence。
  - 降低 clash。
  - 允许必要的 state-specific 局部位移，但不允许 pose 飞离 source interface family。

验收指标：

- contact-F1 至少接近 B0：目标 `>=0.19`。
- clash_rate 接近 B1：目标 `<=0.55`。
- persistence 不低于 B0 太多：目标 `>=0.68`。
- RMSD 不显著坏于 B0：目标 `<=23 Å`。

这个方向比继续多 seed 随机采样更合理，因为 Stage09F 已经证明 source pose transfer 是强先验，而不是弱 baseline。