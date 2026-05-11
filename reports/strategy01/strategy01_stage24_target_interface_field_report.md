# Strategy01 Stage24 target-interface field 阶段报告

## 1. 阶段目标

Stage22-23 后的主要瓶颈已经不是单纯 clash，也不是 replay 不够，而是 **target-relative interface localization**：模型能生成一些 target contact，也能通过 safe-select 避免大部分严重冲突，但 contact 往往落在错误 target 表面或跨状态不一致。

Stage24 因此加入 target-only 的 interface field：只从多状态 target 构象、hotspot、mask 和 target encoder token 预测 target-side interface site，不输入 source pose、真实 binder 坐标或真实 binder 序列。科学目标仍是：**target-only 输入下生成一个共享 binder 序列，同时为每个 target state 生成合理 complex geometry**。

## 2. 代码改动

### 2.1 `local_latents_transformer_multistate.py`

文件：`src/proteinfoundation/nn/local_latents_transformer_multistate.py`

新增：

- `target_interface_site_head`：从 `state_target_tokens [B,K,Nt,D]` 预测 `target_interface_logits_states [B,K,Nt]`。
- `target_interface_guidance_scale`：可学习 guidance scale，初值 `0.05`。
- `_stage24_target_interface_field()`：根据 target-only token 和 hotspot prior 计算 target-side interface logits 与 differentiable target interface center。
- `_stage24_apply_interface_guidance()`：可选地对 `bb_ca_states` 做小幅、刚体式 target-only translation guidance，最大 shift 由 `stage24_interface_guidance_max_shift_nm` 控制。

关键约束：

- 该 field 不读取 source pose、真实 binder 坐标、真实 binder residue type。
- guidance 只做整体平移式偏置，不做 per-residue deform，避免把 binder backbone 拉坏。
- guidance 默认需要显式 `stage24_enable_interface_guidance=True` 才启用。

### 2.2 `multistate_loss.py`

文件：`src/proteinfoundation/flow_matching/multistate_loss.py`

新增 loss：

- `multistate_target_interface_site_justlog`：从 `interface_contact_labels [B,K,Nt,Nb]` 聚合 target-side binary interface labels，对 `target_interface_logits_states [B,K,Nt]` 做 BCE。
- `multistate_target_interface_center_justlog`：用预测 target interface probability center 对齐真实 target-side contact center。
- 每个 state 也记录：
  - `multistate_state_k_target_interface_site_justlog`
  - `multistate_state_k_target_interface_center_justlog`

新增权重：

- `lambda_target_interface_site`
- `lambda_target_interface_center`

科学意义：

Stage23 的 contact count / shell selector 仍可能选择到“接触数量合适但位置错误”的候选。target-interface site/center loss 直接监督 target 表面哪一片应该被 binder 接触，是 target-only de novo 任务中必须补上的空间定位信号。

### 2.3 训练与采样脚本

修改：

- `scripts/strategy01/stage12_de_novo_multistate_training.py`
- `scripts/strategy01/stage12c_de_novo_smoke.py`
- `scripts/strategy01/stage17_per_sample_multiseed_probe.py`

新增 CLI：

- `--stage24-enable-target-interface-field`
- `--stage24-enable-interface-guidance`
- `--stage24-interface-guidance-scale`
- `--stage24-interface-guidance-max-shift-nm`
- `--stage24-interface-hotspot-prior`
- `--lambda-target-interface-site`
- `--lambda-target-interface-center`

新增 Slurm：

- `scripts/strategy01/slurm/stage24_target_interface_field_probe.sbatch`
- `scripts/strategy01/slurm/stage24_sampling_from_probe.sbatch`

## 3. 测试与错误修复

### 3.1 静态编译

覆盖文件：

```bash
python -m py_compile \
  src/proteinfoundation/nn/local_latents_transformer_multistate.py \
  src/proteinfoundation/flow_matching/multistate_loss.py \
  scripts/strategy01/stage12_de_novo_multistate_training.py \
  scripts/strategy01/stage12c_de_novo_smoke.py \
  scripts/strategy01/stage17_per_sample_multiseed_probe.py
```

结果：通过。

### 3.2 CPU dry-run

输出：`reports/strategy01/probes/stage24_target_interface_field_cpu_dryrun_results.json`

结果：`dry_run_passed`。

说明：dry-run 只验证数据/config，不执行 forward loss，因此不能证明新 loss 接通。

### 3.3 CPU forward probe

输出：`reports/strategy01/probes/stage24_target_interface_field_cpu_forward_results.json`

结果：`passed`。

确认新增 loss 出现：

- `multistate_target_interface_site_justlog = 0.7411`
- `multistate_target_interface_center_justlog = 0.9168`

说明：target-interface field head 和 loss 已真实接入 forward/loss。

### 3.4 CPU sampling smoke 报错并修复

首次报错：

```text
AttributeError: 'Namespace' object has no attribute 'lambda_target_interface_site'
```

根因：`stage12c_de_novo_smoke.py` 调用 `build_model_stack()` 的 `stack_args` 没有补 `lambda_target_interface_site/center`。

修复：给 `stage12c` 和 `stage17` 的 `stack_args` 补齐这两个字段，并加入 CLI。

第二次报错：

```text
RuntimeError: The size of tensor a (2) must match the size of tensor b (3)
```

根因：`_stage24_apply_interface_guidance()` 中 binder center 分母 `denom[..., None]` 多加了一维，导致 `[B,K,3] / [B,K,1,1]` 广播错误。

修复：改为 `[B,K,3] / [B,K,1]`。

修复后：`reports/strategy01/probes/stage24_target_interface_guidance_cpu_smoke.json` 通过。

### 3.5 GPU Stage24 probe

训练 job：`2049136`

训练输出：

- `reports/strategy01/probes/stage24_target_interface_field_probe_results.json`
- checkpoint：`ckpts/stage12_de_novo_multistate/runs/stage24_target_interface_field_probe/stage12_pilot_final.pt`

训练部分完成；后续采样首次失败，因为 `stage17` 仍缺 `lambda_target_interface_site/center` 的 `stack_args`。修复后只重跑采样/评估，不重复训练。

采样 job：`2049142`

采样/评估输出：

- `reports/strategy01/probes/stage24_target_interface_field_val12_candidates.json`
- `reports/strategy01/probes/stage24_target_interface_field_val12_geometry.json`
- `reports/strategy01/probes/stage24_target_interface_field_val12_selection_sweep.json`

## 4. Warning 影响判断

### 4.1 `CCD_MIRROR_PATH / PDB_MIRROR_PATH`

当前 Stage24 训练和 tensor/PDB CA geometry benchmark 不依赖这些 mirror，因此不影响本阶段结论。

### 4.2 optional CA/residue feature disabled

这是 target-only no-leak 契约下的预期 warning。真实 binder CA / residue type 不能作为 de novo 输入，否则任务会退化成 repair/inverse folding。

### 4.3 真实影响结果的报错

本阶段影响结果的报错有两个：`stack_args` 缺字段、guidance center shape bug。均已修复并重跑对应 probe。

## 5. Stage24 结果

### 5.1 Stage23 val12 baseline

文件：`reports/strategy01/probes/stage23_safe_replay_native_coupling_val12_geometry.json`

| 指标 | Stage23 val12 |
|---|---:|
| mean_contact_f1 | 0.0439 |
| worst_contact_f1 | 0.0234 |
| mean_direct_rmsd_A | 44.39 |
| mean_aligned_rmsd_A | 28.27 |
| severe_clash_rate | 0.0000 |
| contact_persistence | 0.2889 |
| mean_generated_contact_count | 73.49 |

### 5.2 Stage24 val12 current hard gate

文件：`reports/strategy01/probes/stage24_target_interface_field_val12_geometry.json`

| 指标 | Stage24 val12 |
|---|---:|
| mean_contact_f1 | 0.0506 |
| worst_contact_f1 | 0.0328 |
| mean_direct_rmsd_A | 47.01 |
| mean_aligned_rmsd_A | 30.13 |
| severe_clash_rate | 0.0000 |
| contact_persistence | 0.1452 |
| mean_generated_contact_count | 61.97 |

### 5.3 Stage24 selection sweep

文件：`reports/strategy01/probes/stage24_target_interface_field_val12_selection_sweep.json`

| selector | mean_contact_f1 | worst_contact_f1 | mean_direct_rmsd_A | contact_persistence | safe_rate | shared_identity_posthoc |
|---|---:|---:|---:|---:|---:|---:|
| current_hard_gate | 0.0506 | 0.0328 | 47.01 | 0.1452 | 1.0000 | 0.2466 |
| persistence_contact | 0.0443 | 0.0289 | 46.63 | 0.2009 | 1.0000 | 0.2391 |
| contact_shell | 0.0452 | 0.0289 | 47.46 | 0.1716 | 1.0000 | 0.2391 |

## 6. 结论

Stage24 的 target-interface field 是有用的，但不够。

正向信号：

- mean_contact_f1 从 Stage23 val12 的 `0.0439` 提升到 `0.0506`。
- worst_contact_f1 从 `0.0234` 提升到 `0.0328`。
- severe_clash_rate 仍保持 `0.0`。

负面信号：

- mean_direct_rmsd_A 从 `44.39` 变差到 `47.01`。
- contact_persistence 从 `0.2889` 降到 `0.1452`。
- shared sequence identity 仍只有约 `0.24`，没有达到共享 binder 序列质量目标。

科学解释：

target-interface center guidance 能把 binder 拉向更可能形成 contact 的 target 表面，因此 contact-F1 上升。但它只约束“靠近哪片区域”，没有约束 binder 相对 target 的 orientation、contact distribution、跨状态 persistent anchor。因此它可能牺牲跨状态一致性和整体 pose RMSD。

## 7. 下一步：Stage25 orientation/contact-distribution guidance

不建议继续只加 Stage24 训练步数。下一步应解决更细的界面定位问题：

1. **target-residue contact distribution loss**
   - 不只预测 target-side interface binary site，还预测每个 target residue 的 expected binder contact count 或 contact density。
   - 目标是避免只靠 center 对齐导致 contact 分布塌缩。

2. **binder center + orientation frame guidance**
   - 从 target interface patch 预测 local surface frame：normal/tangent 或 PCA frame。
   - bb_ca flow 不只平移到 center，还要约束 binder 主轴/接触面朝向 target patch。

3. **persistent-anchor distribution consistency**
   - 对跨状态共同出现的 target-side anchors 加强一致性，而不是只优化单 state contact。
   - 这直接服务于“一个共享 binder 序列兼容多个功能构象”。

4. **sampling selector 改为多目标 hard gate**
   - severe clash 继续 hard fail。
   - contact-F1 提升必须同时满足 persistence 不显著下降，否则不判定为科学改进。

Stage24 应保留为 Stage25 起点，但不能作为成功模型。
