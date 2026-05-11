# Strategy01 Stage22-23 native flow coupling 与 safe replay 阶段报告

## 1. 阶段目标

本阶段继续围绕 Strategy01 的核心科学目标：**只输入多状态 target 构象，模型从零生成一个共享 binder 序列，并为每个 target state 生成合理的 state-specific target-binder complex geometry**。

前一阶段已经发现，Stage12/Stage13 虽然有 `shared_seq_logits` 和 state-wise 输出，但自由 de novo sampling 仍然差，说明问题不只在 loss 数值，而在模型内部是否真的让多条 state-specific flow 共同约束同一条共享序列和界面几何。

本阶段具体做两件事：

- 修正 native multistate flow 路径中 shared sequence / cross-state coupling 没有反馈到 `bb_ca_states` 和 `local_latents_states` 的实现缺口。
- 把 replay curriculum 从过强 hard replay 改成 safe replay，避免模型为了无 clash 把 binder 推离真实界面。

## 2. 关键代码改动

### 2.1 `local_latents_transformer_multistate.py`

文件：`src/proteinfoundation/nn/local_latents_transformer_multistate.py`

改动：

- 新增 `native_flow_coupling_scale`，默认初值 `0.05`。
- 在 Stage13 native-state path 中，当输入字段 `stage22_enable_native_flow_coupling=True` 时启用 coupled update。
- 对同一 binder residue index 的 K 个 state token 做 cross-state attention / fusion。
- 由融合后的 shared token 预测 `shared_seq_logits`，再把 soft amino-acid distribution 投影回 state token。
- 用耦合后的 state token 通过已有 `ca_linear` 和 `local_latents_linear` 产生 residual delta，反馈到 `bb_ca_states` 和 `local_latents_states`。
- 更新 `state_seq_logits`，并在 `arch_debug` 中记录：
  - `stage22_native_flow_coupling_enabled`
  - `stage22_native_flow_coupling_scale`
  - `stage22_native_seq_feedback_norm_mean`

科学意义：

原来的 native path 实际上把 K 个 state 展平成 batch，先走原始 Complexa denoiser，之后再读出 shared sequence。这样 shared sequence 是“后验读出”，没有参与 `bb_ca/z` 的 denoising。这个实现不符合 Stage12 目标，因为它仍可能退化成 K 个独立 binder flow 后再投票。本次改动让 shared sequence signal 进入 state-specific flow output，使多状态共享序列约束可以反向影响几何和 latent。

### 2.2 `stage12_de_novo_multistate_training.py`

文件：`scripts/strategy01/stage12_de_novo_multistate_training.py`

改动：

- 新增 CLI 参数 `--stage22-enable-native-flow-coupling`。
- 在 teacher-forced batch、replay batch 和 rollout batch 中传递该 flag。
- 修复 replay builder 函数签名，保证 replay 条件下也能启用 native flow coupling。
- 保留 `de_novo_multistate_mode=True` 的 no-leak contract：真实 binder sequence 只能进入 loss，不能作为输入特征；source pose / init pose 不能进入 target-only de novo 主线。

科学意义：

如果 replay batch 没有启用同一 coupling 逻辑，训练和采样仍会走不同模型路径。这个修复确保 teacher-forced、replay、sampling 三条路径使用同一 Stage22 coupling contract。

### 2.3 `stage12c_de_novo_smoke.py`

文件：`scripts/strategy01/stage12c_de_novo_smoke.py`

改动：

- 新增 `--stage22-enable-native-flow-coupling`。
- 在 de novo smoke / final rollout 中传递该 flag。

科学意义：

保证 smoke benchmark 真实测试 coupled multistate product-space flow，而不是仍然测试旧的 uncoupled native path。

### 2.4 `stage17_per_sample_multiseed_probe.py`

文件：`scripts/strategy01/stage17_per_sample_multiseed_probe.py`

改动：

- 新增 `--stage22-enable-native-flow-coupling`。
- 在多 seed 候选生成和 selection contract 中记录该 flag。

科学意义：

多候选 safe-select 是目前评估 target-only de novo 输出的重要工具。这个改动保证 Stage23 结果确实来自启用 coupling 的模型，而不是混用旧 sampler。

## 3. 执行与错误修复记录

### 3.1 静态编译

命令覆盖：

```bash
python -m py_compile \
  src/proteinfoundation/nn/local_latents_transformer_multistate.py \
  scripts/strategy01/stage12_de_novo_multistate_training.py \
  scripts/strategy01/stage12c_de_novo_smoke.py \
  scripts/strategy01/stage17_per_sample_multiseed_probe.py
```

结果：通过。

### 3.2 Stage22 第一次 GPU probe 失败

现象：

- Slurm job `2048775` 立即失败。
- 报错：`unrecognized args --stage22-enable-native-flow-coupling --report-json`。

根因：

- training parser 尚未注册 `--stage22-enable-native-flow-coupling`。
- training script 不接受 `--report-json`，报告路径由 `run_name` 推导。

修复：

- 在 training parser 中加入 `--stage22-enable-native-flow-coupling`。
- 修改 sbatch，不再向 training script 传 `--report-json`。

### 3.3 Stage22 第二次 GPU probe 失败

现象：

- Slurm job `2048985` 失败。
- 报错：`build_replay_condition_batch() got unexpected keyword argument 'stage22_enable_native_flow_coupling'`。

根因：

- CLI 和主 batch 已接入新 flag，但 replay builder 函数签名未同步。

修复：

- 补齐 replay builder 参数，并在 replay rollout 中传递 `stage22_enable_native_flow_coupling`。

### 3.4 Stage22 几何评估脚本参数错误

现象：

- training 完成，但 geometry evaluator 初始调用失败。
- 原因是 evaluator 使用 `--output`，不是 `--report-json`。

修复：

- 修正 sbatch / 手动重跑 CPU geometry evaluator。

科学影响：

这些错误属于执行接线错误，不是模型科学假设错误。修复后对应 stage 已重新运行并产出有效 JSON。

## 4. Warning 影响判断

### 4.1 `optional CA/residue feature disabled`

判断：不影响科学目标，属于预期 no-leak 行为。

原因：Stage12 target-only de novo 主线不能把真实 binder optional CA 或 residue type 作为输入，否则任务会退化成 pose repair / inverse folding。

### 4.2 `CCD_MIRROR_PATH / PDB_MIRROR_PATH` warning

判断：当前 Stage22/23 tensor rollout、PDB bridge 和 exact geometry evaluator 不依赖这些 mirror 路径，不影响本阶段结果。

后续如果进入全原子构建、外部结构库解析或更完整 PDB 下载流程，需要重新检查。

## 5. 结果对照

### 5.1 Stage21：强 replay interface/clash 方案失败

文件：`reports/strategy01/probes/stage21_interface_replay_probe_val12_geometry.json`

val12：

| 指标 | 数值 |
|---|---:|
| mean_contact_f1 | 0.0184 |
| worst_contact_f1 | 0.0000 |
| mean_direct_rmsd_A | 49.62 |
| contact_persistence | 0.0699 |
| severe_clash_rate | 0.0000 |

结论：

强 replay/interface/clash 权重虽然消除了 clash，但把 binder 推离真实界面，contact 和 persistence 显著变差。因此不能继续沿“强排斥/强 replay”方向盲目加权。

### 5.2 Stage22：native flow coupling 修复后有必要但不充分

文件：`reports/strategy01/probes/stage22_native_coupling_probe_val12_geometry.json`

val12：

| 指标 | 数值 |
|---|---:|
| mean_contact_f1 | 0.0352 |
| worst_contact_f1 | 0.0051 |
| mean_direct_rmsd_A | 55.10 |
| contact_persistence | 0.1379 |
| severe_clash_rate | 0.0000 |

结论：

Stage22 相比 Stage21 恢复了部分 interface contact，但 replay 仍太 hard，training replay total 在验证侧恶化，说明模型仍被 off-manifold replay 拉坏。

### 5.3 Stage23：safe replay curriculum 有正向信号

文件：`reports/strategy01/probes/stage23_safe_replay_native_coupling_val12_geometry.json`

val12：

| 指标 | 数值 |
|---|---:|
| mean_contact_f1 | 0.0439 |
| worst_contact_f1 | 0.0234 |
| mean_direct_rmsd_A | 44.39 |
| mean_aligned_rmsd_A | 28.27 |
| contact_persistence | 0.2889 |
| severe_clash_rate | 0.0000 |

文件：`reports/strategy01/probes/stage23_safe_replay_native_coupling_val36_geometry.json`

val36：

| 指标 | 数值 |
|---|---:|
| mean_contact_f1 | 0.0448 |
| worst_contact_f1 | 0.0219 |
| mean_direct_rmsd_A | 37.63 |
| worst_direct_rmsd_A | 42.82 |
| mean_aligned_rmsd_A | 23.00 |
| contact_persistence | 0.2175 |
| severe_clash_rate | 0.0185 |
| mean_generated_contact_count | 58.29 |

selection sweep 中 `contact_shell` 在 val36 上进一步得到：

| 指标 | 数值 |
|---|---:|
| mean_contact_f1 | 0.0488 |
| worst_contact_f1 | 0.0256 |
| mean_direct_rmsd_A | 37.91 |
| contact_persistence | 0.2297 |
| safe_rate | 0.9444 |
| shared_identity_posthoc | 0.1909 |

### 5.4 与 Stage19/20 的关系

Stage19/20 是当前比较基准：多候选 safe-select + relief，但没有 Stage22 native flow coupling。

Stage20 `contact_shell` val36：

| 指标 | 数值 |
|---|---:|
| mean_contact_f1 | 0.0436 |
| worst_contact_f1 | 0.0231 |
| mean_direct_rmsd_A | 36.07 |
| contact_persistence | 0.3225 |
| safe_rate | 0.9722 |
| shared_identity_posthoc | 0.2115 |

Stage23 `contact_shell` val36：

| 指标 | 数值 |
|---|---:|
| mean_contact_f1 | 0.0488 |
| worst_contact_f1 | 0.0256 |
| mean_direct_rmsd_A | 37.91 |
| contact_persistence | 0.2297 |
| safe_rate | 0.9444 |
| shared_identity_posthoc | 0.1909 |

判断：

- Stage23 对 exact contact-F1 有小幅正向提升。
- Stage23 的 RMSD、contact persistence、safe_rate、posthoc sequence identity 没有同步提升。
- 因此 Stage23 是“方向有用但不达标”，不能判定已实现科学目标。

## 6. 对科学目标的实现程度判断

当前代码比 Stage12B 更接近科学目标，原因是：

- `shared_seq_logits` 不再只是后验读出，而是可以通过 Stage22 coupling 反馈到 state-specific `bb_ca/z` 输出。
- replay curriculum 不再把模型直接推入过强 off-manifold hard replay，而是通过 safe replay 控制 replay 距离。
- 输出和评估仍使用 `bb_ca_states[k] / local_latents_states[k]`，没有把 weighted-average legacy 输出当主科学结果。

但仍没有达到科学目标，主要证据是：

- val36 `shared_identity_posthoc` 仍约 `0.19`，共享序列质量不足。
- contact-F1 虽提升，但绝对值仍低，说明生成的界面位置/接触图仍与 exact state-specific complex 差距大。
- direct RMSD 仍在 `37-38 Å` 量级，说明 target-relative pose localization 仍不可靠。
- contact persistence 不稳定，说明同一共享 binder 在多个 target state 的界面一致性还没学好。

## 7. 当前瓶颈

本阶段结果说明，主要瓶颈已经不是单纯 clash，也不是仅仅缺少 replay，而是：

**target-relative interface localization 不够。**

模型可以生成不严重 clash 的 binder，也能产生一些 target contacts，但这些 contacts 经常落在错误位置或状态间不一致。继续只加 replay 步数或单纯加强 clash/contact loss，很可能重复 Stage21/Stage22 的问题：要么推离界面，要么产生错误接触。

## 8. 下一步建议：Stage24 target-interface field / hotspot-conditioned bb_ca velocity

下一阶段不建议继续盲目增加 Stage23 训练步数。更合理的方向是增加 target-only、非泄漏的 interface localization 信号：

1. **target interface field head**
   - 从 target ensemble encoder 预测每个 target residue 的 binder-contact probability。
   - label 来自 exact/hybrid 复合物的 target-side interface residues。
   - inference 时只用 target 构象和 hotspot，不用 source pose 或真实 binder。

2. **bb_ca velocity interface guidance**
   - 在 denoising 中给 `bb_ca_states` 加一个 target-conditioned center/contact field bias。
   - 目标是让 binder 先靠近正确 target surface shell，再优化细节 contact。
   - 这比事后 safe-select 更符合 flow matching 生成过程。

3. **target-contact distribution matching loss**
   - 不只看 generated contact count，也要求 target-side contact distribution 与 exact/hybrid interface field 对齐。
   - 避免模型生成“很多 contact 但在错位置”的伪优解。

4. **state-specific binder center prior**
   - 对每个 target state 预测 binder center / interface shell。
   - 作为 full de novo bb_ca flow 的空间定位监督，替代 source pose transfer。

进入 Stage24 前的判据：保留 Stage23 checkpoint 作为当前最好 target-only coupled-flow 起点，但不要把它作为成功模型。

## 9. 文件与复现路径

主要代码文件：

- `src/proteinfoundation/nn/local_latents_transformer_multistate.py`
- `scripts/strategy01/stage12_de_novo_multistate_training.py`
- `scripts/strategy01/stage12c_de_novo_smoke.py`
- `scripts/strategy01/stage17_per_sample_multiseed_probe.py`

Slurm 脚本：

- `scripts/strategy01/slurm/stage21_interface_replay_probe.sbatch`
- `scripts/strategy01/slurm/stage22_native_coupling_probe.sbatch`
- `scripts/strategy01/slurm/stage23_safe_replay_native_coupling.sbatch`
- `scripts/strategy01/slurm/stage23_val36_sampling.sbatch`

关键输出：

- `reports/strategy01/probes/stage21_interface_replay_probe_results.json`
- `reports/strategy01/probes/stage21_interface_replay_probe_val12_geometry.json`
- `reports/strategy01/probes/stage22_native_coupling_probe_results.json`
- `reports/strategy01/probes/stage22_native_coupling_probe_val12_geometry.json`
- `reports/strategy01/probes/stage23_safe_replay_native_coupling_results.json`
- `reports/strategy01/probes/stage23_safe_replay_native_coupling_val12_geometry.json`
- `reports/strategy01/probes/stage23_safe_replay_native_coupling_val36_geometry.json`
- `reports/strategy01/probes/stage23_safe_replay_native_coupling_val36_selection_sweep.json`

训练产物 checkpoint 位于：

- `ckpts/stage12_de_novo_multistate/runs/stage23_safe_replay_native_coupling/stage12_pilot_final.pt`

注意：checkpoint 和大体积 PDB/结果目录不进入 git。
