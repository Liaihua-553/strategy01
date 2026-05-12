# Strategy01 Stage25A-C 报告：可训练目标界面场、interface-shell 负结果与 shared-sequence 诊断

## 阶段目标

本阶段继续围绕 Strategy01 的核心科学目标：只输入多状态 target 构象，模型从零生成一个共享 binder 序列，并为每个 target state 生成合理的 state-specific target-binder complex geometry。

前一阶段 Stage24 发现 target-interface field 有轻微信号，但代码审计显示新加入的 `target_interface_site_head` 和 `target_interface_guidance_scale` 没有进入训练可更新参数。因此 Stage25A 先修正 trainable scope；Stage25B 尝试更弱的 interface-shell/coverage 监督；Stage25C 补充 shared-head 与 final latent 序列诊断，判断失败发生在序列 head 还是 AE/local latent 流形。

## 代码改动

### 1. Stage25A：修复 target-interface field 未训练问题

修改文件：

- `scripts/strategy01/stage12_de_novo_multistate_training.py`
- `scripts/strategy01/stage11_flow_sequence_training.py`
- `src/proteinfoundation/flow_matching/multistate_loss.py`

具体改动：

- 将 `target_interface_site_head` 和 `target_interface_guidance_scale` 加入 Stage12/Stage11 的 trainable prefix。
- 在 loss 配置中加入 `target_interface_positive_weight`。
- 在 target-interface site BCE 中对 positive target-interface residues 加权，避免稀疏界面标签被大量非界面 residue 淹没。

科学意义：Stage24 的 target-interface field 只有在真正可训练后才可能学到 target-only 的界面定位先验；否则 sampling guidance 只是随机或弱初始化场。

### 2. Stage25B：新增 interface-shell/coverage loss

修改文件：

- `src/proteinfoundation/flow_matching/multistate_loss.py`
- `scripts/strategy01/stage12_de_novo_multistate_training.py`

新增字段：

- `lambda_interface_shell`
- `interface_shell_target_weight`
- `interface_shell_binder_weight`
- `interface_shell_positive_weight`
- `multistate_interface_shell_justlog`
- `multistate_state_{k}_interface_shell_justlog`

设计目的：pairwise contact BCE 对 de novo 多状态生成过于严格，因为同一个 shared binder 在不同状态中可能保留 target-side anchors，但具体 binder residue-contact assignment 有小幅变化。interface-shell loss 只监督 target-side 和 binder-side interface coverage，允许局部接触重排。

### 3. Stage25C：补充 shared-head vs final_x AE identity 诊断

修改文件：

- `scripts/strategy01/stage17_per_sample_multiseed_probe.py`

新增候选字段：

- `shared_head_identity_mean`
- `ae_pred_state_identity_mean`
- `shared_identity_mean_posthoc`
- `ae_state_identity_mean_posthoc`

科学意义：之前只记录 `final_x` 经过 AE 解码后的 sequence identity，无法判断是 explicit shared sequence head 失败，还是 integrated local_latents 偏离 AE sequence manifold。现在两者分开记录。

## 验证结果

### 静态与 no-leak 检查

- `py_compile` 通过：`multistate_loss.py`、`stage12_de_novo_multistate_training.py`、`stage11_flow_sequence_training.py`、`stage17_per_sample_multiseed_probe.py`。
- `stage25a_trainable_contract.json` 确认：`target_interface_site_head_trainable=true`，`target_interface_guidance_scale_trainable=true`。
- `stage25c_no_leak_invariance_probe.json` 通过：固定 target 与 `x_t_states` 后，随机改 `binder_seq_shared`、`x_1_states`、`interface labels`，模型输出最大差异全部为 `0.0`。这说明真实 binder sequence 和 x1 标签没有进入 model forward，只进入 loss/diagnostic。

### Warning/error 评估

- `CCD_MIRROR_PATH/PDB_MIRROR_PATH not set`：当前 Stage25 不调用依赖 atomworks 镜像的功能，只影响相关外部数据库函数，不影响本阶段训练/采样。
- `OptionalCaCoorsNanometers* disabled` 与 `OptionalResidueType* disabled`：这是 de novo no-leak 契约下刻意关闭真实 binder CA/residue-type optional features 的结果，不影响科学结论。
- 未发现 `Traceback`、`OOM`、`NaN` 或会改变结果有效性的运行错误。

## 指标对比

### Stage25A：可训练 target-interface field

val12 current hard gate：

- no-guidance：mean contact-F1 `0.0369`，worst contact-F1 `0.0148`，RMSD `46.85 A`，persistence `0.3143`。
- guidance：mean contact-F1 `0.0467`，worst contact-F1 `0.0208`，RMSD `44.24 A`，persistence `0.3314`。

val12 selection sweep 最好项：

- guidance + contact_shell：mean contact-F1 `0.0568`，worst contact-F1 `0.0302`，RMSD `46.28 A`，persistence `0.3219`。
- guidance + persistence_contact：mean contact-F1 `0.0539`，worst contact-F1 `0.0319`，RMSD `46.82 A`，persistence `0.3541`。

结论：Stage25A 有正信号，但仍远未达到科学目标。它能稍微改善接触和 safe selection，但不能解决 shared sequence 与 state-specific geometry 的根本耦合问题。

### Stage25B：interface-shell loss

val12 current hard gate：

- no-guidance：mean contact-F1 `0.0397`，worst contact-F1 `0.0233`，RMSD `39.53 A`，persistence `0.1499`。
- guidance：mean contact-F1 `0.0336`，worst contact-F1 `0.0159`，RMSD `39.74 A`，persistence `0.1567`。

训练结果：

- overfit1 final total 反而上升，drop `-27.4%`。
- pilot total 只下降约 `9.2%`。
- replay final total 从 `2.69` 升到 `4.90`。
- validation shared identity `0.75`，AE state identity `0.667`，但自由采样候选 identity 下降。

结论：interface-shell loss 让 binder 更接近 target，因此 RMSD 降低，但破坏了 contact persistence 和 sequence quality。这个 loss 不能作为默认主线继续加权；最多保留为可选诊断项，默认 `lambda=0`。

### Stage25C：shared-head 与 final_x AE identity

val4 诊断：

- Stage25A proxy-selected final_x identity `0.2386`，shared-head identity `0.2500`。
- Stage25B proxy-selected final_x identity `0.1818`，shared-head identity `0.1932`。

结论：自由 rollout 下 explicit shared sequence head 本身也低，不是单纯 AE decoder 或 final local latent 解码失败。因此下一步不能只修 AE latent repair；必须让 shared sequence head 在模型自产生的 on-policy `x_t_states` 条件下学习。

## 当前科学判断

本阶段没有达到最终科学目标。证据支持以下判断：

1. Stage24 的一部分问题是代码问题：target-interface head 未进入 trainable scope，Stage25A 已修复。
2. 目标界面中心或壳层监督只能提供粗定位，不能学到跨状态共享 anchor/contact graph。
3. Stage25B 证明继续加强粗粒度 interface coverage 会损害 persistence 和 shared sequence，不应继续沿这个方向扩大训练。
4. 当前最主要瓶颈是训练-采样分布错配：teacher-forced 验证 identity 可高，但自由 rollout 下 shared head 与 final_x AE identity 都低。
5. no-leak probe 已确认失败不是标签泄漏造成的伪结论，而是真正 target-only generation 还没学好。

## 下一阶段建议：Stage26 sequence-only on-policy replay

下一步不应继续加 interface loss，也不应直接扩大数据规模。建议进入 Stage26：

- 从 Stage25A checkpoint 出发，而不是 Stage25B。
- 冻结几何 flow 主干和 bb/local latent readout。
- 只训练 shared sequence head、state sequence head、cross-state sequence fusion、sequence feedback 相关模块。
- replay cache 使用更接近真实自由采样的 on-policy `x_t_states`，但 loss 只主要作用于 sequence CE、state-shared KL、AE-decoded CE，不继续强推几何。
- 验收指标先看 val4/val12：shared-head identity 是否明显高于 `0.25`，同时 severe clash/contact persistence 不退化。
- 如果 shared head 提升但 final_x AE 不提升，再回到 latent repair；如果 shared head 仍不提升，说明 target-only data/task underdetermined，需要加强 target hotspot/pocket conditioning 或引入更高质量 exact/hybrid 数据。

## 复现路径

关键输出：

- `reports/strategy01/probes/stage25a_trainable_contract.json`
- `reports/strategy01/probes/stage25a_trainable_interface_field_results.json`
- `reports/strategy01/probes/stage25a_trainable_interface_field_val12_guidance_geometry.json`
- `reports/strategy01/probes/stage25b_interface_shell_results.json`
- `reports/strategy01/probes/stage25b_interface_shell_val12_guidance_geometry.json`
- `reports/strategy01/probes/stage25c_identity_gap_stage25a_trainable_interface_field_val4.json`
- `reports/strategy01/probes/stage25c_identity_gap_stage25b_interface_shell_val4.json`
- `reports/strategy01/probes/stage25c_no_leak_invariance_probe.json`

关键 Slurm 脚本：

- `scripts/strategy01/slurm/stage25a_trainable_interface_field.sbatch`
- `scripts/strategy01/slurm/stage25b_interface_shell.sbatch`
- `scripts/strategy01/slurm/stage25c_identity_gap.sbatch`

结论：Stage25A 保留为当前较优路线；Stage25B 是有价值的负结果；Stage26 应修 shared sequence on-policy 学习，而不是继续拉近 target shell。