# Strategy01 Stage09D：几何有效 sampling-time guidance 评估报告

## 阶段目标
Stage09C 已证明 `source_state0` interface anchors 能把多状态 B1 从无约束状态拉向真实界面附近，但 `clash_rate=0.7431` 仍然过高。本阶段目标是尝试把 `anchor_contact + clash_repulsion + interface_shell + safe_accept` 接入更严格的 state-specific sampling，让模型在保持 shared binder / state-specific pose 的同时减少 target-binder 穿模。

核心科学标准没有变化：一个共享 binder 序列必须兼容 target 的真实功能构象变化，并且每个 state 都应有合理复合物几何。单纯降低 clash 但丢失真实界面，不算成功。

## 代码改动
- 修改 `scripts/strategy01/stage09_guided_state_specific_sampling.py`。
- 新增 `set_sampling_optional_features(...)`：允许把当前 `bb_ca` 作为 `ca_coors_nm` 传给 baseline optional CA feature factory。
- 明确不启用 `residue_type` optional feature：不能用 `binder_seq_shared`，否则会把 V_exact 真实 binder 序列泄漏给 B1；后续只能用 generated sequence self-conditioning。
- 新增 `clash_loss_mode`：
  - `mean`：保留 Stage09C legacy 行为，避免旧脚本重跑时语义漂移。
  - `active`：只在实际 violation 上归一化，并加入 max-clash 项，避免原先全 pair mean 把少数严重 clash 稀释掉。
- 新增 `overcontact_loss`：用 soft contact count 约束 generated contact 不要远超 reference contact。
- 新增 `clamp_guidance_displacement(...)`：限制每次 guidance 对单个 residue 的最大位移，避免 repulsion 把 binder 推飞。
- 增强 `safe_accept`：
  - hard min distance 未改善则拒绝。
  - contact-F1 明显下降则拒绝。
  - anchor loss 变差且没有足够 F1 收益则拒绝。

## 运行的 smoke ablation
所有 smoke 都使用 8 个样本、24 个 state，来自同一 V_exact 子集。

| 方法 | contact-F1 | interface RMSD (Å) | aligned CA RMSD (Å) | clash rate | contact persistence | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Stage09C source-state0 | 0.1388 | 24.55 | 10.73 | 0.6667 | 0.4859 | 当前可用基准 |
| Stage09D hard-safe + CA | 0.1274 | 84.38 | 73.30 | 0.0417 | 0.4055 | clash 降了，但 binder 被推飞，不可用 |
| Stage09D hard-safe no-CA | 0.1529 | 121.36 | 110.56 | 0.0417 | 0.2715 | contact 表面好，但 pose 严重失真，不可用 |
| Stage09D balanced | 0.0898 | 23.32 | 10.07 | 0.2083 | 0.3972 | 几何不过远，但真实界面损失太大 |
| Stage09D contact-preserve | 0.0799 | 23.38 | 10.12 | 0.2500 | 0.3317 | contact 继续下降，不可用 |
| Stage09D strict-contact | 0.0765 | 23.58 | 10.15 | 0.3333 | 0.3264 | 严格拒绝 F1 drop 仍不能保界面 |
| Stage09D late-soft + CA | 0.0631 | 24.24 | 10.20 | 0.5417 | 0.3011 | 低扰动也损失界面 |
| Stage09D late-soft no-CA | 0.1060 | 25.53 | 10.76 | 0.7500 | 0.3463 | 接近原条件但 clash 没改善 |

## 关键判断
本阶段没有进入 full run。原因是没有任何 Stage09D smoke 变体同时满足“clash 下降”和“contact-F1/contact persistence 不塌”。继续跑 full 只会消耗 A100 并产生科学上不可接受的结果。

最重要的失败模式是：单轨迹内的强 repulsion 会把 binder 从 target core 推开，但也会破坏真实 interface contacts。换句话说，当前问题不是简单把 `clash_weight` 调大就能解决。多状态 binder 的界面是一个窄窗口，不能只做 attraction 或 repulsion 的单目标优化。

## Warning / Error 评估
- `OptionalCaCoorsNanometers*` warning：当启用 CA feature 后消失，但 CA feature 使用当前 noisy/generated CA 作为输入，会改变模型训练时的条件分布，并在 smoke 中损害 contact。因此本阶段不接受“为了消 warning 而启用 CA feature”作为默认方案。
- `OptionalResidueTypeSeqFeat` warning：保留并记录。原因是 reference `binder_seq_shared` 不能作为生成输入，否则泄漏真实 binder 序列。该 warning 对公平性是必要代价，后续应改成 generated-sequence self-conditioning，而不是用标签序列。
- `torch nested_tensor` warning：性能提示，不影响数值结果。
- `CCD_MIRROR_PATH/PDB_MIRROR_PATH` 环境提示：依赖导入提示，Stage09D 没有调用相关 mirror 功能，不影响当前结果。
- Lightning autoencoder missing keys 长 warning：未造成运行失败；本阶段只用 Strategy01 checkpoint 生成 state-specific CA，报告保留为后续 checkpoint 加载清理项。
- 本地 PowerShell 管道命令曾触发 quote 错误；已改回临时脚本远程执行。

## 对科学目标的影响
Stage09D 的结果说明：在当前模型和采样器里，单条 trajectory 上强行加入 clash repulsion 不是正确主线。它能降低 clash，但会牺牲共享 binder 在多个 state 的真实界面兼容性。这与 Strategy01 目标冲突，因此不能把这些 hard-safe 结果作为改进版。

Stage09C 的 source-state0 anchor 仍是当前更可靠的主线：它保持了界面信号，并且从 oracle anchor 退到了更接近生产场景的 source interface anchor。

## 下一步建议：Stage09E
下一步不再继续加大单轨迹 repulsion，而应改成 candidate-level selection / relax：

1. 多 seed 采样：每个 sample 生成 `N=4-8` 个 state-specific candidates，使用 Stage09C/source-state0 guidance 保持界面。
2. safe-select 排序：按 `no severe clash -> contact-F1 -> contact persistence -> RMSD` 选择，而不是在单轨迹内强制 repulsion。
3. 可选轻量 relax：对候选做局部 rigid-body clash relief 或 Rosetta/FastRelax 风格后处理；只接受 contact 不下降且 clash 改善的结果。
4. residue self-conditioning：后续若要消除 residue_type warning，只能使用上一轮 generated sequence logits/tokens，不能使用 reference binder sequence。
5. 完成 full benchmark：只有 smoke 中同时改善 clash 和 contact 后，再跑 48-sample full。

## 关键输出
- `reports/strategy01/probes/stage09d_smoke_comparison_summary.json`
- `reports/strategy01/probes/stage09d_hard_safe_smoke_summary.json`
- `reports/strategy01/probes/stage09d_balanced_smoke_summary.json`
- `reports/strategy01/probes/stage09d_contact_preserve_smoke_summary.json`
- `reports/strategy01/probes/stage09d_strict_contact_smoke_summary.json`
- `reports/strategy01/probes/stage09d_late_soft_smoke_summary.json`
- `reports/strategy01/probes/stage09d_no_ca_late_soft_smoke_summary.json`
