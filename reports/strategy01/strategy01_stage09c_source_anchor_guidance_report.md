# Strategy01 Stage09C：source-state0 interface anchor 接入 state-specific sampling 报告

## 阶段目标
本阶段接着 Stage09B 推进一个更接近真实生产条件的 sampling-time interface guidance。Stage09B 使用的是 per-state exact anchor，科学上只能作为诊断上限，因为真实设计时通常不可能提前知道每个状态的精确目标复合物界面。本阶段新增 `anchor_source=source_state0`：只从第 0 个已知/source complex 提取 target-binder interface anchors，再把这些 anchor 转移到其它 target state 上，检验“一个已知界面能否约束共享 binder 兼容多个功能构象”。

这一步直接服务 Strategy01 的核心科学目标：输出一个共享 binder 序列，同时给每个 target state 生成合理的 state-specific complex geometry；不再用多个 state 坐标平均后的“四不像”结构作为主评估对象。

## 代码改动
- 修改 `scripts/strategy01/stage09_guided_state_specific_sampling.py`：新增 `--anchor-source {oracle,source_state0}` 参数。
- 新增 `_state_target_and_binder_ca_nm(...)`：从每个 state 的 exact/source complex 中读取 target CA、binder CA、binder residue index 和 binder mask。
- 扩展 `build_guidance_labels(...)`：
  - `oracle`：沿用每个 state 自己的 exact contacts，作为诊断上限。
  - `source_state0`：只用 state0 的 contact pairs 和 binder pose 作为 source anchors；对 state k 使用 target state k 的 target CA，同时保持 source binder anchor index 与 source contact distances。
- 新增 SLURM 脚本：
  - `scripts/strategy01/stage09c_source_smoke.sbatch`
  - `scripts/strategy01/stage09c_source_full.sbatch`

## 关键结果
| 方法 | samples | states | contact-F1 mean | worst/interface RMSD mean (Å) | contact persistence | clash rate |
|---|---:|---:|---:|---:|---:|---:|
| Stage08B unguided B1 | 48 | 144 | 0.1171 | 26.25 | 0.4988 | 0.7847 |
| Stage09B oracle exact anchors | 48 | 144 | 0.1622 | 24.63 | 0.4753 | 0.7431 |
| Stage09C source-state0 anchors | 48 | 144 | 0.1632 | 24.64 | 0.5229 | 0.7431 |

相对 Stage08B unguided，Stage09C source-state0 anchors 的变化：
- contact-F1 mean：`0.1171 -> 0.1632`，绝对提升 `0.0461`，相对提升 `39.4%`。
- worst/interface RMSD mean：`26.25 Å -> 24.64 Å`，下降 `1.61 Å`，相对变化 `-6.1%`。
- clash rate：`0.7847 -> 0.7431`，下降 `0.0417`，相对变化 `-5.3%`。
- contact persistence：`0.4988 -> 0.5229`，绝对提升 `0.0242`。

和 Stage09B oracle anchors 相比，source-state0 anchors 几乎没有损失：contact-F1 还略高 `0.0009`，RMSD 只高 `0.01 Å`，clash rate 持平。这说明当前 exact 验证集里，很多多状态样本存在可由 source interface 转移的保守 anchor 逻辑；这是比 per-state oracle 更接近可落地的数据/采样假设。

## Warning / Error 影响评估
- Stage09C smoke/full 日志没有 `Traceback`、`OOM`、`NaN` 或 Boltz 失败。
- `torch.nn.modules.transformer` 的 nested tensor warning 属于性能提示，不改变数值结果；不影响本阶段科学结论。
- `OptionalCaCoorsNanometersSeqFeat`、`OptionalResidueTypeSeqFeat`、`OptionalCaCoorsNanometersPairwiseDistancesPairFeat` 返回零的 warning 需要重视：这不会破坏 Stage08B/09B/09C 的公平比较，因为三者处在同一 fallback 条件下；但它可能限制绝对生成质量。下一阶段应明确二选一：要么把 multistate sampling batch 中的 optional CA/res type 字段接上，要么在 config 中关闭这些 feature factories，避免模型以为存在有效额外条件。
- 早期 `stage09_guided_pair` 出现过 `--seed` 参数不识别错误，这是已解决的 CLI 接线错误；最终 Stage09B/09C full 结果没有使用该失败作业。
- 旧 Stage07 的 Boltz PDB 缺失、checkpoint 副本缺失错误属于历史阶段失败，已经被后续 direct exact geometry benchmark 路线替代；不影响本阶段 source-state0 结论。

## 科学解释
Stage09C 的意义不是“把指标调高一点”，而是把 Strategy01 从 oracle 诊断推进到可生产假设：如果某个 target-binder pair 至少有一个已知/source interface，那么模型可以用该界面中的 persistent anchors 来引导其它 target state 的 state-specific flow trajectory。这个设定符合多构象 binder 的生物学直觉：共享 binder 不需要在每个状态复制完全相同的全界面，但需要保留一组跨状态可转移的 anchor contacts，同时允许非 anchor 区域发生局部重排。

当前结果支持这个方向：source-state0 anchors 在 48 个样本、144 个 state 上达到与 oracle anchors 接近的效果，说明“从一个已知界面提取 anchors，再跨 state 引导采样”是有效的下一阶段主线。

## 仍然存在的问题
- `clash_rate=0.7431` 仍然很高，说明当前 clash repulsion 还不够，或者缺少后处理 relax / safe-accept gate。
- contact-F1 只有 `0.1632`，虽然比 unguided 明显提升，但距离可用设计还有差距。
- `source_state0` 仍要求至少一个 source complex/interface；纯 de novo target-only 场景还需要 hotspot/pocket shell 或预测 anchor 生成器。
- B0 native fair baseline 尚未完全解决；后续应把 baseline 输出也放到同一 exact geometry evaluator 下，并避免使用对 B1 有利的额外信息。

## 下一步建议
1. 接入 `source_interface_manifest`：允许从训练/实验/模板来源指定 source complex，而不是硬编码 state0。
2. 增强 `safe-accept`：候选只有在 contact-F1 不下降且 clash 不升时才接受 guidance 内步更新。
3. 把 optional CA/res type feature warning 处理掉：优先在 sampling batch 中显式提供字段，确认是否改善绝对质量。
4. 实现 hotspot-shell anchors：没有 source complex 时，用 persistent hotspot、pocket shell 和 target motion intersection 生成弱 anchors。
5. 完成 B0/B1/B2 的公平 exact benchmark：B0、B1 都直接对 V_exact contact/interface geometry 评估，B2 是实验上限。

## 复现命令
```bash
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
sbatch scripts/strategy01/stage09c_source_smoke.sbatch
sbatch scripts/strategy01/stage09c_source_full.sbatch
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python -m py_compile scripts/strategy01/stage09_guided_state_specific_sampling.py
```

## 关键输出
- `reports/strategy01/probes/stage09c_source_state0_smoke_summary.json`
- `reports/strategy01/probes/stage09c_source_state0_smoke_exact_benchmark.json`
- `reports/strategy01/probes/stage09c_source_state0_full_summary.json`
- `reports/strategy01/probes/stage09c_source_state0_full_exact_benchmark.json`
- `reports/strategy01/probes/stage09c_source_anchor_comparison_summary.json`
