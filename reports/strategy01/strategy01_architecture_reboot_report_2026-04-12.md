# Strategy01 架构重开报告（2026-04-12）

## 1. 基线冻结与策略仓来源
- 远程可跑 benchmark 的原始工作仓：`/data/kfliao/general_model/Proteina-complexa`
- 本轮唯一有效基线冻结仓：`/data/kfliao/general_model/strategy_repos/proteina-complexa-benchmark-baseline`
- 本轮架构重开所对应的基线 commit：`2db8e1df838354db079ce8e0e4b88aaebd31f35f`
- 本轮重新复制并实施改动的 Strategy01 仓：`/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase`
- 当前工作分支：`codex/strategy01-arch-reboot`

边界说明：
- 没有改动远程原始可跑 benchmark 工作仓。
- 没有改动 benchmark baseline 仓。
- 本轮所有多状态新架构相关修改，都只落在新的 Strategy01 仓中。

## 2. 本轮科学目标
本轮不直接进入完整训练与 benchmark 评估，而是先把最关键的架构问题敲定并验证通路：
1. 将单状态 target 输入改成显式多状态输入。
2. 保留 Complexa 原始单状态特征抽取优势，不推翻已有单态 target 编码。
3. 把多状态融合从简单 concat 升级为“残基对齐的跨状态融合 + 全局 refinement”。
4. 让模型一次性输出：
   - 一个共享 binder 序列头
   - 所有状态对应的结构输出头
5. 用真实 probe 验证输入、前向、checkpoint 接入与 backward 梯度链是否成立。

这一轮对应的核心科学目标是：
- 输入：同一 target 的多个功能相关构象
- 输出：一个共享兼容的 binder 序列，以及该序列在每个状态下的复合物结构表示

## 3. 本轮新增文件
本轮全部改动都采用“新增文件”的方式实现，避免污染基线实现。

### 3.1 新增架构文件
- `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/src/proteinfoundation/nn/modules/cross_state_fusion.py`
- `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/src/proteinfoundation/nn/ensemble_target_encoder.py`
- `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/src/proteinfoundation/nn/local_latents_transformer_multistate.py`

### 3.2 新增配置文件
- `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/configs/nn/local_latents_score_nn_160M_multistate.yaml`
- `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/configs/training_local_latents_multistate_probe.yaml`

### 3.3 新增探针与输出
- `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/scripts/strategy01/probe_multistate_arch.py`
- `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/architecture_probe_results_2026-04-12.json`

## 4. 改动块逐项说明

### A. 输入契约：从单一 target 改成显式 state 轴
科学动机：
- baseline Complexa 默认把 target 当作单一静态条件。
- 本轮改造后，模型显式接收多构象 target，而不是只靠 `target_state_id` 之类的弱标记。

baseline 输入模式：
- `x_target [B, Nt, 37, 3]`
- `target_mask [B, Nt, 37]`
- `seq_target [B, Nt]`
- `target_hotspot_mask [B, Nt]`

Strategy01 输入模式：
- `x_target_states [B, K, Nt, 37, 3]`
- `target_mask_states [B, K, Nt, 37]`
- `seq_target_states [B, K, Nt]`
- `target_hotspot_mask_states [B, K, Nt]`
- `seq_target_mask_states [B, K, Nt]`
- `target_state_weights [B, K]`
- `target_state_roles [B, K]`
- `state_present_mask [B, K]`

实现位置：
- `EnsembleTargetEncoder._extract_state_axis_batch()`

前后代码对比：
- baseline target 数据入口主要来自 `gen_dataset.py` 中的 `TargetFeatures`
- Strategy01 新增显式多状态输入契约，由 `ensemble_target_encoder.py` 负责解析与整理

probe 实测维度：
- 在真实 TNFalpha 双态 probe 上，已验证得到：
  - `state_target_tokens [1, 2, 438, 768]`
  - `ensemble_target_memory [1, 438, 768]`

### B. 保留原始单状态 target 特征提取，再外包成多状态编码器
科学动机：
- 本轮不希望推翻 Complexa 已有价值的单状态 target 几何与序列特征提取。
- 所以不是从零设计全新 target encoder，而是复用 baseline 的单态 target feature，再抬升到多状态融合框架里。

实现位置：
- `ensemble_target_encoder.py`

复用的 baseline 组件：
- `target_feats.py` 中的 `TargetConcatSeqFeat`

前后代码对比：
- baseline：`TargetConcatSeqFeat` 负责把单态 target 的几何、序列、mask、热点等特征拼成一个单态 feature 表示
- Strategy01：通过 `self.single_state_feature_extractor = TargetConcatSeqFeat(**kwargs)` 复用这一套单态编码，再把每个 state 的输出投影到统一 token 空间

probe 实测维度：
- 单状态 target feature 原始维度：`505`
- 投影后每个 state 的 token 表示：`state_target_tokens [B, K, Nt, 768]`

这说明 baseline 的单态信息提取能力被保留下来，而不是被新架构替换掉。

### C. 多状态融合：从简单 concat 升级为“跨状态注意力 + 门控残差 + 全局 refinement”
科学动机：
- 简单 concat 只是把更多 target token 拼进去，让下游自己消化，这对于多构象兼容设计不够强。
- 本轮按既定方案，实现了混合式融合：
  1. residue-aligned cross-state self-attention
  2. gated residual fusion
  3. global refinement
- 同时让 `state_weight` 和 `state_role` 真正进入融合计算，而不再只是数据层附带字段。

实现位置：
- `cross_state_fusion.py`

前后代码对比：
- baseline 中 target cross-attn 路径只读单态 `target_rep`
- Strategy01 中先做 `cross_state_fusion(...)`，再把融合后的 `ensemble_target_memory` 输入 binder trunk

probe 实测维度：
- `state_target_tokens [1, 2, 438, 768]`
- `cross_state_attention_mean [1, 438, 2]`
- `state_summary_tokens [1, 2, 768]`
- `ensemble_target_memory [1, 438, 768]`

这一步证明：state 轴与跨态融合已经真实进入模型，而不是只停留在计划说明里。

### D. 主干融合方式：不用 concat target，改为 binder 对融合后 target memory 做 cross-attention
科学动机：
- baseline 的 concat target 路线，本质上是把 target token 拼到主序列里，最后再裁掉。
- 这种方式不适合当前多状态主线。
- 本轮改成：先得到 `ensemble_target_memory`，再让 binder token 对它做 cross-attention。

实现位置：
- `local_latents_transformer_multistate.py`

前后变化：
- baseline binder 输出：
  - `bb_ca.v [1, 32, 3]`
  - `local_latents.v [1, 32, 8]`
- Strategy01 多状态输出：
  - `seq_logits_shared [1, 32, 20]`
  - `bb_ca_states [1, 2, 32, 3]`
  - `local_latents_states [1, 2, 32, 8]`

### E. 输出契约：从单结构头改为“一个共享序列头 + K 个状态结构头”
科学动机：
- 你的科学目标不是“每个状态各出一个独立 binder”，而是“一个共享 binder 序列兼容多个状态”。
- 因此共享序列必须是一级输出，而不能靠后处理从多个结构里猜出来。

实现位置：
- `local_latents_transformer_multistate.py`

probe 实测输出：
- baseline：
  - `bb_ca.v [1, 32, 3]`
  - `local_latents.v [1, 32, 8]`
- Strategy01：
  - `seq_logits_shared [1, 32, 20]`
  - `bb_ca_states [1, 2, 32, 3]`
  - `local_latents_states [1, 2, 32, 8]`

这一步已经真实达到了“一个共享序列 + 全状态结构输出”的架构目标。

### F. checkpoint 兼容性审计
科学动机：
- 本轮必须搞清楚：新架构到底能复用多少 benchmark-valid Complexa checkpoint，而不是留到后面训练时再看。

probe 实测结果：
- shape-compatible 可直接加载的 checkpoint 键数：`455`
- unexpected key 数：`0`
- 主要 shape mismatch 有两处：
  - `init_repr_factory.linear_out.weight`
    - 新模型 shape：`(768, 45)`
    - baseline checkpoint shape：`(768, 302)`
  - `pair_repr_builder.init_repr_factory.linear_out.weight`
    - 新模型 shape：`(256, 217)`
    - baseline checkpoint shape：`(256, 219)`

意义解释：
- 这说明大部分 trunk 权重其实已经可复用。
- 当前真正需要重点处理的，并不是整个多状态架构“完全 incompatible”，而是 binder 侧输入投影布局还没有完全对齐 baseline checkpoint。
- 这为下一阶段把问题聚焦到“输入投影对齐 / adapter / 部分加载微调”提供了非常明确的证据。

## 5. Probe 结果汇总
原始结果文件：
- `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/architecture_probe_results_2026-04-12.json`

### P0：静态架构探针
通过。
- 新模块可实例化。
- 实测：
  - `single_state_target_feature_dim = 505`
  - `token_dim = 768`
  - `pair_repr_dim = 256`
  - `cross_state_heads = 8`

### P1：前向探针（synthetic + 真实双态 target）
通过。
- synthetic 双态前向通过。
- 真实 TNFalpha 双态前向通过。
- 真实状态提取后的 target 长度为 `438`。

### P2：checkpoint 接入探针
通过，但结论是“部分兼容”。
- `455` 个 shape-compatible 键成功加载。
- 主要 mismatch 已被定位并记录。
- 在部分加载后，多状态模型仍能继续真实前向。

### P3：单步 backward 探针
通过。
- loss：`0.43263593316078186`
- 关键梯度范数非零：
  - `ensemble_target_encoder.target_feature_projection.1.weight = 15.9127`
  - `ensemble_target_encoder.cross_state_fusion.q_proj.weight = 0.2296`
  - `shared_seq_head.1.weight = 7.1072`
  - `ca_linear.1.weight = 1.2308`
  - `local_latents_linear.1.weight = 1.6260`

这说明：
- 新增的 target encoder 通路有梯度
- cross-state fusion 有梯度
- shared sequence head 有梯度
- 结构头也有梯度
- 整个新架构不是“只前向能跑、实际不连通”的假通路

## 6. 这一阶段已经证明了什么
这一轮已经真实证明：
1. 新 Strategy01 架构可以显式接收多状态 target 输入。
2. baseline 的单状态 target 编码能力可以被保留下来并抬升为多状态编码器。
3. 跨状态 self-attention + gated residual + global refinement 已经正确接进模型。
4. 模型已经可以输出你要求的契约：
   - 一个共享 binder 序列头
   - K 个状态结构输出头
5. benchmark-valid Complexa checkpoint 大量权重已经可以直接复用。
6. 当前最关键的不兼容点已经缩小到 binder 侧输入投影，而不是整个架构失败。

## 7. 这一阶段还没有做的事
这一轮还没有证明训练效果，也还没有证明 benchmark 性能提升。
本轮也还没有最终决定：
- 是继续保持现在这版新输入布局，只做部分加载微调
- 还是先把 binder 侧输入布局对齐到 baseline checkpoint，再做更强的权重复用
- 还是后面再做从零训练对照

这些问题现在已经从“猜测”变成了“基于真实 probe 结果的下一阶段决策问题”。

## 8. 下一步最值得优先解决的问题
下一步最关键的技术问题已经很清晰：
- 是否先把 Strategy01 的 binder 侧输入投影形状对齐到 benchmark baseline checkpoint，从而最大化 checkpoint 复用？
- 还是接受当前这版更干净的多状态架构，直接走部分加载 + 微调？

因为 P0-P3 都已通过，这个决策现在已经有了可靠依据，而不再是拍脑袋定路线。