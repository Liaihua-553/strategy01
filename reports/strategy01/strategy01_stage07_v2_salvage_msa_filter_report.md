# Strategy01 Stage07：科学筛选、MSA 诊断、Wave1 数据生产与多状态训练闭环报告

更新时间：2026-04-29

## 1. 阶段目标

本阶段围绕 Strategy01 的核心科学目标推进：设计一个共享 binder 序列，使它能够兼容 target 的真实功能构象变化，并且在每个 target state 上都有各自合理的复合物几何。也就是说，本阶段的成功标准不是单个静态快照分数变好，也不是把多个 state 的坐标平均成一个“中间结构”，而是：

- 输入端保留真实多状态 target ensemble。
- 模型输出端保留 state-specific flow trajectories：`bb_ca_states [B,K,Nb,3]` 与 `local_latents_states [B,K,Nb,8]`。
- 序列端用多个 state-specific trajectories 共同约束一个 `seq_logits_shared [B,Nb,20]`。
- 训练与评估都关注 worst-state / CVaR / state variance，避免 easiest state 掩盖困难 state。

本阶段没有修改 benchmark baseline 仓，也没有覆盖 baseline checkpoint。所有工作都在策略仓：

`/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase`

## 2. 数据与资源约束

远程资源约束按用户确认后的规则执行：

- GPU：当前 `kfliao` 只按 `new/gu02` 上 1 张 A100 使用，不并发提交多个 GPU 任务。
- CPU：可较积极使用集群 CPU 做 seed mining、过滤、MSA 诊断和 manifest 处理，但至少给其他用户保留两个 CPU 节点。
- Boltz2：默认使用 `/data/kfliao/general_model/envs/boltz_cb04aec/bin/boltz`，缓存目录 `/data/kfliao/general_model/boltz_cache`，`--no_kernels`，模型为 Boltz2。

## 3. 高质量验证集与训练 seed 构建

### 3.1 V_exact 扩展

执行 `stage07_vexact_expand` 后得到：

- 扫描条目：2258
- `V_exact`：64 条
- exact families：55 个
- motion 分层：small 17、medium 27、large 20
- hybrid seed：0
- 错误数：0

科学意义：这给后续 B0/B1/B2 复评提供了更可靠的 exact-only 验证口径。`V_hybrid` 不替代 exact leaderboard。

### 3.2 训练 seed mining

执行 `stage07_mine_train_seeds.py` 后得到：

- 请求条目：3000
- 实际扫描：2258
- accepted seeds：256
- excluded groups：22
- 输出：`data/strategy01/stage07_seed_pool/T_prod_seed_manifest.json`
- 主要失败：`no_pair=774`、`download:HTTPError=10`、`excluded_group=20`、`duplicate_pair=5`
- 耗时：2008.95 秒

这一步主要使用 CPU 与网络/文件 IO，不占 A100。

### 3.3 v2 科学过滤

执行 `stage07_filter_seed_pool_v2.py` 后：

- 输入 seed：256
- 总保留：152
- strict `<=8 Å`：139
- extended `8-10 Å`：13
- motion bins：small 59、medium 52、large 28、extra_large_review 13
- 剔除 `motion_gt_10A`：61
- 剔除 family leakage：21
- 剔除 weak exact interface：10
- target 长度超限：5
- binder 长度超限：7

过滤阈值：

- `1.0 <= target_motion_rmsd <= 8.0 Å` 为默认主集。
- `8-10 Å` 只作为有 persistent anchor 证据的扩展区。
- `>10 Å` 不进入自动训练主集，只进入挑战分析。
- 每个 state 至少 8 个 contacts。
- target persistent anchors 至少 3 个，binder persistent anchors 至少 2 个。

科学意义：真实功能构象变化可以从局部小幅调整到多 Å 级域运动，但自动训练主集不应被极端大运动、界面锚点缺失或 family 泄漏污染。这里的 motion RMSD 不是唯一科学指标，而是与 persistent interface anchors、contact persistence 一起作为数据可靠性过滤器。

## 4. MSA 诊断与结论

执行 `stage07_msa_ab_probe.py` 后得到：

- 状态：completed
- 测试样本：`1jmq_A_P`, state 0
- 云端 MSA 来源：salvaged cloud MSA Boltz output
- cache replay：完成，但 Bronze/Silver 未通过
- 本地工具：`mmseqs` 可用，`hhblits` 可用，`/share/HHsuite/databases` 存在
- 结论：云端 ColabFold MSA 继续作为质量基准；本地 HHblits/MMseqs2 暂时只做 diagnostic branch，不能直接替换主生产。

重要澄清：`--use_msa_server` 指 Boltz2 调用远程 ColabFold MSA server，不是在 `kfliao` 集群本地完整跑 MSA。Stage07 的 evidence 来自 Boltz 输出里的 MSA 目录、缓存 replay、以及云端 MSA 结果文件。后续如果要改为本地 MSA，必须做质量 A/B：不能只比速度，还要比较 contacts、interchain PAE、protein ipTM、Bronze/Silver pass。

## 5. 旧低收益任务 salvage 与停止

旧任务 `1908299` 经过 salvage 后：

- sample dirs present：71
- base seeds with any finished pass：59
- all-state Bronze `>=2`：13
- all-state Bronze `>=3`：5
- accepted variants estimate：56
- Silver all-states：0
- state-level attempted：508
- state-level selected：34

结论：旧生产配置 all-state Bronze 命中率低且无 Silver，不值得继续占用 A100。已 salvage 可用结果并转向 v2/v23 过滤与 anchor-first/soft-anchor 生产。

## 6. v23 wave1 数据生产结果

最终完成的 SLURM 任务：`1909566`。

输出目录：

`data/strategy01/stage07_production_v23_wave1/`

summary：

`reports/strategy01/probes/stage07_data_production_v23_wave1_summary.json`

结果：

- status：passed
- exact_count：24
- hybrid_built_count：0
- train_count：110
- val_count：28
- base_seed_count：142
- accepted_bases：27
- rejected_bases：115
- accepted_variants：138
- quality：全部为 Bronze，没有 Silver

解释：本轮数据规模未达到最初 `>=256 train + >=64 val` 的目标，但已经足够进行 pilot fine-tune 和多状态 loss/结构训练链路验证。低命中率的主因仍是 state-level anchor 失败，尤其大量失败发生在 `state0:anchor`，说明 seed 排序、state ordering、anchor 约束和 Boltz pass 策略还需要优化。

## 7. `/tmp` 输出持久化

`1909566` 的 prediction paths 初始指向 gu02 `/tmp`，为了避免节点临时目录丢失，新增并执行：

- `scripts/strategy01/stage07_materialize_predictions.py`
- `scripts/strategy01/stage07_materialize_v23_wave1.sbatch`

任务：`1911454`

结果：

- status：passed
- materialized unique sources：602
- missing unique sources：0
- elapsed：12.24 秒
- 生成持久化数据：
  - `stage06_predictor_pilot_samples_materialized.pt`
  - `T_prod_train_materialized.pt`
  - `T_prod_val_materialized.pt`
  - `materialized_predictions/` 下的 PAE、pLDDT、PDE、PDB、confidence 等文件

科学/工程意义：后续 AE latent、训练、复评不再依赖 gu02 `/tmp`，数据可复现性明显提高。

## 8. AE latent 接入

执行 `stage07_postprocess.sbatch` 的 AE 步骤后：

- summary：`reports/strategy01/probes/stage07_v23_wave1_pilot_ae_latent_extract_summary.json`
- status：passed
- num_samples_total：138
- processed_states：304
- elapsed：14.83 秒
- 输出：`stage06_predictor_pilot_samples_materialized_ae_latents.pt`

本阶段确认 `local_latents` 已由 `complexa_ae_encoder_mean` 生成，不再使用 geometry proxy 混入主训练。

## 9. 多状态训练结果：structfix5

最终成功任务：`1911500`。

输出：

- 结果 JSON：`reports/strategy01/probes/stage07_v23_wave1_pilot_structfix5_results.json`
- 曲线目录：`reports/strategy01/figures/stage07/stage07_v23_wave1_pilot_structfix5/`
- checkpoint：`ckpts/stage07_sequence_consensus/runs/stage07_v23_wave1_pilot_structfix5/mini_final.pt`

### 9.1 训练配置与资源

- device：cuda
- train samples：110
- val samples：28
- selected batch size：4
- training elapsed：1515.92 秒
- GPU 显存峰值：约 6.33 GB
- 图数量：30 张，包括 train/eval total loss 与各 state total loss 的 linear/log 版本

### 9.2 Probe 结果

`loss_unit`：passed

输出 shape：

- `seq_logits_shared [2,10,20]`
- `seq_logits_base_shared [2,10,20]`
- `state_seq_logits [2,3,10,20]`
- `bb_ca_states [2,3,10,3]`
- `local_latents_states [2,3,10,8]`
- `interface_quality_logits [2,3,5]`

输入/标签 shape：

- `x_target_states [2,3,62,37,3]`
- `interface_contact_labels [2,3,62,10]`
- `interface_quality_labels [2,3,5]`

`sequence_consensus`：passed

关键梯度非零：

- `shared_seq_head.1.weight`
- `state_seq_head.1.weight`
- `shared_seq_consensus_gate.1.weight`
- `state_condition_projector.1.weight`

`grad_route`：passed

关键梯度非零：

- `ensemble_target_encoder.cross_state_fusion.q_proj.weight`
- `target2binder_cross_attention_layer.0.mhba.mha.to_q_a.weight`
- `shared_seq_head.1.weight`
- `state_seq_head.1.weight`
- `ca_linear.1.weight`
- `local_latents_linear.1.weight`
- `interface_quality_head.1.weight`

这说明 shared sequence head 确实受到多 state trajectories、state consensus、结构 heads 和 interface proxy 的共同约束。

### 9.3 overfit1

- train_total：36.56 -> 5.27
- eval total：30.44 -> 24.21，下降 20.47%
- eval struct：28.27 -> 24.14，下降 14.62%
- eval seq：3.36 -> 0.0296，下降 99.12%
- CVaR：31.62 -> 26.88，下降 15.01%
- state0：31.74 -> 28.47，下降 10.32%
- state1：31.50 -> 25.29，下降 19.72%
- state2：26.54 -> 24.74，下降 6.77%

### 9.4 overfit4

- train_total：32.65 -> 5.02
- eval total：27.69 -> 16.77，下降 39.44%
- eval struct：25.32 -> 16.73，下降 33.93%
- eval seq：3.65 -> 0.0133，下降 99.64%
- mean：27.46 -> 18.20，下降 33.74%
- CVaR：28.17 -> 18.60，下降 33.97%
- variance：2.89 -> 1.73，下降 39.95%
- state0：29.06 -> 17.81，下降 38.70%
- state1：27.28 -> 17.45，下降 36.01%
- state2：11.43 -> 10.54，下降 7.83%

### 9.5 mini 训练

- train_total：33.35 -> 5.67
- eval total：27.19 -> 7.27，下降 73.25%
- eval struct：25.31 -> 7.25，下降 71.35%
- eval seq：2.87 -> 0.000245，下降 99.99%
- mean：27.28 -> 7.93，下降 70.94%
- CVaR：27.86 -> 8.10，下降 70.92%
- variance：4.95 -> 0.37，下降 92.47%
- state0：29.58 -> 7.26，下降 75.46%
- state1：26.13 -> 8.33，下降 68.10%
- state2：11.36 -> 4.34，下降 61.77%

结论：这次不是只优化 easiest state。三个 state 都明显下降，CVaR 和 variance 也下降，说明多状态鲁棒训练目标开始起作用。

## 10. 本阶段关键代码改动

### 10.1 多状态 loss：只对真实存在的 state 计算 flow matching

文件：`src/proteinfoundation/flow_matching/multistate_loss.py`

问题：早期版本把 `[B,K]` 展平后，即使某些 padding state 的 mask 全 False，也会进入 base flow matching loss。虽然后面用 `torch.where` 或 state mask 试图隐藏缺失 state，但 autograd 里已经可能产生 NaN 梯度。

修正：对每个 data mode 先构造 `flat_present = state_present.reshape(B*K)`，只把真实存在的 state 送进 `compute_fm_loss`，再把 valid losses scatter 回 `[B,K]`。缺失 state 不再进入 base loss。

科学意义：variable-K 多状态训练必须让缺失构象完全不参与结构噪声路径。否则模型会被不存在的 state 影响，轻则 loss 不可信，重则梯度 NaN。

### 10.2 Cross-state fusion：避免 masked softmax 反向 NaN

文件：`src/proteinfoundation/nn/modules/cross_state_fusion.py`

问题：cross-state attention 中如果 key mask 全 false，用 `-inf` masked softmax 容易在反向传播中产生 NaN，即使 forward 后续做了 `nan_to_num`。

修正：将 `-inf` 替换为有限大负数 `-1.0e4`，并保留显式 mask 后处理。

意义：提高缺失 state、padding state 和异常 batch 下的数值稳定性。

### 10.3 Stage07 训练脚本：恢复结构训练，不只训练 sequence head

文件：`scripts/strategy01/stage07_sequence_consensus_training.py`

问题：早期 `mini` 阶段沿用了 `trainable_phase='seq_consensus'` 或过低 lr，导致 sequence loss 下降但 structure loss 基本不动。这样“能跑通”但不符合科学目标。

修正：`overfit4/mini` 使用可训练结构 heads、top trunk、多状态模块的 phase，并在 loss 修复后恢复 `trainable_phase='mini'`。

意义：让 state-specific flow trajectories 真的被训练，而不是只让 shared sequence head 记住序列。

### 10.4 Stage04 调试工具：增加中间冻结 phase 以定位 NaN

文件：`scripts/strategy01/stage04_real_complex_loss_debug.py`

新增过的诊断 phase：

- `mini_no_encoder`
- `mini_heads_only`

这些 phase 帮助确认 NaN 不是单独来自 encoder、cross-attn 或 top transformer，而是来自 padding states 进入 base FM loss 的更底层问题。

### 10.5 prediction materialization

新增文件：

- `scripts/strategy01/stage07_materialize_predictions.py`
- `scripts/strategy01/stage07_materialize_v23_wave1.sbatch`

作用：把 Boltz2 在 `/tmp` 下生成的 PDB、confidence、PAE、pLDDT、PDE 等文件复制到策略仓持久化目录，并重写 dataset `.pt` 中的路径。

### 10.6 postprocess sbatch

文件：`scripts/strategy01/stage07_postprocess.sbatch`

修改：默认读取 materialized dataset，若 AE dataset 已存在则跳过 AE 重算，并使用明确的 `STAGE07_RAW_DATASET`、`STAGE07_AE_DATASET`、`STAGE07_RUN_NAME` 环境变量。

## 11. 错误与修复日志

### 11.1 远程 PowerShell 引号/管道问题

表现：Windows PowerShell 中直接运行包含 `||`、嵌套 ssh、heredoc 或复杂 Bash 变量的命令容易失败。

修复：本阶段继续采用“本地写 LF 脚本 -> scp -> ssh bash 执行”的方式。

### 11.2 `ssh gu02` 权限问题

表现：不能直接 `ssh gu02`。

修复：不依赖直接登录计算节点，而是通过 SLURM 日志、scratch salvage 和登录节点文件系统检查结果。

### 11.3 AE checkpoint 被清理导致失败

表现：`stage07_postprocess` 第一次失败，找不到 `ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt`。

根因：之前清理策略仓时删除了阶段专用 AE checkpoint 副本。

修复：从只读 baseline checkpoint 恢复到策略仓：

`ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt`

未修改 baseline checkpoint。

### 11.4 磁盘 quota 接近上限

表现：恢复 checkpoint 时 quota 接近满。

修复：删除旧 Stage03 debug runs 下的大体积过时 checkpoint，保留当前阶段所需 checkpoint、代码、报告与小型 JSON。

### 11.5 main checkpoint 缺失

表现：AE latent 已成功，但训练启动时找不到 `complexa_init_readonly_copy.ckpt`。

修复：从 baseline 恢复阶段专用 main checkpoint 副本：

`ckpts/stage03_multistate_loss/complexa_init_readonly_copy.ckpt`

### 11.6 结构训练 NaN

表现：`structfix`、`structfix2`、`structfix3`、`structfix4` 分别在 encoder、cross-attn、top transformer 或 heads-only 阶段出现 non-finite gradients。

最初假设：Cross-state attention masked softmax 可能反向 NaN。

中间修复：把 `-inf` mask 改为有限 `-1e4`，提升稳定性，但未完全解决。

最终根因：padding/missing states 仍进入 base flow matching loss。即使 later aggregation 使用 mask，base loss 内部已经可能产生 NaN 梯度。

最终修复：`compute_multistate_loss` 只对 `state_present=True` 的 state 调 base `compute_fm_loss`，缺失 state 不进入结构损失计算图。

修后验证：`1911500 / structfix5` 训练完整通过，overfit4 与 mini 的结构 loss 均显著下降。

## 12. 当前结论

本阶段已经完成以下闭环：

- 高质量 exact validation 挖掘：64 exact / 55 families。
- 训练 seed 构建与科学过滤：256 -> 152 filtered seeds。
- 云端 MSA 与本地 MSA 诊断：云端继续作为主线质量基准。
- 低收益旧任务 salvage 并停止。
- v23 wave1 Boltz2 数据生产：110 train + 28 val，138 variants。
- `/tmp` prediction 持久化：602 个 unique files，missing=0。
- AE latent 实测接入：138 samples / 304 states。
- batch size 校准：batch=4 成功，显存约 6.33GB。
- 多状态训练：overfit1、overfit4、mini 全部跑通，结构 loss、CVaR、state variance、三态 loss 均下降。
- `compileall` 已覆盖关键修改文件并通过。

最重要的阶段性结论：

**Strategy01 的多状态结构训练链路现在不是“只会背共享序列”，而是 state-specific trajectories 也可以被真实 AE latent + Boltz2 pseudo-complex 标签驱动下降。**

## 13. 仍未完成的关键项

### 13.1 B0/B1/B2 生成复评尚未完成

目前还没有把 `mini_final.pt` 接入正式 generation 入口，对 `V_exact` 做 baseline vs Strategy01 的生成与 Boltz2 refold 对比。因此不能宣称 Strategy01 已经在最终设计指标上优于 baseline。

需要下一步做：

- B0：原始 Complexa 单状态生成 shared binder sequence，逐 state Boltz2 refold。
- B1：Strategy01 ensemble-input 生成 shared binder sequence，逐 state Boltz2 refold。
- B2：V_exact 实验真实复合物与真实 binder sequence 的参考上限，包括 exact complex 指标和真实 binder sequence 的 Boltz2 refold ceiling。

### 13.2 数据质量仍以 Bronze 为主

v23 wave1 没有 Silver 样本。当前 pilot 可以证明训练链路和方向，但不足以支撑大规模结论。下一步必须提高 accepted base 命中率和 Silver 占比，尤其要解决 state-level anchor failure。

### 13.3 训练集规模仍不足

当前只有 110 train / 28 val，未达到 Stage07 目标的 `>=256 train + >=64 val`。不过这次已证明 batch=4 和结构训练可行，后续可以继续扩大数据。

## 14. 下一步优化方案

我建议下一阶段分成两个并行但不混淆的目标。

### 14.1 先做小规模 B0/B1/B2 smoke

不要直接对 64 个 `V_exact` 全量复评。先选 4-8 个 exact targets，覆盖 small/medium/large motion bins：

- 每个 target 候选预算先降到 `N=4-8`。
- 先验证 Strategy01 `mini_final.pt` 能否走 ensemble-input generation。
- 验证输出是不是 `shared_sequence + K 个 state-specific poses`，而不是 averaged `bb_ca`。
- 对输出序列逐 state 跑 Boltz2 refold，记录 all-state pass、worst interchain PAE、protein ipTM、contacts、clash。

如果 smoke 失败，先修 generation adapter，不进入全量 B0/B1。

### 14.2 继续扩大数据，但先优化 seed/anchor

数据生产不应简单扩大旧配置。建议：

- 优先分析失败的 `state0:anchor` 样本，检查 state ordering、exact contact anchor 是否过硬、目标构象对齐是否错位。
- 对 `8-10 Å` motion 样本只保留 persistent anchors 明确者。
- 引入 softer anchor pass：anchor 不作为硬失败先验，而作为 Boltz potentials / constraints 的弱指导。
- 对已知成功 families 做 family-level hard negative removal，避免只扩大相似容易样本。
- 数据目标：下一轮至少 `256 train + 64 val`，如果 Silver 仍为 0，则不要盲目扩大到 800。

### 14.3 1000 样本算力更新估计

基于当前实测：

- v23 wave1 生产 138 variants 需要约一个长 GPU 作业，瓶颈是 Boltz2 而非训练。
- 训练 110/28，batch=4，3000 step，耗时约 1516 秒，显存约 6.33GB。

外推：

- 1000 samples 训练-only：1 张 A100 约 3-6 小时仍合理；如果 sequence length 明显变长或加入更重 eval，可能到 6-10 小时。
- 1000 samples 含 Boltz2 数据生产：仍可能是 2-4 天量级，主要瓶颈在 predictor 与 MSA，而不是 fine-tune。
- 2 张 A100 若可用，训练可接近 1.5-2 倍加速；但当前权限主线仍按 1 张 A100 规划。

## 15. 复现路径

从当前策略仓复现本阶段结果：

1. 确认 baseline checkpoint 只读存在。
2. 确认策略仓阶段 checkpoint 副本存在：
   - `ckpts/stage03_multistate_loss/complexa_init_readonly_copy.ckpt`
   - `ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt`
3. 使用 `stage07_materialize_v23_wave1.sbatch` 持久化 `/tmp` prediction 输出。
4. 使用 `stage07_postprocess.sbatch`：
   - 输入 `stage06_predictor_pilot_samples_materialized.pt`
   - 生成 `stage06_predictor_pilot_samples_materialized_ae_latents.pt`
   - 运行 `stage07_sequence_consensus_training.py`
5. 检查：
   - `reports/strategy01/probes/stage07_v23_wave1_pilot_structfix5_results.json`
   - `reports/strategy01/figures/stage07/stage07_v23_wave1_pilot_structfix5/`
   - `ckpts/stage07_sequence_consensus/runs/stage07_v23_wave1_pilot_structfix5/mini_final.pt`

## 16. 本阶段状态

阶段状态：部分完成但有重要突破。

已完成：数据生产、持久化、AE latent、batch=4、结构训练、曲线图、关键 bug 修复。

未完成：B0/B1/B2 exact validation generation-refold benchmark。

建议：先提交当前稳定代码与报告作为 Stage07-pilot checkpoint，然后继续单独推进 B0/B1/B2 smoke 与 generation adapter。

## 17. 2026-04-29 续执行：B1 checkpoint 接入与 state-specific sampler smoke

### 17.1 为什么不能直接用 `mini_final.pt` 跑原始 generation

`ckpts/stage07_sequence_consensus/runs/stage07_v23_wave1_pilot_structfix5/mini_final.pt` 是 Stage07 训练脚本保存的轻量权重文件，内容只有：

- `model_state_dict`
- `phase=mini`
- `steps=3000`

它不是 Lightning checkpoint，因此不能直接交给 `Proteina.load_from_checkpoint` 或原始 `proteinfoundation.generate`。

同时，原始 `generate.py` 使用的 `TargetFeatures` 仍是单状态/扁平 target conditioning，不能提供 `EnsembleTargetEncoder` 需要的显式 state-axis 字段：

- `x_target_states`
- `target_mask_states`
- `seq_target_states`
- `target_hotspot_mask_states`

因此，B1 不能直接复用原始 generation 入口；必须先补 Strategy01 专用 multistate generation/sampling adapter。

### 17.2 新增 checkpoint 转换脚本

新增文件：

`/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/scripts/strategy01/stage07_convert_mini_to_lightning_ckpt.py`

作用：

- 以 `complexa_init_readonly_copy.ckpt` 作为 metadata 模板。
- 读取 Stage07 `mini_final.pt` 中的 `model_state_dict`。
- 写入 `nn.*` 前缀权重。
- 将 `cfg_exp.nn` 替换为 `local_latents_transformer_multistate` 配置。
- 设置当前策略仓 AE checkpoint 路径。
- 不复制 optimizer states，避免生成过大的训练 checkpoint。

输出：

`ckpts/stage07_sequence_consensus/runs/stage07_v23_wave1_pilot_structfix5/mini_final_lightning.ckpt`

转换结果：

- status：passed
- 输出大小：约 1.3 GB
- 写入 `nn` keys：1035
- `cfg_nn_name=local_latents_transformer_multistate`
- `latent_dim=8`
- source phase：mini
- source steps：3000

Load smoke：

- `Proteina.load_from_checkpoint` 成功。
- `nn_class=LocalLatentsTransformerMultistate`
- 参数量：338,600,691
- `has_state_seq_head=true`
- `has_ensemble_target_encoder=true`

报告文件：

- `reports/strategy01/probes/stage07_mini_lightning_ckpt_conversion_summary.json`
- `reports/strategy01/probes/stage07_mini_lightning_load_smoke_summary.json`

### 17.3 为什么还不能用 legacy averaged sampler 做 B1

当前 `LocalLatentsTransformerMultistate.forward()` 同时输出：

- legacy-compatible `bb_ca` / `local_latents`：按 state weights 加权平均。
- state-specific `bb_ca_states` / `local_latents_states`：每个 state 的分支输出。
- shared sequence：`seq_logits_shared`。

原始 `ProductSpaceFlowMatcher.full_simulation()` 只消费 legacy `bb_ca/local_latents`，因此如果直接跑原始 `generate.py`，采样轨迹仍会被平均结构驱动。这与 Strategy01 的科学目标不一致，因为多个 state 的合理 binder pose 不应被坐标平均成中间构象。

### 17.4 新增 state-specific sampling smoke

新增文件：

- `scripts/strategy01/stage07_multistate_sampling_smoke.py`
- `scripts/strategy01/stage07_multistate_sampling_smoke.sbatch`

核心逻辑：

1. 读取带真实 AE latent 的多状态样本。
2. 每个 state 单独维护 `x_states['bb_ca'] [B,K,Nb,3]` 和 `x_states['local_latents'] [B,K,Nb,8]`。
3. 每个 step 用 state weights 形成 legacy `x_t` 给主干读取，但更新时不再使用 averaged output。
4. 用 `bb_ca_states/local_latents_states` 分别更新每个 state 的 trajectory。
5. 最后导出一个 `shared_sequence.fasta` 和 K 个 `stateXX_binder_ca.pdb`。

这一步是 B1 generation adapter 的最小可运行版本。

### 17.5 GPU smoke 结果

SLURM job：`1911541`

结果文件：

`reports/strategy01/probes/stage07_multistate_sampling_smoke_summary.json`

结果：

- status：passed
- sample：`2ofq_A_B__v00__k3__hexact`
- split：val
- valid states：3
- nsteps：24
- 输出 shape：
  - `bb_ca [1,3,22,3]`
  - `local_latents [1,3,22,8]`
- `has_shared_seq_logits=true`
- `has_state_seq_logits=true`
- 预测 shared sequence：`APSSSSSSSSSNLNSYYYPPNS`
- 参考 shared binder sequence：`PPPEPDWSNTVPVNKTIPVDTQ`
- sequence identity：0.136
- state CA RMSD to label：
  - state0：1.773 nm
  - state1：1.546 nm
  - state2：1.349 nm
- generated state pairwise CA RMSD：约 0.149-0.182 nm

输出目录：

`results/strategy01/stage07_sampling_smoke/2ofq_A_B__v00__k3__hexact/`

### 17.6 解释与结论

这次 smoke 的意义是工程正确性，不是最终效果证明。

已证明：

- Stage07 mini checkpoint 可以转换成 generation 可加载的 Lightning checkpoint。
- Strategy01 可以绕开 averaged legacy sampler，维护 K 个 state-specific flow trajectories。
- 模型能导出一个 shared sequence 与 K 个 state-specific binder CA 结构。

尚未证明：

- 输出 shared sequence 是否真的优于 baseline。
- 输出序列逐 state Boltz2 refold 后是否 all-state pass。
- B1 是否优于 B0。

当前 smoke 的质量信号偏弱：预测序列与伪标注 reference identity 只有 13.6%，state CA RMSD 也较大。因此下一步不能直接宣称 B1 成功，而应做小规模 B0/B1/B2 refold smoke。

### 17.7 下一步执行建议

下一步建议做 4-8 个 `V_exact` target 的小规模 refold smoke，而不是直接跑全量 benchmark：

- B0：baseline Complexa 单状态生成 shared sequence，逐 state Boltz2 refold。
- B1：Strategy01 state-specific sampler 生成 shared sequence，逐 state Boltz2 refold。
- B2：真实 binder sequence 的 Boltz2 refold ceiling 与实验 exact complex 指标。

如果 B1 在 smoke 中没有任何 worst-state 指标改善，应优先修：

- 更长/更稳定 multistate training。
- 更高质量 Silver 数据占比。
- state-specific sampler 的 sequence consensus 温度与 anchor 权重。
- 训练时让 state-specific flow head 更直接监督 clean prediction，而不是只作为 raw velocity/interface proxy。

## 18. B1/B2 refold smoke 接续检查、修复与结论

### 18.1 接续时远程状态

接续检查时间：远程时间 `2026-04-30`。

队列状态：

- `squeue -u kfliao` 为空，没有残留 GPU 作业。
- 上一阶段已完成的 `1911541` state-specific sampling smoke 仍为通过状态。
- 未提交的 B1/B2 refold smoke 首次结果为 `partial_failed`。

本次继续只操作策略仓：

`/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase`

没有改动 benchmark baseline 仓与 baseline checkpoint。

### 18.2 首次 B1/B2 refold 失败根因

失败文件：

- `scripts/strategy01/stage07_b1_b2_refold_smoke.py`
- `scripts/strategy01/stage04_predictors/boltz_adapter.py`

失败现象：

- 6 个 Boltz2 refold 全部没有生成 prediction PDB。
- summary 里都是 `FileNotFoundError: Boltz prediction PDB not found`。
- Boltz 日志里的真正根因是 template 解析阶段 `IndexError: list index out of range`。

进一步检查发现两个独立问题：

- target template PDB 是从 seed state 提取的 PDB，残基号保留为 `177..271`，且没有完整 polymer sequence metadata。Boltz2 内部会用 gemmi 把 PDB 转 mmCIF 再解析；没有 sequence metadata 时，`parse_polymer()` 得到空序列，导致越界。
- YAML 里写了 `msa: empty`。虽然命令带了 `--use_msa_server`，但 `msa: empty` 会强制 single-sequence mode，所以旧结果并没有真实使用云端 MSA。

### 18.3 代码修复 1：Boltz2 MSA 写法修正

修改文件：

`scripts/strategy01/stage04_predictors/boltz_adapter.py`

修改前：

```yaml
sequences:
  - protein:
      id: A
      sequence: ...
      msa: empty
  - protein:
      id: B
      sequence: ...
      msa: empty
```

问题：

- `msa: empty` 明确表示不使用 MSA。
- 这会使 `--use_msa_server` 只显示启用，但不会对该链真正生成 MSA。

修改后：

- `msa=""` 或 `msa=None` 时不再写 `msa:` 字段。
- 只有存在完整缓存路径时才写入 `msa: /path/to/cache.csv`。
- 如果 A/B 两条链只有一条有 custom MSA 缓存，adapter 会省略两条链的 MSA，让 `--use_msa_server` 对两条链一起生成 paired MSA。

科学意义：

- 多链复合物预测不能混用“一条链 custom MSA、另一条链 auto MSA”。Boltz2 会直接报错。
- 对 B1/B2 refold 来说，MSA 行为必须真实可控，否则评估结果不可解释。

验证结果：

- 日志出现 `Calling MSA server for target ... with 2 sequences`。
- 不再出现 `Found explicit empty MSA`。
- 后续相同序列会写入 `data/strategy01/stage07_boltz_msa_cache/` 并复用。

### 18.4 代码修复 2：target template 改成带 full sequence 的 mmCIF

修改文件：

`scripts/strategy01/stage07_b1_b2_refold_smoke.py`

新增逻辑：

- `reindex_pdb_residues()`：把 seed-state target PDB 的残基号重排为 `1..N`。
- `write_boltz_template_cif()`：使用 gemmi 读取重编号 PDB，显式写入 target full sequence，再导出 Boltz2 可解析的 `.cif`。
- adapter 根据 template 后缀自动写 `cif:` 或 `pdb:`。

修改前：

```yaml
templates:
  - pdb: .../2ofq_state00_target_A.pdb
    chain_id: A
    force: false
    threshold: 2.000
```

修改后：

```yaml
templates:
  - cif: .../state00_target_template.cif
    chain_id: A
    force: true
    threshold: 2.000
```

验证结果：

- 单独调用 Boltz2 parser 测试通过：
  - `parse_cif_ok`
  - chain 数：1
  - target length：95
- GPU refold 不再在 template parse 阶段失败。

科学意义：

- B1/B2 refold 的 target state 必须尽量固定在给定构象，否则就变成“只输入 target 序列”的 docking，不再是多状态构象兼容评估。
- `.cif + full_sequence` 是比临时修改 PDB header 更稳的 template 表示。

### 18.5 代码修复 3：anchor-constrained refold smoke

修改文件：

`scripts/strategy01/stage07_b1_b2_refold_smoke.py`

新增逻辑：

- 从 accepted pseudo-complex label 中提取每个 state 的 reference contact。
- 默认最多取 8 个非重复 target/binder 接触对。
- 将这些接触作为 Boltz2 `contact` constraints 写入 YAML。

修改文件：

`scripts/strategy01/stage04_predictors/boltz_adapter.py`

约束写法从：

```yaml
force: false
```

改为：

```yaml
force: true
```

原因：

- `--use_potentials` 只是允许使用 potentials。
- 每条 contact constraint 仍需要 `force: true` 才会真正作为约束 potential。

### 18.6 B1/B2 refold smoke 实测结果

作业记录：

- `1911545`：无 anchor 约束版本，status=`passed`
- `1911549`：anchor 约束但 contact `force=false`，status=`passed`
- `1911550`：anchor 约束且 contact `force=true`，status=`passed`

当前最终 summary：

`reports/strategy01/probes/stage07_b1_b2_refold_smoke_summary.json`

保留的无约束 summary：

`reports/strategy01/probes/stage07_b1_b2_refold_smoke_unconstrained_summary.json`

样本：

- sample：`2ofq_A_B__v00__k3__hexact`
- K：3
- B1 sequence：`APSSSSSSSSSNLNSYYYPPNS`
- B2 reference sequence：`PPPEPDWSNTVPVNKTIPVDTQ`

最终 anchor-force 结果摘要：

| 模式 | all-state pass | worst interchain PAE | worst protein ipTM | worst complex pLDDT | mean contact count |
|---|---:|---:|---:|---:|---:|
| B1 Strategy01 | false | 24.37 Å | 0.092 | 0.645 | 0 |
| B2 reference refold | false | 21.54 Å | 0.213 | 0.754 | 0 |

解释：

- 工程链路已经跑通，Boltz2 能产出 PDB、PAE、pLDDT、ipTM。
- MSA server 真实生效，且缓存复用逻辑已修正。
- 但这个 refold 协议无法恢复界面：即使 B2 reference binder sequence 也得到 `contact_count=0`。
- 因此这次结果不能用来断言 Strategy01 模型一定失败；它首先说明当前 “target template + binder sequence + sparse contact constraints” 的 Boltz2 docking 评估协议不够可靠。

### 18.7 为什么 anchor 约束仍然没有恢复界面

当前最可能原因：

- contact constraints 是从 pseudo-complex 抽出的稀疏 residue-level anchors，但没有提供 binder 初始 pose 或完整界面几何。
- Boltz2 对短 peptide/protein binder 的自由 docking 不一定能在少量 diffusion samples 下找到正确结合模式。
- 本次 `sampling_steps=5`、`diffusion_samples=1` 是 smoke 设置，偏向快速验证，不适合作为最终 docking benchmark。
- 该样本属于 `train_boltz` pseudo-label，不是 `V_exact` 全实验样本；B2 不是严格实验上限。

对 Strategy01 科学目标的影响：

- 这一步验证的是“评估协议是否可用”，不是模型最终能力。
- 当前协议对 reference sequence 也失败，所以不能用它判断 B1 是否优于 B0。
- 后续 B0/B1/B2 必须转到 `V_exact` exact-complex 主口径，B2 应优先直接使用实验复合物几何作为上限，而不是只做无初始姿态的 Boltz2 redocking。

### 18.8 下一步修正后的 B0/B1/B2 协议

推荐下一阶段改为：

- B0：baseline Complexa 单状态生成 shared sequence；评估时对每个 state 用同一套 target hotspot/anchor 信息做受控 refold。
- B1：Strategy01 多状态 state-specific sampler 生成 shared sequence 和 K 个 state-specific binder pose；评估优先使用模型输出 pose + Boltz2/结构指标复评分，而不是让 Boltz2 从零 dock。
- B2：`V_exact` 实验真实复合物几何上限；如做 Boltz2 ceiling，必须明确标为 predictor ceiling，不能替代实验上限。

必须保留的科学标准：

- 一个 shared binder sequence。
- 每个 target state 有各自合理的 binder pose。
- 指标看 worst-state 和 all-state，不看 averaged structure。
- 如果 reference binder 在某个 predictor redocking 协议下也失败，该 predictor 协议不能作为主判据。

### 18.9 本次接续结论

已完成：

- 修复 Boltz2 template PDB 解析失败。
- 修复 `--use_msa_server` 被 `msa: empty` 抵消的问题。
- 修复 custom MSA 与 auto MSA 混用问题。
- 完成无约束与 anchor-constrained B1/B2 smoke。
- 证明当前 state-specific sampling 输出可以被送入 Boltz2 refold 流程。

未完成或不能宣称完成：

- 不能宣称 B1 已优于 B0。
- 当前 B1/B2 smoke 不能作为最终 benchmark，因为 B2 reference redocking 也没有恢复界面。
- 需要在 `V_exact` 上重新设计 B0/B1/B2 评估，尤其要把实验复合物几何作为 B2 主口径。
