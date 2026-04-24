# Strategy01 Stage06：高质量多状态复合物验证集、规模化 Boltz2 数据生产、AE latent 实测与 pilot 微调报告

## 1. 阶段目标

Stage06 的目标不是直接启动 `>=1000` 条大规模微调，而是先把下面四个闭环做实：

1. 构建高质量、多 family、多 binder type 的实验多状态验证集 `V_exact`。
2. 用真实 protein/peptide binder + Boltz2 建立可扩展的 predictor-derived 训练数据工厂 `T_prod`。
3. 把 predictor-derived 样本中的 `local_latents` 从 geometry proxy 替换成真实 Complexa AE `encoder.mean`。
4. 在真实 AE latent 数据上完成 `1-sample / 4-sample / pilot` 三层训练、完整收敛图、算力/时间评估，并据此判断是否进入下一阶段的大规模数据生产。

本报告记录 Stage06 本轮实施的全部关键过程、代码改动、报错与修复、结果与下一步判断。

## 2. 环境、仓库与保护规则

- 策略仓：
  - `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase`
- baseline 仓与 baseline checkpoint 保持只读，未被修改。
- Stage06 继续使用 Stage03 的只读初始化 checkpoint：
  - `ckpts/stage03_multistate_loss/complexa_init_readonly_copy.ckpt`
  - `ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt`
- GPU 规则：
  - 本阶段全部按 `1x A100 40G` 运行
  - CUDA 作业固定 `new` 分区，落在 `gu02`
- Git/数据规则：
  - 大体积 Boltz 输出、训练 checkpoint、tensor 数据不进入 git
  - 代码、报告、关键 probe JSON、复现脚本进入 git

## 3. 本阶段代码改动

### 3.1 多状态模型补齐 baseline-compatible sampling outputs

修改文件：
- `src/proteinfoundation/nn/local_latents_transformer_multistate.py`

改动内容：
- 在原有多状态输出之外，新增 baseline-compatible clean prediction keys：
  - `bb_ca`
  - `local_latents`
- 这两个 shared clean prediction 由各 state 的 `bb_ca_states / local_latents_states` 按
  `target_state_weights * state_present_mask` 做归一化加权平均得到。

改动意义：
- 让多状态模型可以继续走原始 Complexa 的 flow-matching sampling 接口。
- 不改变多状态监督主线，但使得 Stage06 后续把 multistate model 当成“可采样生成器”成为可能。

已验证：
- probe 确认 `nn_out` 中同时存在：
  - `seq_logits_shared`
  - `bb_ca_states`
  - `local_latents_states`
  - `bb_ca`
  - `local_latents`

### 3.2 Boltz2 adapter 扩展

修改文件：
- `scripts/strategy01/stage04_predictors/boltz_adapter.py`

改动内容：
- `BoltzRunConfig` 新增：
  - `use_msa_server`
- `build_command()` 支持追加：
  - `--use_msa_server`
- `predict(...)` 新增：
  - `contact_constraints`

改动意义：
- Stage06 真实复合物数据生产需要：
  - MSA server 支撑更稳的复合物预测
  - anchor-guided contact constraints 做最后一轮界面纠偏

### 3.3 Stage06 exact validation 挖掘脚本

新增文件：
- `scripts/strategy01/stage06_mine_multistate_validation.py`

功能：
- 从实验结构中挖掘：
  - 同一 target
  - 同一 binder
  - 同一 binder sequence
  - 多状态实验复合物
- 输出统一 Stage06 manifest

最终结果：
- manifest：
  - `data/strategy01/stage06_validation/V_exact_manifest.json`
- summary：
  - `reports/strategy01/probes/stage06_validation_mining_summary.json`

### 3.4 Stage06 规模化 Boltz2 数据生产脚本

新增文件：
- `scripts/strategy01/stage06_build_boltz_production_dataset.py`

功能：
- 读取真实 multistate target + 真实 shared binder
- 对每个 state 用 Boltz2 做逐态复合物预测
- 多轮 pass：
  - `scout`
  - `second`
  - `anchor`
- 产出训练标签：
  - binder CA
  - interface contacts
  - distance labels
  - clash
  - quality proxy
- 输出：
  - `stage06_predictor_pilot_samples.pt`
  - `T_prod_train.pt`
  - `T_prod_val.pt`
  - manifests
  - summary JSON

Stage06 期间加过的关键修复：
- target template 先重写为带 `SEQRES` 的 target-only PDB，修复 Boltz 模板解析 `IndexError`
- 单个 pass 异常不再中止整个 seed
- `msa=""` 配合 `--use_msa_server` 真正启用远程 MSA
- Bronze 判定改成四舍五入比较，避免边界误杀
- `val_target <= 0` 时不再错误分配验证 family
- 允许 partial-state acceptance：`K=3` seed 只要有 `>=2` 个 states 过关，仍可产生 `K=2` 样本

### 3.5 Stage06 收敛图脚本

新增并本轮修正文件：
- `scripts/strategy01/stage06_plot_training_curves.py`

本轮修正内容：
- 补入 `overfit4`
- 将 loss key 正确映射到 Stage03/04/05 以来的 `_justlog` 新字段：
  - `multistate_struct_justlog`
  - `multistate_seq_justlog`
  - `multistate_mean_justlog`
  - `multistate_cvar_justlog`
  - `multistate_var_justlog`
  - `multistate_contact_justlog`
  - `multistate_distance_justlog`
  - `multistate_quality_proxy_justlog`
  - `multistate_clash_justlog`
- 新增 per-state 曲线：
  - `state0_total`
  - `state1_total`
  - `state2_total`

作用：
- 现在可以对 `overfit1 / overfit4 / mini` 三段训练都画出完整 raw + EMA、linear/log 曲线。

### 3.6 Stage05 predictor pilot debug 扩展为 Stage06 三层训练入口

修改文件：
- `scripts/strategy01/stage05_predictor_pilot_debug.py`

改动内容：
- 新增 `overfit4` 阶段参数：
  - `--overfit4-steps`
  - `--overfit4-samples`
  - `--overfit4-batch-size`
- 训练结果现在包括：
  - `overfit1`
  - `overfit4`
  - `mini`

改动意义：
- 与 Stage06 规格对齐，不再只有 `1-sample + mini`。
- 明确把“4 样本可否拟合”单独拿出来看，有利于区分：
  - plumbing 是否正常
  - 多状态损失是否可学
  - tiny real-data 是否太少

## 4. V_exact：高质量实验多状态验证集（已完成）

文件：
- `data/strategy01/stage06_validation/V_exact_manifest.json`

统计结果：
- exact samples：`24`
- families：`22`
- binder types：
  - peptide：`18`
  - protein：`6`
- state count：全部 `K=3`
- motion bins：
  - small：`9`
  - medium：`8`
  - large：`7`

解释：
- Stage06 已经满足并超过 exact validation 的最低门槛：
  - `>=16` exact
  - `>=8` families
- 这一批验证集不再是单一 target demo，而是覆盖多 family、多态幅度、多 binder type 的真实实验集。
- 当前 exact 数量已经达到目标 `24`，因此本阶段不需要再用 `V_hybrid` 去“补数量”。

## 5. predictor-derived 生产：pilot8 正式结果

SLURM 作业：
- job `1907411`
- 分区：`new`
- 节点：`gu02`
- 运行时间：`01:28:56`

summary：
- `reports/strategy01/probes/stage06_pilot8_build_summary.json`

正式结果：
- `base_seed_count = 8`
- `accepted_bases = 2`
- `rejected_bases = 6`
- `accepted_variants = 16`
- `train_count = 8`
- `val_count = 4`
- `hybrid_built_count = 0`

失败种子：
- `1hls_B_A`
- `1hv2_A_B`
- `1i5h_W_B`
- `1ibx_B_A`
- `1jgn_A_B`
- `1jh4_A_B`

成功种子：
- `1jco_B_A`
- `1jeg_A_B`

说明：
- 8 个真实 multistate target+binder seeds 里，2 个能够完整通过多状态 Bronze 生产门槛。
- 当前 acceptance rate 约 `25%`（按 base seed 计）。
- 虽然总数还不大，但它已经是“真实多状态 target + 真实 shared binder + Boltz2 复合物 + 真实 AE latent”的闭环数据，而不是 synthetic toy set。

## 6. AE latent 批处理（已完成）

命令入口：
- `scripts/strategy01/stage05_extract_ae_latents.py`

输入：
- `data/strategy01/stage06_production_pilot8/stage06_predictor_pilot_samples.pt`

输出：
- `data/strategy01/stage06_production_pilot8/stage06_predictor_pilot_samples_ae_latents.pt`
- `reports/strategy01/probes/stage06_pilot8_ae_latent_extract_summary.json`

结果：
- `num_samples_total = 12`
- `processed_states = 28`
- `device = cpu`
- `elapsed_sec = 2.343`

解释：
- Stage06 当前这 12 条样本已经全部完成 geometry proxy -> 真实 AE `encoder.mean` 替换。
- 所以本轮训练用的 `local_latents` 已经不是临时几何占位，而是 Complexa AE 的真实编码。

## 7. 数据组成分析

从 `stage06_predictor_pilot_samples_ae_latents.pt` 统计得到：

- split：
  - train：`8`
  - val：`4`
- K 分布：
  - 全部样本中：`K=2 -> 8`，`K=3 -> 4`
  - train：`K=2 -> 6`，`K=3 -> 2`
  - val：`K=2 -> 2`，`K=3 -> 2`
- source sample 分布：
  - `1jco_B_A -> 8`
  - `1jeg_A_B -> 4`
- 当前 train/val family 实际上退化成：
  - train 基本全来自 `1jco_B_A`
  - val 基本全来自 `1jeg_A_B`

科学含义：
- 这批数据足以验证 Stage06 loss / AE latent / 真实 predictor-derived train loop 是否能跑通并下降。
- 但它还不足以代表“泛化能力已经成立”，因为 train/val 只有 2 个 base families。
- 这也是为什么本阶段的训练结果主要用于判断：
  - 结构是否可学
  - loss 是否有效
  - 工程链路是否完整
- 而不能直接把当前 validation 数字当成最终模型性能结论。

## 8. GPU 训练：真实 AE latent 数据上的三层训练结果

训练脚本：
- `scripts/strategy01/stage05_predictor_pilot_debug.py`

本轮新增阶段：
- `overfit1`
- `overfit4`
- `mini`

GPU 作业：
- job `1907412`

结果 JSON：
- `reports/strategy01/probes/stage06_pilot8_debug_results.json`

### 8.1 loss / interface / grad probes

全部通过：
- `loss_unit`
- `synthetic_interface`
- `grad_route`

说明：
- 多状态 loss 路径、interface 项、梯度路由都是真实可训练的。
- 关键多状态模块和 interface head 梯度非零。

### 8.2 overfit1

结果：
- 初始 eval total：`24.9242`
- 最终 eval total：`22.9171`
- 最佳 eval total：`21.8115 @ step 100`
- drop fraction：`0.0805`
- step time：`0.3474 s`
- 显存：`6.303 GB`

判断：
- **可以下降，但下降不够强。**
- 这说明：
  - plumbing 没坏
  - 但单样本过拟合强度不足

### 8.3 overfit4

结果：
- 初始 eval total：`20.8643`
- 最终 eval total：`17.5717`
- 最佳 eval total：`16.9345 @ step 200`
- drop fraction：`0.1578`
- step time：`0.2996 s`
- 显存：`6.303 GB`

判断：
- **4-sample 比 1-sample 更稳定，但仍然不是“轻松吃透”的程度。**
- 说明多状态真实样本可以学，但当前训练设置下仍偏硬。

### 8.4 mini（8 train samples）

结果：
- 初始 eval total：`19.7855`
- 最终 eval total：`7.8262`
- 最佳 eval total：`7.2407 @ step 900`
- drop fraction：`0.6045`
- step time：`0.2914 s`
- 显存：`6.303 GB`
- 总训练时间：`291.4 s`（1000 steps，含评估）

判断：
- **mini 阶段有明显有效学习信号。**
- `L_total` 下降超过 `60%`，并且中期最低值接近 `7.24`，说明真实 AE latent predictor-derived 样本上的 Stage06 loss 不只是“能跑”，而是真能驱动拟合。

### 8.5 validation

validation total：`55.2460`
validation cvar：`55.6035`

判断：
- validation 明显比 train/mini 高很多。
- 在当前数据组成下，这并不意外：
  - train 基本来自 `1jco_B_A`
  - val 基本来自 `1jeg_A_B`
- 所以这里反映的是：
  - 当前 pilot 数据量太小
  - family diversity 太低
  - 还不能证明跨 family 泛化

## 9. 收敛图（已完成）

图目录：
- `reports/strategy01/figures/stage06/pilot8_debug`

图数量：
- `99` 张

覆盖内容：
- `overfit1`
- `overfit4`
- `mini`

每个 phase 均包含：
- `train_total`
- `eval_total`
- `multistate_total`
- `multistate_struct`
- `multistate_seq`
- `multistate_mean`
- `multistate_cvar`
- `multistate_var`
- `interface_contact`
- `interface_distance`
- `interface_quality_proxy`
- `interface_clash`
- `state0_total`
- `state1_total`
- `state2_total`
- `cuda_max_mem_gb`
- `elapsed_sec`

并同时提供：
- linear-y
- log-y
- raw + EMA

## 10. Boltz2 runtime 与规模估计

### 10.1 真实 pilot8 端到端生产 walltime

`1907411` 总运行时间：
- `01:28:56 = 5336 s`

对应规模：
- `8` 个 base seeds
- 每个 seed 最多 `3` 个 target states
- 总共尝试了 `24` 个 state-level contexts
- 生成了 `69` 个 Boltz log/pass 文件

据此可得更有工程意义的 walltime 标尺：
- `~667 s / seed`（平均）
- `~222 s / state-context`（按 24 个 state 计）
- `~77 s / Boltz pass log`（按 69 个 pass 计）

### 10.2 从日志中抽取到的 core runtime

文件：
- `reports/strategy01/probes/stage06_pilot8_runtime_summary.json`

日志直接解析得到的近似核心耗时（不包含整个 job 的全部等待/重试开销）：
- median by pass：
  - scout：`8.0 s`
  - second：`8.5 s`
  - anchor：`10.0 s`
- 按输入总长度分组：
  - small：`8.0 s`
  - medium：`10.0 s`

解释：
- 日志中解析出的 `8-10s/pass` 更接近 Boltz 内核和局部预处理时间。
- 真正的生产 walltime 应优先参考整任务统计，因为它已经包含：
  - 多 pass 重试
  - MSA server 往返
  - I/O 与解析
  - builder 级 Python 开销

### 10.3 1000 sample 外推

当前 pilot8 的经验值：
- `8 seeds -> 16 accepted variants -> 12 kept in 8/4 split`
- 如果按 accepted variants 算，平均约 `2 variants / seed`
- 若目标是 `1000` 条最终样本，粗略需要：
  - `~500 seeds`

用当前真实 walltime 外推数据生产：
- 1 GPU：
  - `500 seeds * 667 s ≈ 333,500 s ≈ 92.6 h ≈ 3.9 天`
- 2 GPUs：
  - 理想均分约 `46.3 h ≈ 1.9 天`

说明：
- 当前阶段真正的瓶颈已经不是训练，而是 **高质量 predictor-derived 数据生产**。

## 11. 训练耗时与显存估计

### 11.1 本轮实测

- `overfit1`：
  - `300 steps`
  - `98.1 s`
  - `0.347 s/step`
  - `6.303 GB`
- `overfit4`：
  - `600 steps`
  - `173.3 s`
  - `0.300 s/step`
  - `6.303 GB`
- `mini`：
  - `1000 steps`
  - `291.4 s`
  - `0.291 s/step`
  - `6.303 GB`

### 11.2 1000-sample fine-tune 估计

如果后续数据集已经准备好，且仍沿用当前长度范围、batch=1、1 卡 A100：

- `3000 steps`：
  - 纯训练估计 `~15 min`
  - 含评估/日志/保存，保守估计 `20-30 min`
- `6000 steps`：
  - 纯训练估计 `~29 min`
  - 含评估/日志/保存，保守估计 `35-60 min`

2 卡 A100（DDP）目前没有在 Stage06 实测，但按当前模型规模与 batch 设置粗略估计：
- `3000 steps`：`~11-20 min`
- `6000 steps`：`~22-40 min`

结论：
- **1000 sample 的训练本身并不贵。**
- 真正贵的是前面的高质量数据生产和后面的 refold/benchmark。

## 12. 错误与修复日志

### 12.1 RCSB result_set 解析错误

现象：
- `TypeError: string indices must be integers, not 'str'`

根因：
- RCSB `result_set` 有时是字符串列表，不是 dict 列表。

修复：
- 挖掘脚本兼容两种格式。

### 12.2 exact validation 混入 homooligomer/self-interface

现象：
- 初始候选中出现 target/binder 近乎相同的 self-pair。

修复：
- 排除 target/binder identical 或高 identity 且同长度的 self-interface 候选。

### 12.3 Boltz 模板解析 `IndexError`

根因：
- 直接抽出的 target-only PDB 不满足 Boltz 模板输入期望。

修复：
- 使用 `write_chain_pdb(...)` 重写带 `SEQRES` 的 target-only template。

### 12.4 anchor pass SVD 异常中断 builder

现象：
- `torch._C._LinAlgError: linalg.svd ... failed to converge`

修复：
- Stage06 builder 逐 pass 捕获异常，单 pass 失败不再终止整个 seed。

### 12.5 auto-MSA 使用方式错误

现象：
- 仅在 YAML 写 `msa: ""` 不能真正启用远程 MSA。

修复：
- adapter 显式新增并启用 `--use_msa_server`。

### 12.6 Bronze 边界误杀

现象：
- `0.5467` vs `0.55` 这类边界值会误杀几乎合格的样本。

修复：
- Bronze 判定使用四舍五入比较。

### 12.7 val_target = 0 时 split 错误

现象：
- 无验证目标时，旧逻辑仍可能保留验证 family。

修复：
- `val_target <= 0` 时直接不分配 val family。

### 12.8 PowerShell -> SSH -> Bash 复杂命令串扰

本轮复现了两类：
- `$(basename ...)`
- `@hydra.main`

根因：
- 本地 PowerShell 会先解析 `@`、`$()` 等特殊语法，污染远程命令。

修复策略：
- 凡是复杂远程命令统一改成：
  - 本地临时脚本
  - `scp`
  - 远程 `python/bash` 执行

### 12.9 sbatch 脚本 BOM 问题

现象：
- `sbatch: This does not look like a batch script`
- 第一行 shebang 未被识别

根因：
- Windows 写出的 `.sbatch` 文件带 BOM

修复：
- 使用 **UTF-8 no BOM + LF** 重写后重新提交，成功得到 job `1907412`

### 12.10 Stage06 中途解析脚本链长 bug

现象：
- 第一次中途验收得到夸张 contact 数和 `interchain_pae = null`

根因：
- 临时脚本错误使用 padding 后长度，而不是 `source_residue_count`

修复：
- 改用真实 residue count，之后得到合理的 Bronze/非 Bronze 中途结果。

### 12.11 远程默认 Python 无 torch

现象：
- 统计样本构成时 `ModuleNotFoundError: No module named 'torch'`

根因：
- 远程默认 `python` 不是 `proteina-complexa` 环境。

修复：
- 统一改用：
  - `/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python`

## 13. 这次结果说明了什么

### 13.1 已经证明成立的部分

1. **V_exact 已经建成。**
   多 family、多运动尺度、多 binder type 的实验多状态验证集已经到位。

2. **真实 predictor-derived -> AE latent -> multistate loss -> GPU train 闭环已打通。**
   不再是 toy data 或 purely synthetic debug。

3. **Stage06 loss 在真实数据上是可学的。**
   mini 阶段 `L_total` 下降超过 `60%`，说明这条 loss 设计有实质训练信号。

4. **算力瓶颈不在 fine-tune，而在数据生产。**
   1 张 A100 足以完成当前架构的 predictor-derived pilot 训练；更耗时的是高质量 Boltz2 复合物生产。

### 13.2 还没有被证明的部分

1. **跨 family 泛化还没有证明。**
   目前 train/val 只有 `1jco` vs `1jeg` 两个 base seeds。

2. **B0/B1 生成复评还没有在本阶段完成。**
   原因不是训练没保存，而是当前生成链路还缺少一个“把 `mini_final.pt` 这类 `model_state_dict` 包装进 Strategy01 生成入口”的推理脚手架。

3. **当前 pilot 数据还不足以进入 Stage07。**
   12 条样本只能证明工程闭环与 loss 有效，不能直接支撑大规模泛化微调结论。

## 14. 为什么 overfit1 / overfit4 不够强，而 mini 明显下降

这是本阶段最值得认真解释的现象。

### 14.1 不是 plumbing 坏了

因为：
- `loss_unit` 通过
- `synthetic_interface` 通过
- `grad_route` 通过
- `mini` 明显下降

所以不是“模型没接上”或“梯度没传到”。

### 14.2 更可能的原因

1. **单样本/4样本仍在承受完整多状态 interface 目标，约束过硬。**
   当前 loss 不只是 `fm`，还包含：
   - contact
   - distance
   - clash
   - anchor persistence
   - quality proxy
   对 tiny real-data overfit 来说，这比纯结构/纯序列 overfit 更难。

2. **本轮 still 使用相当多 trainable parameters。**
   新增模块 + 顶层 trunk 在小数据上不一定能快速压到极低 loss。

3. **当前 pilot 数据来自两个 base seeds 的 2/3-state 变体。**
   这些变体之间存在共享结构，但也有真实多状态差异；因此“单样本轻松背下来”不会像 synthetic debug 那样容易。

4. **validation 很高说明 family shift 很强。**
   这也侧面说明 Stage06 当前更像“闭环可行性证明”，而不是“泛化结果已经成熟”。

## 15. B0/B1 当前状态与下一步缺口

本阶段没有完成 `V_exact-only` 上的正式 B0/B1 生成复评。

当前阻塞点：
- `stage06_pilot8_debug` 已经保存了：
  - `ckpts/stage05_predictor_pilot/runs/stage06_pilot8_debug/mini_final.pt`
- 但它的格式是：
  - `{'model_state_dict', 'phase', 'steps'}`
- 还没有一个现成的 Stage06 inference wrapper 把这个 `model_state_dict` 装进多状态生成入口，并对 `V_exact` 逐 target 产生 shared binder 候选。

因此当前最合理的工程判断是：
- **Stage06 已经完成数据/训练/收敛图闭环，但还没完成生成 benchmark 闭环。**
- 进入下一阶段前，应先补一个：
  - `stage06_export_or_load_generation_checkpoint.py`
  - 或 `stage06_generate_multistate_candidates.py`
- 然后再做：
  - B0：baseline single-state generation + Boltz2 refold
  - B1：Strategy01 multistate generation + Boltz2 refold

## 16. 阶段结论与下一步建议

### 16.1 本阶段结论

Stage06 到目前为止是 **有效推进**，但还 **不应该直接扩到 1000 sample 微调**。

原因：
- 好消息：
  - exact validation 到位
  - predictor-derived 真实数据工厂跑通
  - AE latent 接入成功
  - 1 卡 A100 训练闭环成立
  - mini loss 有明显下降
- 需要继续补强：
  - 数据 family 多样性远远不够
  - validation 还高
  - B0/B1 生成 benchmark 尚未打通

### 16.2 下一步最优先事项

优先顺序建议：

1. **先补生成侧 checkpoint wrapper**
   - 让 `mini_final.pt` 能进入 Strategy01 真实生成链路
   - 这是完成 B0/B1 的必要前提

2. **把 predictor-derived 数据从 2 个 accepted bases 扩到 >=16 accepted bases**
   - 不是盲目扩到 1000 样本
   - 先做到 `64 train + 16 val` 且至少 `8-10 families`

3. **再重跑 pilot fine-tune + exact-val/B0/B1**
   - 如果 exact-val 仍没有改善，再回头修 loss/数据/anchor strategy
   - 如果 exact-val 开始改善，再进入更大规模 `500-1000 curated samples`

## 17. 关键输出路径

### 17.1 数据
- `data/strategy01/stage06_validation/V_exact_manifest.json`
- `data/strategy01/stage06_production_pilot8/stage06_predictor_pilot_samples.pt`
- `data/strategy01/stage06_production_pilot8/stage06_predictor_pilot_samples_ae_latents.pt`
- `data/strategy01/stage06_production_pilot8/T_prod_train.pt`
- `data/strategy01/stage06_production_pilot8/T_prod_val.pt`

### 17.2 probes / results
- `reports/strategy01/probes/stage06_validation_mining_summary.json`
- `reports/strategy01/probes/stage06_pilot8_build_summary.json`
- `reports/strategy01/probes/stage06_pilot8_ae_latent_extract_summary.json`
- `reports/strategy01/probes/stage06_pilot8_runtime_summary.json`
- `reports/strategy01/probes/stage06_pilot8_debug_results.json`

### 17.3 figures
- `reports/strategy01/figures/stage06/pilot8_debug/`

### 17.4 checkpoints
- `ckpts/stage05_predictor_pilot/runs/stage06_pilot8_debug/overfit1_final.pt`
- `ckpts/stage05_predictor_pilot/runs/stage06_pilot8_debug/overfit4_final.pt`
- `ckpts/stage05_predictor_pilot/runs/stage06_pilot8_debug/mini_final.pt`