# Strategy01 第三阶段报告：多状态 Loss、32 条调试集与微调闭环

日期：2026-04-16
策略仓：`/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase`
基线仓：`/data/kfliao/general_model/strategy_repos/proteina-complexa-benchmark-baseline`

## 1. 阶段目标

本阶段目标是把 Strategy01 的多状态训练闭环从“架构可前向”推进到“loss、数据、probe、overfit、32-sample debug fine-tune 均真实跑通”。本阶段不修改 benchmark baseline 仓，也不覆盖原始 baseline checkpoint。

完成状态：

| 项目 | 结果 |
|---|---|
| baseline 仓 | 未改动 |
| baseline checkpoint | 未移动、未覆盖、未重写 |
| 阶段只读 checkpoint 副本 | 已创建并设为只读 |
| 多状态 loss 路径 | 已接入 `ProductSpaceFlowMatcher` 和 `Proteina.training_step()` |
| 12/32 条调试集 | 已生成 stage03 工程调试伪标注集 |
| loss 单元 probe | 通过 |
| worst-state 敏感性 probe | 通过 |
| 梯度路由 probe | 通过 |
| oracle loss probe | 通过，理论正确输出 loss 约 `2.7e-08` |
| 1-sample overfit | 通过，loss 下降 `99.79%` |
| 4-sample overfit | 通过，loss 下降 `99.94%` |
| 32-sample debug fine-tune | 跑满 1500 steps，训练 loss 从 `4.40` 降到 `1.04` |

## 2. Checkpoint 保护

基线只读来源：

- `/data/kfliao/general_model/strategy_repos/proteina-complexa-benchmark-baseline/ckpts/complexa.ckpt`
- `/data/kfliao/general_model/strategy_repos/proteina-complexa-benchmark-baseline/ckpts/complexa_ae.ckpt`

本阶段策略仓副本：

- `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/ckpts/stage03_multistate_loss/complexa_init_readonly_copy.ckpt`
- `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt`

规则执行结果：

- 副本使用 `cp -n` 创建，避免覆盖已有文件。
- 副本执行 `chmod 444`，作为本阶段初始化权重只读来源。
- 训练输出写入 `ckpts/stage03_multistate_loss/runs/`。
- 大 checkpoint 目录被 `.gitignore` 忽略，不纳入 Git 提交。
- 为避免用户 `/data` quota 爆掉，仅保留本阶段必要最终 checkpoint，删除了重复中间 `.pt`。

## 3. 多状态 Loss 设计

本阶段样本固定为 3 个 `required_bind` 状态。每个状态都有一个 binder backbone endpoint 和 local latent endpoint，所有状态共享一个 binder 序列。

每状态结构项：

```text
L_state_k = lambda_bb * L_fm_bb_k + lambda_lat * L_fm_lat_k
```

共享序列项：

```text
L_seq = CE(seq_logits_shared, binder_seq_shared, binder_seq_mask)
```

鲁棒聚合：

```text
L_mean = sum_k w_k * L_state_k
L_cvar = mean(top-2 L_state_k)
L_var = weighted_variance(L_state_k)
```

总损失：

```text
L_total = lambda_seq * L_seq + lambda_struct * (alpha * L_mean + beta * L_cvar + gamma * L_var)
```

默认系数：

| 系数 | 值 |
|---|---:|
| `lambda_bb` | 1.0 |
| `lambda_lat` | 1.0 |
| `lambda_seq` | 0.5 |
| `lambda_struct` | 1.0 |
| `alpha_mean` | 0.5 |
| `beta_cvar` | 0.4 |
| `gamma_var` | 0.1 |
| `cvar_topk` | 2 |

科学意义：

- `mean` 保证整体三态平均可学。
- `CVaR/top-k worst` 强化最差状态，避免平均值掩盖坏状态。
- `variance` 惩罚状态间偏科，促使同一个 binder 序列在三个 required states 上更均衡。
- Pareto 不进入反向传播，只作为后续搜索/诊断视图，避免训练主目标不可微或不稳定。

## 4. 关键代码改动

### 4.1 新增多状态 loss 文件

文件：

- `src/proteinfoundation/flow_matching/multistate_loss.py`

新增能力：

- `corrupt_multistate_batch(flow_matcher, batch)`
- `compute_multistate_loss(flow_matcher, batch, nn_out)`
- `_masked_topk_mean`
- `_weighted_variance`
- `_sequence_loss`

重要修正：

- 初版每个 state 独立采样 `x0/t`，但模型输入只有一个共享 binder `x_t`，导致每个 state 的随机噪声是隐藏变量，结构 overfit 无法真正收敛。
- 修正后同一个样本内 3 个 state 共享 `x0/t`，再分别插值到各自 `x_1_state_k`。
- 这样符合共享 binder 序列的物理语义：同一个 binder 从同一 corruption path 学习不同 state endpoint，而不是让模型猜不可见的 per-state 随机噪声。

### 4.2 接入 ProductSpaceFlowMatcher

文件：

- `src/proteinfoundation/flow_matching/product_space_flow_matcher.py`

新增方法：

```python
corrupt_multistate_batch(self, batch)
compute_multistate_loss(self, batch, nn_out)
```

意义：

- 不破坏原单状态 `corrupt_batch()` 和 `compute_loss()`。
- 多状态训练路径显式调用新方法，不会静默 fallback 到单状态。

### 4.3 接入 Proteina.training_step()

文件：

- `src/proteinfoundation/proteina.py`

改动：

- 支持 `cfg_exp.nn.name == "local_latents_transformer_multistate"`。
- 当 `loss.multistate.enabled=True` 时走多状态训练路径。
- 强制 `n_recycle=0` 和 `self_cond=False`，不允许本阶段隐藏复杂条件。

意义：

- 保留原始单状态训练链路不变。
- 多状态路径独立、可开关、可 probe。

### 4.4 新增训练配置

文件：

- `configs/training_local_latents_multistate_loss_probe.yaml`

关键点：

- checkpoint 显式指向 stage03 只读副本。
- `loss.multistate.enabled=True`。
- `target_dropout_rate=0.0`、`motif_dropout_rate=0.0`、`n_recycle=0`、`self_cond=False`。

### 4.5 新增 stage03 调试脚本

文件：

- `scripts/strategy01/stage03_multistate_loss_debug.py`
- `scripts/strategy01/stage03_oracle_loss_probe.py`

功能：

- 创建 12/32 条工程调试集。
- 运行 loss unit probe、worst-state probe、gradient route probe。
- 运行 stage A、1-sample overfit、4-sample overfit、32-sample debug fine-tune。
- 保存结果 JSON。

## 5. Batch / Loss Schema 前后对比

旧单状态训练输入核心字段：

```text
x_1['bb_ca'] [B,Nb,3]
x_1['local_latents'] [B,Nb,Z]
mask [B,Nb]
```

新增多状态训练输入核心字段：

```text
binder_seq_shared [B,Nb]
binder_seq_mask [B,Nb]
x_1_states['bb_ca'] [B,K,Nb,3]
x_1_states['local_latents'] [B,K,Nb,Z]
state_mask [B,K,Nb]
state_present_mask [B,K]
target_state_weights [B,K]
target_state_roles [B,K]
x_target_states [B,K,Nt,37,3]
target_mask_states [B,K,Nt,37]
seq_target_states [B,K,Nt]
target_hotspot_mask_states [B,K,Nt]
```

模型输出契约：

```text
seq_logits_shared [B,Nb,20]
bb_ca_states [B,K,Nb,3]
local_latents_states [B,K,Nb,Z]
```

本阶段实测：

```text
B=2, K=3, Nb=24, Nt=48, Z=8
seq_logits_shared: [2,24,20]
bb_ca_states: [2,3,24,3]
local_latents_states: [2,3,24,8]
```

## 6. 32 条调试集构造

目标起点：

- `1tnf`
- `3di3`
- `5o45`

每条样本包含：

- 1 条 target canonical sequence 近似表示。
- 3 个 target state 结构。
- 1 条共享 binder sequence。
- 3 个 target-binder complex 的 binder state 伪结构监督。
- 3 个 state 权重，均匀权重。
- 3 个 state role，全部为 `required_bind`。

现实说明：

- target state 尽量读取 benchmark baseline 仓中的真实 PDB。
- `1tnf_cropped_fixed.pdb` 是空文件，因此该 state 使用前一 state 的 fallback，并在 manifest 中记录。
- binder complex 监督是工程伪标注，用来验证 loss、shape、gradient、overfit 和训练闭环；它不是生物学效果 benchmark，不应用来声称真实设计性能。

调试集路径：

- `data/strategy01/stage03_multistate_debug/manifest_stage03_debug.json`
- `data/strategy01/stage03_multistate_debug/stage03_debug_samples.pt`

注意：`data/` 当前被 `.gitignore` 忽略，因此 Git 只提交生成脚本和报告，不提交 `.pt` 数据。

## 7. 实测结果

### 7.1 Probe

结果文件：

- `reports/strategy01/probes/stage03_shared_corruption_probes_results.json`
- `reports/strategy01/probes/stage03_oracle_loss_probe.json`

关键结果：

| Probe | 结果 |
|---|---|
| loss unit | `multistate_total=19.99`，无 NaN |
| worst-state sensitivity | robust delta `1.7006` > mean delta `1.2500` |
| missing state mask | absent state loss 变为 `0.0`，总 loss 非 NaN |
| gradient route | cross-state fusion、target-to-binder cross-attn、shared seq head、CA/local latent heads 梯度非零 |
| oracle loss | `multistate_total=2.704e-08` |

### 7.2 Stage A: loss plumbing smoke

结果文件：

- `reports/strategy01/probes/stage03_shared_corruption_full_300_500_800_1500_results.json`

结果：

| 指标 | 值 |
|---|---:|
| steps | 300 |
| batch | 2 |
| initial eval total | 27.482 |
| final eval total | 2.185 |
| drop fraction | 92.05% |
| final struct | 0.705 |
| final state losses | 0.712 / 0.762 / 0.838 |
| trainable params | 178.79M |

### 7.3 1-sample overfit

结果：

| 指标 | 值 |
|---|---:|
| steps | 500 |
| initial eval total | 2.388 |
| final eval total | 0.0050 |
| drop fraction | 99.79% |
| final seq CE | 0.00023 |
| final struct | 0.00492 |
| final state losses | 0.00312 / 0.00622 / 0.00578 |

结论：通过 `>=60%` 下降门槛，且三个 state 都下降，不是 easiest-state 单独下降。

### 7.4 4-sample overfit

结果：

| 指标 | 值 |
|---|---:|
| steps | 800 |
| initial eval total | 3.967 |
| final eval total | 0.00238 |
| drop fraction | 99.94% |
| final seq CE | 0.00014 |
| final struct | 0.00231 |
| final state losses | 0.00201 / 0.00210 / 0.00298 |

结论：通过，且三个状态均衡下降。

### 7.5 32-sample debug fine-tune

结果：

| 指标 | 值 |
|---|---:|
| steps | 1500 |
| batch | 2 |
| train total step 1 | 4.402 |
| train total step 1500 | 1.044 |
| final eval total | 1.596 |
| final struct | 0.524 |
| final mean | 0.573 |
| final CVaR | 0.593 |
| final variance | 0.00190 |
| final state losses | 0.532 / 0.553 / 0.633 |
| max GPU memory | 5.72GB |
| total elapsed | 645.8s |

解释：

- 32-sample 阶段从前面的 overfit 阶段继续，因此固定 first-two eval 初值非常低，不适合作为“泛化下降”指标。
- 更合理的观察是训练 batch loss 从 `4.402` 降到 `1.044`，结构项最终保持在 `0.524`，三个 state 没有出现一个 state 独降、另一个 state 崩掉。
- 下一阶段应把 stage A、overfit diagnostic、32-sample debug fine-tune 拆成独立 run，避免 overfit 阶段污染 debug32 初始点。

## 8. GPU 与耗时

最终成功 run 使用 1 张 GPU。原因：

- 本阶段脚本是单进程调试闭环，不需要 DDP。
- `srun --immediate` 指定 gu02 时曾遇到 `.to(cuda)` OOM，说明被分到的卡并非干净空闲。
- 取消指定节点后，SLURM 自动分配可用 GPU，完整 run 成功。

实测：

- 完整 300/500/800/1500 steps 用时 `645.8s`，约 10.8 分钟。
- 峰值显存约 `5.72GB`。
- 当前 24-res binder / 48-res target 调试尺寸在 1 张 40G A100 上非常充裕。

## 9. 错误与修正日志

| 错误现象 | 根因 | 修正动作 | 修后验证 |
|---|---|---|---|
| PowerShell + ssh heredoc 多次失败 | 本地 PowerShell 解析引号、管道、heredoc，导致远程 Python 被截断 | 改为上传脚本后远程执行，或使用单层命令 | 后续主脚本均用 `scp + ssh/srun` |
| `git apply` 小补丁失败 | patch 不是标准 unified diff 或 hunk 计数不对 | 改用确定性临时 Python 脚本替换目标块 | 文件编译通过 |
| missing state probe 出现 NaN | absent state 的 `nres=0` 先进入 FM loss，`NaN*0` 仍是 NaN | `_state_loss_from_flat` 改为 `torch.where(state_present, loss, 0)` | 缺失 state loss 为 `0.0`，总 loss 非 NaN |
| 1-sample overfit 初版只降 13.46% | 每个 state 独立采样 `x0/t`，但模型只看到共享 `x_t`，隐藏噪声不可预测 | 改成同一样本内 K state 共享 corruption path | overfit 下降 99.79% |
| oracle GPU 运行 CUDA driver 初始化失败 | 分配环境 CUDA 初始化异常 | oracle 改用 CPU 强制执行 | oracle loss `2.704e-08` |
| gu02 指定节点 full run OOM | 指定节点时可能拿到非空闲显存卡 | 不指定节点，使用 `srun --immediate` 让 SLURM 选可用卡 | 完整 run 成功 |
| `/data` quota exceeded | 多次 run 保存 31 个大 `.pt` checkpoint，占用约 34GB | 只保留必要最终 checkpoint，删除中间重复 `.pt` | quota 从超限恢复到安全范围 |

## 10. 本阶段结论

本阶段已经完成“多状态 loss + 工程调试集 + 首轮微调链路”的闭环验证。最关键的科学/工程修正是：多状态共享 binder 模型不能给各 state 采独立不可见噪声，而应共享 corruption path，再学习 state-specific endpoint。这个修正后，1-sample 和 4-sample overfit 都从失败变为稳定通过，说明多状态监督可以被当前 Strategy01 架构吸收。

当前结果不能解读为真实多状态 binder 设计性能提升，因为本阶段数据是工程伪标注集。它证明的是：

- loss 公式可微且 oracle 正确；
- state mask、state role/weight、CVaR 聚合路径可运行；
- 共享序列 head 和三态结构 head 有梯度；
- 小规模 fine-tune 在 1 张 A100 上可在十几分钟内跑通。

下一阶段建议：

- 把 stage A、overfit diagnostic、debug32 fine-tune 拆成独立入口，避免评估互相污染。
- 用真实或高置信 refold 伪标注 complex 替换当前工程 binder 伪结构。
- 加入 validation split，报告固定 train batch 与 held-out target/binder 的双曲线。
- 开始接入真实 `ipAE/pLDDT/clash` 后评估，而不是只看训练 loss。
