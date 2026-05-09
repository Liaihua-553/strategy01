# Strategy01 Stage13：Native State-wise Complexa Path 审计与下一步修正报告

## 1. 阶段目标

本阶段围绕同一个科学目标：**只输入 target 的多个构象，模型从零生成一个共享 binder 序列，并为每个 target state 生成合理 target-binder complex geometry**。

Stage12 的结果显示，旧 multistate wrapper 的自由 de novo sampling 失败明显：K=1 情况下 sequence identity 与界面几何都不理想。这说明问题不是多状态本身，而是代码路径可能没有真正继承原始 Proteina-Complexa 的生成先验。因此 Stage13 先做一个更基础的审计：

- K=1 时，Strategy01 wrapper 是否等价于原始 Complexa target-conditioned denoiser。
- 如果不等价，先修代码路径，而不是继续训练。
- 在恢复原始 denoiser 后，再测试共享序列 head、replay curriculum、latent repair 是否能把目标往前推进。

## 2. 核心代码改动

### 2.1 新增 native state-wise path

修改文件：`src/proteinfoundation/nn/local_latents_transformer_multistate.py`

新增逻辑：

- `stage13_native_state_path=True` 时，把 `[B,K]` 的 state tensors 展平成 `[B*K]`。
- 对每个 state 复用原始 Complexa 的 `ConcatFeaturesFactory`、`ConcatPairFeaturesFactory`、target concat、pair update、transformer layers、`ca_linear`、`local_latents_linear`。
- 再把输出 reshape 回 `bb_ca_states [B,K,Nb,3]`、`local_latents_states [B,K,Nb,Z]`、`state_seq_logits [B,K,Nb,20]`、`seq_logits_shared [B,Nb,20]`。

科学意义：这不是 pose transfer，也不是 inverse folding，而是把原始 Complexa 的 target-conditioned product-space flow 作为每个 state 的基础生成器。多状态模块后续只应该学习跨状态 coupling 和 shared sequence，而不是从小数据重新学一个新的 binder generator。

### 2.2 新增等价性 probe

新增文件：`scripts/strategy01/stage13_native_path_equivalence_probe.py`

用途：对同一份 flattened input，同时跑原始 `LocalLatentsTransformer` 和 multistate wrapper 的 native branch，比较 `bb_ca` 与 `local_latents` 输出是否逐元素一致。

实测结果：

```json
{
  "bb_ca_states_vs_original": {"max_abs": 0.0, "mean_abs": 0.0},
  "local_latents_states_vs_original": {"max_abs": 0.0, "mean_abs": 0.0}
}
```

结论：native branch 的代码实现已经逐元素等价于原始 Complexa denoiser。

### 2.3 训练与采样脚本开关

修改文件：

- `scripts/strategy01/stage12c_de_novo_smoke.py`
- `scripts/strategy01/stage13_single_state_de_novo_audit.py`
- `scripts/strategy01/stage12_de_novo_multistate_training.py`

新增参数：

- `--stage13-native-state-path`：启用 checkpoint-compatible 原始 Complexa state-wise denoiser。
- `--stage13-heads-only`：冻结 native denoiser 主体，只训练多状态/shared-sequence heads。
- `--stage13-train-latent-repair`：在 heads-only 基础上额外训练 `local_latents_linear` 与 latent sequence repair 模块。

## 3. 实验结果汇总

完整机器可读结果见：`reports/strategy01/probes/stage13_native_path_summary.json`。

| 实验 | 关键设置 | 结果 | 结论 |
|---|---|---:|---|
| native path equivalence | 原始 transformer vs wrapper native branch | `max_abs=0.0` | 代码等价，native path 没有实现错误 |
| original ckpt native K=1 rollout | `ckpts/complexa.ckpt`, native path | shared `0.018`, AE `0.050` | exact-label audit 下原始 checkpoint 不会恢复指定 binder 序列，低 identity 不等于原始模型无效 |
| initial teacher-forced probe | native path, no training | shared `0.0`, AE `0.433-0.883` | AE/local latent 有序列信息，shared head 是随机未训练 |
| heads-only training | 冻结 native denoiser，只训练 shared heads | teacher-forced shared `0.8-0.95` | shared head 可以学习标签 |
| heads-only de novo audit | heads-only ckpt rollout | shared `0.0227`, AE `0.050` | teacher-forced 好不代表 rollout 好，仍有分布错配 |
| replay heads training | 中间 replay | de novo shared `0.0545` | 有小幅改善，但不够 |
| final-state replay heads | 16-step final replay | de novo shared `0.1591` | 明显改善，但仍未达到科学目标 |
| latent-repair training | 解冻 local_latents_linear/repair | replay/teacher 指标变差 | 小样本下直接训 latent readout 会破坏 shared readout，不作为主线 |

## 4. 科学判断

### 4.1 已解决的问题

- Stage12 之前最大的代码风险已经确认：旧 multistate wrapper 替代了原始 Complexa target concat denoiser，导致 K=1 都不能代表原始 Complexa。
- Stage13 已修复为 native state-wise denoiser，并用逐元素等价性 probe 证明没有代码偏差。
- no-leak contract 仍保持：target-only de novo 模式不输入 source pose、真实 binder sequence、真实 binder optional CA/residue feature。

### 4.2 尚未达成科学目标

目前还不能说 Strategy01 已达到“共享 binder 序列兼容多状态 target”的最终目标。原因：

- de novo rollout 后 shared identity 最好只有 `0.1591`，仍远低于机制目标。
- AE-state identity 在 de novo rollout 后仍约 `0.05`，说明 sampled `local_latents` 没有落在能被 AE decoder 解出正确 sequence 的 manifold 上。
- heads-only 能解决 readout 问题，但不能修 local latent manifold。
- 简单解冻 `local_latents_linear/latent repair` 在小数据和高 replay loss 下不稳定，反而破坏 shared readout。

### 4.3 为什么不能继续盲目加步数

当前 replay hard cases 的结构/FM loss 达到几千到上万量级。这说明 sampled states 离 clean states 太远，loss 被结构项主导。继续增加训练步数会让模型在非常坏的 off-manifold 条件上硬拟合，容易破坏原始 Complexa 生成先验。

## 5. Warning/Error 影响判断

- `CCD_MIRROR_PATH/PDB_MIRROR_PATH not set`：本阶段使用 tensorized dataset 和 checkpoint，不调用需要 mirror 的结构读取功能；不影响本阶段结论。
- optional CA/residue feature disabled：这是 target-only/no-leak 预期行为，避免真实 binder 几何或序列泄漏；不影响科学目标。
- `torch.load weights_only=False FutureWarning`：安全/未来兼容 warning，不影响当前数值结果；后续可改为显式可信 checkpoint 加载策略。
- PowerShell 写 sbatch 产生 CRLF：已通过远程 `perl -pi -e 's/\r$//'` 修复；属于执行层问题，不影响模型结论。
- `PYTHONPATH` unbound：已修成 `${PYTHONPATH:-}`；属于作业脚本问题。

## 6. 下一步方案

Stage13 的结果把问题缩小到一个核心：**native Complexa flow 可以保留，但 sampled local latent manifold 仍不稳定；shared sequence 不能只靠最终 readout 学出来。**

下一步不应继续扩大数据或盲目加训练步数，而应做 Stage14：

1. K=1 native warmup：用原始 Complexa native path 做单状态 full-flow supervised warmup，先让 sampled local_latents 在 rollout 条件下回到 AE sequence manifold。
2. 分离 loss scale：对 replay hard cases 降低巨大 FM loss 权重，先优化 `AE-decoded CE + state-shared KL + contact/clash`，避免结构 loss 淹没序列 manifold 学习。
3. latent repair 改为残差小步：不直接训练完整 `local_latents_linear`，而是新增 bounded residual adapter：`z_repaired = z_native + gate * delta_z(shared_seq, state_context)`，并限制 `||delta_z||`。
4. K=2 near-state curriculum：只在 K=1 rollout sequence 稳定后，进入 K=2 近构象多状态训练，再扩到 K=3。
5. 评估不看单个 identity：同时报告 shared sequence、AE decoded state sequence、一致性、contact/clash、state persistence。

## 7. 当前结论

Stage13 没有完成最终科学目标，但完成了关键纠偏：

- 代码已经恢复并验证了原始 Complexa 生成先验的 native state-wise 路径。
- 证明 shared sequence head 可在 teacher-forced/native-token 条件下学习。
- 证明真正阻塞点是 de novo rollout 后的 local latent manifold，而不是 AE decoder 或最终 readout。

因此下一阶段应集中修 latent manifold repair 与 K=1 到 K=2 到 K=3 curriculum，而不是继续扩大训练数据或做旧 B0/B1/B2 胜负比较。
