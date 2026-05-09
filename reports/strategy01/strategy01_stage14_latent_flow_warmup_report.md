# Strategy01 Stage14：native latent flow warmup、self-conditioning 修复与 replay 诊断报告

## 阶段目标

Stage14 继续围绕 Strategy01 的核心科学目标推进：**只输入多个 target 构象，模型从零生成 binder 空间位置、backbone、local latent，并输出一个能兼容多状态 target 的共享 binder 序列**。

前一阶段的关键现象是：teacher-forced 条件下 shared sequence head 可以学到监督序列，但 target-only de novo rollout 的 sampled local latent 仍明显偏离 AE 可解码序列流形。因此 Stage14 的重点不是扩大数据，而是定位并修复训练-采样路径中的实现漏洞。

## 代码改动

### 1. Stage14 bounded latent repair

相关文件：

- `src/proteinfoundation/nn/local_latents_transformer_multistate.py`
- `scripts/strategy01/stage11_flow_sequence_training.py`
- `scripts/strategy01/stage12_de_novo_multistate_training.py`
- `scripts/strategy01/stage12c_de_novo_smoke.py`
- `scripts/strategy01/stage13_single_state_de_novo_audit.py`

改动内容：

- 新增可选 `stage14_enable_bounded_latent_repair`。
- 使用模型自己的 `shared_seq_logits` 形成 soft sequence embedding。
- 通过 `latent_sequence_repair_projector/gate` 对 clean-space `local_latents_states` 做小幅 bounded residual。
- `attach_ae_sequence_logits()` 优先使用 `local_latents_states_clean_repaired`，让 AE-decoded sequence loss 能监督 repair 模块。

科学意义：

- 这个模块尝试把 sampled latent 拉回 AE sequence manifold，但不把真实 binder 序列作为输入，避免泄漏。

结果：

- CPU probe 通过。
- GPU 训练在 teacher/replay-clean 条件可收敛。
- 但 de novo audit 从 Stage13 best `shared=0.1591 / AE=0.0500` 变为 `shared=0.0591 / AE=0.0818`，没有成为主线。

结论：

- bounded final repair 不是当前主要瓶颈解法，至少当前实现会损害 target-only rollout。

### 2. Native latent warmup

相关文件：

- `scripts/strategy01/stage12_de_novo_multistate_training.py`

改动内容：

- 新增 `--stage14-native-latent-warmup`。
- 在 `stage13_heads_only` 的基础上额外解冻：
  - `local_latents_linear`
  - `transformer_layers.10`
  - `transformer_layers.11`
  - `transformer_layers.12`
  - `transformer_layers.13`

科学意义：

- Stage13/Stage14 结果显示问题不只是最终 readout，而是 sampled `local_latents` 本身不在 AE sequence manifold 上。
- 因此需要让原生 Complexa denoiser 的高层和 latent head 适配多状态/共享序列监督，而不是只训练外接 head。

结果：

- `stage14_native_latent_warmup_161`
  - train/val：`161 train / 36 val`
  - batch size：4
  - pilot steps：800
  - trainable params：约 53.85M / 345.75M
  - validation：`shared_identity=0.5795`，`AE_identity=0.4962`
  - de novo audit n16：`shared_identity=0.2136`，`AE_identity=0.2136`

结论：

- 这是 Stage14 中最明确的正向结果。相比 Stage13 best `shared=0.1591 / AE=0.0500`，native latent warmup 提升了真实 rollout 的 shared/AE sequence identity。
- 但该结果仍远低于科学目标，不能视为完成。

### 3. 直接解码 final integrated x 的审计修复

相关文件：

- `scripts/strategy01/stage12c_de_novo_smoke.py`

改动内容：

- 新增 `final_x_identity`。
- 直接用 AE decode 积分结束后的 `x_states["bb_ca"]` 与 `x_states["local_latents"]`，而不是只看最终额外一次模型 clean prediction。

科学意义：

- 原始 Complexa 生成路径最终解码的是 flow 积分后的样本 `x`。
- 如果只评估最后额外一次 clean prediction，可能错误判断生成结果。

结果：

- 对当前最佳 warmup checkpoint，`final_x_identity` 与原 clean-pred identity 一致：
  - `final_x_ae_shared_identity=0.2136`
  - `final_x_ae_state_identity=0.2136`

结论：

- 当前低 identity 不是审计口径误用造成的；真正问题仍是 sampled latent/trajectory 不够好。

### 4. Self-conditioning 修复

相关文件：

- `src/proteinfoundation/nn/local_latents_transformer_multistate.py`
- `scripts/strategy01/stage12_de_novo_multistate_training.py`
- `scripts/strategy01/stage12c_de_novo_smoke.py`

发现的问题：

- 原始 `configs/pipeline/model_sampling.yaml` 中 `args.self_cond=True`。
- 原始 Complexa 采样会在 step > 0 时把上一轮 clean prediction 作为 `x_sc` 输入。
- Stage12/13 自写的 multistate rollout 没有保留该逻辑。
- 更严重的是 `_native_state_flatten_input()` 每次都把 `flat["x_sc"]` 强制置零，导致 native state path 没有真正复用原始 Complexa sampling 语义。

改动内容：

- `_native_state_flatten_input()` 现在优先使用 `x_sc_states`，否则使用已有 `x_sc`，最后才回退零张量。
- `add_weighted_legacy_fields()` 现在会从 `x_sc_states` 生成 legacy `x_sc`。
- `build_replay_condition_batch()` 和 `rollout_final()` 中新增 `x_sc_states` 递推：
  - 每一步由当前 `nn_out` 计算 clean prediction `x_1`。
  - 下一步把该 `x_1` 作为 `x_sc_states`。

科学意义：

- 这是影响 target-only de novo flow 轨迹的真实实现漏洞。没有 self-conditioning，Strategy01 的 state-wise native path 并没有完整继承原始 Complexa 的生成动力学。

结果：

- native path 等价性 probe 在无 self-conditioning 输入时仍保持与原始 Complexa 完全一致：
  - `bb_ca max_abs=0.0`
  - `local_latents max_abs=0.0`
- 对最佳 warmup checkpoint，self-conditioning 后 n16 de novo audit：
  - 修复前：`shared=0.2136 / AE=0.2136`
  - 修复后：`shared=0.2273 / AE=0.2273`
- n32 仍较差：`shared=0.1364 / AE=0.1409`。

结论：

- self-conditioning 修复是正确且必要的，但单独不足以达到科学目标。

## 实验结果汇总

| 实验 | checkpoint/设置 | 关键结果 | 结论 |
| --- | --- | --- | --- |
| Stage14 bounded repair CPU probe | bounded repair on | AE-state `0.4167` | 接线能跑 |
| Stage14 bounded repair train | heads + bounded repair | teacher/replay-clean 可收敛 | 训练链路可用 |
| bounded repair de novo audit | n16 | shared `0.0591`, AE `0.0818` | 不作为主线 |
| native latent warmup | 800 steps | val shared `0.5795`, val AE `0.4962` | 最强正向训练信号 |
| native warmup de novo | n16 | shared `0.2136`, AE `0.2136` | Stage14 best before self-cond |
| warmup + hard replay | replay refine | de novo shared `0.0545`, AE `0.0591` | hard replay 破坏 rollout |
| longer warmup | +3000 steps | de novo shared `0.1682`, AE `0.1682` | 继续训练会过拟合 teacher-forced，不提升 de novo |
| latent stop sweep | stop 0.55/0.65/0.75/0.85 | best shared `0.2045` | 简单停 latent 不能解决 |
| final x direct decode | n16 | 与 clean-pred identity 一致 | 审计口径不是主因 |
| self-cond bugfix | n16 | shared `0.2273`, AE `0.2273` | 必要修复，小幅改善 |
| self-cond replay refine | conservative replay | shared `0.2227`, AE `0.2182` | 不超过 self-cond warmup |

当前最佳：

- checkpoint：`ckpts/stage12_de_novo_multistate/runs/stage14_native_latent_warmup_161/stage12_pilot_final.pt`
- sampling/audit 路径：启用修复后的 self-conditioning
- n16 de novo identity：`shared=0.2273 / AE=0.2273`

## 警告与报错处理

### 1. PowerShell / shell 脚本错误

- `--report-json` 被误传给 CPU probe：该脚本实际通过 `--run-name` 写报告。已改正。
- 状态解析脚本 here-doc 结尾写错，导致 `NameError: PY`。这只影响监控解析脚本，不影响训练输出。
- PowerShell 将 `$(sbatch ...)` 误解释为本地命令。后续改为远程 submit 脚本。
- CRLF 让远程 shell 读到 `/tmp/stage14_parse.py\r`。已用 `sed -i 's/\r$//'` 规避。

这些属于执行层错误，均已修正，不影响模型结果本身。

### 2. Optional feature warning

日志多次出现：

- `use_ca_coors_nm_feature disabled or not in batch`
- `use_residue_type_feature disabled or not in batch`

判断：

- 当前 `de_novo_multistate_mode=True` 明确禁止真实 binder CA / residue type optional feature 进入模型。
- 这些 warning 是 no-leak 契约下的预期行为，不影响科学结果。

### 3. `CCD_MIRROR_PATH/PDB_MIRROR_PATH` warning

判断：

- 当前 Stage14 使用已经 tensorized 的 Strategy01 数据和 checkpoint，不调用需要 PDB/CCD mirror 的功能。
- 该 warning 不影响本阶段结果。

### 4. `torch.load(weights_only=False)` FutureWarning

判断：

- 出现在等价性 probe 中，加载的是用户自己策略仓和 baseline 的受控 checkpoint。
- 当前不影响结果正确性，但后续可把 probe 脚本切换为 `weights_only=True` 或显式记录受控来源。

## 是否达到科学目标

尚未达到。

已经确认的正向点：

- Stage12/13 native path 能逐 state 复用原始 Complexa target-conditioned denoiser。
- 不传 `x_sc_states` 时仍与原始 Complexa 完全等价。
- self-conditioning 缺失已修复。
- native latent warmup 证明 sampled latent manifold 可以被训练改善。

仍未解决的问题：

- target-only de novo rollout 的 sequence identity 仍只有约 `0.23`。
- nsteps 增加会恶化，说明 latent trajectory 会随采样步数漂移。
- replay 训练如果直接作用于 off-manifold rollout，会损害已有 flow。
- 当前还没有进入真正 K>1 多状态 exact benchmark 的成功条件。

## 下一步建议

不要继续盲目增加训练步数，也不要再使用强 replay。下一步应做 Stage15：

1. 保留 self-conditioning bugfix。
2. 以 `stage14_native_latent_warmup_161` 为当前 best checkpoint。
3. 实现 self-conditioned **near-manifold trajectory distillation**：
   - 不再从严重 off-manifold hard replay 直接回归真实 `x1`。
   - 只在接近 clean 的 rollout 状态上训练 sequence/AE consistency。
   - flow FM loss 对 replay 只做很小权重或冻结 native flow。
4. 增加 early-step sequence supervision：
   - 明确覆盖 low/mid `t`，因为当前 step0/mid identity 低。
5. 再做 K=1 de novo gate：
   - n16 shared/AE identity 必须超过当前 best `0.2273`。
   - 如果 K=1 不过，不进入 K>1 多状态 benchmark。
6. K=1 通过后，再跑 K=2/K=3 true multistate exact benchmark，评估 shared sequence 是否同时兼容多个 target states。
