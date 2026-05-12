# Strategy01 Stage26/27：Sequence-only Replay 与 K=1 Curriculum 诊断报告

## 阶段目标

本阶段继续围绕核心科学目标：**只输入多状态 target 构象，模型自己生成一个共享 binder 序列，并为每个 target state 生成合理 target-binder complex geometry**。

前面 Stage25 说明 target-interface field 训练有小幅正信号，但共享序列仍弱。本阶段没有扩大数据，也没有引入外部 teacher，而是做两个最小可验证诊断：

1. **Stage26 sequence-only on-policy replay**：冻结 geometry/interface 模块，只训练共享序列相关头，检验 on-policy replay 能否单独修复 sampled-condition 下的共享序列。
2. **Stage27 K=1 curriculum**：把现有 K=2/K=3 数据拆成 K=1 state-level 样本，先验证单状态 target-only rollout 能否稳定，再看是否能迁移回 K>1 多状态。

## 代码改动

### 1. `stage12_de_novo_multistate_training.py`

文件：`scripts/strategy01/stage12_de_novo_multistate_training.py`

新增可配置 loss 权重：

- `--lambda-seq`
- `--lambda-struct`
- `--lambda-state-seq`
- `--lambda-seq-consensus`
- `--lambda-anchor-disagreement`

科学意义：此前这些权重在 `multistate_loss.py` 中存在默认值，但训练 CLI 不能显式调控。Stage26 需要把结构 loss 关到 `0`，只测序列 replay 是否有效，因此必须把权重暴露出来。

同时修复了一个真实 bug：Stage17 sampling probe 复用 `build_model_stack()` 时构造的兼容 `Namespace` 没有新加的 loss 字段，导致：

```text
AttributeError: 'Namespace' object has no attribute 'lambda_seq'
```

修复方式：`configure_stage12_loss()` 对新增字段使用 `getattr(args, key, default)`，保证训练脚本和旧 sampling/probe 脚本都兼容。

### 2. Stage26 sequence-only trainable mode

新增参数：

```text
--stage26-sequence-only-trainable
```

该模式只训练以下前缀：

- `cross_state_shared_seq_attention`
- `shared_seq_token_norm`
- `shared_seq_token_update`
- `shared_seq_token_head`
- `shared_seq_head`
- `state_seq_head`
- `shared_seq_consensus_gate`
- `state_seq_condition_projector`

明确不训练：

- `state_xt_bb_ca_projector`
- `state_xt_latent_projector`
- `state_condition_projector`
- `target_interface_site_head`
- `target_interface_guidance_scale`
- `ca_linear`
- `local_latents_linear`
- native transformer 主体

科学意义：隔离变量。如果共享序列变好，就说明问题主要在 sequence readout/on-policy replay；如果不变，就不能继续把失败归因于“训练步数太少”。

CPU contract 结果：

- `trainable_params = 4,782,337`
- `lambda_struct = 0.0`
- `lambda_seq = 1.0`
- `lambda_state_seq = 0.75`
- `lambda_seq_consensus = 0.30`
- `lambda_anchor_disagreement = 0.10`
- `lambda_ae_seq = 2.0`
- `lambda_seq_ae_consistency = 0.5`

### 3. `stage27_build_k1_curriculum_dataset.py`

新增文件：`scripts/strategy01/stage27_build_k1_curriculum_dataset.py`

功能：从现有 curated multistate tensor bundle 拆出 K=1 state-level curriculum 数据。

输入：

```text
data/strategy01/stage10_exactaug_training/stage10_exactaug_trainval.pt
```

输出：

```text
data/strategy01/stage27_k1_curriculum/stage27_k1_trainval.pt
```

数据规模：

- 总样本：`470`
- train：`386`
- val：`84`

科学意义：如果 K=1 target-only native flow 都无法在 free rollout 下维持合理 sequence/geometry，那么 K=2/K=3 多状态共享序列学习不会稳定。K=1 curriculum 是必要 gate，而不是最终任务。

## 执行结果

### Stage26：sequence-only replay

训练命令核心设置：

- init checkpoint：`ckpts/stage12_de_novo_multistate/runs/stage25a_trainable_interface_field/stage12_pilot_final.pt`
- batch：`4`
- train samples：`64`
- val samples：`16`
- pilot steps：`480`
- trainable params：`4.78M`
- peak GPU memory：`4.52GB`
- step time：`0.148s/step`

训练 probe：

- pilot drop：`0.041`
- teacher-forced final shared identity：`1.0`
- replay final AE-state identity：`0.825`

free rollout val12：

| 指标 | Stage25A guidance | Stage26 |
|---|---:|---:|
| proxy selected shared-head identity | Stage25C val4 `0.250` | val4 `0.261`, val12 `0.269` |
| proxy selected posthoc identity | val4 `0.239` | val4 `0.216`, val12 `0.230` |
| mean contact-F1 | `0.0467` | `0.0463` |
| worst contact-F1 | `0.0208` | `0.0192` |
| contact persistence | `0.3314` | `0.3328` |
| severe clash rate | `0.0` | `0.0` |

结论：sequence-only replay 有很小 shared-head 正信号，但没有改善最终 posthoc 序列，也没有改善 geometry。它不是主解，不能继续只加 sequence-only 步数。

### Stage27：K=1 curriculum

训练命令核心设置：

- dataset：`stage27_k1_trainval.pt`
- init checkpoint：Stage25A final
- batch：`4`
- train samples：`256`
- val samples：`64`
- pilot steps：`800`
- trainable params：`53.86M`
- peak GPU memory：`4.69GB`
- step time：`0.165s/step`

训练 probe：

- pilot drop：`0.204`
- teacher-forced final shared identity：`1.0`
- replay final AE-state identity：`1.0`

free rollout K=1 val12：

- proxy selected identity：`0.1818`
- oracle identity：`0.2538`
- mean contact-F1：`0.0242`
- mean direct RMSD：`41.95 Å`
- severe clash rate：`0.0`
- contact persistence：`1.0`（K=1 下该指标恒容易偏高，不能代表多状态成功）

free rollout multistate val12：

| 指标 | Stage26 | Stage27 K=1 warmup |
|---|---:|---:|
| proxy selected identity | `0.2303` | `0.2460` |
| proxy selected shared-head identity | `0.2688` | `0.2520` |
| mean contact-F1 | `0.0463` | `0.0184` |
| worst contact-F1 | `0.0192` | `0.0037` |
| contact persistence | `0.3328` | `0.1593` |
| severe clash rate | `0.0` | `0.0` |

结论：K=1 curriculum 没有解决目标问题，且迁移回 K>1 后明显破坏 interface geometry。不能把 Stage27 作为下一阶段主线 checkpoint。

## Warning / Error 处理

### 已修复错误

Stage26 sampling eval 首次失败：

```text
AttributeError: 'Namespace' object has no attribute 'lambda_seq'
```

根因：新增 CLI loss 字段后，Stage17 兼容 Namespace 未同步。

修复：`configure_stage12_loss()` 使用 `getattr(..., default)`。修复后 Stage26 sampling eval 和 Stage27 全流程通过。

### 已评估 warning

1. `CCD_MIRROR_PATH/PDB_MIRROR_PATH not set`

来源：atomworks 可选镜像路径。Stage26/27 没有调用需要本地 CCD/PDB mirror 的功能，不影响本阶段科学结论。

2. `use_ca_coors_nm_feature disabled...` / `use_residue_type_feature disabled...`

来源：no-leak 设计。de novo 模式下禁用真实 binder CA / residue type optional feature，返回零特征是预期行为，不影响结果，反而是科学契约的一部分。

3. `torch.load weights_only=False`

在 Stage27 builder 初版出现。数据是本仓自生成 trusted tensor bundle，但已显式传入 `weights_only=False` 消除未来 PyTorch 默认行为变化带来的歧义。

未发现影响结果的 Traceback/NaN/OOM。

## 对科学目标的判断

本阶段没有达到最终科学目标。

原因不是单个 bug，而是更深的机制问题：

1. teacher-forced / replay-blended 条件下 identity 可以很高，但 free rollout 仍低，说明训练条件仍不足以覆盖真正 de novo sampling 的分布。
2. sequence-only 只修 readout 不够，因为最终失败同时包含 sequence 和 interface placement。
3. K=1 curriculum 没有迁移到 K>1，说明只先修单状态 sampled latent manifold 也不够。
4. 当前 target-interface/hotspot 条件主要通过弱 guidance 或后验 selection 起作用，还没有作为强一等条件稳定控制 full bb_ca flow 的初始位置与 denoising 轨迹。

## 下一步建议

不要继续扩大 Stage26/27 训练步数。

下一阶段应改为 **Stage28：显式 target-hotspot/interface-conditioned de novo flow**：

1. 把 target hotspot / predicted target-interface field 从“弱 prior/guidance”升级为 sampling 初始化和每步 denoising 的硬条件。
2. 对 bb_ca 初始噪声使用 hotspot-centered translation prior，而不是普通 target-centered/random x0。
3. 在采样期间加入 bounded target-shell projection，并记录是否破坏 contact。
4. 默认使用 no-leak 的 `contact_shell` selection 作为候选选择策略，因为 Stage26 中它比当前 hard-gate selection 略好。
5. 训练目标重新平衡：保留 sequence/AE consistency，但把 interface contact/distance/clash 作为主约束；sequence 改善不能以 interface geometry 下降为代价。
6. Stage28 成功 gate：在 val12 上至少要优于 Stage25A/Stage26 的 mean contact-F1 `0.046` 和 persistence `0.33`，同时 shared sequence identity 不低于 Stage26。

当前最可靠 checkpoint 仍是 Stage25A/Stage26，而不是 Stage27 K=1 warmup。
