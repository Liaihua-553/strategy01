# Strategy01 第二阶段报告：binder 输入投影与 checkpoint 对齐

## 1. 阶段目标

本阶段只在远程策略仓 `Strategy01_complexa_multistate_benchmarkbase` 内实施，不改动 benchmark 基线仓。  
目标是把多状态架构中 **binder 侧输入投影** 精确恢复到 benchmark 基线 checkpoint 保存的 `cfg_exp.nn`，并重新完成一次全模型 checkpoint 审计，确认：

- binder 侧输入维度恢复正确；
- 所有仍然保留且应当复用的旧层参数都能与基线 checkpoint 形状对齐；
- 仅新增的多状态模块保持随机初始化；
- 探针 `P0-P3` 全部通过。

## 2. baseline 对齐依据

本阶段唯一对齐真值来自：

- 基线 checkpoint：
  `/data/kfliao/general_model/strategy_repos/proteina-complexa-benchmark-baseline/ckpts/complexa.ckpt`
- 对齐字段：
  `ckpt["hyper_parameters"]["cfg_exp"]["nn"]`

从 checkpoint 中读取到的 binder 侧特征清单为：

### `feats_seq`

```python
[
  "xt_bb_ca",
  "xt_local_latents",
  "x_sc_bb_ca",
  "x_sc_local_latents",
  "optional_ca_coors_nm_seq_feat",
  "optional_res_type_seq_feat",
  "hotspot_mask_seq",
  "res_seq_pdb_idx",
]
```

### `feats_pair_repr`

```python
[
  "rel_seq_sep",
  "xt_bb_ca_pair_dists",
  "x_sc_bb_ca_pair_dists",
  "optional_ca_pair_dist",
  "chain_idx_pair",
  "hotspot_mask_pair",
]
```

## 3. 本阶段改动清单

### 3.1 恢复 multistate 配置中的 binder 特征

修改文件：

- `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/configs/nn/local_latents_score_nn_160M_multistate.yaml`

本次恢复了 4 个先前缺失、但 checkpoint 明确存在的 binder 特征：

- 序列特征新增：
  - `hotspot_mask_seq`
  - `res_seq_pdb_idx`
- pair 特征新增：
  - `chain_idx_pair`
  - `hotspot_mask_pair`

### 3.2 新增第二阶段专用 probe 配置

新增文件：

- `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/configs/training_local_latents_multistate_ckpt_align_probe.yaml`

用途：

- 不覆盖第一阶段 probe 配置；
- 单独输出第二阶段 JSON 结果；
- 保持阶段结果隔离与可追溯。

### 3.3 升级 architecture probe 脚本

修改文件：

- `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/scripts/strategy01/probe_multistate_arch.py`

主要升级点：

- 支持 `--config` 指定 probe 配置；
- 直接从 checkpoint 读取 `cfg_exp.nn`，不再依赖 baseline repo 中可能漂移的 yaml；
- probe 中显式补齐 baseline-native binder 字段：
  - `hotspot_mask`
  - `residue_pdb_idx`
  - `chains`
- 增加全模型 checkpoint 审计分类：
  - `baseline-reusable`
  - `baseline-retired`
  - `multistate-new`
- 增加断言，保证本阶段如果未真正对齐成功会直接报错退出。

## 4. binder 输入前后对比

### 4.1 输入特征前后对比

| 项目 | 第一阶段 | 第二阶段 |
|---|---:|---:|
| `feats_seq` 数量 | 6 | 8 |
| `feats_pair_repr` 数量 | 4 | 6 |
| `binder_seq_input_dim` | 45 | 302 |
| `binder_pair_input_dim` | 217 | 219 |

### 4.2 恢复原因

- `hotspot_mask_seq`：恢复 binder 侧热点显式输入，避免 checkpoint 中对应投影维度缺失。
- `res_seq_pdb_idx`：恢复基线 256 维 residue index embedding，是 `302 - 45` 的主要来源。
- `chain_idx_pair`：恢复 pair 级链关系输入。
- `hotspot_mask_pair`：恢复 pair 级热点提示输入。

## 5. checkpoint 审计结果

### 5.1 baseline 控制组

用基线模型 `LocalLatentsTransformer` 与基线 checkpoint 直接核对：

- `shape_compatible_key_count = 459`
- `baseline_reusable_mismatch_count = 0`
- `baseline_retired_key_count = 0`
- `unexpected_ckpt_only_count = 0`
- `unexpected_model_only_count = 0`

说明 benchmark baseline 本身与 checkpoint 完全一致，可作为本阶段审计基准。

### 5.2 Strategy01 multistate 审计

- `shape_compatible_key_count = 457`
- `baseline_reusable_mismatch_count = 0`
- `multistate_new_mismatch_count = 0`
- `unexpected_ckpt_only_count = 0`
- `unexpected_model_only_count = 0`

这说明：

- 所有 **仍在 Strategy01 中保留且应当复用的旧层参数** 都已经 shape-compatible；
- 新增多状态模块没有错误地与旧 checkpoint 键发生重名或错配；
- binder 输入投影恢复后，没有再出现新的旧层 shape mismatch。

### 5.3 仍未直接载入的参数分类

#### `baseline-retired`：2 个

```text
concat_factory.linear_out.weight
concat_pair_factory.linear_out.weight
```

这 2 个键来自 baseline 的 target concat 路径。  
当前 Strategy01 主线已经明确改为：

- `EnsembleTargetEncoder`
- `target2binder_cross_attention`

因此不再保留 baseline 的 target concat 结构，这 2 个参数被归类为 **架构替换后的 retired key**，而不是 mismatch。

#### `multistate-new`：562 个

这些键全部来自新增模块，例如：

- `ensemble_target_encoder.*`
- `target2binder_cross_attention_layer.*`
- `shared_seq_head.*`
- `state_condition_projector.*`
- `state_token_norm.*`

它们符合本阶段预期，应保持随机初始化。

## 6. 探针结果

第二阶段 JSON 结果文件：

- `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/checkpoint_alignment_probe_results_2026-04-13.json`

### P0：静态维度

关键结果：

- `single_state_target_feature_dim = 505`
- `token_dim = 768`
- `pair_repr_dim = 256`
- `binder_seq_input_dim = 302`
- `binder_pair_input_dim = 219`

结论：

- binder 输入投影已经恢复到 checkpoint 期望值。

### P1：前向探针

synthetic 2-state 输出：

- `seq_logits_shared = [1, 32, 20]`
- `bb_ca_states = [1, 2, 32, 3]`
- `local_latents_states = [1, 2, 32, 8]`

real TNFalpha 2-state 输出：

- `seq_logits_shared = [1, 32, 20]`
- `bb_ca_states = [1, 2, 32, 3]`
- `local_latents_states = [1, 2, 32, 8]`

结论：

- 输出契约在 binder 对齐后保持稳定，没有被破坏。

### P2：checkpoint 加载

关键结果：

- `shape_compatible_key_count = 457`
- `baseline_reusable_mismatch_count = 0`
- `unexpected_ckpt_only_count = 0`
- `unexpected_model_only_count = 0`

结论：

- 第二阶段对齐目标已达成。

### P3：单步 backward

关键梯度范数：

- `ensemble_target_encoder.target_feature_projection.1.weight = 1.5045`
- `ensemble_target_encoder.cross_state_fusion.q_proj.weight = 0.0165`
- `shared_seq_head.1.weight = 7.3428`
- `ca_linear.1.weight = 2.8267`
- `local_latents_linear.1.weight = 1.3674`

结论：

- 新增多状态模块与共享输出头在 binder 输入恢复后仍能正常回传梯度。

## 7. 仍未载入参数的原因说明

本阶段结束后，未直接从 baseline checkpoint 载入的参数分两类：

1. `baseline-retired`
   - 原因：旧 target concat 路径已被新架构替换，不再保留。
2. `multistate-new`
   - 原因：这些模块是 Strategy01 第二阶段之前 benchmark baseline 中不存在的新模块，理应随机初始化。

因此当前状态满足本阶段定义的“成功通过”：

- 旧层中 **应当复用的部分** 已全部对齐；
- 未载入部分都能被清晰解释为“架构替换”或“新增模块”。

## 8. PowerShell 远程执行经验与规避方式

本阶段确认了一条需要长期遵守的规则：

- 不要在 Windows PowerShell 中直接嵌套复杂的 `ssh "python <<'PY' ..."` 多层 heredoc 命令；
- PowerShell 很容易吞掉引号、把 Linux 路径当成本地命令、或把 `|` 误解析为本地管道；
- 更稳的做法是：
  - 本地生成临时脚本；
  - 用 `scp` 传到远程；
  - 再通过 `ssh` 单行调用远程脚本。

本阶段所有真正执行的远程 probe 都改为遵守这个约束。

## 9. 结论

第二阶段已经完成并通过：

- binder 侧输入投影从第一阶段的漂移状态恢复到 checkpoint 真值；
- 所有应当继承的旧层实现了 zero mismatch；
- 多状态前向与反向探针继续成立；
- 第二阶段结果可单独追踪、单独复现。

这为下一阶段继续做多状态微调/加载策略打下了更稳的 checkpoint 兼容基础。
