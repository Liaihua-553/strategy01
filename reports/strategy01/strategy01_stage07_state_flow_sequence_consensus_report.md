# Strategy01 Stage07：多状态序列共识、MSA 优化与几百条训练数据闭环报告

## 1. 阶段目标

本阶段目标是修正 Stage06 暴露出的关键科学风险：多个 state 的 `bb_ca_states / local_latents_states` 不能按权重平均后作为科学意义上的 binder pose。Stage07 保留 legacy 平均输出只用于兼容原 Complexa sampling 接口，主线改为：

- 每个 target state 保留自己的 state-specific flow trajectory。
- 每个 state trajectory 产生自己的 `state_seq_logits`。
- 最终 `seq_logits_shared` 由 base shared logits 和多状态 robust consensus 共同决定。
- 输出一个共享 binder sequence，同时输出 K 个 state-specific binder poses。

## 2. 本阶段代码改动记录

### 2.1 `LocalLatentsTransformerMultistate`

新增模块：

- `state_seq_head`：每个 state branch 产生 `state_seq_logits [B,K,Nb,20]`。
- `shared_seq_consensus_gate`：按 residue 控制 state consensus 对 shared sequence 的影响强度。

前后变化：

- Stage06：`seq_logits_shared = shared_seq_head(seqs)`。
- Stage07：`seq_logits_shared = base_shared_logits + gate * robust_consensus(state_seq_logits)`。

科学意义：

- 共享序列不再只来自一个平均化或全局化 token 路径。
- 每个 state-specific trajectory 都能对同一个 binder sequence 提出氨基酸偏好。
- 使用 quality/uncertainty-biased state weights，让困难 state 不容易被容易 state 平均掉。

### 2.2 `multistate_loss.py`

新增 loss 项：

- `multistate_state_seq_justlog`
- `multistate_seq_consensus_justlog`
- `multistate_anchor_disagreement_justlog`
- `multistate_state_k_seq_justlog`

新增训练目标：

```text
L_total += lambda_state_seq * L_state_seq
        + lambda_seq_consensus * L_seq_consensus
        + lambda_anchor_disagreement * L_anchor_disagreement
```

默认系数：

- `lambda_state_seq = 0.15`
- `lambda_seq_consensus = 0.05`
- `lambda_anchor_disagreement = 0.03`

科学意义：

- `L_state_seq` 让每个 state branch 都学习同一个共享 binder sequence。
- `L_seq_consensus` 让 shared sequence 和 state-specific sequence preference 对齐。
- `L_anchor_disagreement` 重点压低 persistent interface anchors 上的跨状态氨基酸偏好冲突。

### 2.3 `stage04_real_complex_loss_debug.py`

新增 trainable prefixes：

- `state_seq_head`
- `shared_seq_consensus_gate`

新增梯度检查项：

- `state_seq_head.1.weight`
- `shared_seq_consensus_gate.1.weight`

意义：

- 确认新增序列共识路径真实参与训练，而不是只在 forward 中产生无用张量。

### 2.4 Boltz2 MSA 支持

`boltz_adapter.py` 新增：

- `msa` 支持 dict：可分别给 chain A/B 传入不同 MSA 路径。
- `sequence_hash()`。
- `cache_msa_csv()`。
- `cached_msa_csv()`。

意义：

- Stage07 可以检测 `--use_msa_server` 产物，并把 per-chain MSA CSV 缓存下来。
- 后续相同 target/binder sequence 不必重复远程 MSA。
- 如果远程 MSA 慢，可以切换到 CPU 本地 MSA 生成，再把结果路径传给 GPU Boltz2。

### 2.5 新增 Stage07 脚本

- `scripts/strategy01/stage07_sequence_consensus_training.py`
  - 运行 loss/probe。
  - 实测 batch size `4 -> 2 -> 1` fallback。
  - 运行 overfit1 / overfit4 / mini 训练。
- `scripts/strategy01/stage07_msa_diagnostics.py`
  - 扫描 Boltz2 log。
  - 检查 MSA server 是否生效。
  - 建立 MSA CSV sequence-hash cache 索引。
- `scripts/strategy01/stage07_plot_loss_curves.py`
  - 只画 train/eval total loss 和各 state total loss。
  - 输出 raw + EMA，linear/log 两套图。

## 3. Tensor 维度变化

| 字段 | Stage06 | Stage07 |
|---|---:|---:|
| `seq_logits_shared` | `[B,Nb,20]` | `[B,Nb,20]`，但由 robust consensus 修饰 |
| `state_seq_logits` | 无 | `[B,K,Nb,20]` |
| `bb_ca_states` | `[B,K,Nb,3]` | 不变，主结构输出 |
| `local_latents_states` | `[B,K,Nb,Z]` | 不变，主 latent 输出 |
| legacy `bb_ca` | `[B,Nb,3]` 加权平均 | 保留，仅兼容/debug |
| legacy `local_latents` | `[B,Nb,Z]` 加权平均 | 保留，仅兼容/debug |

## 4. 探针与训练结果

待本阶段脚本运行后自动补充：

- sequence consensus probe。
- batch size 4/2/1 fallback 实测。
- 训练/验证 total loss 曲线。
- 各 state total loss 曲线。
- MSA server 生效性检查。

## 5. MSA 结论

待 `stage07_msa_diagnostics.py` 和必要的 Boltz2 对照作业补充。

当前设计判断：

- `--use_msa_server` 是远程 ColabFold MSA server，不是在 kfliao 本地做完整 MSA 搜索。
- 如果远程 MSA server 生效且速度可接受，Stage07 pilot 继续用远程 server + cache。
- 如果慢或不稳定，转为 CPU 本地 `colabfold_search/MMseqs2` 生成 MSA，GPU 只做 Boltz2 prediction。

## 6. 数据计划执行状态

目标：

- accepted training samples 尽可能多。
- 最低目标：`>=256 train + >=64 val`。
- 若 Boltz2 生产顺利，扩展到 `500-800 accepted samples`。

待运行数据生产后补充实际数量、family 覆盖、Bronze/Silver 通过率和 leakage audit。

## 7. 报错与修复日志

本节随执行实时补充。

| 时间 | 阶段 | 错误现象 | 根因 | 修复 | 验证 |
|---|---|---|---|---|---|
| 2026-04-25 | 远程连接 | 不带严格 ssh 选项时部分命令超时 | PowerShell/ssh 和远程路径访问偶发卡顿 | 改用 `ssh -T -o BatchMode=yes -o ConnectTimeout=10`，复杂操作用脚本 | 远程 git/status/file scp 恢复 |

## 8. 复现命令

```bash
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python -m compileall src/proteinfoundation/nn/local_latents_transformer_multistate.py src/proteinfoundation/flow_matching/multistate_loss.py scripts/strategy01/stage07_sequence_consensus_training.py scripts/strategy01/stage07_msa_diagnostics.py scripts/strategy01/stage07_plot_loss_curves.py
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python scripts/strategy01/stage07_msa_diagnostics.py
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python scripts/strategy01/stage07_sequence_consensus_training.py --device cpu --skip-training --run-name stage07_cpu_probe
```

GPU 训练通过 SLURM 在 `new` 分区执行，脚本路径待生成后补充。

## 9. Stage07 已完成实测结果（自动追加）

### 9.1 CPU/GPU 探针

- CPU probe：`reports/strategy01/probes/stage07_cpu_probe_pilot8_results.json`，通过。
- GPU probe：`reports/strategy01/probes/stage07_gpu_probe_pilot8_results.json`，通过。
- batch size 选择：`4`。
- batch 尝试记录：

```json
[
  {
    "batch_size": 4,
    "ok": true,
    "error": null,
    "cuda_max_mem_gb": 2.548643112182617
  }
]
```

### 9.2 训练探针摘要

- `overfit1`：初始 `25.502370834350586`，最终 `22.441280364990234`，显存 `6.327080249786377` GB。
- `overfit4`：初始 `27.838375091552734`，最终 `27.796621322631836`，显存 `6.327080249786377` GB。
- `mini`：初始 `27.825313568115234`，最终 `27.752273559570312`，显存 `6.327080249786377` GB。

图文件目录：`reports/strategy01/figures/stage07/gpu_probe_pilot8/`。

### 9.3 MSA 诊断摘要

- 扫描 Boltz log 数：`18`。
- 检测到远程 MSA server 生效的 log 数：`9`。
- 索引进 sequence-hash cache 的 MSA CSV 数：`18`。
- 缓存目录：`/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/data/strategy01/stage07_msa_cache`。

结论：已有日志中确实能检测到 `api.colabfold.com`，说明 `--use_msa_server` 在部分 Stage06/07 Boltz2 run 中真实生效；但并非所有 run 都启用或成功，因此 Stage07 后续生产必须继续记录每个 run 的 MSA 行为。若远程 MSA 变慢，下一步切换到 CPU 本地 `colabfold_search/MMseqs2` 生成 MSA，GPU 只跑 Boltz2 prediction。

### 9.4 错误修复补充

| 时间 | 阶段 | 错误现象 | 根因 | 修复 | 验证 |
|---|---|---|---|---|---|
| 2026-04-25 | GPU overfit4 | CUDA BCE assert，随后 NaN | 结构距离/概率中出现非有限值，BCE CUDA kernel 直接断言 | 对 target-binder 距离和概率使用 `nan_to_num` 与 clamp | BCE assert 消失 |
| 2026-04-25 | GPU overfit4 | batch=4 一步后 NaN | `state_condition_projector` 同时影响结构 pose 和序列共识，序列训练会扰动结构头 | 新增 `state_seq_condition_projector`，把序列 state bias 与结构 state bias 解耦 | job `1908275` 通过，batch=4 显存约 6.33GB |
| 2026-04-25 | 训练稳定性 | batch=4 误判可用 | 原 batch 选择只看 forward/backward，不检查梯度 finite | batch fallback 检查 trainable gradients 是否全部 finite | GPU probe 通过 |

### 9.5 正在运行的长作业

- Stage07 数据生产 job：`1908276`。
- 分区/节点：`new` / `gu02`。
- 目标上限：`512 train + 128 val`，允许 short dataset 输出。
- 输出目录：`data/strategy01/stage07_production/`。
- summary：`reports/strategy01/probes/stage07_data_production_summary.json`。

该作业是长作业，报告将在作业完成后继续追加实际 accepted samples、family 覆盖、Bronze/Silver 通过率和后续 AE latent/训练结果。
