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
- 2026-04-26 起，`stage06_build_boltz_production_dataset.py` 的 `run_boltz_passes()` 已接入该缓存：预测前先查 sequence-hash cache，预测后再把新生成的 `*_0.csv / *_1.csv` 回填到 persistent cache，供后续 rerun 复用。

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

### 9.5 CPU seed mining 实测

- CPU seed mining job：`1908298`。
- 分区/节点：`new` / `cu06`，仅使用 CPU，不占 GPU。
- 脚本：`scripts/strategy01/stage07_seed_mine.sbatch`。
- 目标：先在 CPU 上筛出真实 multistate target + shared binder 的训练 seed，再把 GPU 留给 Boltz2 complex prediction。
- 实测 summary：`reports/strategy01/probes/stage07_seed_mining_summary.json`。

实测结果：

- `entries_seen = 2258`。
- `accepted_seeds = 256`。
- `elapsed_sec = 2008.95`，约 `33.5` 分钟。
- failure 主因：`no_pair = 774`、`download:HTTPError = 10`、`excluded_group = 20`、`duplicate_pair = 5`。

seed manifest 统计：

- 路径：`data/strategy01/stage07_seed_pool/T_prod_seed_manifest.json`。
- seed 数：`256`。
- unique family/split groups：`241`。
- `K` 分布：全部为 `3-state`。
- binder 长度：最短 `6`、均值 `45.70`、最长 `119`。
- target 长度：最短 `30`、均值 `92.11`、最长 `432`。
- target motion RMSD：最小 `1.023`、均值 `9.051`、最大 `68.962`。

### 9.6 正在运行的 GPU 数据生产

- Stage07 数据生产 job：`1908299`。
- 分区/节点：`new` / `gu02`。
- GPU 约束：仅占用 `1 x A100`。
- 脚本：`scripts/strategy01/stage07_data_prod.sbatch`。
- 输入 seed manifest：`data/strategy01/stage07_seed_pool/T_prod_seed_manifest.json`。
- 输出目录：`data/strategy01/stage07_production/`。
- scratch 目录：`/tmp/kfliao_strategy01_stage07_$SLURM_JOB_ID`。
- 目标上限：`512 train + 128 val`，允许 short dataset 输出。
- summary：`reports/strategy01/probes/stage07_data_production_summary.json`。

说明：当前运行的 `1908299` 使用的是“CPU 先挖 seed，GPU 只做 Boltz2”的拆分版流程。因为该 job 在本次 heartbeat 改造之前已经启动，所以实时日志仍然较少；若需要重跑，新的 builder 已加入 `seed_idx / accept / reject / split_done` 级别进度打印。

实时探针（通过 `scripts/strategy01/stage07_inspect_running_job.sh 1908299` 获取）：

- 运行约 `45` 分钟时，compute-node scratch 大小约 `249M`。
- `boltz_outputs` 下已出现 `5` 个 sample 目录：`2jma_A_B`、`2m1k_A_B`、`2mlx_A_B`、`7jyz_A_B`、`9r4w_B_A`。
- 当时活跃 Boltz 子进程为：`2mlx_A_B_scout_state01_boltz`。

结论：`1908299` 不是空转或卡死，而是在计算节点 `/tmp` 中持续产出中间预测结果，只是最终 tensor/manifest 会在数据生产阶段收尾时一次性写回 `/data`。

### 9.7 自动后处理依赖作业

- Stage07 postprocess job：`1908301`。
- 依赖：`afterok:1908299`，即数据生产成功后才启动。
- 脚本：`scripts/strategy01/stage07_postprocess.sbatch`。
- 内容：对 `data/strategy01/stage07_production/stage06_predictor_pilot_samples.pt` 提取 AE latent，生成 `stage06_predictor_pilot_samples_ae_latents.pt`，随后运行 `stage07_sequence_consensus_training.py` 的 `stage07_prod_pilot`，并绘制 `reports/strategy01/figures/stage07/prod_pilot/` loss 曲线。
- 脚本已升级：加入 `PYTHONUNBUFFERED=1`、`step=ae_latents/train/plot` 和开始/结束时间 heartbeat。

### 9.8 数据生产失败诊断、quota 修复与流程重构

- 第一版 GPU 数据生产 job：`1908276`。
- 第一版 afterok job：`1908277`。
- 初始失败原因：`OSError: [Errno 122] Disk quota exceeded`，发生在创建 `data/strategy01/stage07_production/boltz_outputs` 时。
- 连带影响：`1908277` 进入 `DependencyNeverSatisfied`，运行时间保持为 `0`；已取消。
- 诊断结果：`/data` 用户 quota 接近 hard limit；失败残留 `stage07_production` 约 `3.7GB`，其中 cache 约 `3.5GB`。
- 第一轮修复：新增 `stage06_build_boltz_production_dataset.py --work-dir` 参数，把 RCSB cache、seed states、Boltz outputs 放到计算节点 `/tmp`。
- 第二轮诊断：虽然 scratch 修复了 quota，但 `1908293` 仍然把“RCSB seed 挖掘 + GPU Boltz2”混在同一个 A100 job 里，导致 GPU 在长时间网络/CPU 阶段被占住且日志接近黑箱。
- 第二轮修复：新增 CPU-only `scripts/strategy01/stage07_mine_train_seeds.py` 和 `scripts/strategy01/stage07_seed_mine.sbatch`，把 seed 挖掘前置到 CPU；GPU `stage07_data_prod.sbatch` 改成必须读取现成 `T_prod_seed_manifest.json`。
- 当前采用的正式链路：`1908298 (CPU seed mining) -> 1908299 (GPU Boltz2 production) -> 1908301 (AE latent + training + plotting)`。


### 9.9 2026-04-26 当前中间产出快照

运行中作业 `1908299` 的 allocation 内部检查结果：

- scratch 大小：约 `2.1G`。
- 已创建 `54` 个 base-seed sample 目录。
- 已生成 `372` 个 `*_model_0.pdb` 与对应 `372` 个 `confidence_*.json`。
- 已生成 `926` 个 MSA CSV、`463` 个 `pair.a3m`、`926` 个 `out.tar.gz`。
- 当前活跃 Boltz 子进程示例：`2mkc_C_A_second_state00_boltz`。

按当前 scratch 中已经完成的 state-level 结果重建 Bronze/Silver 统计：

- `accepted_bases_bronze_ge2 = 9`。
- `accepted_bases_bronze_ge3 = 5`。
- `accepted_variants_est = 48`。
- `silver_bases_all_states = 0`。
- `partial_or_failed_bases = 34`。
- `state_level_bronze = 26`，`state_level_silver = 0`。

解释：

- 这里的 `accepted_variants_est` 是按当前已经满足条件的 base seed，套用 `variant_state_subsets()` 和 `hotspot_mode in {exact, pred}` 的规则估出来的可生成训练 variant 数。
- 由于 `1908299` 仍在运行，这不是最终数据集规模，而是“当前已经跑出来且按 Stage07 Bronze 规则可用”的中间快照。

### 9.10 云端 MSA 在 `1908299` 中的真实使用情况

结论：云端 MSA 在当前 `1908299` 的 Boltz2 生产中是**真实生效**的，不是只有代码开关或日志占位。

证据来自运行中 sample log，例如：

- `COMMAND ... --use_msa_server ...`
- `MSA server enabled: https://api.colabfold.com`
- `Calling MSA server for target ... with 2 sequences`
- `MSA server URL: https://api.colabfold.com`

并且当前 scratch 中已经实际生成：

- per-chain MSA CSV：`*_0.csv`、`*_1.csv`
- paired MSA：`pair.a3m`
- unpaired/环境数据库结果：`uniref.a3m`、`bfd.mgnify30.metaeuk30.smag30.a3m`
- 进一步加工后的 Boltz feature：`processed/msa/*.npz`

这说明生产流程是：

1. `stage06_build_boltz_production_dataset.py` 调用 `BoltzAdapter.predict()`。
2. `BoltzAdapter.build_command()` 在 CLI 中显式加入 `--use_msa_server`。
3. Boltz2 访问远程 ColabFold MSA server (`https://api.colabfold.com`) 获取 target/binder 两条链的 MSA。
4. 返回的 MSA 被写成 CSV/A3M，再转换成 `processed/msa/*.npz`。
5. 这些 `msa npz` 被后续结构预测真正读入，所以它们是 prediction feature，不是只写日志。

补充说明：

- 当前正在运行的 `1908299` 是在“MSA persistent cache 接入主生产链”之前启动的，所以这一次运行虽然**真实使用了远程 MSA server**，但还没有享受到我们后来补上的 sequence-hash 持久化复用。
- 也就是说：
  - `1908299`：真实用了云端 MSA，但主要是“每次 Boltz 调用就远程请求并在本次 scratch 内使用”。
  - 后续 rerun：会先查 `stage07_msa_cache`，命中时直接复用缓存 CSV，再把新的 MSA 结果回填缓存，速度会更好。

### 9.11 扩展验证集进度

CPU-only 扩展作业 `1908304` 已成功完成：

- `n_exact = 64`
- `n_exact_families = 55`
- `n_hybrid_seed = 0`
- `small/medium/large = 17/27/20`
- 状态：`passed`

说明当前高质量实验验证集已经从原来的 24 条扩到 64 条，可作为后续 B0/B1/B2 的更强 exact-only 验证基础。

### 9.12 当前可见错误状态

- `1908299`：目前没有 fatal error，`stderr` 为空；只是最终 tensor 尚未写回 `/data`。
- `1908304`：`stderr` 里有一条 `task/cgroup: unable to add task ... to memory cg '(null)'` 警告，但作业本身已成功结束并产出 summary，可视为非阻塞性集群告警。
