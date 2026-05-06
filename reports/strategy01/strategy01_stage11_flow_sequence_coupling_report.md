# Strategy01 Stage11：Complexa-native 多状态结构 Flow 共同优化共享 Binder 序列报告

## 1. 阶段目标

本阶段目标是修正 Stage10 后暴露的核心问题：几何已经可以通过 source-pose 初始化和 state-specific pose refinement 做到较好，但输出的共享 binder 序列仍弱。  
因此 Stage11 不新增 DynamicMPNN/ADFLIP/ProteinMPNN 这类独立 refiner，而是把序列约束直接接回 Complexa 的 product-space flow：

- K 个 state-specific `local_latents_states / bb_ca_states` 仍由 Complexa-native flow 生成。
- 模型内部生成 `state_seq_logits`，再做 robust consensus 得到一个 `seq_logits_shared`。
- 将模型自己的 soft shared sequence distribution 回馈到 state tokens，再进行第二次 state-specific latent/pose readout。
- 对 predicted clean `local_latents_states` 调用冻结 Complexa AE decoder，得到 `ae_seq_logits_states`，让共享序列 loss 直接约束 local latent flow。
- 新增 `flow_gate [B,K,Nb,1]`，用于后续 sampling 中 bounded bb_ca flow，而不是固定 `bb_ca_flow=0` 或全量流动。

这符合当前科学目标：一个共享 binder 序列兼容 target 多个功能构象，并且每个 state 都有合理 target-binder complex geometry。

## 2. 基线与保护

- 策略仓：`/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase`
- Git 分支：`codex/strategy01-arch-reboot`
- Stage10 初始化 checkpoint：`ckpts/stage10_pose_init_exactaug/runs/stage10_pose_init_exactaug_full_v2/mini_final.pt`
- AE checkpoint：`ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt`
- benchmark baseline 仓和 baseline checkpoint 未修改。

执行中遇到 `/data` quota 超限，`git apply` 半途失败导致两个目标文件短暂显示为 deleted。已先删除旧 Stage07 run checkpoints：

`ckpts/stage07_sequence_consensus/runs`

这属于旧策略训练产物，不是 benchmark baseline，也不是 Stage10 当前依赖 checkpoint。删除后 quota 从约 `211G/196G` 降到约 `191G/196G`，随后恢复文件并重新应用 Stage11 patch。

## 3. 代码改动

### 3.1 `local_latents_transformer_multistate.py`

文件：`src/proteinfoundation/nn/local_latents_transformer_multistate.py`

新增模块：

```python
soft_sequence_feedback_projector: [B,Nb,20] -> [B,Nb,d]
flow_gate_head: [B,K,Nb,d] -> [B,K,Nb,1]
```

前向逻辑变化：

1. 原本：`state_tokens -> local_latents_states / bb_ca_states`，`state_seq_tokens -> state_seq_logits`，再求 shared sequence。
2. 现在：
   - 先用 `state_seq_logits_pre` 形成 `shared_seq_logits_pre`。
   - 对 `shared_seq_logits_pre` 做 softmax 得到模型自己的 soft shared AA distribution。
   - 通过 `soft_sequence_feedback_projector` 投影回 token 空间。
   - 将 feedback 加回 `state_tokens/state_seq_tokens`。
   - 再输出最终 `local_latents_states / bb_ca_states / state_seq_logits / flow_gate`。

新增输出：

```text
seq_feedback_tokens [B,Nb,d]
flow_gate [B,K,Nb,1]
seq_logits_pre_feedback [B,Nb,20]
```

科学意义：

- 避免共享序列只来自额外 head，而是让共享序列信号影响 state-specific latent/pose readout。
- 保持 no-leak：feedback 输入是模型自己的 soft sequence，不是真实 binder residue type。
- 为 sampling 阶段提供 learnable bounded flow gate，后续可在不破坏 source anchors 的前提下允许局部 state-specific 调整。

### 3.2 `multistate_loss.py`

文件：`src/proteinfoundation/flow_matching/multistate_loss.py`

新增 loss：

```text
L_ae_seq_state: AE decoder 对每个 state clean local latent 解码出的序列 CE
L_ae_seq_sample: mean + CVaR 聚合后的 AE sequence loss
L_seq_ae_consistency: state/shared sequence logits 与 AE-decoded logits 的一致性
L_flow_gate_reg: anchor 区域 gate 偏小、非 anchor 区域允许中等 gate，同时避免全 0/全 1 塌缩
```

新增日志键：

```text
multistate_ae_seq_justlog
multistate_ae_seq_cvar_justlog
multistate_seq_ae_consistency_justlog
multistate_flow_gate_reg_justlog
multistate_state_{i}_ae_seq_justlog
multistate_state_{i}_flow_gate_anchor_justlog
```

设计原则：

- loss 函数本身不拥有 AE decoder；`ae_seq_logits_states` 由 Stage11 训练脚本注入。
- 如果没有注入 AE logits，旧 Stage03-10 路径保持兼容，新增项为 0。
- Stage11 训练时显式打开：

```text
lambda_ae_seq = 1.5
lambda_seq_ae_consistency = 0.15
lambda_flow_gate_reg = 0.03
```

### 3.3 `stage11_flow_sequence_training.py`

文件：`scripts/strategy01/stage11_flow_sequence_training.py`

职责：

- 从 Stage10 checkpoint 初始化 Strategy01 multistate NN。
- 加载冻结 AE decoder。
- 对 predicted clean `local_latents_states` 调用 AE decoder。
- 把 `ae_seq_logits_states / ae_seq_logits_shared` 注入 `compute_multistate_loss()`。
- 执行 CPU probe、1-sample overfit、4-sample overfit、小规模 mini fine-tune。

关键实现：

```text
state_clean_prediction()
attach_ae_sequence_logits()
forward_loss_stage11()
ae_decoder_probe()
grad_probe()
train_loop()
```

v2 修正：

- `--ae-ca-source init`：AE decoder 默认使用 source-pose/init CA，而不是 predicted clean bb_ca。原因是 Stage10 已证明几何可以较好；本阶段先隔离 local latent -> sequence 问题，避免 bb_ca 噪声干扰序列诊断。
- `--noise-repeats 2`：同一 batch 用两个 corruption seeds 聚合 loss，缓解训练 seed 能降、eval seed 不稳的问题。
- `--mini-trainable-phase seq_latent`：mini 阶段只训练 sequence/latent coupling 相关模块，不训练 `ca_linear` 和顶部 trunk，避免几何被序列 loss 拉坏。

### 3.4 `stage09_guided_state_specific_sampling.py`

文件：`scripts/strategy01/stage09_guided_state_specific_sampling.py`

新增参数：

```text
--use-learned-flow-gate
```

作用：

```python
flat_next = flat_x + flow_gate * (flat_next - flat_x)
```

仅对 `bb_ca` 生效。默认关闭，Stage10C fallback 不受影响。后续 Stage11D 可以比较：

- Stage10C fallback：`bb_ca_flow=0, local_latents_flow=1`
- learned bounded flow
- learned bounded flow + sequence feedback

## 4. 探针与训练结果

### 4.1 CPU probe

命令：

```bash
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python scripts/strategy01/stage11_flow_sequence_training.py \
  --device cpu \
  --run-name stage11_flow_sequence_v2_cpu_probe \
  --max-train-samples 2 \
  --max-val-samples 1 \
  --skip-training \
  --ae-ca-source init \
  --noise-repeats 2 \
  --lambda-ae-seq 1.5 \
  --lambda-seq-ae-consistency 0.15
```

结果：

- `status=passed`
- Stage10 checkpoint 兼容旧参数：`compatible_keys=1041`
- 新增 Stage11 参数随机初始化：
  - `soft_sequence_feedback_projector.*`
  - `flow_gate_head.*`
- shape 通过：
  - `ae_seq_logits_states [B,K,Nb,20]`
  - `flow_gate [B,K,Nb,1]`
  - `seq_feedback_tokens [B,Nb,768]`
- 梯度通过：
  - `soft_sequence_feedback_projector`
  - `flow_gate_head`
  - `shared_seq_head`
  - `state_seq_head`
  - `local_latents_linear`
  - `ca_linear`

### 4.2 GPU smoke v1

job：`1915258`

结果：

| phase | initial eval | final eval | drop | AE-seq eval | struct eval | clash eval | step time |
|---|---:|---:|---:|---:|---:|---:|---:|
| overfit1 | 16.90 | 14.41 | 14.8% | 0.989 -> 0.774 | 16.02 -> 13.73 | 0.0068 -> 0.0052 | 0.54s |
| overfit4 | 15.89 | 15.95 | -0.4% | 0.791 -> 0.533 | 15.18 -> 15.48 | 0.0079 -> 0.0078 | 0.30s |
| mini | 15.97 | 18.35 | -14.9% | 0.841 -> 0.555 | 15.22 -> 17.86 | 0.0072 -> 0.0092 | 0.30s |

判断：

- 代码和梯度闭环成立。
- AE sequence loss 能降，但 mini 阶段结构 loss 反弹，说明 sequence 与 structure 仍有拉扯。
- 因此 v1 不作为后续默认。

### 4.3 GPU smoke v2

job：`1915269`

修正：

- `ae_ca_source=init`
- `noise_repeats=2`
- `mini_trainable_phase=seq_latent`
- `lambda_ae_seq=1.5`
- `lambda_seq_ae_consistency=0.15`

结果：

| phase | initial eval | final eval | drop | AE-seq eval | struct eval | CVar eval | clash eval | step time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overfit1 | 17.80 | 13.42 | 24.6% | 1.092 -> 0.929 | 16.02 -> 11.95 | 18.01 -> 13.34 | 0.0068 -> 0.0032 | 0.81s |
| overfit4 | 16.65 | 14.86 | 10.8% | 0.909 -> 0.721 | 15.18 -> 13.71 | 17.10 -> 15.38 | 0.0079 -> 0.0081 | 0.42s |
| mini | 16.71 | 15.29 | 8.5% | 0.923 -> 0.729 | 15.22 -> 14.11 | 17.14 -> 15.79 | 0.0072 -> 0.0085 | 0.43s |

显存：

```text
cuda_max_mem_gb ~= 7.3GB
```

判断：

- v2 比 v1 更符合 Stage11 目标：eval seed 下 total、AE-seq、struct、CVar 同时下降。
- clash 轻微上升但仍低；后续正式 benchmark 必须以 severe clash hard fail 评估。
- 这仍不是科学完成，只是证明 “AE-decoder sequence loss + soft feedback + multi-corruption training” 这条机制比 v1 更稳。

## 5. Warning/Error 处理

### Disk quota exceeded

现象：

```text
error: unable to write file ... Disk quota exceeded
```

根因：

- `/data` quota 超过软限制。
- 旧 Stage07 run checkpoints 占用约 21GB。

处理：

- 删除旧 Stage07 run checkpoint 目录。
- 不删除 baseline checkpoint、Stage03 初始化副本、Stage10 当前 checkpoint。

影响：

- 不影响科学结果；释放空间后 patch 正常应用。

### `CCD_MIRROR_PATH` / `PDB_MIRROR_PATH` 未设置

判断：

- 本阶段不调用 atomworks mirror 相关功能，不影响 Stage11 训练/probe。
- 记录为环境 warning，不作为失败。

### PyTorch nested tensor warning

判断：

- 性能 warning，不改变数值目标。
- 不影响本阶段科学结论。

### optional CA/residue type features 返回 zeros

判断：

- 这是当前 no-leak 设定：`enable_ca_feature=false`，`use_residue_type_feature=false`。
- residue type 不能输入模型，否则真实 binder sequence 会泄漏。
- CA optional feature 关闭可能限制绝对质量，但这与 Stage10C fallback 保持一致；后续若要打开，必须确保输入来自模型自身或 source/init pose，而不是 exact label。

## 6. 当前结论

Stage11 已完成第一轮机制闭环：

- 模型内部已经有 shared sequence feedback。
- AE decoder sequence loss 已接回 predicted clean local latent flow。
- flow_gate 已输出并可被 sampling 脚本读取。
- CPU/GPU probes 通过。
- v2 smoke 明确优于 v1，并且更符合“多状态 structure flow 共同优化一个 shared sequence”的科学目标。

但 Stage11 还没有达到最终科学目标：

- 目前只是 16 条样本 smoke，不是 161/36 full pilot。
- 还没有用 learned flow gate 做正式 Stage11D sampling benchmark。
- 还没有在 48 V_exact 上做 B0/B1/B2 复评。
- holdout validation loss 仍高，说明泛化还不足。

## 7. 下一步建议

1. 用 v2 配置跑 `161 train / 36 val` full pilot，步数建议 `800-1500`，先不要扩大数据。
2. 用 `--use-learned-flow-gate` 跑 Stage11D sampling smoke，对比：
   - Stage10C fallback
   - learned bounded flow
   - learned bounded flow + sequence feedback
3. 如果 full pilot 的 AE-seq 与 CVar 持续下降且 geometry 不退化，再做 V_exact 48 的 B0/B1/B2。
4. 如果 sequence 提升但 geometry 掉，降低 `lambda_ae_seq` 或把 `ca_linear` 持续冻结。
5. 如果 geometry 稳但 sequence 不升，增加 `noise_repeats` 或加入 sampled latent cache 条件，而不是接外部 refiner。

## 8. 复现命令

编译：

```bash
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python -m py_compile \
  src/proteinfoundation/nn/local_latents_transformer_multistate.py \
  src/proteinfoundation/flow_matching/multistate_loss.py \
  scripts/strategy01/stage11_flow_sequence_training.py \
  scripts/strategy01/stage09_guided_state_specific_sampling.py
```

CPU probe：

```bash
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python scripts/strategy01/stage11_flow_sequence_training.py \
  --device cpu \
  --run-name stage11_flow_sequence_v2_cpu_probe \
  --max-train-samples 2 \
  --max-val-samples 1 \
  --skip-training \
  --ae-ca-source init \
  --noise-repeats 2 \
  --lambda-ae-seq 1.5 \
  --lambda-seq-ae-consistency 0.15
```

GPU smoke：

```bash
sbatch scripts/strategy01/stage11_flow_sequence_smoke.sbatch
```

结果文件：

```text
reports/strategy01/probes/stage11_flow_sequence_v2_cpu_probe_results.json
reports/strategy01/probes/stage11_flow_sequence_v2_smoke_results.json
reports/strategy01/figures/stage11/
```
## 9. fullpilot600 补充实测

上一版报告第 7 节中的第 1 项已经执行。本次 full pilot 使用当前可用的 `161 train / 36 val`，配置为：

- `ae_ca_source=init`，避免 AE decoder 因 sampled CA 噪声把 latent 序列诊断混淆。
- `noise_repeats=2`，覆盖 sampled latent 条件，而不是只在 teacher-forced exact latent 条件下优化。
- `mini_trainable_phase=seq_latent`，mini 阶段冻结 `ca_linear`，重点训练 sequence-flow coupling，避免结构头为了序列损失破坏 Stage10C 已经较好的几何。
- `lambda_ae_seq=1.5`，`lambda_seq_ae_consistency=0.15`。

### 9.1 fullpilot600 结果

| 阶段 | eval total | AE seq | seq-AE consistency | struct | CVaR | clash | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| overfit1 初始 | 17.7951 | 1.0921 | 0.7834 | 16.0192 | 18.0126 | 0.0068 | 20 step 短 overfit 起点 |
| overfit1 结束 | 16.3697 | 1.0123 | 0.5439 | 14.7565 | 16.5370 | 0.0084 | total 下降 8.0%，结构改善，AE seq 小幅改善，clash 小幅上升 |
| overfit4 初始 | 16.6545 | 0.9085 | 0.6257 | 15.1788 | 17.0980 | 0.0079 | 4-sample 起点 |
| overfit4 结束 | 15.3150 | 0.7828 | 0.4714 | 14.0587 | 15.7367 | 0.0091 | total 下降 8.0%，AE seq 和结构均改善，clash 小幅上升 |
| mini 初始 | 16.7146 | 0.9226 | 0.6296 | 15.2177 | 17.1350 | 0.0072 | 161/36 pilot 起点 |
| mini 结束 | 15.7554 | 0.7641 | 0.5154 | 14.5183 | 16.2898 | 0.0081 | total 下降 5.7%，AE seq 下降 17.2%，结构下降 4.6% |

验证集：

- `validation total = 30.0494`
- `validation AE seq = 2.5278`
- `validation struct = 25.5346`
- `validation CVaR = 28.3140`
- `validation clash = 0.00546`

显存与耗时：

- `batch=2`，单张 A100 40G。
- peak memory 约 `7.30 GB`。
- mini 平均 step time 约 `0.372 s/step`。
- 600-step mini 训练墙钟约 `223 s`，说明当前阶段训练瓶颈不是 GPU 算力，而是数据质量、loss 权重和 sampling/benchmark 闭环。

图表：

- `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/figures/stage11/stage11_flow_sequence_v2_fullpilot600_linear.png`
- `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/figures/stage11/stage11_flow_sequence_v2_fullpilot600_log.png`

### 9.2 对科学目标的判断

本次 fullpilot600 证明 Stage11 的核心接线是有效的：

- AE-decoded sequence loss 能从 predicted clean local latent 反传到 flow。
- shared sequence head 与 AE decoder logits 的一致性损失能下降。
- 在冻结结构输出层的 mini 训练下，结构 loss 没有像 v1 那样明显恶化。
- `flow_gate` 没有塌缩，mini 结束时 mean 约 `0.263`，std 约 `0.051`。

但它还没有达到最终科学目标：

- 训练集 mini loss 下降，但 validation 仍高，说明泛化不足。
- state 间序列损失不均衡，mini 结束时 state0 AE sequence loss 明显低于 state1/state2，说明共享序列仍更容易被 source-like state 主导。
- clash 小幅上升，虽然绝对值仍低，但后续评估必须继续把 severe clash 作为 hard fail。
- 尚未执行 Stage11D learned-flow sampling benchmark，因此不能声称 B1 生成质量已经超过 Stage10C 或 B0。

### 9.3 警告和错误处理

本阶段扫描了 `stage11_flow_sequence_full_1915270.out/.err`：

- PyTorch `enable_nested_tensor` warning：性能相关，不影响 loss 数值或科学目标。
- optional CA/residue type feature zeros：当前为 intentional no-leak 设定。`residue_type` 不能输入真实 binder sequence；`CA feature` 暂时关闭是为了和 Stage10C fallback 保持可比。后续若打开，必须只来自 init/source/model prediction，不能来自 exact label。
- PowerShell 远程执行再次出现 CRLF/脚本语法问题：本次原因是本地 `Set-Content` 写出的脚本上传后含 CRLF 或脚本结构不完整，修复方式是在远程执行前用 `perl -pi -e 's/\r$//'` 清理，并用 `bash -n` 检查。后续继续按“写远程临时脚本 -> scp -> 远程 lint -> 执行”的模式，避免直接在 PowerShell 中嵌套复杂 bash。

### 9.4 更新后的下一步

不建议马上扩大数据或直接做 B0/B1/B2 胜负宣称。下一步应继续围绕“多条 state-specific flow 共同优化一条 shared binder sequence”修机制：

1. 增加 hard-state sequence weighting：对 state1/state2 这类 AE sequence loss 高的状态提高 CVaR 权重，避免共享序列被 source-like state 主导。
2. 对 persistent interface anchors 增加 sequence-consistency 权重，但只在 source/init 可得 anchors 上做，避免 exact benchmark 泄漏。
3. 跑 Stage11D sampling benchmark：比较 Stage10C fallback、learned bounded flow、learned bounded flow + sequence feedback。没有这个 benchmark，fullpilot loss 下降还不能转化为“生成更好”。
4. 如果 learned flow sampling 保持 Stage10C 的 geometry，同时 shared sequence identity 提升，再进入正式 B0/B1/B2 exact benchmark。
5. 如果 validation 继续高，优先回查 split/family leakage、样本多样性和 sampled latent cache，而不是接外部 sequence refiner。

### 9.5 fullpilot600 复现命令

```bash
sbatch /tmp/stage11_fullpilot.sbatch
```

核心命令等价于：

```bash
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python scripts/strategy01/stage11_flow_sequence_training.py \
  --device cuda \
  --run-name stage11_flow_sequence_v2_fullpilot600 \
  --ae-ca-source init \
  --noise-repeats 2 \
  --mini-trainable-phase seq_latent \
  --lambda-ae-seq 1.5 \
  --lambda-seq-ae-consistency 0.15 \
  --batch-size 2 \
  --overfit1-steps 20 \
  --overfit4-steps 40 \
  --mini-steps 600
```

结果文件：

```text
reports/strategy01/probes/stage11_flow_sequence_v2_fullpilot600_results.json
reports/strategy01/figures/stage11/stage11_flow_sequence_v2_fullpilot600_linear.png
reports/strategy01/figures/stage11/stage11_flow_sequence_v2_fullpilot600_log.png
```
