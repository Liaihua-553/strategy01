# Strategy01 Stage04 报告：真实多状态复合物数据集、接口感知 Loss 与 1 卡执行诊断

## 1. 阶段目标与结论

本阶段目标是把 Strategy01 从 Stage03 的工程伪标注多状态 loss，推进到真实 target-binder 复合物结构可监督的 Stage04：每条样本包含多个 target states、一个 shared binder sequence、每个 state 对应的 target-binder complex 标签，并在训练 loss 中加入 interface contact、distance、clash、anchor persistence、self-geometry consistency 和 quality proxy。

本阶段已经完成代码闭环和 CPU 级真实复合物 debug 闭环：

- 已构造 `16` 条 experimental-complex-derived debug 样本，其中 `12` 条 train、`4` 条 validation。
- 每条样本 `K=2` 个真实结构 state，来自 benchmark baseline 仓内已存在的 `1tnf_cropped.pdb` 与 `1tnf_repacked.pdb` 多链复合物。
- 总计 `32` 个 state-level complexes，所有 state 的 `contact_count >= 8`，平均 `16.25`，最大 `26`，没有 severe clash。
- 已扩展 `LocalLatentsTransformerMultistate`，新增 `interface_quality_head`，输出 `interface_quality_logits [B,K,5]`。
- 已扩展 `compute_multistate_loss`，加入 contact/distance/clash/anchor/self-geometry/quality proxy loss。
- CPU smoke、loss probe、synthetic bad-interface probe、gradient route probe、CPU 短 overfit/mini fine-tune 均通过。
- 1 卡 GPU 正式 run 未完成，原因是 SLURM `srun --immediate` 无法立即分配 GPU，报错为 `Immediate execution impossible, insufficient priority`。按照你的规则，没有排队。

因此本阶段当前状态是：**Stage04 数据与 loss 代码已实现并验证可跑；1 卡 GPU 正式 300/500 步验收因集群即时资源不可用而等待下一次有卡时补跑**。

## 2. 为什么区分小分子 ligand、peptide/protein ligand 与 protein binder

本阶段继续沿用你确认后的科学定义。

小分子 ligand 没有氨基酸序列，因此不能提供 `shared_binder_sequence`。如果一个数据条目只是 ligand-binding protein 的 apo/holo 结构，那么该蛋白序列是 target/receptor 序列，不是 binder 序列。它适合训练 target ensemble encoder 理解 pocket 构象变化、apo/holo 权重、clash/contact proxy，但不能直接作为 protein binder 的共享序列监督。

peptide ligand、protein ligand、天然 protein partner、抗体链、受体配体、抑制蛋白这类对象有氨基酸序列，且在复合物中作为另一条链与 target 结合。因此它们可以成为 `shared_binder_sequence`。这类数据才是 Stage04 protein binder 主监督最优先的数据。

本阶段 debug set 使用的是已有多链蛋白复合物中的真实 protein chain。对 `1tnf_homotrimer` 来说，shared binder sequence 来自同一真实 PDB 复合物的 binder chain，target states 来自同一 target-chain role 在 `cropped/repacked` 两个真实结构文件中的坐标。

## 3. 数据构造流程

数据构造脚本：

```bash
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python \
  scripts/strategy01/stage04_build_real_multistate_complex_dataset.py \
  --num-samples 16 \
  --train-count 12 \
  --val-count 4 \
  --target-len 48 \
  --binder-len 24 \
  --kmax 5
```

输出路径：

- `data/strategy01/stage04_real_complex_multistate/manifest_stage04_debug.json`
- `data/strategy01/stage04_real_complex_multistate/stage04_debug_samples.pt`
- `data/strategy01/stage04_real_complex_multistate/complexes/*.pdb`

每条样本包含以下核心字段：

- `target_id`：本次为 `1tnf_homotrimer`。
- `target_state_paths`：真实 target state 来源 PDB。
- `target_state_chain_ids`：state 对应 target chain。
- `shared_binder_sequence`：同一 binder chain 的氨基酸序列。
- `predicted_complex_paths`：本阶段为 experimental/cached complex-derived two-chain PDB，target 写作 chain A，binder 写作 chain B。
- `x_target_states [K,Nt,37,3]`。
- `target_mask_states [K,Nt,37]`。
- `seq_target_states [K,Nt]`。
- `binder_seq_shared [Nb]`。
- `x_1_states['bb_ca'] [K,Nb,3]`。
- `x_1_states['local_latents'] [K,Nb,8]`。
- `interface_contact_labels [K,Nt,Nb]`。
- `interface_distance_labels [K,Nt,Nb]`。
- `interface_label_mask [K,Nt,Nb]`。
- `interface_quality_labels [K,5]`。

本阶段 `local_latents` 使用 deterministic geometry-proxy latent，而不是完整 AE encoder mean。原因是 AE encoder checkpoint 的输入特征需要 `x1_a37coors_nm/x1_a37coors_nm_rel/x1_bb_angles/x1_sidechain_angles/chain_break_per_res` 的完整训练前处理管线。本阶段先把 Stage04 dataset/loss/training schema 打通；下一阶段应把 geometry-proxy latent 替换为真实 `complexa_ae` encoder mean，并把这一步作为单独 probe 验收。

## 4. 数据构造错误与修复

### 错误 1：固定 residue offset 裁剪导致界面被裁掉

初版 builder 按固定 `target_offsets` 和 `binder_offsets` 裁剪。结果第一条样本出现 `contact_count=0`，说明裁剪没有覆盖 target-binder 界面。

根因：protein complex 中界面残基不是固定在序列开头，固定 offset 会把真实 interface 裁掉。

修复：新增 `contact_windows()`，先计算全链 target-binder CA 距离，从真实链间接触或最近距离 pair 反推 target/binder crop window，再围绕界面裁剪。

### 错误 2：同一三聚体中混入非邻接链作为 required state

第二版虽然能裁到界面，但同一 binder chain 对某些 oligomer neighbor 没有直接界面，仍出现部分 state `contact_count=0`。

根因：把不同邻接链混成同一 required-bind state set。对 homotrimer 来说，chain A 的真实界面可能主要面对 chain B，而不是 chain C。

修复：重写 `build_samples()`，每条样本固定一个 `target_chain_for_states`，只在多个 source files 中跟踪同一个 target-chain role；并用 `full_chain_contact_count() >= 4` 和裁剪后 `contact_count >= 4` 双重过滤。

修复后数据统计：

- `samples = 16`
- `states = 32`
- `K = 2` for all samples
- `contact_count min/mean/max = 8 / 16.25 / 26`
- `targets = {'1tnf_homotrimer': 16}`
- `severe_clash = 0`

## 5. 代码改动清单

### 5.1 模型输出改动

文件：`src/proteinfoundation/nn/local_latents_transformer_multistate.py`

新增模块：

```python
self.interface_quality_head = nn.Sequential(
    nn.LayerNorm(self.token_dim),
    nn.Linear(self.token_dim, 5, bias=False),
)
```

新增输出：

```python
interface_quality_logits [B,K,5]
```

这 5 个质量代理维度当前对应：

- `1 - iPAE_proxy / 31`
- `interface_pLDDT_proxy / 100`
- `ipTM_proxy`
- `pDockQ2_proxy`
- `no_severe_clash`

科学意义：不在训练时调用外部 AF2/Boltz，也不把不可微 predictor 指标直接塞进主 loss，而是让 Complexa 内部学一个 lightweight quality proxy，使共享 binder 的生成过程能感知界面质量趋势。

### 5.2 多状态 loss 改动

文件：`src/proteinfoundation/flow_matching/multistate_loss.py`

Stage03 原 loss：

```text
L_total = lambda_seq * L_seq + lambda_struct * (alpha*L_mean + beta*L_cvar + gamma*L_var)
```

Stage04 新增 per-state 几何项：

```text
L_state_k = L_fm_k
          + lambda_contact * L_contact_k
          + lambda_distance * L_distance_k
          + lambda_clash * L_clash_k
          + lambda_quality_proxy * L_quality_proxy_k
```

Stage04 新增 sample-level 项：

```text
L_total = lambda_seq * L_seq
        + lambda_struct * L_struct
        + lambda_anchor_persistence * L_anchor
        + lambda_self_geometry * L_self_geometry
```

默认权重来自 `configs/training_local_latents_multistate_stage04_probe.yaml`：

```yaml
lambda_contact: 0.20
lambda_distance: 0.10
lambda_clash: 0.05
lambda_anchor_persistence: 0.05
lambda_self_geometry: 0.05
lambda_quality_proxy: 0.02
alpha_mean: 0.45
beta_cvar: 0.45
gamma_var: 0.10
cvar_topk: auto
```

每个 loss 项的意义：

- `L_contact`：让生成 binder 在每个 target state 上恢复真实/预测复合物的界面 contact map。
- `L_distance`：只在 interface label mask 上约束 target-binder 距离，避免只学到二值接触而不学几何尺度。
- `L_clash`：对 target-binder 过近距离加 softplus hinge，防止模型用穿插结构换取 contact。
- `L_anchor`：鼓励跨状态持续出现的 interface anchors 被共同保留。
- `L_self_geometry`：约束同一 shared binder 在不同 target states 的内部 CA 距离不要乱漂，符合“一个序列兼容多状态”的主线目标。
- `L_quality_proxy`：用离线 predictor 或 geometry proxy 的质量标签监督模型内部质量头。

### 5.3 数据构造脚本

新增：`scripts/strategy01/stage04_build_real_multistate_complex_dataset.py`

功能：

- 从 benchmark baseline 仓读取真实多链 PDB。
- 选择真实 protein chain 作为 shared binder。
- 选择与 binder 有真实界面接触的 target chain/state。
- 围绕 interface contact 自动裁剪 target/binder window。
- 写出 two-chain complex PDB。
- 计算 contact、distance、clash 和 quality proxy。
- 生成 manifest 和 tensor dataset。

### 5.4 Predictor adapter

新增目录：`scripts/strategy01/stage04_predictors/`

新增文件：

- `base.py`
- `af2_colabdesign_adapter.py`
- `boltz_adapter.py`
- `protenix_adapter.py`
- `__init__.py`

本阶段 adapter 的作用是固定接口和明确失败模式。当前远程环境检测到：

- `colabdesign` Python 包存在。
- `AF2_DIR` 未设置。
- `COLABDESIGN_DATA_DIR` 未设置。
- `boltz` CLI 不存在。
- `protenix` Python 包不存在。

因此本阶段没有假装跑 AF2/Boltz，而是使用 `experimental_chain_extractor_geometry_proxy` 作为 debug predictor。

### 5.5 Stage04 probe/training 脚本

新增：`scripts/strategy01/stage04_real_complex_loss_debug.py`

功能：

- 读取 Stage04 tensor dataset。
- 支持 variable-K padding。
- 构造 Stage04 batch。
- 初始化 Strategy01 multistate model 和 ProductSpaceFlowMatcher。
- 加载可复用 checkpoint 参数，新模块随机初始化。
- 执行 loss unit probe。
- 执行 bad-interface synthetic probe。
- 执行 gradient route probe。
- 执行 1-sample overfit。
- 执行 mini fine-tune。
- 输出 JSON probe report。

### 5.6 配置文件

新增：`configs/training_local_latents_multistate_stage04_probe.yaml`

这个配置保留 Stage03 的 shared corruption 和多状态 robust aggregation，同时打开 Stage04 interface loss。

## 6. 测试结果

### 6.1 静态编译

命令：

```bash
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python -m compileall \
  src/proteinfoundation/flow_matching/multistate_loss.py \
  src/proteinfoundation/nn/local_latents_transformer_multistate.py \
  scripts/strategy01/stage04_build_real_multistate_complex_dataset.py \
  scripts/strategy01/stage04_real_complex_loss_debug.py \
  scripts/strategy01/stage04_predictors
```

结果：通过。

### 6.2 CPU smoke

命令：

```bash
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python \
  scripts/strategy01/stage04_real_complex_loss_debug.py \
  --device cpu \
  --run-name stage04_cpu_smoke \
  --overfit1-steps 1 \
  --mini-steps 1 \
  --eval-every 1 \
  --mini-batch-size 1
```

输出：`reports/strategy01/probes/stage04_cpu_smoke_results.json`

结果：通过。

### 6.3 CPU 短训练闭环

命令：

```bash
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python \
  scripts/strategy01/stage04_real_complex_loss_debug.py \
  --device cpu \
  --run-name stage04_cpu_short_real_complex_80_120 \
  --overfit1-steps 80 \
  --mini-steps 120 \
  --eval-every 40 \
  --mini-batch-size 1
```

输出：`reports/strategy01/probes/stage04_cpu_short_real_complex_80_120_results.json`

关键结果：

- `loss_unit`: passed
- `synthetic_interface`: passed
- `grad_route`: passed
- synthetic bad-interface probe：`good_contact_loss = 1.27`，`bad_contact_loss = 50.49`
- synthetic distance probe：`bad_distance_loss > good_distance_loss`
- 1-sample overfit：`46.83 -> 11.43`，下降 `75.60%`
- mini fine-tune：`11.43 -> 7.08`，下降 `38.03%`
- validation total：`12.47`

解释：CPU 短训练没有替代 GPU 正式验收，但已经证明 Stage04 的真实复合物 labels、interface loss、quality head 和训练梯度链路是可学习的。

### 6.4 GPU 1 卡正式 run

命令：

```bash
srun --immediate=1 -p new --gres=gpu:1 --time=02:00:00 \
  /home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python \
  scripts/strategy01/stage04_real_complex_loss_debug.py \
  --device cuda \
  --run-name stage04_gpu_1x_real_complex_300_500 \
  --overfit1-steps 300 \
  --mini-steps 500 \
  --eval-every 100 \
  --mini-batch-size 1
```

结果：未启动，SLURM 返回：

```text
srun: error: Unable to allocate resources: Immediate execution impossible, insufficient priority
```

随后尝试 `-p all --gres=gpu:1 --immediate=1` 的最小 CUDA probe，也返回同样错误。

处理：根据“不能排队、只有 1 张 GPU 权限”的规则，本阶段没有提交排队任务。GPU 正式 run 待资源可立即分配时重跑。

## 7. 本阶段仍未完成的部分

### 7.1 外部 predictor 真正生成 predicted complex

本阶段没有跑 AF2/Boltz/Protenix 生成复合物。原因是当前远程环境缺少必要权重路径或命令：

- `AF2_DIR`: missing
- `COLABDESIGN_DATA_DIR`: missing
- `boltz`: missing
- `protenix`: missing

本阶段用 cached/experimental multichain complex 构造 debug set，目的是先验证 Stage04 schema 和 loss，而不是伪装完成大规模 predictor pipeline。

下一步需要在有 GPU 时完成：

1. 找到原论文 benchmark 可跑环境中 AF2/ColabDesign 的参数加载方式。
2. 用同一 shared binder sequence 对每个 target state 做 refold/predict。
3. 解析真实 predictor 输出的 pLDDT、PAE/ipAE、ipTM。
4. 替换本阶段的 geometry proxy confidence labels。

### 7.2 AE deterministic local latent

本阶段 `x_1_states['local_latents']` 是 geometry proxy，不是 `complexa_ae` encoder mean。

原因：AE encoder 需要完整 atom37 与 torsion feature preprocessing，本阶段为了先验证 Stage04 数据/loss/training schema，未把 AE transform pipeline 接入。

下一步建议单独做 `stage04_ae_latent_extractor_probe.py`：

- 输入 predicted complex binder chain。
- 构造 AE 需要的 `coords_nm/coord_mask/residue_type/residue_pdb_idx`。
- 运行 `complexa_ae_init_readonly_copy.ckpt` encoder。
- 保存 `mean [Nb,8]`。
- 对比 geometry proxy 与 AE mean 的 shape、范围、训练稳定性。

### 7.3 数据多样性

当前 16 条 debug 样本全部来自 `1tnf_homotrimer`。这足够测试 schema 和 loss，但不足以证明泛化。

下一步扩展顺序：

1. 从 PINDER/DIPS-Plus 或 RCSB 下载更多 protein-protein/peptide-protein complexes。
2. 从 CoDNaS-Q/PDBFlex/ProteinBench 补 target state ensemble。
3. 用真实 predictor refold 统一 shared binder sequence 到每个 state。
4. 按 target/family 切分 validation。

## 8. 复现流程

从空 Stage04 输出目录复现当前结果：

```bash
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase

/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python -m compileall \
  src/proteinfoundation/flow_matching/multistate_loss.py \
  src/proteinfoundation/nn/local_latents_transformer_multistate.py \
  scripts/strategy01/stage04_build_real_multistate_complex_dataset.py \
  scripts/strategy01/stage04_real_complex_loss_debug.py \
  scripts/strategy01/stage04_predictors

/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python \
  scripts/strategy01/stage04_build_real_multistate_complex_dataset.py \
  --num-samples 16 \
  --train-count 12 \
  --val-count 4 \
  --target-len 48 \
  --binder-len 24 \
  --kmax 5

/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python \
  scripts/strategy01/stage04_real_complex_loss_debug.py \
  --device cpu \
  --run-name stage04_cpu_short_real_complex_80_120 \
  --overfit1-steps 80 \
  --mini-steps 120 \
  --eval-every 40 \
  --mini-batch-size 1
```

有 1 张 GPU 可立即使用时，补跑：

```bash
srun --immediate=1 -p new --gres=gpu:1 --time=02:00:00 \
  /home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python \
  scripts/strategy01/stage04_real_complex_loss_debug.py \
  --device cuda \
  --run-name stage04_gpu_1x_real_complex_300_500 \
  --overfit1-steps 300 \
  --mini-steps 500 \
  --eval-every 100 \
  --mini-batch-size 1
```

## 9. 完成判定

已完成：

- Stage04 interface-aware loss 代码。
- `interface_quality_head` 模型输出。
- Stage04 debug dataset builder。
- Predictor adapter skeleton 与明确环境失败模式。
- 16 条真实多链复合物 derived debug set。
- CPU smoke、loss probe、synthetic interface probe、gradient route probe。
- CPU 短 overfit/mini fine-tune。
- 中文报告。

未完成但已明确阻塞原因：

- 1 卡 GPU 正式 300/500 steps run：SLURM immediate 无法分配 GPU。
- AF2/Boltz/Protenix 真 predictor 生成：当前环境缺权重路径或命令。
- AE deterministic latent extractor：需要单独接入 AE feature preprocessing。

本阶段的科学意义是把 Strategy01 的多状态目标从“只拟合 state-wise backbone/local latent”推进到“共享 binder 序列 + 多状态复合物界面几何 + 质量代理”的监督形式。它仍然不是最终 benchmark 结果，但已经把下一步真实 predictor 数据生产和 1 卡微调的接口、loss、报告路径打通。
