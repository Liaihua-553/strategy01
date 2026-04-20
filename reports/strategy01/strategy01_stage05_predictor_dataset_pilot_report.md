# Strategy01 Stage05：真实 Boltz-2 复合物伪标注数据工厂、Pilot 微调与 B0/B1 复评报告

## 1. 阶段目标

Stage05 的目标是先做真实 predictor-derived 数据闭环，再决定是否进入大规模微调。本阶段不直接构建 500-2000 条 curated samples，因为 Stage04 只证明了多状态 loss 可学习、1 张 A100 可跑通，但数据仍主要来自 `1tnf_homotrimer` 的 16 条 debug 样本，并且当时的 `local_latents` 是 geometry proxy。若现在直接放大训练，会把数据偏差和伪标签噪声一起放大。

本阶段实际完成的闭环：

1. 新增可运行 Boltz-2 adapter。
2. 新增 Stage05 predictor-derived dataset builder。
3. 用 Boltz-2 真实预测 `1` 个样本的 `2` 个 target states。
4. 从 Boltz 输出解析 PDB、confidence、PAE、pLDDT、PDE，并抽取 interface labels。
5. 用 Complexa AE checkpoint 把 predicted binder chain 编码为真实 `encoder.mean [K,Nb,8]`。
6. 用 Stage05 tensor 跑通 GPU loss probe、300-step overfit 和 100-step mini debug fine-tune。
7. 记录所有失败、根因与修复。

结论先写在前面：**Stage05 pipeline 已跑通，但当前 tiny Boltz smoke 样本质量很差，不能进入有效训练；因此不应进入 Stage06 大规模微调。下一步应先改进数据来源和 Boltz 约束/候选筛选。**

## 2. 科学假设与实现边界

多状态兼容 binder 的核心不是“每个状态单独看起来能生成”，而是同一条 binder 序列在多个 target states 上都形成合理复合物界面。Boltz-2 在这里是离线预测器，不进入在线反向传播；它负责给每个 state 生成伪标注复合物和质量指标。

进入训练主 loss 的仍然是可微几何项：flow-matching structure loss、interface contact、interface distance、clash、anchor persistence、quality proxy head。`ipAE / pLDDT / ipTM / pDockQ2 proxy` 用于 hard filter、sample weight、validation report 和 quality proxy supervision，不每步调用外部 predictor 反传。

本阶段主要做 robust multi-state binding 数据闭环，不做 negative design、不接 DynamicMPNN/ADFLIP、不做 B0/B1 大评测。原因是当前 pilot 没有得到 Silver-quality validation set，直接做 B0/B1 没有科学意义。

## 3. 代码改动清单

### 3.1 `.gitignore`

文件：`.gitignore`

新增忽略：

```text
data/strategy01/stage05_predictor_multistate/
reports/strategy01/probes/stage05_boltz_outputs/
ckpts/stage05_predictor_pilot/
```

意义：Stage05 会产生 Boltz PDB/NPZ、tensor dataset 和 checkpoint，这些是大体积实验产物，不进 Git。

### 3.2 Boltz-2 adapter

文件：`scripts/strategy01/stage04_predictors/boltz_adapter.py`

从 Stage04 placeholder 改为真实 adapter：

- 新增 `BoltzRunConfig`。
- 固定默认 CLI：`/data/kfliao/general_model/envs/boltz_cb04aec/bin/boltz`。
- 固定默认 cache：`/data/kfliao/general_model/boltz_cache`。
- 支持写 Boltz YAML：target chain `A`，binder chain `B`，`msa: empty`。
- 支持 target-template PDB：默认写入 `templates: - pdb: ... chain_id: A`。
- 支持 `use_template=False` 作为 fallback smoke。
- 调用命令默认：`--model boltz2 --output_format pdb --no_kernels --write_full_pae --write_full_pde --override`。
- 解析输出：PDB、confidence JSON、PAE NPZ、pLDDT NPZ、PDE NPZ。
- 统一输出 `PredictorResult`。
- 新增指标：`interchain_pae_A`、`interchain_pde_A`、`binder_plddt_norm`、`pDockQ2_proxy`。

科学意义：让真实 predictor 进入离线数据工厂，同时保持训练循环不依赖昂贵不可微 predictor。

### 3.3 Stage05 数据构造入口

文件：`scripts/strategy01/stage05_build_predictor_multistate_dataset.py`

新增功能：

- 从 Stage04 seed tensor 读取 target states 和 shared binder sequence。
- 为每个 state 写 target-template PDB。
- 逐 state 调 Boltz-2 或解析已有 Boltz 输出。
- 从 predicted complex 中抽取 binder CA、contact/distance/clash、quality labels。
- 保存 manifest 和 tensor dataset。
- 支持 `--prepare-only`、`--run-boltz`、`--disable-template`、`--require-bronze`、`--allow-short-dataset`。

关键 shape：

| 字段 | shape | 含义 |
|---|---:|---|
| `x_target_states` | `[K,Nt,37,3]` | 模型输入 target 多状态结构，单位 nm |
| `target_mask_states` | `[K,Nt,37]` | target atom mask |
| `seq_target_states` | `[K,Nt]` | target residue type |
| `target_hotspot_mask_states` | `[K,Nt]` | 从 predicted interface 反推的 hotspot mask |
| `binder_seq_shared` | `[Nb]` | 共享 binder 序列 |
| `x_1_states['bb_ca']` | `[K,Nb,3]` | predictor-derived binder CA label |
| `x_1_states['local_latents']` | `[K,Nb,8]` | 初始 geometry latent，后续可由 AE mean 替换 |
| `interface_contact_labels` | `[K,Nt,Nb]` | target-binder contact label |
| `interface_distance_labels` | `[K,Nt,Nb]` | target-binder CA distance |
| `interface_label_mask` | `[K,Nt,Nb]` | interface label mask |
| `interface_quality_labels` | `[K,5]` | `[PAE_score, ipLDDT, protein_ipTM, pDockQ2_proxy, no_clash]` |

### 3.4 Boltz PDB 稳健解析器

文件：`scripts/strategy01/stage05_build_predictor_multistate_dataset.py`

新增：`_iter_pdb_atom_records()`、`list_chain_ids()`、`chain_to_atom37_any()`。

原因：Boltz 输出 PDB 可能出现坐标数值很大，固定宽度 PDB 列会溢出，例如：

```text
ATOM    168  C   ASP A  21     214.660-362.467 168.860  1.00 49.38           C
```

Biopython `PDBParser` 对这种行会报 `Invalid or missing coordinate(s)`。新解析器优先按空白分隔解析，失败时再回退固定列解析。

科学意义：这不是改变标签含义，只是让数据工厂对 predictor 输出更鲁棒，避免真实预测结果因为格式细节被丢掉。

### 3.5 Template PDB 写入 `SEQRES`

文件：`scripts/strategy01/stage05_build_predictor_multistate_dataset.py`

修复：`write_chain_pdb()` 现在在 ATOM 记录前写入标准 `SEQRES`。

原因：Boltz 内部通过 gemmi 解析 PDB template。没有 `SEQRES` 时，`entity.full_sequence` 长度为 0，Boltz template parser 在 sequence/polymer 对齐时越界。

修复前：Boltz 日志报错 `IndexError: list index out of range`。

修复后：带 `SEQRES` 的 template 可以被 Boltz parser 正常读取，最终 template-conditioned Boltz smoke 通过。

### 3.6 AE latent extractor

文件：`scripts/strategy01/stage05_extract_ae_latents.py`

新增功能：

- 读取 Stage05 tensor dataset。
- 加载只读 AE checkpoint：`ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt`。
- 对每个 predicted complex 的 binder chain B 构造 AE batch。
- 调用 `AutoEncoder.encode(batch)['mean']`。
- 用真实 AE `encoder.mean [K,Nb,8]` 替换 geometry-proxy `local_latents`。

意义：把 Stage05 标签空间拉回 Complexa 原生 local latent 空间，减少 Stage04 geometry proxy 与主模型 latent 空间不一致的问题。

### 3.7 Stage05 pilot debug 入口

文件：`scripts/strategy01/stage05_predictor_pilot_debug.py`

新增功能：

- 显式读取 Stage05 predictor-derived dataset。
- 复用 Stage04 已验证的 multistate/interface loss、probe、overfit 逻辑。
- 训练输出写入 `ckpts/stage05_predictor_pilot/runs/`。
- 结果写入 `reports/strategy01/probes/<run_name>_results.json`。

### 3.8 SLURM 作业脚本

新增脚本：

- `scripts/strategy01/stage05_boltz_smoke.sbatch`
- `scripts/strategy01/stage05_boltz_seqonly.sbatch`
- `scripts/strategy01/stage05_boltz_seqnew.sbatch`
- `scripts/strategy01/stage05_boltz_template.sbatch`
- `scripts/strategy01/stage05_pilot_debug.sbatch`

说明：前几个脚本记录了失败与修复路径；最终有效的是 `stage05_boltz_template.sbatch` 和 `stage05_pilot_debug.sbatch`。

## 4. 执行与测试结果

### 4.1 compileall

命令：

```bash
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python -m compileall \
  scripts/strategy01/stage04_predictors \
  scripts/strategy01/stage05_build_predictor_multistate_dataset.py \
  scripts/strategy01/stage05_extract_ae_latents.py \
  scripts/strategy01/stage05_predictor_pilot_debug.py
```

结果：通过。

### 4.2 Adapter 离线解析已有 Boltz smoke 输出

输出文件：`reports/strategy01/probes/stage05_boltz_adapter_parse_test.json`

关键结果：

```json
{
  "predictor_name": "boltz2",
  "complex_plddt_norm": 0.59047532081604,
  "protein_iptm": 0.027503905817866325,
  "interchain_pae_A": 23.65525245666504,
  "interchain_pde_A": 10.464077949523926,
  "pDockQ2_proxy": 0.0038477894698350874
}
```

结论：adapter 能解析 confidence、PAE、pLDDT、PDE 和 PDB。

### 4.3 prepare-only 测试

命令：

```bash
python scripts/strategy01/stage05_build_predictor_multistate_dataset.py \
  --prepare-only --max-samples 1 --train-count 1 --val-count 0 \
  --states-per-sample 2 \
  --out-dir data/strategy01/stage05_predictor_multistate_prepare_test2
```

结果：通过，生成 `2` 个 state YAML。

### 4.4 Boltz template-conditioned predictor smoke

有效 job：`1899704`

节点：`gu02`

GPU：A100-PCIE-40GB，Driver `550.54.14`，CUDA `12.4`

输出：

- `data/strategy01/stage05_predictor_multistate_smoke_template/manifest_stage05_predictor_pilot.json`
- `data/strategy01/stage05_predictor_multistate_smoke_template/stage05_predictor_pilot_samples.pt`

运行时间：

```json
{
  "num_samples": 1,
  "train_count": 1,
  "val_count": 0,
  "runtime_sec_total": 107.71859431266785,
  "runtime_sec_per_state_median": 53.49950408935547
}
```

质量指标：

```json
{
  "worst_interchain_pae_A": 27.924875259399414,
  "worst_protein_iptm": 0.030764712020754814,
  "worst_complex_plddt_norm": 0.49617353081703186,
  "worst_pDockQ2_proxy": 0.0,
  "worst_contact_count": 0.0,
  "state_metric_std_interchain_pae_A": 0.010142326354980469
}
```

每 state：

| state | contact_count | severe_clash | interchain_pae_A | complex_plddt_norm | protein_iptm | pDockQ2_proxy | bronze_pass |
|---:|---:|---|---:|---:|---:|---:|---|
| 0 | 0 | false | 27.9046 | 0.4962 | 0.0308 | 0.0 | false |
| 1 | 0 | false | 27.9249 | 0.4971 | 0.0318 | 0.0 | false |

解释：pipeline 成功，但这个样本没有形成 target-binder interface，不应进入训练。原因很可能是 seed binder 来自同源 oligomer crop，不等价于一个可由 Boltz 在单序列/弱模板条件下稳定重构的 binder；同时没有 MSA、没有明确 interface constraints、没有候选筛选。

### 4.5 AE latent extractor

命令：

```bash
python scripts/strategy01/stage05_extract_ae_latents.py \
  --dataset data/strategy01/stage05_predictor_multistate_smoke_template/stage05_predictor_pilot_samples.pt \
  --output data/strategy01/stage05_predictor_multistate_smoke_template/stage05_predictor_pilot_samples_ae_latents.pt \
  --device cpu \
  --max-samples 1 \
  --summary reports/strategy01/probes/stage05_ae_latent_extract_smoke_summary.json
```

结果：

```json
{
  "status": "passed",
  "processed_states": 2,
  "elapsed_sec": 0.36960315704345703
}
```

结论：AE `encoder.mean [K,Nb,8]` 路径已跑通。

### 4.6 GPU pilot debug fine-tune smoke

job：`1899708`

节点：`gu02`

输入：`stage05_predictor_pilot_samples_ae_latents.pt`

命令核心：

```bash
python scripts/strategy01/stage05_predictor_pilot_debug.py \
  --dataset data/strategy01/stage05_predictor_multistate_smoke_template/stage05_predictor_pilot_samples_ae_latents.pt \
  --device cuda \
  --run-name stage05_predictor_smoke_template_ae \
  --overfit1-steps 300 \
  --mini-steps 100 \
  --eval-every 100 \
  --mini-batch-size 1
```

结果文件：`reports/strategy01/probes/stage05_predictor_smoke_template_ae_results.json`

关键结果：

```json
{
  "train_count": 1,
  "val_count": 0,
  "overfit1_initial_eval": 8594.5185546875,
  "overfit1_final_eval": 1653.1461181640625,
  "overfit1_drop_fraction": 0.8076511083611045,
  "overfit1_step_time_sec": 0.36672290007273356,
  "mini_initial_eval": 1653.1461181640625,
  "mini_final_eval": 1460.7178955078125,
  "mini_drop_fraction": 0.11640121858674862,
  "mini_step_time_sec": 0.36081169605255126,
  "cuda_max_mem_gb_overfit1": 6.303045272827148,
  "cuda_max_mem_gb_mini": 6.303045272827148
}
```

结论：模型训练链路可跑，loss 可下降，1 张 40GB A100 显存占用约 6.3GB。但因为该样本 `contact_count=0`，`L_contact/L_distance` 为 0，这次只能证明链路，不证明策略有效。

## 5. 错误与修复日志

| 编号 | 错误现象 | 根因 | 修复 | 修后验证 |
|---:|---|---|---|---|
| 1 | `.gitignore` 被写入字面量 `ndata...nreports...` | PowerShell 到 ssh 的换行转义错误 | 用远程 Python 二进制/文本重写 `.gitignore` | `.gitignore` 末尾正确显示三条 Stage05 ignore |
| 2 | `sbatch` 报 `DOS line breaks` | 本地 PowerShell 写出的 `.sbatch` 是 CRLF | 所有 sbatch 先用远程 Python 替换 `\r\n -> \n` | 后续 sbatch 正常提交 |
| 3 | `prepare-only` 写出所有 seed 样本，且 sample id 重复 | `prepare-only` 不增加 `tensor_samples`，循环停止条件失效 | 增加 `prepared_count`，prepare 模式按它计数 | `max-samples=1` 时只生成 2 个 state YAML |
| 4 | Boltz template parser 报 `IndexError: list index out of range` | target-template PDB 没有 `SEQRES`，gemmi `entity.full_sequence` 为空 | `write_chain_pdb()` 写入标准 `SEQRES` | template-conditioned Boltz job `1899704` 通过 |
| 5 | Boltz GPU 初始化报 NVIDIA driver 过旧 | 作业未指定 partition，被调度到旧驱动节点 | Stage05 Boltz/训练作业固定加 `#SBATCH -p new` | 作业运行在 `gu02`，driver `550.54.14` |
| 6 | Biopython PDBParser 解析 Boltz PDB 第 168 行失败 | Boltz 输出坐标很大，固定宽度 PDB 列溢出 | 新增空白分隔优先的稳健 PDB 解析器 | 成功解析 Boltz PDB 并生成 tensor dataset |
| 7 | no-template Boltz smoke 质量极差 | sequence-only、无 MSA、无 interface constraints，binder 不形成界面 | 保留作为失败证据，不作为训练数据；修复 template 路径 | template smoke 也能跑通，但仍质量差，说明需要候选筛选/约束 |

## 6. 当前数据质量判断

本阶段的最重要科学结论不是“可以开始大规模训练”，而是：**数据工厂跑通了，但 naive 的 target-template + shared binder sequence 直接丢给 Boltz，不保证形成 protein-protein interface。**

这意味着 Stage05 的下一步不能是直接 64/128 样本批量生成，而应该先加入以下机制：

1. shared binder sequence 必须来自真实 PPI partner、peptide/protein ligand，或 Complexa/BindCraft 生成且经过初筛的候选。
2. Boltz 输入应加入 interface constraints 或 pocket/contact hints，必要时启用 `--use_potentials`。
3. 若可行，补 MSA 或使用 Boltz MSA server，否则 sequence-only 对 protein-protein docking 类任务质量偏低。
4. 先按 target/family 扩候选预算，保留 Bronze/Silver 样本，不要把 `contact_count=0` 的样本加入训练。
5. 只有形成至少 `12 train + 4 val` 的 Bronze/Silver predictor-derived debug set 后，才做 1-sample/4-sample/mini fine-tune 的效果判断。

## 7. 是否执行 B0/B1 生成复评

本阶段没有执行 B0/B1 生成复评。原因不是代码链路不可行，而是没有得到 Silver-quality validation set。若用 `contact_count=0`、`protein_iptm≈0.03` 的 smoke 样本做 B0/B1，会把评估变成噪声比较，不能指导模型改进。

B0/B1 复评应等到至少满足以下条件后执行：

- validation 中至少 4 个 target/family 或至少 16 条样本；
- 每条样本的 required states 至少 Bronze pass；
- `contact_count >= 8`，`interchain_pae <= 15 Å` 或等价 Boltz interface uncertainty 合格；
- 至少一部分样本达到 Silver 门槛：`complex_plddt_norm >= 0.70`、`protein_iptm >= 0.35`、`pDockQ2_proxy >= 0.23`。

## 8. 下一步建议

不进入 Stage06 大规模微调。建议先做 Stage05b：高质量 predictor-derived 数据源修复。

优先级：

1. 从 PINDER 或 benchmark-valid 复合物中抽真实 protein/peptide partner chain 作为 shared binder sequence，而不是随意裁 oligomer chain。
2. 对每个 target-state 使用真实 bound/unbound target ensemble，并把真实 partner 作为 binder sequence。
3. Boltz 输入增加 contact constraints：从 experimental complex 或 Complexa 初筛结构提取 4-8 个 interface anchors。
4. 对每个 target/binder 运行 `N=8-32` 个 Boltz seeds/constraints，筛 Bronze/Silver，而不是单次预测。
5. 数据集形成后再跑：`12+4` debug、`64+16` pilot、B0/B1 refold。
6. 如果 Boltz template + constraints 仍不稳定，再考虑 Chai/Protenix 或 AF2-Multimer/ColabDesign 作为 predictor 对照。

进入 Stage06 的 gate 维持不变：heldout `L_cvar` 下降至少 15%，且 B1 相比 B0 的 `all_state_pass_rate` 提升至少 10 个百分点，或 `worst_interchain_PAE` 下降至少 10%。

## 9. 复现入口

### 9.1 编译

```bash
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python -m compileall \
  scripts/strategy01/stage04_predictors \
  scripts/strategy01/stage05_build_predictor_multistate_dataset.py \
  scripts/strategy01/stage05_extract_ae_latents.py \
  scripts/strategy01/stage05_predictor_pilot_debug.py
```

### 9.2 Boltz template smoke

```bash
sbatch scripts/strategy01/stage05_boltz_template.sbatch
```

### 9.3 AE latent extract

```bash
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python scripts/strategy01/stage05_extract_ae_latents.py \
  --dataset data/strategy01/stage05_predictor_multistate_smoke_template/stage05_predictor_pilot_samples.pt \
  --output data/strategy01/stage05_predictor_multistate_smoke_template/stage05_predictor_pilot_samples_ae_latents.pt \
  --device cpu \
  --max-samples 1 \
  --summary reports/strategy01/probes/stage05_ae_latent_extract_smoke_summary.json
```

### 9.4 GPU pilot debug

```bash
sbatch scripts/strategy01/stage05_pilot_debug.sbatch
```

## 10. 最终结论

Stage05 的工程闭环已经打通：Boltz adapter、真实 predictor-derived 数据工厂、AE latent extractor、GPU loss/pilot debug 都能跑。但当前 smoke 样本没有形成界面，指标不达 Bronze，不能用于有效微调，更不能直接扩成大规模数据集。

因此，下一阶段应聚焦“高质量共享 binder 数据源 + Boltz interface constraints + 候选筛选”，而不是马上扩大训练规模。这一点和 Strategy01 的科学目标一致：我们要训练模型输出一个能兼容多个 target states 的共享 binder 序列，而不是让模型学习没有界面的伪复合物。
