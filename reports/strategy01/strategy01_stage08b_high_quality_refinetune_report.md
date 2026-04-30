# Strategy01 Stage08B 高质量数据补完、再微调与 exact benchmark 闭环报告

## 1. 阶段结论

本阶段在远程策略仓 `/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase` 内执行，未修改 benchmark baseline 仓，也未覆盖 baseline checkpoint。阶段目标原本是补齐 curated exact/hybrid 数据、接入真实 AE latent、跑 `>=256 train / >=64 val` pilot、完成 B0/B1/B2 exact benchmark。实际完成情况如下：

- 数据源接入完成了 PINDER HF 小分片与 AF2 分片；远程 DNS 不稳定，所以采用本地下载后 scp 到远程。
- 严格 constrained-hybrid PINDER 只接受 `11` 条，训练/验证 `9/2`，远低于目标 `256/64`。
- 已将 PINDER accepted 样本全部接入 Complexa AE latent，`processed_states=22`，无 geometry proxy 混入这 11 条。
- 合并 Stage07 已有 predictor-derived 数据后得到 `119/30` 的 pilot 数据，仍低于目标 `256/64`，因此本轮定位为 **pilot finetune**，不是完整规模训练。
- 单张 gu02 A100 上 `batch_size=4` 成功，mini 1500 steps 跑完，训练曲线与 30 张图已生成。
- exact-only 目前完成 B2 reference 几何审计；已有 B1 只是一条 predictor-derived smoke，不是 V_exact-only 全量生成 benchmark。B0 没有可用 baseline 生成产物，本阶段不伪造 B0。

严格判断：Stage08B 已完成“数据补充 + AE latent + pilot 微调 + exact/reference 几何审计”的闭环，但 **未达到 curated 256/64 数据规模，也未完成 full V_exact B0/B1/B2 生成对比**。下一步应先构建 V_exact tensor dataset 与 baseline/Strategy01 exact generation artifact，再做正式胜负评估。

## 2. 代码改动清单

| 文件 | 改动目的 | 关键输入 | 关键输出 |
|---|---|---|---|
| `scripts/strategy01/stage08b_build_pinder_dataset.py` | 从 PINDER parquet 构造 constrained-hybrid 多状态样本；用 bound receptor/binder 作 source complex，apo/pred receptor 通过 target alignment 迁移 binder pose | PINDER parquet (`complex`, `apo_receptor`, `pred_receptor`) | `stage08b_pinder_hybrid_samples.pt`、train/val tensor、manifest、summary |
| `scripts/strategy01/stage08b_merge_training_sets.py` | 合并 Stage07 predictor-derived 数据与 Stage08B PINDER hybrid 数据，保持 split 和 source 标记 | 多个 `.pt` dataset | `stage08b_merged_pilot_samples.pt` 与 merge summary |
| `scripts/strategy01/stage08b_train_pilot.sbatch` | 单 A100 执行 1-sample、4-sample、mini pilot，并绘制 loss 曲线 | merged pilot dataset | `stage08b_merged_pilot_results.json`、checkpoint、figures |
| `scripts/strategy01/stage08b_exact_geometry_benchmark.py` | 审计 `V_exact` exact reference 几何，并对已有 B1 smoke 做 contact/RMSD 诊断；不伪造 B0 | `V_exact_main_manifest.json`、sampling smoke | `stage08b_exact_geometry_benchmark_summary.json` |
| `scripts/strategy01/stage08b_write_report.py` | 汇总所有 summary、训练结果、benchmark 诊断，生成中文报告 | probe JSONs | 本报告 |

另外修正了 `stage08b_build_pinder_dataset.py` 的 summary 状态逻辑：只有达到请求的 train/val 数量才是 `passed`；有少量样本但未达标应标为 `short_target`。这是为了避免 11 条样本被误读成完整数据集。

## 3. 数据源与筛选结果

### 3.1 PINDER 数据接入

实际可用 parquet：

- `pinder_s-00000-of-00001.parquet`：250 行。
- `pinder_af2-00000-of-00001.parquet`：180 行。
- `pinder_xl` 本地下载多次只得到不完整 partial 文件，远程 pyarrow 无法读取，所以本阶段未使用。

严格筛选结果：

| 配置 | accepted | train | val | families train | families val | 主要拒绝原因 |
|---|---:|---:|---:|---:|---:|---|
| strict | 11 | 9 | 2 | 9 | 2 | `{'low_contacts': 2, 'binder_length': 164, 'insufficient_motion_states': 180, 'severe_clash': 5, 'target_length': 51, 'low_contact_f1': 17}` |
| binder_max_len=300 probe | 15 | 12 | 3 | 12 | 3 | `{'low_contacts': 3, 'insufficient_motion_states': 271, 'binder_length': 59, 'low_contact_f1': 22, 'severe_clash': 7, 'target_length': 51, 'low_persistent_anchors': 2}` |
| bronze-like relaxed probe | 17 | 14 | 3 | 14 | 3 | `{'low_contacts': 3, 'insufficient_motion_states': 249, 'low_contact_f1': 27, 'binder_length': 59, 'severe_clash': 18, 'low_persistent_anchors': 6, 'target_length': 51}` |

主要瓶颈不是脚本错误，而是这两个小分片中可同时满足“合理 binder 长度、足够 target motion、contact-F1、clash 和 persistent anchors”的样本太少。最大拒绝项是 `insufficient_motion_states` 与 `binder_length`，这说明要达到几百条训练集必须接入更完整的 PINDER 分片、Dockground/ProtCID/PepBDB/Propedia/PepX 或自行构造 constrained hybrid，而不能依赖当前两个小分片。

### 3.2 AE latent 接入

AE 提取使用阶段副本：

`ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt`

结果：

- 输入：`data/strategy01/stage08b_pinder_hybrid/stage08b_pinder_hybrid_samples.pt`
- 输出：`data/strategy01/stage08b_pinder_hybrid/stage08b_pinder_hybrid_samples_ae_latents.pt`
- 样本数：`11`
- 状态数：`22`
- 设备：`cpu`
- 结论：PINDER accepted 样本已全部替换为真实 `encoder.mean [K,Nb,8]`，未回退 geometry proxy。

### 3.3 合并训练集

合并后：

- 总样本：`149`
- train/val：`119/30`
- 来源：`{'stage07_v23_wave1': 138, 'pinder_hybrid_silver': 11}`
- train families：`31`
- val families：`7`

这个规模足够测试训练链路和收敛行为，但不足以宣称完成了 Stage08B 原定的 `256/64` curated 训练集。

## 4. 微调训练结果

训练作业：`stage08b_train_pilot.sbatch`，SLURM job `1911816`，运行在 `new/gu02` 单张 A100。

batch 选择：

```json
{
  "selected_batch_size": 4,
  "attempts": [
    {
      "batch_size": 4,
      "ok": true,
      "error": null,
      "cuda_max_mem_gb": 2.5549144744873047
    }
  ]
}
```

训练总耗时：`1432.1 sec`。

| 阶段 | steps | batch | eval total 起点 | eval total 终点 | 下降 | eval CVaR 起点 | eval CVaR 终点 | 显存GB | sec/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overfit1 | 300 | 1 | 31.631 | 24.209 | 23.5% | 31.621 | 26.876 | 6.33 | 1.042 |
| overfit4 | 600 | 4 | 31.091 | 16.767 | 46.1% | 28.168 | 18.599 | 6.33 | 0.925 |
| mini | 1500 | 4 | 30.157 | 8.185 | 72.9% | 27.855 | 9.165 | 6.33 | 0.368 |

各 state eval total：

| 阶段 | state0 起点→终点 | state1 起点→终点 | state2 起点→终点 |
|---|---:|---:|---:|
| overfit1 | 31.745→28.468 | 31.498→25.285 | 26.539→24.743 |
| overfit4 | 29.061→17.813 | 27.276→17.452 | 11.430→10.535 |
| mini | 29.584→8.844 | 26.127→9.487 | 11.355→3.585 |

观察：

- `mini` eval total 从 `30.157` 降到 `8.185`，下降 `72.9%`。
- `mini` CVaR 从 `27.855` 降到 `9.165`。
- 三个 state loss 都下降，说明 state-specific trajectories 和 shared sequence consensus 的训练路径能共同学习，不是只优化 easiest state。
- 但 final full validation total 为 `84.073`，明显高于 mini 曲线中的 eval subset，说明当前数据混合仍有分布差异，不能进入大规模训练。

训练图输出目录：

`reports/strategy01/figures/stage08b/merged_pilot/`

代表图：

![mini eval total](/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/figures/stage08b/merged_pilot/mini_eval_total_linear.png)

![mini state0](/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/figures/stage08b/merged_pilot/mini_multistate_state_0_justlog_linear.png)

![mini state1](/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/figures/stage08b/merged_pilot/mini_multistate_state_1_justlog_linear.png)

![mini state2](/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/figures/stage08b/merged_pilot/mini_multistate_state_2_justlog_linear.png)

## 5. Exact/reference benchmark 现状

运行脚本：

`scripts/strategy01/stage08b_exact_geometry_benchmark.py`

输出：

`reports/strategy01/probes/stage08b_exact_geometry_benchmark_summary.json`

### 5.1 B2 exact-only reference

- exact samples：`48`
- contact cutoff：`10.0 Å`
- B2 contact count summary：`{'n': 144, 'mean': 125.92361111111111, 'worst': 34.0, 'best': 417.0}`
- B2 contact persistence summary：`{'n': 48, 'mean': 0.658085925508294, 'worst': 0.411214953271028, 'best': 0.9316239316239316}`
- target motion RMSD summary：`{'n': 48, 'mean': 4.2023125, 'worst': 26.965, 'best': 1.046}`

这部分是 exact/reference 上限，用来判断实验复合物本身是否有足够可持续界面锚点。

### 5.2 B1 Strategy01 smoke

当前已有 B1 生成 artifact 来自 Stage07 sampling smoke：

```json
{
  "status": "smoke_only",
  "sample_id": "2ofq_A_B__v00__k3__hexact",
  "source_tier": "train_boltz",
  "dataset": "/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/data/strategy01/stage07_production_v23_wave1/stage06_predictor_pilot_samples_materialized_ae_latents.pt",
  "smoke_summary": "/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/reports/strategy01/probes/stage07_multistate_sampling_smoke_summary.json",
  "note": "This is B1 against predictor-derived labels, not V_exact-only.",
  "b1_strategy01": {
    "contact_f1_vs_label": {
      "n": 3,
      "mean": 0.06674329562485976,
      "worst": 0.009153318077803205,
      "best": 0.14444444444444446
    },
    "direct_binder_ca_rmsd_A": {
      "n": 3,
      "mean": 21.613474527994793,
      "worst": 22.483097076416016,
      "best": 20.485050201416016
    },
    "aligned_binder_ca_rmsd_A": {
      "n": 3,
      "mean": 15.56036885179428,
      "worst": 17.734740865329318,
      "best": 13.487628568559401
    }
  },
  "state_rows": [
    {
      "state_index": 0,
      "status": "ok",
      "reference_contact_count": 175,
      "generated_contact_count": 365,
      "contact_f1_vs_reference_label": 0.14444444444444446,
      "direct_binder_ca_rmsd_A": 21.872276306152344,
      "aligned_binder_ca_rmsd_A": 17.734740865329318
    },
    {
      "state_index": 1,
      "status": "ok",
      "reference_contact_count": 119,
      "generated_contact_count": 267,
      "contact_f1_vs_reference_label": 0.046632124352331605,
      "direct_binder_ca_rmsd_A": 22.483097076416016,
      "aligned_binder_ca_rmsd_A": 15.458737121494119
    },
    {
      "state_index": 2,
      "status": "ok",
      "reference_contact_count": 109,
      "generated_contact_count": 328,
      "contact_f1_vs_reference_label": 0.009153318077803205,
      "direct_binder_ca_rmsd_A": 20.485050201416016,
      "aligned_binder_ca_rmsd_A": 13.487628568559401
    }
  ],
  "sequence_identity_to_reference": 0.13636363636363635,
  "pred_sequence": "APSSSSSSSSSNLNSYYYPPNS",
  "reference_sequence": "PPPEPDWSNTVPVNKTIPVDTQ"
}
```

它的科学含义有限：它证明 state-specific sampler 能输出 K 个 binder pose 和 shared sequence，但它不是 V_exact-only full benchmark。现有 B1 sequence identity 很低，且仅在 predictor-derived label 上做了 smoke 几何比较。

### 5.3 B0 状态

`B0` 没有可用的 baseline Complexa 单状态生成 artifact。本阶段没有伪造 B0，因此 benchmark summary 中标为 `not_run`。要完成正式 B0/B1/B2，下一步必须先把 V_exact 转成 generation-ready tensor dataset，并跑 baseline 与 Strategy01 的同预算生成。

## 6. 错误、根因与修正

| 问题 | 根因 | 修正 | 验证 |
|---|---|---|---|
| 远程直接下载 PINDER 失败 | 远程 DNS/网络解析不稳定 | 本地下载 parquet 后 scp 到远程 | `pinder_s` 和 `pinder_af2` 可由远程 pandas/pyarrow 读取 |
| Hugging Face Dataset Viewer parquet API 失败 | Viewer 不能处理该数据集的 `Array1D` feature schema | 改用直接 parquet URL | 两个小分片读取成功 |
| `pinder_xl` 文件不完整 | 大文件下载中断，只有 partial | 不纳入本阶段主数据，报告中记录 | 远程 pyarrow 读取失败，未使用 |
| PINDER chain 解析最初错位 | `chain_id` 是按 residue 存，`coords/atom_name` 是按 atom 存 | 通过 `residue_starts` 将 residue chain 映射到 atom interval | strict builder 可完成并输出 11 条 |
| interface label 字段名不一致 | 实际函数返回 `contact_labels/distance_labels/label_mask` | 修正读取字段 | 数据 tensor 构建成功 |
| state target 长度不同导致 tensor stack 问题 | apo/pred receptor 与 bound receptor 长度不完全一致 | 按 sample 内最大 target length padding | builder 成功输出 variable target states |
| motion 过滤单位错误 | Kabsch RMSD 是 nm，阈值是 Å | 比较前乘以 10 | accepted 数量恢复到合理水平 |
| PINDER summary 误标 passed | 原状态逻辑只要求至少 1 条 train/val | 修成达到请求数量才 `passed`，未达标为 `short_target` | 代码已修改 |

## 7. 复现命令

```bash
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python

$PY scripts/strategy01/stage08b_build_pinder_dataset.py \
  --parquet data/strategy01/stage08b_curated_sources/pinder_hf/pinder_s-00000-of-00001.parquet \
  --parquet data/strategy01/stage08b_curated_sources/pinder_hf/pinder_af2-00000-of-00001.parquet \
  --out-dir data/strategy01/stage08b_pinder_hybrid \
  --summary reports/strategy01/probes/stage08b_pinder_build_summary.json \
  --train-count 256 \
  --val-count 64

$PY scripts/strategy01/stage05_extract_ae_latents.py \
  --dataset data/strategy01/stage08b_pinder_hybrid/stage08b_pinder_hybrid_samples.pt \
  --output data/strategy01/stage08b_pinder_hybrid/stage08b_pinder_hybrid_samples_ae_latents.pt \
  --device cpu \
  --summary reports/strategy01/probes/stage08b_pinder_ae_latent_summary.json

$PY scripts/strategy01/stage08b_merge_training_sets.py \
  --dataset data/strategy01/stage07_production_v23_wave1/stage06_predictor_pilot_samples_materialized_ae_latents.pt \
  --source-label stage07_v23_wave1 \
  --dataset data/strategy01/stage08b_pinder_hybrid/stage08b_pinder_hybrid_samples_ae_latents.pt \
  --source-label pinder_hybrid_silver \
  --out-dir data/strategy01/stage08b_merged_training \
  --summary reports/strategy01/probes/stage08b_merge_summary.json

$PY scripts/strategy01/stage07_sequence_consensus_training.py \
  --dataset data/strategy01/stage08b_merged_training/stage08b_merged_pilot_samples.pt \
  --device cpu \
  --run-name stage08b_schema_probe \
  --batch-candidates 4,2,1 \
  --skip-training

sbatch scripts/strategy01/stage08b_train_pilot.sbatch

$PY scripts/strategy01/stage08b_exact_geometry_benchmark.py
$PY scripts/strategy01/stage08b_write_report.py
```

## 8. 下一步建议

1. 先不要扩大到 1000 条训练。当前 full validation loss 高、PINDER strict accepted 太少，继续放大会放大标签偏差。
2. 优先补 `V_exact -> tensor dataset` 转换器，让 exact-only 可以直接跑 Strategy01 state-specific sampler。
3. 同预算跑正式 B0/B1/B2：B0 baseline 单状态生成，B1 Strategy01 多状态生成，B2 exact geometry 上限。
4. 数据生产主线改成完整 PINDER / Dockground / ProtCID-PepBDB curated exact/hybrid，而不是小分片或 free redocking。
5. 对 hybrid silver 增加 source complex template 与 anchor constrained reconstruction；没有 source interface 的自由 docking 不进主监督。

阶段判断：本轮证明了训练链路、AE latent 接入和 state-specific consensus loss 可以跑通并下降；真正限制模型效果的是高质量 exact/hybrid 数据和 exact benchmark artifact 还没补齐。

## 9. Stage08B 未完成项补完：V_exact tensor、B1 exact sampling 与 full benchmark

在上一版报告之后，本阶段继续补完了原先未完成的 exact benchmark 链路。

### 9.1 V_exact tensor dataset

- 构建脚本：`scripts/strategy01/stage08b_build_vexact_tensor_dataset.py`
- 输入 manifest：`data/strategy01/stage08_high_quality_dataset/V_exact_main_manifest.json`
- 输出 tensor：`data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples.pt`
- 样本数：`48`
- reject 数：`0`
- external validation families：`41`

这一步把原来只能用于审计的 `V_exact` manifest 转成了 Strategy01 sampler 能直接读取的 tensor dataset。每条样本包含 target 多状态结构、共享 binder 序列、K 个 exact complex label、binder CA、interface contact/distance labels 和 state mask。所有样本都作为 external validation，不混入训练。

### 9.2 V_exact AE latent

- 脚本：`scripts/strategy01/stage05_extract_ae_latents.py`
- 输出：`data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt`
- 样本数：`48`
- processed states：`144`
- AE checkpoint：`/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt`

结论：`V_exact` 的 local latents 已全部替换为 Complexa AE `encoder.mean [K,Nb,8]`，没有保留 geometry proxy 作为主评估输入。

### 9.3 Stage08B checkpoint 转换与 exact sampling

- 转换脚本：`scripts/strategy01/stage07_convert_mini_to_lightning_ckpt.py`
- 输出 checkpoint：`ckpts/stage07_sequence_consensus/runs/stage08b_merged_pilot/mini_final_lightning.ckpt`
- source phase/steps：`mini/1500`
- exact sampling 脚本：`scripts/strategy01/stage08b_multistate_exact_sampling.py`
- exact sampling 样本数：`48`
- nsteps：`24`
- 总耗时：`76.200 sec`
- 平均耗时：`1.587 sec/sample`

这一步完成了 `V_exact` 上的 B1 Strategy01 多状态生成：一个 shared sequence 加每个 state 的 state-specific binder CA pose。输出目录为 `results/strategy01/stage08b_vexact_sampling`。

### 9.4 Full exact benchmark 结果

运行脚本：`scripts/strategy01/stage08b_full_exact_benchmark.py`

输出：`reports/strategy01/probes/stage08b_full_exact_benchmark_summary.json`

| 项 | 状态 | 说明 |
|---|---|---|
| B0 baseline Complexa | `not_run` | `Original Complexa baseline does not expose a shared sequence head in this Strategy01 tensor sampling path; no baseline generation artifact was available.` |
| B1 Strategy01 | `passed` | V_exact tensor 上 48 条 exact samples 的 Strategy01 生成结果 |
| B2 exact reference | `passed` | 实验 exact complex geometry 上限 |

B1 exact 关键指标：

| 指标 | 结果 |
|---|---|
| sample_count | `48` |
| state_count_ok | `144` |
| contact-F1 | `{'n': 144, 'mean': 0.11708181922641514, 'worst': 0.0, 'best': 0.45810055865921795}` |
| direct/worst interface RMSD Å | `{'n': 144, 'mean': 26.2497934434149, 'worst': 101.48299407958984, 'best': 8.060057640075684}` |
| aligned binder CA RMSD Å | `{'n': 144, 'mean': 11.861835042321196, 'worst': 85.13838813740836, 'best': 4.460184516758356}` |
| generated contact persistence | `{'n': 48, 'mean': 0.49876340509741074, 'worst': 0.0, 'best': 0.7533333333333333}` |
| state metric std contact-F1 | `{'n': 48, 'mean': 0.015135371070050719, 'worst': 0.06758688503862295, 'best': 0.0}` |
| clash rate | `0.7847222222222222` |

B2 exact reference 关键指标：

| 指标 | 结果 |
|---|---|
| exact contact count | `{'n': 144, 'mean': 125.92361111111111, 'worst': 34.0, 'best': 417.0}` |
| exact contact persistence | `{'n': 48, 'mean': 0.658085925508294, 'worst': 0.411214953271028, 'best': 0.9316239316239316}` |
| target motion RMSD Å | `{'n': 48, 'mean': 4.2023125, 'worst': 26.965, 'best': 1.046}` |

### 9.5 科学判断

这个补完步骤给出了明确的负结果：当前 Stage08B pilot 能在训练/内部验证 loss 上下降，但在真正 `V_exact` exact geometry 上泛化很弱。B1 的 mean contact-F1 只有约 `0.117`，clash rate 约 `0.785`。这说明现有 119/30 pilot 数据和 loss 还不足以学到真实多状态 interface geometry。

因此不能进入 1000 样本大训练。下一阶段应先解决三件事：

1. 补原始 Complexa B0 artifact：原模型生成结构后需要接入 ProteinMPNN/sequence_hallucination 或原论文 pipeline 的序列确定步骤，否则不能和 Strategy01 的 shared sequence head 做公平比较。
2. 提升 exact/hybrid 训练数据质量：把 `V_exact` 的一部分或同源 exact/hybrid 数据转成训练集，同时保持 family holdout，避免只用 predictor-derived bronze。
3. 修正采样约束：当前 B1 pose clash 高，说明只靠 flow trajectory 还不够，需要把 exact interface anchors/contact constraints 或 clash-aware guidance 前移到 sampling/reward，而不是只在训练 loss 里弱监督。

