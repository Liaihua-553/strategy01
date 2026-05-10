# Strategy01 Stage17 多候选 target-only 采样与安全选择报告

## 阶段目标

本阶段继续围绕核心科学目标：**只输入多状态 target 构象，模型从零生成一个共享 binder 序列，并为每个 state 生成合理 target-binder 复合物几何**。Stage15/16 主要修训练条件，Stage17 不再只看单条采样轨迹，而是检查 target-only 多 seed 采样里是否已经存在更好的候选，以及能否用不泄漏真实 binder 标签的指标选出来。

## 相对 Stage12/15 的代码改动

- `scripts/strategy01/stage12_de_novo_multistate_training.py`：新增 Stage16 near-clean `t` window 训练开关，用于验证 late-denoising 监督是否能改善 sampled latent 质量。
- `scripts/strategy01/stage12c_de_novo_smoke.py`：新增 `target_binder_geometry_proxy()`，只用 target CA/hotspot 与生成的 binder CA 计算 label-free 几何代理，包括最小距离、接触数量、hotspot 接触数量和 severe clash rate。
- `scripts/strategy01/stage17_multiseed_denovo_probe.py`：新增全局多 seed 采样探针，用 no-leak proxy 选候选，真实序列 identity 只作为后验诊断。
- `scripts/strategy01/stage17_per_sample_multiseed_probe.py`：新增逐样本多 seed 选择探针，避免一个全局 seed 掩盖 target-specific 差异。
- `scripts/strategy01/slurm/stage16_twindow_selfcond.sbatch`、`stage17_*.sbatch`：新增单 A100 运行脚本，继续使用 `new/gu02`，并设置 runtime cache 避免 compute node `/tmp` 空间问题。

## Stage15 结果回顾：teacher-forced self-conditioning 有正信号

Stage15 从 warmup checkpoint 启动 teacher-forced self-conditioning，结果如下：

- validation total loss：`34.762115478515625`
- validation shared sequence identity：`0.6477273106575012`
- validation AE-state identity：`0.4962121248245239`
- pilot final shared identity：`NA`
- target-only single-state audit n16：`{'final_x_ae_shared_identity_mean': 0.2454545497894287, 'final_x_ae_state_identity_mean': 0.2454545497894287}`
- target-only single-state audit n32：`{'final_x_ae_shared_identity_mean': 0.20000000298023224, 'final_x_ae_state_identity_mean': 0.20000000298023224}`

解释：teacher-forced self-conditioning 让训练和模型 readout 更一致，target-only rollout 比 Stage14 有小幅提升，但离共享 binder 序列可用还远。

## Stage16 结果：near-clean t-window 不是主线

Stage16 将监督偏向 late denoising 区间，teacher-forced validation 继续变好，但 target-only rollout 变差：

- validation total loss：`27.857648849487305`
- validation shared identity：`0.7386363744735718`
- validation AE-state identity：`0.5568181872367859`
- target-only n16：`{'final_x_ae_shared_identity_mean': 0.21818183362483978, 'final_x_ae_state_identity_mean': 0.21818183362483978}`
- target-only n32：`{'final_x_ae_shared_identity_mean': 0.13181817531585693, 'final_x_ae_state_identity_mean': 0.13181817531585693}`

结论：只把 supervised interpolant 压到 near-clean 区间，会让 teacher-forced 指标更好，但不能自动修复 target-only rollout 分布。这个结果说明下一步不能盲目继续降低训练 loss，而要看采样轨迹和候选物理合理性。

## Stage17 多 seed 全局选择

Stage17 对 Stage15 checkpoint 做多 seed target-only 采样。选择指标不使用真实 binder 序列或 exact pose，只用模型自身置信度与 state 间一致性。

| 数据 | 样本数 | proxy 选中 seed | proxy identity | oracle identity | identity range | proxy 是否等于 oracle |
|---|---:|---:|---:|---:|---|---|
| val4 n16 | 4 | `1249` | `0.27272728085517883` | `0.27272728085517883` | `[0.23863637447357178, 0.27272728085517883]` | `True` |
| val12 n16 | 12 | `1249` | `0.2861599326133728` | `0.2861599326133728` | `[0.2710084319114685, 0.2861599326133728]` | `True` |
| val36 n16 | 36 | `1249` | `0.2142124722401301` | `0.2142124722401301` | `[0.19960208733876547, 0.2142124722401301]` | `True` |

结论：多 seed 确实能找到略好的候选，说明 test-time scaling 对 Strategy01 仍有价值。但全局 seed 选择增益很小，不能解决本质问题。

## Stage17 逐样本选择与几何代理

逐样本选择结果：

- 仅 entropy/disagreement proxy：`{'first_seed_identity_mean': 0.27858416736125946, 'proxy_selected_identity_mean': 0.28237204626202583, 'oracle_identity_mean': 0.2861599251627922, 'proxy_matches_oracle_rate': 0.0}`
- 第一次 geometry proxy 运行：`{'first_seed_identity_mean': 0.27858416736125946, 'proxy_selected_identity_mean': 0.28237204626202583, 'oracle_identity_mean': 0.2861599251627922, 'proxy_matches_oracle_rate': 0.0}`。这次运行后发现候选行未写入 geometry 字段，selection 实际未使用几何指标，作为 bug 记录，不作为有效科学结论。
- 修复后的 geometry proxy：`{'first_seed_identity_mean': 0.27858416736125946, 'proxy_selected_identity_mean': 0.2747962884604931, 'oracle_identity_mean': 0.2861599251627922, 'proxy_matches_oracle_rate': 0.0}`

修复后，geometry proxy 将 severe clash 作为高惩罚项。结果出现一个重要现象：proxy-selected identity 从 `0.2824` 降到 `0.2748`，但这是合理的，因为高 identity 候选经常存在 severe clash。生成的 96 个候选中：

- severe clash rate 均值约 `0.724`
- target contact count 均值约 `80.17`
- hotspot contact count 均值约 `21.16`
- 最小 target-binder 距离均值约 `0.291 nm`

解释：当前模型不是完全不靠近 target；相反，它经常过度靠近 target 并产生硬碰撞。对真实 binder 设计来说，severe clash 是 hard fail，不能因为 reference sequence identity 更高就接受。

## warning / error 审查

- `OptionalCaCoorsNanometersSeqFeat`、`OptionalResidueTypeSeqFeat`、`OptionalCaCoorsNanometersPairwiseDistancesPairFeat` 返回 zeros：这是 no-leak 设计的一部分。Stage12/17 target-only de novo 主线禁止真实 binder CA/residue type 进入输入，因此这些 warning 不影响科学有效性。
- `CCD_MIRROR_PATH`、`PDB_MIRROR_PATH` 未设置：本阶段使用已张量化数据和已加载 checkpoint，没有调用需要本地 PDB/CCD mirror 的解析功能；不影响本阶段结论。
- 没有发现 Traceback/OOM/NaN。早期 compute-node `/tmp` 空间问题已通过 SLURM runtime cache 环境变量修复。

## 当前结论

Stage15 证明 self-conditioning 有帮助；Stage16 证明只优化 teacher-forced late-denoising 不够；Stage17 证明多候选选择有小增益，但也暴露了更关键的几何问题：**target-only 生成候选经常出现 severe clash**。这直接违反科学目标，因为有硬碰撞的 binder 不可能形成合理结合界面。

因此，下一阶段不能继续只追 sequence identity 或增加训练步数。应进入 Stage18：将 severe clash hard gate、target-shell/contact guidance 和物理安全 candidate selection 接入 target-only sampling，同时保持 no-leak 和 state-specific 输出。

## 下一步 Stage18 建议

1. 采样端加入 target-only hard gate：有 severe clash 的候选不能作为成功候选。
2. 加入 bounded rigid-body clash relief，但只允许小幅整体位移，避免 Stage09D 式把界面推飞。
3. selection 排序改为：no severe clash -> target/hotspot contact shell -> state consistency -> sequence entropy/disagreement。
4. 先在 val12 n16 做 smoke，若 clash 明显下降且 contact 不崩，再跑 val36/48。
5. 若安全采样仍不能得到 no-clash 候选，则说明训练中的 bb_ca flow 还没有学到 target-relative placement，需要回到数据/loss 层增强 clash/contact 监督。

## 复现命令摘要

```bash
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python -m py_compile   scripts/strategy01/stage12_de_novo_multistate_training.py   scripts/strategy01/stage12c_de_novo_smoke.py   scripts/strategy01/stage17_multiseed_denovo_probe.py   scripts/strategy01/stage17_per_sample_multiseed_probe.py
sbatch scripts/strategy01/slurm/stage17_multiseed36.sbatch
sbatch scripts/strategy01/slurm/stage17_geo2_persample.sbatch
```
