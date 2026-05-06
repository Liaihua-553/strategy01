# Strategy01 Stage11G 代码与科学目标审查报告

生成时间：2026-05-07T00:52:14

## 1. 本次审查目标

本次审查不是只看代码能不能运行，而是按 Strategy01 的最终科学目标审查：**一个共享 binder 序列要同时兼容 target 的多个真实功能构象，并且每个 state 都有合理的 target-binder complex 几何**。

审查结论分三层：

- 工程层：Stage11G 新增 sampled-latent cache 诊断路径后，代码可编译、GPU 训练和 48-sample 采样可跑通。
- 科学层：当前实现仍未达到最终科学目标。主要瓶颈是 sampled `local_latents_states` 分布不携带足够序列信息，导致 shared sequence 差。
- 方向层：继续把 sampled cache 直接当作 `x_t` 做常规 flow matching 不是正确方向；下一步应做 **latent sequence repair / bounded correction**，把 sampled latent 修正到 AE 可解码出共享序列的区域，而不是让全 FM 结构损失主导训练。

## 2. 审查过的关键代码路径

- `src/proteinfoundation/nn/local_latents_transformer_multistate.py`
  - 已有多状态 target encoder、state-specific heads、`state_seq_logits`、`seq_logits_shared`、soft sequence feedback、`flow_gate`。
  - 代码结构符合“一个 shared sequence + K 个 state-specific pose/latent”的主目标。
- `src/proteinfoundation/flow_matching/multistate_loss.py`
  - 已有 robust mean/CVaR/variance 聚合、state sequence CE、AE decoded sequence CE、anchor-weighted consistency、contact/distance/clash/interface loss。
  - loss 形式科学上比单状态平均更合理，但当前权重和训练条件仍不能解决 sampled latent 退化。
- `scripts/strategy01/stage09_guided_state_specific_sampling.py`
  - 仍是目前几何最稳的采样入口，`bb_ca_flow=0` fallback 能保持 source-pose geometry。
  - learned gate/safe accept 可运行，但未证明优于 fallback。
- `scripts/strategy01/stage11_flow_sequence_training.py`
  - Stage11G 新增 `load_sampled_latent_cache`、`inject_sampled_latent_cache`、`--sampled-cache-dir`、`--cache-only-samples`。
  - 这是诊断 sampled latent 分布的工具，不应作为最终训练默认策略。

## 3. 已确认的代码问题和科学偏差

### 3.1 第一版 sampled-cache 训练没有真正命中 cache

现象：`stage11g_sampled_cache_48pilot600_results.json` 显示 loaded cache entries = 48，但训练阶段 `matched=0`。

根因：代码先按 `--max-train-samples` 截断 train samples，再去 batch 内匹配 cache；被截断到前 48 个样本时，未必包含 sampling cache 的 sample ids。

修正：新增 `--cache-only-samples`，先按 cache ids 过滤 train/val，再做 max sample 截断。

修后证据：`stage11g_sampled_cache_filtered_42pilot600_results.json` 中 cache 信息为 `{"dir": "results/strategy01/stage11e_sampling48/fallback", "entries": 48, "t": 0.75, "cache_only_samples": true}`，训练阶段命中：

- overfit1: `{'enabled': True, 'matched': 1, 'total': 1, 'cache_t': 0.75}`
- overfit4: `{'enabled': True, 'matched': 2, 'total': 2, 'cache_t': 0.75}`
- mini fixed eval: `{'enabled': True, 'matched': 2, 'total': 2, 'cache_t': 0.75}`

### 3.2 过滤后 sampled-cache 训练 loss 会下降，但最终 sequence 更差

关键结果：

- 过滤后 overfit1 loss drop：`0.8230740076054714`
- 过滤后 overfit4 loss drop：`0.177759148224231`
- 过滤后 mini loss drop：`0.2884410135965614`
- 但 48-sample 采样 sequence identity：
  - Stage11E fallback shared: `0.44141384611828893`
  - Stage11G filtered shared: `0.218345021626033`
  - Stage11G filtered AE: `0.22830512504840098`

解释：这说明训练 loss 下降不等于科学目标改善。当前 cache-as-`x_t` 训练把 sampled latent 放进常规 FM 框架后，结构/latent FM loss 变成极大主导项，模型在拟合一个过大的 latent 偏移，而不是学会“如何把 sampled latent 修到能解码共享序列的区域”。这会损害 final shared sequence。

### 3.3 exact latent 与 sampled latent 的差距已经定量确认

`stage11g_exact_vs_sampled_ae_decode_summary.json` 显示：

- exact AE latent decode sequence identity mean: `0.9504862428842504`
- sampled AE latent decode sequence identity mean: `0.44141384611828893`
- exact min: `0.06451612903225806`
- sampled min: `0.06666666666666667`

这直接定位瓶颈：**AE decoder 和 exact label 基本可用，问题发生在 sampled `local_latents_states` 的生成分布**。

几个典型样本说明 sampled latent 会产生低复杂度序列倾向，例如 Pro/Ser/Gly 过多：

```json
[
  {
    "sample_id": "1a0n_B_A",
    "ref_seq": "PPRPLPVAPGSSKT",
    "exact_seq": "PPRPLPVAPGSSKT",
    "exact_id": 1.0,
    "sampled_seq": "PPPPVPPPPGSSPT",
    "sampled_id": 0.6428571428571429,
    "pred_seq": "PPPPVPPPPGSSPT"
  },
  {
    "sample_id": "1ai0_L_K",
    "ref_seq": "GIVEQCCTSICSLYQLENYCN",
    "exact_seq": "GIVEQCCTSICSLYQLENYCN",
    "exact_id": 1.0,
    "sampled_seq": "GLVEQCCTSICSLYQLENYCN",
    "sampled_id": 0.9523809523809523,
    "pred_seq": "GLVEQCCTSICSLYQLENYCN"
  },
  {
    "sample_id": "1aze_A_B",
    "ref_seq": "VPPPVPPRRR",
    "exact_seq": "VPPPVPPRRR",
    "exact_id": 1.0,
    "sampled_seq": "PPVPVPPVPP",
    "sampled_id": 0.5,
    "pred_seq": "PPVPVPPVPP"
  },
  {
    "sample_id": "1bxp_A_B",
    "ref_seq": "MRYYESSLKSYPD",
    "exact_seq": "MRYYESSLKSYPD",
    "exact_id": 1.0,
    "sampled_seq": "GSPSSSSNIPRPY",
    "sampled_id": 0.23076923076923078,
    "pred_seq": "GSPSSSSNIPRPY"
  },
  {
    "sample_id": "1cee_A_B",
    "ref_seq": "KKKISKADIGAPSGFKHVSHVGWDPQNGFDVNNLDPDLRSLFSRAGISEAQLTDAETSK",
    "exact_seq": "KKKISKADIGAPSGFKHVSHVGWDPQNGFDVNNLDPDLRSLFSRAGISEAQLTDAETSK",
    "exact_id": 1.0,
    "sampled_seq": "QITSSPPPSKPPISIPSSSPSSNNPSSSSPSSSSLSSLSPLLLSLLSSNPSLGLNPNLS",
    "sampled_id": 0.13559322033898305,
    "pred_seq": "QITSSPPPSKPPISIPSSSPSSNNPSSSSPSSSSLSSLSPLLLSLLSSNPSLGLNPNLS"
  }
]
```

## 4. 当前代码是否偏离科学目标

结论：**部分符合，但没有完全按科学目标走通。**

符合目标的部分：

- 多状态输入、state-specific outputs、shared sequence head、state sequence consensus 都在代码中真实存在。
- `bb_ca_states/local_latents_states` 没有只剩平均坐标；legacy `bb_ca/local_latents` 平均输出只用于兼容旧接口。
- loss 已经使用 worst/CVaR 风格，不只是平均状态。
- 几何采样已经有 source-anchor、safe accept、clash hard-fail 思路。

偏离或不足的部分：

- 训练条件和采样条件仍不一致：训练主要面对 synthetic corruption / exact latent 附近样本，而最终采样得到的是模型自身退化的 sampled latent。
- `seq_logits_shared` 与 `AE decoded local_latents` 已建立联系，但这种联系在 sampled latent 区域太弱；exact latent 区域表现好不能代表采样时也好。
- Naive sampled-cache flow matching 会把问题变成“拟合巨大 latent 偏移”，不是“修复 sequence-informative latent manifold”。
- 当前 optional CA/residue feature 被禁用是为了避免标签泄漏，但也可能限制绝对质量；后续必须用非泄漏的 predicted/init pose features，而不是用真实 binder residue type。
- learned flow gate 还没有超过 fallback；不能把它设为默认科学结论。

## 5. 警告和报错影响评估

已扫描 Stage11G 相关 `.out/.err`：

- `CCD_MIRROR_PATH/PDB_MIRROR_PATH not set`：本阶段没有调用需要镜像数据库的结构下载或解析功能，不影响 Stage11G 训练/采样结论；如果后续在线解析 mmCIF/PDB，应显式设置。
- PyTorch nested tensor warning：性能警告，不改变数值结果。
- `OptionalCaCoorsNanometersSeqFeat/OptionalResidueTypeSeqFeat` 返回 zeros：不是 runtime error，但有科学影响。当前禁用 residue type 是必要的 no-leak 选择；CA feature 当前也禁用，因为以前实测 noisy/generated CA feature 会损害 contact。后续如果重新启用，必须只用 non-oracle init/pred pose，并单独做 no-leak probe。
- 未发现 Traceback、OOM、NaN、failed job。

## 6. 下一步建议：Stage11H latent sequence repair，而不是继续 naive cache-FM

### 6.1 核心思想

不再把 sampled local latent 当普通 `x_t` 去做完整 FM。改成在 sampled latent 附近训练一个 **bounded latent correction / sequence repair**：

- 输入：sampled `local_latents_states`、init/source `bb_ca_states`、ensemble target memory、state role/weight。
- 输出：`delta_local_latents_states` 或 corrected `local_latents_states_repaired`。
- 主监督：冻结 AE decoder 的 shared sequence CE / state AE sequence CE。
- 辅监督：小幅 latent delta L2、anchor residue trust-region、interface contact 不退化。
- 几何：默认继续 `bb_ca_flow=0`，避免为修序列破坏已较好的 pose。

### 6.2 为什么这比 Stage11G 更科学

- Stage11G 证明 sampled latent 区域错得很远，完整 FM loss 会被结构项压倒。
- 真实科学目标需要的是“同一 shared sequence 与多个 state-specific complexes 兼容”，不是把 sampled latent 强行回归 exact latent 的所有维度。
- 因此应只修复与序列可解码性和界面锚点相关的 latent 子空间，并限制修正幅度。

### 6.3 Stage11H 可执行验收

- exact-vs-sampled-vs-repaired AE decode probe：repaired identity 必须显著高于 sampled，目标 48-sample mean 从 `0.441` 提到 `>=0.55`，且不损害 geometry RMSD。
- no-leak probe：`binder_seq_shared` 只进入 loss，不进入 repair 输入。
- trust-region probe：anchor residues 的 latent/geometry 更新小于配置阈值。
- 48-sample exact benchmark：shared sequence identity 提升，同时 contact-F1/clash/RMSD 不比 Stage10C/Stage11E fallback 差。

## 7. 当前是否应继续扩大数据或训练

不应直接扩大。当前瓶颈不是样本数，而是 sampled latent 分布与 sequence manifold 错位。扩大训练会放大这个目标错配。正确顺序是：

1. 先做 Stage11H latent repair probe。
2. 如果 repaired latent 确认能改善 sampled AE sequence identity，再把它并入主 multistate sampler。
3. 只有当 48-sample exact benchmark 看到 sequence 和 geometry 同时改善，再扩大到更多 curated data。

## 8. 复现路径

关键结果文件：

- `reports/strategy01/probes/stage11g_sampled_cache_48pilot600_results.json`
- `reports/strategy01/probes/stage11g_sampled_cache_filtered_42pilot600_results.json`
- `reports/strategy01/probes/stage11g_sampling48_comparison_summary.json`
- `reports/strategy01/probes/stage11g_sampling48_filtered_comparison_summary.json`
- `reports/strategy01/probes/stage11g_exact_vs_sampled_ae_decode_summary.json`

关键代码：

- `scripts/strategy01/stage11_flow_sequence_training.py`
- `src/proteinfoundation/nn/local_latents_transformer_multistate.py`
- `src/proteinfoundation/flow_matching/multistate_loss.py`

## 9. 总结

当前代码没有单纯的运行错误，但有一个真正影响科学目标的设计问题：**shared sequence 的监督没有有效塑造最终采样得到的 local latent manifold**。这就是为什么 exact latent decode 很好，而 sampled latent 和最终 shared sequence 仍不好。

Stage11G 的价值是把这个问题实证定位清楚；它本身不应作为默认训练方案。下一步必须转向 bounded latent repair / sequence-informative latent correction。
