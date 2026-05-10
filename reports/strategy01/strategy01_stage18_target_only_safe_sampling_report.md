# Strategy01 Stage18 target-only safe sampling 报告

## 阶段目标

Stage17 证明多 seed 采样能带来小幅 sequence improvement，但也暴露了核心物理问题：很多 target-only 生成候选虽然有 target contact，却存在 severe clash。对 binder 设计来说，硬碰撞意味着复合物不物理，不能因为 sequence identity 或 contact 数高就接受。因此 Stage18 的目标是：在不泄漏真实 binder 序列、source pose、exact contact 的前提下，把 **severe clash hard gate + bounded rigid-body clash relief + target-only safe selection** 接到采样端。

这仍然服务于最终科学目标：只输入多状态 target，输出一个共享 binder 序列和每个 state 合理的复合物几何。

## 代码改动

- `scripts/strategy01/stage12c_de_novo_smoke.py`
  - 新增 `stage18_bounded_rigid_clash_relief()`。
  - 该函数只使用 target CA、target mask、生成的 binder CA 和 state mask，不使用真实 binder pose、真实序列或 exact interface。
  - 对存在 severe target-binder CA clash 的 state，按 clashing pair 的平均方向对整个 binder 做小幅刚体平移。
  - 接受条件：最小 target-binder 距离改善，并且保留最小接触数量，避免 Stage09D 那种“排斥强但把 binder 推飞”的失败模式。
  - 新增 CLI：`--stage18-enable-clash-relief`、`--stage18-clash-min-distance-nm`、`--stage18-relief-step-nm`、`--stage18-relief-iters`、`--stage18-min-contact-count`。
- `scripts/strategy01/stage17_per_sample_multiseed_probe.py`
  - 增加 Stage18 relief 参数透传。
  - 候选行记录 `stage18_clash_relief` 统计。
  - selection 从软惩罚改成 hard gate：若某个样本存在 no-clash 候选，只允许在 no-clash 候选内排序；如果所有候选都有 clash，该样本标记为 unsafe。
  - summary 增加 `safe_candidate_available_rate` 和 `selected_safe_rate`。
- `scripts/strategy01/slurm/stage18_geo_relief_persample*.sbatch`
  - 新增单 A100 smoke 脚本，继续使用 `new/gu02`，并保留 runtime cache 设置。

## 测试结果

### CPU probe

`stage18_cpu_relief_probe.json` 使用 CPU、1 sample、2 seeds、2 steps 验证接口。结果通过；该低步数 probe 中候选远离 target，没有触发 relief。临时脚本第一次多了一个 `PY` 尾标导致 shell exit 1，但模型 probe 已完成；已修正后重跑并返回 exit 0。

### val12 smoke

对 Stage15 checkpoint，val12、8 seeds、n16 运行 Stage18 relief：

- summary：`{'first_seed_identity_mean': 0.28237204626202583, 'proxy_selected_identity_mean': 0.28237204626202583, 'oracle_identity_mean': 0.2861599251627922, 'proxy_matches_oracle_rate': 0.25}`
- selected stats：`{'target_severe_clash_rate': {'min': 0.0, 'mean': 0.0, 'max': 0.0}, 'target_contact_count_mean': {'min': 5.5, 'mean': 65.56944444444444, 'max': 107.0}, 'target_hotspot_contact_count_mean': {'min': 0.0, 'mean': 19.98611111111111, 'max': 42.5}, 'target_min_distance_nm_mean': {'min': 0.2905392348766327, 'mean': 0.34836941626336837, 'max': 0.6356346309185028}, 'shared_identity_mean_posthoc': {'min': 0.2142857164144516, 'mean': 0.28237204626202583, 'max': 0.4117647111415863}, 'no_leak_score': {'min': 0.013231997421826236, 'mean': 0.5633296388684054, 'max': 2.9495486579835415}, 'relief_attempts': {'mean': 23.083333333333332, 'max': 37.0}, 'relief_accepted': {'mean': 20.0, 'max': 31.0}, 'relief_accept_rate': {'mean': 0.8938141804989631, 'max': 1.0}, 'relief_min_before_mean': {'mean': 0.1679054173031716, 'max': 0.23041652888059616}, 'relief_min_after_mean': {'mean': 0.23379589158049188, 'max': 0.28145869448781013}, 'relief_contacts_before_mean': {'mean': 153.96689243238157, 'max': 227.47619047619048}, 'relief_contacts_after_mean': {'mean': 153.2082176791416, 'max': 228.33333333333334}}`

与 Stage17 geometry proxy 有效结果相比：

- Stage17 geometry selected severe clash mean：`0.3888888888888889`
- Stage18 selected severe clash mean：`0.0`
- Stage17 selected identity mean：`0.2747962884604931`
- Stage18 selected identity mean：`0.28237204626202583`

解释：Stage18 同时把 selected clash 降到 0，并把 selected identity 恢复到 entropy/disagreement proxy 水平，说明 bounded rigid relief 比单纯几何惩罚更合适。

### val36 / 当前完整 val split

`--max-samples 48` 实际仍只跑 36 条，因为当前 dataset 的 `val` split 只有 36 条；因此 val36 和 val48 输出等价。

- summary：`{'first_seed_identity_mean': 0.20645633213118547, 'proxy_selected_identity_mean': 0.21030433607908586, 'oracle_identity_mean': 0.22371548222791818, 'proxy_matches_oracle_rate': 0.16666666666666666, 'safe_candidate_available_rate': 0.9722222222222222, 'selected_safe_rate': 0.9722222222222222}`
- selected stats：`{'target_severe_clash_rate': {'min': 0.0, 'mean': 0.013888888888888888, 'max': 0.5}, 'target_contact_count_mean': {'min': 5.5, 'mean': 59.875, 'max': 129.66666666666666}, 'target_hotspot_contact_count_mean': {'min': 0.0, 'mean': 17.11111111111111, 'max': 91.66666666666667}, 'target_min_distance_nm_mean': {'min': 0.2654961496591568, 'mean': 0.5041066048045953, 'max': 5.10989511013031}, 'shared_identity_mean_posthoc': {'min': 0.0, 'mean': 0.21030433607908586, 'max': 0.4444444477558136}, 'no_leak_score': {'min': 0.013231997421826236, 'mean': 0.875913854483197, 'max': 5.015670042019337}, 'relief_attempts': {'mean': 21.38888888888889, 'max': 39.0}, 'relief_accepted': {'mean': 18.97222222222222, 'max': 33.0}, 'relief_accept_rate': {'mean': 0.9088076946470667, 'max': 1.0}, 'relief_min_before_mean': {'mean': 0.17214026119760278, 'max': 0.2458499769369761}, 'relief_min_after_mean': {'mean': 0.24565730719447476, 'max': 0.3066399196783702}, 'relief_contacts_before_mean': {'mean': 123.7510332400429, 'max': 397.8974358974359}, 'relief_contacts_after_mean': {'mean': 122.57959398841041, 'max': 396.6666666666667}}`
- unsafe samples：`[{'idx': 13, 'sample_id': '5ixf_A_B__v01__k2__hpred', 'target_id': '5ixf', 'clash': 0.5}]`

结论：36 条中 35 条存在 no-clash 候选并被选中，`selected_safe_rate=0.9722`。唯一失败样本是 `5ixf_A_B__v01__k2__hpred`，8 个 seed 全部仍有 `0.5` severe clash rate；这个样本不能算 Stage18 成功，应进入后续失败分析或更强 target-shell/pose initialization 研究。

## warning / error 审查

- optional CA/residue feature warning：仍是 target-only no-leak 设计结果，不影响科学结论。
- `CCD_MIRROR_PATH/PDB_MIRROR_PATH` 未设置：本阶段只使用已张量化数据，不调用本地 mirror 解析，不影响结果。
- `slurm_load_jobs error: Invalid job id specified` 出现在本地检查命令里，是因为查询时 job 已结束，不是作业失败。
- 未发现 Traceback、OOM、NaN。

## 科学结论

Stage18 没有解决全部问题，但修复了一个关键科学漏洞：候选选择不再把 severe clash 当作可被高 contact 或高 identity 抵消的普通分数项。现在 Stage18 证明：在当前模型已经能产生的一批 candidates 中，多数样本可以通过 label-free bounded relief 找到 no-clash 候选。

这说明下一步不应该只增加训练步数，而应该把 **target-only 物理可行性** 纳入完整 B1 评估：先 hard fail severe clash，再看 shared sequence consistency、hotspot/contact shell 和 state consistency。

## 下一步

1. 对 unsafe 样本 `5ixf` 做单独失败分析：是 target 构象过近、binder 长度/pose 不适合、还是 relief 步长/次数不足。
2. 把 Stage18 safe-select 接入正式 B1 benchmark，输出 no-clash pass rate、contact/hotspot proxy、sequence identity 后验诊断。
3. 若 unsafe 样本比例在更大 exact/hybrid 集上仍高，则回到训练层加强 bb_ca flow 的 target-relative placement 和 clash/contact loss，而不是继续调 sequence head。
4. 若 safe-select 后 exact geometry 指标仍差，则需要引入 target-only contact-shell guidance 或训练一个 lightweight candidate ranker，但不能用 exact 标签做生产选择。

## 复现命令

```bash
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python -m py_compile   scripts/strategy01/stage12c_de_novo_smoke.py   scripts/strategy01/stage17_per_sample_multiseed_probe.py
sbatch scripts/strategy01/slurm/stage18_geo_relief_persample.sbatch
sbatch scripts/strategy01/slurm/stage18_geo_relief_persample36.sbatch
```
