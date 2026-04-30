# Strategy01 Stage09B：sampling-time interface guidance 接入报告

## 1. 阶段目标
Stage09 第一部分只证明了 post-hoc oracle anchor rigid guidance 能改善已有 B1 pose，但它不在生成过程中发挥作用。本阶段把 `anchor_contact + clash_repulsion + interface_shell + safe-accept` 前移到真正的 state-specific flow sampling 循环中，直接更新 `bb_ca_states`。

## 2. Stage09 第一部分结果分析
第一部分 post-hoc 诊断：contact-F1 `0.1171 -> 0.1570`，direct RMSD `26.2498 -> 24.6183 A`，clash_rate `0.7847 -> 0.7639`。
这个结果说明界面 anchor 和 clash/shell 约束方向是有效的，但 post-hoc 刚体修正不能代表模型真实生成能力。因此本阶段必须在 sampling loop 内更新状态轨迹。

## 3. 代码改动
- `scripts/strategy01/stage09_guided_state_specific_sampling.py`：新增真实 sampling-time guided sampler。它保留 Stage07 的多状态 `bb_ca_states/local_latents_states` flow 更新方式，在每个 step 的 `bb_ca` 更新后额外计算 guidance energy 并对 `bb_ca_states[k]` 做梯度步。
- `scripts/strategy01/stage09_guided_sampling.sbatch`：单卡 smoke 入口。
- `scripts/strategy01/stage09_guided_pair.sbatch`：同 seed 下的 unguided/guided paired comparison。
- `scripts/strategy01/stage09_guided_sweep2.sbatch`：更强 late-stage guidance 参数扫。
- `scripts/strategy01/stage09_guided_full.sbatch`：全量 V_exact guided sampling 与 exact benchmark。

## 4. Guidance 实现方式
每个 state 从 exact complex 中抽取 target CA 和 binder CA，按 10 A contact cutoff 选 anchor pairs，并转换到 nm 单位进入 sampling。当前仍是 oracle anchor 诊断，不能作为公平 leaderboard；后续生产版必须替换成 persistent hotspot/source-interface anchors。

每个 sampling step 后对 `bb_ca_states` 加如下能量：

```text
E = L_anchor_contact + w_clash * L_clash_repulsion + w_shell * L_interface_shell + w_tether * L_tether
```

- `L_anchor_contact`：生成 binder 的 anchor residue 到 target anchor residue 的距离靠近参考接触距离。
- `L_clash_repulsion`：target-binder 距离小于 2.8 A 时二次惩罚。
- `L_interface_shell`：binder residue 离 target 表面过远时惩罚，防止漂走。
- `L_tether`：约束 guidance 不要一次把 flow step 的输出拉坏。
- `safe-accept`：如果 guidance 造成新 severe clash、已有 clash 更近、或 contact-F1 下降，则拒绝该 state 的 guidance 更新。

## 5. 参数扫
第一版 weak guidance 跑通但效果偏弱，因此加入 seed 控制并做同 seed paired comparison。

| 配置 | contact-F1 mean | RMSD mean A | clash_rate |
| --- | ---: | ---: | ---: |
| stage09_guided_state_sampling_control_seed123_exact_benchmark.json | 0.1032 | 25.6410 | 0.7500 |
| stage09_guided_state_sampling_guided_seed123_s065_lr005_i3_exact_benchmark.json | 0.1058 | 25.4191 | 0.7500 |
| stage09_guided_state_sampling_guided_seed123_s070_lr012_i8_a64_exact_benchmark.json | 0.1239 | 24.8635 | 0.7500 |
| stage09_guided_state_sampling_guided_seed123_s075_lr008_i5_exact_benchmark.json | 0.1112 | 25.2738 | 0.7500 |
| stage09_guided_state_sampling_guided_seed123_s075_lr015_i8_a32_exact_benchmark.json | 0.1360 | 24.8412 | 0.6667 |
| stage09_guided_state_sampling_guided_seed123_s085_lr020_i10_a32_exact_benchmark.json | 0.1389 | 24.6815 | 0.6667 |

最优 smoke 配置为 `start=0.85, lr=0.20, inner_steps=10, max_anchor_pairs=32, clash_weight=80`。直观解释：早期 flow 还在定全局形状，过早 guidance 会被后续模型更新覆盖；后期聚焦少数核心 anchors 更能改变最终 interface。

## 6. 全量 V_exact 结果
| 指标 | Stage08B unguided B1 | Stage09B sampling-guided B1 | 变化 |
| --- | ---: | ---: | ---: |
| contact-F1 mean | 0.1171 | 0.1622 | 0.0451 |
| direct binder CA RMSD mean A | 26.2498 | 24.6331 | -1.6167 |
| clash_rate | 0.7847 | 0.7431 | -0.0417 |

全量运行耗时 `113.4968 s`，平均 `2.3645 s/sample`，1 张 A100。

## 7. 结果判断
本阶段达成了“接入真正 state-specific sampling”的目标，并且三个核心指标同向改善：contact-F1 增加、RMSD 降低、clash_rate 降低。说明 guidance 不只是后处理有效，放进 flow sampling 后也能改善生成轨迹。

但结果仍未达到科学成功标准：contact-F1 只有 0.162，clash_rate 仍高达 0.743。主要原因是当前 guidance 使用 oracle exact anchors 做诊断，且只作用在 CA 层；模型本身还没有学到足够强的界面放置先验，B0 公平基线也还没有跑通。

## 8. 下一步
1. 把 oracle exact anchors 替换为 production 可用的 persistent anchors：来自 hotspot、source complex interface、constrained hybrid label 或 target ensemble 中稳定 pocket/interface residues。
2. 将 guidance 从脚本级实验迁入正式 Strategy01 sampler 配置，支持 `guidance.anchor_source=oracle|hotspot|source_interface|predicted_persistent`。
3. 继续补 B0-native/B0-same-refiner artifact；没有公平 B0 前不能声称 B1 超过 baseline。
4. 训练侧提高 interface/contact/clash loss 权重，让模型内化 guidance，而不是完全依赖 sampling-time 外力。

## 9. 复现命令
```bash
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
sbatch scripts/strategy01/stage09_guided_pair.sbatch
sbatch scripts/strategy01/stage09_guided_sweep2.sbatch
sbatch scripts/strategy01/stage09_guided_full.sbatch
```
