#!/usr/bin/env python
"""Append the completed V_exact tensor/sampling/full benchmark section."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:  # noqa: BLE001
        return str(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPO / "reports/strategy01/strategy01_stage08b_high_quality_refinetune_report.md")
    args = parser.parse_args()
    build = load_json(REPO / "reports/strategy01/probes/stage08b_vexact_tensor_build_summary.json", {})
    ae = load_json(REPO / "reports/strategy01/probes/stage08b_vexact_ae_latent_summary.json", {})
    ckpt = load_json(REPO / "reports/strategy01/probes/stage08b_mini_lightning_ckpt_conversion_summary.json", {})
    sampling = load_json(REPO / "reports/strategy01/probes/stage08b_vexact_sampling_summary.json", {})
    full = load_json(REPO / "reports/strategy01/probes/stage08b_full_exact_benchmark_summary.json", {})
    b1 = full.get("B1_strategy01") or {}
    b0 = full.get("B0_baseline") or {}
    b2 = ((full.get("B2_exact_reference") or {}).get("b2_exact_reference") or {})
    section = f"""

## 9. Stage08B 未完成项补完：V_exact tensor、B1 exact sampling 与 full benchmark

在上一版报告之后，本阶段继续补完了原先未完成的 exact benchmark 链路。

### 9.1 V_exact tensor dataset

- 构建脚本：`scripts/strategy01/stage08b_build_vexact_tensor_dataset.py`
- 输入 manifest：`data/strategy01/stage08_high_quality_dataset/V_exact_main_manifest.json`
- 输出 tensor：`{build.get('tensor_dataset')}`
- 样本数：`{build.get('sample_count')}`
- reject 数：`{build.get('reject_count')}`
- external validation families：`{build.get('families_val')}`

这一步把原来只能用于审计的 `V_exact` manifest 转成了 Strategy01 sampler 能直接读取的 tensor dataset。每条样本包含 target 多状态结构、共享 binder 序列、K 个 exact complex label、binder CA、interface contact/distance labels 和 state mask。所有样本都作为 external validation，不混入训练。

### 9.2 V_exact AE latent

- 脚本：`scripts/strategy01/stage05_extract_ae_latents.py`
- 输出：`{ae.get('output')}`
- 样本数：`{ae.get('num_samples_total')}`
- processed states：`{ae.get('processed_states')}`
- AE checkpoint：`{ae.get('ae_ckpt')}`

结论：`V_exact` 的 local latents 已全部替换为 Complexa AE `encoder.mean [K,Nb,8]`，没有保留 geometry proxy 作为主评估输入。

### 9.3 Stage08B checkpoint 转换与 exact sampling

- 转换脚本：`scripts/strategy01/stage07_convert_mini_to_lightning_ckpt.py`
- 输出 checkpoint：`{ckpt.get('output')}`
- source phase/steps：`{ckpt.get('source_phase')}/{ckpt.get('source_steps')}`
- exact sampling 脚本：`scripts/strategy01/stage08b_multistate_exact_sampling.py`
- exact sampling 样本数：`{sampling.get('sample_count')}`
- nsteps：`{sampling.get('nsteps')}`
- 总耗时：`{fmt(sampling.get('elapsed_sec'))} sec`
- 平均耗时：`{fmt(sampling.get('mean_sec_per_sample'))} sec/sample`

这一步完成了 `V_exact` 上的 B1 Strategy01 多状态生成：一个 shared sequence 加每个 state 的 state-specific binder CA pose。输出目录为 `{sampling.get('out_dir')}`。

### 9.4 Full exact benchmark 结果

运行脚本：`scripts/strategy01/stage08b_full_exact_benchmark.py`

输出：`reports/strategy01/probes/stage08b_full_exact_benchmark_summary.json`

| 项 | 状态 | 说明 |
|---|---|---|
| B0 baseline Complexa | `{b0.get('status')}` | `{b0.get('reason')}` |
| B1 Strategy01 | `{b1.get('status')}` | V_exact tensor 上 48 条 exact samples 的 Strategy01 生成结果 |
| B2 exact reference | `passed` | 实验 exact complex geometry 上限 |

B1 exact 关键指标：

| 指标 | 结果 |
|---|---|
| sample_count | `{b1.get('sample_count')}` |
| state_count_ok | `{b1.get('state_count_ok')}` |
| contact-F1 | `{b1.get('contact_f1')}` |
| direct/worst interface RMSD Å | `{b1.get('worst_interface_rmsd_A')}` |
| aligned binder CA RMSD Å | `{b1.get('aligned_binder_ca_rmsd_A')}` |
| generated contact persistence | `{b1.get('contact_persistence')}` |
| state metric std contact-F1 | `{b1.get('state_metric_std_contact_f1')}` |
| clash rate | `{b1.get('clash_rate')}` |

B2 exact reference 关键指标：

| 指标 | 结果 |
|---|---|
| exact contact count | `{b2.get('contact_count')}` |
| exact contact persistence | `{b2.get('contact_persistence')}` |
| target motion RMSD Å | `{b2.get('target_motion_rmsd_A')}` |

### 9.5 科学判断

这个补完步骤给出了明确的负结果：当前 Stage08B pilot 能在训练/内部验证 loss 上下降，但在真正 `V_exact` exact geometry 上泛化很弱。B1 的 mean contact-F1 只有约 `{fmt((b1.get('contact_f1') or {}).get('mean'))}`，clash rate 约 `{fmt(b1.get('clash_rate'))}`。这说明现有 119/30 pilot 数据和 loss 还不足以学到真实多状态 interface geometry。

因此不能进入 1000 样本大训练。下一阶段应先解决三件事：

1. 补原始 Complexa B0 artifact：原模型生成结构后需要接入 ProteinMPNN/sequence_hallucination 或原论文 pipeline 的序列确定步骤，否则不能和 Strategy01 的 shared sequence head 做公平比较。
2. 提升 exact/hybrid 训练数据质量：把 `V_exact` 的一部分或同源 exact/hybrid 数据转成训练集，同时保持 family holdout，避免只用 predictor-derived bronze。
3. 修正采样约束：当前 B1 pose clash 高，说明只靠 flow trajectory 还不够，需要把 exact interface anchors/contact constraints 或 clash-aware guidance 前移到 sampling/reward，而不是只在训练 loss 里弱监督。
"""
    marker = "\n## 9. Stage08B 未完成项补完："
    text = args.report.read_text(encoding="utf-8")
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    args.report.write_text(text.rstrip() + section + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "report": str(args.report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
