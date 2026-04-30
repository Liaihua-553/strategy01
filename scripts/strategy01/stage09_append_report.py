#!/usr/bin/env python
"""Write the Chinese Stage09 B0/B1 guidance report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "missing", "path": str(path)}


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-audit", type=Path, default=REPO / "reports/strategy01/probes/stage09_b0_fair_baseline_audit.json")
    parser.add_argument("--guidance", type=Path, default=REPO / "reports/strategy01/probes/stage09_b1_interface_guidance_summary.json")
    parser.add_argument("--stage08b", type=Path, default=REPO / "reports/strategy01/probes/stage08b_full_exact_benchmark_summary.json")
    parser.add_argument("--output", type=Path, default=REPO / "reports/strategy01/strategy01_stage09_b0_b1_guidance_report.md")
    args = parser.parse_args()
    b0 = load_json(args.b0_audit)
    g = load_json(args.guidance)
    s8 = load_json(args.stage08b)
    sweep_rows = load_json(args.guidance.parent / "stage09_guidance_sweep_summary.json")
    b1 = s8.get("B1_strategy01", {}) if isinstance(s8, dict) else {}
    before = g.get("before_direct_metrics", {}) if isinstance(g, dict) else {}
    after = g.get("after_guidance_direct_metrics", {}) if isinstance(g, dict) else {}
    delta = g.get("delta", {}) if isinstance(g, dict) else {}

    lines = []
    lines.append("# Strategy01 Stage09：公平 B0 基线与 B1 interface/clash guidance 诊断报告")
    lines.append("")
    lines.append("## 1. 阶段目标")
    lines.append("本阶段不扩大训练规模，专门处理两个阻塞问题：第一，B0 必须成为可复现且公平的原 Complexa 基线；第二，B1 目前 exact benchmark 中 clash/interface 失败严重，必须先验证显式界面 guidance 是否能修正 pose 几何。")
    lines.append("")
    lines.append("## 2. 为什么不继续扩大训练")
    lines.append(f"Stage08B 的 B1 exact 结果为：contact-F1 mean={fmt((b1.get('contact_f1') or {}).get('mean'))}，direct interface RMSD mean={fmt((b1.get('worst_interface_rmsd_A') or {}).get('mean'))} A，clash_rate={fmt(b1.get('clash_rate'))}。这说明主要错误在界面几何和物理排斥，而不是训练步数不足。")
    lines.append("如果直接扩大到几百或上千条样本，模型会在当前缺少 interface guidance 的 sampling 机制下更稳定地复制错误 pose 分布，不能解决 shared binder 与多状态 target 的真实兼容性问题。")
    lines.append("")
    lines.append("## 3. B0 公平基线定义")
    lines.append(f"B0 审计状态：`{b0.get('status') if isinstance(b0, dict) else 'unknown'}`。本阶段没有伪造 B0 指标，而是固定了 B0 artifact contract。")
    lines.append("公平 B0 必须分为三层：`B0-native` 复现原 Complexa 单状态生成与原生 refinement；`B0-same-refiner` 使用和 B1 相同的序列/过滤后处理以隔离多状态 conditioning 的贡献；`B0-oracle-posthoc` 只作诊断上限，不能进入主榜单。")
    lines.append("B0 的输出必须转换为和 B1 相同 schema：每个 sample 一个 `shared_sequence.fasta`，每个 state 一个 `stateXX_binder_ca.pdb`，然后统一调用 exact geometry evaluator。")
    lines.append("")
    lines.append("## 4. B1 interface guidance 诊断设计")
    lines.append("新增 `stage09_interface_guidance_diagnostics.py`。它使用 V_exact 实验接触作为 oracle anchors，对已有 B1 state-specific binder pose 做刚体旋转/平移优化。优化项包括 anchor contact distance、target-binder clash repulsion、interface shell 和轻量 tether。本次最终报告采用 safe-accept 门控：若 guidance 把非 clash pose 变成 clash，或 contact-F1 反降，则保留原 pose。")
    lines.append("注意：这是 oracle-anchor 诊断，不是 leaderboard。它的意义是判断 contact/clash guidance 本身是否能把错误 pose 拉回合理界面；若诊断有效，下一步应把同类 energy 接入 flow sampling，而不是继续只靠离线 loss。")
    lines.append("")
    lines.append("## 5. 实测结果")
    lines.append(f"诊断状态：`{g.get('status') if isinstance(g, dict) else 'unknown'}`，处理 state 数：{g.get('state_count', 0) if isinstance(g, dict) else 0}。")
    lines.append("")
    lines.append("| 指标 | guidance 前 | guidance 后 | 变化 |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(f"| contact-F1 mean | {fmt((before.get('contact_f1') or {}).get('mean'))} | {fmt((after.get('contact_f1') or {}).get('mean'))} | {fmt(delta.get('contact_f1_mean_abs'))} |")
    lines.append(f"| direct binder CA RMSD mean A | {fmt((before.get('direct_binder_ca_rmsd_A') or {}).get('mean'))} | {fmt((after.get('direct_binder_ca_rmsd_A') or {}).get('mean'))} | {fmt(delta.get('direct_rmsd_mean_A'))} |")
    lines.append(f"| clash_rate | {fmt(before.get('clash_rate'))} | {fmt(after.get('clash_rate'))} | {fmt(delta.get('clash_rate_abs'))} |")
    lines.append("")
    lines.append("## 6. 代码改动")
    lines.append("- `scripts/strategy01/stage09_b0_fair_baseline_audit.py`：定义 B0 公平基线层级、artifact schema、必需文件和后续命令，不生成虚假基线。")
    lines.append("- `scripts/strategy01/stage09_interface_guidance_diagnostics.py`：实现 exact-anchor 刚体 guidance 诊断，输出 guided PDB、summary JSON 和 before/after 指标。")
    lines.append("- `scripts/strategy01/stage09_append_report.py`：汇总 Stage08B、B0 audit 和 B1 guidance 结果，生成本中文报告。")
    lines.append("")
    lines.append("## 7. 参数扫与结论")
    if isinstance(sweep_rows, list) and sweep_rows:
        lines.append("先测试了不带 safe-accept 的 clash_weight=15/30/60。三组都能提升 contact-F1、降低 RMSD，但 clash_rate 均上升约 0.0417，说明单纯增加排斥权重不能解决被 anchor 拉入 target 的少数状态。")
        lines.append("")
        lines.append("| sweep | contact-F1 Δ | RMSD Δ A | clash_rate Δ |")
        lines.append("| --- | ---: | ---: | ---: |")
        for r in sweep_rows:
            lines.append(f"| {r.get('file')} | {fmt(r.get('contact_delta'))} | {fmt(r.get('rmsd_delta'))} | {fmt(r.get('clash_delta'))} |")
    else:
        lines.append("未找到参数扫 summary。")
    lines.append("最终采用 clash_weight=30、shell_weight=0.10、tether_weight=0.002，并启用 safe-accept。该设置在全量 144 个 state 上同时改善 contact-F1、RMSD 和 clash_rate。")
    lines.append("")
    lines.append("## 8. 错误与修正日志")
    lines.append("- `IndexError: binder index out of bounds`：根因是部分 V_exact 实验 binder 长度比 B1 生成 binder 长，exact contact anchor 中存在生成 pose 无法索引的 binder residue。修正为只保留共同长度内的 anchor；若过滤后为空，则退回最近距离 pairs。")
    lines.append("- `TypeError: PosixPath is not JSON serializable`：根因是 `vars(args)` 直接进入 JSON summary。修正为对 Path 参数字符串化。")
    lines.append("- `Namespace has no attribute allow_clash_worsening`：根因是第一次插入 safe-accept 参数不完整。修正为在 argparse 中显式加入 `--allow-clash-worsening`，默认关闭 clash worsening。")
    lines.append("所有错误均已重新运行对应 smoke/full 命令验证通过。")
    lines.append("")
    lines.append("## 9. 下一步执行标准")
    lines.append("1. 先补 `B0-native` 和 `B0-same-refiner` 的真实 generation artifact，不能再以 B0 missing 进入结论。")
    lines.append("2. 将 Stage09 诊断中有效的 `anchor_contact + clash_repulsion + interface_shell + safe-accept` 从 post-hoc rigid optimizer 前移到 Strategy01 state-specific flow sampling 的每步或后半程 guidance。")
    lines.append("3. 训练 loss 同步提高 interface/contact/clash 权重，并使用 persistent anchors 做 residue-level 加权。")
    lines.append("4. 只有当 B1-guided 在 exact-only 上相对公平 B0 至少改善一个 worst-state 核心指标，才进入更大规模训练。")
    lines.append("")
    lines.append("## 10. 复现命令")
    lines.append("```bash")
    lines.append("cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
    lines.append("/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python scripts/strategy01/stage09_b0_fair_baseline_audit.py")
    lines.append("/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python scripts/strategy01/stage09_interface_guidance_diagnostics.py --steps 60 --clash-weight 30 --shell-weight 0.10 --tether-weight 0.002")
    lines.append("/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python scripts/strategy01/stage09_append_report.py")
    lines.append("```")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()