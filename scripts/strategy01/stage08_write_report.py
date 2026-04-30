#!/usr/bin/env python
"""Write the Stage08 Chinese report from audit/mining summaries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(obj: dict[str, Any] | None) -> str:
    if not obj:
        return "无"
    return ", ".join(f"{k}: {v}" for k, v in obj.items())


def fmt_rejected_groups(items: list[dict[str, Any]] | None) -> str:
    if not items:
        return "无。"
    lines = []
    for item in items[:10]:
        entries = ",".join(item.get("entries", []))
        lines.append(
            f"- `{item.get('group_key')}`：entries={entries}，reason={item.get('reason')}，"
            f"motion={item.get('max_target_motion_A')} Å，persistent_anchors={item.get('persistent_anchor_count')}，"
            f"min_contacts={item.get('min_contacts')}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-summary", type=Path, default=REPO_ROOT / "reports/strategy01/probes/stage08_exact_audit_summary.json")
    parser.add_argument("--cross-entry-summary", type=Path, default=REPO_ROOT / "reports/strategy01/probes/stage08_cross_entry_mining_smoke_summary.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports/strategy01/strategy01_stage08_high_quality_data_report.md")
    args = parser.parse_args()

    audit = read_json(args.audit_summary)
    cross = read_json(args.cross_entry_summary)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    audit_output_dir = audit.get("output_dir", str(REPO_ROOT / "data/strategy01/stage08_high_quality_dataset"))
    text = f"""# Strategy01 Stage08 高质量多状态复合物数据审计与下一步微调报告

生成时间：{now}

## 1. 阶段目标

Stage08 的目标不是继续扩大 Stage07 的低质量数据生产，而是先把数据可信度做实。当前 Strategy01 的科学目标固定为：一个 shared binder sequence 能兼容 target 的多个真实功能构象，同时每个 state 都有合理的 state-specific target-binder complex geometry。这个目标不能用坐标平均结构、单态静态分数、或无初始 pose 的 Boltz2 redocking 作为主判断。

Stage07 已经证明训练链路、state-specific sampler 和 shared sequence head 可以跑通，但也暴露两个关键瓶颈。第一，`110/28` pilot 可以让 loss 明显下降，但生成序列仍弱。第二，Boltz2 redocking 即使对 B2 reference sequence 也出现 `contact_count=0`，说明无 pose docking 不能作为主 benchmark。Stage08 因此把主线改为构建可审计的 exact/hybrid 多状态复合物监督数据，再基于这些数据做微调。

## 2. 文献和数据源依据

DynamicMPNN 支持把多构象兼容序列作为显式学习目标，而不是先做单态设计再 post-hoc 聚合：[DynamicMPNN](https://arxiv.org/abs/2507.21938)。

RECON/MSD 说明多状态设计的本质是让同一序列兼容多个局部结构环境：[RECON](https://pmc.ncbi.nlm.nih.gov/articles/PMC7032724/)。

ProChoreo 的任务定义强调 explicitly incorporates conformational ensembles，支持把 target ensemble 作为 binder 设计的一等条件：[ProChoreo](https://www.lifescience.net/preprints/7333/prochoreo-de-novo-binder-design-from-conformationa/)。

CoDNaS、PDBFlex 和 ATLAS 可提供 target conformational ensemble 来源，但它们本身不等于 target-binder exact complex 监督：[CoDNaS](https://academic.oup.com/database/article/doi/10.1093/database/baw038/2630306)、[PDBFlex](https://pdbflex.org/)、[ATLAS](https://academic.oup.com/nar/article/52/D1/D384/7438909)。

PINDER、Dockground、DIPS-Plus、ProtCID、PepBDB/Propedia/PepX 是后续 exact/hybrid 候选的主要来源：[PINDER](https://pinder-org.github.io/pinder/index.html)、[Dockground](https://dockground.compbio.ku.edu/unbound/unbound-docking-benchmarks.php)、[DIPS-Plus](https://www.nature.com/articles/s41597-023-02409-3)、[ProtCID](https://www.nature.com/articles/s41467-020-14301-4)、[PepBDB](https://academic.oup.com/bioinformatics/article/35/1/175/5050021)。

pDockQ2/AlphaFold-Multimer 评估文献支持评估接口时必须结合 PAE/ipAE，而不是只看 pLDDT/contact：[pDockQ2](https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714)。

RCSB Search/Data API 是本阶段自动挖掘 PDB 元数据和结构条目的基础接口：[RCSB Search API](https://search.rcsb.org/) 和 [RCSB Data API](https://data.rcsb.org/)。

## 3. 本阶段新增代码

新增 `scripts/strategy01/stage08_exact_dataset_audit.py`。它读取 Stage06/Stage07 产生的 manifest，逐条检查是否满足 Stage08 的 exact 数据契约：至少两个 target states、至少两个 experimental complex states、同一 target chain、同一 binder chain、同一 shared binder sequence、每个 state 有足够 target-binder contacts、target motion 在合理区间，并计算 persistent anchors。输出分为 `V_exact_main`、`V_exact_challenge`、`hybrid_candidate`、`invalid_or_auxiliary`。

新增 `scripts/strategy01/stage08_mine_rcsb_cross_entry.py`。它通过 RCSB Search API 做小规模跨条目 smoke mining，寻找同 target sequence + 同 binder sequence 在多个 PDB 条目中重复出现的候选。该脚本输出只标为 `exact_experimental_candidate_requires_assembly_audit`，不会直接晋升为 V_exact，因为还需要 ProtCID/biological assembly 支持。

新增 `scripts/strategy01/stage08_write_report.py`。它读取审计和挖掘 summary，生成本中文报告，确保每次 Stage08 审计都能复现并留下结论。

## 4. Exact Audit 结果

输入 manifest：

```text
{chr(10).join(str(x) for x in audit.get("input_manifests", []))}
```

输出目录：`{audit_output_dir}`

状态：`{audit.get("status")}`

总样本数：`{audit.get("n_total")}`

按类别统计：{fmt_counts(audit.get("counts_by_class"))}

按 exact kind 统计：{fmt_counts(audit.get("counts_by_exact_kind"))}

按 motion bin 统计：{fmt_counts(audit.get("counts_by_motion_bin"))}

按 source_db 统计：{fmt_counts(audit.get("counts_by_source_db"))}

按类别 family 数：{fmt_counts(audit.get("families_by_class"))}

当前审计结论：如果 `V_exact_main` 低于 `32` 或 family 低于 `12`，则不能进入大规模微调验收。此时应优先继续挖 cross-entry exact 和 curated exact，而不是扩大 hybrid/bronze 伪标注。

## 5. Cross-entry RCSB Mining Smoke 结果

状态：`{cross.get("status")}`

查询条目数：`{cross.get("n_entries_queried")}`

有可用 chain pair 的条目数：`{cross.get("n_entries_with_usable_pair")}`

序列 group 数：`{cross.get("n_sequence_groups")}`

至少两个 entry 的 group 数：`{cross.get("n_groups_with_ge2_entries")}`

cross-entry 候选数：`{cross.get("n_cross_entry_candidates")}`

候选 manifest：`{cross.get("manifest")}`

被拒绝的跨条目同序列 group 诊断：

{fmt_rejected_groups(cross.get("rejected_group_diagnostics"))}

解释：该 mining smoke 的目的是验证自动挖掘链路，而不是最终建立 benchmark。任何 cross-entry 候选都必须继续做 biological assembly 支持、interface area、ProtCID/ProtCAD 或 author assembly 复核，否则不能作为 V_exact 主口径。

## 6. 数据分层策略

`V_exact` 只保留全实验多状态复合物，用作最终判断。这里的 exact 不只是 target 有多个结构，还必须是同 target、同 binder sequence、多个 state 均有实验复合物几何。

`T_exact` 来自多余 exact 数据，可用于训练，但优先保留 family-holdout external benchmark。exact 太少时不要把全部 exact 拿去训练。

`T_hybrid_silver` 使用真实 target states 和真实 binder sequence，但缺失 state 的 complex label 必须由 constrained reconstruction 产生。由于 Stage07 已证明 free redocking 不可靠，hybrid 不允许使用无约束 Boltz2 从零 docking 作为主 label。必须使用实验 source interface anchors、target-state template、binder template 或已有 pose，并通过 contact-F1、clash、interchain PAE 和 protein ipTM 过滤。

`T_aux` 包括小分子 apo/holo 或只有 target ensemble 没有 protein/peptide binder sequence 的样本。它只能训练 target ensemble encoder、pocket/interface proxy 或 target flexibility，不进入 shared binder 主监督 CE。

## 7. Motion 与 Anchor 过滤原则

主训练保留 `1.0-8.0 Å` target/interface motion。这个范围覆盖局部口袋变化、小到中等 hinge motion，以及部分大幅但仍可能保留共享 anchors 的构象变化。`8-10 Å` 只进入 challenge 或 silver，且必须有 persistent interface anchors。`>10 Å` 不进入自动主训练，因为此时同一 binder 兼容多个状态的假设很容易变成跨拓扑或不同界面问题。

这个过滤不是为了追求 RMSD 本身，而是为了服务核心科学问题：binder 需要兼容 target 的功能构象变化。RMSD 只是粗筛，真正决定是否可学的是 persistent anchor count、contact persistence、interface RMSD 和 state-specific complex geometry。

## 8. 微调前置门槛

进入下一轮微调前，最低门槛保持为：

- `T_train >= 256`
- `T_val >= 64`
- `V_exact >= 32`
- `V_exact` 覆盖 `>=12` 个 families
- train/val/test 按 family、UniProt pair 和 binder sequence cluster 严格切分
- exact/hybrid 样本都有 contact/distance/interface labels
- local latent 必须来自 Complexa AE encoder mean，不允许 geometry proxy 混入主训练

如果这些门槛不满足，下一步应该继续数据挖掘和标签质量控制，而不是跑更长训练。

## 9. B0/B1/B2 Benchmark 修正

B0 是原始 Complexa 单状态生成 shared sequence 和 pose，然后对各 state 直接和 exact contact/interface geometry 对齐评估。

B1 是 Strategy01 ensemble-input 生成 shared sequence 和 K 个 state-specific poses，直接与 V_exact 的 per-state complex contact map、interface RMSD、clash、contact persistence 对比。

B2 是 V_exact 实验真实复合物几何上限，另可附加真实 binder sequence 的 constrained relaxation 结果作为辅助。Boltz2 不再作为无 pose docking 主裁判，只能作为 constrained relaxation/refold 辅助表。

## 10. 复现命令

```bash
cd /data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase
PY=/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/python

$PY -m compileall -q \\
  scripts/strategy01/stage08_exact_dataset_audit.py \\
  scripts/strategy01/stage08_mine_rcsb_cross_entry.py \\
  scripts/strategy01/stage08_write_report.py

$PY scripts/strategy01/stage08_exact_dataset_audit.py \\
  --input-manifest data/strategy01/stage07_validation_expand/V_exact_manifest.json \\
  --input-manifest data/strategy01/stage06_validation/V_exact_manifest.json \\
  --out-dir data/strategy01/stage08_high_quality_dataset \\
  --summary reports/strategy01/probes/stage08_exact_audit_summary.json

$PY scripts/strategy01/stage08_mine_rcsb_cross_entry.py \\
  --max-entries 80 \\
  --out-dir data/strategy01/stage08_high_quality_dataset \\
  --summary reports/strategy01/probes/stage08_cross_entry_mining_smoke_summary.json

$PY scripts/strategy01/stage08_write_report.py \\
  --audit-summary reports/strategy01/probes/stage08_exact_audit_summary.json \\
  --cross-entry-summary reports/strategy01/probes/stage08_cross_entry_mining_smoke_summary.json \\
  --out reports/strategy01/strategy01_stage08_high_quality_data_report.md
```

## 11. 错误与修正日志

本阶段代码层面要重点防止三类错误。第一，不能把 NMR 多模型 exact-like 数据误报成跨条目 exact benchmark；报告中已用 `exact_kind` 拆分。第二，不能把 cross-entry smoke candidate 直接当作 V_exact；脚本强制标记 `needs_assembly_audit=true`。第三，不能让 hybrid/predicted complex 进入 exact 主口径；audit 里缺少 experimental complex paths 的样本会降级为 `hybrid_candidate`。

实际执行中还出现了一次 Windows PowerShell 远程脚本 CRLF 问题：早期 `bash` 续行参数混入 `\\r`，导致 summary 文件被写成 `stage08_exact_audit_summary.json\\r` 和 `stage08_cross_entry_mining_smoke_summary.json\\r`，报告脚本因此读不到 summary 并显示 `missing`。修正方式是用远程 Python 扫描并重命名带 `\\r` 的文件名，之后统一用 LF 写远程临时脚本并重新生成报告。修正后两个 summary 均可正常读取，报告状态分别为 `{audit.get("status")}` 和 `{cross.get("status")}`。

## 12. 下一步执行建议

下一步优先扩展数据源，而不是先大训练。具体顺序是：先接 ProtCID/ProtCAD 或可下载的 biological assembly/interface cluster 信息，再接 PepBDB/Propedia/PepX 的 peptide cases，然后接 PINDER/Dockground 的真实 PPI bound/unbound pairs。只有当 `V_exact_main` 达到至少 `32` 且 family 覆盖足够，才进入 `256/64` pilot finetune。若 exact 不足，训练可以用 `T_hybrid_silver`，但最终胜负必须仍以 exact-only benchmark 为主。
"""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(json.dumps({"report": str(args.out), "audit_status": audit.get("status"), "cross_entry_status": cross.get("status")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
