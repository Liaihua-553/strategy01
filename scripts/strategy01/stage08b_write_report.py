#!/usr/bin/env python
"""Write the Chinese Stage08B report from probe summaries."""

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


def pct(drop_fraction: float | None) -> str:
    if drop_fraction is None:
        return "NA"
    return f"{drop_fraction * 100:.1f}%"


def num(x: Any, digits: int = 3) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:  # noqa: BLE001
        return str(x)


def phase_table(training: dict[str, Any]) -> str:
    lines = [
        "| 阶段 | steps | batch | eval total 起点 | eval total 终点 | 下降 | eval CVaR 起点 | eval CVaR 终点 | 显存GB | sec/step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in ["overfit1", "overfit4", "mini"]:
        obj = training.get(phase) or {}
        hist = obj.get("history") or []
        first = hist[0].get("eval_losses", {}) if hist else {}
        last = hist[-1].get("eval_losses", {}) if hist else {}
        lines.append(
            "| {phase} | {steps} | {batch} | {start} | {end} | {drop} | {c0} | {c1} | {mem} | {step_time} |".format(
                phase=phase,
                steps=obj.get("steps", "NA"),
                batch=obj.get("batch_size", "NA"),
                start=num(obj.get("initial_eval_total")),
                end=num(obj.get("final_eval_total")),
                drop=pct(obj.get("drop_fraction")),
                c0=num(first.get("multistate_cvar_justlog")),
                c1=num(last.get("multistate_cvar_justlog")),
                mem=num(obj.get("cuda_max_mem_gb"), 2),
                step_time=num(obj.get("step_time_sec"), 3),
            )
        )
    return "\n".join(lines)


def state_table(training: dict[str, Any]) -> str:
    lines = [
        "| 阶段 | state0 起点→终点 | state1 起点→终点 | state2 起点→终点 |",
        "|---|---:|---:|---:|",
    ]
    for phase in ["overfit1", "overfit4", "mini"]:
        obj = training.get(phase) or {}
        hist = obj.get("history") or []
        if not hist:
            lines.append(f"| {phase} | NA | NA | NA |")
            continue
        first = hist[0].get("eval_losses", {})
        last = hist[-1].get("eval_losses", {})
        vals = []
        for i in range(3):
            vals.append(f"{num(first.get(f'multistate_state_{i}_justlog'))}→{num(last.get(f'multistate_state_{i}_justlog'))}")
        lines.append(f"| {phase} | {vals[0]} | {vals[1]} | {vals[2]} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "reports/strategy01/strategy01_stage08b_high_quality_refinetune_report.md")
    args = parser.parse_args()

    probes = REPO / "reports/strategy01/probes"
    figs = REPO / "reports/strategy01/figures/stage08b/merged_pilot"
    pinder = load_json(probes / "stage08b_pinder_build_summary.json", {})
    pinder_len = load_json(probes / "stage08b_pinder_len300_probe_summary.json", {})
    pinder_bronze = load_json(probes / "stage08b_pinder_bronze_probe_summary.json", {})
    ae = load_json(probes / "stage08b_pinder_ae_latent_summary.json", {})
    merge = load_json(probes / "stage08b_merge_summary.json", {})
    schema = load_json(probes / "stage08b_schema_probe_results.json", {})
    train = load_json(probes / "stage08b_merged_pilot_results.json", {})
    benchmark = load_json(probes / "stage08b_exact_geometry_benchmark_summary.json", {})

    batch = (train.get("batch_selection") or {}).get("selected_batch_size")
    exact = benchmark.get("B2_exact_only_reference") or {}
    b2 = exact.get("b2_exact_reference") or {}
    b1 = (benchmark.get("B1_strategy01_smoke") or {}).get("b1_strategy01") or {}

    report = f"""# Strategy01 Stage08B 高质量数据补完、再微调与 exact benchmark 闭环报告

## 1. 阶段结论

本阶段在远程策略仓 `{REPO}` 内执行，未修改 benchmark baseline 仓，也未覆盖 baseline checkpoint。阶段目标原本是补齐 curated exact/hybrid 数据、接入真实 AE latent、跑 `>=256 train / >=64 val` pilot、完成 B0/B1/B2 exact benchmark。实际完成情况如下：

- 数据源接入完成了 PINDER HF 小分片与 AF2 分片；远程 DNS 不稳定，所以采用本地下载后 scp 到远程。
- 严格 constrained-hybrid PINDER 只接受 `{pinder.get('accepted_total')}` 条，训练/验证 `{pinder.get('train_count')}/{pinder.get('val_count')}`，远低于目标 `256/64`。
- 已将 PINDER accepted 样本全部接入 Complexa AE latent，`processed_states={ae.get('processed_states')}`，无 geometry proxy 混入这 11 条。
- 合并 Stage07 已有 predictor-derived 数据后得到 `{merge.get('train_count')}/{merge.get('val_count')}` 的 pilot 数据，仍低于目标 `256/64`，因此本轮定位为 **pilot finetune**，不是完整规模训练。
- 单张 gu02 A100 上 `batch_size={batch}` 成功，mini 1500 steps 跑完，训练曲线与 30 张图已生成。
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
| strict | {pinder.get('accepted_total')} | {pinder.get('train_count')} | {pinder.get('val_count')} | {pinder.get('families_train')} | {pinder.get('families_val')} | `{pinder.get('reject_counts')}` |
| binder_max_len=300 probe | {pinder_len.get('accepted_total')} | {pinder_len.get('train_count')} | {pinder_len.get('val_count')} | {pinder_len.get('families_train')} | {pinder_len.get('families_val')} | `{pinder_len.get('reject_counts')}` |
| bronze-like relaxed probe | {pinder_bronze.get('accepted_total')} | {pinder_bronze.get('train_count')} | {pinder_bronze.get('val_count')} | {pinder_bronze.get('families_train')} | {pinder_bronze.get('families_val')} | `{pinder_bronze.get('reject_counts')}` |

主要瓶颈不是脚本错误，而是这两个小分片中可同时满足“合理 binder 长度、足够 target motion、contact-F1、clash 和 persistent anchors”的样本太少。最大拒绝项是 `insufficient_motion_states` 与 `binder_length`，这说明要达到几百条训练集必须接入更完整的 PINDER 分片、Dockground/ProtCID/PepBDB/Propedia/PepX 或自行构造 constrained hybrid，而不能依赖当前两个小分片。

### 3.2 AE latent 接入

AE 提取使用阶段副本：

`ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt`

结果：

- 输入：`{ae.get('dataset')}`
- 输出：`{ae.get('output')}`
- 样本数：`{ae.get('num_samples_total')}`
- 状态数：`{ae.get('processed_states')}`
- 设备：`{ae.get('device')}`
- 结论：PINDER accepted 样本已全部替换为真实 `encoder.mean [K,Nb,8]`，未回退 geometry proxy。

### 3.3 合并训练集

合并后：

- 总样本：`{merge.get('total_count')}`
- train/val：`{merge.get('train_count')}/{merge.get('val_count')}`
- 来源：`{merge.get('source_counts')}`
- train families：`{merge.get('families_train')}`
- val families：`{merge.get('families_val')}`

这个规模足够测试训练链路和收敛行为，但不足以宣称完成了 Stage08B 原定的 `256/64` curated 训练集。

## 4. 微调训练结果

训练作业：`stage08b_train_pilot.sbatch`，SLURM job `1911816`，运行在 `new/gu02` 单张 A100。

batch 选择：

```json
{json.dumps(train.get('batch_selection'), indent=2, ensure_ascii=False)}
```

训练总耗时：`{num(train.get('training_elapsed_sec'), 1)} sec`。

{phase_table(train.get('training') or {})}

各 state eval total：

{state_table(train.get('training') or {})}

观察：

- `mini` eval total 从 `{num((train.get('training') or {}).get('mini', {}).get('initial_eval_total'))}` 降到 `{num((train.get('training') or {}).get('mini', {}).get('final_eval_total'))}`，下降 `{pct((train.get('training') or {}).get('mini', {}).get('drop_fraction'))}`。
- `mini` CVaR 从 `{num(((train.get('training') or {}).get('mini', {}).get('history') or [{}])[0].get('eval_losses', {}).get('multistate_cvar_justlog'))}` 降到 `{num(((train.get('training') or {}).get('mini', {}).get('history') or [{}])[-1].get('eval_losses', {}).get('multistate_cvar_justlog'))}`。
- 三个 state loss 都下降，说明 state-specific trajectories 和 shared sequence consensus 的训练路径能共同学习，不是只优化 easiest state。
- 但 final full validation total 为 `{num((train.get('validation') or {}).get('total'))}`，明显高于 mini 曲线中的 eval subset，说明当前数据混合仍有分布差异，不能进入大规模训练。

训练图输出目录：

`reports/strategy01/figures/stage08b/merged_pilot/`

代表图：

![mini eval total]({figs / 'mini_eval_total_linear.png'})

![mini state0]({figs / 'mini_multistate_state_0_justlog_linear.png'})

![mini state1]({figs / 'mini_multistate_state_1_justlog_linear.png'})

![mini state2]({figs / 'mini_multistate_state_2_justlog_linear.png'})

## 5. Exact/reference benchmark 现状

运行脚本：

`scripts/strategy01/stage08b_exact_geometry_benchmark.py`

输出：

`reports/strategy01/probes/stage08b_exact_geometry_benchmark_summary.json`

### 5.1 B2 exact-only reference

- exact samples：`{exact.get('sample_count')}`
- contact cutoff：`{exact.get('contact_cutoff_A')} Å`
- B2 contact count summary：`{b2.get('contact_count')}`
- B2 contact persistence summary：`{b2.get('contact_persistence')}`
- target motion RMSD summary：`{b2.get('target_motion_rmsd_A')}`

这部分是 exact/reference 上限，用来判断实验复合物本身是否有足够可持续界面锚点。

### 5.2 B1 Strategy01 smoke

当前已有 B1 生成 artifact 来自 Stage07 sampling smoke：

```json
{json.dumps(benchmark.get('B1_strategy01_smoke'), indent=2, ensure_ascii=False)[:4000]}
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

$PY scripts/strategy01/stage08b_build_pinder_dataset.py \\
  --parquet data/strategy01/stage08b_curated_sources/pinder_hf/pinder_s-00000-of-00001.parquet \\
  --parquet data/strategy01/stage08b_curated_sources/pinder_hf/pinder_af2-00000-of-00001.parquet \\
  --out-dir data/strategy01/stage08b_pinder_hybrid \\
  --summary reports/strategy01/probes/stage08b_pinder_build_summary.json \\
  --train-count 256 \\
  --val-count 64

$PY scripts/strategy01/stage05_extract_ae_latents.py \\
  --dataset data/strategy01/stage08b_pinder_hybrid/stage08b_pinder_hybrid_samples.pt \\
  --output data/strategy01/stage08b_pinder_hybrid/stage08b_pinder_hybrid_samples_ae_latents.pt \\
  --device cpu \\
  --summary reports/strategy01/probes/stage08b_pinder_ae_latent_summary.json

$PY scripts/strategy01/stage08b_merge_training_sets.py \\
  --dataset data/strategy01/stage07_production_v23_wave1/stage06_predictor_pilot_samples_materialized_ae_latents.pt \\
  --source-label stage07_v23_wave1 \\
  --dataset data/strategy01/stage08b_pinder_hybrid/stage08b_pinder_hybrid_samples_ae_latents.pt \\
  --source-label pinder_hybrid_silver \\
  --out-dir data/strategy01/stage08b_merged_training \\
  --summary reports/strategy01/probes/stage08b_merge_summary.json

$PY scripts/strategy01/stage07_sequence_consensus_training.py \\
  --dataset data/strategy01/stage08b_merged_training/stage08b_merged_pilot_samples.pt \\
  --device cpu \\
  --run-name stage08b_schema_probe \\
  --batch-candidates 4,2,1 \\
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
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
