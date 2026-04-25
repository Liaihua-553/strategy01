#!/usr/bin/env python
"""Plot compact Stage07 train/eval total and per-state total-loss curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
DEFAULT_RESULTS = REPO_ROOT / "reports" / "strategy01" / "probes" / "stage07_sequence_consensus_results.json"
DEFAULT_OUT = REPO_ROOT / "reports" / "strategy01" / "figures" / "stage07"


def ema(values: list[float], alpha: float = 0.25) -> list[float]:
    out: list[float] = []
    cur = None
    for value in values:
        cur = value if cur is None else alpha * value + (1.0 - alpha) * cur
        out.append(cur)
    return out


def collect_phase(history: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in history:
        if key == "train_total":
            value = row.get("train_total")
        elif key == "eval_total":
            value = row.get("eval_total")
        else:
            value = row.get("eval_losses", {}).get(key)
        if value is not None:
            xs.append(int(row["step"]))
            ys.append(float(value))
    return xs, ys


def plot_metric(phase: str, history: list[dict[str, Any]], metric: str, out_dir: Path) -> list[str]:
    xs, ys = collect_phase(history, metric)
    if not xs:
        return []
    paths = []
    for logy in (False, True):
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, label="raw", alpha=0.55)
        if len(ys) >= 2:
            plt.plot(xs, ema(ys), label="EMA", linewidth=2.0)
        plt.xlabel("step")
        plt.ylabel(metric)
        plt.title(f"Stage07 {phase} {metric}")
        if logy:
            plt.yscale("log")
        plt.grid(alpha=0.25)
        plt.legend()
        suffix = "log" if logy else "linear"
        path = out_dir / f"{phase}_{metric}_{suffix}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["train_total", "eval_total"] + [f"multistate_state_{idx}_justlog" for idx in range(5)]
    written: list[str] = []
    for phase, phase_data in data.get("training", {}).items():
        history = phase_data.get("history", [])
        for metric in metrics:
            written.extend(plot_metric(phase, history, metric, args.out_dir))
    summary = {"results": str(args.results), "figure_count": len(written), "figures": written}
    out_json = args.out_dir / "stage07_loss_curve_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()