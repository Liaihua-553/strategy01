#!/usr/bin/env python
"""Plot Stage06 training curves from JSON history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
DEFAULT_RESULTS = REPO_ROOT / "reports" / "strategy01" / "probes" / "stage06_predictor_pilot_results.json"
DEFAULT_OUT = REPO_ROOT / "reports" / "strategy01" / "figures" / "stage06"

LOSS_KEY_MAP = {
    "multistate_total": "multistate_total",
    "multistate_struct": "multistate_struct_justlog",
    "multistate_seq": "multistate_seq_justlog",
    "multistate_mean": "multistate_mean_justlog",
    "multistate_cvar": "multistate_cvar_justlog",
    "multistate_var": "multistate_var_justlog",
    "interface_contact": "multistate_contact_justlog",
    "interface_distance": "multistate_distance_justlog",
    "interface_quality_proxy": "multistate_quality_proxy_justlog",
    "interface_clash": "multistate_clash_justlog",
    "state0_total": "multistate_state_0_justlog",
    "state1_total": "multistate_state_1_justlog",
    "state2_total": "multistate_state_2_justlog",
}


def ema(values: list[float], alpha: float = 0.25) -> list[float]:
    if not values:
        return []
    out = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1.0 - alpha) * out[-1])
    return out


def collect_history(section: dict) -> dict[str, list[float]]:
    history = section.get("history", [])
    rows = {
        "step": [],
        "train_total": [],
        "eval_total": [],
        "cuda_max_mem_gb": [],
        "elapsed_sec": [],
    }
    for k in LOSS_KEY_MAP:
        rows[k] = []
    for row in history:
        rows["step"].append(float(row["step"]))
        rows["train_total"].append(float(row["train_total"]))
        rows["eval_total"].append(float(row["eval_total"]))
        rows["cuda_max_mem_gb"].append(float(row.get("cuda_max_mem_gb", 0.0)))
        rows["elapsed_sec"].append(float(row.get("elapsed_sec", 0.0)))
        losses = row.get("eval_losses", {})
        for out_key, loss_key in LOSS_KEY_MAP.items():
            if loss_key in losses:
                rows[out_key].append(float(losses[loss_key]))
            else:
                rows[out_key].append(float("nan"))
    return rows


def plot_pair(x: list[float], y_raw: list[float], out_path: Path, title: str, y_label: str, log_y: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y_raw, alpha=0.35, linewidth=1.5, label="raw")
    ax.plot(x, ema(y_raw), linewidth=2.0, label="ema")
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(y_label)
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.results_json.read_text(encoding="utf-8"))
    for phase_name in ["overfit1", "overfit4", "mini"]:
        phase = data.get("training", {}).get(phase_name)
        if not phase:
            continue
        rows = collect_history(phase)
        metrics = [
            ("train_total", "train_total"),
            ("eval_total", "eval_total"),
            ("multistate_total", "multistate_total"),
            ("multistate_struct", "multistate_struct"),
            ("multistate_seq", "multistate_seq"),
            ("multistate_mean", "multistate_mean"),
            ("multistate_cvar", "multistate_cvar"),
            ("multistate_var", "multistate_var"),
            ("interface_contact", "interface_contact"),
            ("interface_distance", "interface_distance"),
            ("interface_quality_proxy", "interface_quality_proxy"),
            ("interface_clash", "interface_clash"),
            ("state0_total", "state0_total"),
            ("state1_total", "state1_total"),
            ("state2_total", "state2_total"),
            ("cuda_max_mem_gb", "cuda_max_mem_gb"),
            ("elapsed_sec", "elapsed_sec"),
        ]
        for key, label in metrics:
            y = [v for v in rows[key] if v == v]
            x = [rows["step"][idx] for idx, v in enumerate(rows[key]) if v == v]
            if not x:
                continue
            base = args.out_dir / f"{phase_name}_{key}"
            plot_pair(x, y, base.with_name(base.name + "_linear.png"), f"{phase_name} {label}", label, log_y=False)
            if key != "elapsed_sec" and all(v > 0 for v in y):
                plot_pair(x, y, base.with_name(base.name + "_log.png"), f"{phase_name} {label}", label, log_y=True)
    print(json.dumps({"status": "passed", "out_dir": str(args.out_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()