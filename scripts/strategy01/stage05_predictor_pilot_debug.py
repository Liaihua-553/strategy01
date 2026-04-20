#!/usr/bin/env python
"""Stage05 predictor-derived dataset loss probes and pilot fine-tune wrapper.

This reuses the Stage04 multistate/interface loss implementation while making
the dataset and output directory explicit for predictor-derived Stage05 data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.strategy01.stage04_real_complex_loss_debug as s4  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "data" / "strategy01" / "stage05_predictor_multistate" / "stage05_predictor_pilot_samples_ae_latents.pt"
FALLBACK_DATASET = REPO_ROOT / "data" / "strategy01" / "stage05_predictor_multistate" / "stage05_predictor_pilot_samples.pt"
STAGE05_DIR = REPO_ROOT / "ckpts" / "stage05_predictor_pilot"
RUNS_DIR = STAGE05_DIR / "runs"
REPORT_PROBE_DIR = REPO_ROOT / "reports" / "strategy01" / "probes"


def jsonable(obj: Any) -> Any:
    return s4.jsonable(obj)


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.exists() and path == DEFAULT_DATASET and FALLBACK_DATASET.exists():
        path = FALLBACK_DATASET
    data = torch.load(path, map_location="cpu")
    return data["samples"], data.get("manifest", {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="stage05_predictor_pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit1-steps", type=int, default=300)
    parser.add_argument("--mini-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    args = parser.parse_args()
    s4.set_seed(args.seed)
    device = s4.choose_device(args.device)
    samples, manifest = load_dataset(args.dataset)
    train_samples = [s for s in samples if s["split"] == "train"]
    val_samples = [s for s in samples if s["split"] == "val"]
    if not train_samples:
        raise RuntimeError(f"No train samples found in {args.dataset}")
    model, fm, model_meta = s4.build_model(device)
    REPORT_PROBE_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / args.run_name
    results = {
        "run_name": args.run_name,
        "dataset": str(args.dataset),
        "device": str(device),
        "manifest_summary": {k: v for k, v in manifest.items() if k != "samples"},
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "model_meta": model_meta,
        "probes": {},
        "training": {},
    }
    results["probes"]["loss_unit"] = s4.loss_unit_probe(model, fm, train_samples, device)
    results["probes"]["synthetic_interface"] = s4.synthetic_interface_probe(model, fm, train_samples, device)
    results["probes"]["grad_route"] = s4.grad_route_probe(model, fm, train_samples, device)
    results["training"]["overfit1"] = s4.train_loop(model, fm, train_samples[:1], device, "overfit1", args.overfit1_steps, 1, args.eval_every, 2e-4, run_dir, fixed_seed=999)
    results["training"]["mini"] = s4.train_loop(model, fm, train_samples[: min(16, len(train_samples))], device, "mini", args.mini_steps, args.mini_batch_size, args.eval_every, 1e-4, run_dir, fixed_seed=None)
    if val_samples:
        model.eval()
        with torch.no_grad():
            val_batch = s4.collate_samples(val_samples[: min(4, len(val_samples))], device)
            val_total, val_losses, _ = s4.forward_loss(model, fm, val_batch, seed=888)
        results["validation"] = {"total": float(val_total.detach().cpu().item()), "losses": s4.summarize_losses(val_losses)}
    output_path = REPORT_PROBE_DIR / f"{args.run_name}_results.json"
    output_path.write_text(json.dumps(jsonable(results), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "passed", "results": str(output_path)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
