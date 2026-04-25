#!/usr/bin/env python
"""Stage07 sequence-consensus probes and batch-size fallback fine-tuning.

This script keeps Stage04/05 data plumbing but verifies the Stage07 contract:
state-specific sequence logits must route gradients into the one shared sequence
head, and training starts with batch=4 before falling back to 2 or 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.strategy01.stage04_real_complex_loss_debug as s4  # noqa: E402

DEFAULT_DATASETS = [
    REPO_ROOT / "data" / "strategy01" / "stage06_production_pilot8" / "stage06_predictor_pilot_samples_ae_latents.pt",
    REPO_ROOT / "data" / "strategy01" / "stage06_production_pilot8" / "stage06_predictor_pilot_samples.pt",
    REPO_ROOT / "data" / "strategy01" / "stage06_production" / "stage06_predictor_pilot_samples_ae_latents.pt",
    REPO_ROOT / "data" / "strategy01" / "stage06_production" / "stage06_predictor_pilot_samples.pt",
    REPO_ROOT / "data" / "strategy01" / "stage05_predictor_multistate" / "stage05_predictor_pilot_samples_ae_latents.pt",
    REPO_ROOT / "data" / "strategy01" / "stage05_predictor_multistate" / "stage05_predictor_pilot_samples.pt",
]
REPORT_PROBE_DIR = REPO_ROOT / "reports" / "strategy01" / "probes"
RUNS_DIR = REPO_ROOT / "ckpts" / "stage07_sequence_consensus" / "runs"


def jsonable(obj: Any) -> Any:
    return s4.jsonable(obj)


def resolve_dataset(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for candidate in DEFAULT_DATASETS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No Stage05/06 predictor-derived dataset found.")


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["samples"], data.get("manifest", {})


def sequence_consensus_probe(model, fm, samples, device) -> dict[str, Any]:
    model.train()
    s4.set_trainable(model, "overfit")
    batch = s4.collate_samples(samples[: min(2, len(samples))], device)
    total, losses, nn_out = s4.forward_loss(model, fm, batch, seed=7001)
    required = ["seq_logits_shared", "state_seq_logits", "seq_logits_base_shared"]
    missing = [key for key in required if key not in nn_out]
    total.backward()
    params = dict(model.named_parameters())
    grad_names = [
        "shared_seq_head.1.weight",
        "state_seq_head.1.weight",
        "shared_seq_consensus_gate.1.weight",
        "state_condition_projector.1.weight",
    ]
    grad_norms = {}
    for name in grad_names:
        p = params.get(name)
        grad_norms[name] = None if p is None or p.grad is None else float(p.grad.detach().norm().cpu().item())
    model.zero_grad(set_to_none=True)
    shape_summary = {key: list(nn_out[key].shape) for key in required if key in nn_out}
    return {
        "status": "passed" if not missing and all(v is not None and v > 0 for v in grad_norms.values()) else "failed",
        "missing_outputs": missing,
        "shapes": shape_summary,
        "grad_norms": grad_norms,
        "loss_subset": {k: float(v.detach().float().mean().cpu().item()) for k, v in losses.items() if "seq" in k or "anchor_disagreement" in k},
    }


def try_batch_size(model, fm, train_samples, device, batch_size: int) -> dict[str, Any]:
    torch.cuda.empty_cache() if device.type == "cuda" else None
    model.train()
    s4.set_trainable(model, "mini")
    before_mem = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
    try:
        batch = s4.collate_samples(train_samples[: min(batch_size, len(train_samples))], device)
        total, _, _ = s4.forward_loss(model, fm, batch, seed=8000 + batch_size)
        total.backward()
        ok = bool(torch.isfinite(total).item())
        err = None
    except RuntimeError as exc:
        ok = False
        err = str(exc)
    finally:
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    after_mem = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else before_mem
    return {"batch_size": batch_size, "ok": ok, "error": err, "cuda_max_mem_gb": after_mem}


def choose_batch_size(model, fm, train_samples, device, candidates: list[int]) -> tuple[int, list[dict[str, Any]]]:
    attempts = []
    for batch_size in candidates:
        result = try_batch_size(model, fm, train_samples, device, batch_size)
        attempts.append(result)
        if result["ok"]:
            return batch_size, attempts
    return 1, attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="stage07_sequence_consensus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-candidates", default="4,2,1")
    parser.add_argument("--overfit1-steps", type=int, default=300)
    parser.add_argument("--overfit4-steps", type=int, default=600)
    parser.add_argument("--mini-steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--no-reset-between-phases", action="store_true")
    args = parser.parse_args()

    s4.set_seed(args.seed)
    device = s4.choose_device(args.device)
    dataset = resolve_dataset(args.dataset)
    samples, manifest = load_dataset(dataset)
    train_samples = [s for s in samples if s.get("split") == "train"]
    val_samples = [s for s in samples if s.get("split") == "val"]
    if not train_samples:
        raise RuntimeError(f"No train samples found in {dataset}")

    model, fm, model_meta = s4.build_model(device)
    REPORT_PROBE_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / args.run_name
    results: dict[str, Any] = {
        "run_name": args.run_name,
        "dataset": str(dataset),
        "device": str(device),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "manifest_summary": {k: v for k, v in manifest.items() if k != "samples"},
        "model_meta": model_meta,
        "probes": {},
        "training": {},
    }

    results["probes"]["loss_unit"] = s4.loss_unit_probe(model, fm, train_samples, device)
    results["probes"]["sequence_consensus"] = sequence_consensus_probe(model, fm, train_samples, device)
    results["probes"]["grad_route"] = s4.grad_route_probe(model, fm, train_samples, device)

    candidates = [int(x) for x in args.batch_candidates.split(",") if x.strip()]
    selected_batch, batch_attempts = choose_batch_size(model, fm, train_samples, device, candidates)
    results["batch_selection"] = {"selected_batch_size": selected_batch, "attempts": batch_attempts}

    if not args.skip_training:
        start = time.time()
        results["training"]["overfit1"] = s4.train_loop(model, fm, train_samples[:1], device, "overfit1", args.overfit1_steps, 1, args.eval_every, 2e-4, run_dir, fixed_seed=999)
        if not args.no_reset_between_phases:
            model, fm, _ = s4.build_model(device)
        results["training"]["overfit4"] = s4.train_loop(
            model,
            fm,
            train_samples[: min(4, len(train_samples))],
            device,
            "overfit4",
            args.overfit4_steps,
            min(selected_batch, 4),
            args.eval_every,
            1e-6,
            run_dir,
            fixed_seed=1234,
            trainable_phase="seq_consensus",
        )
        if not args.no_reset_between_phases:
            model, fm, _ = s4.build_model(device)
        pilot_samples = train_samples[: min(max(16, selected_batch), len(train_samples))]
        results["training"]["mini"] = s4.train_loop(
            model,
            fm,
            pilot_samples,
            device,
            "mini",
            args.mini_steps,
            selected_batch,
            args.eval_every,
            1e-6,
            run_dir,
            fixed_seed=None,
            trainable_phase="seq_consensus",
        )
        results["training_elapsed_sec"] = time.time() - start
        if val_samples:
            model.eval()
            with torch.no_grad():
                val_batch = s4.collate_samples(val_samples[: min(selected_batch, len(val_samples))], device)
                val_total, val_losses, _ = s4.forward_loss(model, fm, val_batch, seed=888)
            results["validation"] = {"total": float(val_total.detach().cpu().item()), "losses": s4.summarize_losses(val_losses)}

    output_path = REPORT_PROBE_DIR / f"{args.run_name}_results.json"
    output_path.write_text(json.dumps(jsonable(results), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "passed", "results": str(output_path), "selected_batch_size": selected_batch}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
