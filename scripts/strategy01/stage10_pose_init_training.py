#!/usr/bin/env python
"""Stage10 pose-initialized exact-augmented multistate fine-tuning.

Key differences from Stage07:
- variable-length collate for mixed Boltz/hybrid/exact samples;
- transferred source-pose initialization `init_bb_ca_states`;
- probes that verify init-pose gradients reach the model;
- exact-aug train/val dataset support.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import scripts.strategy01.stage04_real_complex_loss_debug as s4  # noqa: E402

DEFAULT_DATASET = REPO / "data/strategy01/stage10_exactaug_training/stage10_exactaug_trainval.pt"
REPORT_DIR = REPO / "reports/strategy01/probes"
RUNS_DIR = REPO / "ckpts/stage10_pose_init_exactaug/runs"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s4.jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["samples"], data.get("manifest", {})


def kabsch_apply(source_points: torch.Tensor, target_points: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
    if source_points.shape[0] < 3 or target_points.shape[0] < 3:
        return moving.clone()
    p = source_points.float()
    q = target_points.float()
    m = moving.float()
    p_mean = p.mean(dim=0)
    q_mean = q.mean(dim=0)
    pc = p - p_mean
    qc = q - q_mean
    h = pc.transpose(0, 1) @ qc
    try:
        u, _, vh = torch.linalg.svd(h)
        r = vh.transpose(0, 1) @ u.transpose(0, 1)
        if torch.det(r) < 0:
            vh = vh.clone()
            vh[-1, :] *= -1
            r = vh.transpose(0, 1) @ u.transpose(0, 1)
        t = q_mean - p_mean @ r
        return m @ r + t
    except RuntimeError:
        return moving.clone()


def build_init_bb_ca_states(sample: dict[str, Any], source_state: int = 0) -> torch.Tensor:
    target = sample["x_target_states"].float()
    target_mask = sample["target_mask_states"].bool()
    binder = sample["x_1_states"]["bb_ca"].float()
    valid_k = int(sample["state_present_mask"].bool().sum().item())
    source_state = min(max(0, source_state), max(valid_k - 1, 0))
    source_target_ca = target[source_state, :, 1, :]
    source_target_mask = target_mask[source_state, :, 1]
    source_binder = binder[source_state]
    init = torch.zeros_like(binder)
    for k in range(binder.shape[0]):
        if k >= valid_k:
            continue
        target_ca = target[k, :, 1, :]
        mask = source_target_mask & target_mask[k, :, 1]
        init[k] = kabsch_apply(source_target_ca[mask], target_ca[mask], source_binder)
    return init


def copy_tensor(dst: torch.Tensor, src: torch.Tensor, slices: tuple[slice, ...]) -> None:
    dst[slices] = src.to(dtype=dst.dtype)


def collate_variable(samples: list[dict[str, Any]], device: torch.device, source_state: int = 0) -> dict[str, Any]:
    b = len(samples)
    max_k = max(int(s["state_present_mask"].shape[0]) for s in samples)
    max_nb = max(int(s["binder_seq_shared"].shape[0]) for s in samples)
    max_nt = max(int(s["x_target_states"].shape[1]) for s in samples)
    latent_dim = int(samples[0]["x_1_states"]["local_latents"].shape[-1])

    batch: dict[str, Any] = {
        "mask": torch.zeros(b, max_nb, dtype=torch.bool),
        "binder_seq_shared": torch.zeros(b, max_nb, dtype=torch.long),
        "binder_seq_mask": torch.zeros(b, max_nb, dtype=torch.bool),
        "x_1_states": {
            "bb_ca": torch.zeros(b, max_k, max_nb, 3, dtype=torch.float32),
            "local_latents": torch.zeros(b, max_k, max_nb, latent_dim, dtype=torch.float32),
        },
        "init_bb_ca_states": torch.zeros(b, max_k, max_nb, 3, dtype=torch.float32),
        "state_mask": torch.zeros(b, max_k, max_nb, dtype=torch.bool),
        "state_present_mask": torch.zeros(b, max_k, dtype=torch.bool),
        "target_state_weights": torch.zeros(b, max_k, dtype=torch.float32),
        "target_state_roles": torch.zeros(b, max_k, dtype=torch.long),
        "x_target_states": torch.zeros(b, max_k, max_nt, 37, 3, dtype=torch.float32),
        "target_mask_states": torch.zeros(b, max_k, max_nt, 37, dtype=torch.bool),
        "seq_target_states": torch.zeros(b, max_k, max_nt, dtype=torch.long),
        "target_hotspot_mask_states": torch.zeros(b, max_k, max_nt, dtype=torch.bool),
        "seq_target_mask_states": torch.zeros(b, max_k, max_nt, dtype=torch.bool),
        "interface_contact_labels": torch.zeros(b, max_k, max_nt, max_nb, dtype=torch.float32),
        "interface_distance_labels": torch.zeros(b, max_k, max_nt, max_nb, dtype=torch.float32),
        "interface_label_mask": torch.zeros(b, max_k, max_nt, max_nb, dtype=torch.bool),
        "interface_quality_labels": torch.zeros(b, max_k, 5, dtype=torch.float32),
        "interface_quality_mask": torch.zeros(b, max_k, 5, dtype=torch.bool),
        "hotspot_mask": torch.zeros(b, max_nb, dtype=torch.bool),
        "residue_pdb_idx": torch.zeros(b, max_nb, dtype=torch.float32),
        "chains": torch.zeros(b, max_nb, dtype=torch.long),
        "use_ca_coors_nm_feature": False,
        "use_residue_type_feature": False,
        "strict_feats": False,
    }

    for i, s in enumerate(samples):
        nb = int(s["binder_seq_shared"].shape[0])
        k = int(s["state_present_mask"].shape[0])
        nt = int(s["x_target_states"].shape[1])
        batch["mask"][i, :nb] = s["binder_seq_mask"].bool()
        batch["binder_seq_shared"][i, :nb] = s["binder_seq_shared"].long()
        batch["binder_seq_mask"][i, :nb] = s["binder_seq_mask"].bool()
        batch["x_1_states"]["bb_ca"][i, :k, :nb] = s["x_1_states"]["bb_ca"].float()
        batch["x_1_states"]["local_latents"][i, :k, :nb] = s["x_1_states"]["local_latents"].float()
        batch["init_bb_ca_states"][i, :k, :nb] = build_init_bb_ca_states(s, source_state=source_state).float()
        batch["state_mask"][i, :k, :nb] = s["state_mask"].bool()
        batch["state_present_mask"][i, :k] = s["state_present_mask"].bool()
        batch["target_state_weights"][i, :k] = s["target_state_weights"].float()
        batch["target_state_roles"][i, :k] = s["target_state_roles"].long()
        batch["x_target_states"][i, :k, :nt] = s["x_target_states"].float()
        batch["target_mask_states"][i, :k, :nt] = s["target_mask_states"].bool()
        batch["seq_target_states"][i, :k, :nt] = s["seq_target_states"].long()
        batch["target_hotspot_mask_states"][i, :k, :nt] = s["target_hotspot_mask_states"].bool()
        batch["seq_target_mask_states"][i, :k, :nt] = s["target_mask_states"].bool().any(dim=-1)
        batch["interface_contact_labels"][i, :k, :nt, :nb] = s["interface_contact_labels"].float()
        batch["interface_distance_labels"][i, :k, :nt, :nb] = s["interface_distance_labels"].float()
        batch["interface_label_mask"][i, :k, :nt, :nb] = s["interface_label_mask"].bool()
        batch["interface_quality_labels"][i, :k] = s["interface_quality_labels"].float()
        batch["interface_quality_mask"][i, :k] = s["interface_quality_mask"].bool()
        batch["hotspot_mask"][i, : min(4, nb)] = True
        batch["residue_pdb_idx"][i, :nb] = torch.arange(1, nb + 1, dtype=torch.float32)
    return s4.move_batch(batch, device)


def forward_loss(model, fm, samples: list[dict[str, Any]], device: torch.device, seed: int | None = None):
    batch = collate_variable(samples, device)
    return s4.forward_loss(model, fm, batch, seed=seed)


def probe(model, fm, train_samples, device) -> dict[str, Any]:
    model.train()
    s4.set_trainable(model, "overfit")
    subset = train_samples[: min(2, len(train_samples))]
    batch = collate_variable(subset, device)
    corrupt = fm.corrupt_multistate_batch(s4.clone_batch(batch))
    init_delta = (corrupt["x_0_states"]["bb_ca"] - batch["init_bb_ca_states"]).abs()
    init_delta_mean = float(init_delta[corrupt["state_mask"]].mean().detach().cpu().item()) if corrupt["state_mask"].any() else 0.0
    total, losses, nn_out = s4.forward_loss(model, fm, batch, seed=1010)
    total.backward()
    params = dict(model.named_parameters())
    grad_names = [
        "init_pose_projector.1.weight",
        "shared_seq_head.1.weight",
        "state_seq_head.1.weight",
        "ca_linear.1.weight",
        "local_latents_linear.1.weight",
    ]
    grad_norms = {}
    for name in grad_names:
        p = params.get(name)
        grad_norms[name] = None if p is None or p.grad is None else float(p.grad.detach().norm().cpu().item())
    model.zero_grad(set_to_none=True)
    return {
        "status": "passed" if all(v is not None and v > 0 for v in grad_norms.values()) else "failed",
        "loss_total": float(total.detach().cpu().item()),
        "losses": s4.summarize_losses(losses),
        "output_shapes": {k: list(v.shape) for k, v in nn_out.items() if torch.is_tensor(v)},
        "init_delta_mean_nm": init_delta_mean,
        "grad_norms": grad_norms,
    }


def sample_cost(sample: dict[str, Any]) -> int:
    k = int(sample["state_present_mask"].shape[0])
    nb = int(sample["binder_seq_shared"].shape[0])
    nt = int(sample["x_target_states"].shape[1])
    return k * nb * nt


def choose_batch(model, fm, samples, device, candidates: list[int]) -> tuple[int, list[dict[str, Any]]]:
    attempts = []
    stress_samples = sorted(samples, key=sample_cost, reverse=True)
    for bs in candidates:
        try:
            subset = stress_samples[: min(bs, len(stress_samples))]
            total, _, _ = forward_loss(model, fm, subset, device, seed=2000 + bs)
            total.backward()
            ok = bool(torch.isfinite(total).item())
            err = None
        except RuntimeError as exc:
            ok = False
            err = str(exc)[:1000]
        finally:
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                mem = torch.cuda.max_memory_allocated() / (1024**3)
                torch.cuda.empty_cache()
            else:
                mem = 0.0
        attempts.append({"batch_size": bs, "ok": ok, "error": err, "cuda_max_mem_gb": mem})
        if ok:
            return bs, attempts
    return 1, attempts




def save_checkpoint_verified(obj: dict[str, Any], path: Path) -> dict[str, Any]:
    """Save a checkpoint atomically and verify it is readable.

    Stage10 full training exposed corrupt/zero-byte NFS checkpoint files even
    though the Python process completed.  We now write a temporary legacy-format
    file, load it back, then atomically replace the target.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    torch.save(obj, tmp, _use_new_zipfile_serialization=False)
    loaded = torch.load(tmp, map_location="cpu", weights_only=False)
    if "model_state_dict" not in loaded or not loaded["model_state_dict"]:
        raise RuntimeError(f"Checkpoint verification failed for {tmp}")
    tmp.replace(path)
    return {"path": str(path), "bytes": path.stat().st_size, "verified_keys": len(loaded["model_state_dict"])}


def train_loop(model, fm, samples, device, phase: str, steps: int, batch_size: int, eval_every: int, lr: float, run_dir: Path, trainable_phase: str, fixed_seed: int | None = None, save_checkpoint: bool = True) -> dict[str, Any]:
    model.train()
    trainable = s4.set_trainable(model, trainable_phase)
    opt = s4.make_optimizer(model, lr)
    fixed_subset = samples[: min(batch_size, len(samples))]
    with torch.no_grad():
        init_total, _, _ = forward_loss(model, fm, fixed_subset, device, seed=777)
    init_value = float(init_total.detach().cpu().item())
    history = []
    start = time.time()
    for step in range(1, steps + 1):
        subset = [samples[((step - 1) * batch_size + j) % len(samples)] for j in range(batch_size)]
        opt.zero_grad(set_to_none=True)
        total, losses, _ = forward_loss(model, fm, subset, device, seed=fixed_seed)
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}: {total}")
        total.backward()
        bad = []
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all().item():
                bad.append(name)
                if len(bad) >= 8:
                    break
        if bad:
            raise FloatingPointError(f"non-finite gradients at step {step}: {bad}")
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            with torch.no_grad():
                eval_total, eval_losses, _ = forward_loss(model, fm, fixed_subset, device, seed=777)
            row = {
                "step": step,
                "train_total": float(total.detach().cpu().item()),
                "eval_total": float(eval_total.detach().cpu().item()),
                "eval_losses": s4.summarize_losses(eval_losses),
                "elapsed_sec": time.time() - start,
                "cuda_max_mem_gb": torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0,
            }
            print(json.dumps({"phase": phase, **row}, ensure_ascii=False), flush=True)
            history.append(row)
    with torch.no_grad():
        final_total, final_losses, _ = forward_loss(model, fm, fixed_subset, device, seed=777)
    ckpt_info = save_checkpoint_verified({"model_state_dict": model.state_dict(), "phase": phase, "steps": steps}, run_dir / f"{phase}_final.pt") if save_checkpoint else None
    final_value = float(final_total.detach().cpu().item())
    return {
        "phase": phase,
        "steps": steps,
        "batch_size": batch_size,
        "initial_eval_total": init_value,
        "final_eval_total": final_value,
        "drop_fraction": (init_value - final_value) / max(abs(init_value), 1e-8),
        "final_losses": s4.summarize_losses(final_losses),
        "history": history,
        "elapsed_sec": time.time() - start,
        "step_time_sec": (time.time() - start) / max(steps, 1),
        "cuda_max_mem_gb": torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0,
        "trainable_info": trainable,
        "checkpoint": ckpt_info,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--run-name", default="stage10_pose_init_exactaug")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-candidates", default="4,2,1")
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-val-samples", type=int, default=0)
    ap.add_argument("--overfit1-steps", type=int, default=300)
    ap.add_argument("--overfit4-steps", type=int, default=600)
    ap.add_argument("--mini-steps", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--save-phases", default="mini", help="Comma-separated phases to checkpoint; default saves only mini to avoid quota-heavy overfit checkpoints.")
    args = ap.parse_args()

    s4.set_seed(args.seed)
    device = s4.choose_device(args.device)
    samples, manifest = load_dataset(args.dataset)
    train_samples = [s for s in samples if s.get("split") == "train"]
    val_samples = [s for s in samples if s.get("split") == "val"]
    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_samples = val_samples[: args.max_val_samples]
    if not train_samples:
        raise RuntimeError("No training samples selected")

    model, fm, model_meta = s4.build_model(device)
    report_path = REPORT_DIR / f"{args.run_name}_results.json"
    run_dir = RUNS_DIR / args.run_name
    results: dict[str, Any] = {
        "status": "running",
        "run_name": args.run_name,
        "dataset": str(args.dataset),
        "device": str(device),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "manifest_summary": {k: v for k, v in manifest.items() if k not in {"samples"}},
        "model_meta": model_meta,
        "probes": {},
        "training": {},
    }
    results["probes"]["init_pose_and_grad"] = probe(model, fm, train_samples, device)
    candidates = [int(x) for x in args.batch_candidates.split(",") if x.strip()]
    selected_batch, attempts = choose_batch(model, fm, train_samples, device, candidates)
    results["batch_selection"] = {"selected_batch_size": selected_batch, "attempts": attempts}
    write_json(report_path, results)

    if not args.skip_training:
        start = time.time()
        save_phases = {x.strip() for x in args.save_phases.split(",") if x.strip()}
        results["save_phases"] = sorted(save_phases)
        results["training"]["overfit1"] = train_loop(model, fm, train_samples[:1], device, "overfit1", args.overfit1_steps, 1, args.eval_every, 2e-4, run_dir, "overfit", fixed_seed=999, save_checkpoint="overfit1" in save_phases)
        model, fm, _ = s4.build_model(device)
        results["training"]["overfit4"] = train_loop(model, fm, train_samples[: min(4, len(train_samples))], device, "overfit4", args.overfit4_steps, min(selected_batch, 4), args.eval_every, 8e-5, run_dir, "mini", fixed_seed=1234, save_checkpoint="overfit4" in save_phases)
        model, fm, _ = s4.build_model(device)
        # Stage10 uses the full selected train set by default.  Stage07 used a
        # 16-sample mini subset, which is insufficient once experimental V_exact
        # samples are intentionally added to strengthen supervision.
        pilot_samples = train_samples
        results["training"]["mini"] = train_loop(model, fm, pilot_samples, device, "mini", args.mini_steps, selected_batch, args.eval_every, 5e-5, run_dir, "mini", fixed_seed=None, save_checkpoint="mini" in save_phases)
        results["training_elapsed_sec"] = time.time() - start
        if val_samples:
            model.eval()
            with torch.no_grad():
                val_subset = val_samples[: min(selected_batch, len(val_samples))]
                val_total, val_losses, _ = forward_loss(model, fm, val_subset, device, seed=888)
            results["validation"] = {"total": float(val_total.detach().cpu().item()), "losses": s4.summarize_losses(val_losses)}
    results["status"] = "passed"
    write_json(report_path, results)
    print(json.dumps({"status": "passed", "results": str(report_path), "selected_batch_size": selected_batch}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
