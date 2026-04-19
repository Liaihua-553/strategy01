#!/usr/bin/env python
"""Stage04 real-complex-derived loss probes and one-GPU debug fine-tune."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
DATA_DIR = REPO_ROOT / "data" / "strategy01" / "stage04_real_complex_multistate"
REPORT_PROBE_DIR = REPO_ROOT / "reports" / "strategy01" / "probes"
STAGE_DIR = REPO_ROOT / "ckpts" / "stage04_real_complex_interface_loss"
RUNS_DIR = STAGE_DIR / "runs"
STAGE_COMPLEXA = REPO_ROOT / "ckpts" / "stage03_multistate_loss" / "complexa_init_readonly_copy.ckpt"
STAGE_AE = REPO_ROOT / "ckpts" / "stage03_multistate_loss" / "complexa_ae_init_readonly_copy.ckpt"
NN_CONFIG = REPO_ROOT / "configs" / "nn" / "local_latents_score_nn_160M_multistate.yaml"
TRAIN_CONFIG = REPO_ROOT / "configs" / "training_local_latents_multistate_stage04_probe.yaml"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from proteinfoundation.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher  # noqa: E402
from proteinfoundation.nn.local_latents_transformer_multistate import LocalLatentsTransformerMultistate  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.detach().cpu().item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def load_dataset() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tensor_path = DATA_DIR / "stage04_debug_samples.pt"
    if not tensor_path.exists():
        raise FileNotFoundError(f"Stage04 tensor dataset not found: {tensor_path}. Run stage04_build_real_multistate_complex_dataset.py first.")
    data = torch.load(tensor_path, map_location="cpu", weights_only=False)
    return data["samples"], data["manifest"]


def pad_states(values: list[torch.Tensor], pad_value: float | int | bool = 0) -> torch.Tensor:
    max_k = max(v.shape[0] for v in values)
    tail_shape = values[0].shape[1:]
    out_shape = (len(values), max_k, *tail_shape)
    dtype = values[0].dtype
    out = torch.full(out_shape, pad_value, dtype=dtype)
    for i, v in enumerate(values):
        out[i, : v.shape[0]] = v
    return out


def collate_samples(samples: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    b = len(samples)
    nb = samples[0]["binder_seq_shared"].shape[0]
    state_present = pad_states([s["state_present_mask"] for s in samples], False).bool()
    state_mask = pad_states([s["state_mask"] for s in samples], False).bool()
    state_weights = pad_states([s["target_state_weights"] for s in samples], 0.0).float()
    state_roles = pad_states([s["target_state_roles"] for s in samples], 0).long()
    batch = {
        "mask": torch.stack([s["binder_seq_mask"] for s in samples]).bool(),
        "binder_seq_shared": torch.stack([s["binder_seq_shared"] for s in samples]).long(),
        "binder_seq_mask": torch.stack([s["binder_seq_mask"] for s in samples]).bool(),
        "x_1_states": {
            "bb_ca": pad_states([s["x_1_states"]["bb_ca"] for s in samples], 0.0).float(),
            "local_latents": pad_states([s["x_1_states"]["local_latents"] for s in samples], 0.0).float(),
        },
        "state_mask": state_mask,
        "state_present_mask": state_present,
        "target_state_weights": state_weights,
        "target_state_roles": state_roles,
        "x_target_states": pad_states([s["x_target_states"] for s in samples], 0.0).float(),
        "target_mask_states": pad_states([s["target_mask_states"] for s in samples], False).bool(),
        "seq_target_states": pad_states([s["seq_target_states"] for s in samples], 0).long(),
        "target_hotspot_mask_states": pad_states([s["target_hotspot_mask_states"] for s in samples], False).bool(),
        "seq_target_mask_states": pad_states([s["target_mask_states"].any(dim=-1) for s in samples], False).bool(),
        "interface_contact_labels": pad_states([s["interface_contact_labels"] for s in samples], 0.0).float(),
        "interface_distance_labels": pad_states([s["interface_distance_labels"] for s in samples], 0.0).float(),
        "interface_label_mask": pad_states([s["interface_label_mask"] for s in samples], False).bool(),
        "interface_quality_labels": pad_states([s["interface_quality_labels"] for s in samples], 0.0).float(),
        "interface_quality_mask": pad_states([s["interface_quality_mask"] for s in samples], False).bool(),
        "hotspot_mask": torch.zeros(b, nb, dtype=torch.bool),
        "residue_pdb_idx": torch.arange(1, nb + 1, dtype=torch.float32).unsqueeze(0).expand(b, -1),
        "chains": torch.zeros(b, nb, dtype=torch.long),
        "use_ca_coors_nm_feature": False,
        "use_residue_type_feature": False,
        "strict_feats": False,
    }
    batch["hotspot_mask"][:, : min(4, nb)] = True
    return move_batch(batch, device)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.clone()
        elif isinstance(value, dict):
            out[key] = {kk: vv.clone() if isinstance(vv, torch.Tensor) else vv for kk, vv in value.items()}
        else:
            out[key] = value
    return out


def load_cfg() -> tuple[Any, dict[str, Any], int, dict[str, Any]]:
    ckpt = torch.load(STAGE_COMPLEXA, map_location="cpu", weights_only=False)
    latent_dim = int(ckpt["state_dict"]["nn.local_latents_linear.1.weight"].shape[0])
    train_cfg = OmegaConf.load(TRAIN_CONFIG)
    nn_cfg = OmegaConf.to_container(OmegaConf.load(NN_CONFIG), resolve=True)
    train_cfg.nn = OmegaConf.create(nn_cfg)
    train_cfg.pretrain_ckpt_path = str(STAGE_COMPLEXA)
    train_cfg.autoencoder_ckpt_path = str(STAGE_AE)
    return train_cfg, nn_cfg, latent_dim, ckpt


def strip_nn_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k[len("nn.") :]: v for k, v in state_dict.items() if k.startswith("nn.")}


def build_model(device: torch.device) -> tuple[LocalLatentsTransformerMultistate, ProductSpaceFlowMatcher, dict[str, Any]]:
    train_cfg, nn_cfg, latent_dim, ckpt = load_cfg()
    model = LocalLatentsTransformerMultistate(**nn_cfg, latent_dim=latent_dim).to(device)
    nn_state = strip_nn_prefix(ckpt["state_dict"])
    model_state = model.state_dict()
    compatible = {k: v for k, v in nn_state.items() if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)}
    incompatible_shape = [k for k, v in nn_state.items() if k in model_state and tuple(v.shape) != tuple(model_state[k].shape)]
    load_info = model.load_state_dict(compatible, strict=False)
    fm = ProductSpaceFlowMatcher(train_cfg).to(device)
    return model, fm, {
        "latent_dim": latent_dim,
        "checkpoint_compatible_keys": len(compatible),
        "checkpoint_incompatible_shape_keys": incompatible_shape[:50],
        "missing_keys_count": len(load_info.missing_keys),
        "missing_keys_sample": load_info.missing_keys[:50],
        "train_config": str(TRAIN_CONFIG),
    }


def forward_loss(model: torch.nn.Module, fm: ProductSpaceFlowMatcher, batch: dict[str, Any], seed: int | None = None):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    work_batch = clone_batch(batch)
    work_batch = fm.corrupt_multistate_batch(work_batch)
    nn_out = model(work_batch)
    losses = fm.compute_multistate_loss(work_batch, nn_out)
    return losses["multistate_total"].mean(), losses, nn_out


def summarize_losses(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: float(v.detach().float().mean().cpu().item()) for k, v in losses.items()}


def set_trainable(model: torch.nn.Module, phase: str) -> dict[str, Any]:
    for p in model.parameters():
        p.requires_grad = False
    prefixes = [
        "ensemble_target_encoder",
        "target2binder_cross_attention_layer",
        "state_condition_projector",
        "shared_seq_head",
        "state_token_norm",
        "interface_quality_head",
    ]
    if phase == "overfit":
        for p in model.parameters():
            p.requires_grad = True
        prefixes = ["ALL_PARAMETERS_FOR_STAGE04_OVERFIT"]
    elif phase == "mini":
        prefixes += ["local_latents_linear", "ca_linear"]
        prefixes += [f"transformer_layers.{i}" for i in range(10, 14)]
    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            p.requires_grad = True
    return {
        "phase": phase,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "prefixes": prefixes,
    }


def make_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)


def loss_unit_probe(model, fm, samples, device):
    model.eval()
    batch = collate_samples(samples[:2], device)
    with torch.no_grad():
        total, losses, nn_out = forward_loss(model, fm, batch, seed=101)
    return {
        "status": "passed",
        "total": float(total.detach().cpu().item()),
        "losses": summarize_losses(losses),
        "output_shapes": {k: list(v.shape) for k, v in nn_out.items() if isinstance(v, torch.Tensor)},
        "batch_shapes": {
            "x_target_states": list(batch["x_target_states"].shape),
            "interface_contact_labels": list(batch["interface_contact_labels"].shape),
            "interface_quality_labels": list(batch["interface_quality_labels"].shape),
        },
        "nan_detected": bool(any(torch.isnan(v).any().item() for v in losses.values())),
    }


def synthetic_interface_probe(model, fm, samples, device):
    model.eval()
    batch = collate_samples(samples[:2], device)
    with torch.no_grad():
        _, good_losses, _ = forward_loss(model, fm, batch, seed=222)
    bad_batch = clone_batch(batch)
    bad_batch["interface_distance_labels"] = torch.zeros_like(bad_batch["interface_distance_labels"])
    bad_batch["interface_contact_labels"] = 1.0 - bad_batch["interface_contact_labels"]
    bad_batch["interface_contact_labels"] = bad_batch["interface_contact_labels"] * bad_batch["interface_label_mask"].float()
    with torch.no_grad():
        _, bad_losses, _ = forward_loss(model, fm, bad_batch, seed=222)
    return {
        "status": "passed",
        "good_contact_loss": float(good_losses["multistate_contact_justlog"].mean().cpu().item()),
        "bad_contact_loss": float(bad_losses["multistate_contact_justlog"].mean().cpu().item()),
        "good_distance_loss": float(good_losses["multistate_distance_justlog"].mean().cpu().item()),
        "bad_distance_loss": float(bad_losses["multistate_distance_justlog"].mean().cpu().item()),
        "bad_distance_increases": bool(bad_losses["multistate_distance_justlog"].mean() > good_losses["multistate_distance_justlog"].mean()),
    }


def grad_route_probe(model, fm, samples, device):
    model.train()
    trainable = set_trainable(model, "overfit")
    batch = collate_samples(samples[:2], device)
    total, _, _ = forward_loss(model, fm, batch, seed=303)
    total.backward()
    params = dict(model.named_parameters())
    names = [
        "ensemble_target_encoder.cross_state_fusion.q_proj.weight",
        "target2binder_cross_attention_layer.0.mhba.mha.to_q_a.weight",
        "shared_seq_head.1.weight",
        "ca_linear.1.weight",
        "local_latents_linear.1.weight",
        "state_condition_projector.1.weight",
        "interface_quality_head.1.weight",
    ]
    grad_norms = {}
    for name in names:
        p = params.get(name)
        grad_norms[name] = None if p is None or p.grad is None else float(p.grad.detach().norm().cpu().item())
    model.zero_grad(set_to_none=True)
    return {
        "status": "passed",
        "loss": float(total.detach().cpu().item()),
        "trainable_info": trainable,
        "grad_norms": grad_norms,
        "all_required_grad_nonzero": bool(all(v is not None and v > 0 for v in grad_norms.values())),
    }


def train_loop(model, fm, samples, device, phase, steps, batch_size, eval_every, lr, run_dir, fixed_seed=None):
    model.train()
    trainable = set_trainable(model, "overfit" if phase == "overfit1" else "mini")
    opt = make_optimizer(model, lr)
    fixed_eval = collate_samples(samples[: min(batch_size, len(samples))], device)
    with torch.no_grad():
        initial, _, _ = forward_loss(model, fm, fixed_eval, seed=777)
    initial_value = float(initial.detach().cpu().item())
    history = []
    start = time.time()
    for step in range(1, steps + 1):
        batch_samples = [samples[((step - 1) * batch_size + j) % len(samples)] for j in range(batch_size)]
        batch = collate_samples(batch_samples, device)
        opt.zero_grad(set_to_none=True)
        total, _, _ = forward_loss(model, fm, batch, seed=fixed_seed)
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}: {total}")
        total.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            with torch.no_grad():
                eval_total, eval_losses, _ = forward_loss(model, fm, fixed_eval, seed=777)
            row = {
                "step": step,
                "train_total": float(total.detach().cpu().item()),
                "eval_total": float(eval_total.detach().cpu().item()),
                "eval_losses": summarize_losses(eval_losses),
                "elapsed_sec": time.time() - start,
                "cuda_max_mem_gb": torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0,
            }
            history.append(row)
            print(json.dumps({"phase": phase, **row}), flush=True)
    with torch.no_grad():
        final, final_losses, _ = forward_loss(model, fm, fixed_eval, seed=777)
    final_value = float(final.detach().cpu().item())
    drop = (initial_value - final_value) / max(abs(initial_value), 1e-8)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "phase": phase, "steps": steps}, run_dir / f"{phase}_final.pt")
    return {
        "phase": phase,
        "steps": steps,
        "batch_size": batch_size,
        "initial_eval_total": initial_value,
        "final_eval_total": final_value,
        "drop_fraction": drop,
        "final_losses": summarize_losses(final_losses),
        "history": history,
        "elapsed_sec": time.time() - start,
        "step_time_sec": (time.time() - start) / max(steps, 1),
        "cuda_max_mem_gb": torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0,
        "trainable_info": trainable,
    }


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="stage04_real_complex_interface_loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit1-steps", type=int, default=300)
    parser.add_argument("--mini-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    args = parser.parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    samples, manifest = load_dataset()
    train_samples = [s for s in samples if s["split"] == "train"]
    val_samples = [s for s in samples if s["split"] == "val"]
    model, fm, model_meta = build_model(device)
    REPORT_PROBE_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / args.run_name
    results = {
        "run_name": args.run_name,
        "device": str(device),
        "manifest_summary": {k: v for k, v in manifest.items() if k != "samples"},
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "model_meta": model_meta,
        "probes": {},
        "training": {},
    }
    results["probes"]["loss_unit"] = loss_unit_probe(model, fm, train_samples, device)
    results["probes"]["synthetic_interface"] = synthetic_interface_probe(model, fm, train_samples, device)
    results["probes"]["grad_route"] = grad_route_probe(model, fm, train_samples, device)
    results["training"]["overfit1"] = train_loop(model, fm, train_samples[:1], device, "overfit1", args.overfit1_steps, 1, args.eval_every, 2e-4, run_dir, fixed_seed=999)
    results["training"]["mini"] = train_loop(model, fm, train_samples[: min(16, len(train_samples))], device, "mini", args.mini_steps, args.mini_batch_size, args.eval_every, 1e-4, run_dir, fixed_seed=None)
    if val_samples:
        model.eval()
        with torch.no_grad():
            val_batch = collate_samples(val_samples[: min(4, len(val_samples))], device)
            val_total, val_losses, _ = forward_loss(model, fm, val_batch, seed=888)
        results["validation"] = {"total": float(val_total.detach().cpu().item()), "losses": summarize_losses(val_losses)}
    output_path = REPORT_PROBE_DIR / f"{args.run_name}_results.json"
    output_path.write_text(json.dumps(jsonable(results), indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "results": str(output_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
