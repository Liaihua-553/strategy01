#!/usr/bin/env python
"""Stage11 Complexa-native sequence-flow coupling probes and fine-tuning.

This stage keeps Strategy01 inside the Complexa product-space flow instead of
adding an external sequence refiner.  The frozen AE decoder reads predicted
clean local latents and returns sequence logits; those logits are injected into
the multistate loss so the shared binder sequence constrains local-latent flow.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import scripts.strategy01.stage04_real_complex_loss_debug as s4  # noqa: E402
import scripts.strategy01.stage05_extract_ae_latents as s5ae  # noqa: E402
import scripts.strategy01.stage10_pose_init_training as s10  # noqa: E402

DEFAULT_DATASET = REPO / "data/strategy01/stage10_exactaug_training/stage10_exactaug_trainval.pt"
DEFAULT_STAGE10_CKPT = REPO / "ckpts/stage10_pose_init_exactaug/runs/stage10_pose_init_exactaug_full_v2/mini_final.pt"
DEFAULT_AE_CKPT = REPO / "ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt"
REPORT_DIR = REPO / "reports/strategy01/probes"
RUNS_DIR = REPO / "ckpts/stage11_flow_sequence/runs"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s4.jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def load_strategy_checkpoint(model: torch.nn.Module, path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"loaded": False, "reason": f"checkpoint not found: {path}"}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    cleaned = {}
    for key, value in state.items():
        if key.startswith("nn."):
            key = key[3:]
        cleaned[key] = value
    model_state = model.state_dict()
    compatible = {k: v for k, v in cleaned.items() if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)}
    incompatible = [k for k, v in cleaned.items() if k in model_state and tuple(v.shape) != tuple(model_state[k].shape)]
    info = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": True,
        "path": str(path),
        "compatible_keys": len(compatible),
        "incompatible_shape_keys": incompatible[:40],
        "missing_keys": info.missing_keys[:80],
        "unexpected_keys": info.unexpected_keys[:40],
    }


def configure_stage11_loss(fm: Any) -> dict[str, float]:
    cfg = fm.cfg_exp.loss.multistate
    OmegaConf.set_struct(cfg, False)
    values = {
        "lambda_ae_seq": 0.80,
        "lambda_seq_ae_consistency": 0.10,
        "lambda_flow_gate_reg": 0.03,
        "flow_gate_anchor_target": 0.12,
        "flow_gate_flex_target": 0.45,
        "ae_seq_cvar_topk": 2,
    }
    for key, value in values.items():
        cfg[key] = value
    return values


def state_clean_prediction(fm: Any, batch: dict[str, Any], nn_out: dict[str, torch.Tensor], data_mode: str, out_key: str) -> torch.Tensor:
    pred = nn_out[out_key]
    param = fm.cfg_exp.nn.output_parameterization[data_mode]
    if param == "x_1":
        return pred
    if param == "v":
        x_t = batch["x_t_states"][data_mode].to(device=pred.device, dtype=pred.dtype)
        t = batch["t_states"][data_mode].to(device=pred.device, dtype=pred.dtype)[:, :, None, None]
        return x_t + pred * (1.0 - t)
    raise ValueError(f"Unsupported output parameterization for {data_mode}: {param}")


def normalized_state_weights(batch: dict[str, Any], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    present = batch["state_present_mask"].to(device=device).bool()
    weights = batch.get("target_state_weights")
    if weights is None:
        weights = torch.ones_like(present, dtype=dtype, device=device)
    else:
        weights = weights.to(device=device, dtype=dtype)
    weights = weights * present.float()
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return weights / denom


def attach_ae_sequence_logits(ae: torch.nn.Module, fm: Any, batch: dict[str, Any], nn_out: dict[str, torch.Tensor], ca_source: str = "init") -> dict[str, torch.Tensor]:
    clean_lat = state_clean_prediction(fm, batch, nn_out, "local_latents", "local_latents_states")
    clean_bb = state_clean_prediction(fm, batch, nn_out, "bb_ca", "bb_ca_states")
    if ca_source == "init" and "init_bb_ca_states" in batch:
        clean_bb = batch["init_bb_ca_states"].to(device=clean_lat.device, dtype=clean_lat.dtype)
    elif ca_source == "exact":
        clean_bb = batch["x_1_states"]["bb_ca"].to(device=clean_lat.device, dtype=clean_lat.dtype)
    elif ca_source != "pred":
        raise ValueError(f"Unsupported AE CA source: {ca_source}")
    b, k, n, z = clean_lat.shape
    mask = batch["state_mask"].to(device=clean_lat.device).bool()
    flat_lat = clean_lat.reshape(b * k, n, z)
    flat_bb = clean_bb.reshape(b * k, n, 3)
    flat_mask = mask.reshape(b * k, n)
    # AE parameters stay frozen, but gradients must flow from seq CE back to
    # local_latents_states and bb_ca_states through the decoder computation.
    dec = ae.decode(z_latent=flat_lat, ca_coors_nm=flat_bb, mask=flat_mask)
    ae_logits = dec["seq_logits"].reshape(b, k, n, 20)
    weights = normalized_state_weights(batch, ae_logits.dtype, ae_logits.device)
    nn_out["ae_seq_logits_states"] = ae_logits
    nn_out["ae_seq_logits_shared"] = (ae_logits * weights[:, :, None, None]).sum(dim=1)
    return nn_out


def sequence_identity_from_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=-1)
    valid = mask.bool()
    match = (pred == target.to(pred.device)) & valid
    return match.float().sum(dim=-1) / valid.float().sum(dim=-1).clamp_min(1.0)


def forward_loss_stage11(
    model: torch.nn.Module,
    ae: torch.nn.Module,
    fm: Any,
    samples: list[dict[str, Any]],
    device: torch.device,
    seed: int | None = None,
    ae_ca_source: str = "init",
    noise_repeats: int = 1,
):
    if noise_repeats > 1:
        total_accum = None
        last_losses = None
        last_nn_out = None
        last_batch = None
        for repeat_idx in range(noise_repeats):
            repeat_seed = None if seed is None else seed + repeat_idx * 1009
            total, losses, nn_out, work_batch = forward_loss_stage11(
                model,
                ae,
                fm,
                samples,
                device,
                seed=repeat_seed,
                ae_ca_source=ae_ca_source,
                noise_repeats=1,
            )
            total_accum = total if total_accum is None else total_accum + total
            last_losses, last_nn_out, last_batch = losses, nn_out, work_batch
        return total_accum / float(noise_repeats), last_losses, last_nn_out, last_batch
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    batch = s10.collate_variable(samples, device)
    work_batch = s4.clone_batch(batch)
    work_batch = fm.corrupt_multistate_batch(work_batch)
    nn_out = model(work_batch)
    nn_out = attach_ae_sequence_logits(ae, fm, work_batch, nn_out, ca_source=ae_ca_source)
    losses = fm.compute_multistate_loss(work_batch, nn_out)
    return losses["multistate_total"].mean(), losses, nn_out, work_batch


def decode_exact_ae(ae: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    lat = batch["x_1_states"]["local_latents"]
    bb = batch["x_1_states"]["bb_ca"]
    mask = batch["state_mask"].bool()
    b, k, n, z = lat.shape
    dec = ae.decode(
        z_latent=lat.reshape(b * k, n, z),
        ca_coors_nm=bb.reshape(b * k, n, 3),
        mask=mask.reshape(b * k, n),
    )
    return dec["seq_logits"].reshape(b, k, n, 20)


def ae_decoder_probe(model: torch.nn.Module, ae: torch.nn.Module, fm: Any, samples: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    subset = samples[: min(4, len(samples))]
    batch = s10.collate_variable(subset, device)
    with torch.no_grad():
        exact_logits = decode_exact_ae(ae, batch)
    target = batch["binder_seq_shared"]
    mask = batch["binder_seq_mask"]
    exact_state_id = sequence_identity_from_logits(exact_logits, target[:, None, :].expand_as(exact_logits[..., 0]), batch["state_mask"]).detach()
    total, losses, nn_out, work_batch = forward_loss_stage11(model, ae, fm, subset, device, seed=1101)
    ae_state_id = sequence_identity_from_logits(
        nn_out["ae_seq_logits_states"],
        work_batch["binder_seq_shared"][:, None, :].expand_as(nn_out["ae_seq_logits_states"][..., 0]),
        work_batch["state_mask"],
    ).detach()
    shared_id = sequence_identity_from_logits(nn_out["seq_logits_shared"], work_batch["binder_seq_shared"], mask.to(device)).detach()
    return {
        "status": "passed",
        "loss_total": float(total.detach().cpu().item()),
        "losses": s4.summarize_losses(losses),
        "exact_ae_state_identity_mean": float(exact_state_id.mean().cpu().item()),
        "sampled_clean_ae_state_identity_mean": float(ae_state_id.mean().cpu().item()),
        "shared_head_identity_mean": float(shared_id.mean().cpu().item()),
        "shapes": {k: list(v.shape) for k, v in nn_out.items() if torch.is_tensor(v)},
        "flow_gate_mean": float(nn_out["flow_gate"].detach().mean().cpu().item()),
        "flow_gate_std": float(nn_out["flow_gate"].detach().std().cpu().item()),
    }


def set_trainable_stage11(model: torch.nn.Module, phase: str) -> dict[str, Any]:
    for p in model.parameters():
        p.requires_grad = False
    if phase == "overfit":
        prefixes = ["ALL_PARAMETERS"]
        for p in model.parameters():
            p.requires_grad = True
    else:
        prefixes = [
            "soft_sequence_feedback_projector",
            "flow_gate_head",
            "shared_seq_head",
            "state_seq_head",
            "shared_seq_consensus_gate",
            "state_seq_condition_projector",
            "state_condition_projector",
            "state_token_norm",
            "local_latents_linear",
            "ca_linear",
            "interface_quality_head",
        ]
        if phase == "seq_latent":
            prefixes = [
                "soft_sequence_feedback_projector",
                "flow_gate_head",
                "shared_seq_head",
                "state_seq_head",
                "shared_seq_consensus_gate",
                "state_seq_condition_projector",
                "state_condition_projector",
                "state_token_norm",
                "local_latents_linear",
            ]
        if phase == "mini":
            prefixes += [f"transformer_layers.{i}" for i in range(10, 14)]
        for name, p in model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                p.requires_grad = True
    return {
        "phase": phase,
        "prefixes": prefixes,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
    }


def grad_probe(model: torch.nn.Module, ae: torch.nn.Module, fm: Any, samples: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    model.train()
    info = set_trainable_stage11(model, "mini")
    total, losses, _, _ = forward_loss_stage11(model, ae, fm, samples[: min(2, len(samples))], device, seed=1201)
    total.backward()
    params = dict(model.named_parameters())
    names = [
        "soft_sequence_feedback_projector.1.weight",
        "flow_gate_head.1.weight",
        "shared_seq_head.1.weight",
        "state_seq_head.1.weight",
        "local_latents_linear.1.weight",
        "ca_linear.1.weight",
    ]
    grad_norms = {}
    for name in names:
        p = params.get(name)
        grad_norms[name] = None if p is None or p.grad is None else float(p.grad.detach().norm().cpu().item())
    model.zero_grad(set_to_none=True)
    return {
        "status": "passed" if all(v is not None and v > 0 for v in grad_norms.values()) else "failed",
        "loss_total": float(total.detach().cpu().item()),
        "losses": s4.summarize_losses(losses),
        "trainable": info,
        "grad_norms": grad_norms,
    }


def train_loop(
    model: torch.nn.Module,
    ae: torch.nn.Module,
    fm: Any,
    samples: list[dict[str, Any]],
    device: torch.device,
    phase: str,
    steps: int,
    batch_size: int,
    eval_every: int,
    lr: float,
    run_dir: Path,
    trainable_phase: str,
    fixed_seed: int | None = None,
    ae_ca_source: str = "init",
    noise_repeats: int = 1,
) -> dict[str, Any]:
    model.train()
    trainable = set_trainable_stage11(model, trainable_phase)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    fixed_subset = samples[: min(batch_size, len(samples))]
    with torch.no_grad():
        init_total, init_losses, _, _ = forward_loss_stage11(model, ae, fm, fixed_subset, device, seed=777, ae_ca_source=ae_ca_source)
    init_value = float(init_total.detach().cpu().item())
    history = []
    started = time.time()
    for step in range(1, steps + 1):
        subset = [samples[((step - 1) * batch_size + j) % len(samples)] for j in range(batch_size)]
        opt.zero_grad(set_to_none=True)
        train_seed = fixed_seed if fixed_seed is not None else step * 991
        total, losses, _, _ = forward_loss_stage11(
            model,
            ae,
            fm,
            subset,
            device,
            seed=train_seed,
            ae_ca_source=ae_ca_source,
            noise_repeats=noise_repeats,
        )
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
                eval_total, eval_losses, eval_out, _ = forward_loss_stage11(model, ae, fm, fixed_subset, device, seed=777, ae_ca_source=ae_ca_source)
            row = {
                "step": step,
                "train_total": float(total.detach().cpu().item()),
                "eval_total": float(eval_total.detach().cpu().item()),
                "train_losses": s4.summarize_losses(losses),
                "eval_losses": s4.summarize_losses(eval_losses),
                "flow_gate_mean": float(eval_out["flow_gate"].detach().mean().cpu().item()),
                "flow_gate_std": float(eval_out["flow_gate"].detach().std().cpu().item()),
                "elapsed_sec": time.time() - started,
                "cuda_max_mem_gb": torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0,
            }
            print(json.dumps({"phase": phase, **row}, ensure_ascii=False), flush=True)
            history.append(row)
    with torch.no_grad():
        final_total, final_losses, _, _ = forward_loss_stage11(model, ae, fm, fixed_subset, device, seed=777, ae_ca_source=ae_ca_source)
    return {
        "phase": phase,
        "steps": steps,
        "batch_size": batch_size,
        "initial_eval_total": init_value,
        "initial_losses": s4.summarize_losses(init_losses),
        "final_eval_total": float(final_total.detach().cpu().item()),
        "drop_fraction": (init_value - float(final_total.detach().cpu().item())) / max(abs(init_value), 1e-8),
        "final_losses": s4.summarize_losses(final_losses),
        "history": history,
        "elapsed_sec": time.time() - started,
        "step_time_sec": (time.time() - started) / max(steps, 1),
        "cuda_max_mem_gb": torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0,
        "trainable": trainable,
        "ae_ca_source": ae_ca_source,
        "noise_repeats": noise_repeats,
    }


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--stage10-ckpt", type=Path, default=DEFAULT_STAGE10_CKPT)
    ap.add_argument("--ae-ckpt", type=Path, default=DEFAULT_AE_CKPT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--run-name", default="stage11_flow_sequence")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-val-samples", type=int, default=0)
    ap.add_argument("--overfit1-steps", type=int, default=120)
    ap.add_argument("--overfit4-steps", type=int, default=200)
    ap.add_argument("--mini-steps", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--ae-ca-source", choices=["init", "exact", "pred"], default="init")
    ap.add_argument("--noise-repeats", type=int, default=1)
    ap.add_argument("--mini-trainable-phase", default="seq_latent", choices=["seq_latent", "mini", "overfit"])
    ap.add_argument("--lambda-ae-seq", type=float, default=1.5)
    ap.add_argument("--lambda-seq-ae-consistency", type=float, default=0.15)
    ap.add_argument("--lambda-flow-gate-reg", type=float, default=0.03)
    ap.add_argument("--skip-training", action="store_true")
    args = ap.parse_args()

    s4.set_seed(args.seed)
    device = choose_device(args.device)
    samples, manifest = s10.load_dataset(args.dataset)
    train_samples = [s for s in samples if s.get("split") == "train"]
    val_samples = [s for s in samples if s.get("split") == "val"]
    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_samples = val_samples[: args.max_val_samples]
    if not train_samples:
        raise RuntimeError("No train samples selected")

    model, fm, model_meta = s4.build_model(device)
    ckpt_meta = load_strategy_checkpoint(model, args.stage10_ckpt)
    loss_cfg = configure_stage11_loss(fm)
    fm.cfg_exp.loss.multistate.lambda_ae_seq = args.lambda_ae_seq
    fm.cfg_exp.loss.multistate.lambda_seq_ae_consistency = args.lambda_seq_ae_consistency
    fm.cfg_exp.loss.multistate.lambda_flow_gate_reg = args.lambda_flow_gate_reg
    loss_cfg.update({
        "lambda_ae_seq": args.lambda_ae_seq,
        "lambda_seq_ae_consistency": args.lambda_seq_ae_consistency,
        "lambda_flow_gate_reg": args.lambda_flow_gate_reg,
        "ae_ca_source": args.ae_ca_source,
        "noise_repeats": args.noise_repeats,
    })
    ae = s5ae.load_autoencoder(args.ae_ckpt, device)
    for p in ae.parameters():
        p.requires_grad = False
    ae.eval()

    report_path = REPORT_DIR / f"{args.run_name}_results.json"
    run_dir = RUNS_DIR / args.run_name
    result: dict[str, Any] = {
        "status": "running",
        "run_name": args.run_name,
        "dataset": str(args.dataset),
        "device": str(device),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "manifest_summary": {k: v for k, v in manifest.items() if k != "samples"},
        "model_meta": model_meta,
        "stage10_ckpt_meta": ckpt_meta,
        "stage11_loss_cfg": loss_cfg,
        "probes": {},
        "training": {},
    }
    result["probes"]["ae_decoder"] = ae_decoder_probe(model, ae, fm, train_samples, device)
    result["probes"]["grad_route"] = grad_probe(model, ae, fm, train_samples, device)
    write_json(report_path, result)

    if not args.skip_training:
        result["training"]["overfit1"] = train_loop(model, ae, fm, train_samples[:1], device, "overfit1", args.overfit1_steps, 1, args.eval_every, 2e-4, run_dir, "overfit", fixed_seed=999, ae_ca_source=args.ae_ca_source, noise_repeats=args.noise_repeats)
        model, fm, _ = s4.build_model(device)
        load_strategy_checkpoint(model, args.stage10_ckpt)
        configure_stage11_loss(fm)
        fm.cfg_exp.loss.multistate.lambda_ae_seq = args.lambda_ae_seq
        fm.cfg_exp.loss.multistate.lambda_seq_ae_consistency = args.lambda_seq_ae_consistency
        fm.cfg_exp.loss.multistate.lambda_flow_gate_reg = args.lambda_flow_gate_reg
        result["training"]["overfit4"] = train_loop(model, ae, fm, train_samples[: min(4, len(train_samples))], device, "overfit4", args.overfit4_steps, min(args.batch_size, 4), args.eval_every, 1e-4, run_dir, args.mini_trainable_phase, fixed_seed=1234, ae_ca_source=args.ae_ca_source, noise_repeats=args.noise_repeats)
        model, fm, _ = s4.build_model(device)
        load_strategy_checkpoint(model, args.stage10_ckpt)
        configure_stage11_loss(fm)
        fm.cfg_exp.loss.multistate.lambda_ae_seq = args.lambda_ae_seq
        fm.cfg_exp.loss.multistate.lambda_seq_ae_consistency = args.lambda_seq_ae_consistency
        fm.cfg_exp.loss.multistate.lambda_flow_gate_reg = args.lambda_flow_gate_reg
        result["training"]["mini"] = train_loop(model, ae, fm, train_samples, device, "mini", args.mini_steps, args.batch_size, args.eval_every, 5e-5, run_dir, args.mini_trainable_phase, fixed_seed=None, ae_ca_source=args.ae_ca_source, noise_repeats=args.noise_repeats)
        if val_samples:
            model.eval()
            with torch.no_grad():
                val_total, val_losses, _, _ = forward_loss_stage11(model, ae, fm, val_samples[: min(args.batch_size, len(val_samples))], device, seed=888, ae_ca_source=args.ae_ca_source)
            result["validation"] = {"total": float(val_total.detach().cpu().item()), "losses": s4.summarize_losses(val_losses)}

    result["status"] = "passed"
    write_json(report_path, result)
    print(json.dumps({"status": "passed", "report": str(report_path)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
