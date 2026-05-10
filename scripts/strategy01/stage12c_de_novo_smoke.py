#!/usr/bin/env python3
"""Stage12C target-only de-novo rollout smoke test.

This script performs an actual state-specific product-space rollout from
target-only noise and evaluates the final sampled states against held-out
multistate labels. The true binder sequence is used only as an evaluation
label, never as model input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import scripts.strategy01.stage04_real_complex_loss_debug as s4  # noqa: E402
import scripts.strategy01.stage10_pose_init_training as s10  # noqa: E402
import scripts.strategy01.stage11_flow_sequence_training as s11  # noqa: E402
import scripts.strategy01.stage12_de_novo_multistate_training as s12  # noqa: E402


DEFAULT_REPORT = REPO / "reports/strategy01/probes/stage12c_de_novo_smoke_results.json"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s4.jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def sequence_entropy(logits: torch.Tensor, mask: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1).clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1)
    value = (entropy * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
    return float(value.detach().cpu().item())


def state_seq_disagreement(state_logits: torch.Tensor, state_mask: torch.Tensor) -> float:
    probs = torch.softmax(state_logits, dim=-1)
    valid = state_mask.float()
    mean = (probs * valid[..., None]).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)[..., None]
    sq = ((probs - mean) ** 2).sum(dim=-1)
    value = (sq * valid).sum() / valid.sum().clamp_min(1.0)
    return float(value.detach().cpu().item())


def target_binder_geometry_proxy(batch: dict[str, torch.Tensor], x_states: dict[str, torch.Tensor]) -> dict[str, float]:
    """Target-only geometry proxy for de-novo candidates.

    It uses target coordinates and generated binder CA only.  No exact binder pose
    or interface labels are read, so this is safe for candidate selection.
    Units are nanometers because Strategy01 tensors follow Complexa internals.
    """
    if "x_target_states" not in batch or "target_mask_states" not in batch:
        return {}
    target_ca = batch["x_target_states"].to(device=x_states["bb_ca"].device)[..., 1, :]
    target_mask = batch["target_mask_states"].to(device=x_states["bb_ca"].device).bool()[..., 1]
    binder_ca = x_states["bb_ca"]
    binder_mask = batch["state_mask"].to(device=binder_ca.device).bool()
    b, k, nb, _ = binder_ca.shape
    rows = []
    hotspot = batch.get("target_hotspot_mask_states")
    hotspot_mask = hotspot.to(device=binder_ca.device).bool() if hotspot is not None else target_mask
    for bi in range(b):
        for si in range(k):
            tmask = target_mask[bi, si]
            bmask = binder_mask[bi, si]
            if not bool(tmask.any()) or not bool(bmask.any()):
                continue
            dist = torch.cdist(target_ca[bi, si, tmask].float(), binder_ca[bi, si, bmask].float())
            min_dist = float(dist.min().detach().cpu().item())
            contact_count = int((dist < 1.0).sum().detach().cpu().item())
            severe_clash = min_dist < 0.28
            hmask = hotspot_mask[bi, si] & tmask
            hotspot_contacts = 0
            if bool(hmask.any()):
                hdist = torch.cdist(target_ca[bi, si, hmask].float(), binder_ca[bi, si, bmask].float())
                hotspot_contacts = int((hdist < 1.0).sum().detach().cpu().item())
            rows.append((min_dist, contact_count, hotspot_contacts, severe_clash))
    if not rows:
        return {}
    return {
        "target_min_distance_nm_mean": float(sum(r[0] for r in rows) / len(rows)),
        "target_min_distance_nm_min": float(min(r[0] for r in rows)),
        "target_contact_count_mean": float(sum(r[1] for r in rows) / len(rows)),
        "target_hotspot_contact_count_mean": float(sum(r[2] for r in rows) / len(rows)),
        "target_severe_clash_rate": float(sum(1 for r in rows if r[3]) / len(rows)),
    }


def final_x_ae_identity(
    ae: torch.nn.Module,
    x_states: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: torch.Tensor,
) -> dict[str, float]:
    """Decode the actually integrated final sample, matching native Complexa generation."""
    local_latents = x_states["local_latents"]
    bb_ca = x_states["bb_ca"]
    state_mask = batch["state_mask"].to(device=local_latents.device).bool()
    target = batch["binder_seq_shared"].to(device=local_latents.device).long()
    binder_mask = batch["binder_seq_mask"].to(device=local_latents.device).bool()
    b, k, n, z = local_latents.shape
    dec = ae.decode(
        z_latent=local_latents.reshape(b * k, n, z),
        ca_coors_nm=bb_ca.reshape(b * k, n, 3),
        mask=state_mask.reshape(b * k, n),
    )
    state_logits = dec["seq_logits"].reshape(b, k, n, 20)
    shared_logits = (state_logits * weights[:, :, None, None]).sum(dim=1)
    state_identity = s11.sequence_identity_from_logits(
        state_logits,
        target[:, None, :].expand_as(state_logits[..., 0]),
        state_mask,
    )
    shared_identity = s11.sequence_identity_from_logits(shared_logits, target, binder_mask)
    return {
        "final_x_ae_shared_identity_mean": float(shared_identity.detach().mean().cpu().item()),
        "final_x_ae_state_identity_mean": float(state_identity.detach().mean().cpu().item()),
    }


def rollout_final(
    model: torch.nn.Module,
    ae: torch.nn.Module,
    fm: Any,
    samples: list[dict[str, Any]],
    device: torch.device,
    sampling_cfg: dict[str, Any],
    nsteps: int,
    seed: int,
    local_latents_stop_t: float | None = None,
    target_shell_max_center_distance_nm: float = 0.0,
    stage13_native_state_path: bool = False,
    stage14_enable_bounded_latent_repair: bool = False,
    stage14_latent_repair_max_norm: float = 0.25,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    batch = s10.collate_variable(samples, device)
    work = s12.make_de_novo_batch(batch)
    if stage13_native_state_path:
        work["stage13_native_state_path"] = True
    work = fm.corrupt_multistate_batch(work)
    state_mask = work["state_mask"].to(device=device).bool()
    weights = s12.normalize_state_weights(work, device)
    x_states = {dm: value.detach().clone() for dm, value in work["x_0_states"].items()}
    b, k, n = state_mask.shape
    ts, gt = fm.sample_schedule(nsteps=nsteps, sampling_model_args=sampling_cfg)
    ts = {dm: value.to(device) for dm, value in ts.items()}
    gt = {dm: value.to(device) for dm, value in gt.items()}
    schedule_steps = min(int(value.numel()) - 1 for value in ts.values())
    diagnostics: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        x_sc_states: dict[str, torch.Tensor] | None = None
        for step in range(schedule_steps):
            work["x_t_states"] = {dm: value.detach() for dm, value in x_states.items()}
            work["t_states"] = {dm: ts[dm][step].expand(b, k).to(device) for dm in fm.data_modes}
            if x_sc_states is not None:
                work["x_sc_states"] = {dm: value.detach() for dm, value in x_sc_states.items()}
            else:
                work.pop("x_sc_states", None)
            if stage13_native_state_path:
                work["stage13_native_state_path"] = True
            if stage14_enable_bounded_latent_repair:
                work["stage14_enable_bounded_latent_repair"] = True
                work["stage14_latent_repair_max_norm"] = float(stage14_latent_repair_max_norm)
            s12.add_weighted_legacy_fields(work, weights)
            work["de_novo_multistate_mode"] = True
            nn_out = model(work)
            if step in {0, schedule_steps // 2, schedule_steps - 1}:
                diag_out = s11.attach_ae_sequence_logits(ae, fm, work, dict(nn_out), ca_source="pred")
                diag_identity = s12.identity_metrics(diag_out, work)
                arch_debug = nn_out.get("arch_debug", {})
                diagnostics.append(
                    {
                        "step": step,
                        "t": {dm: float(ts[dm][step].detach().cpu().item()) for dm in fm.data_modes},
                        "shared_entropy": sequence_entropy(nn_out["seq_logits_shared"], work["binder_seq_mask"]),
                        "state_seq_disagreement": state_seq_disagreement(nn_out["state_seq_logits"], work["state_mask"]),
                        "identity": diag_identity,
                        "early_sequence_feedback_norm_mean": float(
                            arch_debug.get("early_sequence_feedback_norm_mean", torch.tensor(0.0)).detach().cpu().item()
                        ),
                        "sequence_feedback_norm_mean": float(
                            arch_debug.get("sequence_feedback_norm_mean", torch.tensor(0.0)).detach().cpu().item()
                        ),
                        "latent_sequence_repair_gate_mean": float(
                            arch_debug.get("latent_sequence_repair_gate_mean", torch.tensor(0.0)).detach().cpu().item()
                        ),
                        "latent_sequence_repair_norm_mean": float(
                            arch_debug.get("latent_sequence_repair_norm_mean", torch.tensor(0.0)).detach().cpu().item()
                        ),
                    }
                )
            updated: dict[str, torch.Tensor] = {}
            next_x_sc_states: dict[str, torch.Tensor] = {}
            for dm in fm.data_modes:
                if (
                    local_latents_stop_t is not None
                    and dm == "local_latents"
                    and float(ts[dm][step].detach().cpu().item()) >= float(local_latents_stop_t)
                ):
                    updated[dm] = x_states[dm].detach()
                    continue
                flat_x = x_states[dm].reshape(b * k, n, x_states[dm].shape[-1])
                flat_mask = state_mask.reshape(b * k, n)
                flat_t = ts[dm][step].expand(b * k).to(device)
                dt = ts[dm][step + 1] - ts[dm][step]
                param = fm.cfg_exp.nn.output_parameterization[dm]
                flat_nn = {param: nn_out[f"{dm}_states"].reshape(b * k, n, x_states[dm].shape[-1])}
                base = fm.base_flow_matchers[dm]
                flat_nn = base.nn_out_add_clean_sample_prediction(flat_x, flat_t, flat_mask, flat_nn)
                next_x_sc_states[dm] = flat_nn["x_1"].reshape(b, k, n, x_states[dm].shape[-1]).detach()
                flat_nn = base.nn_out_add_simulation_tensor(flat_x, flat_t, flat_mask, flat_nn)
                flat_nn = base.nn_out_add_guided_simulation_tensor(flat_nn, None, None, guidance_w=1.0, ag_ratio=0.0)
                flat_next = base.simulation_step(
                    x_t=flat_x,
                    nn_out=flat_nn,
                    t=flat_t,
                    dt=dt,
                    gt=gt[dm][step],
                    mask=flat_mask,
                    simulation_step_params=sampling_cfg[dm]["simulation_step_params"],
                )
                state_next = flat_next.reshape(b, k, n, x_states[dm].shape[-1]) * state_mask[..., None]
                if dm == "bb_ca" and target_shell_max_center_distance_nm > 0.0:
                    state_next, _ = s12.project_bb_ca_to_target_shell(work, state_next, state_mask, target_shell_max_center_distance_nm)
                updated[dm] = state_next
            x_states = updated
            x_sc_states = next_x_sc_states
        final = s12.make_de_novo_batch(batch)
        if stage13_native_state_path:
            final["stage13_native_state_path"] = True
        if stage14_enable_bounded_latent_repair:
            final["stage14_enable_bounded_latent_repair"] = True
            final["stage14_latent_repair_max_norm"] = float(stage14_latent_repair_max_norm)
        final["x_0_states"] = work["x_0_states"]
        final["x_t_states"] = {dm: value.detach() for dm, value in x_states.items()}
        if x_sc_states is not None:
            final["x_sc_states"] = {dm: value.detach() for dm, value in x_sc_states.items()}
        final["t_states"] = {
            dm: torch.clamp(
                ts[dm][-1].expand(b, k).to(device),
                max=min(0.95, float(local_latents_stop_t)) if (dm == "local_latents" and local_latents_stop_t is not None) else 0.95,
            )
            for dm in fm.data_modes
        }
        final["state_mask"] = state_mask
        final["stage12_primary_state_tensors"] = True
        s12.add_weighted_legacy_fields(final, weights)
        final_out = model(final)
        final_out = s11.attach_ae_sequence_logits(ae, fm, final, final_out, ca_source="pred")
        losses = fm.compute_multistate_loss(final, final_out)
        identity = s12.identity_metrics(final_out, final)
        final_x_identity = final_x_ae_identity(ae, x_states, final, weights)
        target_geometry_proxy = target_binder_geometry_proxy(final, x_states)
    return {
        "losses": s4.summarize_losses(losses),
        "identity": identity,
        "final_x_identity": final_x_identity,
        "target_geometry_proxy": target_geometry_proxy,
        "diagnostics": diagnostics,
        "nsteps": nsteps,
        "sample_count": len(samples),
        "stage13_native_state_path": bool(stage13_native_state_path),
        "stage14_enable_bounded_latent_repair": bool(stage14_enable_bounded_latent_repair),
        "stage14_latent_repair_max_norm": float(stage14_latent_repair_max_norm),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Stage12C target-only de-novo rollout smoke")
    parser.add_argument("--dataset", type=Path, default=s12.DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ae-ckpt", type=Path, default=s12.DEFAULT_AE_CKPT)
    parser.add_argument("--sampling-config", type=Path, default=s12.DEFAULT_SAMPLING_CONFIG)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--nsteps", type=int, default=16)
    parser.add_argument(
        "--local-latents-stop-t",
        type=float,
        default=None,
        help="Optional sampling diagnostic: stop local_latents updates after this t while bb_ca keeps flowing.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1207)
    parser.add_argument(
        "--target-shell-max-center-distance-nm",
        type=float,
        default=0.0,
        help="Target-only diagnostic guidance: rigidly translate binder states back inside this target/hotspot center shell.",
    )
    parser.add_argument(
        "--stage13-native-state-path",
        action="store_true",
        help="Diagnostic only: run K state tensors through the native Complexa target-concat denoiser state-wise.",
    )
    parser.add_argument(
        "--stage14-enable-bounded-latent-repair",
        action="store_true",
        help="Use the Stage14 bounded clean latent repair output for final AE sequence diagnostics.",
    )
    parser.add_argument("--stage14-latent-repair-max-norm", type=float, default=0.25)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = s12.choose_device(args.device)
    samples, manifest = s10.load_dataset(args.dataset)
    selected = [s for s in samples if s.get("split") == args.split][: args.max_samples]
    if not selected:
        raise RuntimeError(f"No samples selected for split={args.split}")
    stack_args = argparse.Namespace(
        init_ckpt=args.checkpoint,
        ae_ckpt=args.ae_ckpt,
        target_center_noise_scale_nm=0.20,
        latent_state_residual_noise_scale=0.05,
        lambda_ae_seq=1.5,
        lambda_seq_ae_consistency=0.15,
        lambda_flow_gate_reg=0.03,
        ae_seq_hard_state_alpha=0.0,
        ae_seq_hard_state_gamma=0.0,
    )
    model, fm, ae, model_meta, ckpt_meta, loss_cfg = s12.build_model_stack(device, stack_args)
    sampling_cfg = s12.load_sampling_cfg(args.sampling_config)
    result = {
        "stage": "stage12c_de_novo_smoke",
        "status": "running",
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "split": args.split,
        "sample_count": len(selected),
        "manifest_summary": {k: v for k, v in manifest.items() if k != "samples"},
        "model_meta": model_meta,
        "checkpoint_meta": ckpt_meta,
        "loss_cfg": loss_cfg,
        "contract": {
            "de_novo_multistate_mode": True,
            "forbidden_inputs": s12.FORBIDDEN_MODEL_INPUT_KEYS + ["optional_real_ca_feature", "optional_real_residue_type_feature"],
            "primary_outputs": ["bb_ca_states", "local_latents_states", "shared_seq_logits"],
            "legacy_average_outputs": "not used for main smoke evaluation",
            "stage13_native_state_path": bool(args.stage13_native_state_path),
        },
    }
    smoke = rollout_final(
        model,
        ae,
        fm,
        selected,
        device,
        sampling_cfg,
        args.nsteps,
        args.seed,
        local_latents_stop_t=args.local_latents_stop_t,
        target_shell_max_center_distance_nm=args.target_shell_max_center_distance_nm,
        stage13_native_state_path=args.stage13_native_state_path,
        stage14_enable_bounded_latent_repair=args.stage14_enable_bounded_latent_repair,
        stage14_latent_repair_max_norm=args.stage14_latent_repair_max_norm,
    )
    result["local_latents_stop_t"] = args.local_latents_stop_t
    result["target_shell_max_center_distance_nm"] = args.target_shell_max_center_distance_nm
    result.update(smoke)
    result["status"] = "passed"
    write_json(args.report_json, result)
    print(json.dumps({"status": "passed", "report": str(args.report_json)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
