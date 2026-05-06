#!/usr/bin/env python
"""Stage09 sampling-time interface guidance for Strategy01.

This is the first real integration of interface guidance into the state-specific
flow sampling loop. It updates `bb_ca_states` during sampling with differentiable
anchor/contact, clash-repulsion, and interface-shell energies. The default label
source is V_exact oracle anchors, so this script is a diagnostic implementation,
not a fair leaderboard result. The same code path is designed to later accept
non-oracle persistent anchors mined from target ensembles or predicted source
interfaces.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from openfold.np import residue_constants

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from proteinfoundation.proteina import Proteina  # noqa: E402
import scripts.strategy01.stage04_real_complex_loss_debug as s4  # noqa: E402
import scripts.strategy01.stage05_extract_ae_latents as s5ae  # noqa: E402
import scripts.strategy01.stage07_multistate_sampling_smoke as s7  # noqa: E402
from scripts.strategy01.stage08b_exact_geometry_benchmark import parse_ca  # noqa: E402

AA = residue_constants.restypes
DEFAULT_CKPT = REPO / "ckpts/stage07_sequence_consensus/runs/stage08b_merged_pilot/mini_final_lightning.ckpt"
DEFAULT_AE = REPO / "ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt"
DEFAULT_DATASET = REPO / "data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt"
DEFAULT_SAMPLING = REPO / "configs/pipeline/model_sampling.yaml"
DEFAULT_OUT = REPO / "results/strategy01/stage09_guided_state_sampling"
DEFAULT_REPORT = REPO / "reports/strategy01/probes/stage09_guided_state_sampling_summary.json"


@dataclass
class StateGuidanceLabels:
    target_ca_nm: torch.Tensor
    ref_binder_ca_nm: torch.Tensor
    anchor_target_idx: torch.Tensor
    anchor_binder_idx: torch.Tensor
    anchor_dist_nm: torch.Tensor


def as_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s7.tensor_to_json(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def load_samples(path: Path) -> list[dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["samples"]


def select_samples(samples: list[dict[str, Any]], split: str, sample_ids: set[str] | None, max_samples: int) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    for idx, sample in enumerate(samples):
        if sample_ids is not None and sample.get("sample_id") not in sample_ids:
            continue
        if split != "any" and sample.get("split") != split:
            continue
        if int(sample["state_present_mask"].bool().sum().item()) < 2:
            continue
        selected.append((idx, sample))
        if max_samples > 0 and len(selected) >= max_samples:
            break
    if not selected:
        raise RuntimeError("No samples selected for guided sampling")
    return selected


class PlainStrategySamplingModel(torch.nn.Module):
    """Minimal Proteina-like wrapper for Strategy01 plain model_state_dict checkpoints."""

    def __init__(self, nn: torch.nn.Module, fm: Any):
        super().__init__()
        self.nn = nn
        self.fm = fm
        self.cfg_exp = fm.cfg_exp


def load_sampling_model(ckpt_path: Path, autoencoder_ckpt: Path, device: torch.device) -> torch.nn.Module:
    try:
        model = Proteina.load_from_checkpoint(str(ckpt_path), strict=False, autoencoder_ckpt_path=str(autoencoder_ckpt), map_location="cpu")
        return model.to(device).eval()
    except Exception as exc:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise RuntimeError(f"Failed to load checkpoint as Lightning or plain model_state_dict: {ckpt_path}") from exc
        nn, fm, _ = s4.build_model(torch.device("cpu"))
        state = payload["model_state_dict"]
        model_state = nn.state_dict()
        compatible = {k: v for k, v in state.items() if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)}
        missing_or_bad = [k for k, v in state.items() if k in model_state and tuple(v.shape) != tuple(model_state[k].shape)]
        info = nn.load_state_dict(compatible, strict=False)
        if missing_or_bad:
            print(json.dumps({"plain_ckpt_warning": "shape_mismatch", "keys": missing_or_bad[:20]}, ensure_ascii=False), flush=True)
        print(json.dumps({
            "plain_ckpt_loaded": str(ckpt_path),
            "compatible_keys": len(compatible),
            "missing_keys_count": len(info.missing_keys),
            "unexpected_keys_count": len(info.unexpected_keys),
        }, ensure_ascii=False), flush=True)
        model = PlainStrategySamplingModel(nn, fm)
        return model.to(device).eval()


def contact_pairs(target_ca_nm: torch.Tensor, binder_ca_nm: torch.Tensor, cutoff_nm: float, max_pairs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = torch.cdist(target_ca_nm.float(), binder_ca_nm.float())
    idx = torch.nonzero(d <= cutoff_nm, as_tuple=False)
    if idx.numel() == 0:
        flat = torch.argsort(d.flatten())[:max_pairs]
        ti = (flat // d.shape[1]).long()
        bj = (flat % d.shape[1]).long()
    else:
        dist = d[idx[:, 0], idx[:, 1]]
        order = torch.argsort(dist)[:max_pairs]
        picked = idx[order]
        ti = picked[:, 0].long()
        bj = picked[:, 1].long()
    rd = d[ti, bj].float().clamp(min=0.35, max=cutoff_nm)
    return ti, bj, rd


def _state_target_and_binder_ca_nm(
    sample: dict[str, Any],
    state_index: int,
    target_chain: str,
    binder_chain: str,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    exact_paths = sample.get("exact_complex_paths") or sample.get("predicted_complex_paths") or []
    target_paths = sample.get("target_state_paths") or []
    target_path = as_path(target_paths[state_index])
    complex_path = as_path(exact_paths[state_index])
    target_rows = parse_ca(target_path, target_chain)
    if len(target_rows) < 3:
        target_rows = parse_ca(complex_path, target_chain)
    binder_rows = parse_ca(complex_path, binder_chain)
    if len(target_rows) < 3 or len(binder_rows) < 3:
        return None, None
    target_ca_nm = torch.stack([xyz for _, xyz in target_rows]).float().to(device) / 10.0
    binder_ca_nm = torch.stack([xyz for _, xyz in binder_rows]).float().to(device) / 10.0
    return target_ca_nm, binder_ca_nm


def build_guidance_labels(
    sample: dict[str, Any],
    device: torch.device,
    contact_cutoff_nm: float,
    max_pairs: int,
    anchor_source: str = "oracle",
) -> list[StateGuidanceLabels | None]:
    """Build state guidance labels.

    anchor_source modes:
    - oracle: use each state's exact complex contacts. Diagnostic upper bound.
    - source_state0: use only state0/source-complex binder residue anchors and
      transfer them to every target state by residue index. This approximates a
      production scenario where one source interface is known and compatibility
      with other conformations is required.
    """
    valid_k = int(sample["state_present_mask"].bool().sum().item())
    target_chain = sample.get("target_chain_id") or "A"
    binder_chain = sample.get("binder_chain_id") or "B"
    labels: list[StateGuidanceLabels | None] = []

    source_ti = source_bj = source_rd = None
    source_binder_ca_nm = None
    if anchor_source == "source_state0":
        source_target_ca_nm, source_binder_ca_nm = _state_target_and_binder_ca_nm(sample, 0, target_chain, binder_chain, device)
        if source_target_ca_nm is not None and source_binder_ca_nm is not None:
            source_ti, source_bj, source_rd = contact_pairs(source_target_ca_nm, source_binder_ca_nm, contact_cutoff_nm, max_pairs)

    for k in range(valid_k):
        target_ca_nm, ref_binder_ca_nm = _state_target_and_binder_ca_nm(sample, k, target_chain, binder_chain, device)
        if target_ca_nm is None or ref_binder_ca_nm is None:
            labels.append(None)
            continue
        if anchor_source == "oracle":
            ti, bj, rd = contact_pairs(target_ca_nm, ref_binder_ca_nm, contact_cutoff_nm, max_pairs)
        elif anchor_source == "source_state0":
            if source_ti is None or source_bj is None or source_rd is None or source_binder_ca_nm is None:
                labels.append(None)
                continue
            keep = (source_ti < target_ca_nm.shape[0]) & (source_bj < source_binder_ca_nm.shape[0])
            ti, bj, rd = source_ti[keep], source_bj[keep], source_rd[keep]
            if ti.numel() == 0:
                labels.append(None)
                continue
            ref_binder_ca_nm = source_binder_ca_nm
        else:
            raise ValueError(f"Unknown anchor_source={anchor_source}")
        labels.append(
            StateGuidanceLabels(
                target_ca_nm=target_ca_nm,
                ref_binder_ca_nm=ref_binder_ca_nm,
                anchor_target_idx=ti.to(device),
                anchor_binder_idx=bj.to(device),
                anchor_dist_nm=rd.to(device),
            )
        )
    return labels


def contact_f1_and_clash(target_ca_nm: torch.Tensor, ref_binder_ca_nm: torch.Tensor, pred_ca_nm: torch.Tensor, cutoff_nm: float, clash_cutoff_nm: float) -> tuple[float, bool, float]:
    d_ref = torch.cdist(target_ca_nm.float(), ref_binder_ca_nm.float())
    d_pred = torch.cdist(target_ca_nm.float(), pred_ca_nm.float())
    ref = d_ref <= cutoff_nm
    pred = d_pred <= cutoff_nm
    inter = (ref & pred[:, : ref.shape[1]]).sum().float() if pred.shape[1] >= ref.shape[1] else (ref[:, : pred.shape[1]] & pred).sum().float()
    pred_count = pred.sum().float().clamp_min(1.0)
    ref_count = ref.sum().float().clamp_min(1.0)
    precision = inter / pred_count
    recall = inter / ref_count
    f1 = 0.0 if float(precision + recall) <= 0 else float((2.0 * precision * recall / (precision + recall)).detach().cpu().item())
    min_d = float(d_pred.min().detach().cpu().item()) if d_pred.numel() else math.inf
    return f1, min_d < clash_cutoff_nm, min_d


def ae_consensus_logits(ae: torch.nn.Module, x_states: dict[str, torch.Tensor], state_mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    lat = x_states["local_latents"]
    bb = x_states["bb_ca"].to(dtype=lat.dtype)
    b, k, n, z = lat.shape
    dec = ae.decode(
        z_latent=lat.reshape(b * k, n, z),
        ca_coors_nm=bb.reshape(b * k, n, 3),
        mask=state_mask.reshape(b * k, n).bool(),
    )
    logits = dec["seq_logits"].reshape(b, k, n, 20)
    return (logits * weights[:, :, None, None].to(dtype=logits.dtype)).sum(dim=1)


def choose_sequence_logits(final_nn_out: dict[str, torch.Tensor], ae: torch.nn.Module | None, x_states: dict[str, torch.Tensor], state_mask: torch.Tensor, weights: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    shared = final_nn_out["seq_logits_shared"]
    if args.sequence_source == "shared_head":
        return shared
    if ae is None:
        raise RuntimeError("sequence_source requires autoencoder, but ae is None")
    ae_logits = ae_consensus_logits(ae, x_states, state_mask, weights)
    if args.sequence_source == "ae_consensus":
        return ae_logits
    if args.sequence_source == "hybrid":
        w = float(args.sequence_hybrid_ae_weight)
        return (1.0 - w) * shared + w * ae_logits
    raise ValueError(f"Unknown sequence_source={args.sequence_source}")


def safe_accept_model_flow_update(
    before_flat: torch.Tensor,
    after_flat: torch.Tensor,
    labels_batch: list[list[StateGuidanceLabels | None]],
    state_mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reject learned model-flow bb_ca updates that damage source-anchor geometry.

    This is deliberately source-anchor based, not exact benchmark based.  The goal
    is to let learned flow refine flexible regions while preventing it from
    destroying the source/interface pose that Stage10C showed was the reliable
    geometric prior.
    """
    if not getattr(args, "gate_safe_accept", True):
        return after_flat, {"type": "model_flow_safe_accept", "applied": False, "accepted": 0, "rejected": 0}
    b, k, n = state_mask.shape
    out = after_flat.clone()
    accepted = 0
    rejected = 0
    final_logs: list[dict[str, Any]] = []
    for bi in range(b):
        for ki in range(k):
            label = labels_batch[bi][ki]
            flat_idx = bi * k + ki
            if label is None:
                accepted += 1
                continue
            mask = state_mask[bi, ki].bool()
            before = before_flat[flat_idx]
            after = after_flat[flat_idx]
            before_f1, before_clash, before_min_d = contact_f1_and_clash(
                label.target_ca_nm,
                label.ref_binder_ca_nm,
                before[: min(before.shape[0], label.ref_binder_ca_nm.shape[0])],
                args.contact_cutoff_nm,
                args.clash_cutoff_nm,
            )
            after_f1, after_clash, after_min_d = contact_f1_and_clash(
                label.target_ca_nm,
                label.ref_binder_ca_nm,
                after[: min(after.shape[0], label.ref_binder_ca_nm.shape[0])],
                args.contact_cutoff_nm,
                args.clash_cutoff_nm,
            )
            reject_reason = None
            hard_min = float(args.hard_min_dist_nm)
            min_improved = after_min_d >= before_min_d + float(args.min_dist_improvement_nm)
            if after_min_d < hard_min and not min_improved:
                reject_reason = "model_flow_hard_min_distance_not_improved"
            elif (not before_clash) and after_clash:
                reject_reason = "model_flow_would_create_new_clash"
            elif before_clash and after_min_d + 1e-6 < before_min_d:
                reject_reason = "model_flow_existing_clash_became_closer"
            elif before_f1 - after_f1 > float(args.gate_f1_drop_tolerance):
                reject_reason = "model_flow_contact_f1_decreased"
            if reject_reason is not None:
                out[flat_idx] = before * mask[:, None] + after * (~mask)[:, None]
                rejected += 1
            else:
                accepted += 1
            final_logs.append({
                "batch": bi,
                "state": ki,
                "accepted": reject_reason is None,
                "reject_reason": reject_reason,
                "before_f1": before_f1,
                "after_f1": after_f1,
                "before_clash": before_clash,
                "after_clash": after_clash,
                "before_min_dist_nm": before_min_d,
                "after_min_dist_nm": after_min_d,
            })
    return out, {"type": "model_flow_safe_accept", "applied": True, "accepted": accepted, "rejected": rejected, "final_step_state_logs": final_logs}


def set_sampling_optional_features(batch: dict[str, Any], args: argparse.Namespace) -> None:
    """Expose current CA coordinates to baseline optional feature factories.

    Residue-type optional features intentionally stay disabled in de novo
    sampling: using `binder_seq_shared` would leak the exact reference sequence
    into B1. A future self-conditioning path can feed previous-step generated
    sequence tokens without using labels.
    """
    if args.enable_ca_feature:
        batch["ca_coors_nm"] = batch["x_t"]["bb_ca"].detach()
        batch["use_ca_coors_nm_feature"] = True
    else:
        batch["use_ca_coors_nm_feature"] = False
    batch["use_residue_type_feature"] = False


def guidance_energy(
    x_ca_nm: torch.Tensor,
    label: StateGuidanceLabels,
    valid_mask: torch.Tensor,
    contact_cutoff_nm: float,
    clash_cutoff_nm: float,
    shell_max_nm: float,
    max_pairs: int,
    clash_weight: float,
    shell_weight: float,
    tether_weight: float,
    overcontact_weight: float,
    clash_max_weight: float,
    contact_temperature_nm: float,
    clash_loss_mode: str,
    x_anchor: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    common_len = min(x_ca_nm.shape[0], label.ref_binder_ca_nm.shape[0])
    if common_len < 1:
        zero = x_ca_nm.sum() * 0.0
        return zero, {"anchor_pairs": 0.0, "anchor_loss": 0.0, "clash_loss": 0.0, "shell_loss": 0.0, "tether_loss": 0.0}
    bj = label.anchor_binder_idx
    ti = label.anchor_target_idx
    rd = label.anchor_dist_nm
    keep = bj < min(common_len, x_ca_nm.shape[0])
    bj, ti, rd = bj[keep], ti[keep], rd[keep]
    if bj.numel() == 0:
        ti, bj, rd = contact_pairs(label.target_ca_nm, label.ref_binder_ca_nm[:common_len], contact_cutoff_nm, max_pairs)
        bj, ti, rd = bj.to(x_ca_nm.device), ti.to(x_ca_nm.device), rd.to(x_ca_nm.device)
    pred_valid = x_ca_nm[:common_len]
    anchor_d = torch.linalg.norm(pred_valid[bj] - label.target_ca_nm[ti], dim=-1)
    anchor_loss = F.smooth_l1_loss(anchor_d, rd.clamp(min=0.35, max=contact_cutoff_nm))
    all_d = torch.cdist(label.target_ca_nm.float(), pred_valid.float())
    clash_violation = torch.relu(clash_cutoff_nm - all_d)
    if contact_temperature_nm < 0:
        raise ValueError("contact_temperature_nm must be non-negative")
    if clash_loss_mode == "mean":
        clash_active = clash_violation.pow(2).mean()
        clash_max = clash_violation.pow(2).max()
        clash_loss = clash_active
    elif clash_loss_mode == "active" and bool((clash_violation > 0).any()):
        clash_active = clash_violation[clash_violation > 0].pow(2).mean()
        clash_max = clash_violation.pow(2).max()
        clash_loss = clash_active + clash_max_weight * clash_max
    elif clash_loss_mode != "active":
        raise ValueError(f"Unknown clash_loss_mode={clash_loss_mode}")
    else:
        clash_active = all_d.sum() * 0.0
        clash_max = all_d.sum() * 0.0
        clash_loss = all_d.sum() * 0.0
    shell_loss = torch.relu(all_d.min(dim=0).values - shell_max_nm).pow(2).mean()
    ref_d = torch.cdist(label.target_ca_nm.float(), label.ref_binder_ca_nm[:common_len].float())
    ref_contact_count = (ref_d <= contact_cutoff_nm).float().sum().clamp_min(1.0)
    soft_contact_count = torch.sigmoid((contact_cutoff_nm - all_d) / max(contact_temperature_nm, 1e-4)).sum()
    overcontact_loss = torch.relu(soft_contact_count - 1.25 * ref_contact_count).pow(2) / ref_contact_count.pow(2)
    tether_loss = ((x_ca_nm[valid_mask] - x_anchor[valid_mask]) ** 2).mean() if int(valid_mask.sum()) else x_ca_nm.sum() * 0.0
    loss = anchor_loss + clash_weight * clash_loss + shell_weight * shell_loss + tether_weight * tether_loss + overcontact_weight * overcontact_loss
    return loss, {
        "anchor_pairs": float(bj.numel()),
        "anchor_loss": float(anchor_loss.detach().cpu().item()),
        "clash_loss": float(clash_loss.detach().cpu().item()),
        "clash_active_loss": float(clash_active.detach().cpu().item()),
        "clash_max_loss": float(clash_max.detach().cpu().item()),
        "shell_loss": float(shell_loss.detach().cpu().item()),
        "overcontact_loss": float(overcontact_loss.detach().cpu().item()),
        "soft_contact_count": float(soft_contact_count.detach().cpu().item()),
        "ref_contact_count": float(ref_contact_count.detach().cpu().item()),
        "tether_loss": float(tether_loss.detach().cpu().item()),
    }


def clamp_guidance_displacement(candidate: torch.Tensor, x0: torch.Tensor, mask: torch.Tensor, max_disp_nm: float) -> torch.Tensor:
    if max_disp_nm <= 0:
        return candidate
    delta = candidate - x0
    delta_norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-8)
    scale = torch.clamp(max_disp_nm / delta_norm, max=1.0)
    clamped = x0 + delta * scale
    return clamped * mask[:, None] + x0 * (~mask)[:, None]


def apply_sampling_guidance(
    bb_states: torch.Tensor,
    state_mask: torch.Tensor,
    labels_batch: list[list[StateGuidanceLabels | None]],
    step: int,
    nsteps: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if step < int(args.guidance_start_fraction * nsteps):
        return bb_states, {"applied": False, "reason": "before_guidance_start"}
    guided = bb_states.detach().clone()
    accepted = 0
    rejected = 0
    state_logs: list[dict[str, Any]] = []
    bsz, kmax, _, _ = bb_states.shape
    for b in range(bsz):
        for k in range(kmax):
            label = labels_batch[b][k] if k < len(labels_batch[b]) else None
            if label is None or int(state_mask[b, k].sum().item()) < 1:
                continue
            x0 = bb_states[b, k].detach()
            x = x0.clone().requires_grad_(True)
            mask = state_mask[b, k].bool()
            last_meta = {}
            for _ in range(max(1, args.guidance_inner_steps)):
                loss, meta = guidance_energy(
                    x,
                    label,
                    mask,
                    args.contact_cutoff_nm,
                    args.clash_cutoff_nm,
                    args.shell_max_nm,
                    args.max_anchor_pairs,
                    args.clash_weight,
                    args.shell_weight,
                    args.tether_weight,
                    args.overcontact_weight,
                    args.clash_max_weight,
                    args.contact_temperature_nm,
                    args.clash_loss_mode,
                    x0,
                )
                grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                step_size = float(args.guidance_lr) * (0.5 + 0.5 * step / max(1, nsteps - 1))
                x = (x - step_size * grad).detach().requires_grad_(True)
                x = x * mask[:, None] + x0 * (~mask)[:, None]
                last_meta = meta
            candidate = x.detach()
            candidate = clamp_guidance_displacement(candidate, x0, mask, args.max_guidance_displacement_nm)
            _, before_meta = guidance_energy(
                x0,
                label,
                mask,
                args.contact_cutoff_nm,
                args.clash_cutoff_nm,
                args.shell_max_nm,
                args.max_anchor_pairs,
                args.clash_weight,
                args.shell_weight,
                args.tether_weight,
                args.overcontact_weight,
                args.clash_max_weight,
                args.contact_temperature_nm,
                args.clash_loss_mode,
                x0,
            )
            _, after_meta = guidance_energy(
                candidate,
                label,
                mask,
                args.contact_cutoff_nm,
                args.clash_cutoff_nm,
                args.shell_max_nm,
                args.max_anchor_pairs,
                args.clash_weight,
                args.shell_weight,
                args.tether_weight,
                args.overcontact_weight,
                args.clash_max_weight,
                args.contact_temperature_nm,
                args.clash_loss_mode,
                x0,
            )
            before_f1, before_clash, before_min_d = contact_f1_and_clash(
                label.target_ca_nm,
                label.ref_binder_ca_nm,
                x0[: min(x0.shape[0], label.ref_binder_ca_nm.shape[0])],
                args.contact_cutoff_nm,
                args.clash_cutoff_nm,
            )
            after_f1, after_clash, after_min_d = contact_f1_and_clash(
                label.target_ca_nm,
                label.ref_binder_ca_nm,
                candidate[: min(candidate.shape[0], label.ref_binder_ca_nm.shape[0])],
                args.contact_cutoff_nm,
                args.clash_cutoff_nm,
            )
            accept = True
            reject_reason = None
            if args.safe_accept:
                hard_min = float(args.hard_min_dist_nm)
                min_improved = after_min_d >= before_min_d + float(args.min_dist_improvement_nm)
                f1_drop = before_f1 - after_f1
                anchor_worsened = after_meta["anchor_loss"] > before_meta["anchor_loss"] + float(args.anchor_loss_tolerance)
                if after_min_d < hard_min and not min_improved:
                    accept = False
                    reject_reason = "hard_min_distance_not_improved"
                elif (not before_clash) and after_clash:
                    accept = False
                    reject_reason = "would_create_new_clash"
                elif before_clash and after_min_d + 1e-6 < before_min_d:
                    accept = False
                    reject_reason = "existing_clash_became_closer"
                elif f1_drop > float(args.f1_drop_tolerance):
                    accept = False
                    reject_reason = "contact_f1_decreased"
                elif anchor_worsened and after_f1 < before_f1 + float(args.f1_gain_for_anchor_worsen):
                    accept = False
                    reject_reason = "anchor_loss_worsened_without_f1_gain"
            if accept:
                guided[b, k] = candidate * mask[:, None] + x0 * (~mask)[:, None]
                accepted += 1
            else:
                rejected += 1
            if step == nsteps - 1:
                state_logs.append(
                    {
                        "batch": b,
                        "state": k,
                        "accepted": accept,
                        "reject_reason": reject_reason,
                        "before_f1": before_f1,
                        "after_f1": after_f1,
                        "before_clash": before_clash,
                        "after_clash": after_clash,
                        "before_min_dist_nm": before_min_d,
                        "after_min_dist_nm": after_min_d,
                        "hard_min_dist_nm": args.hard_min_dist_nm,
                        "before_energy_meta": before_meta,
                        "after_energy_meta": after_meta,
                        "energy_meta": last_meta,
                    }
                )
    return guided * state_mask[..., None], {"applied": True, "accepted": accepted, "rejected": rejected, "final_step_state_logs": state_logs}


def guided_state_specific_sample(
    model: Proteina,
    batch: dict[str, Any],
    labels_batch: list[list[StateGuidanceLabels | None]],
    nsteps: int,
    sampling_cfg: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[dict[str, Any]]]:
    fm = model.fm
    model.eval()
    mask = batch["mask"].bool().to(device)
    state_mask = batch["state_mask"].bool().to(device)
    b, k, n = state_mask.shape
    weights = s7.normalize_weights(batch, device)
    x_states = {}
    for dm in fm.data_modes:
        dim = int(batch["x_1_states"][dm].shape[-1])
        base = fm.base_flow_matchers[dm]
        flat_mask = state_mask.reshape(b * k, n)
        # MODIFIED 2026-05-05 Stage10:
        # Optional pose-initialized B1 sampling.  If init_bb_ca_states is in
        # the batch, bb_ca trajectories start from transferred source poses;
        # otherwise Stage09 noise initialization is unchanged.
        if dm == "bb_ca" and "init_bb_ca_states" in batch:
            x0 = batch["init_bb_ca_states"].to(device=device, dtype=batch["x_1_states"][dm].dtype)
            x_states[dm] = x0 * state_mask[..., None]
        else:
            x0 = base.sample_noise(n=n, shape=(b * k,), device=device, mask=flat_mask, training=False)
            x_states[dm] = x0.reshape(b, k, n, dim) * state_mask[..., None]

    ts, gt = fm.sample_schedule(nsteps=nsteps, sampling_model_args=sampling_cfg)
    ts = {dm: val.to(device) for dm, val in ts.items()}
    gt = {dm: val.to(device) for dm, val in gt.items()}
    guidance_trace: list[dict[str, Any]] = []
    for step in range(nsteps):
        batch["x_t"] = s7.weighted_average_states(x_states, weights, mask)
        set_sampling_optional_features(batch, args)
        batch["t"] = {dm: ts[dm][step].expand(b).to(device) for dm in fm.data_modes}
        batch["x_sc"] = {dm: torch.zeros_like(batch["x_t"][dm]) for dm in fm.data_modes}
        with torch.no_grad():
            nn_out = model.nn(batch)
        updated = {}
        for dm in fm.data_modes:
            state_key = f"{dm}_states"
            param = model.cfg_exp.nn.output_parameterization[dm]
            flat_x = x_states[dm].reshape(b * k, n, x_states[dm].shape[-1])
            flat_mask = state_mask.reshape(b * k, n)
            flat_t = ts[dm][step].expand(b * k).to(device)
            dt = ts[dm][step + 1] - ts[dm][step]
            flat_nn = {param: nn_out[state_key].reshape(b * k, n, x_states[dm].shape[-1])}
            base = fm.base_flow_matchers[dm]
            flat_nn = base.nn_out_add_clean_sample_prediction(flat_x, flat_t, flat_mask, flat_nn)
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
            # MODIFIED 2026-05-05 Stage10B:
            # Pose-initialized multistate design should behave as bounded
            # refinement, not as a full re-generation that can destroy the
            # transferred source interface.  The default scale is 1.0, so old
            # Stage09 behavior is unchanged unless explicitly overridden.
            flow_scale = float(getattr(args, "flow_step_scale", 1.0))
            if dm == "bb_ca" and getattr(args, "bb_ca_flow_step_scale", None) is not None:
                flow_scale = float(args.bb_ca_flow_step_scale)
            if dm == "local_latents" and getattr(args, "local_latents_flow_step_scale", None) is not None:
                flow_scale = float(args.local_latents_flow_step_scale)
            if flow_scale != 1.0:
                flat_next = flat_x + flow_scale * (flat_next - flat_x)
            if dm == "bb_ca" and getattr(args, "use_learned_flow_gate", False) and "flow_gate" in nn_out:
                gate = nn_out["flow_gate"].to(device=flat_x.device, dtype=flat_x.dtype)
                gate = gate.reshape(b * k, n, 1).clamp(0.0, 1.0)
                flat_next = flat_x + gate * (flat_next - flat_x)
            max_flow_disp = float(getattr(args, "max_flow_displacement_nm", 0.0))
            if dm == "bb_ca" and getattr(args, "bb_ca_max_flow_displacement_nm", None) is not None:
                max_flow_disp = float(args.bb_ca_max_flow_displacement_nm)
            if dm == "local_latents" and getattr(args, "local_latents_max_flow_displacement_nm", None) is not None:
                max_flow_disp = float(args.local_latents_max_flow_displacement_nm)
            if max_flow_disp > 0.0:
                delta = flat_next - flat_x
                delta_norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-8)
                scale = torch.clamp(max_flow_disp / delta_norm, max=1.0)
                flat_next = flat_x + delta * scale
            if dm == "bb_ca" and getattr(args, "use_learned_flow_gate", False) and getattr(args, "gate_safe_accept", True):
                flat_next, gate_trace = safe_accept_model_flow_update(flat_x, flat_next, labels_batch, state_mask, args)
                if gate_trace.get("applied"):
                    gate_trace["step"] = step
                    guidance_trace.append(gate_trace)
            updated[dm] = flat_next.reshape(b, k, n, x_states[dm].shape[-1]) * state_mask[..., None]
        guided_bb, trace = apply_sampling_guidance(updated["bb_ca"], state_mask, labels_batch, step, nsteps, args)
        updated["bb_ca"] = guided_bb
        if trace.get("applied"):
            trace["step"] = step
            guidance_trace.append(trace)
        x_states = updated

    batch["x_t"] = s7.weighted_average_states(x_states, weights, mask)
    set_sampling_optional_features(batch, args)
    batch["t"] = {dm: torch.ones(b, device=device) for dm in fm.data_modes}
    batch["x_sc"] = {dm: torch.zeros_like(batch["x_t"][dm]) for dm in fm.data_modes}
    with torch.no_grad():
        final_nn_out = model.nn(batch)
    return x_states, final_nn_out, guidance_trace


def sample_one(model: Proteina, ae: torch.nn.Module | None, sample_idx: int, sample: dict[str, Any], sampling_cfg: Any, out_root: Path, nsteps: int, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "enable_init_pose", False):
        import scripts.strategy01.stage10_pose_init_training as s10  # local import avoids changing Stage09 default behavior
        batch = s10.collate_variable([sample], device)
    else:
        batch = s4.collate_samples([sample], device)
    labels = [build_guidance_labels(sample, device, args.contact_cutoff_nm, args.max_anchor_pairs, args.anchor_source)]
    started = time.time()
    x_states, nn_out, guidance_trace = guided_state_specific_sample(model, batch, labels, nsteps, sampling_cfg, device, args)
    elapsed = time.time() - started
    seq_logits = choose_sequence_logits(nn_out, ae, x_states, batch["state_mask"], s7.normalize_weights(batch, device), args)
    pred_seq = s7.seq_from_logits(seq_logits, batch["mask"])
    ref_seq = sample.get("shared_binder_sequence") or s7.seq_from_tokens(sample["binder_seq_shared"], sample["binder_seq_mask"])
    seq_identity = sum(a == b for a, b in zip(pred_seq, ref_seq)) / max(1, min(len(pred_seq), len(ref_seq)))
    sample_out = out_root / str(sample.get("sample_id", f"sample_{sample_idx}"))
    sample_out.mkdir(parents=True, exist_ok=True)
    (sample_out / "shared_sequence.fasta").write_text(f">strategy01_stage09_guided|{sample.get('sample_id')}\n{pred_seq}\n", encoding="utf-8")
    (sample_out / "reference_sequence.fasta").write_text(f">reference|{sample.get('sample_id')}\n{ref_seq}\n", encoding="utf-8")
    pred_ca = x_states["bb_ca"][0].detach().cpu()
    label_ca = batch["x_1_states"]["bb_ca"][0].detach().cpu()
    sm = batch["state_mask"][0].detach().cpu()
    valid_k = int(sample["state_present_mask"].bool().sum().item())
    state_summaries = []
    for k in range(valid_k):
        s7.write_ca_pdb(sample_out / f"state{k:02d}_binder_ca.pdb", pred_ca[k, sm[k]], pred_seq)
        state_summaries.append({"state_index": k, "ca_rmsd_to_exact_label_nm": s7.kabsch_rmsd(pred_ca[k], label_ca[k], sm[k])})
    torch.save({"sample_id": sample.get("sample_id"), "pred_seq": pred_seq, "ref_seq": ref_seq, "x_states": {k: v.detach().cpu() for k, v in x_states.items()}}, sample_out / "state_specific_outputs.pt")
    return {
        "status": "passed",
        "sample_index": sample_idx,
        "sample_id": sample.get("sample_id"),
        "split": sample.get("split"),
        "target_id": sample.get("target_id"),
        "nsteps": nsteps,
        "valid_states": valid_k,
        "sequence_source": args.sequence_source,
        "pred_sequence": pred_seq,
        "reference_sequence": ref_seq,
        "sequence_identity_to_reference": seq_identity,
        "state_summaries": state_summaries,
        "guidance_trace_tail": guidance_trace[-3:],
        "out_dir": str(sample_out),
        "elapsed_sec": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--autoencoder-ckpt", type=Path, default=DEFAULT_AE)
    parser.add_argument("--sequence-source", choices=["shared_head", "ae_consensus", "hybrid"], default="shared_head")
    parser.add_argument("--sequence-hybrid-ae-weight", type=float, default=0.5)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sampling-config", type=Path, default=DEFAULT_SAMPLING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--nsteps", type=int, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--anchor-source", choices=["oracle", "source_state0"], default="oracle")
    parser.add_argument("--guidance-start-fraction", type=float, default=0.35)
    parser.add_argument("--guidance-inner-steps", type=int, default=1)
    parser.add_argument("--guidance-lr", type=float, default=0.015)
    parser.add_argument("--contact-cutoff-nm", type=float, default=1.0)
    parser.add_argument("--clash-cutoff-nm", type=float, default=0.28)
    parser.add_argument("--shell-max-nm", type=float, default=1.2)
    parser.add_argument("--max-anchor-pairs", type=int, default=128)
    parser.add_argument("--clash-weight", type=float, default=30.0)
    parser.add_argument("--clash-loss-mode", choices=["mean", "active"], default="mean")
    parser.add_argument("--clash-max-weight", type=float, default=4.0)
    parser.add_argument("--shell-weight", type=float, default=0.10)
    parser.add_argument("--tether-weight", type=float, default=0.002)
    parser.add_argument("--overcontact-weight", type=float, default=0.02)
    parser.add_argument("--contact-temperature-nm", type=float, default=0.06)
    parser.add_argument("--hard-min-dist-nm", type=float, default=0.28)
    parser.add_argument("--min-dist-improvement-nm", type=float, default=0.005)
    parser.add_argument("--f1-drop-tolerance", type=float, default=0.02)
    parser.add_argument("--anchor-loss-tolerance", type=float, default=0.25)
    parser.add_argument("--f1-gain-for-anchor-worsen", type=float, default=0.02)
    parser.add_argument("--max-guidance-displacement-nm", type=float, default=0.25)
    parser.add_argument("--flow-step-scale", type=float, default=1.0, help="Stage10B: multiply each model flow update; <1 keeps pose-initialized sampling closer to transferred source poses")
    parser.add_argument("--max-flow-displacement-nm", type=float, default=0.0, help="Stage10B: optional per-step displacement clamp for model flow updates; 0 disables")
    parser.add_argument("--bb-ca-flow-step-scale", type=float, default=None, help="Stage10B: override flow-step-scale for bb_ca only")
    parser.add_argument("--local-latents-flow-step-scale", type=float, default=None, help="Stage10B: override flow-step-scale for local_latents only")
    parser.add_argument("--bb-ca-max-flow-displacement-nm", type=float, default=None, help="Stage10B: override max-flow-displacement-nm for bb_ca only")
    parser.add_argument("--local-latents-max-flow-displacement-nm", type=float, default=None, help="Stage10B: override max-flow-displacement-nm for local_latents only")
    parser.add_argument("--use-learned-flow-gate", action="store_true", default=False, help="Stage11: use nn_out['flow_gate'] as bounded bb_ca flow blend instead of a fixed scalar only")
    parser.add_argument("--gate-safe-accept", action="store_true", default=True, help="Stage11E: reject learned-gate model flow updates that damage source-anchor contact/clash geometry")
    parser.add_argument("--unsafe-no-gate-safe-accept", dest="gate_safe_accept", action="store_false")
    parser.add_argument("--gate-f1-drop-tolerance", type=float, default=0.01)
    parser.add_argument("--enable-ca-feature", action="store_true", default=False)
    parser.add_argument("--disable-ca-feature", dest="enable_ca_feature", action="store_false")
    parser.add_argument("--enable-init-pose", action="store_true", default=False, help="Stage10: collate init_bb_ca_states and start bb_ca trajectories from transferred source pose")
    parser.add_argument("--safe-accept", action="store_true", default=True)
    parser.add_argument("--unsafe-no-safe-accept", dest="safe_accept", action="store_false")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    samples = load_samples(args.dataset)
    selected = select_samples(samples, args.split, set(args.sample_id) if args.sample_id else None, args.max_samples)
    model = load_sampling_model(args.ckpt, args.autoencoder_ckpt, device)
    ae = None
    if args.sequence_source != "shared_head":
        ae = s5ae.load_autoencoder(args.autoencoder_ckpt, device)
        ae.eval()
        for p in ae.parameters():
            p.requires_grad = False
    sampling_cfg = OmegaConf.to_container(OmegaConf.load(args.sampling_config).model, resolve=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    for sample_idx, sample in selected:
        rows.append(sample_one(model, ae, sample_idx, sample, sampling_cfg, args.out_dir, args.nsteps, device, args))
        write_json(args.report, {"status": "running", "rows": rows, "completed": len(rows), "total": len(selected)})
    summary = {
        "status": "passed",
        "oracle_warning": "Sampling-time guidance currently uses V_exact experimental contacts as anchors. This validates mechanics only; it is not a fair production benchmark until anchors are non-oracle.",
        "dataset": str(args.dataset),
        "ckpt": str(args.ckpt),
        "out_dir": str(args.out_dir),
        "nsteps": args.nsteps,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "sample_count": len(rows),
        "elapsed_sec": time.time() - started,
        "mean_sec_per_sample": (time.time() - started) / max(1, len(rows)),
        "rows": rows,
    }
    write_json(args.report, summary)
    print(json.dumps({"status": summary["status"], "sample_count": len(rows), "elapsed_sec": summary["elapsed_sec"], "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
