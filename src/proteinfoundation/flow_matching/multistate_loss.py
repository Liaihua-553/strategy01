"""Multistate flow-matching loss utilities for Strategy01.

Stage03 introduced robust K-state flow matching.  Stage04 extends that path
with interface supervision extracted from real/predicted target-binder complexes:
contact labels, distance constraints, clash penalties, cross-state anchor
persistence, binder self-geometry consistency, and a light quality-proxy head.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _multistate_cfg(flow_matcher: Any) -> Any:
    return _cfg_get(_cfg_get(flow_matcher.cfg_exp, "loss", {}), "multistate", {})


def _normalize_state_weights(batch: dict[str, Tensor], b: int, k: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    weights = batch.get("target_state_weights")
    if weights is None:
        weights = torch.ones(b, k, device=device, dtype=dtype)
    else:
        weights = weights.to(device=device, dtype=dtype)
    present = batch.get("state_present_mask")
    if present is None:
        present = torch.ones(b, k, device=device, dtype=torch.bool)
    else:
        present = present.to(device=device).bool()
    weights = weights * present.float()
    denom = weights.sum(dim=1, keepdim=True)
    fallback = present.float() / present.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.where(denom > 1e-8, weights / denom.clamp_min(1e-8), fallback)


def _require_multistate_keys(batch: dict) -> None:
    required = ["x_1_states", "binder_seq_shared", "mask"]
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Multistate loss requires missing batch keys: {missing}")
    for dm in ("bb_ca", "local_latents"):
        if dm not in batch["x_1_states"]:
            raise KeyError(f"Multistate loss requires x_1_states['{dm}']")


def corrupt_multistate_batch(flow_matcher: Any, batch: dict) -> dict:
    """Create state-wise noisy samples and a single weighted binder input."""
    _require_multistate_keys(batch)
    x_1_states = batch["x_1_states"]
    bb_ca = x_1_states["bb_ca"]
    b, k, n = bb_ca.shape[:3]
    device = bb_ca.device
    dtype = bb_ca.dtype
    base_mask = batch["mask"].to(device=device).bool()
    state_present = batch.get("state_present_mask")
    if state_present is None:
        state_present = torch.ones(b, k, device=device, dtype=torch.bool)
    else:
        state_present = state_present.to(device=device).bool()
    state_mask = batch.get("state_mask")
    if state_mask is None:
        state_mask = base_mask[:, None, :] & state_present[:, :, None]
    else:
        state_mask = state_mask.to(device=device).bool()
    state_weights = _normalize_state_weights(batch, b, k, device, dtype)

    # Use one shared corruption path per sample across all states.  The model sees
    # a single binder x_t plus state-conditioned heads; independent per-state noise
    # would create hidden random variables that the state heads cannot infer.
    base_shape = (b,)
    t_base = flow_matcher.sample_t(shape=base_shape, device=device)
    t = {dm: t_base[dm][:, None].expand(b, k).contiguous() for dm in flow_matcher.data_modes}
    x_0_states = {}
    x_t_states = {}
    for data_mode in flow_matcher.data_modes:
        x_1 = x_1_states[data_mode].to(device=device)
        x_0_base = flow_matcher.base_flow_matchers[data_mode].sample_noise(
            n=n,
            shape=base_shape,
            device=device,
            mask=base_mask,
            training=flow_matcher.training,
        )
        x_0 = x_0_base[:, None, :, :].expand(b, k, n, x_0_base.shape[-1]).contiguous()
        x_0 = x_0 * state_mask[..., None]
        x_t = flow_matcher.base_flow_matchers[data_mode].interpolate(
            x_0=x_0,
            x_1=x_1,
            t=t[data_mode],
            mask=state_mask,
        )
        x_0_states[data_mode] = x_0
        x_t_states[data_mode] = x_t

    batch["x_0_states"] = x_0_states
    batch["x_t_states"] = x_t_states
    batch["t_states"] = t
    batch["state_mask"] = state_mask
    batch["x_1"] = {dm: torch.sum(x_1_states[dm].to(device=device) * state_weights[:, :, None, None], dim=1) for dm in flow_matcher.data_modes}
    batch["x_0"] = {dm: torch.sum(x_0_states[dm] * state_weights[:, :, None, None], dim=1) for dm in flow_matcher.data_modes}
    batch["x_t"] = {dm: torch.sum(x_t_states[dm] * state_weights[:, :, None, None], dim=1) for dm in flow_matcher.data_modes}
    batch["t"] = {dm: torch.sum(t[dm] * state_weights, dim=1) for dm in flow_matcher.data_modes}
    if "x_sc" not in batch:
        batch["x_sc"] = {dm: torch.zeros_like(batch["x_t"][dm]) for dm in flow_matcher.data_modes}
    return batch


def _state_loss_from_flat(flat_loss: Tensor, b: int, k: int, state_present: Tensor) -> Tensor:
    state_loss = flat_loss.reshape(b, k)
    return torch.where(state_present.bool(), state_loss, torch.zeros_like(state_loss))


def _weighted_variance(values: Tensor, weights: Tensor) -> Tensor:
    mean = torch.sum(values * weights, dim=1)
    return torch.sum(weights * (values - mean[:, None]) ** 2, dim=1)


def _masked_topk_mean(values: Tensor, present: Tensor, topk: int | None) -> Tensor:
    if topk is None or topk <= 0:
        valid_counts = present.long().sum(dim=1).clamp_min(1)
        topk = int(torch.ceil(valid_counts.float().max() / 2).item())
    masked = values.masked_fill(~present.bool(), float("-inf"))
    valid_counts = present.long().sum(dim=1).clamp_min(1)
    k_eff = torch.clamp(valid_counts, max=topk)
    top_values = torch.topk(masked, k=min(topk, values.shape[1]), dim=1).values
    top_values = torch.where(torch.isfinite(top_values), top_values, torch.zeros_like(top_values))
    selector = torch.arange(top_values.shape[1], device=values.device)[None, :] < k_eff[:, None]
    return (top_values * selector.float()).sum(dim=1) / k_eff.float()


def _sequence_loss(nn_out: dict, batch: dict) -> Tensor:
    if "seq_logits_shared" not in nn_out:
        raise KeyError("Multistate loss requires nn_out['seq_logits_shared']")
    target = batch["binder_seq_shared"].to(nn_out["seq_logits_shared"].device).long()
    mask = batch.get("binder_seq_mask", batch["mask"]).to(nn_out["seq_logits_shared"].device).bool()
    logits = nn_out["seq_logits_shared"]
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1), reduction="none")
    ce = ce.reshape(target.shape)
    return (ce * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)


def _interface_enabled(cfg: Any) -> bool:
    if bool(_cfg_get(cfg, "use_interface_loss", False)):
        return True
    for key in (
        "lambda_contact",
        "lambda_distance",
        "lambda_clash",
        "lambda_anchor_persistence",
        "lambda_quality_proxy",
        "lambda_self_geometry",
    ):
        if float(_cfg_get(cfg, key, 0.0)) != 0.0:
            return True
    return False


def _masked_pair_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask.float()).sum(dim=(-2, -1)) / mask.float().sum(dim=(-2, -1)).clamp_min(1.0)


def _target_ca_and_mask(batch: dict, device: torch.device) -> tuple[Tensor, Tensor]:
    if "x_target_states" not in batch or "target_mask_states" not in batch:
        raise KeyError("Stage04 interface loss requires x_target_states and target_mask_states")
    x_target = batch["x_target_states"].to(device=device)
    target_mask = batch["target_mask_states"].to(device=device).bool()
    return x_target[..., 1, :], target_mask[..., 1]


def _pairwise_target_binder_dist(target_ca: Tensor, binder_ca: Tensor) -> Tensor:
    b, k, nt = target_ca.shape[:3]
    nb = binder_ca.shape[2]
    d = torch.cdist(target_ca.reshape(b * k, nt, 3), binder_ca.reshape(b * k, nb, 3))
    return d.reshape(b, k, nt, nb)


def _compute_interface_losses(cfg: Any, batch: dict, nn_out: dict, state_present: Tensor, state_mask: Tensor) -> dict[str, Tensor]:
    if "bb_ca_states" not in nn_out:
        raise KeyError("Stage04 interface loss requires nn_out['bb_ca_states']")
    pred_bb = nn_out["bb_ca_states"]
    device = pred_bb.device
    dtype = pred_bb.dtype
    b, k, nb = pred_bb.shape[:3]
    target_ca, target_ca_mask = _target_ca_and_mask(batch, device)
    dists = _pairwise_target_binder_dist(target_ca.to(dtype=dtype), pred_bb)
    pair_mask = target_ca_mask[:, :, :, None] & state_mask[:, :, None, :].to(device).bool()
    label_mask = batch.get("interface_label_mask")
    if label_mask is not None:
        pair_mask = pair_mask & label_mask.to(device).bool()

    contact_labels = batch.get("interface_contact_labels")
    distance_labels = batch.get("interface_distance_labels")
    require_labels = bool(_cfg_get(cfg, "require_interface_labels", False))
    if require_labels and (contact_labels is None or distance_labels is None):
        raise KeyError("Stage04 interface loss requires interface_contact_labels and interface_distance_labels")
    contact_cutoff = float(_cfg_get(cfg, "contact_cutoff_nm", 0.8))
    contact_temperature = float(_cfg_get(cfg, "contact_temperature_nm", 0.08))
    clash_min = float(_cfg_get(cfg, "clash_min_nm", 0.28))
    clash_temperature = float(_cfg_get(cfg, "clash_temperature_nm", 0.04))

    zeros_state = torch.zeros(b, k, device=device, dtype=dtype)
    if contact_labels is not None:
        labels = contact_labels.to(device=device, dtype=dtype)
        logits = (contact_cutoff - dists) / max(contact_temperature, 1e-6)
        contact_raw = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="none").to(dtype)
        l_contact_state = _masked_pair_mean(contact_raw, pair_mask)
    else:
        l_contact_state = zeros_state

    if distance_labels is not None:
        labels = distance_labels.to(device=device, dtype=dtype)
        dist_mask = pair_mask
        contact_for_dist = batch.get("interface_distance_mask")
        if contact_for_dist is not None:
            dist_mask = dist_mask & contact_for_dist.to(device).bool()
        elif contact_labels is not None:
            dist_mask = dist_mask & contact_labels.to(device).bool()
        dist_raw = F.smooth_l1_loss(dists.float(), labels.float(), reduction="none", beta=0.1).to(dtype)
        l_distance_state = _masked_pair_mean(dist_raw, dist_mask)
    else:
        l_distance_state = zeros_state

    clash_raw = F.softplus((clash_min - dists) / max(clash_temperature, 1e-6)).to(dtype)
    l_clash_state = _masked_pair_mean(clash_raw, pair_mask)

    l_quality_state = zeros_state
    if "interface_quality_logits" in nn_out and "interface_quality_labels" in batch:
        quality_logits = nn_out["interface_quality_logits"]
        quality_labels = batch["interface_quality_labels"].to(device=device, dtype=quality_logits.dtype)
        quality_mask = batch.get("interface_quality_mask")
        if quality_mask is None:
            quality_mask = torch.ones_like(quality_labels, dtype=torch.bool, device=device)
        else:
            quality_mask = quality_mask.to(device).bool()
        q_loss = F.mse_loss(torch.sigmoid(quality_logits).float(), quality_labels.float(), reduction="none").to(dtype)
        l_quality_state = (q_loss * quality_mask.float()).sum(dim=-1) / quality_mask.float().sum(dim=-1).clamp_min(1.0)

    l_anchor = torch.zeros(b, device=device, dtype=dtype)
    if contact_labels is not None:
        labels = contact_labels.to(device=device, dtype=dtype)
        valid_state = state_present.to(device).bool()
        state_denom = valid_state.float().sum(dim=1).clamp_min(1.0)
        persistent = (labels * valid_state[:, :, None, None].float()).sum(dim=1) / state_denom[:, None, None]
        persistent = (persistent >= float(_cfg_get(cfg, "anchor_persistence_threshold", 0.5))).float()
        pred_prob = torch.sigmoid((contact_cutoff - dists) / max(contact_temperature, 1e-6))
        pred_mean = (pred_prob * valid_state[:, :, None, None].float()).sum(dim=1) / state_denom[:, None, None]
        anchor_mask = pair_mask.any(dim=1)
        anchor_raw = F.binary_cross_entropy(pred_mean.float().clamp(1e-6, 1 - 1e-6), persistent.float(), reduction="none").to(dtype)
        l_anchor = (anchor_raw * anchor_mask.float()).sum(dim=(-2, -1)) / anchor_mask.float().sum(dim=(-2, -1)).clamp_min(1.0)

    l_self = torch.zeros(b, device=device, dtype=dtype)
    if bool(_cfg_get(cfg, "use_self_geometry_loss", True)) and k > 1:
        binder_pair = torch.cdist(pred_bb.reshape(b * k, nb, 3), pred_bb.reshape(b * k, nb, 3)).reshape(b, k, nb, nb)
        valid_state = state_present.to(device).bool()
        denom = valid_state.float().sum(dim=1).clamp_min(1.0)
        mean_pair = (binder_pair * valid_state[:, :, None, None].float()).sum(dim=1) / denom[:, None, None]
        var_pair = ((binder_pair - mean_pair[:, None]) ** 2 * valid_state[:, :, None, None].float()).sum(dim=1) / denom[:, None, None]
        binder_mask = state_mask.any(dim=1).to(device).bool()
        pair_binder_mask = binder_mask[:, :, None] & binder_mask[:, None, :]
        eye = torch.eye(nb, dtype=torch.bool, device=device)[None]
        pair_binder_mask = pair_binder_mask & ~eye
        l_self = (var_pair * pair_binder_mask.float()).sum(dim=(-2, -1)) / pair_binder_mask.float().sum(dim=(-2, -1)).clamp_min(1.0)

    return {
        "contact_state": torch.where(state_present, l_contact_state, torch.zeros_like(l_contact_state)),
        "distance_state": torch.where(state_present, l_distance_state, torch.zeros_like(l_distance_state)),
        "clash_state": torch.where(state_present, l_clash_state, torch.zeros_like(l_clash_state)),
        "quality_state": torch.where(state_present, l_quality_state, torch.zeros_like(l_quality_state)),
        "anchor_sample": l_anchor,
        "self_geometry_sample": l_self,
    }


def compute_multistate_loss(flow_matcher: Any, batch: dict, nn_out: dict[str, Tensor]) -> dict[str, Tensor]:
    """Compute Strategy01 robust multistate loss."""
    _require_multistate_keys(batch)
    cfg = _multistate_cfg(flow_matcher)
    x_1_states = batch["x_1_states"]
    b, k, n = x_1_states["bb_ca"].shape[:3]
    device = x_1_states["bb_ca"].device
    state_present = batch.get("state_present_mask", torch.ones(b, k, device=device, dtype=torch.bool)).to(device).bool()
    state_mask = batch["state_mask"].to(device).bool()
    weights = _normalize_state_weights(batch, b, k, device, x_1_states["bb_ca"].dtype)
    mode_weights = {"bb_ca": float(_cfg_get(cfg, "lambda_bb", 1.0)), "local_latents": float(_cfg_get(cfg, "lambda_lat", 1.0))}
    state_losses = []
    output_names = {"bb_ca": "bb_ca_states", "local_latents": "local_latents_states"}
    for data_mode in flow_matcher.data_modes:
        out_key = output_names[data_mode]
        if out_key not in nn_out:
            raise KeyError(f"Multistate loss requires nn_out['{out_key}']")
        pred = nn_out[out_key]
        flat_nn_out = {flow_matcher.cfg_exp.nn.output_parameterization[data_mode]: pred.reshape(b * k, n, pred.shape[-1])}
        flat_loss = flow_matcher.base_flow_matchers[data_mode].compute_fm_loss(
            x_0=batch["x_0_states"][data_mode].reshape(b * k, n, -1),
            x_1=batch["x_1_states"][data_mode].reshape(b * k, n, -1),
            x_t=batch["x_t_states"][data_mode].reshape(b * k, n, -1),
            mask=state_mask.reshape(b * k, n),
            t=batch["t_states"][data_mode].reshape(b * k),
            nn_out=flat_nn_out,
        )
        state_losses.append(mode_weights[data_mode] * _state_loss_from_flat(flat_loss, b, k, state_present))
    l_fm_state = torch.stack(state_losses, dim=0).sum(dim=0)

    l_contact_state = torch.zeros_like(l_fm_state)
    l_distance_state = torch.zeros_like(l_fm_state)
    l_clash_state = torch.zeros_like(l_fm_state)
    l_quality_state = torch.zeros_like(l_fm_state)
    l_anchor = torch.zeros(b, device=device, dtype=l_fm_state.dtype)
    l_self = torch.zeros(b, device=device, dtype=l_fm_state.dtype)
    if _interface_enabled(cfg):
        interface = _compute_interface_losses(cfg, batch, nn_out, state_present, state_mask)
        l_contact_state = interface["contact_state"]
        l_distance_state = interface["distance_state"]
        l_clash_state = interface["clash_state"]
        l_quality_state = interface["quality_state"]
        l_anchor = interface["anchor_sample"]
        l_self = interface["self_geometry_sample"]

    l_state = (
        l_fm_state
        + float(_cfg_get(cfg, "lambda_contact", 0.0)) * l_contact_state
        + float(_cfg_get(cfg, "lambda_distance", 0.0)) * l_distance_state
        + float(_cfg_get(cfg, "lambda_clash", 0.0)) * l_clash_state
        + float(_cfg_get(cfg, "lambda_quality_proxy", 0.0)) * l_quality_state
    )
    l_mean = torch.sum(l_state * weights, dim=1)
    cvar_topk = _cfg_get(cfg, "cvar_topk", 2)
    if str(cvar_topk).lower() in {"auto", "none"}:
        cvar_topk = None
    else:
        cvar_topk = int(cvar_topk)
    l_cvar = _masked_topk_mean(l_state, state_present, cvar_topk)
    l_var = _weighted_variance(l_state, weights)
    l_seq = _sequence_loss(nn_out, batch)
    alpha = float(_cfg_get(cfg, "alpha_mean", 0.5))
    beta = float(_cfg_get(cfg, "beta_cvar", 0.4))
    gamma = float(_cfg_get(cfg, "gamma_var", 0.1))
    lambda_seq = float(_cfg_get(cfg, "lambda_seq", 0.5))
    lambda_struct = float(_cfg_get(cfg, "lambda_struct", 1.0))
    l_struct = alpha * l_mean + beta * l_cvar + gamma * l_var
    l_total = (
        lambda_seq * l_seq
        + lambda_struct * l_struct
        + float(_cfg_get(cfg, "lambda_anchor_persistence", 0.0)) * l_anchor
        + float(_cfg_get(cfg, "lambda_self_geometry", 0.0)) * l_self
    )
    losses = {
        "multistate_total": l_total,
        "multistate_seq_justlog": l_seq,
        "multistate_struct_justlog": l_struct,
        "multistate_fm_mean_justlog": torch.sum(l_fm_state * weights, dim=1),
        "multistate_contact_justlog": torch.sum(l_contact_state * weights, dim=1),
        "multistate_distance_justlog": torch.sum(l_distance_state * weights, dim=1),
        "multistate_clash_justlog": torch.sum(l_clash_state * weights, dim=1),
        "multistate_quality_proxy_justlog": torch.sum(l_quality_state * weights, dim=1),
        "multistate_anchor_persistence_justlog": l_anchor,
        "multistate_self_geometry_justlog": l_self,
        "multistate_mean_justlog": l_mean,
        "multistate_cvar_justlog": l_cvar,
        "multistate_var_justlog": l_var,
    }
    for state_idx in range(k):
        losses[f"multistate_state_{state_idx}_justlog"] = l_state[:, state_idx]
        losses[f"multistate_state_{state_idx}_fm_justlog"] = l_fm_state[:, state_idx]
        losses[f"multistate_state_{state_idx}_contact_justlog"] = l_contact_state[:, state_idx]
        losses[f"multistate_state_{state_idx}_distance_justlog"] = l_distance_state[:, state_idx]
        losses[f"multistate_state_{state_idx}_clash_justlog"] = l_clash_state[:, state_idx]
    return losses
