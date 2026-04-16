
"""Multistate flow-matching loss utilities for Strategy01.

The helpers in this file keep the original single-state flow matcher intact while
adding an explicit K-state supervision path for the multistate binder model.
"""

from __future__ import annotations

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


def _masked_topk_mean(values: Tensor, present: Tensor, topk: int) -> Tensor:
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
    l_state = torch.stack(state_losses, dim=0).sum(dim=0)
    l_mean = torch.sum(l_state * weights, dim=1)
    l_cvar = _masked_topk_mean(l_state, state_present, int(_cfg_get(cfg, "cvar_topk", 2)))
    l_var = _weighted_variance(l_state, weights)
    l_seq = _sequence_loss(nn_out, batch)
    alpha = float(_cfg_get(cfg, "alpha_mean", 0.5))
    beta = float(_cfg_get(cfg, "beta_cvar", 0.4))
    gamma = float(_cfg_get(cfg, "gamma_var", 0.1))
    lambda_seq = float(_cfg_get(cfg, "lambda_seq", 0.5))
    lambda_struct = float(_cfg_get(cfg, "lambda_struct", 1.0))
    l_struct = alpha * l_mean + beta * l_cvar + gamma * l_var
    l_total = lambda_seq * l_seq + lambda_struct * l_struct
    losses = {
        "multistate_total": l_total,
        # Proteina.training_step sums all losses that do not end in _justlog.
        # Keep decomposed terms visible without double-counting them in backward.
        "multistate_seq_justlog": l_seq,
        "multistate_struct_justlog": l_struct,
        "multistate_mean_justlog": l_mean,
        "multistate_cvar_justlog": l_cvar,
        "multistate_var_justlog": l_var,
    }
    for state_idx in range(k):
        losses[f"multistate_state_{state_idx}_justlog"] = l_state[:, state_idx]
    return losses
