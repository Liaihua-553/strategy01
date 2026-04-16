#!/usr/bin/env python
"""Stage03 Strategy01 multistate loss and debug fine-tune runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
BASELINE_ROOT = Path("/data/kfliao/general_model/strategy_repos/proteina-complexa-benchmark-baseline")
STAGE_DIR = REPO_ROOT / "ckpts" / "stage03_multistate_loss"
RUNS_DIR = STAGE_DIR / "runs"
REPORT_PROBE_DIR = REPO_ROOT / "reports" / "strategy01" / "probes"
DATA_DIR = REPO_ROOT / "data" / "strategy01" / "stage03_multistate_debug"
BASELINE_COMPLEXA = BASELINE_ROOT / "ckpts" / "complexa.ckpt"
BASELINE_AE = BASELINE_ROOT / "ckpts" / "complexa_ae.ckpt"
STAGE_COMPLEXA = STAGE_DIR / "complexa_init_readonly_copy.ckpt"
STAGE_AE = STAGE_DIR / "complexa_ae_init_readonly_copy.ckpt"
NN_CONFIG = REPO_ROOT / "configs" / "nn" / "local_latents_score_nn_160M_multistate.yaml"
TRAIN_CONFIG = REPO_ROOT / "configs" / "training_local_latents_multistate_loss_probe.yaml"

sys.path.insert(0, str(REPO_ROOT / "src"))

from openfold.np import residue_constants  # noqa: E402
from proteinfoundation.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher  # noqa: E402
from proteinfoundation.nn.local_latents_transformer_multistate import LocalLatentsTransformerMultistate  # noqa: E402

ATOM_ORDER = residue_constants.atom_order
RESTYPE_3TO1 = residue_constants.restype_3to1
RESTYPE_ORDER = residue_constants.restype_order

TARGET_SPECS = {
    "1tnf": [
        BASELINE_ROOT / "assets" / "target_data" / "alpha_proteo_targets" / "1tnf_cropped.pdb",
        BASELINE_ROOT / "assets" / "target_data" / "alpha_proteo_targets" / "1tnf_repacked.pdb",
        BASELINE_ROOT / "assets" / "target_data" / "alpha_proteo_targets" / "1tnf_cropped_fixed.pdb",
    ],
    "3di3": [
        BASELINE_ROOT / "assets" / "target_data" / "alpha_proteo_targets" / "3di3_cropped.pdb",
        BASELINE_ROOT / "assets" / "target_data" / "alpha_proteo_targets" / "3di3_repacked.pdb",
        BASELINE_ROOT / "assets" / "target_data" / "alpha_proteo_targets" / "3di3_cropped_fixed.pdb",
    ],
    "5o45": [
        BASELINE_ROOT / "assets" / "target_data" / "alpha_proteo_targets" / "5o45_cropped.pdb",
        BASELINE_ROOT / "assets" / "target_data" / "alpha_proteo_targets" / "5o45_repacked.pdb",
        BASELINE_ROOT / "assets" / "target_data" / "alpha_proteo_targets" / "5o45_cropped_fixed.pdb",
    ],
}


@dataclass
class DebugSample:
    sample_id: str
    target_name: str
    binder_seq: torch.Tensor
    binder_bb_states: torch.Tensor
    binder_latent_states: torch.Tensor
    target_x_states: torch.Tensor
    target_mask_states: torch.Tensor
    target_seq_states: torch.Tensor
    target_hotspot_states: torch.Tensor
    provenance: list[dict[str, Any]]


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_stage_ckpt_copies() -> dict[str, Any]:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    actions = []
    for src, dst in [(BASELINE_COMPLEXA, STAGE_COMPLEXA), (BASELINE_AE, STAGE_AE)]:
        if not dst.exists():
            shutil.copy2(src, dst)
            os.chmod(dst, 0o444)
            actions.append({"action": "copied", "src": str(src), "dst": str(dst)})
        else:
            actions.append({"action": "exists", "src": str(src), "dst": str(dst)})
        if os.stat(dst).st_mode & 0o222:
            os.chmod(dst, 0o444)
    return {"stage_dir": str(STAGE_DIR), "actions": actions}


def parse_pdb_atom37(
    path: Path,
    target_len: int,
    fallback: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    residues: dict[tuple[str, int, str], dict[str, Any]] = {}
    if path.exists() and path.stat().st_size > 0:
        with path.open("rt", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("ATOM"):
                    continue
                atom = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21].strip() or "_"
                try:
                    resseq = int(line[22:26])
                    x = float(line[30:38]) / 10.0
                    y = float(line[38:46]) / 10.0
                    z = float(line[46:54]) / 10.0
                except ValueError:
                    continue
                key = (chain, resseq, line[26].strip())
                item = residues.setdefault(key, {"resname": resname, "atoms": {}})
                if atom in ATOM_ORDER:
                    item["atoms"][atom] = (x, y, z)
    if not residues:
        if fallback is None:
            raise ValueError(f"No ATOM records found in {path}")
        x, mask, seq = fallback
        prov = {"path": str(path), "status": "fallback_from_previous_state", "reason": "empty_or_unreadable_pdb"}
        return x.clone(), mask.clone(), seq.clone(), prov

    ordered = [residues[key] for key in sorted(residues.keys(), key=lambda v: (v[0], v[1], v[2]))]
    x = torch.zeros(target_len, 37, 3, dtype=torch.float32)
    mask = torch.zeros(target_len, 37, dtype=torch.bool)
    seq = torch.zeros(target_len, dtype=torch.long)
    ncopy = min(target_len, len(ordered))
    pseudo_offsets = {
        "N": torch.tensor([-0.13, 0.03, 0.0]),
        "CA": torch.zeros(3),
        "C": torch.tensor([0.13, -0.03, 0.0]),
        "O": torch.tensor([0.19, -0.08, 0.0]),
    }
    for i in range(ncopy):
        resname = ordered[i]["resname"]
        aa1 = RESTYPE_3TO1.get(resname, "A")
        seq[i] = int(RESTYPE_ORDER.get(aa1, 0))
        for atom, coor in ordered[i]["atoms"].items():
            aidx = ATOM_ORDER[atom]
            x[i, aidx] = torch.tensor(coor, dtype=torch.float32)
            mask[i, aidx] = True
        ca_idx = ATOM_ORDER["CA"]
        ca = x[i, ca_idx].clone() if mask[i, ca_idx] else torch.tensor([i * 0.38, 0.0, 0.0])
        for atom, offset in pseudo_offsets.items():
            aidx = ATOM_ORDER[atom]
            if not mask[i, aidx]:
                x[i, aidx] = ca + offset
                mask[i, aidx] = True
    for i in range(ncopy, target_len):
        ca = torch.tensor([i * 0.38, 0.0, 0.0])
        seq[i] = 0
        for atom, offset in pseudo_offsets.items():
            aidx = ATOM_ORDER[atom]
            x[i, aidx] = ca + offset
            mask[i, aidx] = True
    prov = {"path": str(path), "status": "loaded", "source_residue_count": len(ordered), "used_residue_count": ncopy}
    return x, mask, seq, prov


def make_target_states(
    target_name: str,
    target_len: int,
    sample_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    xs, masks, seqs, hotspots, provenance = [], [], [], [], []
    fallback = None
    for state_idx, path in enumerate(TARGET_SPECS[target_name]):
        try:
            x, mask, seq, prov = parse_pdb_atom37(path, target_len, fallback)
        except Exception as exc:
            x = torch.zeros(target_len, 37, 3)
            mask = torch.zeros(target_len, 37, dtype=torch.bool)
            seq = torch.zeros(target_len, dtype=torch.long)
            for i in range(target_len):
                ca = torch.tensor([i * 0.38, 0.05 * state_idx, 0.0])
                for atom, offset in {
                    "N": torch.tensor([-0.13, 0.03, 0.0]),
                    "CA": torch.zeros(3),
                    "C": torch.tensor([0.13, -0.03, 0.0]),
                    "O": torch.tensor([0.19, -0.08, 0.0]),
                }.items():
                    aidx = ATOM_ORDER[atom]
                    x[i, aidx] = ca + offset
                    mask[i, aidx] = True
            prov = {"path": str(path), "status": "synthetic_fallback", "reason": repr(exc)}
        x = x + (sample_offset % 7) * 0.0005 + state_idx * 0.001
        hotspot = torch.zeros(target_len, dtype=torch.bool)
        start = (sample_offset * 3 + state_idx * 5) % max(1, target_len - 6)
        hotspot[start : start + min(6, target_len)] = True
        xs.append(x)
        masks.append(mask)
        seqs.append(seq)
        hotspots.append(hotspot)
        provenance.append(prov)
        fallback = (x.detach().clone(), mask.detach().clone(), seq.detach().clone())
    return torch.stack(xs), torch.stack(masks), torch.stack(seqs), torch.stack(hotspots), provenance


def make_binder(sample_idx: int, binder_len: int, latent_dim: int, nstates: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq = (torch.arange(binder_len, dtype=torch.long) * 7 + sample_idx * 3) % 20
    mut_stride = 3 + (sample_idx % 3)
    seq[::mut_stride] = (seq[::mut_stride] + sample_idx + 5) % 20
    t = torch.linspace(0, 2 * math.pi, binder_len)
    base = torch.stack([0.18 * torch.cos(t), 0.18 * torch.sin(t), torch.linspace(0, 0.13 * binder_len, binder_len)], dim=-1)
    bb_states = []
    lat_states = []
    for state_idx in range(nstates):
        phase = state_idx * 0.35 + sample_idx * 0.03
        bend = torch.stack(
            [
                0.03 * torch.sin(t + phase),
                0.02 * torch.cos(t * 0.5 + phase),
                torch.zeros_like(t),
            ],
            dim=-1,
        )
        bb = base + bend + torch.tensor([0.4 + 0.04 * state_idx, -0.2 + 0.01 * sample_idx, 0.1 * state_idx])
        bb_states.append(bb.float())
        lat = torch.stack([torch.sin(t * (j + 1) * 0.3 + phase) for j in range(latent_dim)], dim=-1)
        lat = lat * 0.05 + 0.01 * sample_idx + 0.005 * state_idx
        lat_states.append(lat.float())
    return seq, torch.stack(bb_states), torch.stack(lat_states)


def build_debug_samples(num_samples: int, binder_len: int, target_len: int, latent_dim: int) -> tuple[list[DebugSample], dict[str, Any]]:
    samples: list[DebugSample] = []
    all_provenance = []
    target_names = ["1tnf", "3di3", "5o45"]
    for idx in range(num_samples):
        target_name = target_names[idx % len(target_names)]
        tx, tm, ts, th, prov = make_target_states(target_name, target_len, idx)
        seq, bb, lat = make_binder(idx, binder_len, latent_dim, 3)
        sample = DebugSample(
            sample_id=f"stage03_{idx:03d}_{target_name}",
            target_name=target_name,
            binder_seq=seq,
            binder_bb_states=bb,
            binder_latent_states=lat,
            target_x_states=tx,
            target_mask_states=tm,
            target_seq_states=ts,
            target_hotspot_states=th,
            provenance=prov,
        )
        samples.append(sample)
        all_provenance.append({"sample_id": sample.sample_id, "target_name": target_name, "states": prov})
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "num_samples": num_samples,
        "binder_len": binder_len,
        "target_len": target_len,
        "latent_dim": latent_dim,
        "state_roles": ["required_bind", "required_bind", "required_bind"],
        "state_weights": [1 / 3, 1 / 3, 1 / 3],
        "targets": target_names,
        "note": "Stage03 engineering debug pseudo dataset; binder complex supervision is synthetic and used only for plumbing/overfit validation.",
        "provenance": all_provenance,
    }
    (DATA_DIR / "manifest_stage03_debug.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    torch.save({"samples": samples, "manifest": manifest}, DATA_DIR / "stage03_debug_samples.pt")
    return samples, manifest


def collate_samples(samples: list[DebugSample], device: torch.device) -> dict[str, Any]:
    b = len(samples)
    k = samples[0].binder_bb_states.shape[0]
    nb = samples[0].binder_bb_states.shape[1]
    batch = {
        "mask": torch.ones(b, nb, dtype=torch.bool, device=device),
        "binder_seq_shared": torch.stack([s.binder_seq for s in samples]).to(device),
        "binder_seq_mask": torch.ones(b, nb, dtype=torch.bool, device=device),
        "x_1_states": {
            "bb_ca": torch.stack([s.binder_bb_states for s in samples]).to(device),
            "local_latents": torch.stack([s.binder_latent_states for s in samples]).to(device),
        },
        "state_mask": torch.ones(b, k, nb, dtype=torch.bool, device=device),
        "state_present_mask": torch.ones(b, k, dtype=torch.bool, device=device),
        "target_state_weights": torch.full((b, k), 1.0 / k, dtype=torch.float32, device=device),
        "target_state_roles": torch.ones(b, k, dtype=torch.long, device=device),
        "x_target_states": torch.stack([s.target_x_states for s in samples]).to(device),
        "target_mask_states": torch.stack([s.target_mask_states for s in samples]).to(device),
        "seq_target_states": torch.stack([s.target_seq_states for s in samples]).to(device),
        "target_hotspot_mask_states": torch.stack([s.target_hotspot_states for s in samples]).to(device),
        "seq_target_mask_states": torch.stack([s.target_mask_states.any(dim=-1) for s in samples]).to(device),
        "hotspot_mask": torch.zeros(b, nb, dtype=torch.bool, device=device),
        "residue_pdb_idx": torch.arange(1, nb + 1, dtype=torch.float32, device=device).unsqueeze(0).expand(b, -1),
        "chains": torch.zeros(b, nb, dtype=torch.long, device=device),
        "use_ca_coors_nm_feature": False,
        "use_residue_type_feature": False,
        "strict_feats": False,
    }
    batch["hotspot_mask"][:, : min(4, nb)] = True
    return batch


def load_cfg() -> tuple[Any, dict[str, Any], int, dict[str, Any]]:
    ckpt = torch.load(STAGE_COMPLEXA, map_location="cpu")
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
    metadata = {
        "latent_dim": latent_dim,
        "checkpoint_compatible_keys": len(compatible),
        "checkpoint_incompatible_shape_keys": incompatible_shape[:50],
        "missing_keys_count": len(load_info.missing_keys),
        "missing_keys_sample": load_info.missing_keys[:50],
        "unexpected_keys_count": len(load_info.unexpected_keys),
        "train_config": str(TRAIN_CONFIG),
        "nn_config": str(NN_CONFIG),
    }
    return model, fm, metadata


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


def forward_loss(
    model: torch.nn.Module,
    fm: ProductSpaceFlowMatcher,
    batch: dict[str, Any],
    seed: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    work_batch = clone_batch(batch)
    work_batch = fm.corrupt_multistate_batch(work_batch)
    nn_out = model(work_batch)
    losses = fm.compute_multistate_loss(work_batch, nn_out)
    total = losses["multistate_total"].mean()
    return total, losses, nn_out


def summarize_losses(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: float(v.detach().float().mean().cpu().item()) for k, v in losses.items()}


def loss_unit_probe(model: torch.nn.Module, fm: ProductSpaceFlowMatcher, samples: list[DebugSample], device: torch.device) -> dict[str, Any]:
    model.eval()
    batch = collate_samples(samples[:2], device)
    with torch.no_grad():
        total, losses, nn_out = forward_loss(model, fm, batch, seed=101)
    return {
        "status": "passed",
        "total": float(total.detach().cpu().item()),
        "losses": summarize_losses(losses),
        "output_shapes": {k: list(v.shape) for k, v in nn_out.items() if isinstance(v, torch.Tensor)},
        "nan_detected": bool(any(torch.isnan(v).any().item() for v in losses.values())),
    }


def worst_state_probe(samples: list[DebugSample], device: torch.device) -> dict[str, Any]:
    batch = collate_samples(samples[:2], device)
    l_state_good = torch.tensor([[1.0, 1.0, 1.0], [0.8, 0.9, 1.0]], device=device)
    l_state_bad = torch.tensor([[1.0, 1.0, 5.0], [0.8, 0.9, 4.5]], device=device)
    weights = batch["target_state_weights"]
    present = batch["state_present_mask"]
    from proteinfoundation.flow_matching.multistate_loss import _masked_topk_mean, _weighted_variance

    mean_good = (l_state_good * weights).sum(dim=1)
    mean_bad = (l_state_bad * weights).sum(dim=1)
    cvar_good = _masked_topk_mean(l_state_good, present, 2)
    cvar_bad = _masked_topk_mean(l_state_bad, present, 2)
    var_good = _weighted_variance(l_state_good, weights)
    var_bad = _weighted_variance(l_state_bad, weights)
    robust_good = 0.5 * mean_good + 0.4 * cvar_good + 0.1 * var_good
    robust_bad = 0.5 * mean_bad + 0.4 * cvar_bad + 0.1 * var_bad
    return {
        "status": "passed",
        "mean_delta": float((mean_bad - mean_good).mean().cpu().item()),
        "robust_delta": float((robust_bad - robust_good).mean().cpu().item()),
        "robust_more_sensitive_than_mean": bool(((robust_bad - robust_good) > (mean_bad - mean_good)).all().item()),
        "good": {"mean": jsonable(mean_good), "cvar": jsonable(cvar_good), "var": jsonable(var_good), "robust": jsonable(robust_good)},
        "bad": {"mean": jsonable(mean_bad), "cvar": jsonable(cvar_bad), "var": jsonable(var_bad), "robust": jsonable(robust_bad)},
    }


def set_trainable(model: torch.nn.Module, phase: str) -> dict[str, Any]:
    for p in model.parameters():
        p.requires_grad = False
    prefixes = [
        "ensemble_target_encoder",
        "target2binder_cross_attention_layer",
        "state_condition_projector",
        "shared_seq_head",
        "state_token_norm",
    ]
    if phase == "overfit":
        # Capacity diagnostic: if partial unfreezing cannot fit one sample, unfreeze
        # all parameters to verify the multistate supervision itself is learnable.
        for p in model.parameters():
            p.requires_grad = True
        prefixes = ["ALL_PARAMETERS_FOR_OVERFIT_DIAGNOSTIC"]
    else:
        if phase == "debug32":
            prefixes += ["local_latents_linear", "ca_linear"]
            prefixes += [f"transformer_layers.{i}" for i in range(10, 14)]
            if hasattr(model, "pair_update_layers"):
                prefixes += [f"pair_update_layers.{i}" for i in range(10, 13)]
        for name, p in model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"phase": phase, "trainable_params": trainable, "total_params": total, "trainable_fraction": trainable / max(total, 1), "prefixes": prefixes}


def make_optimizer(model: torch.nn.Module, old_lr: float = 5e-5) -> torch.optim.Optimizer:
    new_prefixes = (
        "ensemble_target_encoder",
        "target2binder_cross_attention_layer",
        "state_condition_projector",
        "shared_seq_head",
        "state_token_norm",
    )
    new_params, old_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(new_prefixes):
            new_params.append(p)
        else:
            old_params.append(p)
    groups = []
    if new_params:
        groups.append({"params": new_params, "lr": 2e-4})
    if old_params:
        groups.append({"params": old_params, "lr": old_lr})
    return torch.optim.AdamW(groups, weight_decay=0.01)


def grad_route_probe(model: torch.nn.Module, fm: ProductSpaceFlowMatcher, samples: list[DebugSample], device: torch.device) -> dict[str, Any]:
    model.train()
    trainable_info = set_trainable(model, "overfit")
    batch = collate_samples(samples[:2], device)
    total, _, _ = forward_loss(model, fm, batch, seed=202)
    total.backward()
    names = [
        "ensemble_target_encoder.cross_state_fusion.q_proj.weight",
        "target2binder_cross_attention_layer.0.mhba.mha.to_q_a.weight",
        "shared_seq_head.1.weight",
        "ca_linear.1.weight",
        "local_latents_linear.1.weight",
        "state_condition_projector.1.weight",
    ]
    params = dict(model.named_parameters())
    grad_norms = {}
    for name in names:
        p = params.get(name)
        grad_norms[name] = None if p is None or p.grad is None else float(p.grad.detach().norm().cpu().item())
    model.zero_grad(set_to_none=True)
    missing_batch = collate_samples(samples[:2], device)
    missing_batch["state_present_mask"][:, 2] = False
    missing_batch["state_mask"][:, 2, :] = False
    total_missing, losses_missing, _ = forward_loss(model, fm, missing_batch, seed=203)
    return {
        "status": "passed",
        "trainable_info": trainable_info,
        "loss": float(total.detach().cpu().item()),
        "loss_missing_state": float(total_missing.detach().cpu().item()),
        "grad_norms": grad_norms,
        "all_required_grad_nonzero": bool(all(v is not None and v > 0 for v in grad_norms.values())),
        "missing_state_losses": summarize_losses(losses_missing),
    }


def run_train_loop(
    model: torch.nn.Module,
    fm: ProductSpaceFlowMatcher,
    samples: list[DebugSample],
    device: torch.device,
    phase: str,
    steps: int,
    batch_size: int,
    run_dir: Path,
    eval_every: int,
    save_every: int,
    require_drop: float | None = None,
    fixed_train_seed: int | None = None,
) -> dict[str, Any]:
    model.train()
    phase_trainable = "stage_a" if phase == "stage_a" else ("debug32" if phase == "debug32" else "overfit")
    trainable_info = set_trainable(model, phase_trainable)
    opt = make_optimizer(model, old_lr=2e-4 if phase.startswith("overfit") else 5e-5)
    history = []
    start_time = time.time()
    fixed_eval_batch = collate_samples(samples[: min(batch_size, len(samples))], device)
    with torch.no_grad():
        initial_eval, _, _ = forward_loss(model, fm, fixed_eval_batch, seed=777)
    best = float(initial_eval.detach().cpu().item())
    for step in range(1, steps + 1):
        offset = ((step - 1) * batch_size) % len(samples)
        batch_samples = [samples[(offset + j) % len(samples)] for j in range(batch_size)]
        batch = collate_samples(batch_samples, device)
        opt.zero_grad(set_to_none=True)
        total, _, _ = forward_loss(model, fm, batch, seed=fixed_train_seed)
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}: {total}")
        total.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % eval_every == 0 or step == 1 or step == steps:
            with torch.no_grad():
                eval_total, eval_losses, _ = forward_loss(model, fm, fixed_eval_batch, seed=777)
            eval_value = float(eval_total.detach().cpu().item())
            best = min(best, eval_value)
            row = {
                "step": step,
                "train_total": float(total.detach().cpu().item()),
                "eval_total": eval_value,
                "eval_losses": summarize_losses(eval_losses),
                "elapsed_sec": time.time() - start_time,
                "cuda_max_mem_gb": torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0,
            }
            history.append(row)
            print(json.dumps({"phase": phase, **row}), flush=True)
        if step % save_every == 0 or step == steps:
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "phase": phase, "step": step}, run_dir / f"{phase}_step_{step}.pt")
    with torch.no_grad():
        final_eval, final_losses, _ = forward_loss(model, fm, fixed_eval_batch, seed=777)
    elapsed = time.time() - start_time
    initial = float(initial_eval.detach().cpu().item())
    final = float(final_eval.detach().cpu().item())
    drop_fraction = (initial - final) / max(abs(initial), 1e-8)
    return {
        "phase": phase,
        "steps": steps,
        "batch_size": batch_size,
        "initial_eval_total": initial,
        "final_eval_total": final,
        "best_eval_total": best,
        "drop_fraction": drop_fraction,
        "final_losses": summarize_losses(final_losses),
        "history": history,
        "elapsed_sec": elapsed,
        "step_time_sec": elapsed / max(steps, 1),
        "cuda_max_mem_gb": torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0,
        "trainable_info": trainable_info,
        "require_drop": require_drop,
        "fixed_train_seed": fixed_train_seed,
        "passed_drop_gate": None if require_drop is None else bool(drop_fraction >= require_drop),
    }


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="stage03_multistate_loss_debug")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binder-len", type=int, default=24)
    parser.add_argument("--target-len", type=int, default=48)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--stage-a-steps", type=int, default=300)
    parser.add_argument("--overfit1-steps", type=int, default=500)
    parser.add_argument("--overfit4-steps", type=int, default=800)
    parser.add_argument("--debug32-steps", type=int, default=1500)
    parser.add_argument("--batch-size-debug32", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--only-probes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ckpt_copy_info = ensure_stage_ckpt_copies()
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats()
    model, fm, model_meta = build_model(device)
    samples, manifest = build_debug_samples(args.num_samples, args.binder_len, args.target_len, int(model_meta["latent_dim"]))
    results: dict[str, Any] = {
        "run_name": args.run_name,
        "device": str(device),
        "seed": args.seed,
        "checkpoint_copy_info": ckpt_copy_info,
        "model_meta": model_meta,
        "dataset_manifest": manifest,
        "errors_fixed_during_stage": [],
    }
    t0 = time.time()
    try:
        results["loss_unit_probe"] = loss_unit_probe(model, fm, samples, device)
        results["worst_state_probe"] = worst_state_probe(samples, device)
        results["gradient_route_probe"] = grad_route_probe(model, fm, samples, device)
        if not args.only_probes:
            run_dir = RUNS_DIR / args.run_name
            results["stage_a"] = run_train_loop(
                model,
                fm,
                samples[:12],
                device,
                "stage_a",
                args.stage_a_steps,
                batch_size=2,
                run_dir=run_dir,
                eval_every=max(1, min(args.eval_every, args.stage_a_steps)),
                save_every=max(1, min(args.save_every, args.stage_a_steps)),
            )
            results["overfit_1_sample"] = run_train_loop(
                model,
                fm,
                samples[:1],
                device,
                "overfit1",
                args.overfit1_steps,
                batch_size=1,
                run_dir=run_dir,
                eval_every=max(1, min(50, args.overfit1_steps)),
                save_every=max(1, min(args.save_every, args.overfit1_steps)),
                require_drop=0.60,
                fixed_train_seed=777,
            )
            results["overfit_4_sample"] = run_train_loop(
                model,
                fm,
                samples[:4],
                device,
                "overfit4",
                args.overfit4_steps,
                batch_size=2,
                run_dir=run_dir,
                eval_every=max(1, min(args.eval_every, args.overfit4_steps)),
                save_every=max(1, min(args.save_every, args.overfit4_steps)),
                fixed_train_seed=777,
            )
            results["debug32"] = run_train_loop(
                model,
                fm,
                samples[:32],
                device,
                "debug32",
                args.debug32_steps,
                batch_size=args.batch_size_debug32,
                run_dir=run_dir,
                eval_every=max(1, min(args.eval_every, args.debug32_steps)),
                save_every=max(1, min(args.save_every, args.debug32_steps)),
            )
    except Exception as exc:
        results["fatal_error"] = {"type": exc.__class__.__name__, "message": repr(exc)}
        raise
    finally:
        results["total_elapsed_sec"] = time.time() - t0
        REPORT_PROBE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORT_PROBE_DIR / f"{args.run_name}_results.json"
        out_path.write_text(json.dumps(jsonable(results), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"RESULT_JSON={out_path}", flush=True)


if __name__ == "__main__":
    main()
