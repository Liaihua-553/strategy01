import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer
from proteinfoundation.nn.local_latents_transformer_multistate import LocalLatentsTransformerMultistate
from proteinfoundation.utils.pdb_utils import load_target_from_pdb


def infer_latent_dim_from_ckpt(ckpt_path: str) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return int(ckpt["state_dict"]["nn.local_latents_linear.1.weight"].shape[0])


def make_binder_stub_batch(batch_size: int, binder_len: int, latent_dim: int, device: torch.device) -> dict[str, torch.Tensor]:
    bb_ca = torch.randn(batch_size, binder_len, 3, device=device) * 0.05
    local_latents = torch.randn(batch_size, binder_len, latent_dim, device=device) * 0.05
    batch = {
        "mask": torch.ones(batch_size, binder_len, dtype=torch.bool, device=device),
        "x_t": {"bb_ca": bb_ca.clone(), "local_latents": local_latents.clone()},
        "x_sc": {"bb_ca": torch.zeros_like(bb_ca), "local_latents": torch.zeros_like(local_latents)},
        "t": {
            "bb_ca": torch.full((batch_size,), 0.5, dtype=torch.float32, device=device),
            "local_latents": torch.full((batch_size,), 0.5, dtype=torch.float32, device=device),
        },
        "use_ca_coors_nm_feature": False,
        "use_residue_type_feature": False,
    }
    return batch


def make_synthetic_target_states(batch_size: int, nstates: int, target_len: int, device: torch.device) -> dict[str, torch.Tensor]:
    x_target_states = torch.randn(batch_size, nstates, target_len, 37, 3, device=device) * 0.1
    target_mask_states = torch.ones(batch_size, nstates, target_len, 37, dtype=torch.bool, device=device)
    seq_target_states = torch.randint(0, 20, (batch_size, nstates, target_len), device=device)
    target_hotspot_mask_states = torch.zeros(batch_size, nstates, target_len, dtype=torch.bool, device=device)
    target_hotspot_mask_states[:, 0, : min(4, target_len)] = True
    target_state_weights = torch.ones(batch_size, nstates, dtype=torch.float32, device=device) / max(nstates, 1)
    target_state_roles = torch.arange(1, nstates + 1, device=device).unsqueeze(0).expand(batch_size, -1)
    state_present_mask = torch.ones(batch_size, nstates, dtype=torch.bool, device=device)
    return {
        "x_target_states": x_target_states,
        "target_mask_states": target_mask_states,
        "seq_target_states": seq_target_states,
        "target_hotspot_mask_states": target_hotspot_mask_states,
        "seq_target_mask_states": target_mask_states.any(dim=-1),
        "target_state_weights": target_state_weights,
        "target_state_roles": target_state_roles,
        "state_present_mask": state_present_mask,
    }


def make_real_target_states(
    state_paths: list[str],
    input_spec: str,
    state_weights: list[float],
    state_roles: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    structures = []
    masks = []
    residue_types = []
    hotspots = []
    target_len = None
    for path in state_paths:
        target_mask, target_structure, target_residue_type, target_hotspot_mask, _ = load_target_from_pdb(
            input_spec,
            path,
            target_hotspots=None,
        )
        if target_len is None:
            target_len = int(target_structure.shape[0])
        assert int(target_structure.shape[0]) == target_len, "All target states must share aligned residue length"
        structures.append(target_structure)
        masks.append(target_mask)
        residue_types.append(target_residue_type)
        hotspots.append(target_hotspot_mask)

    x_target_states = torch.stack(structures, dim=0).unsqueeze(0).to(device)
    target_mask_states = torch.stack(masks, dim=0).unsqueeze(0).to(device)
    seq_target_states = torch.stack(residue_types, dim=0).unsqueeze(0).to(device)
    target_hotspot_mask_states = torch.stack(hotspots, dim=0).unsqueeze(0).to(device)
    target_state_weights = torch.tensor(state_weights, dtype=torch.float32, device=device).unsqueeze(0)
    target_state_roles = torch.tensor(state_roles, dtype=torch.long, device=device).unsqueeze(0)
    state_present_mask = torch.ones(1, len(state_paths), dtype=torch.bool, device=device)
    return {
        "x_target_states": x_target_states,
        "target_mask_states": target_mask_states,
        "seq_target_states": seq_target_states,
        "target_hotspot_mask_states": target_hotspot_mask_states,
        "seq_target_mask_states": target_mask_states.any(dim=-1),
        "target_state_weights": target_state_weights,
        "target_state_roles": target_state_roles,
        "state_present_mask": state_present_mask,
    }


def tensor_shape(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, dict):
        return {k: tensor_shape(v) for k, v in value.items()}
    return value


def summarize_forward(output: dict) -> dict:
    summary = {}
    for key, value in output.items():
        if key == "arch_debug":
            summary[key] = {
                subkey: tensor_shape(subval)
                for subkey, subval in value.items()
                if isinstance(subval, torch.Tensor)
            }
        else:
            summary[key] = tensor_shape(value)
    return summary


def strip_nn_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("nn."):
            stripped[key[len("nn.") :]] = value
    return stripped


def count_shape_compatible_keys(model: torch.nn.Module, nn_state: dict[str, torch.Tensor]) -> tuple[int, list[str], list[str]]:
    model_state = model.state_dict()
    matched = []
    mismatched = []
    for key, value in nn_state.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            matched.append(key)
        elif key in model_state:
            mismatched.append(key)
    return len(matched), matched, mismatched


def main():
    cfg = OmegaConf.load("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase/configs/training_local_latents_multistate_probe.yaml")
    torch.manual_seed(int(cfg.seed))
    device = torch.device("cpu")

    latent_dim = infer_latent_dim_from_ckpt(cfg.baseline_ckpt_path)
    multistate_nn_cfg = OmegaConf.to_container(OmegaConf.load(cfg.nn_config_path), resolve=True)
    baseline_nn_cfg = OmegaConf.to_container(
        OmegaConf.load(f"{cfg.baseline_repo_root}/configs/nn/local_latents_score_nn_160M.yaml"),
        resolve=True,
    )

    baseline_model = LocalLatentsTransformer(**baseline_nn_cfg, latent_dim=latent_dim).to(device)
    multistate_model = LocalLatentsTransformerMultistate(**multistate_nn_cfg, latent_dim=latent_dim).to(device)
    baseline_model.eval()
    multistate_model.eval()

    binder_stub = make_binder_stub_batch(cfg.probe.batch_size, cfg.probe.binder_len, latent_dim, device)
    synthetic_targets = make_synthetic_target_states(
        cfg.probe.batch_size,
        cfg.probe.synthetic_nstates,
        cfg.probe.synthetic_target_len,
        device,
    )
    synthetic_batch = {**binder_stub, **synthetic_targets}

    real_targets = make_real_target_states(
        list(cfg.probe.real_state_paths),
        cfg.probe.real_input_spec,
        list(cfg.probe.real_state_weights),
        list(cfg.probe.real_state_roles),
        device,
    )
    real_batch = {**make_binder_stub_batch(1, cfg.probe.binder_len, latent_dim, device), **real_targets}

    with torch.no_grad():
        baseline_out = baseline_model(binder_stub)
        p1_synthetic_out = multistate_model(synthetic_batch)
        p1_real_out = multistate_model(real_batch)

    ckpt = torch.load(cfg.baseline_ckpt_path, map_location="cpu")
    nn_state = strip_nn_prefix(ckpt["state_dict"])
    shape_match_count, matched_keys, mismatched_keys = count_shape_compatible_keys(multistate_model, nn_state)
    shape_compatible_state = {key: nn_state[key] for key in matched_keys}
    incompatible = multistate_model.load_state_dict(shape_compatible_state, strict=False)

    with torch.no_grad():
        p2_loaded_real_out = multistate_model(real_batch)

    multistate_model.train()
    p3_out = multistate_model(synthetic_batch)
    loss = (
        p3_out["seq_logits_shared"].pow(2).mean()
        + p3_out["bb_ca_states"].pow(2).mean()
        + p3_out["local_latents_states"].pow(2).mean()
    )
    loss.backward()

    grad_names = [
        "ensemble_target_encoder.target_feature_projection.1.weight",
        "ensemble_target_encoder.cross_state_fusion.q_proj.weight",
        "shared_seq_head.1.weight",
        "ca_linear.1.weight",
        "local_latents_linear.1.weight",
    ]
    named_params = dict(multistate_model.named_parameters())
    grad_norms = {}
    for name in grad_names:
        grad = named_params[name].grad
        grad_norms[name] = None if grad is None else float(grad.norm().item())

    results = {
        "probe_date": "2026-04-12",
        "baseline_commit": "2db8e1df838354db079ce8e0e4b88aaebd31f35f",
        "latent_dim_inferred_from_checkpoint": latent_dim,
        "p0_static": {
            "baseline_model_class": baseline_model.__class__.__name__,
            "multistate_model_class": multistate_model.__class__.__name__,
            "single_state_target_feature_dim": int(multistate_model.ensemble_target_encoder.single_state_feat_dim),
            "token_dim": int(multistate_model.token_dim),
            "pair_repr_dim": int(multistate_model.pair_repr_dim),
            "cross_state_heads": int(multistate_model.ensemble_target_encoder.cross_state_fusion.nheads),
        },
        "baseline_forward_shapes": summarize_forward(baseline_out),
        "p1_synthetic_forward_shapes": summarize_forward(p1_synthetic_out),
        "p1_real_forward_shapes": summarize_forward(p1_real_out),
        "p2_checkpoint_loading": {
            "shape_compatible_key_count": shape_match_count,
            "shape_compatible_sample": matched_keys[:20],
            "shape_mismatched_sample": mismatched_keys[:20],
            "missing_key_count": len(incompatible.missing_keys),
            "unexpected_key_count": len(incompatible.unexpected_keys),
            "missing_key_sample": incompatible.missing_keys[:30],
            "unexpected_key_sample": incompatible.unexpected_keys[:30],
            "loaded_real_forward_shapes": summarize_forward(p2_loaded_real_out),
        },
        "p3_backward": {
            "loss": float(loss.item()),
            "gradient_norms": grad_norms,
        },
    }

    report_path = Path(cfg.probe.report_json_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
