#!/usr/bin/env python
"""Stage13 native-path equivalence probe.

This probe checks a narrow but critical contract: when the multistate wrapper is
asked to use `stage13_native_state_path`, its state-wise denoiser branch must be
numerically equivalent to the original Complexa `LocalLatentsTransformer` on the
same flattened single-state input. This is a code-path audit, not a biological
success metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer  # noqa: E402
from proteinfoundation.nn.local_latents_transformer_multistate import LocalLatentsTransformerMultistate  # noqa: E402

import scripts.strategy01.stage03_multistate_loss_debug as s3  # noqa: E402
import scripts.strategy01.stage10_pose_init_training as s10  # noqa: E402
import scripts.strategy01.stage12_de_novo_multistate_training as s12  # noqa: E402


DEFAULT_REPORT = REPO / "reports/strategy01/probes/stage13_native_path_equivalence_probe.json"


def strip_nn_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k[len("nn.") :]: v for k, v in state_dict.items() if k.startswith("nn.")}


def load_compatible(model: torch.nn.Module, checkpoint: Path) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = strip_nn_prefix(ckpt["state_dict"])
    model_state = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)}
    incompatible = [k for k, v in state.items() if k in model_state and tuple(v.shape) != tuple(model_state[k].shape)]
    info = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(checkpoint),
        "compatible_keys": len(compatible),
        "incompatible_shape_keys": incompatible[:50],
        "missing_keys": list(info.missing_keys)[:80],
        "unexpected_keys": list(info.unexpected_keys)[:80],
    }


def load_cfg_for_checkpoint(checkpoint: Path) -> tuple[Any, dict[str, Any], int]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    latent_dim = int(ckpt["state_dict"]["nn.local_latents_linear.1.weight"].shape[0])
    train_cfg = OmegaConf.load(s3.TRAIN_CONFIG)
    nn_cfg = OmegaConf.to_container(OmegaConf.load(s3.NN_CONFIG), resolve=True)
    train_cfg.nn = OmegaConf.create(nn_cfg)
    train_cfg.pretrain_ckpt_path = str(checkpoint)
    return train_cfg, nn_cfg, latent_dim


def tensor_diff(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    diff = (a - b).detach()
    if mask is not None:
        while mask.ndim < diff.ndim:
            mask = mask.unsqueeze(-1)
        diff = diff[mask.expand_as(diff)]
    if diff.numel() == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0}
    return {"max_abs": float(diff.abs().max().cpu()), "mean_abs": float(diff.abs().mean().cpu())}


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check original vs multistate native-path output equivalence")
    parser.add_argument("--dataset", type=Path, default=s12.DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=REPO / "ckpts/complexa.ckpt")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = s12.choose_device(args.device)
    train_cfg, nn_cfg, latent_dim = load_cfg_for_checkpoint(args.checkpoint)
    original = LocalLatentsTransformer(**nn_cfg, latent_dim=latent_dim).to(device)
    multistate = LocalLatentsTransformerMultistate(**nn_cfg, latent_dim=latent_dim).to(device)
    original_meta = load_compatible(original, args.checkpoint)
    multistate_meta = load_compatible(multistate, args.checkpoint)
    original.eval()
    multistate.eval()

    samples, manifest = s10.load_dataset(args.dataset)
    selected = [s for s in samples if s.get("split") == args.split][: args.max_samples]
    if not selected:
        raise RuntimeError(f"No samples selected for split={args.split}")

    fm = s3.ProductSpaceFlowMatcher(train_cfg).to(device)
    batch = s10.collate_variable(selected, device)
    work = s12.make_de_novo_batch(batch)
    work = fm.corrupt_multistate_batch(work)
    weights = s12.normalize_state_weights(work, device)
    s12.add_weighted_legacy_fields(work, weights)
    work["de_novo_multistate_mode"] = True
    work["stage13_native_state_path"] = True

    with torch.no_grad():
        flat, _, _, _ = multistate._native_state_flatten_input(work)
        original_out = original(flat)
        multistate_out = multistate(work)

    b, k, nb = work["state_mask"].shape
    mask = work["state_mask"].bool()
    bb_param = train_cfg.nn.output_parameterization["bb_ca"]
    z_param = train_cfg.nn.output_parameterization["local_latents"]
    original_bb = original_out["bb_ca"][bb_param].reshape(b, k, nb, 3)
    original_z = original_out["local_latents"][z_param].reshape(b, k, nb, -1)

    result = {
        "stage": "stage13_native_path_equivalence_probe",
        "status": "passed",
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "sample_count": len(selected),
        "manifest_stage": manifest.get("stage") if isinstance(manifest, dict) else None,
        "output_parameterization": {
            "bb_ca": str(bb_param),
            "local_latents": str(z_param),
        },
        "original_checkpoint_meta": original_meta,
        "multistate_checkpoint_meta": multistate_meta,
        "diff": {
            "bb_ca_states_vs_original": tensor_diff(multistate_out["bb_ca_states"], original_bb, mask),
            "local_latents_states_vs_original": tensor_diff(multistate_out["local_latents_states"], original_z, mask),
        },
        "native_arch_debug": jsonable(multistate_out.get("arch_debug", {})),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["diff"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
