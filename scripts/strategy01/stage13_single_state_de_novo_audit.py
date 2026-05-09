#!/usr/bin/env python3
"""Stage13 K=1 target-only de-novo audit.

This audit slices multistate samples into single-state target-only cases and
runs the same Stage12 de-novo rollout.  It answers a narrow question before
more multistate tuning: does the Strategy01 architecture still preserve
single-state target-conditioned de-novo generation behavior?

No source pose, true binder optional CA feature, or residue-type feature is
used as model input.  The true binder sequence/complex tensors remain labels
for evaluation only.
"""

from __future__ import annotations

import argparse
import copy
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
import scripts.strategy01.stage12_de_novo_multistate_training as s12  # noqa: E402
import scripts.strategy01.stage12c_de_novo_smoke as s12smoke  # noqa: E402


DEFAULT_REPORT = REPO / "reports/strategy01/probes/stage13_single_state_de_novo_audit.json"

STATE_TENSOR_KEYS = {
    "x_target_states",
    "target_mask_states",
    "seq_target_states",
    "target_hotspot_mask_states",
    "target_state_weights",
    "target_state_roles",
    "state_present_mask",
    "state_mask",
    "interface_contact_labels",
    "interface_distance_labels",
    "interface_label_mask",
    "interface_quality_labels",
    "interface_quality_mask",
}

STATE_LIST_KEYS = {
    "target_state_paths",
    "target_state_chain_ids",
    "state_roles",
    "state_weights",
    "state_metrics",
    "predicted_complex_paths",
}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s4.jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def slice_state_sample(sample: dict[str, Any], state_idx: int) -> dict[str, Any]:
    """Return a copy of one sample with exactly one target state.

    Only state-axis fields are sliced.  Binder sequence, binder mask, target id,
    and other scalar metadata are preserved.
    """
    k = int(sample["state_present_mask"].shape[0])
    if state_idx < 0 or state_idx >= k:
        raise IndexError(f"state_idx={state_idx} out of range K={k}")
    if not bool(sample["state_present_mask"][state_idx].item()):
        raise ValueError(f"state_idx={state_idx} is not present for sample {sample.get('sample_id')}")

    out: dict[str, Any] = {}
    for key, value in sample.items():
        if key == "x_1_states":
            out[key] = {dm: tensor[state_idx : state_idx + 1].clone() for dm, tensor in value.items()}
        elif key in STATE_TENSOR_KEYS and torch.is_tensor(value):
            out[key] = value[state_idx : state_idx + 1].clone()
        elif key in STATE_LIST_KEYS and isinstance(value, list) and len(value) == k:
            out[key] = [copy.deepcopy(value[state_idx])]
        else:
            out[key] = copy.deepcopy(value)

    out["sample_id"] = f"{sample.get('sample_id', 'sample')}__single_state_{state_idx}"
    out["stage13_single_state_source_sample_id"] = sample.get("sample_id")
    out["stage13_single_state_index"] = int(state_idx)
    out["target_state_weights"] = torch.ones_like(out["target_state_weights"], dtype=torch.float32)
    out["state_present_mask"] = torch.ones_like(out["state_present_mask"], dtype=torch.bool)
    out["target_state_roles"] = torch.zeros_like(out["target_state_roles"], dtype=torch.long)
    return out


def select_single_state_cases(
    samples: list[dict[str, Any]],
    split: str,
    max_source_samples: int,
    states: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    source_count = 0
    for sample in samples:
        if sample.get("split") != split:
            continue
        present = sample["state_present_mask"].bool()
        indices = [int(i) for i in torch.where(present)[0].tolist()]
        if states == "first":
            indices = indices[:1]
        for state_idx in indices:
            selected.append(slice_state_sample(sample, state_idx))
        source_count += 1
        if source_count >= max_source_samples:
            break
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage13 K=1 de-novo audit for Strategy01")
    parser.add_argument("--dataset", type=Path, default=s12.DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ae-ckpt", type=Path, default=s12.DEFAULT_AE_CKPT)
    parser.add_argument("--sampling-config", type=Path, default=s12.DEFAULT_SAMPLING_CONFIG)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-source-samples", type=int, default=8)
    parser.add_argument("--states", choices=["first", "all"], default="all")
    parser.add_argument("--nsteps", type=int, default=16)
    parser.add_argument("--local-latents-stop-t", type=float, default=None)
    parser.add_argument("--target-shell-max-center-distance-nm", type=float, default=0.0)
    parser.add_argument(
        "--native-state-path",
        action="store_true",
        help="Diagnostic only: run K=1 through native Complexa target-concat denoiser inside the multistate wrapper.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1313)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = s12.choose_device(args.device)
    samples, manifest = s10.load_dataset(args.dataset)
    selected = select_single_state_cases(samples, args.split, args.max_source_samples, args.states)
    if not selected:
        raise RuntimeError(f"No single-state cases selected for split={args.split}")

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
    smoke = s12smoke.rollout_final(
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
        stage13_native_state_path=args.native_state_path,
    )
    result = {
        "stage": "stage13_single_state_de_novo_audit",
        "status": "passed",
        "scientific_contract": {
            "target_only": True,
            "k": 1,
            "forbidden_source_pose": True,
            "forbidden_true_binder_optional_features": True,
            "purpose": "Audit whether Strategy01 preserves single-state target-conditioned de-novo generation before further multistate tuning.",
            "native_state_path": bool(args.native_state_path),
        },
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "split": args.split,
        "source_sample_count": args.max_source_samples,
        "single_state_case_count": len(selected),
        "states": args.states,
        "model_meta": model_meta,
        "checkpoint_meta": ckpt_meta,
        "loss_cfg": loss_cfg,
        "manifest_stage": manifest.get("stage") if isinstance(manifest, dict) else None,
        "nsteps": args.nsteps,
        "local_latents_stop_t": args.local_latents_stop_t,
        "target_shell_max_center_distance_nm": args.target_shell_max_center_distance_nm,
        "native_state_path": bool(args.native_state_path),
    }
    result.update(smoke)
    write_json(args.report_json, result)
    print(json.dumps(s4.jsonable(result["identity"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
