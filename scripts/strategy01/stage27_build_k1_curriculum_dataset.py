#!/usr/bin/env python
"""Build a K=1 state-level curriculum dataset from curated multistate samples.

Stage27 is a diagnostic curriculum, not a new data source.  Each output sample
keeps only one target state and the matching binder supervision from an
accepted multistate sample.  The true binder sequence and complex geometry stay
as labels for the loss; they are not model input in de_novo_multistate mode.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch


STATE_LIST_KEYS = {
    "target_state_paths",
    "target_state_chain_ids",
    "predicted_complex_paths",
    "state_roles",
    "state_weights",
    "state_metrics",
}

STATE_TENSOR_KEYS = {
    "x_target_states",
    "target_mask_states",
    "seq_target_states",
    "target_hotspot_mask_states",
    "state_mask",
    "state_present_mask",
    "target_state_weights",
    "target_state_roles",
    "interface_contact_labels",
    "interface_distance_labels",
    "interface_label_mask",
    "interface_quality_labels",
    "interface_quality_mask",
}


def _slice_tensor(value: torch.Tensor, state_idx: int) -> torch.Tensor:
    if value.ndim == 0:
        return value.clone()
    if value.shape[0] <= state_idx:
        raise IndexError(f"state_idx={state_idx} out of range for shape={tuple(value.shape)}")
    return value[state_idx : state_idx + 1].clone()


def _slice_state_value(key: str, value: Any, state_idx: int) -> Any:
    if key in STATE_LIST_KEYS and isinstance(value, list):
        return [copy.deepcopy(value[state_idx])]
    if key in STATE_TENSOR_KEYS and torch.is_tensor(value):
        return _slice_tensor(value, state_idx)
    return copy.deepcopy(value)


def make_k1_sample(sample: dict[str, Any], state_idx: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in sample.items():
        if key == "x_1_states":
            out[key] = {
                sub_key: _slice_tensor(sub_value, state_idx)
                for sub_key, sub_value in value.items()
            }
        else:
            out[key] = _slice_state_value(key, value, state_idx)

    out["sample_id"] = f"{sample.get('sample_id', 'sample')}__state{state_idx}_k1curr"
    out["stage27_curriculum_source_sample_id"] = sample.get("sample_id")
    out["stage27_curriculum_state_idx"] = int(state_idx)
    out["stage27_curriculum_k"] = 1
    out["source_tier"] = f"{sample.get('source_tier', 'unknown')}|k1_curriculum"
    out["state_present_mask"] = torch.ones(1, dtype=torch.bool)

    if torch.is_tensor(out.get("target_state_weights")):
        out["target_state_weights"] = torch.ones_like(out["target_state_weights"], dtype=torch.float32)
    if isinstance(out.get("state_weights"), list):
        out["state_weights"] = [1.0]
    if torch.is_tensor(out.get("target_state_roles")):
        out["target_state_roles"] = out["target_state_roles"].clone()
    if isinstance(out.get("state_roles"), list) and out["state_roles"]:
        out["state_roles"] = [out["state_roles"][0]]
    return out


def build_dataset(input_path: Path, output_path: Path) -> dict[str, Any]:
    # This script only reads Strategy01-generated trusted tensor bundles.  Pass
    # weights_only explicitly so future PyTorch default changes do not make the
    # stage emit misleading security warnings.
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    samples = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    if not isinstance(samples, list):
        raise TypeError("Expected a list of samples or a dict containing samples")

    out_samples: list[dict[str, Any]] = []
    source_counts = {"train": 0, "val": 0, "other": 0}
    for sample in samples:
        present = sample.get("state_present_mask")
        if torch.is_tensor(present):
            state_indices = [i for i, flag in enumerate(present.bool().tolist()) if flag]
        else:
            nstates = len(sample.get("target_state_paths", []))
            state_indices = list(range(nstates))
        for state_idx in state_indices:
            out_samples.append(make_k1_sample(sample, state_idx))
            split = str(sample.get("split", "other"))
            source_counts[split if split in source_counts else "other"] += 1

    manifest = copy.deepcopy(payload.get("manifest", {}) if isinstance(payload, dict) else {})
    manifest.update(
        {
            "stage": "stage27_k1_curriculum",
            "source_dataset": str(input_path),
            "samples": len(out_samples),
            "source_counts": source_counts,
            "scientific_role": (
                "Curriculum gate for target-only de novo generation: first prove "
                "state-wise native Complexa flow can keep sampled local latents "
                "and shared sequence readout stable before re-coupling K>1 states."
            ),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": out_samples, "manifest": manifest}, output_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/strategy01/stage10_exactaug_training/stage10_exactaug_trainval.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/strategy01/stage27_k1_curriculum/stage27_k1_trainval.pt"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("reports/strategy01/probes/stage27_k1_curriculum_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(args.input, args.output)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(args.output), "summary": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
