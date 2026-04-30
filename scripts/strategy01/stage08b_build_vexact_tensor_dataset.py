#!/usr/bin/env python
"""Build a generation-ready Strategy01 tensor dataset from V_exact manifest.

The Stage08 V_exact manifest contains exact experimental multi-model/state
complexes, but the sampler/training code consumes tensor samples.  This script
materializes those exact complexes into the same schema used by Stage07/08B:
target ensemble tensors, one shared binder sequence, K exact binder labels,
interface contact/distance labels, and geometry-proxy local latents.  The AE
encoder is run as a separate step to replace the proxy latents.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.strategy01.stage04_build_real_multistate_complex_dataset import (  # noqa: E402
    ATOM_ORDER,
    chain_to_atom37,
    geometry_latents,
    get_chain_residues,
    interface_labels,
)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def as_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def pad_tensor_first_dim(t: torch.Tensor, length: int, value: float | bool | int = 0) -> torch.Tensor:
    if t.shape[0] >= length:
        return t[:length]
    out = torch.full((length,) + tuple(t.shape[1:]), value, dtype=t.dtype)
    out[: t.shape[0]] = t
    return out


def pad_pair_tensor(t: torch.Tensor, target_len: int, binder_len: int, value: float | bool = 0) -> torch.Tensor:
    out = torch.full((target_len, binder_len), value, dtype=t.dtype)
    out[: min(target_len, t.shape[0]), : min(binder_len, t.shape[1])] = t[:target_len, :binder_len]
    return out


def ca_coords(chain: Any) -> torch.Tensor:
    return chain.x[:, ATOM_ORDER["CA"]].float()


def chain_length(path: Path, chain_id: str) -> int:
    return len(get_chain_residues(path, chain_id))


def build_sample(item: dict[str, Any], out_dir: Path, contact_cutoff_nm: float) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    sample_id = str(item["sample_id"])
    target_chain = str(item.get("target_chain_id") or "A")
    binder_chain = str(item.get("binder_chain_id") or "B")
    target_paths = [as_path(p) for p in item.get("target_state_paths") or []]
    complex_paths = [as_path(p) for p in item.get("exact_complex_paths") or []]
    if not target_paths or len(target_paths) != len(complex_paths):
        return None, {"sample_id": sample_id, "reject": "missing_state_or_complex_paths"}

    targets = []
    binders = []
    labels = []
    errors: list[str] = []
    binder_len_ref: int | None = None
    for k, (target_path, complex_path) in enumerate(zip(target_paths, complex_paths)):
        try:
            target_len = chain_length(target_path, target_chain)
            binder_len = chain_length(complex_path, binder_chain)
            if target_len < 3 or binder_len < 3:
                raise ValueError(f"too_short target={target_len} binder={binder_len}")
            if binder_len_ref is None:
                binder_len_ref = binder_len
            elif binder_len != binder_len_ref:
                # Exact multistate supervision assumes one shared binder length.
                raise ValueError(f"binder_len_mismatch state={k} got={binder_len} ref={binder_len_ref}")
            target = chain_to_atom37(target_path, target_chain, 0, target_len)
            binder = chain_to_atom37(complex_path, binder_chain, 0, binder_len)
            lab = interface_labels(target, binder, contact_cutoff_nm=contact_cutoff_nm)
            targets.append(target)
            binders.append(binder)
            labels.append(lab)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"state{k}:{type(exc).__name__}:{exc}")
    if errors or not targets or binder_len_ref is None:
        return None, {"sample_id": sample_id, "reject": "parse_error", "errors": errors}

    max_target_len = max(t.x.shape[0] for t in targets)
    binder_len = binder_len_ref
    state_count = len(targets)
    x_target_states = []
    target_mask_states = []
    seq_target_states = []
    target_hotspot_mask_states = []
    bb_states = []
    lat_states = []
    contact_tensors = []
    distance_tensors = []
    label_masks = []
    quality_labels = []
    state_metrics = []
    contact_counts = []
    severe_clash_flags = []
    for k, (target, binder, lab) in enumerate(zip(targets, binders, labels)):
        x_target_states.append(pad_tensor_first_dim(target.x, max_target_len, 0.0))
        target_mask_states.append(pad_tensor_first_dim(target.mask, max_target_len, False))
        seq_target_states.append(pad_tensor_first_dim(target.seq, max_target_len, 0))
        hotspot = lab["contact_labels"].bool().any(dim=1)
        target_hotspot_mask_states.append(pad_tensor_first_dim(hotspot, max_target_len, False))
        bb_states.append(ca_coords(binder))
        lat_states.append(geometry_latents(ca_coords(binder), binder.seq))
        contact_tensors.append(pad_pair_tensor(lab["contact_labels"].float(), max_target_len, binder_len, 0.0))
        distance_tensors.append(pad_pair_tensor(lab["distance_labels"].float(), max_target_len, binder_len, 0.0))
        label_masks.append(pad_pair_tensor(lab["label_mask"].bool(), max_target_len, binder_len, False))
        quality_labels.append(lab["quality_labels"].float())
        metrics = dict(lab["metrics"])
        contact_counts.append(int(metrics.get("contact_count", 0)))
        severe_clash_flags.append(bool(metrics.get("severe_clash", False)))
        state_metrics.append(
            {
                "state_index": k,
                "state_method": (item.get("state_methods") or [None] * state_count)[k],
                "contact_count": int(metrics.get("contact_count", 0)),
                "min_distance_nm": float(metrics.get("min_distance_nm", 9.9)),
                "severe_clash": bool(metrics.get("severe_clash", False)),
                "source_label_type": "exact_experimental_complex",
            }
        )

    # Persistent anchors across exact states are the biologically important
    # interface core that the shared sequence must support.
    anchor_sets = []
    for lab in labels:
        idx = torch.nonzero(lab["contact_labels"] > 0.5, as_tuple=False)
        anchor_sets.append({(int(i), int(j)) for i, j in idx.tolist()})
    persistent = set.intersection(*anchor_sets) if anchor_sets else set()
    union = set.union(*anchor_sets) if anchor_sets else set()
    contact_persistence = len(persistent) / max(1, len(union)) if union else 0.0

    state_weights = torch.tensor(item.get("state_weights") or [1.0 / state_count] * state_count, dtype=torch.float32)
    state_weights = state_weights / state_weights.sum().clamp_min(1e-8)
    binder_seq = binders[0].seq
    sample = {
        "sample_id": sample_id,
        "target_id": item.get("target_id"),
        "family_split_key": item.get("family_holdout_key") or item.get("split_group") or sample_id,
        "split_group": item.get("split_group") or item.get("family_holdout_key") or sample_id,
        "source_dataset": "stage08_v_exact_manifest",
        "source_tier": "exact_experimental",
        "predictor": "none_exact_experimental",
        "quality_tier": item.get("quality_tier", "silver"),
        "shared_binder_sequence": item.get("shared_binder_sequence") or binders[0].seq_str,
        "binder_chain_id": binder_chain,
        "target_chain_id": target_chain,
        "target_state_chain_ids": [target_chain] * state_count,
        "target_state_paths": [str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p) for p in target_paths],
        "predicted_complex_paths": [str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p) for p in complex_paths],
        "exact_complex_paths": [str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p) for p in complex_paths],
        "exact_or_hybrid_complex_paths": [str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p) for p in complex_paths],
        "state_roles": item.get("state_roles") or ["required_bind"] * state_count,
        "state_weights": [float(x) for x in state_weights.tolist()],
        "leakage_keys": {
            "target_id": item.get("target_id"),
            "sample_id": sample_id,
            "source_db": item.get("source_db"),
            "binder_sequence": item.get("shared_binder_sequence"),
        },
        "x_target_states": torch.stack(x_target_states),
        "target_mask_states": torch.stack(target_mask_states),
        "seq_target_states": torch.stack(seq_target_states),
        "target_hotspot_mask_states": torch.stack(target_hotspot_mask_states),
        "binder_seq_shared": binder_seq,
        "binder_seq_mask": torch.ones_like(binder_seq, dtype=torch.bool),
        "x_1_states": {"bb_ca": torch.stack(bb_states), "local_latents": torch.stack(lat_states)},
        "state_mask": torch.ones(state_count, binder_len, dtype=torch.bool),
        "state_present_mask": torch.ones(state_count, dtype=torch.bool),
        "target_state_weights": state_weights,
        "target_state_roles": torch.zeros(state_count, dtype=torch.long),
        "interface_contact_labels": torch.stack(contact_tensors),
        "interface_distance_labels": torch.stack(distance_tensors),
        "interface_label_mask": torch.stack(label_masks),
        "interface_quality_labels": torch.stack(quality_labels),
        "interface_quality_mask": torch.ones(state_count, 5, dtype=torch.bool),
        "state_metrics": state_metrics,
        "worst_state_metrics": {
            "worst_contact_count": min(contact_counts),
            "persistent_anchor_count": len(persistent),
            "contact_persistence": contact_persistence,
            "any_severe_clash": any(severe_clash_flags),
            "target_motion_rmsd_A": (item.get("motion_metrics") or {}).get("target_backbone_rmsd_A"),
        },
        "stage08b_notes": {
            "label_construction": "exact_experimental_complex_from_v_exact_manifest",
            "local_latents_source": "geometry_proxy_pending_ae",
            "contact_cutoff_A": contact_cutoff_nm * 10.0,
        },
        "ae_latent_source": None,
        "split": "val",
    }
    manifest = {
        "sample_id": sample_id,
        "target_id": item.get("target_id"),
        "source_tier": sample["source_tier"],
        "k_states": state_count,
        "target_len_max": max_target_len,
        "binder_len": binder_len,
        "persistent_anchor_count": len(persistent),
        "contact_persistence": contact_persistence,
        "min_contact_count": min(contact_counts),
        "motion_bin": item.get("motion_bin"),
        "target_motion_rmsd_A": (item.get("motion_metrics") or {}).get("target_backbone_rmsd_A"),
        "split": "val",
    }
    return sample, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO / "data/strategy01/stage08_high_quality_dataset/V_exact_main_manifest.json")
    parser.add_argument("--out-dir", type=Path, default=REPO / "data/strategy01/stage08b_vexact_tensor")
    parser.add_argument("--summary", type=Path, default=REPO / "reports/strategy01/probes/stage08b_vexact_tensor_build_summary.json")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--contact-cutoff-nm", type=float, default=0.8)
    args = parser.parse_args()

    started = time.time()
    items = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.max_samples > 0:
        items = items[: args.max_samples]
    samples = []
    manifests = []
    rejects = []
    for item in items:
        sample, info = build_sample(item, args.out_dir, args.contact_cutoff_nm)
        if sample is None:
            rejects.append(info)
            continue
        manifest = info
        samples.append(sample)
        manifests.append(manifest)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = args.out_dir / "stage08b_vexact_samples.pt"
    val_path = args.out_dir / "V_exact_val.pt"
    manifest_path = args.out_dir / "V_exact_tensor_manifest.json"
    summary = {
        "status": "passed" if samples else "empty",
        "source_manifest": str(args.manifest),
        "sample_count": len(samples),
        "reject_count": len(rejects),
        "rejects": rejects[:50],
        "val_count": len(samples),
        "families_val": len({s["family_split_key"] for s in samples}),
        "tensor_dataset": str(tensor_path),
        "val_tensor": str(val_path),
        "tensor_manifest": str(manifest_path),
        "elapsed_sec": time.time() - started,
        "notes": [
            "All samples are split=val/external benchmark; they are not used as training data in Stage08B.",
            "local_latents are geometry proxy until stage05_extract_ae_latents.py replaces them with Complexa AE encoder.mean.",
        ],
    }
    torch.save({"samples": samples, "manifest": summary}, tensor_path)
    torch.save({"samples": samples, "manifest": summary}, val_path)
    write_json(manifest_path, manifests)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
