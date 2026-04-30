#!/usr/bin/env python
"""Merge Stage07 materialized data with Stage08B curated/hybrid supplements."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")


def load(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["samples"], data.get("manifest", {})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sample_key(sample: dict[str, Any]) -> str:
    # For training we keep multiple candidates from the same family/sequence
    # because they may represent different poses or state selections.  Leakage
    # is handled at split/benchmark time; here only exact duplicate sample IDs
    # should be removed.
    return str(sample.get("sample_id"))


def normalize_sample(sample: dict[str, Any], source_label: str) -> dict[str, Any]:
    sample = dict(sample)
    sample.setdefault("stage08b_merge_source", source_label)
    if sample.get("ae_latent_source") is None and sample.get("stage05_notes", {}).get("local_latents_source") == "complexa_ae_encoder_mean":
        sample["ae_latent_source"] = "complexa_ae_encoder_mean"
    if sample.get("ae_latent_source") is None and sample.get("stage08b_notes", {}).get("local_latents_source") == "complexa_ae_encoder_mean":
        sample["ae_latent_source"] = "complexa_ae_encoder_mean"
    return sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", type=Path, required=True)
    ap.add_argument("--source-label", action="append", default=[])
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/strategy01/stage08b_merged_training")
    ap.add_argument("--summary", type=Path, default=REPO_ROOT / "reports/strategy01/probes/stage08b_merge_summary.json")
    args = ap.parse_args()

    labels = list(args.source_label)
    while len(labels) < len(args.dataset):
        labels.append(f"source{len(labels)}")

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    for path, label in zip(args.dataset, labels, strict=False):
        samples, _ = load(path)
        for sample in samples:
            key = sample_key(sample)
            if key in seen:
                skipped["duplicate"] += 1
                continue
            if "x_1_states" not in sample or "local_latents" not in sample["x_1_states"]:
                skipped["missing_latents"] += 1
                continue
            lat = sample["x_1_states"]["local_latents"]
            if lat.shape[-1] != 8 or not torch.isfinite(lat).all():
                skipped["bad_latents"] += 1
                continue
            seen.add(key)
            merged.append(normalize_sample(sample, label))
            source_counts[label] += 1

    # Preserve existing split when available.  If a source has no val split,
    # hold out by family at roughly 20%.
    train = [s for s in merged if s.get("split") == "train"]
    val = [s for s in merged if s.get("split") == "val"]
    unsplit = [s for s in merged if s.get("split") not in {"train", "val"}]
    if unsplit:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in unsplit:
            groups[str(s.get("family_split_key") or s.get("split_group") or s.get("sample_id"))].append(s)
        target_val = max(1, int(round(0.2 * len(unsplit))))
        val_count = 0
        for group, group_samples in sorted(groups.items()):
            dest = val if val_count < target_val else train
            for s in group_samples:
                s["split"] = "val" if dest is val else "train"
                dest.append(s)
            if dest is val:
                val_count += len(group_samples)

    final_samples = train + val
    summary = {
        "status": "passed" if train and val else "short_dataset",
        "datasets": [str(p) for p in args.dataset],
        "source_counts": dict(source_counts),
        "skipped": dict(skipped),
        "total_count": len(final_samples),
        "train_count": len(train),
        "val_count": len(val),
        "families_train": len({str(s.get("family_split_key")) for s in train}),
        "families_val": len({str(s.get("family_split_key")) for s in val}),
        "notes": [
            "Merged set is Stage08B pilot data. It does not satisfy the desired 256/64 threshold unless train_count/val_count reach that target.",
            "Stage07 samples are predictor-derived bronze; PINDER samples are constrained-template hybrid silver.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tensor = args.out_dir / "stage08b_merged_pilot_samples.pt"
    train_tensor = args.out_dir / "T_stage08b_train.pt"
    val_tensor = args.out_dir / "T_stage08b_val.pt"
    torch.save({"samples": final_samples, "manifest": summary}, tensor)
    torch.save({"samples": train, "manifest": summary}, train_tensor)
    torch.save({"samples": val, "manifest": summary}, val_tensor)
    write_json(args.out_dir / "manifests/stage08b_merged_manifest.json", [{"sample_id": s["sample_id"], "split": s.get("split"), "source_tier": s.get("source_tier"), "family_split_key": s.get("family_split_key")} for s in final_samples])
    summary.update({"tensor_dataset": str(tensor), "train_tensor": str(train_tensor), "val_tensor": str(val_tensor)})
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
