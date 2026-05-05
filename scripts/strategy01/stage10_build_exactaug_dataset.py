#!/usr/bin/env python
"""Stage10 exact-augmented train/val dataset builder.

This merges Stage08B predictor/hybrid training data with experimental V_exact
samples.  V_exact is split by family/split_group so it can be used as stronger
training signal while still leaving an exact holdout for diagnostics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
DEFAULT_TRAIN = REPO / "data/strategy01/stage08b_merged_training/T_stage08b_train.pt"
DEFAULT_VAL = REPO / "data/strategy01/stage08b_merged_training/T_stage08b_val.pt"
DEFAULT_EXACT = REPO / "data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt"
DEFAULT_OUT = REPO / "data/strategy01/stage10_exactaug_training/stage10_exactaug_trainval.pt"
DEFAULT_SUMMARY = REPO / "reports/strategy01/probes/stage10_exactaug_dataset_summary.json"


def load_samples(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["samples"], data.get("manifest", {})


def stable_group(sample: dict[str, Any]) -> str:
    return str(sample.get("split_group") or sample.get("family_split_key") or sample.get("target_id") or sample.get("sample_id"))


def group_hash(group: str) -> float:
    h = hashlib.sha1(group.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / float(0xFFFFFFFF)


def tensor_shape(sample: dict[str, Any], key: str) -> list[int] | None:
    v = sample.get(key)
    return list(v.shape) if torch.is_tensor(v) else None


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    binder_lens = [int(s["binder_seq_shared"].shape[0]) for s in samples]
    target_lens = [int(s["x_target_states"].shape[1]) for s in samples]
    k_counts = [int(s["state_present_mask"].bool().sum().item()) for s in samples]
    return {
        "count": len(samples),
        "splits": dict(Counter(str(s.get("split", "?")) for s in samples)),
        "source_tier": dict(Counter(str(s.get("source_tier", "?")) for s in samples).most_common()),
        "binder_len_minmax": [min(binder_lens), max(binder_lens)] if binder_lens else None,
        "target_len_minmax": [min(target_lens), max(target_lens)] if target_lens else None,
        "K": dict(Counter(k_counts)),
        "family_count": len({stable_group(s) for s in samples}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--val", type=Path, default=DEFAULT_VAL)
    ap.add_argument("--exact", type=Path, default=DEFAULT_EXACT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--exact-train-fraction", type=float, default=0.75)
    args = ap.parse_args()

    train, train_manifest = load_samples(args.train)
    val, val_manifest = load_samples(args.val)
    exact, exact_manifest = load_samples(args.exact)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in exact:
        groups[stable_group(sample)].append(sample)

    exact_train_groups = []
    exact_val_groups = []
    for group in sorted(groups):
        if group_hash(group) < args.exact_train_fraction:
            exact_train_groups.append(group)
        else:
            exact_val_groups.append(group)
    if not exact_val_groups and exact_train_groups:
        exact_val_groups.append(exact_train_groups.pop())
    if not exact_train_groups and exact_val_groups:
        exact_train_groups.append(exact_val_groups.pop())

    exact_train = []
    exact_val = []
    for group in exact_train_groups:
        for sample in groups[group]:
            item = copy.deepcopy(sample)
            item["split"] = "train"
            item["stage10_exactaug_role"] = "exact_train"
            item["stage10_original_split"] = sample.get("split")
            exact_train.append(item)
    for group in exact_val_groups:
        for sample in groups[group]:
            item = copy.deepcopy(sample)
            item["split"] = "val"
            item["stage10_exactaug_role"] = "exact_holdout_val"
            item["stage10_original_split"] = sample.get("split")
            exact_val.append(item)

    merged_train = [copy.deepcopy(s) for s in train] + exact_train
    merged_val = [copy.deepcopy(s) for s in val] + exact_val
    for s in merged_train:
        s["split"] = "train"
    for s in merged_val:
        s["split"] = "val"
    merged = merged_train + merged_val

    manifest = {
        "status": "passed",
        "stage": "stage10_exactaug",
        "notes": [
            "Experimental V_exact is intentionally allowed into Stage10 training to strengthen exact interface supervision.",
            "Exact holdout remains family/split_group separated for diagnostics; it is not a fully external benchmark once V_exact is used in training.",
            "Variable binder/target lengths require Stage10 variable-length collate.",
        ],
        "source_paths": {"train": str(args.train), "val": str(args.val), "exact": str(args.exact)},
        "exact_train_fraction": args.exact_train_fraction,
        "exact_train_groups": exact_train_groups,
        "exact_val_groups": exact_val_groups,
        "input_summary": {"train": summarize(train), "val": summarize(val), "exact": summarize(exact)},
        "output_summary": {"merged_train": summarize(merged_train), "merged_val": summarize(merged_val), "merged_all": summarize(merged)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": merged, "manifest": manifest}, args.out)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "out": str(args.out), "summary": str(args.summary), "train": len(merged_train), "val": len(merged_val)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
