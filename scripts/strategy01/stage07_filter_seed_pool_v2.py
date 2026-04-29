#!/usr/bin/env python
"""Filter Stage07 seed pool by biologically meaningful multistate motion and shared anchors."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.strategy01.stage05_build_predictor_multistate_dataset import chain_to_atom37_any, interface_labels


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def repo_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (REPO / p)


def motion_bin(rmsd: float) -> str:
    if rmsd < 2.5:
        return "small"
    if rmsd < 5.0:
        return "medium"
    if rmsd <= 8.0:
        return "large"
    if rmsd <= 10.0:
        return "extra_large_review"
    return "too_large"


def leakage_keys(manifests: list[Path]) -> set[str]:
    keys: set[str] = set()
    for p in manifests:
        if not p.exists():
            continue
        data = load_json(p)
        if not isinstance(data, list):
            continue
        for row in data:
            for k in ["split_group", "family_split_key", "target_uniprot", "target_id"]:
                v = row.get(k)
                if v:
                    keys.add(str(v))
    return keys


def contact_persistence(sample: dict[str, Any]) -> dict[str, Any]:
    target_len = len(sample["target_sequence"])
    binder_len = len(sample["shared_binder_sequence"])
    target_chain = sample.get("target_chain_id", "A")
    binder_chain = sample.get("binder_chain_id", "B")
    contacts = []
    contact_counts = []
    for p in sample.get("exact_complex_paths", []):
        path = repo_path(p)
        target = chain_to_atom37_any(path, target_chain, target_len)
        binder = chain_to_atom37_any(path, binder_chain, binder_len)
        labels = interface_labels(target, binder)
        c = labels["contact_labels"].bool()
        contacts.append(c)
        contact_counts.append(int(c.sum().item()))
    if not contacts:
        return {"state_count": 0, "error": "no_contacts"}
    stack = torch.stack(contacts).int()
    k = stack.shape[0]
    min_states = max(2, math.ceil(k / 2)) if k >= 2 else 1
    pair_persistent = int((stack.sum(dim=0) >= min_states).sum().item())
    target_persistent = int((stack.any(dim=2).sum(dim=0) >= min_states).sum().item())
    binder_persistent = int((stack.any(dim=1).sum(dim=0) >= min_states).sum().item())
    union_pairs = int((stack.sum(dim=0) > 0).sum().item())
    if union_pairs > 0:
        contact_persistence_ratio = float(pair_persistent / union_pairs)
    else:
        contact_persistence_ratio = 0.0
    return {
        "state_count": int(k),
        "contact_counts": contact_counts,
        "min_contact_count": min(contact_counts) if contact_counts else 0,
        "pair_persistent_contacts": pair_persistent,
        "target_persistent_anchor_residues": target_persistent,
        "binder_persistent_anchor_residues": binder_persistent,
        "contact_persistence_ratio": contact_persistence_ratio,
    }


def keep_reason(sample: dict[str, Any], metrics: dict[str, Any], args: argparse.Namespace, excluded: set[str]) -> tuple[bool, str]:
    motion = float(sample.get("target_motion_rmsd") or 0.0)
    target_len = len(sample.get("target_sequence", ""))
    binder_len = len(sample.get("shared_binder_sequence", ""))
    split_group = str(sample.get("split_group") or sample.get("family_split_key") or "")
    if split_group and split_group in excluded:
        return False, "family_leakage"
    if motion < args.min_motion:
        return False, "motion_too_small"
    if motion > args.extended_max_motion:
        return False, "motion_gt_10A"
    if target_len < args.min_target_len or target_len > args.max_target_len:
        return False, "target_length_out_of_range"
    if binder_len < args.min_binder_len or binder_len > args.max_binder_len:
        return False, "binder_length_out_of_range"
    if metrics.get("state_count", 0) < args.min_states:
        return False, "too_few_states"
    if min(metrics.get("contact_counts", [0])) < args.min_contacts_each_state:
        return False, "weak_exact_interface"
    if metrics.get("target_persistent_anchor_residues", 0) < args.min_target_persistent_anchors:
        return False, "weak_target_persistent_anchors"
    if metrics.get("binder_persistent_anchor_residues", 0) < args.min_binder_persistent_anchors:
        return False, "weak_binder_persistent_anchors"
    if metrics.get("pair_persistent_contacts", 0) < args.min_pair_persistent_contacts:
        return False, "weak_pair_persistent_contacts"
    if motion > args.default_max_motion and metrics.get("contact_persistence_ratio", 0.0) < args.extra_large_min_contact_persistence:
        return False, "extra_large_weak_persistence"
    return True, "keep_strict" if motion <= args.default_max_motion else "keep_extended_8_10A"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REPO / "data/strategy01/stage07_seed_pool/T_prod_seed_manifest.json")
    parser.add_argument("--output", type=Path, default=REPO / "data/strategy01/stage07_seed_pool/T_prod_seed_manifest_v2_filtered.json")
    parser.add_argument("--strict-output", type=Path, default=REPO / "data/strategy01/stage07_seed_pool/T_prod_seed_manifest_v2_strict_le8.json")
    parser.add_argument("--extended-output", type=Path, default=REPO / "data/strategy01/stage07_seed_pool/T_prod_seed_manifest_v2_extended_8to10.json")
    parser.add_argument("--summary", type=Path, default=REPO / "reports/strategy01/probes/stage07_seed_filter_v2_summary.json")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[REPO / "data/strategy01/stage07_validation_expand/V_exact_manifest.json", REPO / "data/strategy01/stage06_validation/V_exact_manifest.json"])
    parser.add_argument("--min-motion", type=float, default=1.0)
    parser.add_argument("--default-max-motion", type=float, default=8.0)
    parser.add_argument("--extended-max-motion", type=float, default=10.0)
    parser.add_argument("--min-states", type=int, default=2)
    parser.add_argument("--min-target-len", type=int, default=30)
    parser.add_argument("--max-target-len", type=int, default=260)
    parser.add_argument("--min-binder-len", type=int, default=6)
    parser.add_argument("--max-binder-len", type=int, default=100)
    parser.add_argument("--min-contacts-each-state", type=int, default=8)
    parser.add_argument("--min-target-persistent-anchors", type=int, default=3)
    parser.add_argument("--min-binder-persistent-anchors", type=int, default=2)
    parser.add_argument("--min-pair-persistent-contacts", type=int, default=1)
    parser.add_argument("--extra-large-min-contact-persistence", type=float, default=0.05)
    args = parser.parse_args()

    seeds = load_json(args.input)
    excluded = leakage_keys(args.exclude_manifest)
    kept = []
    strict = []
    extended = []
    rejected: dict[str, int] = {}
    bins: dict[str, int] = {}
    examples = []

    for sample in seeds:
        sample = dict(sample)
        metrics = contact_persistence(sample)
        keep, reason = keep_reason(sample, metrics, args, excluded)
        rejected[reason] = rejected.get(reason, 0) + (0 if keep else 1)
        sample["stage07_v2_filter"] = {
            "decision": reason,
            "motion_bin": motion_bin(float(sample.get("target_motion_rmsd") or 0.0)),
            "contact_persistence": metrics,
            "max_motion_default_A": args.default_max_motion,
            "max_motion_extended_A": args.extended_max_motion,
            "scientific_target": "shared binder sequence compatible with functional target conformational ensemble",
        }
        if keep:
            kept.append(sample)
            b = sample["stage07_v2_filter"]["motion_bin"]
            bins[b] = bins.get(b, 0) + 1
            if float(sample.get("target_motion_rmsd") or 0.0) <= args.default_max_motion:
                strict.append(sample)
            else:
                extended.append(sample)
            if len(examples) < 20:
                examples.append({"sample_id": sample.get("sample_id"), "motion": sample.get("target_motion_rmsd"), "decision": reason, "metrics": metrics})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
    args.strict_output.write_text(json.dumps(strict, indent=2, ensure_ascii=False), encoding="utf-8")
    args.extended_output.write_text(json.dumps(extended, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "strict_output": str(args.strict_output),
        "extended_output": str(args.extended_output),
        "n_input": len(seeds),
        "n_kept_total": len(kept),
        "n_kept_strict_le8": len(strict),
        "n_kept_extended_8to10": len(extended),
        "motion_bins_kept": bins,
        "rejected_counts": rejected,
        "excluded_leakage_keys": len(excluded),
        "thresholds": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()} | {"exclude_manifest": [str(p) for p in args.exclude_manifest]},
        "examples": examples,
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

