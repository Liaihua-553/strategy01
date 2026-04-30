#!/usr/bin/env python
"""Stage08 audit for high-quality multistate target-binder complex data.

This script does not create new labels.  It audits existing Stage06/Stage07
manifests and separates samples into exact, challenge, hybrid-candidate, and
auxiliary/invalid buckets according to the Stage08 scientific contract:
one shared binder sequence, multiple target states, and per-state experimental
complex geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.strategy01.stage06_mine_multistate_validation import (  # noqa: E402
    ModelRecord,
    calc_contacts,
    coords_for_chain,
    parse_models,
    residue_sequence,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def as_abs(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    p = Path(path_like)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def seq_identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(1 for x, y in zip(a[:n], b[:n], strict=False) if x == y) / n


def kabsch_rmsd(coords_a: list[tuple[float, float, float]], coords_b: list[tuple[float, float, float]]) -> float | None:
    n = min(len(coords_a), len(coords_b))
    if n < 3:
        return None
    a = np.asarray(coords_a[:n], dtype=np.float64)
    b = np.asarray(coords_b[:n], dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    aligned = a @ r
    return float(np.sqrt(np.mean(np.sum((aligned - b) ** 2, axis=1))))


def first_model(path: Path) -> ModelRecord | None:
    models = parse_models(path)
    if not models:
        return None
    return models[0]


def contact_pairs(model: ModelRecord, target_chain: str, binder_chain: str, cutoff_a: float = 8.0) -> set[tuple[int, int]]:
    target = coords_for_chain(model, target_chain)
    binder = coords_for_chain(model, binder_chain)
    if not target or not binder:
        return set()
    cutoff2 = cutoff_a * cutoff_a
    pairs: set[tuple[int, int]] = set()
    for i, (xa, ya, za) in enumerate(target):
        for j, (xb, yb, zb) in enumerate(binder):
            if (xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2 <= cutoff2:
                pairs.add((i, j))
    return pairs


def pairwise_max_rmsd(models: list[ModelRecord], chain_id: str) -> float | None:
    coords = [coords_for_chain(model, chain_id) for model in models]
    coords = [c for c in coords if c]
    if len(coords) < 2:
        return None
    values: list[float] = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            rmsd = kabsch_rmsd(coords[i], coords[j])
            if rmsd is not None:
                values.append(rmsd)
    if not values:
        return None
    return max(values)


def motion_bin_stage08(rmsd: float | None) -> str:
    if rmsd is None:
        return "unknown"
    if rmsd < 1.0:
        return "low_lt1"
    if rmsd <= 2.5:
        return "small_1_2p5"
    if rmsd <= 5.0:
        return "medium_2p5_5"
    if rmsd <= 8.0:
        return "large_5_8"
    if rmsd <= 10.0:
        return "challenge_8_10"
    return "too_large_gt10"


def infer_exact_kind(sample: dict[str, Any]) -> str:
    source_db = str(sample.get("source_db", "")).lower()
    paths = sample.get("exact_complex_paths") or []
    parents = {Path(str(p)).parent.name for p in paths}
    if "nmr" in source_db:
        return "exact_experimental_multimodel_nmr"
    if len(parents) == 1 and paths:
        return "exact_experimental_same_entry_or_same_dir"
    return "exact_experimental_cross_entry_or_mixed"


def load_existing_samples(paths: list[Path]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        data = read_json(path)
        if not isinstance(data, list):
            continue
        for idx, sample in enumerate(data):
            sid = str(sample.get("sample_id") or f"{path.name}:{idx}")
            key = sid + "::" + str(sample.get("source_db", ""))
            if key in seen:
                continue
            seen.add(key)
            sample = dict(sample)
            sample["_stage08_source_manifest"] = str(path)
            merged.append(sample)
    return merged


def audit_sample(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    audited = dict(sample)
    reasons: list[str] = []
    warnings: list[str] = []

    target_chain = str(sample.get("target_chain_id") or sample.get("target_chain") or "A").upper()
    binder_chain = str(sample.get("binder_chain_id") or sample.get("binder_chain") or "B").upper()
    target_paths = [as_abs(p) for p in sample.get("target_state_paths", [])]
    complex_paths = [as_abs(p) for p in sample.get("exact_complex_paths", [])]
    target_paths = [p for p in target_paths if p is not None]
    complex_paths = [p for p in complex_paths if p is not None]

    if len(target_paths) < args.min_states:
        reasons.append(f"target_states_lt_{args.min_states}")
    if len(complex_paths) < args.min_states:
        reasons.append(f"experimental_complex_states_lt_{args.min_states}")
    missing_paths = [str(p) for p in target_paths + complex_paths if not p.exists()]
    if missing_paths:
        reasons.append("missing_structure_paths")

    target_models: list[ModelRecord] = []
    complex_models: list[ModelRecord] = []
    if not missing_paths:
        for p in target_paths:
            model = first_model(p)
            if model is not None:
                target_models.append(model)
        for p in complex_paths:
            model = first_model(p)
            if model is not None:
                complex_models.append(model)

    if len(target_models) != len(target_paths):
        reasons.append("target_parse_failed")
    if len(complex_models) != len(complex_paths):
        reasons.append("complex_parse_failed")

    declared_target_seq = str(sample.get("target_sequence") or "")
    declared_binder_seq = str(sample.get("shared_binder_sequence") or sample.get("binder_uniprot_or_sequence") or "")

    target_seqs = [residue_sequence(model.chains.get(target_chain, [])) for model in target_models]
    binder_seqs = [residue_sequence(model.chains.get(binder_chain, [])) for model in complex_models]
    if target_seqs and declared_target_seq:
        min_target_id = min(seq_identity(declared_target_seq, seq) for seq in target_seqs if seq)
    else:
        min_target_id = 0.0
    if binder_seqs and declared_binder_seq:
        min_binder_id = min(seq_identity(declared_binder_seq, seq) for seq in binder_seqs if seq)
    else:
        min_binder_id = 0.0

    if target_seqs and min_target_id < args.min_sequence_identity:
        reasons.append("target_sequence_inconsistent")
    if binder_seqs and min_binder_id < args.min_sequence_identity:
        reasons.append("binder_sequence_inconsistent")

    contact_sets = [contact_pairs(model, target_chain, binder_chain, args.contact_cutoff_a) for model in complex_models]
    contact_counts = [len(x) for x in contact_sets]
    if contact_counts and min(contact_counts) < args.min_contacts:
        reasons.append("low_experimental_contacts")
    if contact_sets:
        persistent = set.intersection(*contact_sets) if len(contact_sets) > 1 else set(contact_sets[0])
        union = set.union(*contact_sets)
    else:
        persistent = set()
        union = set()
    persistent_anchor_count = len(persistent)
    contact_persistence = persistent_anchor_count / max(1, len(union))

    if contact_sets and persistent_anchor_count < args.min_persistent_anchors:
        warnings.append("low_persistent_anchor_count")

    max_target_rmsd = pairwise_max_rmsd(target_models, target_chain)
    if max_target_rmsd is None:
        mm = sample.get("motion_metrics") or {}
        raw_rmsd = mm.get("target_backbone_rmsd_A") or mm.get("interface_region_rmsd_A")
        try:
            max_target_rmsd = float(raw_rmsd)
        except Exception:
            max_target_rmsd = None
    motion_bin = motion_bin_stage08(max_target_rmsd)

    if max_target_rmsd is None:
        reasons.append("motion_unknown")
    elif max_target_rmsd < 1.0:
        warnings.append("motion_lt_1A_auxiliary")
    elif max_target_rmsd > 10.0:
        reasons.append("motion_gt_10A_excluded_from_main")
    elif max_target_rmsd > 8.0 and persistent_anchor_count < args.min_persistent_anchors:
        reasons.append("motion_8_10A_without_anchor_support")

    exact_kind = infer_exact_kind(sample)
    has_exact = len(complex_paths) >= args.min_states and not missing_paths

    if not has_exact:
        stage08_class = "hybrid_candidate"
    elif reasons:
        stage08_class = "invalid_or_auxiliary"
    elif max_target_rmsd is not None and 1.0 <= max_target_rmsd <= 8.0:
        stage08_class = "V_exact_main"
    elif max_target_rmsd is not None and 8.0 < max_target_rmsd <= 10.0:
        stage08_class = "V_exact_challenge"
    else:
        stage08_class = "invalid_or_auxiliary"

    audited["stage08_audit"] = {
        "class": stage08_class,
        "reasons": reasons,
        "warnings": warnings,
        "exact_kind": exact_kind,
        "k_target_states": len(target_paths),
        "k_complex_states": len(complex_paths),
        "min_target_sequence_identity": round(min_target_id, 4),
        "min_binder_sequence_identity": round(min_binder_id, 4),
        "per_state_contact_counts": contact_counts,
        "min_contact_count": min(contact_counts) if contact_counts else 0,
        "persistent_anchor_count": persistent_anchor_count,
        "contact_persistence": round(contact_persistence, 4),
        "target_motion_rmsd_aligned_A": round(max_target_rmsd, 4) if max_target_rmsd is not None else None,
        "motion_bin_stage08": motion_bin,
        "path_ok": not missing_paths,
        "missing_paths": missing_paths[:20],
    }
    audited["source_tier_stage08"] = {
        "V_exact_main": "exact_experimental",
        "V_exact_challenge": "exact_experimental_challenge",
        "hybrid_candidate": "hybrid_candidate",
    }.get(stage08_class, "auxiliary_or_invalid")
    return audited


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_class = Counter(s["stage08_audit"]["class"] for s in samples)
    by_kind = Counter(s["stage08_audit"]["exact_kind"] for s in samples)
    by_motion = Counter(s["stage08_audit"]["motion_bin_stage08"] for s in samples)
    by_source = Counter(str(s.get("source_db", "unknown")) for s in samples)
    families_by_class: dict[str, set[str]] = defaultdict(set)
    for s in samples:
        families_by_class[s["stage08_audit"]["class"]].add(str(s.get("family_holdout_key") or s.get("split_group") or s.get("target_id") or "unknown"))
    main = [s for s in samples if s["stage08_audit"]["class"] == "V_exact_main"]
    status = "passed" if len(main) >= 32 and len(families_by_class["V_exact_main"]) >= 12 else "short_exact_main"
    return {
        "status": status,
        "n_total": len(samples),
        "counts_by_class": dict(by_class),
        "counts_by_exact_kind": dict(by_kind),
        "counts_by_motion_bin": dict(by_motion),
        "counts_by_source_db": dict(by_source),
        "families_by_class": {k: len(v) for k, v in families_by_class.items()},
        "main_exact_sample_ids": [s.get("sample_id") for s in main[:20]],
        "criteria": {
            "V_exact_main_min_target": 32,
            "V_exact_main_min_families": 12,
            "motion_main_A": "1.0-8.0",
            "motion_challenge_A": "8.0-10.0 with persistent anchors",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/strategy01/stage08_high_quality_dataset")
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "reports/strategy01/probes/stage08_exact_audit_summary.json")
    parser.add_argument("--min-states", type=int, default=2)
    parser.add_argument("--min-contacts", type=int, default=8)
    parser.add_argument("--min-persistent-anchors", type=int, default=3)
    parser.add_argument("--min-sequence-identity", type=float, default=0.95)
    parser.add_argument("--contact-cutoff-a", type=float, default=8.0)
    args = parser.parse_args()

    input_paths = [as_abs(path) if not path.is_absolute() else path for path in args.input_manifest]
    existing = load_existing_samples([p for p in input_paths if p is not None])
    audited = [audit_sample(sample, args) for sample in existing]

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in audited:
        buckets[sample["stage08_audit"]["class"]].append(sample)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "V_exact_main_manifest.json", buckets["V_exact_main"])
    write_json(args.out_dir / "V_exact_challenge_manifest.json", buckets["V_exact_challenge"])
    write_json(args.out_dir / "V_hybrid_candidates_manifest.json", buckets["hybrid_candidate"])
    write_json(args.out_dir / "T_aux_or_invalid_manifest.json", buckets["invalid_or_auxiliary"])
    write_json(args.out_dir / "stage08_all_audited_manifest.json", audited)
    summary = summarize(audited)
    summary["input_manifests"] = [str(p) for p in input_paths if p is not None]
    summary["output_dir"] = str(args.out_dir)
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
