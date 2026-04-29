#!/usr/bin/env python
"""Rank Stage07 v2 filtered seeds by shared-anchor quality and production risk."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def length_penalty(target_len: int, binder_len: int) -> float:
    penalty = 0.0
    if target_len > 180:
        penalty += min(1.5, (target_len - 180) / 120.0)
    if binder_len > 70:
        penalty += min(1.0, (binder_len - 70) / 60.0)
    return penalty


def motion_score(motion: float) -> float:
    # Strategy01 wants real functional motion, not static noise and not extreme incompatible transitions.
    if motion < 1.0:
        return -2.0
    if motion <= 2.5:
        return 1.8
    if motion <= 5.0:
        return 2.2
    if motion <= 8.0:
        return 1.4
    if motion <= 10.0:
        return 0.4
    return -5.0


def score_sample(sample: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    filt = sample.get("stage07_v2_filter", {})
    cp = filt.get("contact_persistence", {})
    target_len = len(sample.get("target_sequence", ""))
    binder_len = len(sample.get("shared_binder_sequence", ""))
    motion = float(sample.get("target_motion_rmsd") or 0.0)
    ratio = float(cp.get("contact_persistence_ratio", 0.0) or 0.0)
    pair = float(cp.get("pair_persistent_contacts", 0.0) or 0.0)
    targ = float(cp.get("target_persistent_anchor_residues", 0.0) or 0.0)
    bind = float(cp.get("binder_persistent_anchor_residues", 0.0) or 0.0)
    min_contacts = float(cp.get("min_contact_count", 0.0) or 0.0)
    score = 0.0
    score += 4.0 * ratio
    score += 0.45 * math.log1p(pair)
    score += 0.25 * math.log1p(targ)
    score += 0.35 * math.log1p(bind)
    score += 0.15 * math.log1p(min_contacts)
    score += motion_score(motion)
    score -= length_penalty(target_len, binder_len)
    # Explicitly demote extra-large samples in the first production wave.
    if motion > 8.0:
        score -= 1.5
    meta = {
        "science_score": score,
        "target_len": target_len,
        "binder_len": binder_len,
        "motion": motion,
        "motion_score": motion_score(motion),
        "length_penalty": length_penalty(target_len, binder_len),
        "contact_persistence_ratio": ratio,
        "pair_persistent_contacts": pair,
        "target_persistent_anchor_residues": targ,
        "binder_persistent_anchor_residues": bind,
        "min_contact_count": min_contacts,
    }
    return score, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REPO / "data/strategy01/stage07_seed_pool/T_prod_seed_manifest_v2_filtered.json")
    parser.add_argument("--output", type=Path, default=REPO / "data/strategy01/stage07_seed_pool/T_prod_seed_manifest_v2_ranked.json")
    parser.add_argument("--top-output", type=Path, default=REPO / "data/strategy01/stage07_seed_pool/T_prod_seed_manifest_v2_ranked_top60.json")
    parser.add_argument("--summary", type=Path, default=REPO / "reports/strategy01/probes/stage07_seed_rank_v2_summary.json")
    parser.add_argument("--top-n", type=int, default=60)
    args = parser.parse_args()

    seeds = load_json(args.input)
    ranked = []
    for s in seeds:
        s = dict(s)
        score, meta = score_sample(s)
        s["stage07_v2_rank"] = meta
        ranked.append(s)
    ranked.sort(key=lambda x: x["stage07_v2_rank"]["science_score"], reverse=True)
    for i, s in enumerate(ranked, start=1):
        s["stage07_v2_rank"]["rank"] = i

    args.output.write_text(json.dumps(ranked, indent=2, ensure_ascii=False), encoding="utf-8")
    args.top_output.write_text(json.dumps(ranked[: args.top_n], indent=2, ensure_ascii=False), encoding="utf-8")
    bins: dict[str, int] = {}
    for s in ranked[: args.top_n]:
        b = s.get("stage07_v2_filter", {}).get("motion_bin", "unknown")
        bins[b] = bins.get(b, 0) + 1
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "top_output": str(args.top_output),
        "n_input": len(seeds),
        "top_n": args.top_n,
        "top_motion_bins": bins,
        "top_examples": [
            {
                "rank": s["stage07_v2_rank"]["rank"],
                "sample_id": s.get("sample_id"),
                "score": s["stage07_v2_rank"]["science_score"],
                "motion": s["stage07_v2_rank"]["motion"],
                "target_len": s["stage07_v2_rank"]["target_len"],
                "binder_len": s["stage07_v2_rank"]["binder_len"],
                "contact_persistence_ratio": s["stage07_v2_rank"]["contact_persistence_ratio"],
                "pair_persistent_contacts": s["stage07_v2_rank"]["pair_persistent_contacts"],
            }
            for s in ranked[:20]
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
