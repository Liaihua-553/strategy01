#!/usr/bin/env python
"""Build Stage07 v2.2 seed order after Boltz runtime/acceptance feedback."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_known_outcomes(paths: list[Path]) -> tuple[set[str], set[str]]:
    accepted: set[str] = set()
    rejected: set[str] = set()
    for p in paths:
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        accepted.update(re.findall(r"accept sample_id=([^\s]+)", text))
        rejected.update(re.findall(r"reject sample_id=([^\s]+)", text))
    rejected -= accepted
    return accepted, rejected


def score_sample(sample: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    cp = sample.get("stage07_v2_filter", {}).get("contact_persistence", {})
    motion = float(sample.get("target_motion_rmsd") or 0.0)
    target_len = len(sample.get("target_sequence", ""))
    binder_len = len(sample.get("shared_binder_sequence", ""))
    ratio = float(cp.get("contact_persistence_ratio", 0.0) or 0.0)
    pair = float(cp.get("pair_persistent_contacts", 0.0) or 0.0)
    min_contacts = float(cp.get("min_contact_count", 0.0) or 0.0)
    targ = float(cp.get("target_persistent_anchor_residues", 0.0) or 0.0)
    bind = float(cp.get("binder_persistent_anchor_residues", 0.0) or 0.0)
    # v2.2 favors binder-like moderate interfaces: enough anchors, not whole-chain broad docking.
    score = 0.0
    score += 2.5 * min(ratio, 0.9)
    score += 0.55 * math.log1p(min(pair, 75.0))
    score += 0.20 * math.log1p(min_contacts)
    score += 0.25 * math.log1p(targ)
    score += 0.35 * math.log1p(bind)
    if 2.0 <= motion <= 5.5:
        score += 2.0
    elif 1.0 <= motion < 2.0:
        score += 1.2
    elif 5.5 < motion <= 8.0:
        score += 1.0
    else:
        score -= 2.0
    if pair > 90:
        score -= 2.5 + (pair - 90) / 50.0
    if target_len > 180:
        score -= (target_len - 180) / 80.0
    if binder_len > 70:
        score -= (binder_len - 70) / 40.0
    if binder_len < 8:
        score -= 0.8
    meta = {
        "science_score_v22": score,
        "motion": motion,
        "target_len": target_len,
        "binder_len": binder_len,
        "contact_persistence_ratio": ratio,
        "pair_persistent_contacts": pair,
        "min_contact_count": min_contacts,
        "target_persistent_anchor_residues": targ,
        "binder_persistent_anchor_residues": bind,
        "rationale": "moderate persistent interface, not oversized broad interface; suited to shared binder sequence across functional states",
    }
    return score, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REPO / "data/strategy01/stage07_seed_pool/T_prod_seed_manifest_v2_filtered.json")
    parser.add_argument("--output", type=Path, default=REPO / "data/strategy01/stage07_seed_pool/T_prod_seed_manifest_v22_ranked.json")
    parser.add_argument("--top-output", type=Path, default=REPO / "data/strategy01/stage07_seed_pool/T_prod_seed_manifest_v22_ranked_top80.json")
    parser.add_argument("--summary", type=Path, default=REPO / "reports/strategy01/probes/stage07_seed_rank_v22_summary.json")
    parser.add_argument("--top-n", type=int, default=80)
    args = parser.parse_args()
    accepted, rejected = collect_known_outcomes([
        REPO / "reports/strategy01/probes/stage07_data_prod_v2_calib_1908308.out",
        REPO / "reports/strategy01/probes/stage07_data_prod_v2_ranked_calib_1908314.out",
    ])
    seeds = load_json(args.input)
    rows = []
    excluded = []
    for s in seeds:
        s = dict(s)
        sid = str(s.get("sample_id"))
        cp = s.get("stage07_v2_filter", {}).get("contact_persistence", {})
        pair = float(cp.get("pair_persistent_contacts", 0.0) or 0.0)
        target_len = len(s.get("target_sequence", ""))
        binder_len = len(s.get("shared_binder_sequence", ""))
        if sid in rejected:
            excluded.append({"sample_id": sid, "reason": "known_failed_in_calibration"})
            continue
        if pair > 120:
            excluded.append({"sample_id": sid, "reason": "oversized_persistent_interface", "pair_persistent_contacts": pair})
            continue
        if target_len > 260 or binder_len > 100:
            excluded.append({"sample_id": sid, "reason": "length_risk", "target_len": target_len, "binder_len": binder_len})
            continue
        score, meta = score_sample(s)
        if sid in accepted:
            score += 3.0
            meta["known_calibration_accept_bonus"] = True
            meta["science_score_v22"] = score
        s["stage07_v22_rank"] = meta
        rows.append(s)
    rows.sort(key=lambda x: x["stage07_v22_rank"]["science_score_v22"], reverse=True)
    for i, s in enumerate(rows, start=1):
        s["stage07_v22_rank"]["rank"] = i
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    args.top_output.write_text(json.dumps(rows[:args.top_n], indent=2, ensure_ascii=False), encoding="utf-8")
    bins: dict[str, int] = {}
    for s in rows[:args.top_n]:
        b = s.get("stage07_v2_filter", {}).get("motion_bin", "unknown")
        bins[b] = bins.get(b, 0) + 1
    summary = {
        "n_input": len(seeds),
        "n_ranked": len(rows),
        "n_excluded": len(excluded),
        "known_accepted": sorted(accepted),
        "known_rejected": sorted(rejected),
        "top_output": str(args.top_output),
        "top_motion_bins": bins,
        "excluded_examples": excluded[:30],
        "top_examples": [
            {
                "rank": s["stage07_v22_rank"]["rank"],
                "sample_id": s.get("sample_id"),
                "score": s["stage07_v22_rank"]["science_score_v22"],
                "motion": s["stage07_v22_rank"]["motion"],
                "target_len": s["stage07_v22_rank"]["target_len"],
                "binder_len": s["stage07_v22_rank"]["binder_len"],
                "pair_persistent_contacts": s["stage07_v22_rank"]["pair_persistent_contacts"],
                "contact_persistence_ratio": s["stage07_v22_rank"]["contact_persistence_ratio"],
                "known_bonus": s["stage07_v22_rank"].get("known_calibration_accept_bonus", False),
            }
            for s in rows[:25]
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
