#!/usr/bin/env python
"""Stage08 cross-entry exact candidate miner.

The goal is not to replace curated resources such as ProtCID/PepBDB/PINDER.
This miner provides a reproducible RCSB smoke path for finding entries that
share the same target sequence and same binder sequence across multiple PDB
entries.  Candidates are marked as requiring biological-assembly audit before
they can become V_exact_main.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
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
    http_download,
    parse_models,
    residue_sequence,
    sequence_identity,
    write_state_pdb,
)

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query?json="
RCSB_PDB_URL = "https://files.rcsb.org/download/{entry}.pdb"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def http_json(payload: dict[str, Any]) -> dict[str, Any]:
    url = RCSB_SEARCH_URL + urllib.parse.quote(json.dumps(payload))
    req = urllib.request.Request(url, headers={"User-Agent": "codex-stage08/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def query_rcsb_complex_entries(max_entries: int) -> list[str]:
    method_nodes = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {"attribute": "exptl.method", "operator": "exact_match", "value": method},
        }
        for method in ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY", "SOLUTION NMR"]
    ]
    payload = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "group", "logical_operator": "or", "nodes": method_nodes},
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                        "operator": "greater_or_equal",
                        "value": 2,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_entries},
            "results_verbosity": "compact",
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }
    data = http_json(payload)
    entries: list[str] = []
    for item in data.get("result_set", []):
        if isinstance(item, str):
            entries.append(item.lower())
        elif isinstance(item, dict) and item.get("identifier"):
            entries.append(str(item["identifier"]).lower())
    return entries


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


def contact_pairs(model: ModelRecord, target_chain: str, binder_chain: str, cutoff_a: float = 8.0) -> set[tuple[int, int]]:
    target = coords_for_chain(model, target_chain)
    binder = coords_for_chain(model, binder_chain)
    cutoff2 = cutoff_a * cutoff_a
    pairs: set[tuple[int, int]] = set()
    for i, (xa, ya, za) in enumerate(target):
        for j, (xb, yb, zb) in enumerate(binder):
            if (xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2 <= cutoff2:
                pairs.add((i, j))
    return pairs


def sha12(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def pick_valid_pairs(model: ModelRecord, args: argparse.Namespace) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for target_chain, target_res in model.chains.items():
        target_len = sum(res.ca is not None for res in target_res)
        if target_len < args.target_min_len:
            continue
        target_seq = residue_sequence(target_res)
        if not target_seq or "X" in target_seq:
            continue
        for binder_chain, binder_res in model.chains.items():
            if binder_chain == target_chain:
                continue
            binder_len = sum(res.ca is not None for res in binder_res)
            if binder_len < args.binder_min_len or binder_len > args.binder_max_len:
                continue
            binder_seq = residue_sequence(binder_res)
            if not binder_seq or "X" in binder_seq:
                continue
            if target_seq == binder_seq:
                continue
            if len(target_seq) == len(binder_seq) and sequence_identity(target_seq, binder_seq) >= 0.95:
                continue
            contacts = calc_contacts(model, target_chain, binder_chain, args.min_contact_cutoff_a)
            if contacts < args.min_contacts:
                continue
            current = {
                "target_chain": target_chain,
                "binder_chain": binder_chain,
                "target_seq": target_seq,
                "binder_seq": binder_seq,
                "target_len": target_len,
                "binder_len": binder_len,
                "contacts": contacts,
                "score": contacts + 0.05 * target_len + 0.1 * binder_len,
            }
            pairs.append(current)
    pairs.sort(key=lambda item: item["score"], reverse=True)
    return pairs[: args.max_pairs_per_entry]


def scan_entries(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    entries = query_rcsb_complex_entries(args.max_entries)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    cache = args.out_dir / "cache" / "rcsb_pdb"
    for idx, entry in enumerate(entries, start=1):
        try:
            pdb_path = http_download(RCSB_PDB_URL.format(entry=entry.upper()), cache / f"{entry}.pdb")
            models = parse_models(pdb_path)
            if not models:
                continue
            pairs = pick_valid_pairs(models[0], args)
            if not pairs:
                continue
            for pair in pairs:
                records.append(
                    {
                        "entry": entry,
                        "pdb_path": str(pdb_path),
                        "model": models[0],
                        "pair": pair,
                        "group_key": sha12(pair["target_seq"]) + "__" + sha12(pair["binder_seq"]),
                    }
                )
        except Exception as exc:
            errors.append({"entry": entry, "error": str(exc)[:500]})
        if args.sleep_s > 0 and idx % 20 == 0:
            time.sleep(args.sleep_s)
    return records, [{"entry": e} for e in entries], errors


def build_group_sample(group_key: str, records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any] | None:
    if len(records) < args.min_states:
        return None
    records = sorted(records, key=lambda r: r["pair"]["contacts"], reverse=True)[: args.max_states]
    ref = records[0]
    target_chain = ref["pair"]["target_chain"]
    binder_chain = ref["pair"]["binder_chain"]
    target_coords_ref = coords_for_chain(ref["model"], target_chain)
    rmsds: list[float] = []
    contact_sets: list[set[tuple[int, int]]] = []
    for rec in records:
        rmsd = kabsch_rmsd(target_coords_ref, coords_for_chain(rec["model"], rec["pair"]["target_chain"]))
        if rmsd is not None:
            rmsds.append(rmsd)
        contact_sets.append(contact_pairs(rec["model"], rec["pair"]["target_chain"], rec["pair"]["binder_chain"], args.min_contact_cutoff_a))
    max_rmsd = max(rmsds) if rmsds else None
    if max_rmsd is None or max_rmsd < args.min_target_motion_a or max_rmsd > args.max_target_motion_a:
        return None
    persistent = set.intersection(*contact_sets) if contact_sets else set()
    if len(persistent) < args.min_persistent_anchors:
        return None

    sample_root = args.out_dir / "cross_entry_states" / group_key
    target_paths: list[str] = []
    complex_paths: list[str] = []
    state_entries: list[str] = []
    contact_counts: list[int] = []
    for state_i, rec in enumerate(records):
        entry = rec["entry"]
        pair = rec["pair"]
        target_path = sample_root / f"{group_key}_state{state_i:02d}_{entry}_target_{pair['target_chain']}.pdb"
        complex_path = sample_root / f"{group_key}_state{state_i:02d}_{entry}_complex_{pair['target_chain']}{pair['binder_chain']}.pdb"
        write_state_pdb(rec["model"], {pair["target_chain"]}, target_path)
        write_state_pdb(rec["model"], {pair["target_chain"], pair["binder_chain"]}, complex_path)
        target_paths.append(str(target_path))
        complex_paths.append(str(complex_path))
        state_entries.append(entry)
        contact_counts.append(int(pair["contacts"]))

    binder_seq = ref["pair"]["binder_seq"]
    binder_type = "peptide" if len(binder_seq) <= 40 else "protein"
    return {
        "sample_id": f"rcsb_cross_{group_key}",
        "target_id": f"rcsb_cross_{sha12(ref['pair']['target_seq'])}",
        "binder_id": f"rcsb_cross_{sha12(binder_seq)}",
        "binder_type": binder_type,
        "target_uniprot": None,
        "binder_uniprot_or_sequence": binder_seq,
        "target_sequence": ref["pair"]["target_seq"],
        "shared_binder_sequence": binder_seq,
        "target_chain_id": target_chain,
        "binder_chain_id": binder_chain,
        "target_state_paths": target_paths,
        "exact_complex_paths": complex_paths,
        "state_methods": ["rcsb_experimental_pdb_smoke"] * len(target_paths),
        "state_roles": ["required_bind"] * len(target_paths),
        "state_weights": [round(1.0 / len(target_paths), 6)] * len(target_paths),
        "exact_pair_flag": True,
        "source_db": "rcsb_cross_entry_smoke",
        "source_tier": "exact_experimental_candidate",
        "source_tier_stage08": "exact_experimental_candidate_requires_assembly_audit",
        "assembly_support": "author_asym_or_pdb_file_unverified",
        "needs_assembly_audit": True,
        "motion_metrics": {
            "target_backbone_rmsd_A": round(float(max_rmsd), 4),
            "interface_region_rmsd_A": round(float(max_rmsd), 4),
            "per_state_contact_counts": contact_counts,
            "persistent_anchor_count": len(persistent),
            "state_entries": state_entries,
        },
        "motion_bin": motion_bin_stage08(max_rmsd),
        "quality_tier": "candidate",
        "split_group": sha12(ref["pair"]["target_seq"]),
        "family_holdout_key": sha12(ref["pair"]["target_seq"]),
    }


def diagnose_group(group_key: str, records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {
        "group_key": group_key,
        "n_records": len(records),
        "entries": sorted({r["entry"] for r in records})[:20],
        "reason": None,
        "max_target_motion_A": None,
        "persistent_anchor_count": None,
        "min_contacts": None,
    }
    if len(records) < args.min_states:
        out["reason"] = "records_lt_min_states"
        return out
    records = sorted(records, key=lambda r: r["pair"]["contacts"], reverse=True)[: args.max_states]
    ref = records[0]
    target_coords_ref = coords_for_chain(ref["model"], ref["pair"]["target_chain"])
    rmsds: list[float] = []
    contact_sets: list[set[tuple[int, int]]] = []
    contacts: list[int] = []
    for rec in records:
        rmsd = kabsch_rmsd(target_coords_ref, coords_for_chain(rec["model"], rec["pair"]["target_chain"]))
        if rmsd is not None:
            rmsds.append(rmsd)
        pairs = contact_pairs(rec["model"], rec["pair"]["target_chain"], rec["pair"]["binder_chain"], args.min_contact_cutoff_a)
        contact_sets.append(pairs)
        contacts.append(len(pairs))
    max_rmsd = max(rmsds) if rmsds else None
    persistent = set.intersection(*contact_sets) if contact_sets else set()
    out["max_target_motion_A"] = round(float(max_rmsd), 4) if max_rmsd is not None else None
    out["persistent_anchor_count"] = len(persistent)
    out["min_contacts"] = min(contacts) if contacts else 0
    if max_rmsd is None:
        out["reason"] = "motion_unknown"
    elif max_rmsd < args.min_target_motion_a:
        out["reason"] = "motion_below_min"
    elif max_rmsd > args.max_target_motion_a:
        out["reason"] = "motion_above_max"
    elif len(persistent) < args.min_persistent_anchors:
        out["reason"] = "persistent_anchors_below_min"
    else:
        out["reason"] = "accepted_or_unexpected"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/strategy01/stage08_high_quality_dataset")
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "reports/strategy01/probes/stage08_cross_entry_mining_smoke_summary.json")
    parser.add_argument("--max-entries", type=int, default=120)
    parser.add_argument("--min-states", type=int, default=2)
    parser.add_argument("--max-states", type=int, default=3)
    parser.add_argument("--target-min-len", type=int, default=30)
    parser.add_argument("--binder-min-len", type=int, default=6)
    parser.add_argument("--binder-max-len", type=int, default=150)
    parser.add_argument("--min-contacts", type=int, default=8)
    parser.add_argument("--min-contact-cutoff-a", type=float, default=8.0)
    parser.add_argument("--min-target-motion-a", type=float, default=1.0)
    parser.add_argument("--max-target-motion-a", type=float, default=10.0)
    parser.add_argument("--min-persistent-anchors", type=int, default=3)
    parser.add_argument("--max-pairs-per-entry", type=int, default=4)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    records, queried, errors = scan_entries(args)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_group[rec["group_key"]].append(rec)
    samples: list[dict[str, Any]] = []
    group_diagnostics: list[dict[str, Any]] = []
    for group_key, group_records in by_group.items():
        if len(group_records) >= args.min_states:
            group_diagnostics.append(diagnose_group(group_key, group_records, args))
        sample = build_group_sample(group_key, group_records, args)
        if sample is not None:
            samples.append(sample)

    manifest = args.out_dir / "cross_entry_candidates_manifest.json"
    write_json(manifest, samples)
    summary = {
        "status": "passed" if samples else "no_cross_entry_candidates_in_smoke_window",
        "n_entries_queried": len(queried),
        "n_entries_with_usable_pair": len(records),
        "n_sequence_groups": len(by_group),
        "n_groups_with_ge2_entries": sum(1 for v in by_group.values() if len(v) >= 2),
        "n_cross_entry_candidates": len(samples),
        "rejected_group_diagnostics": [x for x in group_diagnostics if x.get("reason") != "accepted_or_unexpected"][:50],
        "manifest": str(manifest),
        "candidate_sample_ids": [s["sample_id"] for s in samples[:20]],
        "errors": errors[:50],
        "note": "Candidates require biological-assembly/ProtCID-style audit before being promoted to V_exact_main.",
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
