#!/usr/bin/env python
"""CPU-only Stage07 seed mining for Boltz production.

This script keeps large RCSB downloads in a scratch cache and writes only the
selected multistate target/binder seed PDBs plus a compact manifest to /data.
It avoids occupying a GPU while the pipeline is still doing network/CPU mining.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.strategy01.stage06_mine_multistate_validation import (  # noqa: E402
    RCSB_PDB_URL,
    calc_rmsd,
    exact_sample_from_candidate,
    http_download,
    parse_models,
    pick_chain_pair,
    query_rcsb_nmr_entries,
    select_state_indices,
)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list manifest: {path}")
    return data


def target_motion_rmsd(models: list[Any], target_chain: str, state_indices: list[int]) -> float:
    coords = []
    for idx in state_indices:
        chain = models[idx].chains.get(target_chain, [])
        coords.append([r.ca for r in chain if r.ca is not None])
    best = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            value = calc_rmsd(coords[i], coords[j])
            if value is not None:
                best = max(best, float(value))
    return best


def mine(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    selected_state_dir = args.out_dir / "T_prod_seed_states"
    selected_state_dir.mkdir(parents=True, exist_ok=True)

    excluded_groups: set[str] = set()
    for p in args.exclude_manifest:
        for sample in load_manifest(p):
            key = sample.get("split_group") or sample.get("family_split_key")
            if key:
                excluded_groups.add(str(key))

    entries = query_rcsb_nmr_entries(args.max_entries)
    if args.shuffle:
        random.shuffle(entries)
    seeds: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    failures: dict[str, int] = {}
    started = time.time()

    def bump(name: str) -> None:
        failures[name] = failures.get(name, 0) + 1

    print(f"[stage07_seed] entries={len(entries)} target_seeds={args.target_seeds} excluded_groups={len(excluded_groups)}", flush=True)
    for entry_idx, entry in enumerate(entries, start=1):
        if len(seeds) >= args.target_seeds:
            break
        if entry_idx == 1 or entry_idx % args.progress_every == 0:
            elapsed = time.time() - started
            print(f"[stage07_seed] scanned={entry_idx}/{len(entries)} accepted={len(seeds)} elapsed_sec={elapsed:.1f}", flush=True)
        try:
            pdb_path = http_download(RCSB_PDB_URL.format(entry=entry), args.cache_dir / "rcsb_nmr_pdbs" / f"{entry}.pdb")
        except Exception as exc:
            bump(f"download:{type(exc).__name__}")
            continue
        try:
            models = parse_models(pdb_path)
            pair = pick_chain_pair(
                models=models,
                target_min_len=args.target_min_len,
                binder_min_len=args.binder_min_len,
                binder_max_len=args.binder_max_len,
                min_contacts=args.min_contacts,
                min_target_rmsd=args.min_target_rmsd,
            )
            if pair is None:
                bump("no_pair")
                continue
            split_group = pair.get("split_group") or __import__("hashlib").sha1(pair["target_seq"].encode("utf-8")).hexdigest()[:12]
            if split_group in excluded_groups:
                bump("excluded_group")
                continue
            pair_key = (pair["target_seq"], pair["binder_seq"])
            if pair_key in seen_pairs:
                bump("duplicate_pair")
                continue
            state_indices = select_state_indices(models, pair["target_chain"], args.min_states, args.max_states)
            if len(state_indices) < args.min_states:
                bump("too_few_states")
                continue
            motion = target_motion_rmsd(models, pair["target_chain"], state_indices)
            if motion < args.min_target_rmsd:
                bump("low_motion")
                continue
            sample = exact_sample_from_candidate(entry, models, pair, state_indices, selected_state_dir)
            if sample.get("split_group") in excluded_groups:
                bump("excluded_after_sample")
                continue
            sample["exact_pair_flag"] = False
            sample["source_tier"] = "train_boltz_seed"
            sample["stage07_seed_source"] = "rcsb_nmr_cpu_mining"
            sample["target_motion_rmsd"] = motion
            seeds.append(sample)
            seen_pairs.add(pair_key)
        except Exception as exc:
            bump(f"parse:{type(exc).__name__}")
            continue

    with args.output_manifest.open("w", encoding="utf-8") as f:
        json.dump(seeds, f, indent=2, ensure_ascii=False)
    summary = {
        "entries_requested": args.max_entries,
        "entries_seen": len(entries),
        "accepted_seeds": len(seeds),
        "target_seeds": args.target_seeds,
        "output_manifest": str(args.output_manifest),
        "out_dir": str(args.out_dir),
        "cache_dir": str(args.cache_dir),
        "excluded_groups": len(excluded_groups),
        "failures": failures,
        "elapsed_sec": time.time() - started,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "strategy01" / "stage07_seed_pool")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/kfliao_strategy01_stage07_seed_cache"))
    parser.add_argument("--output-manifest", type=Path, default=REPO_ROOT / "data" / "strategy01" / "stage07_seed_pool" / "T_prod_seed_manifest.json")
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "reports" / "strategy01" / "probes" / "stage07_seed_mining_summary.json")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[REPO_ROOT / "data" / "strategy01" / "stage06_validation" / "V_exact_manifest.json"])
    parser.add_argument("--max-entries", type=int, default=2000)
    parser.add_argument("--target-seeds", type=int, default=256)
    parser.add_argument("--min-states", type=int, default=2)
    parser.add_argument("--max-states", type=int, default=3)
    parser.add_argument("--target-min-len", type=int, default=30)
    parser.add_argument("--binder-min-len", type=int, default=6)
    parser.add_argument("--binder-max-len", type=int, default=120)
    parser.add_argument("--min-contacts", type=int, default=8)
    parser.add_argument("--min-target-rmsd", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()
    mine(args)


if __name__ == "__main__":
    main()