#!/usr/bin/env python
"""Salvage partial Stage07 Boltz2 production results from a running scratch dir.

This script is intended to run on the compute node that owns the scratch path
(for job 1908299 this is gu02).  It parses completed Boltz outputs, keeps only
state predictions that pass Bronze, copies compact evidence for accepted states
into the persistent Strategy01 data directory, and writes a summary explaining
why the old low-yield run should be stopped.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.strategy01.stage04_predictors.boltz_adapter import BoltzAdapter, BoltzRunConfig
from scripts.strategy01.stage05_build_predictor_multistate_dataset import chain_to_atom37_any, interface_labels
from scripts.strategy01.stage06_build_boltz_production_dataset import (
    extract_anchor_constraints,
    passes_bronze_stage06,
    passes_silver,
    variant_state_subsets,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def repo_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (REPO / p)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    try:
        return str(dst.relative_to(REPO))
    except ValueError:
        return str(dst)


def compact_metrics(metrics: dict[str, Any], geom: dict[str, Any], *, pass_name: str, state_idx: int, stem: str) -> dict[str, Any]:
    keys = [
        "confidence_score",
        "ptm",
        "iptm",
        "protein_iptm",
        "complex_plddt_norm",
        "complex_iplddt_norm",
        "binder_plddt_norm",
        "interchain_pae_A",
        "interchain_pde_A",
        "pDockQ2_proxy",
    ]
    out = {k: safe_float(metrics.get(k)) for k in keys}
    out.update(
        {
            "state_index": int(state_idx),
            "pass_name": pass_name,
            "input_stem": stem,
            "contact_count": int(geom.get("contact_count", 0)),
            "severe_clash": bool(geom.get("severe_clash", True)),
        }
    )
    return out


def aggregate_worst(states: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [float(s[key]) for s in states if s.get(key) is not None and math.isfinite(float(s[key]))]

    out: dict[str, Any] = {"valid_states": len(states)}
    for key in ["interchain_pae_A", "interchain_pde_A"]:
        v = vals(key)
        if v:
            out[f"worst_{key}"] = max(v)
            out[f"mean_{key}"] = sum(v) / len(v)
    for key in ["protein_iptm", "complex_plddt_norm", "pDockQ2_proxy"]:
        v = vals(key)
        if v:
            out[f"worst_{key}"] = min(v)
            out[f"mean_{key}"] = sum(v) / len(v)
    contacts = [int(s.get("contact_count", 0)) for s in states]
    if contacts:
        out["worst_contact_count"] = min(contacts)
        out["mean_contact_count"] = sum(contacts) / len(contacts)
    out["any_severe_clash"] = any(bool(s.get("severe_clash", True)) for s in states)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-manifest", type=Path, default=REPO / "data/strategy01/stage07_seed_pool/T_prod_seed_manifest.json")
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/kfliao_strategy01_stage07_1908299"))
    parser.add_argument("--out-dir", type=Path, default=REPO / "data/strategy01/stage07_partial_1908299")
    parser.add_argument("--summary", type=Path, default=REPO / "reports/strategy01/probes/stage07_salvage_1908299_summary.json")
    parser.add_argument("--min-states", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=25)
    args = parser.parse_args()

    started = time.time()
    out_dir = args.out_dir
    evidence_dir = out_dir / "accepted_boltz_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    seeds = load_json(args.seed_manifest)
    seed_by_id = {str(s.get("sample_id")): s for s in seeds}
    out_root = args.scratch / "boltz_outputs"
    adapter = BoltzAdapter(BoltzRunConfig())

    state_attempt_records: list[dict[str, Any]] = []
    accepted_bases: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "scratch": str(args.scratch),
        "seed_manifest": str(args.seed_manifest),
        "out_dir": str(out_dir),
        "sample_dirs_present": 0,
        "base_seeds_with_any_finished_pass": 0,
        "accepted_bases_bronze_ge2": 0,
        "accepted_bases_bronze_ge3": 0,
        "accepted_variants_est": 0,
        "silver_bases_all_states": 0,
        "partial_or_failed_bases": 0,
        "state_level_bronze": 0,
        "state_level_silver": 0,
        "state_level_total_attempted": 0,
        "state_level_total_selected": 0,
        "missing_exact_paths": 0,
        "parse_failures": 0,
        "copy_failures": 0,
    }

    if not out_root.exists():
        raise FileNotFoundError(f"Boltz scratch output root not found: {out_root}")

    sample_dirs = sorted(p for p in out_root.iterdir() if p.is_dir())
    stats["sample_dirs_present"] = len(sample_dirs)

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        seed = seed_by_id.get(sample_id)
        if seed is None:
            continue
        seed = dict(seed)
        seed["exact_complex_paths"] = [str(repo_path(x)) for x in seed.get("exact_complex_paths", [])]
        seed["target_state_paths"] = [str(repo_path(x)) for x in seed.get("target_state_paths", [])]
        if any(not Path(x).exists() for x in seed.get("exact_complex_paths", [])):
            stats["missing_exact_paths"] += 1
            continue

        target_len = len(seed["target_sequence"])
        binder_len = len(seed["shared_binder_sequence"])
        try:
            constraints, _ = extract_anchor_constraints(seed, max_constraints=8)
        except Exception:
            constraints = [[] for _ in seed.get("target_state_paths", [])]

        selected_states: list[dict[str, Any]] = []
        all_silver = True
        had_any = False
        for state_idx, _target_path in enumerate(seed.get("target_state_paths", [])):
            pass_order = [("scout", False), ("second", False)]
            if state_idx < len(constraints) and constraints[state_idx]:
                pass_order.append(("anchor", True))
            state_selected: dict[str, Any] | None = None
            state_attempts: list[dict[str, Any]] = []
            for pass_name, _has_anchor in pass_order:
                stem = f"{sample_id}_{pass_name}_state{state_idx:02d}_boltz"
                try:
                    result = adapter.parse_existing_output(
                        sample_dir,
                        stem,
                        sample_id=f"{sample_id}_{pass_name}",
                        state_index=state_idx,
                        target_len=target_len,
                        binder_len=binder_len,
                    )
                    had_any = True
                except Exception as exc:
                    state_attempts.append({"state_index": state_idx, "pass_name": pass_name, "input_stem": stem, "status": "missing_or_parse_failed", "error": type(exc).__name__})
                    continue
                try:
                    target_pred = chain_to_atom37_any(result.complex_path, "A", target_len)
                    binder_pred = chain_to_atom37_any(result.complex_path, "B", binder_len)
                    geom = interface_labels(target_pred, binder_pred)["metrics"]
                    bronze = passes_bronze_stage06(result.metrics, geom)
                    silver = passes_silver(result.metrics, geom)
                    rec = compact_metrics(result.metrics, geom, pass_name=pass_name, state_idx=state_idx, stem=stem)
                    rec.update({"sample_id": sample_id, "status": "parsed", "bronze": bool(bronze), "silver": bool(silver)})
                    state_attempts.append(rec)
                    stats["state_level_total_attempted"] += 1
                    if bronze and state_selected is None:
                        copied_root = evidence_dir / sample_id / stem
                        pred_dir = adapter.prediction_dir(sample_dir, stem)
                        copied_pred_dir = copy_if_exists(pred_dir, copied_root / "prediction")
                        copied_yaml = copy_if_exists(sample_dir / "inputs" / f"{stem}.yaml", copied_root / "inputs" / f"{stem}.yaml")
                        copied_log = copy_if_exists(sample_dir / "logs" / f"{stem}.log", copied_root / "logs" / f"{stem}.log")
                        msa_dir = sample_dir / f"boltz_results_{stem}" / "msa"
                        copied_msa = copy_if_exists(msa_dir, copied_root / "msa")
                        copied_complex = None
                        if copied_pred_dir:
                            rel_pred = REPO / copied_pred_dir
                            pdbs = sorted(rel_pred.glob("*.pdb"))
                            copied_complex = str(pdbs[0].relative_to(REPO)) if pdbs else None
                        state_selected = dict(rec)
                        state_selected.update(
                            {
                                "complex_path": copied_complex or str(result.complex_path),
                                "confidence_path": str(result.confidence_path),
                                "copied_prediction_dir": copied_pred_dir,
                                "copied_yaml": copied_yaml,
                                "copied_log": copied_log,
                                "copied_msa_dir": copied_msa,
                            }
                        )
                        selected_states.append(state_selected)
                        stats["state_level_bronze"] += 1
                        stats["state_level_total_selected"] += 1
                        if silver:
                            stats["state_level_silver"] += 1
                        else:
                            all_silver = False
                        break
                except Exception as exc:
                    stats["parse_failures"] += 1
                    state_attempts.append({"state_index": state_idx, "pass_name": pass_name, "input_stem": stem, "status": "metric_failed", "error": repr(exc)[:500]})
                    continue
            if state_selected is None:
                all_silver = False
            for rec in state_attempts:
                rec.setdefault("sample_id", sample_id)
                state_attempt_records.append(rec)

        if had_any:
            stats["base_seeds_with_any_finished_pass"] += 1
        if len(selected_states) >= args.min_states:
            nactive = len(selected_states)
            stats["accepted_bases_bronze_ge2"] += 1
            if nactive >= 3:
                stats["accepted_bases_bronze_ge3"] += 1
            stats["accepted_variants_est"] += len(variant_state_subsets(nactive)) * 2
            if all_silver:
                stats["silver_bases_all_states"] += 1
            accepted = {
                "sample_id": sample_id,
                "target_id": seed.get("target_id"),
                "family_split_key": seed.get("split_group") or seed.get("family_split_key"),
                "target_motion_rmsd": seed.get("target_motion_rmsd"),
                "shared_binder_sequence": seed.get("shared_binder_sequence"),
                "target_sequence": seed.get("target_sequence"),
                "target_state_paths": seed.get("target_state_paths"),
                "target_chain_id": seed.get("target_chain_id"),
                "binder_chain_id": seed.get("binder_chain_id"),
                "source_seed_manifest": str(args.seed_manifest),
                "source_scratch": str(args.scratch),
                "accepted_state_count": nactive,
                "selected_states": selected_states,
                "worst_state_metrics": aggregate_worst(selected_states),
                "source_tier": "partial_stage07_boltz_salvage",
            }
            accepted_bases.append(accepted)
            if len(examples) < args.max_examples:
                examples.append({"sample_id": sample_id, "accepted_state_count": nactive, "selected_states": selected_states})
        elif had_any:
            stats["partial_or_failed_bases"] += 1

    stats["examples"] = examples
    stats["elapsed_sec"] = time.time() - started
    stats["recommendation"] = "stop_job_and_rerun_v2" if stats["accepted_bases_bronze_ge2"] < max(16, stats["base_seeds_with_any_finished_pass"] * 0.30) else "continue_or_finish_possible"
    stats["reason"] = "Low all-state Bronze hit rate and zero/low Silver under old scout/second/anchor ordering; salvage accepted partials, then rerun with filtered seeds and anchor-first constraints."

    with (out_dir / "accepted_bases_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(accepted_bases, f, indent=2, ensure_ascii=False)
    with (out_dir / "state_attempts.jsonl").open("w", encoding="utf-8") as f:
        for rec in state_attempt_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out_dir / "salvage_summary.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    with args.summary.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
