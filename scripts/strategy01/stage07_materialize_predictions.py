#!/usr/bin/env python
"""Materialize Stage07 Boltz predicted complexes from node-local /tmp into repo data.

This script is intended to run on the same compute node that produced the Boltz
scratch directory, because Stage07 production keeps accepted prediction paths in
node-local /tmp until materialized. It copies only files referenced by accepted
samples, then rewrites dataset .pt files to point at persistent repo paths.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "strategy01" / "stage07_production_v23_wave1"


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def rel_dest_for(src: Path, dest_root: Path) -> Path:
    parts = src.parts
    if "boltz_outputs" in parts:
        idx = parts.index("boltz_outputs")
        rel = Path(*parts[idx + 1 :])
    else:
        rel = Path(src.name)
    return dest_root / rel


def copy_one(src_str: str, dest_root: Path, copied: dict[str, str], missing: list[str]) -> str:
    if not src_str:
        return src_str
    if src_str in copied:
        return copied[src_str]
    src = Path(src_str)
    dst = rel_dest_for(src, dest_root)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        copied[src_str] = str(dst)
        return str(dst)
    # If the path was already materialized in a previous run, preserve it.
    if dst.exists():
        copied[src_str] = str(dst)
        return str(dst)
    missing.append(src_str)
    return src_str


def materialize_dataset(input_path: Path, output_path: Path, dest_root: Path, max_missing: int, dry_run: bool = False) -> dict[str, Any]:
    data = safe_torch_load(input_path)
    samples = data.get("samples", [])
    copied: dict[str, str] = {}
    missing: list[str] = []
    new_data = copy.deepcopy(data)
    for sample in new_data.get("samples", []):
        new_pred_paths = []
        for p in sample.get("predicted_complex_paths", []) or []:
            new_pred_paths.append(copy_one(str(p), dest_root, copied, missing))
        sample["predicted_complex_paths"] = new_pred_paths
        # Update state metric file references when they exist. These are not needed
        # for AE extraction, but keeping them consistent helps later audit/debug.
        for metric in sample.get("state_metrics", []) or []:
            for key in ("pae_path", "plddt_path", "pde_path"):
                if key in metric and metric[key]:
                    metric[key] = copy_one(str(metric[key]), dest_root, copied, missing)
            if metric.get("prediction_dir"):
                old_dir = Path(str(metric["prediction_dir"]))
                if "boltz_outputs" in old_dir.parts:
                    idx = old_dir.parts.index("boltz_outputs")
                    metric["prediction_dir"] = str(dest_root / Path(*old_dir.parts[idx + 1 :]))
        sample["stage07_materialized_predictions"] = True
    unique_missing = sorted(set(missing))
    if unique_missing and len(unique_missing) > max_missing:
        raise RuntimeError(f"Too many missing source files: {len(unique_missing)} > {max_missing}. First: {unique_missing[:5]}")
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(new_data, output_path)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "dest_root": str(dest_root),
        "num_samples": len(samples),
        "copied_unique_sources": len(copied),
        "missing_unique_sources": len(unique_missing),
        "missing_examples": unique_missing[:20],
        "dry_run": dry_run,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--dest-root", type=Path, default=None)
    ap.add_argument("--summary", type=Path, default=REPO_ROOT / "reports" / "strategy01" / "probes" / "stage07_v23_wave1_materialize_summary.json")
    ap.add_argument("--max-missing", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    data_dir = args.data_dir
    dest_root = args.dest_root or (data_dir / "materialized_predictions")
    started = time.time()
    outputs = {}
    mapping = {
        "all": (data_dir / "stage06_predictor_pilot_samples.pt", data_dir / "stage06_predictor_pilot_samples_materialized.pt"),
        "train": (data_dir / "T_prod_train.pt", data_dir / "T_prod_train_materialized.pt"),
        "val": (data_dir / "T_prod_val.pt", data_dir / "T_prod_val_materialized.pt"),
    }
    for name, (inp, outp) in mapping.items():
        if not inp.exists():
            outputs[name] = {"exists": False, "input": str(inp)}
            continue
        outputs[name] = materialize_dataset(inp, outp, dest_root, args.max_missing, args.dry_run)
    summary = {
        "status": "passed" if all(v.get("missing_unique_sources", 0) == 0 for v in outputs.values() if v.get("exists", True)) else "partial",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_sec": time.time() - started,
        "outputs": outputs,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
