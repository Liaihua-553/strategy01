#!/usr/bin/env python
"""Stage07 MSA A/B probe for cloud MSA vs cached MSA replay.

The probe intentionally keeps local HHblits/MMseqs2 as diagnostics until its
quality is proven.  It uses one already accepted cloud-MSA Boltz prediction from
partial salvage as the online reference, then reruns Boltz2 with the copied MSA
CSV files and no remote MSA server.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.strategy01.stage04_predictors.boltz_adapter import BoltzAdapter, BoltzRunConfig
from scripts.strategy01.stage05_build_predictor_multistate_dataset import chain_to_atom37_any, interface_labels
from scripts.strategy01.stage06_build_boltz_production_dataset import passes_bronze_stage06, passes_silver, write_chain_pdb


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def repo_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (REPO / p)


def run_cmd(cmd: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return {"cmd": cmd, "returncode": proc.returncode, "output": proc.stdout[-2000:]}
    except Exception as exc:
        return {"cmd": cmd, "error": repr(exc)}


def find_msa_csvs(msa_dir: Path) -> dict[str, Path | None]:
    csvs = sorted(msa_dir.glob("*.csv"))
    return {"A": csvs[0] if len(csvs) >= 1 else None, "B": csvs[1] if len(csvs) >= 2 else None}


def metric_record(result, target_len: int, binder_len: int) -> dict[str, Any]:
    target = chain_to_atom37_any(result.complex_path, "A", target_len)
    binder = chain_to_atom37_any(result.complex_path, "B", binder_len)
    labels = interface_labels(target, binder)
    geom = labels["metrics"]
    return {
        "complex_path": str(result.complex_path),
        "confidence_path": str(result.confidence_path),
        "metrics": result.metrics,
        "geometry": geom,
        "bronze_pass": passes_bronze_stage06(result.metrics, geom),
        "silver_pass": passes_silver(result.metrics, geom),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-manifest", type=Path, default=REPO / "data/strategy01/stage07_partial_1908299/accepted_bases_manifest.json")
    parser.add_argument("--seed-manifest", type=Path, default=REPO / "data/strategy01/stage07_seed_pool/T_prod_seed_manifest.json")
    parser.add_argument("--out-dir", type=Path, default=REPO / "data/strategy01/stage07_msa_ab_probe")
    parser.add_argument("--summary", type=Path, default=REPO / "reports/strategy01/probes/stage07_msa_ab_probe_summary.json")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--state-index", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    accepted = load_json(args.accepted_manifest)
    seeds = {s["sample_id"]: s for s in load_json(args.seed_manifest)}
    chosen_base = None
    chosen_state = None
    for base in accepted:
        if args.sample_id and base.get("sample_id") != args.sample_id:
            continue
        for st in base.get("selected_states", []):
            if args.state_index is not None and int(st.get("state_index")) != args.state_index:
                continue
            msa_dir = repo_path(st.get("copied_msa_dir", ""))
            csvs = find_msa_csvs(msa_dir) if msa_dir.exists() else {"A": None, "B": None}
            if csvs["A"] and csvs["B"]:
                chosen_base = base
                chosen_state = st
                break
        if chosen_base:
            break
    if not chosen_base or not chosen_state:
        raise RuntimeError("No accepted state with copied A/B MSA CSV files found.")

    sample_id = chosen_base["sample_id"]
    seed = seeds[sample_id]
    state_idx = int(chosen_state["state_index"])
    target_len = len(seed["target_sequence"])
    binder_len = len(seed["shared_binder_sequence"])
    target_chain = chain_to_atom37_any(repo_path(seed["target_state_paths"][state_idx]), seed.get("target_chain_id", "A"), target_len)
    template_path = args.out_dir / "templates" / f"{sample_id}_state{state_idx:02d}_target_A.pdb"
    write_chain_pdb(template_path, target_chain, chain_id="A")
    msa_dir = repo_path(chosen_state["copied_msa_dir"])
    csvs = find_msa_csvs(msa_dir)

    tool_status = {
        "mmseqs": run_cmd(["/home/kfliao/data/anaconda3/envs/proteina-complexa/bin/mmseqs", "version"], timeout=20),
        "hhblits": run_cmd(["/share/HHsuite/3.0.1/bin/hhblits", "-h"], timeout=20),
        "hhsuite_db_dir": str(Path("/share/HHsuite/databases")),
        "hhsuite_db_dir_exists": Path("/share/HHsuite/databases").exists(),
        "local_msa_policy": "diagnostic_only_until_quality_matches_cloud_msa",
    }

    reference = {
        "source": "salvaged_cloud_msa_boltz_output",
        "sample_id": sample_id,
        "state_index": state_idx,
        "input_stem": chosen_state.get("input_stem"),
        "metrics": chosen_state,
    }

    result_record = None
    if not args.dry_run:
        adapter = BoltzAdapter(
            BoltzRunConfig(
                recycling_steps=1,
                sampling_steps=5,
                diffusion_samples=1,
                use_potentials=False,
                use_no_kernels=True,
                use_msa_server=False,
                num_workers=0,
            )
        )
        result = adapter.predict(
            target_pdb=template_path,
            binder_sequence=seed["shared_binder_sequence"],
            out_dir=args.out_dir / "cache_replay",
            sample_id=f"{sample_id}_cache_replay",
            state_index=state_idx,
            target_sequence=target_chain.seq_str,
            target_template_pdb=template_path,
            use_template=True,
            force_template=True,
            template_threshold=2.0,
            msa={"A": str(csvs["A"]), "B": str(csvs["B"])},
            target_len=target_len,
            binder_len=binder_len,
            timeout_sec=args.timeout_sec,
        )
        result_record = metric_record(result, target_len, binder_len)

    summary = {
        "status": "dry_run" if args.dry_run else "completed",
        "sample_id": sample_id,
        "state_index": state_idx,
        "reference_cloud_msa": reference,
        "cache_replay": result_record,
        "msa_csvs": {k: str(v) if v else None for k, v in csvs.items()},
        "tool_status": tool_status,
        "elapsed_sec": time.time() - started,
        "interpretation": "Use cloud MSA as quality baseline. Cached MSA replay is acceptable only if Bronze/Silver and worst-state metrics do not degrade materially. Local HHblits/MMseqs2 remains a future A/B branch, not the default production path.",
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
