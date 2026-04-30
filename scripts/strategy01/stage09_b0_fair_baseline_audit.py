#!/usr/bin/env python
"""Stage09 fair B0 baseline audit for Strategy01 exact benchmark.

This script does not fabricate a B0 result. It defines the fair baseline
contract, checks the local Complexa pipeline assets, and writes the exact
artifact schema that a native or same-refiner B0 run must satisfy before it can
be compared against B1.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def exists(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None}


def run_quiet(cmd: list[str], cwd: Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)
        return {"cmd": cmd, "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "error": f"{type(exc).__name__}:{exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--v-exact-dataset", type=Path, default=REPO / "data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt")
    parser.add_argument("--v-exact-manifest", type=Path, default=REPO / "data/strategy01/stage08_high_quality_dataset/V_exact_main_manifest.json")
    parser.add_argument("--b1-summary", type=Path, default=REPO / "reports/strategy01/probes/stage08b_vexact_sampling_summary.json")
    parser.add_argument("--output", type=Path, default=REPO / "reports/strategy01/probes/stage09_b0_fair_baseline_audit.json")
    args = parser.parse_args()

    repo = args.repo
    python = sys.executable
    complexa_bin = shutil.which("complexa")
    core_files = {
        "cli_runner": exists(repo / "src/proteinfoundation/cli/cli_runner.py"),
        "generate_py": exists(repo / "src/proteinfoundation/generate.py"),
        "search_pipeline_config": exists(repo / "configs/search_binder_local_pipeline.yaml"),
        "binder_generate_config": exists(repo / "configs/pipeline/binder/binder_generate.yaml"),
        "targets_dict": exists(repo / "configs/targets/targets_dict.yaml"),
        "baseline_ckpt": exists(repo / "ckpts/complexa.ckpt"),
        "stage_init_ckpt_copy": exists(repo / "ckpts/stage03_multistate_loss/complexa_init_readonly_copy.ckpt"),
        "v_exact_dataset": exists(args.v_exact_dataset),
        "v_exact_manifest": exists(args.v_exact_manifest),
        "b1_summary": exists(args.b1_summary),
    }
    cli_probe = run_quiet([python, "-m", "proteinfoundation.cli.cli_runner", "--help"], repo)

    b0_contract = {
        "artifact_schema": {
            "summary_json": {
                "rows": [
                    {
                        "sample_id": "must match V_exact sample_id",
                        "out_dir": "directory containing stateXX_binder_ca.pdb or per-state converted baseline outputs",
                        "pred_sequence": "final B0 shared sequence after native or same-refiner sequence finalization",
                        "sequence_identity_to_reference": "optional diagnostic only; not a selection criterion",
                    }
                ]
            },
            "per_sample_files": [
                "shared_sequence.fasta",
                "state00_binder_ca.pdb",
                "state01_binder_ca.pdb",
                "state02_binder_ca.pdb if K=3",
            ],
        },
        "fair_b0_levels": [
            {
                "name": "B0-native",
                "definition": "Run original Complexa single-state binder generation and its native sequence/refinement pipeline. Re-evaluate the resulting one shared sequence/pose against every target state with the same exact-geometry evaluator used by B1.",
                "allowed_inputs": "A single target state and user-visible hotspot/target config only. No V_exact labels or future states during generation.",
                "selection_rule": "Same candidate budget as B1. Candidate selection may use native Complexa filtering, but not exact V_exact labels.",
            },
            {
                "name": "B0-same-refiner",
                "definition": "Use original Complexa single-state backbone/latent generation, then apply the same sequence/refinement/adapter layer used by B1 so the comparison isolates multistate conditioning rather than sequence-head availability.",
                "allowed_inputs": "Single-state target during generation; same downstream refiner for B0 and B1.",
                "selection_rule": "Same N candidates, same refiner calls, same exact evaluator.",
            },
            {
                "name": "B0-oracle-posthoc-diagnostic",
                "definition": "Post-hoc select among single-state candidates after multi-state exact evaluation. This is an upper-bound diagnostic only and must not be used as the main leaderboard baseline.",
                "allowed_inputs": "May use exact metrics after generation only for diagnostic selection.",
                "selection_rule": "Report separately as oracle/diagnostic.",
            },
        ],
    }
    required_before_leaderboard = [
        "Create V_exact-derived single-state target configs without leaking binder coordinates or exact contacts.",
        "Run B0-native on at least the same V_exact sample subset as B1 with fixed N candidates per target.",
        "Convert native Complexa outputs into the Stage08B summary schema above.",
        "Run scripts/strategy01/stage08b_full_exact_benchmark.py with --b0-summary pointing to the converted B0 summary.",
        "Repeat for B0-same-refiner if native B0 lacks a comparable final shared sequence artifact.",
    ]
    status = "blocked_by_missing_b0_generation_artifact"
    if args.output.exists():
        status = "audit_refreshed_missing_b0_generation_artifact"
    summary = {
        "status": status,
        "repo": str(repo),
        "complexa_bin": complexa_bin,
        "python": python,
        "core_files": core_files,
        "cli_probe": cli_probe,
        "b0_contract": b0_contract,
        "recommended_next_commands": [
            "complexa generate configs/search_binder_local_pipeline.yaml ++generation.task_name=<stage09_single_state_target> ++run_name=stage09_b0_native_<sample>",
            "python scripts/strategy01/stage09_convert_b0_outputs_to_exact_schema.py --input <native_output_dir> --v-exact-dataset data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt --output reports/strategy01/probes/stage09_b0_native_summary.json",
            "python scripts/strategy01/stage08b_full_exact_benchmark.py --b0-summary reports/strategy01/probes/stage09_b0_native_summary.json --output reports/strategy01/probes/stage09_b0_b1_exact_benchmark.json",
        ],
        "required_before_leaderboard": required_before_leaderboard,
        "interpretation": "B0 is now defined as a strict artifact contract. Stage09 does not fabricate a baseline. B1 cannot be claimed better until B0-native or B0-same-refiner produces this artifact.",
    }
    write_json(args.output, summary)
    print(json.dumps({"status": summary["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()