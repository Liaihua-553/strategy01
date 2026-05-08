#!/usr/bin/env python3
"""Stage12 exact benchmark contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

DEFAULT_STATE = REPO / "reports/strategy01/probes/stage12_benchmark_contract.json"


def validate_main_eval_output_source(source: str) -> None:
    if source != "state_specific":
        raise ValueError(
            "Stage12 exact benchmark must evaluate bb_ca_states[k]/local_latents_states[k]; "
            f"legacy or averaged output source is forbidden: {source}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Stage12 B0/B1/B2 benchmark contract")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state-json", default=str(DEFAULT_STATE))
    parser.add_argument("--prediction-source", default="state_specific", choices=["state_specific", "legacy_average"])
    return parser.parse_args()


def main():
    args = parse_args()
    validate_main_eval_output_source(args.prediction_source)
    state = {
        "stage": "stage12_exact_benchmark",
        "status": "dry_run_passed" if args.dry_run else "ready",
        "systems": {
            "B0_static_native": "single-state Complexa-native generation/refold baseline",
            "B0_static_transfer_relax": "source-pose transfer/repair baseline, not de-novo B1",
            "B1_stage12_de_novo": "target-only coupled full multistate product-space flow",
            "B2_exact": "experimental complex geometry and true binder sequence upper bound",
        },
        "hard_failures": ["severe_clash"],
        "metrics": ["shared_sequence_identity", "contact_F1", "contact_persistence", "worst_interface_RMSD", "clash_rate"],
        "prediction_source": args.prediction_source,
        "forbidden_main_eval_output": "legacy weighted-average bb_ca/local_latents",
    }
    out = Path(args.state_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
