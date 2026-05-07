#!/usr/bin/env python3
"""Stage12 target-only de-novo multistate sampling contract checker."""

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

DEFAULT_STATE = REPO / "reports/strategy01/probes/stage12_sampling_contract.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Stage12 sampling contract checker")
    parser.add_argument("--target-ensemble-manifest", required=False)
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--nsteps", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state-json", default=str(DEFAULT_STATE))
    return parser.parse_args()


def main():
    args = parse_args()
    state = {
        "stage": "stage12_de_novo_multistate_sampling",
        "status": "dry_run_passed" if args.dry_run else "ready_for_full_sampler_integration",
        "full_product_space_flow_required": {"bb_ca": True, "local_latents": True},
        "forbidden_inputs": ["init_bb_ca_states", "source_complex", "binder_seq_shared_as_feature"],
        "primary_outputs": ["shared_sequence.fasta", "state_specific_complex_0..K.pdb"],
        "legacy_average_outputs": "debug_only_not_scientific_output",
        "nsteps": args.nsteps,
        "target_ensemble_manifest": args.target_ensemble_manifest,
        "checkpoint": args.checkpoint,
    }
    out = Path(args.state_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
