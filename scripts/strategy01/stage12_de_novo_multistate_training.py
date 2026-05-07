#!/usr/bin/env python3
"""Stage12 de-novo multistate training entrypoint contract."""

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

DEFAULT_STATE = REPO / "reports/strategy01/probes/stage12_training_entrypoint_state.json"

FORBIDDEN_MODEL_INPUT_KEYS = [
    "init_bb_ca_states",
    "source_bb_ca_states",
    "source_complex_paths",
    "source_interface_contact_labels",
    "source_interface_distance_labels",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Stage12 de-novo multistate training wrapper")
    parser.add_argument("--train-manifest", required=False)
    parser.add_argument("--val-manifest", required=False)
    parser.add_argument("--config", default="configs/training_local_latents_multistate_stage12_denovo.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state-json", default=str(DEFAULT_STATE))
    return parser.parse_args()


def main():
    args = parse_args()
    state = {
        "stage": "stage12_de_novo_multistate_training",
        "status": "dry_run_passed" if args.dry_run else "ready_requires_dataset_training_loop",
        "contract": {
            "target_only_model_inputs": True,
            "forbidden_model_input_keys": FORBIDDEN_MODEL_INPUT_KEYS,
            "true_binder_sequence_allowed_only_as_loss_label": True,
            "primary_generated_outputs": [
                "shared_sequence.fasta",
                "state_specific_complexes/*.pdb",
                "bb_ca_states",
                "local_latents_states",
            ],
        },
        "config": args.config,
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
    }
    out = Path(args.state_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
