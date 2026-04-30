#!/usr/bin/env python
"""Full exact-geometry benchmark for available Stage08B artifacts.

B2 is the experimental exact complex geometry.
B1 is Strategy01 state-specific generation on the V_exact tensor dataset.
B0 is accepted only if a baseline generation summary with the same artifact
schema is supplied; otherwise it is explicitly reported as not_run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.strategy01.stage08b_exact_geometry_benchmark import (  # noqa: E402
    as_path,
    contact_set,
    direct_rmsd_a,
    exact_reference_metrics,
    f1,
    kabsch_rmsd_a,
    parse_ca,
    summarize_values,
)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_samples(dataset: Path) -> dict[str, dict[str, Any]]:
    data = torch.load(dataset, map_location="cpu", weights_only=False)
    return {str(s.get("sample_id")): s for s in data["samples"]}


def std(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))


def benchmark_generation_summary(summary_path: Path, dataset_path: Path, label: str, cutoff_a: float) -> dict[str, Any]:
    if not summary_path or not summary_path.exists():
        return {"status": "not_run", "label": label, "reason": f"missing summary {summary_path}"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    samples = load_samples(dataset_path)
    rows_out: list[dict[str, Any]] = []
    all_f1: list[float] = []
    all_direct: list[float] = []
    all_aligned: list[float] = []
    sample_persistence: list[float] = []
    sample_std: list[float] = []
    clash_flags: list[bool] = []
    for row in summary.get("rows", []):
        sample_id = str(row.get("sample_id"))
        sample = samples.get(sample_id)
        if sample is None:
            rows_out.append({"sample_id": sample_id, "status": "missing_sample"})
            continue
        out_dir = as_path(row["out_dir"])
        target_chain = sample.get("target_chain_id") or "A"
        binder_chain = sample.get("binder_chain_id") or "B"
        valid_k = int(sample["state_present_mask"].bool().sum().item())
        state_rows = []
        gen_contact_sets: list[set[tuple[int, int]]] = []
        sample_f1s: list[float] = []
        for k in range(valid_k):
            gen_path = out_dir / f"state{k:02d}_binder_ca.pdb"
            target_path = as_path(sample["target_state_paths"][k])
            exact_complex = as_path((sample.get("exact_complex_paths") or sample.get("predicted_complex_paths"))[k])
            try:
                target_rows = parse_ca(target_path, target_chain)
                if len(target_rows) < 3:
                    target_rows = parse_ca(exact_complex, target_chain)
                ref_binder_rows = parse_ca(exact_complex, binder_chain)
                gen_rows = parse_ca(gen_path, "B")
                if len(target_rows) < 3 or len(ref_binder_rows) < 3 or len(gen_rows) < 3:
                    raise ValueError("too_few_ca")
                target_ca = torch.stack([x for _, x in target_rows])
                ref_binder_ca = torch.stack([x for _, x in ref_binder_rows])
                gen_ca = torch.stack([x for _, x in gen_rows])
                ref_contacts = contact_set(target_ca, ref_binder_ca, cutoff_a)
                gen_contacts = contact_set(target_ca, gen_ca, cutoff_a)
                cf1 = f1(gen_contacts, ref_contacts)
                dr = direct_rmsd_a(gen_ca, ref_binder_ca)
                ar = kabsch_rmsd_a(gen_ca, ref_binder_ca)
                min_target_dist = float(torch.cdist(target_ca.float(), gen_ca.float()).min().item())
                severe_clash = min_target_dist < 2.8
                all_f1.append(cf1)
                all_direct.append(dr)
                all_aligned.append(ar)
                sample_f1s.append(cf1)
                clash_flags.append(severe_clash)
                gen_contact_sets.append(gen_contacts)
                state_rows.append(
                    {
                        "state_index": k,
                        "status": "ok",
                        "reference_contact_count": len(ref_contacts),
                        "generated_contact_count": len(gen_contacts),
                        "contact_f1": cf1,
                        "direct_binder_ca_rmsd_A": dr,
                        "aligned_binder_ca_rmsd_A": ar,
                        "min_target_generated_ca_distance_A": min_target_dist,
                        "severe_clash": severe_clash,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                state_rows.append({"state_index": k, "status": "failed", "error": f"{type(exc).__name__}:{exc}"})
        union = set().union(*gen_contact_sets) if gen_contact_sets else set()
        inter = set(gen_contact_sets[0]) if gen_contact_sets else set()
        for cset in gen_contact_sets[1:]:
            inter &= cset
        persistence = len(inter) / max(1, len(union)) if union else 0.0
        if gen_contact_sets:
            sample_persistence.append(persistence)
        if sample_f1s:
            sample_std.append(std(sample_f1s) or 0.0)
        rows_out.append(
            {
                "sample_id": sample_id,
                "status": "ok" if gen_contact_sets else "failed",
                "target_id": sample.get("target_id"),
                "sequence_identity_to_reference": row.get("sequence_identity_to_reference"),
                "contact_persistence_generated": persistence,
                "state_metric_std_contact_f1": std(sample_f1s),
                "state_rows": state_rows,
            }
        )
    return {
        "status": "passed" if rows_out else "empty",
        "label": label,
        "summary_path": str(summary_path),
        "dataset": str(dataset_path),
        "sample_count": len(rows_out),
        "state_count_ok": len(all_f1),
        "contact_f1": summarize_values(all_f1),
        "worst_interface_rmsd_A": summarize_values(all_direct, lower_is_better=True),
        "aligned_binder_ca_rmsd_A": summarize_values(all_aligned, lower_is_better=True),
        "contact_persistence": summarize_values(sample_persistence),
        "state_metric_std_contact_f1": summarize_values(sample_std, lower_is_better=True),
        "clash_rate": (sum(1 for x in clash_flags if x) / len(clash_flags)) if clash_flags else None,
        "rows": rows_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v-exact-manifest", type=Path, default=REPO / "data/strategy01/stage08_high_quality_dataset/V_exact_main_manifest.json")
    parser.add_argument("--v-exact-dataset", type=Path, default=REPO / "data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt")
    parser.add_argument("--b1-summary", type=Path, default=REPO / "reports/strategy01/probes/stage08b_vexact_sampling_summary.json")
    parser.add_argument("--b0-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=REPO / "reports/strategy01/probes/stage08b_full_exact_benchmark_summary.json")
    parser.add_argument("--contact-cutoff-a", type=float, default=10.0)
    args = parser.parse_args()

    b2 = exact_reference_metrics(args.v_exact_manifest, 0, args.contact_cutoff_a)
    b1 = benchmark_generation_summary(args.b1_summary, args.v_exact_dataset, "B1_strategy01_stage08b", args.contact_cutoff_a)
    if args.b0_summary:
        b0 = benchmark_generation_summary(args.b0_summary, args.v_exact_dataset, "B0_baseline_complexa", args.contact_cutoff_a)
    else:
        b0 = {
            "status": "not_run",
            "label": "B0_baseline_complexa",
            "reason": "Original Complexa baseline does not expose a shared sequence head in this Strategy01 tensor sampling path; no baseline generation artifact was available.",
        }
    summary = {
        "status": "completed_with_b0_missing" if b1.get("status") == "passed" else "incomplete",
        "contact_cutoff_A": args.contact_cutoff_a,
        "B0_baseline": b0,
        "B1_strategy01": b1,
        "B2_exact_reference": b2,
        "acceptance_check": {
            "b1_has_exact_samples": b1.get("sample_count", 0) > 0,
            "b0_available": b0.get("status") == "passed",
            "note": "B1 can be compared to B2 exact geometry; B0 remains unavailable until a baseline generation artifact is produced.",
        },
    }
    write_json(args.output, summary)
    print(json.dumps({"status": summary["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
