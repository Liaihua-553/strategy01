#!/usr/bin/env python
"""Stage08B exact/reference geometry benchmark and B1 smoke diagnostics.

This script intentionally separates two facts:

1. V_exact exact-only currently provides experimental reference geometry. It is
   used to summarize B2 upper-bound contact/interface properties.
2. The available generated B1 artifact is a smoke sample from the predictor-
   derived training set, not a full V_exact generation benchmark. We still
   compute its direct contact/RMSD diagnostics and mark it as a smoke result.

No B0 baseline generation is fabricated here. If no B0 artifact is supplied, the
summary records B0 as not_run.
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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def as_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def parse_ca(path: Path, chain_id: str | None = None) -> list[tuple[str, torch.Tensor]]:
    rows: list[tuple[str, torch.Tensor]] = []
    seen: set[tuple[str, str, str]] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        atom = line[12:16].strip()
        if atom != "CA":
            continue
        ch = line[21].strip() or "_"
        if chain_id is not None and ch != chain_id:
            continue
        resid = line[22:27]
        key = (ch, resid, atom)
        if key in seen:
            continue
        seen.add(key)
        try:
            xyz = torch.tensor([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=torch.float32)
        except ValueError:
            continue
        rows.append((f"{ch}:{resid.strip()}", xyz))
    return rows


def contact_set(target_ca: torch.Tensor, binder_ca: torch.Tensor, cutoff_a: float) -> set[tuple[int, int]]:
    if target_ca.numel() == 0 or binder_ca.numel() == 0:
        return set()
    d = torch.cdist(target_ca.float(), binder_ca.float())
    idx = torch.nonzero(d <= cutoff_a, as_tuple=False)
    return {(int(i), int(j)) for i, j in idx.tolist()}


def f1(pred: set[tuple[int, int]], ref: set[tuple[int, int]]) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    inter = len(pred & ref)
    precision = inter / max(1, len(pred))
    recall = inter / max(1, len(ref))
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def kabsch_rmsd_a(pred: torch.Tensor, ref: torch.Tensor) -> float:
    n = min(pred.shape[0], ref.shape[0])
    if n < 3:
        return math.nan
    x = pred[:n].double()
    y = ref[:n].double()
    x0 = x - x.mean(dim=0, keepdim=True)
    y0 = y - y.mean(dim=0, keepdim=True)
    cov = x0.T @ y0
    u, _, vh = torch.linalg.svd(cov)
    d = torch.sign(torch.det(vh.T @ u.T))
    corr = torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=torch.double))
    rot = vh.T @ corr @ u.T
    xa = x0 @ rot.T
    return float(torch.sqrt(((xa - y0) ** 2).sum(dim=-1).mean()).item())


def direct_rmsd_a(pred: torch.Tensor, ref: torch.Tensor) -> float:
    n = min(pred.shape[0], ref.shape[0])
    if n < 1:
        return math.nan
    return float(torch.sqrt(((pred[:n].float() - ref[:n].float()) ** 2).sum(dim=-1).mean()).item())


def summarize_values(values: list[float], lower_is_better: bool = False) -> dict[str, float | int | None]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "worst": None, "best": None}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "worst": max(vals) if lower_is_better else min(vals),
        "best": min(vals) if lower_is_better else max(vals),
    }


def exact_reference_metrics(manifest_path: Path, max_samples: int, cutoff_a: float) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if max_samples > 0:
        manifest = manifest[:max_samples]
    sample_rows: list[dict[str, Any]] = []
    all_counts: list[float] = []
    persistence_vals: list[float] = []
    motion_vals: list[float] = []
    failures: list[dict[str, Any]] = []
    for item in manifest:
        target_chain = item.get("target_chain_id")
        binder_chain = item.get("binder_chain_id")
        contacts_by_state: list[set[tuple[int, int]]] = []
        counts: list[int] = []
        state_failures: list[str] = []
        for idx, complex_path in enumerate(item.get("exact_complex_paths") or []):
            path = as_path(complex_path)
            try:
                target_rows = parse_ca(path, target_chain)
                binder_rows = parse_ca(path, binder_chain)
                if len(target_rows) < 3 or len(binder_rows) < 3:
                    raise ValueError("too_few_ca")
                cset = contact_set(
                    torch.stack([x for _, x in target_rows]),
                    torch.stack([x for _, x in binder_rows]),
                    cutoff_a,
                )
                contacts_by_state.append(cset)
                counts.append(len(cset))
                all_counts.append(float(len(cset)))
            except Exception as exc:  # noqa: BLE001
                state_failures.append(f"state{idx}:{type(exc).__name__}:{exc}")
        union = set().union(*contacts_by_state) if contacts_by_state else set()
        inter = set(contacts_by_state[0]) if contacts_by_state else set()
        for cset in contacts_by_state[1:]:
            inter &= cset
        persistence = len(inter) / max(1, len(union)) if union else 0.0
        if contacts_by_state:
            persistence_vals.append(persistence)
        motion = (item.get("motion_metrics") or {}).get("target_backbone_rmsd_A")
        if motion is not None:
            motion_vals.append(float(motion))
        row = {
            "sample_id": item.get("sample_id"),
            "target_id": item.get("target_id"),
            "k_states": len(item.get("exact_complex_paths") or []),
            "contact_counts": counts,
            "min_contact_count": min(counts) if counts else 0,
            "mean_contact_count": sum(counts) / len(counts) if counts else 0.0,
            "contact_persistence": persistence,
            "motion_bin": item.get("motion_bin"),
            "target_motion_rmsd_A": motion,
            "failures": state_failures,
        }
        sample_rows.append(row)
        if state_failures:
            failures.append({"sample_id": item.get("sample_id"), "failures": state_failures})
    return {
        "manifest": str(manifest_path),
        "sample_count": len(manifest),
        "contact_cutoff_A": cutoff_a,
        "b2_exact_reference": {
            "contact_count": summarize_values(all_counts),
            "contact_persistence": summarize_values(persistence_vals),
            "target_motion_rmsd_A": summarize_values(motion_vals, lower_is_better=True),
            "samples_with_parse_failures": len(failures),
        },
        "sample_rows": sample_rows,
        "failures": failures[:20],
    }


def load_sample(dataset: Path, sample_id: str) -> dict[str, Any] | None:
    if not dataset.exists():
        return None
    data = torch.load(dataset, map_location="cpu", weights_only=False)
    for sample in data.get("samples", []):
        if sample.get("sample_id") == sample_id:
            return sample
    return None


def smoke_b1_geometry(smoke_summary_path: Path, dataset_path: Path, cutoff_a: float) -> dict[str, Any]:
    if not smoke_summary_path.exists():
        return {"status": "not_run", "reason": f"missing {smoke_summary_path}"}
    smoke = json.loads(smoke_summary_path.read_text(encoding="utf-8"))
    sample_id = smoke.get("sample_id")
    sample = load_sample(dataset_path, sample_id)
    if sample is None:
        return {"status": "not_run", "reason": f"sample {sample_id} not found in {dataset_path}"}
    out_dir = as_path(smoke.get("out_dir"))
    rows: list[dict[str, Any]] = []
    f1s: list[float] = []
    direct_rmsds: list[float] = []
    aligned_rmsds: list[float] = []
    valid_k = int(sample["state_present_mask"].bool().sum().item())
    target_chain = sample.get("target_chain_id") or "A"
    binder_chain = sample.get("binder_chain_id") or "B"
    for k in range(valid_k):
        gen_path = out_dir / f"state{k:02d}_binder_ca.pdb"
        ref_complex = as_path(sample["predicted_complex_paths"][k])
        target_path = as_path(sample["target_state_paths"][k])
        target_rows = parse_ca(target_path, target_chain)
        if len(target_rows) < 3:
            target_rows = parse_ca(ref_complex, target_chain)
        ref_binder_rows = parse_ca(ref_complex, binder_chain)
        gen_rows = parse_ca(gen_path, "B")
        if len(target_rows) < 3 or len(ref_binder_rows) < 3 or len(gen_rows) < 3:
            rows.append({"state_index": k, "status": "failed_parse"})
            continue
        target_ca = torch.stack([x for _, x in target_rows])
        ref_binder_ca = torch.stack([x for _, x in ref_binder_rows])
        gen_ca = torch.stack([x for _, x in gen_rows])
        ref_contacts = contact_set(target_ca, ref_binder_ca, cutoff_a)
        gen_contacts = contact_set(target_ca, gen_ca, cutoff_a)
        cf1 = f1(gen_contacts, ref_contacts)
        dr = direct_rmsd_a(gen_ca, ref_binder_ca)
        ar = kabsch_rmsd_a(gen_ca, ref_binder_ca)
        f1s.append(cf1)
        direct_rmsds.append(dr)
        aligned_rmsds.append(ar)
        rows.append({
            "state_index": k,
            "status": "ok",
            "reference_contact_count": len(ref_contacts),
            "generated_contact_count": len(gen_contacts),
            "contact_f1_vs_reference_label": cf1,
            "direct_binder_ca_rmsd_A": dr,
            "aligned_binder_ca_rmsd_A": ar,
        })
    return {
        "status": "smoke_only",
        "sample_id": sample_id,
        "source_tier": sample.get("source_tier"),
        "dataset": str(dataset_path),
        "smoke_summary": str(smoke_summary_path),
        "note": "This is B1 against predictor-derived labels, not V_exact-only.",
        "b1_strategy01": {
            "contact_f1_vs_label": summarize_values(f1s),
            "direct_binder_ca_rmsd_A": summarize_values(direct_rmsds, lower_is_better=True),
            "aligned_binder_ca_rmsd_A": summarize_values(aligned_rmsds, lower_is_better=True),
        },
        "state_rows": rows,
        "sequence_identity_to_reference": smoke.get("sequence_identity_to_reference"),
        "pred_sequence": smoke.get("pred_sequence"),
        "reference_sequence": smoke.get("reference_sequence"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v-exact-manifest", type=Path, default=REPO / "data/strategy01/stage08_high_quality_dataset/V_exact_main_manifest.json")
    parser.add_argument("--smoke-summary", type=Path, default=REPO / "reports/strategy01/probes/stage07_multistate_sampling_smoke_summary.json")
    parser.add_argument("--smoke-dataset", type=Path, default=REPO / "data/strategy01/stage07_production_v23_wave1/stage06_predictor_pilot_samples_materialized_ae_latents.pt")
    parser.add_argument("--output", type=Path, default=REPO / "reports/strategy01/probes/stage08b_exact_geometry_benchmark_summary.json")
    parser.add_argument("--contact-cutoff-a", type=float, default=10.0)
    parser.add_argument("--max-exact-samples", type=int, default=0)
    args = parser.parse_args()

    exact = exact_reference_metrics(args.v_exact_manifest, args.max_exact_samples, args.contact_cutoff_a)
    b1 = smoke_b1_geometry(args.smoke_summary, args.smoke_dataset, args.contact_cutoff_a)
    summary = {
        "status": "completed_with_limitations",
        "contact_cutoff_A": args.contact_cutoff_a,
        "B2_exact_only_reference": exact,
        "B1_strategy01_smoke": b1,
        "B0_baseline": {
            "status": "not_run",
            "reason": "No baseline Complexa single-state generation artifact exists for the V_exact manifest in Stage08B. The script does not fabricate B0 metrics.",
        },
        "limitations": [
            "V_exact currently summarizes exact reference geometry but has not been converted into a generation-ready tensor dataset in this stage.",
            "Available B1 generated pose is a smoke sample from predictor-derived labels, not exact-only.",
            "Full B0/B1/B2 exact benchmark remains the next blocked item after building a V_exact tensor dataset and baseline generation artifacts.",
        ],
    }
    write_json(args.output, summary)
    print(json.dumps({"status": summary["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
