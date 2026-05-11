#!/usr/bin/env python3
"""Stage19 geometry benchmark for selected target-only candidates.

This evaluates generated state-specific binder CA PDBs against the tensor labels
in the current Strategy01 dataset. It intentionally reports the dataset
`source_tier` and does not claim V_exact-only status unless the input dataset is
exact-only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
for path in [REPO, REPO / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.strategy01.stage04_real_complex_loss_debug as s4  # noqa: E402
import scripts.strategy01.stage10_pose_init_training as s10  # noqa: E402
import scripts.strategy01.stage12_de_novo_multistate_training as s12  # noqa: E402


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s4.jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def as_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def parse_ca_pdb_nm(path: Path) -> torch.Tensor:
    coords = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            coords.append([float(line[30:38]) / 10.0, float(line[38:46]) / 10.0, float(line[46:54]) / 10.0])
        except ValueError:
            continue
    if not coords:
        return torch.empty(0, 3)
    return torch.tensor(coords, dtype=torch.float32)


def contact_set_from_dist(dist: torch.Tensor, cutoff_nm: float) -> set[tuple[int, int]]:
    idx = torch.nonzero(dist <= cutoff_nm, as_tuple=False)
    return {(int(i), int(j)) for i, j in idx.tolist()}


def label_contact_set(label: torch.Tensor, tmask: torch.Tensor, bmask: torch.Tensor) -> set[tuple[int, int]]:
    # Keep original residue indices so generated contact sets align to labels.
    idx = torch.nonzero((label > 0.5) & tmask[:, None] & bmask[None, :], as_tuple=False)
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


def direct_rmsd_nm(pred: torch.Tensor, ref: torch.Tensor) -> float:
    n = min(pred.shape[0], ref.shape[0])
    if n < 1:
        return math.nan
    return float(torch.sqrt(((pred[:n].float() - ref[:n].float()) ** 2).sum(dim=-1).mean()).item())


def aligned_rmsd_nm(pred: torch.Tensor, ref: torch.Tensor) -> float:
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


def summarize(vals: list[float], lower_is_better: bool = False) -> dict[str, Any]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return {"n": 0, "mean": None, "best": None, "worst": None}
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "best": min(xs) if lower_is_better else max(xs),
        "worst": max(xs) if lower_is_better else min(xs),
    }


def evaluate_sample(row: dict[str, Any], sample: dict[str, Any], contact_cutoff_nm: float, clash_cutoff_nm: float) -> dict[str, Any]:
    selected = row.get("selected_by_proxy") or {}
    dirs = selected.get("written_pdb_dirs") or []
    if not dirs:
        return {"sample_id": row.get("sample_id"), "status": "missing_pdb_dir"}
    pdb_dir = as_path(dirs[0])
    state_present = sample["state_present_mask"].bool()
    state_count = int(state_present.sum().item())
    target_ca = sample["x_target_states"][..., 1, :].float()
    target_mask = sample["target_mask_states"][..., 1].bool()
    binder_mask = sample["state_mask"].bool()
    ref_bb = sample["x_1_states"]["bb_ca"].float()
    labels = sample["interface_contact_labels"].float()
    state_rows = []
    gen_contacts_by_state: list[set[tuple[int, int]]] = []
    contact_f1s: list[float] = []
    direct_rmsds: list[float] = []
    aligned_rmsds: list[float] = []
    clash_flags: list[float] = []
    contact_counts: list[float] = []
    for k in range(state_count):
        pdb = pdb_dir / f"state{k:02d}_binder_ca.pdb"
        if not pdb.exists():
            state_rows.append({"state_index": k, "status": "missing_pdb", "path": str(pdb)})
            continue
        gen = parse_ca_pdb_nm(pdb)
        bmask = binder_mask[k]
        tmask = target_mask[k]
        n = min(gen.shape[0], int(bmask.sum().item()))
        if n < 1 or not bool(tmask.any()):
            state_rows.append({"state_index": k, "status": "empty_geometry", "path": str(pdb)})
            continue
        # Map sequential PDB rows back to valid binder residue positions.
        gen_full = torch.zeros_like(ref_bb[k])
        valid_bidx = torch.nonzero(bmask, as_tuple=False).flatten()
        gen_full[valid_bidx[:n]] = gen[:n]
        dist = torch.cdist(target_ca[k].float(), gen_full.float())
        valid_pair = tmask[:, None] & bmask[None, :]
        masked_dist = dist[valid_pair]
        min_dist = float(masked_dist.min().item()) if masked_dist.numel() else math.nan
        pred_contacts = contact_set_from_dist(dist.masked_fill(~valid_pair, 999.0), contact_cutoff_nm)
        ref_contacts = label_contact_set(labels[k], tmask, bmask)
        cf1 = f1(pred_contacts, ref_contacts)
        dr = direct_rmsd_nm(gen_full[bmask], ref_bb[k][bmask])
        ar = aligned_rmsd_nm(gen_full[bmask], ref_bb[k][bmask])
        severe = bool(math.isfinite(min_dist) and min_dist < clash_cutoff_nm)
        gen_contacts_by_state.append(pred_contacts)
        contact_f1s.append(cf1)
        direct_rmsds.append(dr)
        aligned_rmsds.append(ar)
        clash_flags.append(1.0 if severe else 0.0)
        contact_counts.append(float(len(pred_contacts)))
        state_rows.append({
            "state_index": k,
            "status": "ok",
            "pdb": str(pdb),
            "generated_contact_count": len(pred_contacts),
            "reference_contact_count": len(ref_contacts),
            "contact_f1": cf1,
            "direct_binder_ca_rmsd_A": dr * 10.0,
            "aligned_binder_ca_rmsd_A": ar * 10.0,
            "min_target_binder_distance_nm": min_dist,
            "severe_clash": severe,
        })
    union = set().union(*gen_contacts_by_state) if gen_contacts_by_state else set()
    inter = set(gen_contacts_by_state[0]) if gen_contacts_by_state else set()
    for cset in gen_contacts_by_state[1:]:
        inter &= cset
    persistence = len(inter) / max(1, len(union)) if union else 0.0
    return {
        "sample_id": row.get("sample_id"),
        "target_id": row.get("target_id"),
        "source_tier": sample.get("source_tier"),
        "quality_tier": sample.get("quality_tier"),
        "selected_seed": selected.get("seed"),
        "selected_safe": row.get("selected_is_safe"),
        "selected_proxy_identity": selected.get("shared_identity_mean_posthoc"),
        "selected_proxy_clash_rate": selected.get("target_severe_clash_rate"),
        "status": "ok" if state_rows else "no_states",
        "state_rows": state_rows,
        "sample_metrics": {
            "mean_contact_f1": sum(contact_f1s) / len(contact_f1s) if contact_f1s else None,
            "worst_contact_f1": min(contact_f1s) if contact_f1s else None,
            "mean_direct_rmsd_A": (sum(direct_rmsds) / len(direct_rmsds) * 10.0) if direct_rmsds else None,
            "worst_direct_rmsd_A": (max(direct_rmsds) * 10.0) if direct_rmsds else None,
            "mean_aligned_rmsd_A": (sum(aligned_rmsds) / len(aligned_rmsds) * 10.0) if aligned_rmsds else None,
            "worst_aligned_rmsd_A": (max(aligned_rmsds) * 10.0) if aligned_rmsds else None,
            "severe_clash_rate": sum(clash_flags) / len(clash_flags) if clash_flags else None,
            "contact_persistence": persistence,
            "mean_generated_contact_count": sum(contact_counts) / len(contact_counts) if contact_counts else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-json", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=s12.DEFAULT_DATASET)
    ap.add_argument("--split", default="val")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--contact-cutoff-nm", type=float, default=1.0)
    ap.add_argument("--clash-cutoff-nm", type=float, default=0.28)
    args = ap.parse_args()
    samples, manifest = s10.load_dataset(args.dataset)
    by_id = {s.get("sample_id"): s for s in samples if s.get("split") == args.split}
    report = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    rows = []
    for row in report.get("samples", []):
        sid = row.get("sample_id")
        sample = by_id.get(sid)
        if sample is None:
            rows.append({"sample_id": sid, "status": "sample_not_found"})
            continue
        rows.append(evaluate_sample(row, sample, args.contact_cutoff_nm, args.clash_cutoff_nm))
    sample_metrics = [r.get("sample_metrics", {}) for r in rows if r.get("status") == "ok"]
    exact_like = [r for r in rows if "exact" in str(r.get("sample_id", ""))]
    summary = {
        "stage": "stage19_candidate_geometry_benchmark",
        "status": "completed",
        "candidate_json": str(args.candidate_json),
        "dataset": str(args.dataset),
        "split": args.split,
        "candidate_sample_count": len(report.get("samples", [])),
        "evaluated_sample_count": len(sample_metrics),
        "exact_like_name_count": len(exact_like),
        "contact_cutoff_nm": args.contact_cutoff_nm,
        "clash_cutoff_nm": args.clash_cutoff_nm,
        "aggregate": {
            "mean_contact_f1": summarize([m.get("mean_contact_f1") for m in sample_metrics]),
            "worst_contact_f1": summarize([m.get("worst_contact_f1") for m in sample_metrics]),
            "mean_direct_rmsd_A": summarize([m.get("mean_direct_rmsd_A") for m in sample_metrics], lower_is_better=True),
            "worst_direct_rmsd_A": summarize([m.get("worst_direct_rmsd_A") for m in sample_metrics], lower_is_better=True),
            "mean_aligned_rmsd_A": summarize([m.get("mean_aligned_rmsd_A") for m in sample_metrics], lower_is_better=True),
            "severe_clash_rate": summarize([m.get("severe_clash_rate") for m in sample_metrics], lower_is_better=True),
            "contact_persistence": summarize([m.get("contact_persistence") for m in sample_metrics]),
            "mean_generated_contact_count": summarize([m.get("mean_generated_contact_count") for m in sample_metrics]),
        },
        "warning": "This evaluates against the current Stage10/Stage07 tensor labels. It is not V_exact-only unless the input dataset/report is exact-only.",
        "rows": rows,
    }
    write_json(args.output, summary)
    print(json.dumps({"status": "completed", "output": str(args.output), "aggregate": summary["aggregate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
