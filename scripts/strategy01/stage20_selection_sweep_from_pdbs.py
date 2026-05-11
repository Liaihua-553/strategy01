#!/usr/bin/env python3
"""Stage20 label-free selection sweep over generated candidate PDBs.

The sweep computes label-free generated contact statistics for every candidate,
then compares several non-leaking selection rules against held-out tensor labels
for diagnosis. Label metrics are never used to select candidates.
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
import scripts.strategy01.stage19_candidate_geometry_benchmark as s19  # noqa: E402


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s4.jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def finite(x: Any, default: float = 0.0) -> float:
    return float(x) if isinstance(x, (int, float)) and math.isfinite(float(x)) else default


def candidate_geometry(candidate: dict[str, Any], sample: dict[str, Any], contact_cutoff_nm: float, clash_cutoff_nm: float) -> dict[str, Any]:
    dirs = candidate.get("written_pdb_dirs") or []
    if not dirs:
        return {"status": "missing_pdb_dir"}
    pdb_dir = s19.as_path(dirs[0])
    state_present = sample["state_present_mask"].bool()
    state_count = int(state_present.sum().item())
    target_ca = sample["x_target_states"][..., 1, :].float()
    target_mask = sample["target_mask_states"][..., 1].bool()
    binder_mask = sample["state_mask"].bool()
    ref_bb = sample["x_1_states"]["bb_ca"].float()
    labels = sample["interface_contact_labels"].float()
    gen_contacts_by_state=[]
    label_contact_f1=[]
    direct=[]
    clash=[]
    counts=[]
    hotspot_counts=[]
    hotspot = sample.get("target_hotspot_mask_states")
    hotspot_mask = hotspot.bool() if hotspot is not None else target_mask
    for k in range(state_count):
        pdb = pdb_dir / f"state{k:02d}_binder_ca.pdb"
        if not pdb.exists():
            continue
        gen = s19.parse_ca_pdb_nm(pdb)
        bmask = binder_mask[k]
        tmask = target_mask[k]
        n = min(gen.shape[0], int(bmask.sum().item()))
        if n < 1 or not bool(tmask.any()):
            continue
        gen_full = torch.zeros_like(ref_bb[k])
        valid_bidx = torch.nonzero(bmask, as_tuple=False).flatten()
        gen_full[valid_bidx[:n]] = gen[:n]
        dist = torch.cdist(target_ca[k].float(), gen_full.float())
        valid_pair = tmask[:, None] & bmask[None, :]
        pred = s19.contact_set_from_dist(dist.masked_fill(~valid_pair, 999.0), contact_cutoff_nm)
        ref = s19.label_contact_set(labels[k], tmask, bmask)
        cf1 = s19.f1(pred, ref)
        min_dist = float(dist[valid_pair].min().item()) if bool(valid_pair.any()) else math.nan
        hpair = hotspot_mask[k][:, None] & bmask[None, :]
        hcount = int(((dist <= contact_cutoff_nm) & hpair).sum().item()) if bool(hpair.any()) else 0
        gen_contacts_by_state.append(pred)
        label_contact_f1.append(cf1)
        direct.append(s19.direct_rmsd_nm(gen_full[bmask], ref_bb[k][bmask]) * 10.0)
        clash.append(1.0 if math.isfinite(min_dist) and min_dist < clash_cutoff_nm else 0.0)
        counts.append(float(len(pred)))
        hotspot_counts.append(float(hcount))
    union=set().union(*gen_contacts_by_state) if gen_contacts_by_state else set()
    inter=set(gen_contacts_by_state[0]) if gen_contacts_by_state else set()
    for c in gen_contacts_by_state[1:]: inter &= c
    persistence=len(inter)/max(1,len(union)) if union else 0.0
    return {
        "status": "ok" if label_contact_f1 else "no_states",
        "label_mean_contact_f1": sum(label_contact_f1)/len(label_contact_f1) if label_contact_f1 else None,
        "label_worst_contact_f1": min(label_contact_f1) if label_contact_f1 else None,
        "label_mean_direct_rmsd_A": sum(direct)/len(direct) if direct else None,
        "label_worst_direct_rmsd_A": max(direct) if direct else None,
        "generated_contact_persistence": persistence,
        "generated_contact_count_mean": sum(counts)/len(counts) if counts else 0.0,
        "generated_hotspot_contact_count_mean": sum(hotspot_counts)/len(hotspot_counts) if hotspot_counts else 0.0,
        "generated_severe_clash_rate": sum(clash)/len(clash) if clash else 1.0,
    }


def score_current(c: dict[str, Any]) -> float:
    safe_penalty = 1e6 if finite(c.get("generated_severe_clash_rate")) > 0 else 0.0
    return safe_penalty + finite(c.get("no_leak_score"), 999.0)


def score_persistence(c: dict[str, Any]) -> float:
    clash = finite(c.get("generated_severe_clash_rate"), 1.0)
    safe_penalty = 1e6 if clash > 0 else 0.0
    ent = finite(c.get("shared_entropy"), 10.0)
    dis = finite(c.get("state_seq_disagreement"), 10.0)
    persistence = finite(c.get("generated_contact_persistence"), 0.0)
    contact = min(finite(c.get("generated_contact_count_mean"), 0.0), 160.0)
    hotspot = min(finite(c.get("generated_hotspot_contact_count_mean"), 0.0), 80.0)
    # Lower is better. Reward persistent target contact shell while keeping sequence confidence in the score.
    return safe_penalty + ent + 2.0 * dis - 2.0 * persistence - 0.004 * contact - 0.006 * hotspot


def score_contact_shell(c: dict[str, Any]) -> float:
    clash = finite(c.get("generated_severe_clash_rate"), 1.0)
    safe_penalty = 1e6 if clash > 0 else 0.0
    contact = finite(c.get("generated_contact_count_mean"), 0.0)
    hotspot = finite(c.get("generated_hotspot_contact_count_mean"), 0.0)
    persistence = finite(c.get("generated_contact_persistence"), 0.0)
    # Prefer moderate/high contacts, but penalize extremely sticky all-over-target candidates.
    shell = abs(contact - 85.0) / 85.0
    return safe_penalty + shell - 1.5 * persistence - 0.003 * min(hotspot, 80.0)


def summarize(vals: list[float], lower_is_better: bool = False) -> dict[str, Any]:
    xs=[float(v) for v in vals if isinstance(v,(int,float)) and math.isfinite(float(v))]
    if not xs: return {"n":0,"mean":None,"best":None,"worst":None}
    return {"n":len(xs),"mean":sum(xs)/len(xs),"best":min(xs) if lower_is_better else max(xs),"worst":max(xs) if lower_is_better else min(xs)}


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate-json", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=s12.DEFAULT_DATASET)
    ap.add_argument("--split", default="val")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--contact-cutoff-nm", type=float, default=1.0)
    ap.add_argument("--clash-cutoff-nm", type=float, default=0.28)
    args=ap.parse_args()
    samples,_=s10.load_dataset(args.dataset)
    by_id={s.get("sample_id"):s for s in samples if s.get("split")==args.split}
    report=json.loads(args.candidate_json.read_text(encoding="utf-8"))
    scorers={"current_hard_gate":score_current,"persistence_contact":score_persistence,"contact_shell":score_contact_shell}
    sample_rows=[]
    for sample_row in report.get("samples",[]):
        sample=by_id.get(sample_row.get("sample_id"))
        if sample is None:
            continue
        cand=[]
        for c in sample_row.get("candidates",[]):
            cc=dict(c)
            cc.update(candidate_geometry(cc, sample, args.contact_cutoff_nm, args.clash_cutoff_nm))
            cand.append(cc)
        selections={}
        for name, scorer in scorers.items():
            valid=[c for c in cand if c.get("status")=="ok"]
            best=min(valid, key=scorer) if valid else None
            selections[name]=best
        sample_rows.append({"sample_id":sample_row.get("sample_id"),"target_id":sample_row.get("target_id"),"selections":selections,"candidate_count":len(cand)})
    summary={}
    for name in scorers:
        selected=[r["selections"].get(name) for r in sample_rows if r["selections"].get(name)]
        summary[name]={
            "selected_count":len(selected),
            "safe_rate":sum(1 for c in selected if finite(c.get("generated_severe_clash_rate"),1.0)==0)/max(1,len(selected)),
            "mean_contact_f1":summarize([c.get("label_mean_contact_f1") for c in selected]),
            "worst_contact_f1":summarize([c.get("label_worst_contact_f1") for c in selected]),
            "mean_direct_rmsd_A":summarize([c.get("label_mean_direct_rmsd_A") for c in selected], lower_is_better=True),
            "contact_persistence":summarize([c.get("generated_contact_persistence") for c in selected]),
            "generated_contact_count_mean":summarize([c.get("generated_contact_count_mean") for c in selected]),
            "shared_identity_posthoc":summarize([c.get("shared_identity_mean_posthoc") for c in selected]),
        }
    out={"stage":"stage20_selection_sweep_from_pdbs","status":"completed","candidate_json":str(args.candidate_json),"dataset":str(args.dataset),"summary":summary,"rows":sample_rows,"note":"Scoring rules are label-free; label geometry is used only for diagnosis."}
    write_json(args.output,out)
    print(json.dumps({"status":"completed","output":str(args.output),"summary":summary},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
