#!/usr/bin/env python
"""Stage09 interface/clash guidance diagnostic for B1 exact outputs.

The default mode uses exact V_exact contacts as oracle anchors. That is not a
leaderboard result; it tests whether anchor/contact/clash guidance terms are
capable of moving existing B1 state-specific poses toward a physically plausible
interface. If this diagnostic fails, scaling training is not justified.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.strategy01.stage08b_exact_geometry_benchmark import as_path, contact_set, parse_ca  # noqa: E402
from scripts.strategy01.stage08b_full_exact_benchmark import benchmark_generation_summary  # noqa: E402

AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_samples(dataset: Path) -> dict[str, dict[str, Any]]:
    data = torch.load(dataset, map_location="cpu", weights_only=False)
    return {str(s.get("sample_id")): s for s in data["samples"]}


def read_fasta(path: Path) -> str:
    if not path.exists():
        return ""
    lines = [x.strip() for x in path.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip() and not x.startswith(">")]
    return "".join(lines)


def write_ca_pdb(path: Path, coords: torch.Tensor, seq: str, chain: str = "B") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    coords = coords.detach().cpu().float()
    for i, xyz in enumerate(coords, start=1):
        aa = AA3.get(seq[i - 1] if i - 1 < len(seq) else "G", "GLY")
        x, y, z = [float(v) for v in xyz.tolist()]
        lines.append(f"ATOM  {i:5d}  CA  {aa:>3s} {chain}{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00           C")
    lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rodrigues(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec) + 1e-8
    k = rotvec / theta
    kx = torch.tensor(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=rotvec.dtype,
        device=rotvec.device,
    )
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return eye + torch.sin(theta) * kx + (1.0 - torch.cos(theta)) * (kx @ kx)


def rigid_apply(coords: torch.Tensor, rotvec: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    center = coords.mean(dim=0, keepdim=True)
    r = rodrigues(rotvec)
    return (coords - center) @ r.T + center + trans


def select_contact_pairs(target_ca: torch.Tensor, ref_binder_ca: torch.Tensor, cutoff_a: float, max_pairs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cset = contact_set(target_ca, ref_binder_ca, cutoff_a)
    if not cset:
        d = torch.cdist(target_ca.float(), ref_binder_ca.float())
        flat = torch.argsort(d.flatten())[:max_pairs]
        ti = flat // d.shape[1]
        bj = flat % d.shape[1]
        rd = d[ti, bj]
        return ti.long(), bj.long(), rd.float()
    pairs = sorted(cset, key=lambda ij: float(torch.linalg.norm(target_ca[ij[0]] - ref_binder_ca[ij[1]]).item()))[:max_pairs]
    ti = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    bj = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    rd = torch.linalg.norm(target_ca[ti] - ref_binder_ca[bj], dim=-1).float()
    return ti, bj, rd


def metrics(target_ca: torch.Tensor, ref_binder_ca: torch.Tensor, pred_ca: torch.Tensor, cutoff_a: float, clash_cutoff_a: float) -> dict[str, Any]:
    ref = contact_set(target_ca, ref_binder_ca, cutoff_a)
    pred = contact_set(target_ca, pred_ca, cutoff_a)
    inter = len(ref & pred)
    precision = inter / max(1, len(pred)) if pred else 0.0
    recall = inter / max(1, len(ref)) if ref else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    min_dist = float(torch.cdist(target_ca.float(), pred_ca.float()).min().item())
    n = min(pred_ca.shape[0], ref_binder_ca.shape[0])
    direct = float(torch.sqrt(((pred_ca[:n].float() - ref_binder_ca[:n].float()) ** 2).sum(dim=-1).mean()).item()) if n else math.nan
    return {
        "reference_contact_count": len(ref),
        "generated_contact_count": len(pred),
        "contact_precision": precision,
        "contact_recall": recall,
        "contact_f1": f1,
        "direct_binder_ca_rmsd_A": direct,
        "min_target_generated_ca_distance_A": min_dist,
        "severe_clash": min_dist < clash_cutoff_a,
    }


def optimize_pose(
    target_ca: torch.Tensor,
    ref_binder_ca: torch.Tensor,
    gen_ca: torch.Tensor,
    cutoff_a: float,
    max_pairs: int,
    steps: int,
    lr: float,
    clash_cutoff_a: float,
    shell_max_a: float,
    clash_weight: float,
    shell_weight: float,
    tether_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    target_ca = target_ca.float()
    ref_binder_ca = ref_binder_ca.float()
    gen_ca = gen_ca.float()
    ti, bj, ref_d = select_contact_pairs(target_ca, ref_binder_ca, cutoff_a, max_pairs)
    # The generated binder can be shorter than the experimental binder in a few
    # V_exact entries. Keep only anchors addressable by the generated pose;
    # otherwise guidance would silently compare different residue axes.
    valid_anchor = bj < gen_ca.shape[0]
    ti, bj, ref_d = ti[valid_anchor], bj[valid_anchor], ref_d[valid_anchor]
    if bj.numel() == 0:
        common_len = min(gen_ca.shape[0], ref_binder_ca.shape[0])
        if common_len < 1:
            raise ValueError("no_common_binder_residues_for_guidance")
        d = torch.cdist(target_ca.float(), ref_binder_ca[:common_len].float())
        flat = torch.argsort(d.flatten())[:max_pairs]
        ti = (flat // d.shape[1]).long()
        bj = (flat % d.shape[1]).long()
        ref_d = d[ti, bj].float()
    rotvec = torch.zeros(3, dtype=torch.float32, requires_grad=True)
    trans = torch.zeros(3, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([rotvec, trans], lr=lr)
    best_loss = float("inf")
    best_pred = gen_ca.clone()
    loss_trace: list[float] = []
    for step in range(max(1, steps)):
        pred = rigid_apply(gen_ca, rotvec, trans)
        contact_d = torch.linalg.norm(pred[bj] - target_ca[ti], dim=-1)
        anchor_loss = F.smooth_l1_loss(contact_d, ref_d.clamp(min=3.5, max=cutoff_a))
        all_d = torch.cdist(target_ca, pred)
        clash_loss = torch.relu(clash_cutoff_a - all_d).pow(2).mean()
        shell_loss = torch.relu(all_d.min(dim=0).values - shell_max_a).pow(2).mean()
        tether_loss = ((pred - gen_ca) ** 2).mean()
        loss = anchor_loss + clash_weight * clash_loss + shell_weight * shell_loss + tether_weight * tether_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        cur = float(loss.detach().item())
        if cur < best_loss:
            best_loss = cur
            best_pred = pred.detach().clone()
        if step == 0 or step == steps - 1 or (step + 1) % 25 == 0:
            loss_trace.append(cur)
    return best_pred, {"best_loss": best_loss, "loss_trace": loss_trace, "selected_anchor_pairs": int(ti.numel())}


def summarize(vals: list[float], lower_is_better: bool = False) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "best": None, "worst": None}
    return {"n": len(vals), "mean": sum(vals) / len(vals), "best": min(vals) if lower_is_better else max(vals), "worst": max(vals) if lower_is_better else min(vals)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=REPO / "data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt")
    parser.add_argument("--b1-summary", type=Path, default=REPO / "reports/strategy01/probes/stage08b_vexact_sampling_summary.json")
    parser.add_argument("--out-dir", type=Path, default=REPO / "results/strategy01/stage09_b1_oracle_anchor_guided")
    parser.add_argument("--report", type=Path, default=REPO / "reports/strategy01/probes/stage09_b1_interface_guidance_summary.json")
    parser.add_argument("--guided-summary", type=Path, default=REPO / "reports/strategy01/probes/stage09_b1_guided_generation_summary.json")
    parser.add_argument("--contact-cutoff-a", type=float, default=10.0)
    parser.add_argument("--clash-cutoff-a", type=float, default=2.8)
    parser.add_argument("--shell-max-a", type=float, default=12.0)
    parser.add_argument("--max-pairs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--clash-weight", type=float, default=8.0)
    parser.add_argument("--shell-weight", type=float, default=0.15)
    parser.add_argument("--tether-weight", type=float, default=0.001)
    parser.add_argument("--allow-clash-worsening", action="store_true", help="Keep guided pose even if it creates a new severe clash; off by default for safe diagnostic.")
    args = parser.parse_args()

    samples = load_samples(args.dataset)
    b1 = json.loads(args.b1_summary.read_text(encoding="utf-8"))
    rows = b1.get("rows", [])
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    guided_rows: list[dict[str, Any]] = []
    before_f1: list[float] = []
    after_f1: list[float] = []
    before_clash: list[bool] = []
    after_clash: list[bool] = []
    before_rmsd: list[float] = []
    after_rmsd: list[float] = []
    state_debug: list[dict[str, Any]] = []

    for row in rows:
        sample_id = str(row.get("sample_id"))
        sample = samples.get(sample_id)
        if sample is None:
            continue
        src_dir = as_path(row["out_dir"])
        dst_dir = args.out_dir / sample_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        seq = row.get("pred_sequence") or read_fasta(src_dir / "shared_sequence.fasta") or sample.get("shared_binder_sequence", "")
        (dst_dir / "shared_sequence.fasta").write_text(f">stage09_guided|{sample_id}\n{seq}\n", encoding="utf-8")
        ref_fasta = src_dir / "reference_sequence.fasta"
        if ref_fasta.exists():
            shutil.copy2(ref_fasta, dst_dir / "reference_sequence.fasta")
        valid_k = int(sample["state_present_mask"].bool().sum().item())
        target_chain = sample.get("target_chain_id") or "A"
        binder_chain = sample.get("binder_chain_id") or "B"
        for k in range(valid_k):
            gen_path = src_dir / f"state{k:02d}_binder_ca.pdb"
            target_path = as_path(sample["target_state_paths"][k])
            exact_complex = as_path((sample.get("exact_complex_paths") or sample.get("predicted_complex_paths"))[k])
            target_rows = parse_ca(target_path, target_chain)
            if len(target_rows) < 3:
                target_rows = parse_ca(exact_complex, target_chain)
            ref_rows = parse_ca(exact_complex, binder_chain)
            gen_rows = parse_ca(gen_path, "B")
            if len(target_rows) < 3 or len(ref_rows) < 3 or len(gen_rows) < 3:
                continue
            target_ca = torch.stack([x for _, x in target_rows])
            ref_ca = torch.stack([x for _, x in ref_rows])
            gen_ca = torch.stack([x for _, x in gen_rows])
            before = metrics(target_ca, ref_ca, gen_ca, args.contact_cutoff_a, args.clash_cutoff_a)
            guided_ca, opt_meta = optimize_pose(
                target_ca,
                ref_ca,
                gen_ca,
                args.contact_cutoff_a,
                args.max_pairs,
                args.steps,
                args.lr,
                args.clash_cutoff_a,
                args.shell_max_a,
                args.clash_weight,
                args.shell_weight,
                args.tether_weight,
            )
            after = metrics(target_ca, ref_ca, guided_ca, args.contact_cutoff_a, args.clash_cutoff_a)
            accept_guided = True
            reject_reason = None
            if (not args.allow_clash_worsening) and (not before["severe_clash"]) and after["severe_clash"]:
                accept_guided = False
                reject_reason = "would_create_new_severe_clash"
            if after["contact_f1"] < before["contact_f1"]:
                accept_guided = False
                reject_reason = "contact_f1_decreased"
            if not accept_guided:
                guided_ca = gen_ca
                after = before
            opt_meta["accepted_guided_pose"] = bool(accept_guided)
            opt_meta["reject_reason"] = reject_reason
            write_ca_pdb(dst_dir / f"state{k:02d}_binder_ca.pdb", guided_ca, seq, "B")
            before_f1.append(float(before["contact_f1"]))
            after_f1.append(float(after["contact_f1"]))
            before_clash.append(bool(before["severe_clash"]))
            after_clash.append(bool(after["severe_clash"]))
            before_rmsd.append(float(before["direct_binder_ca_rmsd_A"]))
            after_rmsd.append(float(after["direct_binder_ca_rmsd_A"]))
            state_debug.append({"sample_id": sample_id, "state_index": k, "before": before, "after": after, "optimizer": opt_meta})
        guided_rows.append(
            {
                "status": "passed",
                "sample_id": sample_id,
                "split": row.get("split"),
                "target_id": row.get("target_id"),
                "pred_sequence": seq,
                "reference_sequence": row.get("reference_sequence"),
                "sequence_identity_to_reference": row.get("sequence_identity_to_reference"),
                "out_dir": str(dst_dir),
            }
        )
        write_json(args.guided_summary, {"status": "running", "rows": guided_rows})

    guided_summary = {
        "status": "passed",
        "source": "B1_strategy01_stage08b plus oracle exact-anchor rigid guidance",
        "oracle_warning": "Uses V_exact experimental contacts as anchors. This is a diagnostic upper-bound for guidance terms, not a fair leaderboard result.",
        "rows": guided_rows,
    }
    write_json(args.guided_summary, guided_summary)
    guided_exact = benchmark_generation_summary(args.guided_summary, args.dataset, "B1_stage09_oracle_anchor_guided", args.contact_cutoff_a)
    summary = {
        "status": "passed" if guided_rows else "empty",
        "dataset": str(args.dataset),
        "b1_summary": str(args.b1_summary),
        "guided_summary": str(args.guided_summary),
        "out_dir": str(args.out_dir),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "oracle_warning": guided_summary["oracle_warning"],
        "state_count": len(after_f1),
        "before_direct_metrics": {
            "contact_f1": summarize(before_f1),
            "direct_binder_ca_rmsd_A": summarize(before_rmsd, lower_is_better=True),
            "clash_rate": sum(before_clash) / max(1, len(before_clash)),
        },
        "after_guidance_direct_metrics": {
            "contact_f1": summarize(after_f1),
            "direct_binder_ca_rmsd_A": summarize(after_rmsd, lower_is_better=True),
            "clash_rate": sum(after_clash) / max(1, len(after_clash)),
        },
        "delta": {
            "contact_f1_mean_abs": (sum(after_f1) / max(1, len(after_f1))) - (sum(before_f1) / max(1, len(before_f1))),
            "direct_rmsd_mean_A": (sum(after_rmsd) / max(1, len(after_rmsd))) - (sum(before_rmsd) / max(1, len(before_rmsd))),
            "clash_rate_abs": (sum(after_clash) / max(1, len(after_clash))) - (sum(before_clash) / max(1, len(before_clash))),
        },
        "exact_benchmark_after_guidance": {k: v for k, v in guided_exact.items() if k != "rows"},
        "state_debug_sample": state_debug[:20],
    }
    write_json(args.report, summary)
    print(json.dumps({"status": summary["status"], "states": summary["state_count"], "report": str(args.report), "delta": summary["delta"]}, indent=2))


if __name__ == "__main__":
    main()