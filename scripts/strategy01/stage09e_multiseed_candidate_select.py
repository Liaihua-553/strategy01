#!/usr/bin/env python
"""Stage09E multi-seed safe-select for Strategy01 exact benchmark.

This stage keeps the Stage09C source-state0 interface-anchor generation path,
but samples multiple complete candidates per V_exact item and selects one whole
candidate using production-safe source-anchor geometry. Exact V_exact contacts
are used only by the downstream benchmark script, never by this selector.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from proteinfoundation.proteina import Proteina  # noqa: E402
import scripts.strategy01.stage04_real_complex_loss_debug as s4  # noqa: E402
import scripts.strategy01.stage07_multistate_sampling_smoke as s7  # noqa: E402
import scripts.strategy01.stage09_guided_state_specific_sampling as s9  # noqa: E402
from scripts.strategy01.stage08b_exact_geometry_benchmark import parse_ca  # noqa: E402

DEFAULT_CKPT = REPO / "ckpts/stage07_sequence_consensus/runs/stage08b_merged_pilot/mini_final_lightning.ckpt"
DEFAULT_AE = REPO / "ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt"
DEFAULT_DATASET = REPO / "data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt"
DEFAULT_SAMPLING = REPO / "configs/pipeline/model_sampling.yaml"
DEFAULT_OUT = REPO / "results/strategy01/stage09e_multiseed_safe_select"
DEFAULT_REPORT = REPO / "reports/strategy01/probes/stage09e_multiseed_safe_select_summary.json"


def as_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s7.tensor_to_json(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def stats(values: list[float], lower_is_better: bool = False) -> dict[str, float | int | None]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "worst": None, "best": None}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "worst": max(vals) if lower_is_better else min(vals),
        "best": min(vals) if lower_is_better else max(vals),
    }


def std(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return math.inf
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))


def load_candidate_ca_nm(out_dir: Path, state_index: int) -> torch.Tensor | None:
    path = out_dir / f"state{state_index:02d}_binder_ca.pdb"
    if not path.exists():
        return None
    rows = parse_ca(path, "B")
    if len(rows) < 3:
        return None
    return torch.stack([xyz for _, xyz in rows]).float() / 10.0


def state_source_anchor_metrics(
    label: s9.StateGuidanceLabels,
    pred_ca_nm: torch.Tensor,
    contact_cutoff_nm: float,
    clash_cutoff_nm: float,
    contact_temperature_nm: float,
) -> dict[str, float | bool]:
    common_len = min(pred_ca_nm.shape[0], label.ref_binder_ca_nm.shape[0])
    if common_len < 1:
        return {
            "status_ok": False,
            "source_anchor_score": 0.0,
            "source_anchor_distance_error_nm": math.inf,
            "generated_source_contact_ratio": 0.0,
            "min_target_binder_distance_nm": math.inf,
            "severe_clash": True,
            "generated_contact_count": 0.0,
            "source_contact_count": 0.0,
        }
    bj = label.anchor_binder_idx
    ti = label.anchor_target_idx
    rd = label.anchor_dist_nm
    keep = bj < common_len
    bj, ti, rd = bj[keep], ti[keep], rd[keep]
    if bj.numel() == 0:
        return {
            "status_ok": False,
            "source_anchor_score": 0.0,
            "source_anchor_distance_error_nm": math.inf,
            "generated_source_contact_ratio": 0.0,
            "min_target_binder_distance_nm": math.inf,
            "severe_clash": True,
            "generated_contact_count": 0.0,
            "source_contact_count": 0.0,
        }
    pred = pred_ca_nm[:common_len].float()
    target = label.target_ca_nm.float().cpu()
    ref = label.ref_binder_ca_nm[:common_len].float().cpu()
    anchor_d = torch.linalg.norm(pred[bj.cpu()] - target[ti.cpu()], dim=-1)
    score = torch.sigmoid((contact_cutoff_nm - anchor_d) / max(contact_temperature_nm, 1e-4)).mean()
    error = (anchor_d - rd.cpu().float()).abs().mean()
    all_pred_d = torch.cdist(target, pred)
    all_ref_d = torch.cdist(target, ref)
    gen_pairs_tensor = torch.nonzero(all_pred_d <= contact_cutoff_nm, as_tuple=False)
    gen_pairs = [(int(i), int(j)) for i, j in gen_pairs_tensor.tolist()]
    gen_count = float(len(gen_pairs))
    ref_count = float((all_ref_d <= contact_cutoff_nm).float().sum().item())
    min_d = float(all_pred_d.min().item()) if all_pred_d.numel() else math.inf
    ratio = gen_count / max(1.0, ref_count)
    return {
        "status_ok": True,
        "source_anchor_score": float(score.item()),
        "source_anchor_distance_error_nm": float(error.item()),
        "generated_source_contact_ratio": ratio,
        "min_target_binder_distance_nm": min_d,
        "severe_clash": bool(min_d < clash_cutoff_nm),
        "generated_contact_count": gen_count,
        "source_contact_count": ref_count,
        "_generated_contact_pairs": gen_pairs,
    }


def aggregate_candidate_metrics(state_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in state_rows if r.get("status_ok")]
    scores = [float(r["source_anchor_score"]) for r in ok_rows]
    errors = [float(r["source_anchor_distance_error_nm"]) for r in ok_rows]
    ratios = [float(r["generated_source_contact_ratio"]) for r in ok_rows]
    min_ds = [float(r["min_target_binder_distance_nm"]) for r in ok_rows]
    severe_count = sum(1 for r in ok_rows if r.get("severe_clash"))
    ratio_penalties = [abs(math.log(max(1e-3, min(1e3, r)))) for r in ratios]
    generated_contact_sets = [set(tuple(pair) for pair in r.get("_generated_contact_pairs", [])) for r in ok_rows]
    if generated_contact_sets:
        union = set().union(*generated_contact_sets)
        inter = set(generated_contact_sets[0])
        for cset in generated_contact_sets[1:]:
            inter &= cset
        generated_persistence = len(inter) / max(1, len(union)) if union else 0.0
    else:
        generated_persistence = 0.0
    clean_state_rows = []
    for row in state_rows:
        clean = dict(row)
        clean.pop("_generated_contact_pairs", None)
        clean_state_rows.append(clean)
    return {
        "valid_state_count": len(ok_rows),
        "severe_clash_count": severe_count,
        "all_states_no_severe_clash": severe_count == 0 and len(ok_rows) > 0,
        "worst_source_anchor_score": min(scores) if scores else 0.0,
        "mean_source_anchor_score": sum(scores) / len(scores) if scores else 0.0,
        "state_consistency_std_source_anchor_score": std(scores),
        "mean_source_anchor_distance_error_nm": sum(errors) / len(errors) if errors else math.inf,
        "worst_source_anchor_distance_error_nm": max(errors) if errors else math.inf,
        "mean_generated_source_contact_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
        "mean_contact_ratio_penalty": sum(ratio_penalties) / len(ratio_penalties) if ratio_penalties else math.inf,
        "source_contact_persistence_generated": generated_persistence,
        "min_target_binder_distance_nm": min(min_ds) if min_ds else math.inf,
        "state_rows": clean_state_rows,
    }


def source_anchor_metrics_for_outdir(sample: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    labels = s9.build_guidance_labels(sample, torch.device("cpu"), args.contact_cutoff_nm, args.max_anchor_pairs, "source_state0")
    valid_k = int(sample["state_present_mask"].bool().sum().item())
    rows = []
    for k in range(valid_k):
        label = labels[k] if k < len(labels) else None
        pred_ca_nm = load_candidate_ca_nm(out_dir, k)
        if label is None or pred_ca_nm is None:
            rows.append({"state_index": k, "status_ok": False, "reason": "missing_label_or_prediction"})
            continue
        row = state_source_anchor_metrics(label, pred_ca_nm, args.contact_cutoff_nm, args.clash_cutoff_nm, args.contact_temperature_nm)
        row["state_index"] = k
        rows.append(row)
    return aggregate_candidate_metrics(rows)


def selector_key(metrics: dict[str, Any], mode: str = "balanced_safe") -> tuple[Any, ...]:
    ratio = float(metrics.get("mean_generated_source_contact_ratio") or 0.0)
    ratio_ok = 0.6 <= ratio <= 1.6
    severe = int(metrics.get("severe_clash_count", 999))
    anchor = float(metrics.get("worst_source_anchor_score") or 0.0)
    persistence = float(metrics.get("source_contact_persistence_generated") or 0.0)
    ratio_penalty = float(metrics.get("mean_contact_ratio_penalty") or math.inf)
    consistency = float(metrics.get("state_consistency_std_source_anchor_score") or math.inf)
    dist_error = float(metrics.get("mean_source_anchor_distance_error_nm") or math.inf)
    if mode == "lexicographic":
        return (int(severe > 0), -anchor, int(not ratio_ok), ratio_penalty, consistency, dist_error)
    if mode == "persistence_first":
        return (int(severe > 0), -persistence, -anchor, int(not ratio_ok), ratio_penalty, consistency, dist_error)
    if mode == "balanced_safe":
        # Empirical Stage09E smoke analysis showed that strict no-clash-first
        # selection preserved fewer cross-state contacts. This score still uses
        # only source-anchor/generated geometry, but allows one clashing state
        # when it preserves persistent interface anchors substantially better.
        score = 1.5 * persistence + anchor - 0.10 * ratio_penalty - 0.15 * severe
        return (int(severe > 1), -score, int(not ratio_ok), ratio_penalty, consistency, dist_error)
    raise ValueError(f"Unknown selector_mode={mode}")


def translate_clashing_pose(
    label: s9.StateGuidanceLabels,
    pred_ca_nm: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    current = pred_ca_nm.clone().float()
    original = state_source_anchor_metrics(label, current, args.contact_cutoff_nm, args.clash_cutoff_nm, args.contact_temperature_nm)
    accepted_steps = 0
    step_logs = []
    for step in range(max(0, args.relief_steps)):
        d = torch.cdist(label.target_ca_nm.float().cpu(), current.float())
        clashing = torch.nonzero(d < args.clash_cutoff_nm, as_tuple=False)
        if clashing.numel() == 0:
            break
        ti = clashing[:, 0]
        bj = clashing[:, 1]
        vec = current[bj] - label.target_ca_nm.cpu()[ti]
        norm = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(1e-6)
        direction = (vec / norm).mean(dim=0)
        direction_norm = torch.linalg.norm(direction).clamp_min(1e-6)
        direction = direction / direction_norm
        min_d = float(d.min().item())
        delta_nm = min(args.relief_max_translation_nm, max(0.0, args.clash_cutoff_nm - min_d + 0.01))
        proposal = current + direction * delta_nm
        proposed = state_source_anchor_metrics(label, proposal, args.contact_cutoff_nm, args.clash_cutoff_nm, args.contact_temperature_nm)
        score_ok = float(proposed["source_anchor_score"]) >= float(original["source_anchor_score"]) - args.relief_anchor_score_tolerance
        ratio = float(proposed["generated_source_contact_ratio"])
        ratio_ok = args.relief_contact_ratio_min <= ratio <= args.relief_contact_ratio_max
        min_improved = float(proposed["min_target_binder_distance_nm"]) > float(original["min_target_binder_distance_nm"])
        if score_ok and ratio_ok and min_improved:
            current = proposal
            original = proposed
            accepted_steps += 1
            step_logs.append({"step": step, "accepted": True, "delta_nm": delta_nm, "metrics": proposed})
        else:
            step_logs.append({"step": step, "accepted": False, "delta_nm": delta_nm, "metrics": proposed, "score_ok": score_ok, "ratio_ok": ratio_ok, "min_improved": min_improved})
            break
    return current, {"accepted_steps": accepted_steps, "step_logs": step_logs}


def maybe_apply_rigid_relief(sample: dict[str, Any], row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out_dir = as_path(row["out_dir"])
    before = source_anchor_metrics_for_outdir(sample, out_dir, args)
    if before.get("severe_clash_count", 0) <= 0 or not args.enable_rigid_relief:
        return {"applied": False, "accepted": False, "before": before, "after": before, "reason": "no_severe_clash_or_disabled"}
    pt_path = out_dir / "state_specific_outputs.pt"
    if not pt_path.exists():
        return {"applied": True, "accepted": False, "before": before, "after": before, "reason": "missing_state_specific_outputs"}
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    x_states = payload["x_states"]
    bb_raw = x_states["bb_ca"].clone().float()
    has_batch_dim = bb_raw.dim() == 4
    bb = bb_raw[0].clone() if has_batch_dim else bb_raw.clone()
    labels = s9.build_guidance_labels(sample, torch.device("cpu"), args.contact_cutoff_nm, args.max_anchor_pairs, "source_state0")
    valid_k = int(sample["state_present_mask"].bool().sum().item())
    state_logs = []
    any_step = False
    for k in range(valid_k):
        state_before = before["state_rows"][k]
        if not state_before.get("severe_clash"):
            continue
        label = labels[k] if k < len(labels) else None
        if label is None:
            continue
        state_mask = sample["state_mask"][k].bool()
        pred_state = bb[k, state_mask]
        relieved, relief_log = translate_clashing_pose(label, pred_state, args)
        if relief_log["accepted_steps"] > 0:
            bb[k, state_mask] = relieved
            any_step = True
        relief_log["state_index"] = k
        state_logs.append(relief_log)
    if not any_step:
        return {"applied": True, "accepted": False, "before": before, "after": before, "state_logs": state_logs, "reason": "no_step_accepted"}
    backup_dir = out_dir / "pre_relief_backup"
    backup_dir.mkdir(exist_ok=True)
    for k in range(valid_k):
        pdb = out_dir / f"state{k:02d}_binder_ca.pdb"
        if pdb.exists() and not (backup_dir / pdb.name).exists():
            shutil.copy2(pdb, backup_dir / pdb.name)
    pred_seq = row.get("pred_sequence") or payload.get("pred_seq") or sample.get("shared_binder_sequence", "")
    for k in range(valid_k):
        mask = sample["state_mask"][k].bool()
        s7.write_ca_pdb(out_dir / f"state{k:02d}_binder_ca.pdb", bb[k, mask], pred_seq)
    if has_batch_dim:
        x_states["bb_ca"][0] = bb
    else:
        x_states["bb_ca"] = bb
    payload["x_states"] = x_states
    payload["stage09e_rigid_relief"] = {"state_logs": state_logs}
    torch.save(payload, pt_path)
    after = source_anchor_metrics_for_outdir(sample, out_dir, args)
    score_drop = float(before["worst_source_anchor_score"]) - float(after["worst_source_anchor_score"])
    ratio = float(after.get("mean_generated_source_contact_ratio") or 0.0)
    accepted = (
        int(after.get("severe_clash_count", 999)) <= int(before.get("severe_clash_count", 999))
        and score_drop <= args.relief_anchor_score_tolerance
        and args.relief_contact_ratio_min <= ratio <= args.relief_contact_ratio_max
    )
    if not accepted:
        for k in range(valid_k):
            backup = backup_dir / f"state{k:02d}_binder_ca.pdb"
            if backup.exists():
                shutil.copy2(backup, out_dir / backup.name)
        after = source_anchor_metrics_for_outdir(sample, out_dir, args)
    return {"applied": True, "accepted": accepted, "before": before, "after": after, "state_logs": state_logs}


def copy_selected(row: dict[str, Any], selected_root: Path, sample: dict[str, Any]) -> dict[str, Any]:
    src = as_path(row["out_dir"])
    dst = selected_root / str(sample.get("sample_id"))
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    selected = copy.deepcopy(row)
    selected["out_dir"] = str(dst)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--autoencoder-ckpt", type=Path, default=DEFAULT_AE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sampling-config", type=Path, default=DEFAULT_SAMPLING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--selector-mode", choices=["balanced_safe", "lexicographic", "persistence_first"], default="balanced_safe")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seed-stride", type=int, default=997)
    parser.add_argument("--nsteps", type=int, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--guidance-start-fraction", type=float, default=0.85)
    parser.add_argument("--guidance-inner-steps", type=int, default=10)
    parser.add_argument("--guidance-lr", type=float, default=0.20)
    parser.add_argument("--contact-cutoff-nm", type=float, default=1.0)
    parser.add_argument("--clash-cutoff-nm", type=float, default=0.28)
    parser.add_argument("--shell-max-nm", type=float, default=1.2)
    parser.add_argument("--max-anchor-pairs", type=int, default=32)
    parser.add_argument("--clash-weight", type=float, default=80.0)
    parser.add_argument("--clash-loss-mode", choices=["mean", "active"], default="mean")
    parser.add_argument("--clash-max-weight", type=float, default=4.0)
    parser.add_argument("--shell-weight", type=float, default=0.10)
    parser.add_argument("--tether-weight", type=float, default=0.002)
    parser.add_argument("--overcontact-weight", type=float, default=0.02)
    parser.add_argument("--contact-temperature-nm", type=float, default=0.06)
    parser.add_argument("--hard-min-dist-nm", type=float, default=0.28)
    parser.add_argument("--min-dist-improvement-nm", type=float, default=0.005)
    parser.add_argument("--f1-drop-tolerance", type=float, default=0.02)
    parser.add_argument("--anchor-loss-tolerance", type=float, default=0.25)
    parser.add_argument("--f1-gain-for-anchor-worsen", type=float, default=0.02)
    parser.add_argument("--max-guidance-displacement-nm", type=float, default=0.25)
    parser.add_argument("--enable-ca-feature", action="store_true", default=False)
    parser.add_argument("--safe-accept", action="store_true", default=True)
    parser.add_argument("--unsafe-no-safe-accept", dest="safe_accept", action="store_false")
    parser.add_argument("--enable-rigid-relief", action="store_true", default=True)
    parser.add_argument("--disable-rigid-relief", dest="enable_rigid_relief", action="store_false")
    parser.add_argument("--relief-max-translation-nm", type=float, default=0.04)
    parser.add_argument("--relief-steps", type=int, default=8)
    parser.add_argument("--relief-anchor-score-tolerance", type=float, default=0.01)
    parser.add_argument("--relief-contact-ratio-min", type=float, default=0.6)
    parser.add_argument("--relief-contact-ratio-max", type=float, default=1.6)
    args = parser.parse_args()

    device = torch.device(args.device)
    samples = s9.load_samples(as_path(args.dataset))
    selected = s9.select_samples(samples, args.split, set(args.sample_id) if args.sample_id else None, args.max_samples)
    model = Proteina.load_from_checkpoint(str(as_path(args.ckpt)), strict=False, autoencoder_ckpt_path=str(as_path(args.autoencoder_ckpt)), map_location="cpu")
    model.to(device)
    model.eval()
    sampling_cfg = OmegaConf.to_container(OmegaConf.load(as_path(args.sampling_config)).model, resolve=True)
    args.out_dir = as_path(args.out_dir)
    args.report = as_path(args.report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_root = args.out_dir / "selected"
    selected_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    started = time.time()
    for sample_rank, (sample_idx, sample) in enumerate(selected):
        candidates = []
        for cand_i in range(args.num_candidates):
            cand_seed = int(args.seed + sample_rank * args.seed_stride + cand_i)
            torch.manual_seed(cand_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cand_seed)
            cand_root = args.out_dir / "candidates" / str(sample.get("sample_id")) / f"candidate_{cand_i:02d}_seed{cand_seed}"
            cand_args = copy.copy(args)
            cand_args.anchor_source = "source_state0"
            row = s9.sample_one(model, sample_idx, sample, sampling_cfg, cand_root, args.nsteps, device, cand_args)
            before_relief_metrics = source_anchor_metrics_for_outdir(sample, as_path(row["out_dir"]), args)
            relief = maybe_apply_rigid_relief(sample, row, args)
            after_metrics = relief.get("after", before_relief_metrics)
            cand_record = {
                "candidate_index": cand_i,
                "seed": cand_seed,
                "row": row,
                "source_anchor_metrics_before_relief": before_relief_metrics,
                "source_anchor_metrics": after_metrics,
                "rigid_relief": relief,
                "selector_key": list(selector_key(after_metrics, args.selector_mode)),
            }
            candidates.append(cand_record)
            write_json(args.report, {"status": "running", "completed_samples": len(rows), "current_sample": sample.get("sample_id"), "current_candidate": cand_i, "rows": rows})
        candidates_sorted = sorted(candidates, key=lambda c: selector_key(c["source_anchor_metrics"], args.selector_mode))
        chosen = candidates_sorted[0]
        selected_row = copy_selected(chosen["row"], selected_root, sample)
        selected_row["stage09e_selected_candidate_index"] = chosen["candidate_index"]
        selected_row["stage09e_selected_seed"] = chosen["seed"]
        selected_row["stage09e_selection_metrics"] = chosen["source_anchor_metrics"]
        selected_row["stage09e_selector_key"] = chosen["selector_key"]
        selected_row["stage09e_candidate_table"] = [
            {
                "candidate_index": c["candidate_index"],
                "seed": c["seed"],
                "selector_key": c["selector_key"],
                "source_anchor_metrics": {k: v for k, v in c["source_anchor_metrics"].items() if k != "state_rows"},
                "rigid_relief_accepted": bool((c.get("rigid_relief") or {}).get("accepted")),
            }
            for c in candidates_sorted
        ]
        rows.append(selected_row)
        write_json(args.report, {"status": "running", "completed_samples": len(rows), "total_samples": len(selected), "rows": rows})

    all_candidates = [c for row in rows for c in row.get("stage09e_candidate_table", [])]
    selected_metrics = [row.get("stage09e_selection_metrics", {}) for row in rows]
    summary = {
        "status": "passed",
        "selection_warning": "Production-safe selection uses only source_state0 anchors and generated geometry. Exact contacts are reserved for downstream evaluation.",
        "dataset": str(as_path(args.dataset)),
        "ckpt": str(as_path(args.ckpt)),
        "out_dir": str(args.out_dir),
        "nsteps": args.nsteps,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "sample_count": len(rows),
        "candidate_count": len(all_candidates),
        "elapsed_sec": time.time() - started,
        "mean_sec_per_selected_sample": (time.time() - started) / max(1, len(rows)),
        "selection_aggregate": {
            "selected_worst_source_anchor_score": stats([m.get("worst_source_anchor_score", 0.0) for m in selected_metrics]),
            "selected_severe_clash_count": stats([float(m.get("severe_clash_count", 0.0)) for m in selected_metrics], lower_is_better=True),
            "selected_min_target_binder_distance_nm": stats([m.get("min_target_binder_distance_nm", math.nan) for m in selected_metrics], lower_is_better=False),
            "selected_contact_ratio": stats([m.get("mean_generated_source_contact_ratio", math.nan) for m in selected_metrics]),
            "selected_source_contact_persistence_generated": stats([m.get("source_contact_persistence_generated", math.nan) for m in selected_metrics]),
        },
        "rows": rows,
    }
    write_json(args.report, summary)
    print(json.dumps({"status": "passed", "sample_count": len(rows), "candidate_count": len(all_candidates), "elapsed_sec": summary["elapsed_sec"], "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
