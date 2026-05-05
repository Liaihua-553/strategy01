#!/usr/bin/env python
"""Stage09F static-source B0 baseline for exact multistate benchmark.

B0-static-source is a fair static-design baseline for Strategy01:
- Generation and selection see only source state0 target/interface anchors.
- The selected single-state binder pose is rigidly transferred to other target
  states by target CA alignment, then evaluated by the same exact-geometry
  benchmark as B1.
- Future-state exact contacts are never used for B0 selection.

This does not claim to be the native original Complexa CLI baseline. It is a
same-checkpoint/same-refiner ablation that isolates multistate conditioning and
state-specific pose generation from static source-state design.
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
import scripts.strategy01.stage07_multistate_sampling_smoke as s7  # noqa: E402
import scripts.strategy01.stage09_guided_state_specific_sampling as s9  # noqa: E402
import scripts.strategy01.stage09e_multiseed_candidate_select as s9e  # noqa: E402
from scripts.strategy01.stage08b_exact_geometry_benchmark import parse_ca  # noqa: E402

DEFAULT_CKPT = REPO / "ckpts/stage07_sequence_consensus/runs/stage08b_merged_pilot/mini_final_lightning.ckpt"
DEFAULT_AE = REPO / "ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt"
DEFAULT_DATASET = REPO / "data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt"
DEFAULT_SAMPLING = REPO / "configs/pipeline/model_sampling.yaml"
DEFAULT_OUT = REPO / "results/strategy01/stage09f_b0_static_source"
DEFAULT_REPORT = REPO / "reports/strategy01/probes/stage09f_b0_static_source_summary.json"


def as_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s7.tensor_to_json(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def slice_tensor_state(x: torch.Tensor, idx: int) -> torch.Tensor:
    return x[idx : idx + 1].clone()


def make_source_state_sample(sample: dict[str, Any], source_state: int) -> dict[str, Any]:
    out = copy.copy(sample)
    tensor_keys = [
        "x_target_states",
        "target_mask_states",
        "seq_target_states",
        "target_hotspot_mask_states",
        "state_mask",
        "state_present_mask",
        "target_state_weights",
        "target_state_roles",
        "interface_contact_labels",
        "interface_distance_labels",
        "interface_label_mask",
        "interface_quality_labels",
        "interface_quality_mask",
    ]
    for key in tensor_keys:
        if key in sample and isinstance(sample[key], torch.Tensor) and sample[key].shape[0] > source_state:
            out[key] = slice_tensor_state(sample[key], source_state)
    if "x_1_states" in sample:
        out["x_1_states"] = {k: slice_tensor_state(v, source_state) for k, v in sample["x_1_states"].items()}
    for key in ["state_roles", "state_weights"]:
        if key in sample and isinstance(sample[key], list) and len(sample[key]) > source_state:
            out[key] = [sample[key][source_state]]
    for key in ["target_state_paths", "predicted_complex_paths", "exact_complex_paths", "exact_or_hybrid_complex_paths"]:
        if key in sample and isinstance(sample[key], list) and len(sample[key]) > source_state:
            out[key] = [sample[key][source_state]]
    out["state_present_mask"] = torch.ones(1, dtype=torch.bool)
    out["target_state_weights"] = torch.ones(1, dtype=torch.float32)
    out["stage09f_source_state_index"] = source_state
    return out


def kabsch_transform(source: torch.Tensor, target: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
    n = min(source.shape[0], target.shape[0])
    if n < 3:
        raise ValueError("too_few_target_ca_for_alignment")
    x = source[:n].double()
    y = target[:n].double()
    m = moving.double()
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    x0 = x - x_mean
    y0 = y - y_mean
    cov = x0.T @ y0
    u, _, vh = torch.linalg.svd(cov)
    d = torch.sign(torch.det(vh.T @ u.T))
    corr = torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=torch.double))
    rot = vh.T @ corr @ u.T
    return ((m - x_mean) @ rot.T + y_mean).float()


def read_target_ca(sample: dict[str, Any], state_index: int) -> torch.Tensor:
    chain = sample.get("target_chain_id") or "A"
    target_paths = sample.get("target_state_paths") or []
    exact_paths = sample.get("exact_complex_paths") or sample.get("predicted_complex_paths") or []
    rows = []
    if state_index < len(target_paths):
        rows = parse_ca(as_path(target_paths[state_index]), chain)
    if len(rows) < 3 and state_index < len(exact_paths):
        rows = parse_ca(as_path(exact_paths[state_index]), chain)
    if len(rows) < 3:
        raise ValueError(f"too_few_target_ca_state_{state_index}")
    return torch.stack([xyz for _, xyz in rows]).float() / 10.0


def transfer_source_pose_to_all_states(sample: dict[str, Any], selected_out: Path, source_state: int, pred_seq: str) -> dict[str, Any]:
    source_pdb = selected_out / "state00_binder_ca.pdb"
    rows = parse_ca(source_pdb, "B")
    if len(rows) < 3:
        raise ValueError(f"cannot_parse_source_binder_ca:{source_pdb}")
    source_binder_ca = torch.stack([xyz for _, xyz in rows]).float() / 10.0
    source_target_ca = read_target_ca(sample, source_state)
    valid_k = int(sample["state_present_mask"].bool().sum().item())
    transfer_rows = []
    for k in range(valid_k):
        target_ca = read_target_ca(sample, k)
        moved = kabsch_transform(source_target_ca, target_ca, source_binder_ca)
        s7.write_ca_pdb(selected_out / f"state{k:02d}_binder_ca.pdb", moved, pred_seq)
        transfer_rows.append({"state_index": k, "status": "transferred", "source_state": source_state})
    return {"status": "passed", "source_state": source_state, "state_rows": transfer_rows}


def copy_selected(row: dict[str, Any], selected_root: Path, sample_id: str) -> dict[str, Any]:
    src = as_path(row["out_dir"])
    dst = selected_root / sample_id
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    out = copy.deepcopy(row)
    out["out_dir"] = str(dst)
    return out


def sample_b0_one(model: Proteina, sample_idx: int, sample: dict[str, Any], sampling_cfg: Any, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    source_sample = make_source_state_sample(sample, args.source_state_index)
    candidates = []
    for cand_i in range(args.num_candidates):
        seed = int(args.seed + sample_idx * args.seed_stride + cand_i)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cand_root = args.out_dir / "candidates" / str(sample.get("sample_id")) / f"candidate_{cand_i:02d}_seed{seed}"
        cand_args = copy.copy(args)
        cand_args.anchor_source = "source_state0"
        row = s9.sample_one(model, sample_idx, source_sample, sampling_cfg, cand_root, args.nsteps, device, cand_args)
        relief = s9e.maybe_apply_rigid_relief(source_sample, row, args)
        metrics = relief.get("after") or s9e.source_anchor_metrics_for_outdir(source_sample, as_path(row["out_dir"]), args)
        candidates.append({
            "candidate_index": cand_i,
            "seed": seed,
            "row": row,
            "source_anchor_metrics": metrics,
            "rigid_relief": relief,
            "selector_key": list(s9e.selector_key(metrics, args.selector_mode)),
        })
    candidates_sorted = sorted(candidates, key=lambda c: s9e.selector_key(c["source_anchor_metrics"], args.selector_mode))
    chosen = candidates_sorted[0]
    selected = copy_selected(chosen["row"], args.out_dir / "selected", str(sample.get("sample_id")))
    pred_seq = selected.get("pred_sequence") or sample.get("shared_binder_sequence", "")
    transfer = transfer_source_pose_to_all_states(sample, as_path(selected["out_dir"]), args.source_state_index, pred_seq)
    selected.update({
        "status": "passed",
        "sample_index": sample_idx,
        "sample_id": sample.get("sample_id"),
        "split": sample.get("split"),
        "target_id": sample.get("target_id"),
        "valid_states": int(sample["state_present_mask"].bool().sum().item()),
        "stage09f_baseline_type": "B0_static_source_state_same_model",
        "stage09f_source_state_index": args.source_state_index,
        "stage09f_transfer": transfer,
        "stage09f_selected_candidate_index": chosen["candidate_index"],
        "stage09f_selected_seed": chosen["seed"],
        "stage09f_selection_metrics": chosen["source_anchor_metrics"],
        "stage09f_selector_key": chosen["selector_key"],
        "stage09f_candidate_table": [
            {
                "candidate_index": c["candidate_index"],
                "seed": c["seed"],
                "selector_key": c["selector_key"],
                "source_anchor_metrics": {k: v for k, v in c["source_anchor_metrics"].items() if k != "state_rows"},
                "rigid_relief_accepted": bool((c.get("rigid_relief") or {}).get("accepted")),
            }
            for c in candidates_sorted
        ],
    })
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
    parser.add_argument("--source-state-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=321)
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
    parser.add_argument("--safe-accept", action="store_true", default=True)
    parser.add_argument("--unsafe-no-safe-accept", dest="safe_accept", action="store_false")
    parser.add_argument("--enable-ca-feature", action="store_true", default=False)
    parser.add_argument("--enable-rigid-relief", action="store_true", default=True)
    parser.add_argument("--disable-rigid-relief", dest="enable_rigid_relief", action="store_false")
    parser.add_argument("--relief-max-translation-nm", type=float, default=0.04)
    parser.add_argument("--relief-steps", type=int, default=8)
    parser.add_argument("--relief-anchor-score-tolerance", type=float, default=0.01)
    parser.add_argument("--relief-contact-ratio-min", type=float, default=0.6)
    parser.add_argument("--relief-contact-ratio-max", type=float, default=1.6)
    args = parser.parse_args()

    args.out_dir = as_path(args.out_dir)
    args.report = as_path(args.report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    samples = s9.load_samples(as_path(args.dataset))
    selected_samples = s9.select_samples(samples, args.split, set(args.sample_id) if args.sample_id else None, args.max_samples)
    model = Proteina.load_from_checkpoint(str(as_path(args.ckpt)), strict=False, autoencoder_ckpt_path=str(as_path(args.autoencoder_ckpt)), map_location="cpu")
    model.to(device)
    model.eval()
    sampling_cfg = OmegaConf.to_container(OmegaConf.load(as_path(args.sampling_config)).model, resolve=True)
    rows = []
    started = time.time()
    for sample_idx, sample in selected_samples:
        row = sample_b0_one(model, sample_idx, sample, sampling_cfg, args, device)
        rows.append(row)
        write_json(args.report, {"status": "running", "completed": len(rows), "total": len(selected_samples), "rows": rows})
    summary = {
        "status": "passed",
        "baseline_type": "B0_static_source_state_same_model",
        "interpretation": "Generation/selection sees only source state0. Selected source pose is rigidly transferred by target CA alignment to other states for exact evaluation.",
        "dataset": str(as_path(args.dataset)),
        "ckpt": str(as_path(args.ckpt)),
        "out_dir": str(args.out_dir),
        "nsteps": args.nsteps,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "sample_count": len(rows),
        "candidate_count": len(rows) * int(args.num_candidates),
        "elapsed_sec": time.time() - started,
        "mean_sec_per_selected_sample": (time.time() - started) / max(1, len(rows)),
        "rows": rows,
    }
    write_json(args.report, summary)
    print(json.dumps({"status": "passed", "sample_count": len(rows), "candidate_count": summary["candidate_count"], "elapsed_sec": summary["elapsed_sec"], "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()