#!/usr/bin/env python
"""Run Strategy01 state-specific sampling on V_exact tensor samples."""

from __future__ import annotations

import argparse
import json
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

DEFAULT_CKPT = REPO / "ckpts/stage07_sequence_consensus/runs/stage08b_merged_pilot/mini_final_lightning.ckpt"
DEFAULT_AE = REPO / "ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt"
DEFAULT_DATASET = REPO / "data/strategy01/stage08b_vexact_tensor/stage08b_vexact_samples_ae_latents.pt"
DEFAULT_SAMPLING = REPO / "configs/pipeline/model_sampling.yaml"
DEFAULT_OUT = REPO / "results/strategy01/stage08b_vexact_sampling"
DEFAULT_REPORT = REPO / "reports/strategy01/probes/stage08b_vexact_sampling_summary.json"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_samples(path: Path) -> list[dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["samples"]


def select_samples(samples: list[dict[str, Any]], split: str, sample_ids: set[str] | None, max_samples: int) -> list[tuple[int, dict[str, Any]]]:
    selected = []
    for idx, sample in enumerate(samples):
        if sample_ids is not None and sample.get("sample_id") not in sample_ids:
            continue
        if split != "any" and sample.get("split") != split:
            continue
        if int(sample["state_present_mask"].bool().sum().item()) < 2:
            continue
        selected.append((idx, sample))
        if max_samples > 0 and len(selected) >= max_samples:
            break
    if not selected:
        raise RuntimeError("No V_exact samples selected for sampling")
    return selected


def sample_one(
    model: Proteina,
    sample_idx: int,
    sample: dict[str, Any],
    sampling_cfg: Any,
    out_root: Path,
    nsteps: int,
    device: torch.device,
) -> dict[str, Any]:
    batch = s4.collate_samples([sample], device)
    started = time.time()
    x_states, nn_out = s7.state_specific_sample(model, batch, nsteps, sampling_cfg, device)
    elapsed = time.time() - started
    pred_seq = s7.seq_from_logits(nn_out["seq_logits_shared"], batch["mask"])
    ref_seq = sample.get("shared_binder_sequence") or s7.seq_from_tokens(sample["binder_seq_shared"], sample["binder_seq_mask"])
    seq_identity = sum(a == b for a, b in zip(pred_seq, ref_seq)) / max(1, min(len(pred_seq), len(ref_seq)))
    sample_out = out_root / str(sample.get("sample_id", f"sample_{sample_idx}"))
    sample_out.mkdir(parents=True, exist_ok=True)
    (sample_out / "shared_sequence.fasta").write_text(f">strategy01_stage08b|{sample.get('sample_id')}\n{pred_seq}\n", encoding="utf-8")
    (sample_out / "reference_sequence.fasta").write_text(f">reference|{sample.get('sample_id')}\n{ref_seq}\n", encoding="utf-8")
    state_summaries = []
    pred_ca = x_states["bb_ca"][0].detach().cpu()
    label_ca = batch["x_1_states"]["bb_ca"][0].detach().cpu()
    sm = batch["state_mask"][0].detach().cpu()
    valid_k = int(sample["state_present_mask"].bool().sum().item())
    for k in range(valid_k):
        s7.write_ca_pdb(sample_out / f"state{k:02d}_binder_ca.pdb", pred_ca[k, sm[k]], pred_seq)
        state_summaries.append(
            {
                "state_index": k,
                "ca_rmsd_to_exact_label_nm": s7.kabsch_rmsd(pred_ca[k], label_ca[k], sm[k]),
            }
        )
    pairwise = []
    for i in range(valid_k):
        for j in range(i + 1, valid_k):
            common = sm[i] & sm[j]
            pairwise.append(
                {
                    "state_i": i,
                    "state_j": j,
                    "ca_rmsd_between_generated_states_nm": s7.kabsch_rmsd(pred_ca[i], pred_ca[j], common),
                }
            )
    torch.save(
        {
            "sample_id": sample.get("sample_id"),
            "pred_seq": pred_seq,
            "ref_seq": ref_seq,
            "x_states": {k: v.detach().cpu() for k, v in x_states.items()},
        },
        sample_out / "state_specific_outputs.pt",
    )
    return {
        "status": "passed",
        "sample_index": sample_idx,
        "sample_id": sample.get("sample_id"),
        "split": sample.get("split"),
        "target_id": sample.get("target_id"),
        "nsteps": nsteps,
        "valid_states": valid_k,
        "pred_sequence": pred_seq,
        "reference_sequence": ref_seq,
        "sequence_identity_to_reference": seq_identity,
        "state_summaries": state_summaries,
        "generated_state_pairwise": pairwise,
        "out_dir": str(sample_out),
        "elapsed_sec": elapsed,
    }


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
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--nsteps", type=int, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    samples = load_samples(args.dataset)
    selected = select_samples(samples, args.split, set(args.sample_id) if args.sample_id else None, args.max_samples)
    model = Proteina.load_from_checkpoint(str(args.ckpt), strict=False, autoencoder_ckpt_path=str(args.autoencoder_ckpt), map_location="cpu")
    model.to(device)
    model.eval()
    sampling_cfg = OmegaConf.to_container(OmegaConf.load(args.sampling_config).model, resolve=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.time()
    for sample_idx, sample in selected:
        rows.append(sample_one(model, sample_idx, sample, sampling_cfg, args.out_dir, args.nsteps, device))
        write_json(args.report, {"status": "running", "rows": rows, "completed": len(rows), "total": len(selected)})
    summary = {
        "status": "passed",
        "dataset": str(args.dataset),
        "ckpt": str(args.ckpt),
        "out_dir": str(args.out_dir),
        "nsteps": args.nsteps,
        "sample_count": len(rows),
        "elapsed_sec": time.time() - started,
        "mean_sec_per_sample": (time.time() - started) / max(1, len(rows)),
        "rows": rows,
    }
    write_json(args.report, summary)
    print(json.dumps({k: summary[k] for k in ["status", "sample_count", "elapsed_sec", "mean_sec_per_sample", "report"] if k in summary}, indent=2))


if __name__ == "__main__":
    main()
