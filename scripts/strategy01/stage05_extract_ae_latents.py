#!/usr/bin/env python
"""Replace Stage05 geometry-proxy local latents with Complexa AE encoder means.

The script is deliberately separate from the dataset builder because AE encoding
is a checkpoint-dependent post-processing step.  The baseline AE checkpoint is
read only; the output tensor is written as a new Stage05 artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.strategy01.stage05_build_predictor_multistate_dataset import ATOM_ORDER, chain_to_atom37_any  # noqa: E402
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "data" / "strategy01" / "stage05_predictor_multistate" / "stage05_predictor_pilot_samples.pt"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "strategy01" / "stage05_predictor_multistate" / "stage05_predictor_pilot_samples_ae_latents.pt"
DEFAULT_AE_CKPT = REPO_ROOT / "ckpts" / "stage03_multistate_loss" / "complexa_ae_init_readonly_copy.ckpt"


def safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_autoencoder(ckpt_path: Path, device: torch.device) -> AutoEncoder:
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    cfg_ae = ckpt["hyper_parameters"]["cfg_ae"]
    model = AutoEncoder(cfg_ae, store_dir=str(REPO_ROOT / "tmp" / "stage05_ae_extract"))
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        raise RuntimeError(f"AE checkpoint missing parameters: {missing[:20]} (total={len(missing)})")
    if unexpected:
        raise RuntimeError(f"AE checkpoint has unexpected parameters: {unexpected[:20]} (total={len(unexpected)})")
    model.to(device)
    model.eval()
    return model


def build_ae_batch(complex_path: Path, binder_chain_id: str, binder_len: int, device: torch.device) -> dict[str, torch.Tensor]:
    binder = chain_to_atom37_any(complex_path, binder_chain_id, binder_len)
    coords_nm = binder.x.unsqueeze(0).to(device)
    coord_mask = binder.mask.unsqueeze(0).to(device)
    seq = binder.seq.unsqueeze(0).to(device)
    residue_mask = coord_mask[:, :, ATOM_ORDER["CA"]].bool()
    n = binder_len
    batch = {
        "coords_nm": coords_nm,
        "coords": coords_nm * 10.0,
        "coord_mask": coord_mask,
        "residue_type": seq.long(),
        "mask": residue_mask,
        "residue_pdb_idx": torch.arange(1, n + 1, device=device).float().unsqueeze(0),
        "chain_breaks_per_residue": torch.zeros(1, n, device=device),
        "defaults_allowed": True,
    }
    return batch


def extract_latents(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data = safe_torch_load(args.dataset, map_location="cpu")
    samples = data["samples"]
    model = load_autoencoder(args.ae_ckpt, device)
    processed_states = 0
    started = time.time()
    with torch.no_grad():
        for sample_idx, sample in enumerate(samples):
            if args.max_samples > 0 and sample_idx >= args.max_samples:
                break
            predicted_paths = sample.get("predicted_complex_paths") or []
            if not predicted_paths:
                raise RuntimeError(f"Sample {sample.get('sample_id')} has no predicted_complex_paths; Stage05 AE latents require predicted complexes.")
            latents = []
            for state_idx, path_str in enumerate(predicted_paths):
                complex_path = Path(path_str)
                if not complex_path.exists():
                    raise FileNotFoundError(f"Predicted complex missing: {complex_path}")
                binder_chain_id = sample.get("binder_chain_id", "B")
                binder_len = int(sample["binder_seq_shared"].numel())
                batch = build_ae_batch(complex_path, binder_chain_id, binder_len, device)
                enc = model.encode(batch)
                mean = enc["mean"].detach().cpu().squeeze(0).float()
                if mean.shape[-1] != 8:
                    raise RuntimeError(f"Expected AE latent dim 8, got {tuple(mean.shape)} for {complex_path}")
                latents.append(mean)
                processed_states += 1
            sample["x_1_states"]["local_latents"] = torch.stack(latents)
            sample.setdefault("stage05_notes", {})["local_latents_source"] = "complexa_ae_encoder_mean"
    summary = {
        "status": "passed",
        "dataset": str(args.dataset),
        "output": str(args.output),
        "ae_ckpt": str(args.ae_ckpt),
        "device": str(device),
        "num_samples_total": len(samples),
        "processed_states": processed_states,
        "elapsed_sec": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ae-ckpt", type=Path, default=DEFAULT_AE_CKPT)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "reports" / "strategy01" / "probes" / "stage05_ae_latent_extract_summary.json")
    args = parser.parse_args()
    summary = extract_latents(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
