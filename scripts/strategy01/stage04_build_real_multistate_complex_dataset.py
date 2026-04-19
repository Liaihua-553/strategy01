#!/usr/bin/env python
"""Build a Stage04 real-complex-derived multistate debug dataset.

This script uses cached/experimental multichain PDBs that already exist in the
benchmark-valid baseline repository.  It does not call AF2/Boltz online.  Each
state-level complex is written as a two-chain PDB (target=A, binder=B), then the
same parser extracts binder CA, geometry-proxy local latents, interface labels,
and quality-proxy labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from Bio.PDB import PDBParser
from openfold.np import residue_constants

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
BASELINE_ROOT = Path("/data/kfliao/general_model/strategy_repos/proteina-complexa-benchmark-baseline")
OUT_DIR = REPO_ROOT / "data" / "strategy01" / "stage04_real_complex_multistate"

ATOM_ORDER = residue_constants.atom_order
RESTYPE_3TO1 = residue_constants.restype_3to1
RESTYPE_ORDER = residue_constants.restype_order
BACKBONE_OFFSETS = {
    "N": torch.tensor([-0.13, 0.03, 0.0]),
    "CA": torch.zeros(3),
    "C": torch.tensor([0.13, -0.03, 0.0]),
    "O": torch.tensor([0.19, -0.08, 0.0]),
}
PARSER = PDBParser(QUIET=True)

SOURCE_GROUPS = [
    {
        "target_id": "1tnf_homotrimer",
        "files": [
            BASELINE_ROOT / "assets" / "target_data" / "alpha_proteo_targets" / "1tnf_cropped.pdb",
            BASELINE_ROOT / "assets" / "target_data" / "alpha_proteo_targets" / "1tnf_repacked.pdb",
        ],
        "chain_ids": ["A", "B", "C"],
        "binder_offsets": [0, 24, 48, 72],
        "target_offsets": [0, 32],
    },
    {
        "target_id": "5vli_heterodimer",
        "files": [
            BASELINE_ROOT / "assets" / "target_data" / "alpha_proteo_targets" / "5vli_cropped.pdb",
            BASELINE_ROOT / "assets" / "target_data" / "alpha_proteo_targets" / "5vli_cropped_fixed.pdb",
        ],
        "chain_ids": ["A", "B"],
        "binder_offsets": [0, 24, 48, 72],
        "target_offsets": [0, 32],
    },
]


@dataclass
class ChainTensor:
    x: torch.Tensor
    mask: torch.Tensor
    seq: torch.Tensor
    seq_str: str
    source_residue_count: int


def aa3_to_idx(resname: str) -> int:
    aa1 = RESTYPE_3TO1.get(resname.upper(), "A")
    return int(RESTYPE_ORDER.get(aa1, 0))


def aa3_to_1(resname: str) -> str:
    return RESTYPE_3TO1.get(resname.upper(), "A")


def get_chain_residues(path: Path, chain_id: str) -> list[Any]:
    structure = PARSER.get_structure(path.stem, str(path))
    model = next(structure.get_models())
    chain = model[chain_id]
    residues = [r for r in chain.get_residues() if r.id[0] == " "]
    return residues


def chain_to_atom37(path: Path, chain_id: str, start: int, length: int) -> ChainTensor:
    residues = get_chain_residues(path, chain_id)
    if start >= len(residues):
        start = max(0, len(residues) - length)
    selected = residues[start : start + length]
    x = torch.zeros(length, 37, 3, dtype=torch.float32)
    mask = torch.zeros(length, 37, dtype=torch.bool)
    seq = torch.zeros(length, dtype=torch.long)
    seq_chars = []
    for i in range(length):
        if i < len(selected):
            residue = selected[i]
            seq[i] = aa3_to_idx(residue.get_resname())
            seq_chars.append(aa3_to_1(residue.get_resname()))
            for atom in residue.get_atoms():
                name = atom.get_name().strip()
                if name in ATOM_ORDER:
                    x[i, ATOM_ORDER[name]] = torch.tensor(atom.get_coord(), dtype=torch.float32) / 10.0
                    mask[i, ATOM_ORDER[name]] = True
            ca_idx = ATOM_ORDER["CA"]
            ca = x[i, ca_idx].clone() if mask[i, ca_idx] else torch.tensor([i * 0.38, 0.0, 0.0])
        else:
            seq_chars.append("A")
            ca = torch.tensor([i * 0.38, 0.0, 0.0])
        for atom_name, offset in BACKBONE_OFFSETS.items():
            atom_idx = ATOM_ORDER[atom_name]
            if not mask[i, atom_idx]:
                x[i, atom_idx] = ca + offset
                mask[i, atom_idx] = True
    return ChainTensor(x=x, mask=mask, seq=seq, seq_str="".join(seq_chars), source_residue_count=len(residues))


def geometry_latents(ca: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(ca).all(dim=-1)
    center = ca[mask].mean(dim=0) if mask.any() else torch.zeros(3)
    centered = ca - center
    n = ca.shape[0]
    idx = torch.linspace(0.0, 1.0, n).unsqueeze(-1)
    seq_norm = (seq.float() / 19.0).unsqueeze(-1)
    lat = torch.cat(
        [
            centered,
            torch.sin(idx * math.pi),
            torch.cos(idx * math.pi),
            idx,
            seq_norm,
            torch.linalg.norm(centered, dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    return lat[:, :8].float()


def interface_labels(target: ChainTensor, binder: ChainTensor, contact_cutoff_nm: float = 0.8) -> dict[str, Any]:
    target_ca = target.x[:, ATOM_ORDER["CA"], :]
    binder_ca = binder.x[:, ATOM_ORDER["CA"], :]
    target_mask = target.mask[:, ATOM_ORDER["CA"]]
    binder_mask = binder.mask[:, ATOM_ORDER["CA"]]
    dists = torch.cdist(target_ca, binder_ca)
    pair_mask = target_mask[:, None] & binder_mask[None, :]
    contacts = (dists <= contact_cutoff_nm) & pair_mask
    clash = (dists < 0.28) & pair_mask
    if contacts.any():
        mean_contact_dist = float(dists[contacts].mean().item())
    else:
        mean_contact_dist = float(dists[pair_mask].min().item()) if pair_mask.any() else 9.9
    min_dist = float(dists[pair_mask].min().item()) if pair_mask.any() else 9.9
    contact_count = int(contacts.sum().item())
    severe_clash = bool(clash.any().item())
    ipae_proxy = max(2.0, min(31.0, 2.0 + 12.0 * max(0.0, mean_contact_dist - 0.45) + (5.0 if contact_count < 8 else 0.0) + (12.0 if severe_clash else 0.0)))
    interface_plddt_proxy = max(5.0, min(95.0, 90.0 - (30.0 if contact_count < 4 else 0.0) - (55.0 if severe_clash else 0.0)))
    iptm_proxy = max(0.0, min(0.95, contact_count / 32.0)) * (0.25 if severe_clash else 1.0)
    pdockq2_proxy = max(0.0, min(1.0, (interface_plddt_proxy / 100.0) * iptm_proxy * (1.0 - ipae_proxy / 31.0)))
    quality = torch.tensor(
        [1.0 - ipae_proxy / 31.0, interface_plddt_proxy / 100.0, iptm_proxy, pdockq2_proxy, 0.0 if severe_clash else 1.0],
        dtype=torch.float32,
    )
    return {
        "contact_labels": contacts.float(),
        "distance_labels": dists.float(),
        "label_mask": pair_mask,
        "quality_labels": quality,
        "metrics": {
            "contact_count": contact_count,
            "min_distance_nm": min_dist,
            "mean_contact_distance_nm": mean_contact_dist,
            "severe_clash": severe_clash,
            "iPAE_proxy": ipae_proxy,
            "interface_pLDDT_proxy": interface_plddt_proxy,
            "ipTM_proxy": iptm_proxy,
            "pDockQ2_proxy": pdockq2_proxy,
        },
    }


def write_complex_pdb(path: Path, target: ChainTensor, binder: ChainTensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atom_names = ["N", "CA", "C", "O"]
    lines = []
    serial = 1
    for chain_id, chain, res_offset in [("A", target, 1), ("B", binder, 1)]:
        for i in range(chain.x.shape[0]):
            resname = residue_constants.restype_1to3.get(residue_constants.restypes[int(chain.seq[i].item())], "ALA")
            for atom_name in atom_names:
                atom_idx = ATOM_ORDER[atom_name]
                if not chain.mask[i, atom_idx]:
                    continue
                xyz = chain.x[i, atom_idx] * 10.0
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:^4s} {resname:>3s} {chain_id}{i+res_offset:4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 80.00           {atom_name[0]:>2s}\n"
                )
                serial += 1
        lines.append("TER\n")
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")




def full_chain_contact_count(path: Path, target_chain: str, binder_chain: str, cutoff_nm: float = 0.8) -> int:
    target_n = len(get_chain_residues(path, target_chain))
    binder_n = len(get_chain_residues(path, binder_chain))
    target_full = chain_to_atom37(path, target_chain, 0, target_n)
    binder_full = chain_to_atom37(path, binder_chain, 0, binder_n)
    d = torch.cdist(target_full.x[:, ATOM_ORDER["CA"], :], binder_full.x[:, ATOM_ORDER["CA"], :])
    return int((d <= cutoff_nm).sum().item())

def contact_windows(path: Path, target_chain: str, binder_chain: str, target_len: int, binder_len: int, max_windows: int = 8) -> list[tuple[int, int]]:
    """Choose crop windows centered on real target-binder CA contacts."""
    target_n = len(get_chain_residues(path, target_chain))
    binder_n = len(get_chain_residues(path, binder_chain))
    target_full = chain_to_atom37(path, target_chain, 0, target_n)
    binder_full = chain_to_atom37(path, binder_chain, 0, binder_n)
    target_ca = target_full.x[:, ATOM_ORDER["CA"], :]
    binder_ca = binder_full.x[:, ATOM_ORDER["CA"], :]
    d = torch.cdist(target_ca, binder_ca)
    flat = []
    contact_idx = torch.nonzero(d <= 0.8, as_tuple=False)
    if contact_idx.numel() == 0:
        # Fall back to nearest pairs, but the resulting metrics will reveal the weak interface.
        nearest = torch.argsort(d.reshape(-1))[: max_windows * 4]
        contact_idx = torch.stack([nearest // binder_n, nearest % binder_n], dim=1)
    for ti, bi in contact_idx.tolist():
        target_start = max(0, min(ti - target_len // 2, max(0, target_n - target_len)))
        binder_start = max(0, min(bi - binder_len // 2, max(0, binder_n - binder_len)))
        flat.append((target_start, binder_start, float(d[ti, bi].item())))
    flat.sort(key=lambda x: x[2])
    windows = []
    seen = set()
    for target_start, binder_start, _ in flat:
        key = (target_start, binder_start)
        if key in seen:
            continue
        seen.add(key)
        windows.append(key)
        if len(windows) >= max_windows:
            break
    return windows

def build_samples(num_samples: int, train_count: int, val_count: int, target_len: int, binder_len: int, kmax: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    complexes_dir = OUT_DIR / "complexes"
    for group in SOURCE_GROUPS:
        files = [p for p in group["files"] if p.exists() and p.stat().st_size > 0]
        for binder_chain in group["chain_ids"]:
            state_specs = []
            for path in files:
                for state_chain in group["chain_ids"]:
                    if state_chain == binder_chain:
                        continue
                    try:
                        get_chain_residues(path, state_chain)
                        get_chain_residues(path, binder_chain)
                    except Exception:
                        continue
                    state_specs.append((path, state_chain))
            filtered_specs = []
            for state_file, state_chain in state_specs:
                try:
                    if full_chain_contact_count(state_file, state_chain, binder_chain) >= 4:
                        filtered_specs.append((state_file, state_chain))
                except Exception:
                    continue
            state_specs = filtered_specs
            if len(state_specs) < 2:
                continue
            state_specs = state_specs[:kmax]
            first_file, first_target_chain = state_specs[0]
            try:
                crop_windows = contact_windows(first_file, first_target_chain, binder_chain, target_len, binder_len, max_windows=12)
            except Exception:
                crop_windows = [(to, bo) for bo in group["binder_offsets"] for to in group["target_offsets"]]
            for target_offset, binder_offset in crop_windows:
                if len(samples) >= num_samples:
                    break
                try:
                    binder = chain_to_atom37(first_file, binder_chain, binder_offset, binder_len)
                except Exception:
                    continue
                if len(binder.seq_str.replace("A", "")) == 0:
                    continue
                x_targets=[]; target_masks=[]; target_seqs=[]; hotspots=[]
                bb_states=[]; lat_states=[]; contacts=[]; distances=[]; label_masks=[]; quality=[]; state_metrics=[]; complex_paths=[]
                ok=True
                for state_idx, (state_file, state_chain) in enumerate(state_specs):
                    try:
                        target = chain_to_atom37(state_file, state_chain, target_offset, target_len)
                        # Use binder coordinates from the same source file when available, preserving a real complex frame.
                        binder_state = chain_to_atom37(state_file, binder_chain, binder_offset, binder_len)
                    except Exception as exc:
                        ok=False
                        break
                    labels = interface_labels(target, binder_state)
                    complex_path = complexes_dir / f"stage04_{len(samples):03d}_state{state_idx}_{group['target_id']}_{state_chain}_{binder_chain}.pdb"
                    write_complex_pdb(complex_path, target, binder_state)
                    x_targets.append(target.x); target_masks.append(target.mask); target_seqs.append(target.seq)
                    hotspot = labels["contact_labels"].bool().any(dim=1)
                    hotspots.append(hotspot)
                    bb_states.append(binder_state.x[:, ATOM_ORDER["CA"], :])
                    lat_states.append(geometry_latents(binder_state.x[:, ATOM_ORDER["CA"], :], binder_state.seq))
                    contacts.append(labels["contact_labels"]); distances.append(labels["distance_labels"]); label_masks.append(labels["label_mask"])
                    quality.append(labels["quality_labels"]); state_metrics.append(labels["metrics"]); complex_paths.append(str(complex_path))
                if not ok:
                    continue
                split = "train" if len([s for s in samples if s["split"] == "train"]) < train_count else "val"
                sample = {
                        "sample_id": f"stage04_{len(samples):03d}_{group['target_id']}_{binder_chain}_bo{binder_offset}_to{target_offset}",
                        "split": split,
                        "target_id": group["target_id"],
                        "source_kind": "experimental_complex_derived_debug",
                        "predictor": "experimental_chain_extractor_geometry_proxy",
                        "shared_binder_sequence": binder.seq_str,
                        "binder_chain_id": binder_chain,
                        "target_state_paths": [str(p) for p, _ in state_specs],
                        "target_state_chain_ids": [c for _, c in state_specs],
                        "predicted_complex_paths": complex_paths,
                        "x_target_states": torch.stack(x_targets),
                        "target_mask_states": torch.stack(target_masks),
                        "seq_target_states": torch.stack(target_seqs),
                        "target_hotspot_mask_states": torch.stack(hotspots),
                        "binder_seq_shared": binder.seq,
                        "binder_seq_mask": torch.ones(binder_len, dtype=torch.bool),
                        "x_1_states": {"bb_ca": torch.stack(bb_states), "local_latents": torch.stack(lat_states)},
                        "state_mask": torch.ones(len(bb_states), binder_len, dtype=torch.bool),
                        "state_present_mask": torch.ones(len(bb_states), dtype=torch.bool),
                        "target_state_weights": torch.full((len(bb_states),), 1.0 / len(bb_states), dtype=torch.float32),
                        "target_state_roles": torch.ones(len(bb_states), dtype=torch.long),
                        "interface_contact_labels": torch.stack(contacts),
                        "interface_distance_labels": torch.stack(distances),
                        "interface_label_mask": torch.stack(label_masks),
                        "interface_quality_labels": torch.stack(quality),
                        "interface_quality_mask": torch.ones(len(bb_states), 5, dtype=torch.bool),
                        "state_metrics": state_metrics,
                }
                samples.append(sample)
            if len(samples) >= num_samples:
                break
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break
    if len(samples) < train_count + val_count:
        raise RuntimeError(f"Only built {len(samples)} samples; need {train_count + val_count}")
    # Normalize split counts in case generation order exceeded train_count.
    for idx, sample in enumerate(samples):
        sample["split"] = "train" if idx < train_count else "val"
    manifest_samples = []
    for sample in samples:
        manifest_samples.append({
            "sample_id": sample["sample_id"],
            "split": sample["split"],
            "target_id": sample["target_id"],
            "source_kind": sample["source_kind"],
            "predictor": sample["predictor"],
            "K": int(sample["state_present_mask"].numel()),
            "binder_length": int(sample["binder_seq_shared"].numel()),
            "target_length": int(sample["x_target_states"].shape[1]),
            "target_state_paths": sample["target_state_paths"],
            "target_state_chain_ids": sample["target_state_chain_ids"],
            "predicted_complex_paths": sample["predicted_complex_paths"],
            "state_metrics": sample["state_metrics"],
        })
    manifest = {
        "stage": "stage04_real_complex_multistate_debug",
        "note": "Experimental-complex-derived debug set from cached multichain PDBs. Geometry-proxy confidence labels are used because AF2/Boltz weights are not configured in the current environment.",
        "num_samples": len(samples),
        "train_count": train_count,
        "val_count": len(samples) - train_count,
        "target_len": target_len,
        "binder_len": binder_len,
        "kmax": kmax,
        "samples": manifest_samples,
    }
    return samples, manifest



def build_samples(num_samples: int, train_count: int, val_count: int, target_len: int, binder_len: int, kmax: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build samples where each state keeps the same target-chain identity across source files.

    This avoids mixing different oligomer neighbors into one required-bind sample;
    each sample supervises one shared binder against one target-chain role across
    multiple experimentally derived states/files.
    """
    samples: list[dict[str, Any]] = []
    complexes_dir = OUT_DIR / "complexes"
    for group in SOURCE_GROUPS:
        files = [p for p in group["files"] if p.exists() and p.stat().st_size > 0]
        for binder_chain in group["chain_ids"]:
            for target_chain_for_states in group["chain_ids"]:
                if target_chain_for_states == binder_chain:
                    continue
                state_specs = []
                for state_file in files:
                    try:
                        get_chain_residues(state_file, target_chain_for_states)
                        get_chain_residues(state_file, binder_chain)
                        if full_chain_contact_count(state_file, target_chain_for_states, binder_chain) >= 4:
                            state_specs.append((state_file, target_chain_for_states))
                    except Exception:
                        continue
                if len(state_specs) < 2:
                    continue
                state_specs = state_specs[:kmax]
                first_file, first_target_chain = state_specs[0]
                try:
                    crop_windows = contact_windows(first_file, first_target_chain, binder_chain, target_len, binder_len, max_windows=16)
                except Exception:
                    continue
                for target_offset, binder_offset in crop_windows:
                    if len(samples) >= num_samples:
                        break
                    try:
                        binder = chain_to_atom37(first_file, binder_chain, binder_offset, binder_len)
                    except Exception:
                        continue
                    x_targets=[]; target_masks=[]; target_seqs=[]; hotspots=[]
                    bb_states=[]; lat_states=[]; contacts=[]; distances=[]; label_masks=[]; quality=[]; state_metrics=[]; complex_paths=[]
                    ok=True
                    for state_idx, (state_file, state_chain) in enumerate(state_specs):
                        try:
                            target = chain_to_atom37(state_file, state_chain, target_offset, target_len)
                            binder_state = chain_to_atom37(state_file, binder_chain, binder_offset, binder_len)
                        except Exception:
                            ok=False
                            break
                        labels = interface_labels(target, binder_state)
                        if labels["metrics"]["contact_count"] < 4:
                            ok=False
                            break
                        complex_path = complexes_dir / f"stage04_{len(samples):03d}_state{state_idx}_{group['target_id']}_{state_chain}_{binder_chain}.pdb"
                        write_complex_pdb(complex_path, target, binder_state)
                        x_targets.append(target.x); target_masks.append(target.mask); target_seqs.append(target.seq)
                        hotspots.append(labels["contact_labels"].bool().any(dim=1))
                        bb_states.append(binder_state.x[:, ATOM_ORDER["CA"], :])
                        lat_states.append(geometry_latents(binder_state.x[:, ATOM_ORDER["CA"], :], binder_state.seq))
                        contacts.append(labels["contact_labels"]); distances.append(labels["distance_labels"]); label_masks.append(labels["label_mask"])
                        quality.append(labels["quality_labels"]); state_metrics.append(labels["metrics"]); complex_paths.append(str(complex_path))
                    if not ok:
                        continue
                    sample = {
                        "sample_id": f"stage04_{len(samples):03d}_{group['target_id']}_{target_chain_for_states}_{binder_chain}_bo{binder_offset}_to{target_offset}",
                        "split": "train" if len(samples) < train_count else "val",
                        "target_id": group["target_id"],
                        "source_kind": "experimental_complex_derived_debug",
                        "predictor": "experimental_chain_extractor_geometry_proxy",
                        "shared_binder_sequence": binder.seq_str,
                        "binder_chain_id": binder_chain,
                        "target_state_paths": [str(p) for p, _ in state_specs],
                        "target_state_chain_ids": [c for _, c in state_specs],
                        "predicted_complex_paths": complex_paths,
                        "x_target_states": torch.stack(x_targets),
                        "target_mask_states": torch.stack(target_masks),
                        "seq_target_states": torch.stack(target_seqs),
                        "target_hotspot_mask_states": torch.stack(hotspots),
                        "binder_seq_shared": binder.seq,
                        "binder_seq_mask": torch.ones(binder_len, dtype=torch.bool),
                        "x_1_states": {"bb_ca": torch.stack(bb_states), "local_latents": torch.stack(lat_states)},
                        "state_mask": torch.ones(len(bb_states), binder_len, dtype=torch.bool),
                        "state_present_mask": torch.ones(len(bb_states), dtype=torch.bool),
                        "target_state_weights": torch.full((len(bb_states),), 1.0 / len(bb_states), dtype=torch.float32),
                        "target_state_roles": torch.ones(len(bb_states), dtype=torch.long),
                        "interface_contact_labels": torch.stack(contacts),
                        "interface_distance_labels": torch.stack(distances),
                        "interface_label_mask": torch.stack(label_masks),
                        "interface_quality_labels": torch.stack(quality),
                        "interface_quality_mask": torch.ones(len(bb_states), 5, dtype=torch.bool),
                        "state_metrics": state_metrics,
                    }
                    samples.append(sample)
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break
    if len(samples) < train_count + val_count:
        raise RuntimeError(f"Only built {len(samples)} samples; need {train_count + val_count}")
    for idx, sample in enumerate(samples):
        sample["split"] = "train" if idx < train_count else "val"
    manifest_samples = []
    for sample in samples:
        manifest_samples.append({
            "sample_id": sample["sample_id"],
            "split": sample["split"],
            "target_id": sample["target_id"],
            "source_kind": sample["source_kind"],
            "predictor": sample["predictor"],
            "K": int(sample["state_present_mask"].numel()),
            "binder_length": int(sample["binder_seq_shared"].numel()),
            "target_length": int(sample["x_target_states"].shape[1]),
            "target_state_paths": sample["target_state_paths"],
            "target_state_chain_ids": sample["target_state_chain_ids"],
            "predicted_complex_paths": sample["predicted_complex_paths"],
            "state_metrics": sample["state_metrics"],
        })
    manifest = {
        "stage": "stage04_real_complex_multistate_debug",
        "note": "Experimental-complex-derived debug set from cached multichain PDBs. Geometry-proxy confidence labels are used because AF2/Boltz weights are not configured in the current environment.",
        "num_samples": len(samples),
        "train_count": train_count,
        "val_count": len(samples) - train_count,
        "target_len": target_len,
        "binder_len": binder_len,
        "kmax": kmax,
        "samples": manifest_samples,
    }
    return samples, manifest

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--train-count", type=int, default=12)
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--target-len", type=int, default=48)
    parser.add_argument("--binder-len", type=int, default=24)
    parser.add_argument("--kmax", type=int, default=5)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples, manifest = build_samples(args.num_samples, args.train_count, args.val_count, args.target_len, args.binder_len, args.kmax)
    manifest_path = OUT_DIR / "manifest_stage04_debug.json"
    tensor_path = OUT_DIR / "stage04_debug_samples.pt"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save({"samples": samples, "manifest": manifest}, tensor_path)
    print(json.dumps({"status": "passed", "manifest": str(manifest_path), "tensor": str(tensor_path), "num_samples": len(samples)}, indent=2))


if __name__ == "__main__":
    main()
