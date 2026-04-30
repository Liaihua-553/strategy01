#!/usr/bin/env python
"""Build Stage08B constrained-hybrid multistate samples from PINDER parquet.

The Hugging Face PINDER parquet shards include compact coordinate dictionaries
for bound complex, apo receptor/ligand, and predicted receptor/ligand.  This
script constructs Strategy01 samples without free docking:

state0 = experimental bound receptor + experimental binder from complex.
state1/2 = apo/pred receptor, with binder pose transferred by rigidly aligning
bound receptor coordinates to that receptor state.

These are hybrid-silver labels only when persistent anchors and contact-F1 pass.
No sample is promoted to V_exact by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.strategy01.stage04_build_real_multistate_complex_dataset import (  # noqa: E402
    ATOM_ORDER,
    BACKBONE_OFFSETS,
    ChainTensor,
    geometry_latents,
    interface_labels,
)
from openfold.np import residue_constants  # noqa: E402

AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_IDX_TO_1 = {i: aa for i, aa in enumerate(AA_ORDER)}
AA_1_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sha12(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def finite_xyz(x: Any) -> bool:
    try:
        arr = np.asarray(x, dtype=np.float32)
        return arr.shape == (3,) and bool(np.isfinite(arr).all())
    except Exception:
        return False


def structure_to_chain(struct: Any, chain_id: str, max_len: int | None = None) -> ChainTensor | None:
    if struct is None or not isinstance(struct, dict):
        return None
    coords = struct.get("coords")
    atom_names = struct.get("atom_name")
    chain_ids = struct.get("chain_id")
    residue_starts = struct.get("residue_starts")
    restype_index = struct.get("restype_index")
    if coords is None or atom_names is None or chain_ids is None or residue_starts is None or restype_index is None:
        return None
    residue_starts = list(np.asarray(residue_starts, dtype=np.int64))
    if not residue_starts:
        return None
    n_res = len(residue_starts)
    if max_len is not None:
        n_res = min(n_res, max_len)
    x = torch.zeros(n_res, 37, 3, dtype=torch.float32)
    mask = torch.zeros(n_res, 37, dtype=torch.bool)
    seq = torch.zeros(n_res, dtype=torch.long)
    seq_chars: list[str] = []
    coords_list = list(coords)
    atom_names_list = [str(a) for a in atom_names]
    # PINDER stores chain_id per residue, while coords/atom_name are per atom.
    # residue_starts maps residue index to its atom interval.
    chain_ids_list = [str(c) for c in chain_ids]
    restype_arr = np.asarray(restype_index, dtype=np.int64)
    used = 0
    for i in range(n_res):
        start = int(residue_starts[i])
        end = int(residue_starts[i + 1]) if i + 1 < len(residue_starts) else len(coords_list)
        if start >= len(coords_list):
            break
        residue_chain = chain_ids_list[i] if i < len(chain_ids_list) else ""
        if residue_chain != chain_id:
            continue
        aa_idx = int(restype_arr[i]) if i < len(restype_arr) else 0
        aa = AA_IDX_TO_1.get(aa_idx, "A")
        out_i = used
        if out_i >= n_res:
            break
        seq[out_i] = int(AA_1_TO_IDX.get(aa, 0))
        seq_chars.append(aa)
        for j in range(start, min(end, len(coords_list))):
            atom = atom_names_list[j].strip()
            if atom not in ATOM_ORDER:
                continue
            if not finite_xyz(coords_list[j]):
                continue
            x[out_i, ATOM_ORDER[atom]] = torch.tensor(np.asarray(coords_list[j], dtype=np.float32) / 10.0)
            mask[out_i, ATOM_ORDER[atom]] = True
        ca_idx = ATOM_ORDER["CA"]
        if not bool(mask[out_i, ca_idx]):
            # Skip residues without CA. They cannot be used by Complexa labels.
            continue
        ca = x[out_i, ca_idx].clone()
        for atom_name, offset in BACKBONE_OFFSETS.items():
            atom_idx = ATOM_ORDER[atom_name]
            if not bool(mask[out_i, atom_idx]):
                x[out_i, atom_idx] = ca + offset
                mask[out_i, atom_idx] = True
        used += 1
    if used < 3:
        return None
    return ChainTensor(x=x[:used], mask=mask[:used], seq=seq[:used], seq_str="".join(seq_chars[:used]), source_residue_count=used)


def split_complex_chains(complex_struct: dict[str, Any]) -> tuple[ChainTensor | None, ChainTensor | None]:
    return structure_to_chain(complex_struct, "R"), structure_to_chain(complex_struct, "L")


def ca_coords(chain: ChainTensor) -> torch.Tensor:
    return chain.x[:, ATOM_ORDER["CA"]].float()


def kabsch_transform(src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    n = min(src.shape[0], dst.shape[0])
    if n < 3:
        return None
    a = src[:n].double()
    b = dst[:n].double()
    ac = a.mean(dim=0, keepdim=True)
    bc = b.mean(dim=0, keepdim=True)
    aa = a - ac
    bb = b - bc
    h = aa.T @ bb
    u, _, v = torch.linalg.svd(h)
    r = v.T @ u.T
    if torch.det(r) < 0:
        v[-1, :] *= -1
        r = v.T @ u.T
    aligned = aa @ r + bc
    rmsd = torch.sqrt(torch.mean(torch.sum((aligned - b) ** 2, dim=1))).item()
    t = (bc.squeeze(0) - ac.squeeze(0) @ r).float()
    return r.float(), t.float(), float(rmsd)


def transform_chain(chain: ChainTensor, r: torch.Tensor, t: torch.Tensor) -> ChainTensor:
    x = chain.x.clone()
    flat = x.reshape(-1, 3)
    flat = flat @ r + t
    x = flat.reshape_as(x)
    return ChainTensor(x=x, mask=chain.mask.clone(), seq=chain.seq.clone(), seq_str=chain.seq_str, source_residue_count=chain.source_residue_count)


def write_chain_pdb(path: Path, chain: ChainTensor, chain_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    serial = 1
    for i in range(chain.x.shape[0]):
        aa = residue_constants.restypes[int(chain.seq[i].item())] if int(chain.seq[i].item()) < len(residue_constants.restypes) else "A"
        resname = residue_constants.restype_1to3.get(aa, "ALA")
        for atom_name in ["N", "CA", "C", "O"]:
            atom_idx = ATOM_ORDER[atom_name]
            if not bool(chain.mask[i, atom_idx].item()):
                continue
            xyz = chain.x[i, atom_idx] * 10.0
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {resname:>3s} {chain_id}{i+1:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 80.00           {atom_name[0]:>2s}\n"
            )
            serial += 1
    lines.append("TER\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_complex_pdb(path: Path, target: ChainTensor, binder: ChainTensor, target_chain_id: str = "A", binder_chain_id: str = "B") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_t = path.with_suffix(".target.tmp")
    tmp_b = path.with_suffix(".binder.tmp")
    write_chain_pdb(tmp_t, target, target_chain_id)
    write_chain_pdb(tmp_b, binder, binder_chain_id)
    text = tmp_t.read_text(encoding="utf-8") + tmp_b.read_text(encoding="utf-8") + "END\n"
    path.write_text(text, encoding="utf-8")
    tmp_t.unlink(missing_ok=True)
    tmp_b.unlink(missing_ok=True)


def min_ca_distance_nm(target: ChainTensor, binder: ChainTensor) -> float:
    d = torch.cdist(ca_coords(target), ca_coords(binder))
    return float(d.min().item()) if d.numel() else math.inf


def contact_set(labels: dict[str, Any]) -> set[tuple[int, int]]:
    mat = labels["contact_labels"]
    idx = torch.nonzero(mat > 0.5, as_tuple=False)
    return {(int(i), int(j)) for i, j in idx.tolist()}


def f1(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    precision = inter / max(1, len(a))
    recall = inter / max(1, len(b))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def pad_tensor_first_dim(t: torch.Tensor, length: int, value: float | bool | int = 0) -> torch.Tensor:
    if t.shape[0] >= length:
        return t[:length]
    out_shape = (length,) + tuple(t.shape[1:])
    out = torch.full(out_shape, value, dtype=t.dtype)
    out[: t.shape[0]] = t
    return out


def pad_pair_tensor(t: torch.Tensor, target_len: int, binder_len: int, value: float | bool = 0) -> torch.Tensor:
    out = torch.full((target_len, binder_len), value, dtype=t.dtype)
    out[: min(target_len, t.shape[0]), : min(binder_len, t.shape[1])] = t[:target_len, :binder_len]
    return out


def build_row_sample(row: Any, source_name: str, out_root: Path, args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    row_id = str(row.get("id"))
    source = {"row_id": row_id, "pdb_id": str(row.get("pdb_id")), "source_name": source_name}
    bound_target, bound_binder = split_complex_chains(row["complex"])
    if bound_target is None or bound_binder is None:
        return None, {**source, "reject": "missing_bound_chains"}
    if not (args.target_min_len <= bound_target.x.shape[0] <= args.target_max_len):
        return None, {**source, "reject": "target_length"}
    if not (args.binder_min_len <= bound_binder.x.shape[0] <= args.binder_max_len):
        return None, {**source, "reject": "binder_length", "binder_len": int(bound_binder.x.shape[0])}

    state_defs: list[tuple[str, ChainTensor, ChainTensor, float]] = [("bound_exact", bound_target, bound_binder, 0.0)]
    for state_name in ["apo_receptor", "pred_receptor"]:
        state_target = structure_to_chain(row.get(state_name), "R")
        if state_target is None:
            continue
        kt = kabsch_transform(ca_coords(bound_target), ca_coords(state_target))
        if kt is None:
            continue
        r, t, rmsd = kt
        rmsd_a = rmsd * 10.0
        if rmsd_a < args.min_motion_a or rmsd_a > args.max_motion_a:
            # Keep only functionally relevant but not impossible motions.
            continue
        state_binder = transform_chain(bound_binder, r, t)
        state_defs.append((state_name, state_target, state_binder, rmsd))
        if len(state_defs) >= args.max_states:
            break
    if len(state_defs) < args.min_states:
        return None, {**source, "reject": "insufficient_motion_states", "states": len(state_defs)}

    labels = [interface_labels(t, b, contact_cutoff_nm=args.contact_cutoff_nm) for _, t, b, _ in state_defs]
    contacts = [contact_set(x) for x in labels]
    contact_counts = [len(c) for c in contacts]
    if min(contact_counts) < args.min_contacts:
        return None, {**source, "reject": "low_contacts", "contact_counts": contact_counts}
    anchor = set.intersection(*contacts) if contacts else set()
    if len(anchor) < args.min_persistent_anchors:
        return None, {**source, "reject": "low_persistent_anchors", "persistent_anchor_count": len(anchor)}
    source_contacts = contacts[0]
    contact_f1s = [f1(c, source_contacts) for c in contacts]
    if min(contact_f1s[1:] or [1.0]) < args.min_contact_f1:
        return None, {**source, "reject": "low_contact_f1", "contact_f1": contact_f1s}
    min_dists = [min_ca_distance_nm(t, b) for _, t, b, _ in state_defs]
    if min(min_dists) < args.min_ca_distance_nm:
        return None, {**source, "reject": "severe_clash", "min_ca_distance_nm": min(min_dists)}

    max_motion = max(r for _, _, _, r in state_defs)
    sample_hash = sha12(row_id + source_name)
    sample_id = f"pinder_{source_name}_{sample_hash}__k{len(state_defs)}"
    sample_dir = out_root / "pinder_constrained_states" / sample_id
    target_state_paths: list[str] = []
    complex_paths: list[str] = []
    x_target_states: list[torch.Tensor] = []
    target_mask_states: list[torch.Tensor] = []
    seq_target_states: list[torch.Tensor] = []
    target_hotspot_mask_states: list[torch.Tensor] = []
    bb_states: list[torch.Tensor] = []
    lat_states: list[torch.Tensor] = []
    contact_tensors: list[torch.Tensor] = []
    distance_tensors: list[torch.Tensor] = []
    label_masks: list[torch.Tensor] = []
    quality_labels: list[torch.Tensor] = []
    state_metrics: list[dict[str, Any]] = []
    max_target_len = max(target.x.shape[0] for _, target, _, _ in state_defs)
    binder_len = bound_binder.x.shape[0]
    for state_i, (state_name, target, binder, rmsd) in enumerate(state_defs):
        target_path = sample_dir / f"{sample_id}_state{state_i:02d}_{state_name}_target_A.pdb"
        complex_path = sample_dir / f"{sample_id}_state{state_i:02d}_{state_name}_complex_AB.pdb"
        write_chain_pdb(target_path, target, "A")
        write_complex_pdb(complex_path, target, binder, "A", "B")
        target_state_paths.append(str(target_path))
        complex_paths.append(str(complex_path))
        x_target_states.append(pad_tensor_first_dim(target.x, max_target_len, 0.0))
        target_mask_states.append(pad_tensor_first_dim(target.mask, max_target_len, False))
        seq_target_states.append(pad_tensor_first_dim(target.seq, max_target_len, 0))
        hotspot = labels[state_i]["contact_labels"].any(dim=1)
        target_hotspot_mask_states.append(pad_tensor_first_dim(hotspot, max_target_len, False))
        bb_states.append(ca_coords(binder))
        lat_states.append(geometry_latents(ca_coords(binder), binder.seq))
        contact_tensors.append(pad_pair_tensor(labels[state_i]["contact_labels"].float(), max_target_len, binder_len, 0.0))
        distance_tensors.append(pad_pair_tensor(labels[state_i]["distance_labels"].float(), max_target_len, binder_len, 0.0))
        label_masks.append(pad_pair_tensor(labels[state_i]["label_mask"].bool(), max_target_len, binder_len, False))
        # Confidence proxy is geometric only. Predictor confidence is not faked.
        quality_labels.append(
            torch.tensor(
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0 if min_dists[state_i] < args.min_ca_distance_nm else 1.0,
                ],
                dtype=torch.float32,
            )
        )
        state_metrics.append(
            {
                "state_name": state_name,
                "target_motion_rmsd_A": round(rmsd * 10.0, 4),
                "contact_count": int(contact_counts[state_i]),
                "contact_F1_vs_source": round(contact_f1s[state_i], 4),
                "min_ca_distance_A": round(min_dists[state_i] * 10.0, 4),
                "source_label_type": "exact_bound_complex" if state_i == 0 else "constrained_template_reconstruction",
            }
        )

    state_count = len(state_defs)
    state_weights = torch.full((state_count,), 1.0 / state_count, dtype=torch.float32)
    binder_seq = bound_binder.seq
    split_key = str(row.get("cluster_id") or row.get("receptor_uniprot_accession") or row.get("pdb_id") or sample_hash)
    sample = {
        "sample_id": sample_id,
        "target_id": str(row.get("pdb_id")),
        "family_split_key": split_key,
        "split_group": split_key,
        "source_dataset": "pinder_hf_parquet",
        "source_tier": "T_hybrid_silver",
        "predictor": "none_constrained_template",
        "quality_tier": "silver",
        "shared_binder_sequence": bound_binder.seq_str,
        "binder_chain_id": "B",
        "target_chain_id": "A",
        "target_state_chain_ids": ["A"] * state_count,
        "target_state_paths": target_state_paths,
        "predicted_complex_paths": complex_paths,
        "exact_or_hybrid_complex_paths": complex_paths,
        "state_roles": ["required_bind"] * state_count,
        "state_weights": [float(x) for x in state_weights.tolist()],
        "leakage_keys": {
            "pinder_id": row_id,
            "pdb_id": str(row.get("pdb_id")),
            "cluster_id": str(row.get("cluster_id")),
            "receptor_uniprot": str(row.get("receptor_uniprot_accession")),
            "ligand_uniprot": str(row.get("ligand_uniprot_accession")),
            "binder_sequence_hash": sha12(bound_binder.seq_str),
        },
        "x_target_states": torch.stack(x_target_states),
        "target_mask_states": torch.stack(target_mask_states),
        "seq_target_states": torch.stack(seq_target_states),
        "target_hotspot_mask_states": torch.stack(target_hotspot_mask_states),
        "binder_seq_shared": binder_seq,
        "binder_seq_mask": torch.ones_like(binder_seq, dtype=torch.bool),
        "x_1_states": {"bb_ca": torch.stack(bb_states), "local_latents": torch.stack(lat_states)},
        "state_mask": torch.ones(state_count, binder_seq.numel(), dtype=torch.bool),
        "state_present_mask": torch.ones(state_count, dtype=torch.bool),
        "target_state_weights": state_weights,
        "target_state_roles": torch.zeros(state_count, dtype=torch.long),
        "interface_contact_labels": torch.stack(contact_tensors),
        "interface_distance_labels": torch.stack(distance_tensors),
        "interface_label_mask": torch.stack(label_masks),
        "interface_quality_labels": torch.stack(quality_labels),
        "interface_quality_mask": torch.tensor([[False, False, False, False, True]] * state_count, dtype=torch.bool),
        "state_metrics": state_metrics,
        "worst_state_metrics": {
            "worst_contact_count": min(contact_counts),
            "min_contact_F1_vs_source": min(contact_f1s),
            "persistent_anchor_count": len(anchor),
            "max_target_motion_rmsd_A": round(max_motion * 10.0, 4),
            "state_metric_std_motion_A": float(torch.tensor([m["target_motion_rmsd_A"] for m in state_metrics]).std(unbiased=False).item()),
        },
        "stage08b_notes": {
            "label_construction": "bound exact + rigid target-aligned binder transfer for apo/pred states",
            "local_latents_source": "geometry_proxy_pending_ae",
            "contact_cutoff_A": args.contact_cutoff_nm * 10.0,
        },
        "ae_latent_source": None,
    }
    manifest = {
        "sample_id": sample_id,
        "target_id": sample["target_id"],
        "family_split_key": split_key,
        "source_tier": sample["source_tier"],
        "quality_tier": sample["quality_tier"],
        "k_states": state_count,
        "binder_len": int(bound_binder.seq.numel()),
        "target_len": int(bound_target.seq.numel()),
        "persistent_anchor_count": len(anchor),
        "max_target_motion_rmsd_A": round(max_motion * 10.0, 4),
        "min_contact_count": min(contact_counts),
        "min_contact_F1_vs_source": round(min(contact_f1s), 4),
        "row_id": row_id,
        "source_name": source_name,
    }
    return (sample, manifest), {**source, "accepted": True}


def split_samples(samples: list[dict[str, Any]], val_fraction: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        groups[str(s["family_split_key"])].append(s)
    group_keys = sorted(groups)
    rng.shuffle(group_keys)
    n_val_target = max(1, int(round(len(samples) * val_fraction)))
    val_groups: set[str] = set()
    val_count = 0
    for g in group_keys:
        if val_count >= n_val_target:
            break
        val_groups.add(g)
        val_count += len(groups[g])
    for s in samples:
        s["split"] = "val" if str(s["family_split_key"]) in val_groups else "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", action="append", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/strategy01/stage08b_pinder_hybrid")
    ap.add_argument("--summary", type=Path, default=REPO_ROOT / "reports/strategy01/probes/stage08b_pinder_build_summary.json")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--train-count", type=int, default=256)
    ap.add_argument("--val-count", type=int, default=64)
    ap.add_argument("--min-states", type=int, default=2)
    ap.add_argument("--max-states", type=int, default=3)
    ap.add_argument("--target-min-len", type=int, default=30)
    ap.add_argument("--target-max-len", type=int, default=512)
    ap.add_argument("--binder-min-len", type=int, default=6)
    ap.add_argument("--binder-max-len", type=int, default=160)
    ap.add_argument("--min-motion-a", type=float, default=1.0)
    ap.add_argument("--max-motion-a", type=float, default=8.0)
    ap.add_argument("--min-contacts", type=int, default=12)
    ap.add_argument("--min-persistent-anchors", type=int, default=3)
    ap.add_argument("--min-contact-f1", type=float, default=0.4)
    ap.add_argument("--contact-cutoff-nm", type=float, default=0.8)
    ap.add_argument("--min-ca-distance-nm", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    source_rows = 0
    source_counts: Counter[str] = Counter()
    for pq in args.parquet:
        source_name = pq.stem.replace("-00000-of-00001", "").replace("pinder_", "")
        df = pd.read_parquet(pq)
        source_counts[source_name] += int(len(df))
        for _, row in df.iterrows():
            source_rows += 1
            if args.max_rows > 0 and source_rows > args.max_rows:
                break
            try:
                result, info = build_row_sample(row, source_name, args.out_dir, args)
                if result is None:
                    rejects.append(info)
                    continue
                sample, manifest = result
                samples.append(sample)
                manifests.append(manifest)
                if len(samples) >= args.train_count + args.val_count:
                    break
            except Exception as exc:
                rejects.append({"row_id": str(row.get("id")), "source_name": source_name, "reject": "exception", "error": str(exc)[:500]})
        if len(samples) >= args.train_count + args.val_count:
            break
    split_samples(samples, val_fraction=args.val_count / max(1, args.train_count + args.val_count), seed=args.seed)
    train_samples = [s for s in samples if s["split"] == "train"][: args.train_count]
    val_samples = [s for s in samples if s["split"] == "val"][: args.val_count]
    keep_ids = {s["sample_id"] for s in train_samples + val_samples}
    samples = [s for s in samples if s["sample_id"] in keep_ids]
    manifests = [m for m in manifests if m["sample_id"] in keep_ids]
    for m in manifests:
        m["split"] = "train" if any(s["sample_id"] == m["sample_id"] and s["split"] == "train" for s in train_samples) else "val"

    if len(train_samples) >= args.train_count and len(val_samples) >= args.val_count:
        summary_status = "passed"
    elif train_samples and val_samples:
        summary_status = "short_target"
    else:
        summary_status = "insufficient"
    dataset_summary = {
        "status": summary_status,
        "source": "pinder_hf_parquet",
        "parquets": [str(p) for p in args.parquet],
        "source_row_counts": dict(source_counts),
        "accepted_total": len(samples),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "target_train_count": args.train_count,
        "target_val_count": args.val_count,
        "reject_counts": dict(Counter(r.get("reject", "unknown") for r in rejects)),
        "families_train": len({s["family_split_key"] for s in train_samples}),
        "families_val": len({s["family_split_key"] for s in val_samples}),
        "elapsed_sec": time.time() - started,
        "notes": [
            "PINDER rows are constrained-hybrid supervision, not exact multistate experimental benchmark.",
            "local_latents are geometry proxy here and must be replaced by stage05_extract_ae_latents.py before training.",
        ],
    }
    tensor_path = args.out_dir / "stage08b_pinder_hybrid_samples.pt"
    train_path = args.out_dir / "T_hybrid_silver_train.pt"
    val_path = args.out_dir / "T_hybrid_silver_val.pt"
    torch.save({"samples": samples, "manifest": dataset_summary}, tensor_path)
    torch.save({"samples": train_samples, "manifest": dataset_summary}, train_path)
    torch.save({"samples": val_samples, "manifest": dataset_summary}, val_path)
    manifest_dir = args.out_dir / "manifests"
    write_json(manifest_dir / "T_hybrid_silver_manifest.json", manifests)
    write_json(manifest_dir / "T_hybrid_silver_train_manifest.json", [m for m in manifests if m["split"] == "train"])
    write_json(manifest_dir / "T_hybrid_silver_val_manifest.json", [m for m in manifests if m["split"] == "val"])
    write_json(manifest_dir / "rejects_sample.json", rejects[:200])
    dataset_summary.update(
        {
            "tensor_dataset": str(tensor_path),
            "train_tensor": str(train_path),
            "val_tensor": str(val_path),
            "manifest": str(manifest_dir / "T_hybrid_silver_manifest.json"),
        }
    )
    write_json(args.summary, dataset_summary)
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
