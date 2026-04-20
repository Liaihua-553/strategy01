#!/usr/bin/env python
"""Build Strategy01 Stage05 predictor-derived multistate complex dataset.

Input supervision is produced by a complex predictor (Boltz-2 by default):
for every target state and one shared binder sequence, predict a complex, parse
binder coordinates and interface labels, then save a tensor dataset.  Predicted
complexes are labels only; model conditioning remains target-state ensemble +
state metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openfold.np import residue_constants

from scripts.strategy01.stage04_build_real_multistate_complex_dataset import (  # noqa: E402
    ATOM_ORDER,
    BACKBONE_OFFSETS,
    ChainTensor,
    aa3_to_idx,
    geometry_latents,
    get_chain_residues,
    interface_labels,
)
from scripts.strategy01.stage04_predictors.boltz_adapter import BoltzAdapter, BoltzRunConfig  # noqa: E402

DEFAULT_SOURCE = REPO_ROOT / "data" / "strategy01" / "stage04_real_complex_multistate" / "stage04_debug_samples.pt"
DEFAULT_OUT = REPO_ROOT / "data" / "strategy01" / "stage05_predictor_multistate"


def seq_tensor_to_str(seq: torch.Tensor) -> str:
    restypes = residue_constants.restypes
    chars = []
    for idx in seq.detach().cpu().long().tolist():
        if 0 <= idx < len(restypes):
            chars.append(restypes[idx])
        else:
            chars.append("A")
    return "".join(chars)


def write_chain_pdb(path: Path, chain: ChainTensor, chain_id: str = "A") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atom_names = ["N", "CA", "C", "O"]
    lines: list[str] = []
    resnames: list[str] = []
    for i in range(chain.x.shape[0]):
        aa = residue_constants.restypes[int(chain.seq[i].item())] if int(chain.seq[i].item()) < len(residue_constants.restypes) else "A"
        resnames.append(residue_constants.restype_1to3.get(aa, "ALA"))
    for block_idx, start in enumerate(range(0, len(resnames), 13), start=1):
        chunk = resnames[start : start + 13]
        lines.append(f"SEQRES {block_idx:3d} {chain_id:1s} {len(resnames):4d}  {' '.join(chunk)}\n")
    serial = 1
    for i, resname in enumerate(resnames):
        for atom_name in atom_names:
            atom_idx = ATOM_ORDER[atom_name]
            if not bool(chain.mask[i, atom_idx].item()):
                continue
            xyz = chain.x[i, atom_idx] * 10.0
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {resname:>3s} {chain_id}{i+1:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 80.00           {atom_name[0]:>2s}\n"
            )
            serial += 1
    lines.append("TER\nEND\n")
    path.write_text("".join(lines), encoding="utf-8")


def tensor_state_to_chain(x: torch.Tensor, mask: torch.Tensor, seq: torch.Tensor) -> ChainTensor:
    return ChainTensor(
        x=x.detach().cpu().float(),
        mask=mask.detach().cpu().bool(),
        seq=seq.detach().cpu().long(),
        seq_str=seq_tensor_to_str(seq),
        source_residue_count=int(seq.numel()),
    )


def _iter_pdb_atom_records(path: Path):
    """Yield robust atom records from PDB lines, including Boltz lines whose
    coordinates may overflow fixed PDB columns and break Bio.PDB's strict parser.
    """
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        parts = line.split()
        if len(parts) >= 9:
            try:
                yield {
                    "atom_name": parts[2].strip(),
                    "resname": parts[3].strip(),
                    "chain_id": parts[4].strip(),
                    "resseq": int(float(parts[5])),
                    "xyz": torch.tensor([float(parts[6]), float(parts[7]), float(parts[8])], dtype=torch.float32) / 10.0,
                }
                continue
            except Exception:
                pass
        try:
            yield {
                "atom_name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain_id": line[21].strip() or "A",
                "resseq": int(line[22:26]),
                "xyz": torch.tensor([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=torch.float32) / 10.0,
            }
        except Exception:
            continue


def list_chain_ids(path: Path) -> list[str]:
    seen: list[str] = []
    for rec in _iter_pdb_atom_records(path):
        chain_id = rec["chain_id"]
        if chain_id not in seen:
            seen.append(chain_id)
    return seen


def chain_to_atom37_any(path: Path, chain_id: str, length: int) -> ChainTensor:
    residues: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for rec in _iter_pdb_atom_records(path):
        if rec["chain_id"] != chain_id:
            continue
        resseq = int(rec["resseq"])
        if resseq not in residues:
            residues[resseq] = {"resname": rec["resname"], "atoms": {}}
            order.append(resseq)
        residues[resseq]["atoms"][rec["atom_name"]] = rec["xyz"]
    selected_keys = order[:length]
    x = torch.zeros(length, 37, 3, dtype=torch.float32)
    mask = torch.zeros(length, 37, dtype=torch.bool)
    seq = torch.zeros(length, dtype=torch.long)
    seq_chars: list[str] = []
    restype_3to1 = residue_constants.restype_3to1
    restype_order = residue_constants.restype_order
    for i in range(length):
        if i < len(selected_keys):
            record = residues[selected_keys[i]]
            aa1 = restype_3to1.get(str(record["resname"]).upper(), "A")
            seq[i] = int(restype_order.get(aa1, 0))
            seq_chars.append(aa1)
            for atom_name, xyz in record["atoms"].items():
                if atom_name in ATOM_ORDER:
                    x[i, ATOM_ORDER[atom_name]] = xyz
                    mask[i, ATOM_ORDER[atom_name]] = True
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
    return ChainTensor(x=x, mask=mask, seq=seq, seq_str="".join(seq_chars), source_residue_count=len(order))

def quality_from_metrics(metrics: dict[str, Any], severe_clash: bool) -> torch.Tensor:
    interchain_pae = metrics.get("interchain_pae_A")
    if interchain_pae is None or not math.isfinite(float(interchain_pae)):
        pae_score = 0.0
    else:
        pae_score = max(0.0, 1.0 - min(float(interchain_pae), 31.0) / 31.0)
    plddt = metrics.get("complex_iplddt_norm", metrics.get("complex_plddt_norm"))
    protein_iptm = metrics.get("protein_iptm")
    pdockq2 = metrics.get("pDockQ2_proxy")
    return torch.tensor(
        [
            float(pae_score),
            float(plddt) if plddt is not None else 0.0,
            float(protein_iptm) if protein_iptm is not None else 0.0,
            float(pdockq2) if pdockq2 is not None else 0.0,
            0.0 if severe_clash else 1.0,
        ],
        dtype=torch.float32,
    )


def passes_bronze(metrics: dict[str, Any], geom: dict[str, Any]) -> bool:
    contacts = int(geom.get("contact_count", 0))
    severe_clash = bool(geom.get("severe_clash", True))
    plddt = metrics.get("complex_plddt_norm")
    iptm = metrics.get("protein_iptm")
    interchain_pae = metrics.get("interchain_pae_A")
    if contacts < 8 or severe_clash:
        return False
    if plddt is not None and float(plddt) < 0.55:
        return False
    if iptm is not None and float(iptm) < 0.20:
        return False
    if interchain_pae is not None and float(interchain_pae) > 15.0:
        return False
    return True


def aggregate_worst(per_state_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    def finite_values(key: str) -> list[float]:
        vals = []
        for m in per_state_metrics:
            v = m.get(key)
            if v is not None:
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        vals.append(fv)
                except Exception:
                    pass
        return vals

    pae = finite_values("interchain_pae_A")
    iptm = finite_values("protein_iptm")
    plddt = finite_values("complex_plddt_norm")
    pdockq2 = finite_values("pDockQ2_proxy")
    contacts = finite_values("contact_count")
    return {
        "worst_interchain_pae_A": max(pae) if pae else None,
        "worst_protein_iptm": min(iptm) if iptm else None,
        "worst_complex_plddt_norm": min(plddt) if plddt else None,
        "worst_pDockQ2_proxy": min(pdockq2) if pdockq2 else None,
        "worst_contact_count": min(contacts) if contacts else None,
        "state_metric_std_interchain_pae_A": float(torch.tensor(pae).std(unbiased=False).item()) if len(pae) > 1 else 0.0,
    }


def build_predictor_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not args.source_tensor.exists():
        raise FileNotFoundError(f"Source tensor not found: {args.source_tensor}")
    source = torch.load(args.source_tensor, map_location="cpu")
    source_samples = source["samples"]
    out_dir = args.out_dir
    target_template_dir = out_dir / "target_templates"
    boltz_out_dir = out_dir / "boltz_outputs"
    tensor_samples: list[dict[str, Any]] = []
    manifest_samples: list[dict[str, Any]] = []
    adapter = BoltzAdapter(
        BoltzRunConfig(
            recycling_steps=args.boltz_recycling_steps,
            sampling_steps=args.boltz_sampling_steps,
            diffusion_samples=args.boltz_diffusion_samples,
            use_potentials=args.use_potentials,
            use_no_kernels=not args.enable_kernels,
            num_workers=args.num_workers,
        )
    )
    env_status = adapter.environment_status()
    started = time.time()
    state_runtime_sec: list[float] = []
    prepared_count = 0
    for source_index, src in enumerate(source_samples):
        current_count = prepared_count if args.prepare_only else len(tensor_samples)
        if current_count >= args.max_samples:
            break
        k_available = int(src["state_present_mask"].numel())
        k = min(args.states_per_sample, k_available)
        sample_id = f"stage05_{current_count:03d}_{src['sample_id']}"
        target_seq_states: list[str] = []
        predicted_paths: list[str] = []
        per_state_metrics: list[dict[str, Any]] = []
        bb_states: list[torch.Tensor] = []
        lat_states: list[torch.Tensor] = []
        contacts: list[torch.Tensor] = []
        distances: list[torch.Tensor] = []
        label_masks: list[torch.Tensor] = []
        quality: list[torch.Tensor] = []
        hotspots: list[torch.Tensor] = []
        ok = True
        for state_idx in range(k):
            target_chain = tensor_state_to_chain(src["x_target_states"][state_idx], src["target_mask_states"][state_idx], src["seq_target_states"][state_idx])
            target_seq = target_chain.seq_str
            target_seq_states.append(target_seq)
            template_path = target_template_dir / f"{sample_id}_state{state_idx:02d}_target_A.pdb"
            write_chain_pdb(template_path, target_chain, chain_id="A")
            result = None
            input_stem = f"{sample_id}_state{state_idx:02d}_boltz"
            state_start = time.time()
            if args.prepare_only:
                adapter.write_input_yaml(
                    boltz_out_dir / "inputs" / f"{input_stem}.yaml",
                    target_seq,
                    src["shared_binder_sequence"],
                    target_template_pdb=None if args.disable_template else template_path,
                    use_template=not args.disable_template,
                    force_template=args.force_template,
                    template_threshold=args.template_threshold,
                    template_id=args.template_id,
                )
                continue
            if args.run_boltz:
                result = adapter.predict(
                    template_path,
                    src["shared_binder_sequence"],
                    boltz_out_dir,
                    sample_id,
                    state_idx,
                    target_sequence=target_seq,
                    target_template_pdb=None if args.disable_template else template_path,
                    use_template=not args.disable_template,
                    force_template=args.force_template,
                    template_threshold=args.template_threshold,
                    template_id=args.template_id,
                    target_len=int(src["x_target_states"].shape[1]),
                    binder_len=int(src["binder_seq_shared"].numel()),
                    timeout_sec=args.timeout_sec if args.timeout_sec > 0 else None,
                )
            else:
                result = adapter.parse_existing_output(
                    boltz_out_dir,
                    input_stem,
                    sample_id=sample_id,
                    state_index=state_idx,
                    target_len=int(src["x_target_states"].shape[1]),
                    binder_len=int(src["binder_seq_shared"].numel()),
                )
            state_runtime_sec.append(time.time() - state_start)
            chains = list_chain_ids(result.complex_path)
            if len(chains) < 2:
                raise RuntimeError(f"Predicted complex has fewer than two chains: {result.complex_path}; chains={chains}")
            target_chain_id = "A" if "A" in chains else chains[0]
            binder_chain_id = "B" if "B" in chains else chains[1]
            target_pred = chain_to_atom37_any(result.complex_path, target_chain_id, int(src["x_target_states"].shape[1]))
            binder_pred = chain_to_atom37_any(result.complex_path, binder_chain_id, int(src["binder_seq_shared"].numel()))
            labels = interface_labels(target_pred, binder_pred)
            geom_metrics = labels["metrics"]
            merged_metrics = {**result.metrics, **geom_metrics}
            merged_metrics["bronze_pass"] = passes_bronze(result.metrics, geom_metrics)
            if args.require_bronze and not bool(merged_metrics["bronze_pass"]):
                ok = False
                break
            predicted_paths.append(str(result.complex_path))
            per_state_metrics.append(merged_metrics)
            bb_states.append(binder_pred.x[:, ATOM_ORDER["CA"], :])
            lat_states.append(geometry_latents(binder_pred.x[:, ATOM_ORDER["CA"], :], binder_pred.seq))
            contacts.append(labels["contact_labels"])
            distances.append(labels["distance_labels"])
            label_masks.append(labels["label_mask"])
            quality.append(quality_from_metrics(result.metrics, bool(geom_metrics["severe_clash"])))
            hotspots.append(labels["contact_labels"].bool().any(dim=1))
        if args.prepare_only:
            prepared_count += 1
            continue
        if not ok:
            continue
        if len(bb_states) != k:
            continue
        split = "train" if len(tensor_samples) < args.train_count else "val"
        sample = {
            "sample_id": sample_id,
            "split": split,
            "target_id": src.get("target_id", sample_id),
            "family_split_key": src.get("target_id", sample_id),
            "source_dataset": "stage04_seed_plus_boltz2_predicted_complex",
            "predictor": "boltz2",
            "shared_binder_sequence": src["shared_binder_sequence"],
            "binder_chain_id": "B",
            "target_chain_id": "A",
            "target_state_paths": [str(p) for p in src.get("target_state_paths", [])[:k]],
            "target_state_chain_ids": [str(c) for c in src.get("target_state_chain_ids", [])[:k]],
            "predicted_complex_paths": predicted_paths,
            "state_roles": ["required_bind"] * k,
            "state_weights": [float(1.0 / k)] * k,
            "leakage_keys": {"source_sample_id": src.get("sample_id"), "source_index": source_index},
            "x_target_states": src["x_target_states"][:k].clone(),
            "target_mask_states": src["target_mask_states"][:k].clone(),
            "seq_target_states": src["seq_target_states"][:k].clone(),
            "target_hotspot_mask_states": torch.stack(hotspots),
            "binder_seq_shared": src["binder_seq_shared"].clone(),
            "binder_seq_mask": src["binder_seq_mask"].clone(),
            "x_1_states": {"bb_ca": torch.stack(bb_states), "local_latents": torch.stack(lat_states)},
            "state_mask": torch.ones(k, int(src["binder_seq_shared"].numel()), dtype=torch.bool),
            "state_present_mask": torch.ones(k, dtype=torch.bool),
            "target_state_weights": torch.full((k,), 1.0 / k, dtype=torch.float32),
            "target_state_roles": torch.ones(k, dtype=torch.long),
            "interface_contact_labels": torch.stack(contacts),
            "interface_distance_labels": torch.stack(distances),
            "interface_label_mask": torch.stack(label_masks),
            "interface_quality_labels": torch.stack(quality),
            "interface_quality_mask": torch.ones(k, 5, dtype=torch.bool),
            "state_metrics": per_state_metrics,
            "worst_state_metrics": aggregate_worst(per_state_metrics),
        }
        tensor_samples.append(sample)
        manifest_samples.append(
            {
                "sample_id": sample["sample_id"],
                "split": split,
                "target_id": sample["target_id"],
                "family_split_key": sample["family_split_key"],
                "target_state_paths": sample["target_state_paths"],
                "state_roles": sample["state_roles"],
                "state_weights": sample["state_weights"],
                "shared_binder_sequence": sample["shared_binder_sequence"],
                "predicted_complex_paths": sample["predicted_complex_paths"],
                "per_state_metrics": per_state_metrics,
                "worst_state_metrics": sample["worst_state_metrics"],
                "source_dataset": sample["source_dataset"],
                "leakage_keys": sample["leakage_keys"],
            }
        )
    if args.prepare_only:
        manifest = {
            "stage": "stage05_predictor_multistate_prepare_only",
            "status": "prepared_boltz_inputs_only",
            "out_dir": str(out_dir),
            "boltz_environment": env_status,
        }
        return [], manifest
    if len(tensor_samples) < args.train_count + args.val_count:
        if args.allow_short_dataset:
            train_count = min(args.train_count, max(0, len(tensor_samples) - args.val_count))
        else:
            raise RuntimeError(f"Only built {len(tensor_samples)} samples; need {args.train_count + args.val_count}")
    else:
        train_count = args.train_count
    for idx, sample in enumerate(tensor_samples):
        sample["split"] = "train" if idx < train_count else "val"
        manifest_samples[idx]["split"] = sample["split"]
    manifest = {
        "stage": "stage05_predictor_multistate_pilot",
        "note": "Boltz-2 predictor-derived complexes are used as labels only; model inputs remain target-state ensembles.",
        "num_samples": len(tensor_samples),
        "train_count": sum(1 for s in tensor_samples if s["split"] == "train"),
        "val_count": sum(1 for s in tensor_samples if s["split"] == "val"),
        "states_per_sample": args.states_per_sample,
        "source_tensor": str(args.source_tensor),
        "boltz_environment": env_status,
        "runtime_sec_total": time.time() - started,
        "runtime_sec_per_state_median": float(torch.tensor(state_runtime_sec).median().item()) if state_runtime_sec else None,
        "samples": manifest_samples,
    }
    return tensor_samples, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tensor", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--train-count", type=int, default=1)
    parser.add_argument("--val-count", type=int, default=1)
    parser.add_argument("--states-per-sample", type=int, default=2)
    parser.add_argument("--run-boltz", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--require-bronze", action="store_true")
    parser.add_argument("--allow-short-dataset", action="store_true")
    parser.add_argument("--boltz-recycling-steps", type=int, default=1)
    parser.add_argument("--boltz-sampling-steps", type=int, default=5)
    parser.add_argument("--boltz-diffusion-samples", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--force-template", action="store_true")
    parser.add_argument("--disable-template", action="store_true", help="Run Boltz from sequence-only input; used as a smoke fallback when Boltz template parsing fails.")
    parser.add_argument("--use-potentials", action="store_true")
    parser.add_argument("--enable-kernels", action="store_true")
    parser.add_argument("--template-threshold", type=float, default=2.0)
    parser.add_argument("--template-id", type=str, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples, manifest = build_predictor_samples(args)
    manifest_path = args.out_dir / "manifest_stage05_predictor_pilot.json"
    tensor_path = args.out_dir / "stage05_predictor_pilot_samples.pt"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if samples:
        torch.save({"samples": samples, "manifest": manifest}, tensor_path)
    print(json.dumps({"status": "passed", "manifest": str(manifest_path), "tensor": str(tensor_path) if samples else None, "num_samples": len(samples)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()





