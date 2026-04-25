#!/usr/bin/env python
"""Build Stage06 predictor-derived multistate train/val datasets with Boltz2.

The script consumes real experimental multistate target+binder seeds, predicts
complexes state-by-state with Boltz2, extracts geometry/interface supervision,
creates multiple multistate variants per accepted base pair, and writes a
family-holdout train/internal-val dataset for pilot fine-tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.strategy01.stage04_predictors.boltz_adapter import BoltzAdapter, BoltzRunConfig  # noqa: E402
from scripts.strategy01.stage05_build_predictor_multistate_dataset import (  # noqa: E402
    ATOM_ORDER,
    aggregate_worst,
    chain_to_atom37_any,
    geometry_latents,
    interface_labels,
    list_chain_ids,
    passes_bronze,
    quality_from_metrics,
    write_chain_pdb,
)
from scripts.strategy01.stage06_mine_multistate_validation import (  # noqa: E402
    DATA_DIR as VALIDATION_DATA_DIR,
    exact_sample_from_candidate,
    http_download,
    parse_models,
    pick_chain_pair,
    query_rcsb_nmr_entries,
    select_state_indices,
)

RCSB_PDB_URL = "https://files.rcsb.org/download/{entry}.pdb"
DEFAULT_EXACT = VALIDATION_DATA_DIR / "V_exact_manifest.json"
DEFAULT_HYBRID_SEEDS = VALIDATION_DATA_DIR / "V_hybrid_seed_manifest.json"
DEFAULT_OUT = REPO_ROOT / "data" / "strategy01" / "stage06_production"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def passes_silver(metrics: dict[str, Any], geom: dict[str, Any]) -> bool:
    contacts = int(geom.get("contact_count", 0))
    severe_clash = bool(geom.get("severe_clash", True))
    plddt = metrics.get("complex_plddt_norm")
    iptm = metrics.get("protein_iptm")
    interchain_pae = metrics.get("interchain_pae_A")
    pdockq2 = metrics.get("pDockQ2_proxy")
    if contacts < 12 or severe_clash:
        return False
    if plddt is not None and float(plddt) < 0.70:
        return False
    if iptm is not None and float(iptm) < 0.35:
        return False
    if interchain_pae is not None and float(interchain_pae) > 10.0:
        return False
    if pdockq2 is not None and float(pdockq2) < 0.23:
        return False
    return True


def passes_bronze_stage06(metrics: dict[str, Any], geom: dict[str, Any]) -> bool:
    contacts = int(geom.get("contact_count", 0))
    severe_clash = bool(geom.get("severe_clash", True))
    plddt = metrics.get("complex_plddt_norm")
    iptm = metrics.get("protein_iptm")
    interchain_pae = metrics.get("interchain_pae_A")
    if contacts < 8 or severe_clash:
        return False
    if plddt is not None and round(float(plddt), 2) < 0.55:
        return False
    if iptm is not None and round(float(iptm), 2) < 0.20:
        return False
    if interchain_pae is not None and round(float(interchain_pae), 2) > 15.0:
        return False
    return True


def normalize_chain_ids(chains: list[str]) -> tuple[str, str]:
    if "A" in chains and "B" in chains:
        return "A", "B"
    if len(chains) >= 2:
        return chains[0], chains[1]
    raise RuntimeError(f"Predicted complex has fewer than two chains: {chains}")


def extract_anchor_constraints(sample: dict[str, Any], max_constraints: int) -> tuple[list[list[tuple[int, int]]], list[torch.Tensor]]:
    per_state_constraints: list[list[tuple[int, int]]] = []
    per_state_hotspots: list[torch.Tensor] = []
    target_len = len(sample["target_sequence"])
    binder_len = len(sample["shared_binder_sequence"])
    for exact_path in sample["exact_complex_paths"]:
        target_exact = chain_to_atom37_any(Path(exact_path), sample["target_chain_id"], target_len)
        binder_exact = chain_to_atom37_any(Path(exact_path), sample["binder_chain_id"], binder_len)
        labels = interface_labels(target_exact, binder_exact)
        contact = labels["contact_labels"].bool()
        hotspots = contact.any(dim=1)
        pairs = []
        for ti, bi in contact.nonzero(as_tuple=False).tolist():
            pairs.append((int(ti) + 1, int(bi) + 1))
            if len(pairs) >= max_constraints:
                break
        per_state_constraints.append(pairs)
        per_state_hotspots.append(hotspots)
    return per_state_constraints, per_state_hotspots


def base_state_bundle(sample: dict[str, Any], active_state_indices: list[int], per_state_metrics: list[dict[str, Any]], predicted_paths: list[str], exact_hotspots: list[torch.Tensor]) -> dict[str, Any]:
    target_states = []
    target_masks = []
    target_seqs = []
    pred_hotspots = []
    bb_states = []
    lat_states = []
    contacts = []
    distances = []
    label_masks = []
    quality = []
    target_len = len(sample["target_sequence"])
    binder_len = len(sample["shared_binder_sequence"])
    for bundle_idx, (state_idx, complex_path) in enumerate(zip(active_state_indices, predicted_paths, strict=False)):
        target_path = sample["target_state_paths"][state_idx]
        target_chain = chain_to_atom37_any(Path(target_path), sample["target_chain_id"], target_len)
        chains = list_chain_ids(Path(complex_path))
        pred_target_chain_id, pred_binder_chain_id = normalize_chain_ids(chains)
        target_pred = chain_to_atom37_any(Path(complex_path), pred_target_chain_id, target_len)
        binder_pred = chain_to_atom37_any(Path(complex_path), pred_binder_chain_id, binder_len)
        labels = interface_labels(target_pred, binder_pred)
        target_states.append(target_chain.x)
        target_masks.append(target_chain.mask)
        target_seqs.append(target_chain.seq)
        pred_hotspots.append(labels["contact_labels"].bool().any(dim=1))
        bb_states.append(binder_pred.x[:, ATOM_ORDER["CA"], :])
        lat_states.append(geometry_latents(binder_pred.x[:, ATOM_ORDER["CA"], :], binder_pred.seq))
        contacts.append(labels["contact_labels"])
        distances.append(labels["distance_labels"])
        label_masks.append(labels["label_mask"])
        quality.append(quality_from_metrics(per_state_metrics[bundle_idx], bool(labels["metrics"]["severe_clash"])))
    return {
        "x_target_states": torch.stack(target_states),
        "target_mask_states": torch.stack(target_masks),
        "seq_target_states": torch.stack(target_seqs),
        "target_hotspot_mask_states_pred": torch.stack(pred_hotspots),
        "target_hotspot_mask_states_exact": torch.stack(exact_hotspots),
        "binder_seq_shared": torch.tensor([ord(c) for c in sample["shared_binder_sequence"]], dtype=torch.long),  # replaced later
        "x_1_states": {"bb_ca": torch.stack(bb_states), "local_latents": torch.stack(lat_states)},
        "state_mask": torch.ones(len(bb_states), binder_len, dtype=torch.bool),
        "state_present_mask": torch.ones(len(bb_states), dtype=torch.bool),
        "interface_contact_labels": torch.stack(contacts),
        "interface_distance_labels": torch.stack(distances),
        "interface_label_mask": torch.stack(label_masks),
        "interface_quality_labels": torch.stack(quality),
        "interface_quality_mask": torch.ones(len(bb_states), 5, dtype=torch.bool),
    }


def seq_to_tensor(seq: str) -> torch.Tensor:
    from openfold.np import residue_constants

    restype_order = residue_constants.restype_order
    return torch.tensor([int(restype_order.get(aa, 0)) for aa in seq], dtype=torch.long)


def build_variant(sample: dict[str, Any], bundle: dict[str, Any], per_state_metrics: list[dict[str, Any]], state_indices: list[int], hotspot_mode: str, variant_rank: int, quality_tier: str, split_group: str) -> dict[str, Any]:
    hotspot_key = "target_hotspot_mask_states_exact" if hotspot_mode == "exact" else "target_hotspot_mask_states_pred"
    binder_seq = seq_to_tensor(sample["shared_binder_sequence"])
    selected_metrics = [per_state_metrics[i] for i in state_indices]
    state_count = len(state_indices)
    return {
        "sample_id": f"{sample['sample_id']}__v{variant_rank:02d}__k{state_count}__h{hotspot_mode}",
        "split": "unset",
        "target_id": sample["target_id"],
        "family_split_key": split_group,
        "source_dataset": "stage06_real_multistate_boltz",
        "source_tier": "train_boltz",
        "predictor": "boltz2",
        "shared_binder_sequence": sample["shared_binder_sequence"],
        "binder_chain_id": "B",
        "target_chain_id": "A",
        "target_state_paths": [sample["target_state_paths"][i] for i in state_indices],
        "target_state_chain_ids": [sample["target_chain_id"]] * state_count,
        "predicted_complex_paths": [sample["predicted_complex_paths"][i] for i in state_indices],
        "state_roles": ["required_bind"] * state_count,
        "state_weights": [float(1.0 / state_count)] * state_count,
        "leakage_keys": {"source_sample_id": sample["sample_id"], "family_holdout_key": split_group},
        "x_target_states": bundle["x_target_states"][state_indices].clone(),
        "target_mask_states": bundle["target_mask_states"][state_indices].clone(),
        "seq_target_states": bundle["seq_target_states"][state_indices].clone(),
        "target_hotspot_mask_states": bundle[hotspot_key][state_indices].clone(),
        "binder_seq_shared": binder_seq.clone(),
        "binder_seq_mask": torch.ones_like(binder_seq, dtype=torch.bool),
        "x_1_states": {
            "bb_ca": bundle["x_1_states"]["bb_ca"][state_indices].clone(),
            "local_latents": bundle["x_1_states"]["local_latents"][state_indices].clone(),
        },
        "state_mask": bundle["state_mask"][state_indices].clone(),
        "state_present_mask": bundle["state_present_mask"][state_indices].clone(),
        "target_state_weights": torch.full((state_count,), 1.0 / state_count, dtype=torch.float32),
        "target_state_roles": torch.ones(state_count, dtype=torch.long),
        "interface_contact_labels": bundle["interface_contact_labels"][state_indices].clone(),
        "interface_distance_labels": bundle["interface_distance_labels"][state_indices].clone(),
        "interface_label_mask": bundle["interface_label_mask"][state_indices].clone(),
        "interface_quality_labels": bundle["interface_quality_labels"][state_indices].clone(),
        "interface_quality_mask": bundle["interface_quality_mask"][state_indices].clone(),
        "state_metrics": selected_metrics,
        "worst_state_metrics": aggregate_worst(selected_metrics),
        "quality_tier": quality_tier,
        "ae_latent_source": None,
    }


def variant_state_subsets(nstates: int) -> list[list[int]]:
    if nstates >= 3:
        return [[0, 1, 2], [0, 1], [0, 2], [1, 2]]
    if nstates == 2:
        return [[0, 1]]
    return [[0]]


def _work_dir(args: argparse.Namespace) -> Path:
    return args.work_dir if getattr(args, "work_dir", None) is not None else args.out_dir


def mine_extra_seed_pool(args: argparse.Namespace, excluded_groups: set[str], target_count: int) -> list[dict[str, Any]]:
    work_dir = _work_dir(args)
    out_dir = work_dir / "T_prod_seed_states"
    cache_dir = work_dir / "cache"
    entries = query_rcsb_nmr_entries(args.max_entries)
    seeds: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for entry in entries:
        if len(seeds) >= target_count:
            break
        try:
            pdb_path = http_download(RCSB_PDB_URL.format(entry=entry), cache_dir / "rcsb_nmr_pdbs" / f"{entry}.pdb")
            models = parse_models(pdb_path)
            pair = pick_chain_pair(
                models=models,
                target_min_len=args.target_min_len,
                binder_min_len=args.binder_min_len,
                binder_max_len=args.binder_max_len,
                min_contacts=args.min_contacts,
                min_target_rmsd=args.min_target_rmsd,
            )
            if pair is None:
                continue
            split_group = hashlib.sha1(pair["target_seq"].encode("utf-8")).hexdigest()[:12]
            if split_group in excluded_groups:
                continue
            pair_key = (pair["target_seq"], pair["binder_seq"])
            if pair_key in seen_pairs:
                continue
            state_indices = select_state_indices(models, pair["target_chain"], args.min_states, args.max_states)
            if len(state_indices) < args.min_states:
                continue
            sample = exact_sample_from_candidate(entry, models, pair, state_indices, out_dir)
            if sample["split_group"] in excluded_groups:
                continue
            sample["exact_pair_flag"] = False
            sample["source_tier"] = "train_boltz"
            seeds.append(sample)
            seen_pairs.add(pair_key)
        except Exception:
            continue
    return seeds


def run_boltz_passes(sample: dict[str, Any], out_root: Path, adapters: dict[str, BoltzAdapter], max_constraints: int, timeout_sec: int, min_states_required: int) -> tuple[list[int], list[str], list[dict[str, Any]], str]:
    active_state_indices: list[int] = []
    predicted_paths: list[str] = []
    per_state_metrics: list[dict[str, Any]] = []
    constraints, _ = extract_anchor_constraints(sample, max_constraints=max_constraints)
    target_len = len(sample["target_sequence"])
    binder_len = len(sample["shared_binder_sequence"])
    failures: list[str] = []
    for state_idx, target_path in enumerate(sample["target_state_paths"]):
        target_chain = chain_to_atom37_any(Path(target_path), sample["target_chain_id"], target_len)
        target_seq = target_chain.seq_str
        template_path = out_root / sample["sample_id"] / "templates" / f"state{state_idx:02d}_target_A.pdb"
        write_chain_pdb(template_path, target_chain, chain_id="A")
        passes = [
            ("scout", adapters["scout"], None),
            ("second", adapters["second"], None),
        ]
        if constraints[state_idx]:
            passes.append(("anchor", adapters["anchor"], constraints[state_idx]))
        success = False
        for pass_name, adapter, pass_constraints in passes:
            try:
                result = adapter.predict(
                    target_pdb=template_path,
                    binder_sequence=sample["shared_binder_sequence"],
                    out_dir=out_root / sample["sample_id"],
                    sample_id=f"{sample['sample_id']}_{pass_name}",
                    state_index=state_idx,
                    target_sequence=target_seq,
                    target_template_pdb=template_path,
                    use_template=True,
                    force_template=True,
                    template_threshold=2.0,
                    msa="",
                    contact_constraints=pass_constraints,
                    target_len=target_len,
                    binder_len=binder_len,
                    timeout_sec=timeout_sec if timeout_sec > 0 else None,
                )
            except Exception as exc:
                last_failure = f"state{state_idx}:{pass_name}:{type(exc).__name__}"
                continue
            chains = list_chain_ids(result.complex_path)
            target_chain_id, binder_chain_id = normalize_chain_ids(chains)
            target_pred = chain_to_atom37_any(result.complex_path, target_chain_id, target_len)
            binder_pred = chain_to_atom37_any(result.complex_path, binder_chain_id, binder_len)
            labels = interface_labels(target_pred, binder_pred)
            merged_metrics = {**result.metrics, **labels["metrics"], "pass_name": pass_name}
            merged_metrics["bronze_pass"] = passes_bronze_stage06(result.metrics, labels["metrics"])
            merged_metrics["silver_pass"] = passes_silver(result.metrics, labels["metrics"])
            if merged_metrics["bronze_pass"]:
                active_state_indices.append(state_idx)
                predicted_paths.append(str(result.complex_path))
                per_state_metrics.append(merged_metrics)
                success = True
                break
            last_failure = f"state{state_idx}:{pass_name}"
        if not success:
            failures.append(last_failure)
    if len(active_state_indices) >= min_states_required:
        return active_state_indices, predicted_paths, per_state_metrics, "passed" if not failures else "partial_pass"
    return [], [], [], failures[0] if failures else "no_success"


def assign_splits(samples: list[dict[str, Any]], train_target: int, val_target: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    families = sorted({s["family_split_key"] for s in samples})
    if val_target <= 0:
        val_families = set()
    else:
        val_families = set(families[-max(1, round(0.2 * len(families))):])
    train = [s for s in samples if s["family_split_key"] not in val_families]
    val = [s for s in samples if s["family_split_key"] in val_families]
    for sample in train:
        sample["split"] = "train"
    for sample in val:
        sample["split"] = "val"
    return train[:train_target], val[:val_target]


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    exact = load_manifest(args.exact_manifest)
    hybrid_seeds = load_manifest(args.hybrid_seed_manifest)
    excluded_groups = {s["split_group"] for s in exact} | {s["split_group"] for s in hybrid_seeds}
    if args.seed_manifest and args.seed_manifest.exists():
        base_seeds = load_manifest(args.seed_manifest)
    else:
        base_seeds = mine_extra_seed_pool(args, excluded_groups, args.max_base_seeds)
    adapters = {
        "scout": BoltzAdapter(BoltzRunConfig(recycling_steps=1, sampling_steps=5, diffusion_samples=1, use_potentials=False, use_no_kernels=not args.enable_kernels, use_msa_server=True, num_workers=0)),
        "second": BoltzAdapter(BoltzRunConfig(recycling_steps=2, sampling_steps=10, diffusion_samples=1, use_potentials=False, use_no_kernels=not args.enable_kernels, use_msa_server=True, num_workers=0)),
        "anchor": BoltzAdapter(BoltzRunConfig(recycling_steps=2, sampling_steps=10, diffusion_samples=1, use_potentials=True, use_no_kernels=not args.enable_kernels, use_msa_server=True, num_workers=0)),
    }
    accepted_samples: list[dict[str, Any]] = []
    hybrid_built: list[dict[str, Any]] = []
    attrition = {"base_seed_count": len(base_seeds), "accepted_bases": 0, "rejected_bases": 0, "accepted_variants": 0, "hybrid_built": 0, "failures": []}

    def build_from_seed(seed: dict[str, Any], source_tier: str) -> list[dict[str, Any]]:
        active_state_indices, predicted_paths, per_state_metrics, status = run_boltz_passes(
            seed, _work_dir(args) / "boltz_outputs", adapters, args.max_constraints, args.timeout_sec, args.min_states
        )
        if status not in {"passed", "partial_pass"}:
            attrition["failures"].append({"sample_id": seed["sample_id"], "status": status})
            return []
        exact_constraints, exact_hotspots = extract_anchor_constraints(seed, max_constraints=args.max_constraints)
        active_seed = dict(seed)
        active_seed["target_state_paths"] = [seed["target_state_paths"][i] for i in active_state_indices]
        active_seed["exact_complex_paths"] = [seed["exact_complex_paths"][i] for i in active_state_indices]
        active_seed["predicted_complex_paths"] = predicted_paths
        bundle = base_state_bundle(active_seed, list(range(len(active_state_indices))), per_state_metrics, predicted_paths, [exact_hotspots[i] for i in active_state_indices])
        all_silver = all(bool(m["silver_pass"]) for m in per_state_metrics)
        quality_tier = "silver" if all_silver else "bronze"
        variants: list[dict[str, Any]] = []
        for variant_rank, subset in enumerate(variant_state_subsets(len(per_state_metrics))):
            for hotspot_mode in ["exact", "pred"]:
                variant = build_variant(active_seed, bundle, per_state_metrics, subset, hotspot_mode, variant_rank, quality_tier, seed["split_group"])
                variant["source_tier"] = source_tier
                variants.append(variant)
        return variants

    if len(exact) < args.target_exact_count:
        need_hybrid = min(args.target_hybrid_count, args.target_exact_count - len(exact))
        for seed in hybrid_seeds[:need_hybrid]:
            variants = build_from_seed(seed, "hybrid_experimental_boltz")
            if not variants:
                continue
            hybrid_built.extend(variants[:1])
            attrition["hybrid_built"] += 1

    for seed in base_seeds:
        if len(accepted_samples) >= args.train_count + args.val_count + 32:
            break
        variants = build_from_seed(seed, "train_boltz")
        if not variants:
            attrition["rejected_bases"] += 1
            continue
        accepted_samples.extend(variants)
        attrition["accepted_bases"] += 1
        attrition["accepted_variants"] += len(variants)

    train_samples, val_samples = assign_splits(accepted_samples, args.train_count, args.val_count)
    if not args.allow_short_dataset and (len(train_samples) < args.train_count or len(val_samples) < args.val_count):
        raise RuntimeError(f"Short dataset after Boltz screening: train={len(train_samples)} val={len(val_samples)} target={args.train_count}/{args.val_count}")

    combined = train_samples + val_samples
    summary = {
        "status": "passed",
        "exact_count": len(exact),
        "hybrid_built_count": len(hybrid_built),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "attrition": attrition,
        "base_seed_count": len(base_seeds),
        "work_dir": str(_work_dir(args)),
        "out_dir": str(args.out_dir),
        "boltz_environment": adapters["scout"].environment_status(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": combined, "manifest": summary}, args.out_dir / "stage06_predictor_pilot_samples.pt")
    torch.save({"samples": train_samples, "manifest": summary}, args.out_dir / "T_prod_train.pt")
    torch.save({"samples": val_samples, "manifest": summary}, args.out_dir / "T_prod_val.pt")
    (args.out_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifests" / "T_prod_train_manifest.json").write_text(json.dumps([{"sample_id": s["sample_id"], "family_split_key": s["family_split_key"], "quality_tier": s["quality_tier"]} for s in train_samples], indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "manifests" / "T_prod_val_manifest.json").write_text(json.dumps([{"sample_id": s["sample_id"], "family_split_key": s["family_split_key"], "quality_tier": s["quality_tier"]} for s in val_samples], indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "manifests" / "V_hybrid_manifest.json").write_text(json.dumps([{"sample_id": s["sample_id"], "family_split_key": s["family_split_key"], "quality_tier": s["quality_tier"]} for s in hybrid_built], indent=2, ensure_ascii=False), encoding="utf-8")
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-manifest", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--hybrid-seed-manifest", type=Path, default=DEFAULT_HYBRID_SEEDS)
    parser.add_argument("--seed-manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "reports" / "strategy01" / "probes" / "stage06_production_summary.json")
    parser.add_argument("--max-entries", type=int, default=800)
    parser.add_argument("--max-base-seeds", type=int, default=96)
    parser.add_argument("--target-exact-count", type=int, default=24)
    parser.add_argument("--target-hybrid-count", type=int, default=8)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--val-count", type=int, default=64)
    parser.add_argument("--min-states", type=int, default=2)
    parser.add_argument("--max-states", type=int, default=3)
    parser.add_argument("--target-min-len", type=int, default=30)
    parser.add_argument("--binder-min-len", type=int, default=6)
    parser.add_argument("--binder-max-len", type=int, default=120)
    parser.add_argument("--min-contacts", type=int, default=8)
    parser.add_argument("--min-target-rmsd", type=float, default=1.0)
    parser.add_argument("--max-constraints", type=int, default=8)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-kernels", action="store_true")
    parser.add_argument("--allow-short-dataset", action="store_true")
    args = parser.parse_args()
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary = build_dataset(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
