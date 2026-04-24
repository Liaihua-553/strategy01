#!/usr/bin/env python
"""Mine Stage06 exact/hybrid multistate validation seeds.

Current runnable path prioritizes exact experimental multi-state complexes from
RCSB multi-model NMR entries.  This gives a scientifically clean bootstrap set:
same target, same binder, same sequence, multiple experimentally observed
conformers.  The script also prepares hybrid seeds that can later be completed
with Boltz2 when the exact set is smaller than the desired target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
DATA_DIR = REPO_ROOT / "data" / "strategy01" / "stage06_validation"
REPORT_DIR = REPO_ROOT / "reports" / "strategy01" / "probes"

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query?json="
RCSB_PDB_URL = "https://files.rcsb.org/download/{entry}.pdb"

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


@dataclass
class ResidueRecord:
    resseq: int
    icode: str
    resname: str
    ca: tuple[float, float, float] | None


@dataclass
class ModelRecord:
    model_id: int
    chains: dict[str, list[ResidueRecord]]
    atom_lines: list[str]


def http_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        url + urllib.parse.quote(json.dumps(payload)),
        headers={"User-Agent": "codex-stage06/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_download(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    req = urllib.request.Request(url, headers={"User-Agent": "codex-stage06/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        out_path.write_bytes(resp.read())
    return out_path


def query_rcsb_nmr_entries(max_entries: int) -> list[str]:
    payload = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "SOLUTION NMR",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_model_count",
                        "operator": "greater_or_equal",
                        "value": 5,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                        "operator": "greater_or_equal",
                        "value": 2,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_entries},
            "results_verbosity": "compact",
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }
    resp = http_json(RCSB_SEARCH_URL, payload)
    out = []
    for item in resp.get("result_set", []):
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and "identifier" in item:
            out.append(item["identifier"])
    return out


def parse_models(pdb_path: Path) -> list[ModelRecord]:
    models: list[ModelRecord] = []
    current_id = 1
    current_lines: list[str] = []
    current_chain_records: dict[str, dict[tuple[int, str], ResidueRecord]] = {}
    has_model_cards = False

    def flush_model() -> None:
        nonlocal current_lines, current_chain_records, current_id
        if current_chain_records:
            chains = {cid: [current_chain_records[cid][k] for k in sorted(current_chain_records[cid])] for cid in sorted(current_chain_records)}
            models.append(ModelRecord(model_id=current_id, chains=chains, atom_lines=list(current_lines)))
        current_lines = []
        current_chain_records = {}

    for raw_line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True):
        line = raw_line.rstrip("\n")
        if line.startswith("MODEL"):
            has_model_cards = True
            if current_chain_records:
                flush_model()
            try:
                current_id = int(line.split()[-1])
            except Exception:
                current_id = len(models) + 1
            continue
        if line.startswith("ENDMDL"):
            flush_model()
            current_id = len(models) + 1
            continue
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        current_lines.append(raw_line)
        atom_name = line[12:16].strip()
        resname = line[17:20].strip().upper()
        chain_id = (line[21].strip() or "A").upper()
        try:
            resseq = int(line[22:26])
        except Exception:
            continue
        icode = line[26].strip()
        key = (resseq, icode)
        chain_map = current_chain_records.setdefault(chain_id, {})
        if key not in chain_map:
            chain_map[key] = ResidueRecord(resseq=resseq, icode=icode, resname=resname, ca=None)
        if atom_name == "CA":
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except Exception:
                continue
            chain_map[key].ca = xyz
    if current_chain_records:
        flush_model()
    if not has_model_cards and models:
        return models[:1]
    return models


def residue_sequence(residues: list[ResidueRecord]) -> str:
    return "".join(AA3_TO_1.get(res.resname.upper(), "X") for res in residues if res.ca is not None)


def coords_for_chain(model: ModelRecord, chain_id: str) -> list[tuple[float, float, float]]:
    return [res.ca for res in model.chains.get(chain_id, []) if res.ca is not None]


def calc_contacts(model: ModelRecord, target_chain: str, binder_chain: str, cutoff_a: float = 8.0) -> int:
    target = coords_for_chain(model, target_chain)
    binder = coords_for_chain(model, binder_chain)
    if not target or not binder:
        return 0
    cutoff2 = cutoff_a * cutoff_a
    contacts = 0
    for xa, ya, za in target:
        for xb, yb, zb in binder:
            d2 = (xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2
            if d2 <= cutoff2:
                contacts += 1
    return contacts


def calc_rmsd(coords_a: list[tuple[float, float, float]], coords_b: list[tuple[float, float, float]]) -> float | None:
    n = min(len(coords_a), len(coords_b))
    if n < 3:
        return None
    acc = 0.0
    for (xa, ya, za), (xb, yb, zb) in zip(coords_a[:n], coords_b[:n], strict=False):
        acc += (xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2
    return math.sqrt(acc / n)


def motion_bin(max_rmsd: float) -> str:
    if max_rmsd > 5.0:
        return "large"
    if max_rmsd >= 2.5:
        return "medium"
    return "small"


def sequence_identity(seq_a: str, seq_b: str) -> float:
    n = min(len(seq_a), len(seq_b))
    if n == 0:
        return 0.0
    matches = sum(1 for a, b in zip(seq_a[:n], seq_b[:n], strict=False) if a == b)
    return matches / n


def pick_chain_pair(
    models: list[ModelRecord],
    target_min_len: int,
    binder_min_len: int,
    binder_max_len: int,
    min_contacts: int,
    min_target_rmsd: float,
) -> dict[str, Any] | None:
    if not models:
        return None
    ref = models[0]
    best: dict[str, Any] | None = None
    for target_chain, target_res in ref.chains.items():
        target_len = sum(res.ca is not None for res in target_res)
        if target_len < target_min_len:
            continue
        target_seq = residue_sequence(target_res)
        if not target_seq or "X" in target_seq:
            continue
        for binder_chain, binder_res in ref.chains.items():
            if binder_chain == target_chain:
                continue
            binder_len = sum(res.ca is not None for res in binder_res)
            if binder_len < binder_min_len or binder_len > binder_max_len:
                continue
            binder_seq = residue_sequence(binder_res)
            if not binder_seq or "X" in binder_seq:
                continue
            if target_seq == binder_seq:
                continue
            if len(target_seq) == len(binder_seq) and sequence_identity(target_seq, binder_seq) >= 0.95:
                continue
            contacts = calc_contacts(ref, target_chain, binder_chain)
            if contacts < min_contacts:
                continue
            target_ref = coords_for_chain(ref, target_chain)
            binder_ref = coords_for_chain(ref, binder_chain)
            target_rmsds = []
            binder_rmsds = []
            for model in models[1:]:
                target_coords = coords_for_chain(model, target_chain)
                binder_coords = coords_for_chain(model, binder_chain)
                tr = calc_rmsd(target_ref, target_coords)
                br = calc_rmsd(binder_ref, binder_coords)
                if tr is not None:
                    target_rmsds.append(tr)
                if br is not None:
                    binder_rmsds.append(br)
            max_target_rmsd = max(target_rmsds) if target_rmsds else 0.0
            if max_target_rmsd < min_target_rmsd:
                continue
            score = contacts + 5.0 * max_target_rmsd + 0.5 * (max(binder_rmsds) if binder_rmsds else 0.0)
            current = {
                "target_chain": target_chain,
                "target_len": target_len,
                "target_seq": target_seq,
                "binder_chain": binder_chain,
                "binder_len": binder_len,
                "binder_seq": binder_seq,
                "contacts_model1": contacts,
                "max_target_rmsd": round(max_target_rmsd, 3),
                "max_binder_rmsd": round(max(binder_rmsds) if binder_rmsds else 0.0, 3),
                "score": score,
            }
            if best is None or current["score"] > best["score"]:
                best = current
    return best


def select_state_indices(models: list[ModelRecord], target_chain: str, min_states: int, max_states: int) -> list[int]:
    ref_coords = coords_for_chain(models[0], target_chain)
    scores: list[tuple[float, int]] = []
    for idx, model in enumerate(models[1:], start=1):
        rmsd = calc_rmsd(ref_coords, coords_for_chain(model, target_chain))
        if rmsd is not None:
            scores.append((rmsd, idx))
    scores.sort(reverse=True)
    selected = [0]
    for _, idx in scores:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= max_states:
            break
    if len(selected) < min_states:
        return []
    return selected


def write_state_pdb(model: ModelRecord, chain_ids: set[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [line for line in model.atom_lines if (line[21].strip() or "A").upper() in chain_ids]
    out_path.write_text("".join(lines) + "END\n", encoding="utf-8")


def exact_sample_from_candidate(entry: str, models: list[ModelRecord], pair: dict[str, Any], state_indices: list[int], out_dir: Path) -> dict[str, Any]:
    target_chain = pair["target_chain"]
    binder_chain = pair["binder_chain"]
    target_state_paths = []
    exact_complex_paths = []
    state_model_ids = []
    for local_idx, model_index in enumerate(state_indices):
        model = models[model_index]
        state_dir = out_dir / entry.lower()
        target_path = state_dir / f"{entry.lower()}_state{local_idx:02d}_target_{target_chain}.pdb"
        complex_path = state_dir / f"{entry.lower()}_state{local_idx:02d}_complex_{target_chain}{binder_chain}.pdb"
        write_state_pdb(model, {target_chain}, target_path)
        write_state_pdb(model, {target_chain, binder_chain}, complex_path)
        target_state_paths.append(str(target_path))
        exact_complex_paths.append(str(complex_path))
        state_model_ids.append(int(model.model_id))
    split_group = hashlib.sha1(pair["target_seq"].encode("utf-8")).hexdigest()[:12]
    max_rmsd = float(pair["max_target_rmsd"])
    return {
        "sample_id": f"{entry.lower()}_{target_chain}_{binder_chain}",
        "target_id": entry.lower(),
        "binder_id": f"{entry.lower()}_{binder_chain}",
        "binder_type": "peptide" if int(pair["binder_len"]) <= 40 else "protein",
        "target_uniprot": None,
        "binder_uniprot_or_sequence": pair["binder_seq"],
        "target_sequence": pair["target_seq"],
        "shared_binder_sequence": pair["binder_seq"],
        "target_chain_id": target_chain,
        "binder_chain_id": binder_chain,
        "target_state_paths": target_state_paths,
        "exact_complex_paths": exact_complex_paths,
        "state_methods": ["solution_nmr"] * len(target_state_paths),
        "state_roles": ["required_bind"] * len(target_state_paths),
        "state_weights": [round(1.0 / len(target_state_paths), 6)] * len(target_state_paths),
        "exact_pair_flag": True,
        "source_db": "rcsb_nmr",
        "source_tier": "exact_experimental",
        "assembly_support": "author_bioassembly",
        "motion_metrics": {
            "target_backbone_rmsd_A": max_rmsd,
            "interface_region_rmsd_A": max_rmsd,
            "contacts_model1": int(pair["contacts_model1"]),
            "binder_backbone_rmsd_A": float(pair["max_binder_rmsd"]),
            "state_model_ids": state_model_ids,
        },
        "motion_bin": motion_bin(max_rmsd),
        "quality_tier": "silver",
        "ae_latent_source": None,
        "split_group": split_group,
        "family_holdout_key": split_group,
    }


def mine_exact_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    entries = query_rcsb_nmr_entries(args.max_entries)
    exact: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    errors: list[dict[str, str]] = []
    families: set[str] = set()
    pdb_cache = args.cache_dir / "rcsb_nmr_pdbs"
    exact_root = args.out_dir / "V_exact_states"

    for entry in entries:
        try:
            pdb_path = http_download(RCSB_PDB_URL.format(entry=entry), pdb_cache / f"{entry}.pdb")
            models = parse_models(pdb_path)
            if len(models) < args.min_states:
                continue
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
            pair_key = (pair["target_seq"], pair["binder_seq"])
            if pair_key in seen_pairs:
                continue
            state_indices = select_state_indices(models, pair["target_chain"], args.min_states, args.max_states)
            if len(state_indices) < args.min_states:
                continue
            sample = exact_sample_from_candidate(entry, models, pair, state_indices, exact_root)
            seen_pairs.add(pair_key)
            if len(exact) < args.target_exact_count:
                exact.append(sample)
                families.add(sample["split_group"])
            else:
                sample["exact_pair_flag"] = False
                sample["source_tier"] = "hybrid_experimental_boltz"
                extras.append(sample)
            if len(exact) >= args.target_exact_count and len(extras) >= args.target_hybrid_count:
                break
        except Exception as exc:
            errors.append({"entry": entry, "error": str(exc)[:400]})

    hybrid = extras[: max(0, args.target_hybrid_count if len(exact) < args.target_exact_count else 0)]
    summary = {
        "status": "passed" if len(exact) >= args.min_exact_count else "short_exact",
        "n_entries_scanned": len(entries),
        "n_exact": len(exact),
        "n_hybrid_seed": len(hybrid),
        "n_exact_families": len({s["split_group"] for s in exact}),
        "n_motion_small": sum(1 for s in exact if s["motion_bin"] == "small"),
        "n_motion_medium": sum(1 for s in exact if s["motion_bin"] == "medium"),
        "n_motion_large": sum(1 for s in exact if s["motion_bin"] == "large"),
        "errors": errors[:50],
    }
    return exact, hybrid, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DATA_DIR / "cache")
    parser.add_argument("--max-entries", type=int, default=400)
    parser.add_argument("--target-exact-count", type=int, default=24)
    parser.add_argument("--min-exact-count", type=int, default=16)
    parser.add_argument("--target-hybrid-count", type=int, default=8)
    parser.add_argument("--min-states", type=int, default=2)
    parser.add_argument("--max-states", type=int, default=3)
    parser.add_argument("--target-min-len", type=int, default=30)
    parser.add_argument("--binder-min-len", type=int, default=6)
    parser.add_argument("--binder-max-len", type=int, default=120)
    parser.add_argument("--min-contacts", type=int, default=8)
    parser.add_argument("--min-target-rmsd", type=float, default=1.0)
    parser.add_argument("--exact-manifest", type=Path, default=DATA_DIR / "V_exact_manifest.json")
    parser.add_argument("--hybrid-seed-manifest", type=Path, default=DATA_DIR / "V_hybrid_seed_manifest.json")
    parser.add_argument("--summary", type=Path, default=REPORT_DIR / "stage06_validation_mining_summary.json")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    exact, hybrid, summary = mine_exact_samples(args)
    args.exact_manifest.write_text(json.dumps(exact, indent=2, ensure_ascii=False), encoding="utf-8")
    args.hybrid_seed_manifest.write_text(json.dumps(hybrid, indent=2, ensure_ascii=False), encoding="utf-8")
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": summary, "exact_manifest": str(args.exact_manifest), "hybrid_seed_manifest": str(args.hybrid_seed_manifest)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
