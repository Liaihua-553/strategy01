#!/usr/bin/env python
"""Stage07 B1/B2 Boltz2 refold smoke for one multistate sample.

B1: Strategy01 generated shared sequence from stage07_multistate_sampling_smoke.
B2: Reference/experimental shared binder sequence from the same sample.
Each sequence is refolded with every target state via Boltz2 and summarized with
worst-state metrics. This does not include B0 baseline generation yet.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from openfold.np import residue_constants

REPO = Path('/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase')
BOLTZ_SITE_PACKAGES = Path('/data/kfliao/general_model/envs/boltz_cb04aec/lib/python3.11/site-packages')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / 'src') not in sys.path:
    sys.path.insert(0, str(REPO / 'src'))
if BOLTZ_SITE_PACKAGES.exists() and str(BOLTZ_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(BOLTZ_SITE_PACKAGES))

import gemmi  # noqa: E402

from scripts.strategy01.stage04_predictors.boltz_adapter import BoltzAdapter, BoltzRunConfig  # noqa: E402
from scripts.strategy01.stage05_build_predictor_multistate_dataset import chain_to_atom37_any, interface_labels, list_chain_ids  # noqa: E402
from scripts.strategy01.stage06_build_boltz_production_dataset import normalize_chain_ids  # noqa: E402

DEFAULT_DATASET = REPO / 'data/strategy01/stage07_production_v23_wave1/stage06_predictor_pilot_samples_materialized_ae_latents.pt'
DEFAULT_SMOKE = REPO / 'reports/strategy01/probes/stage07_multistate_sampling_smoke_summary.json'
DEFAULT_OUT = REPO / 'data/strategy01/stage07_b1_b2_refold_smoke'
DEFAULT_REPORT = REPO / 'reports/strategy01/probes/stage07_b1_b2_refold_smoke_summary.json'
DEFAULT_MSA_CACHE = REPO / 'data/strategy01/stage07_boltz_msa_cache'
AA = residue_constants.restypes
AA1_TO_AA3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
    'X': 'ALA',
}


def seq_from_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> str:
    return ''.join(AA[int(i)] if int(i) < len(AA) else 'X' for i, ok in zip(tokens.detach().cpu(), mask.detach().cpu()) if bool(ok))


def target_seq(sample: dict[str, Any], state_idx: int) -> str:
    toks = sample['seq_target_states'][state_idx]
    mask = sample['target_mask_states'][state_idx].any(dim=-1)
    return seq_from_tokens(toks, mask)


def load_sample(dataset: Path, sample_id: str) -> dict[str, Any]:
    data = torch.load(dataset, map_location='cpu', weights_only=False)
    for sample in data['samples']:
        if sample.get('sample_id') == sample_id:
            return sample
    raise KeyError(sample_id)


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = {}
    for key in ['interchain_pae_A', 'protein_iptm', 'complex_plddt_norm', 'pDockQ2_proxy', 'contact_count']:
        nums = [r.get(key) for r in rows if r.get(key) is not None]
        nums = [float(x) for x in nums]
        if not nums:
            continue
        if key == 'interchain_pae_A':
            vals['worst_' + key] = max(nums)
            vals['mean_' + key] = sum(nums) / len(nums)
        else:
            vals['worst_' + key] = min(nums)
            vals['mean_' + key] = sum(nums) / len(nums)
    vals['all_state_bronze_like'] = all(bool(r.get('bronze_like')) for r in rows) if rows else False
    vals['n_states'] = len(rows)
    return vals


def reindex_pdb_residues(src: Path, dst: Path, chain_id: str = 'A') -> Path:
    """Write a Boltz-template-safe copy with residue ids reset to 1..N.

    Stage07 seed-state PDBs can retain source residue numbers such as 177..271.
    Boltz2 template parsing maps template residues against the input sequence;
    large source numbering can therefore index beyond a 1..N sequence and fail.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    residue_map: dict[tuple[str, str, str], int] = {}
    next_res = 1
    out_lines: list[str] = []
    for line in src.read_text(encoding='utf-8', errors='ignore').splitlines():
        if line.startswith(('ATOM  ', 'HETATM')) and len(line) >= 27 and line[21].strip() == chain_id:
            key = (line[21], line[22:26], line[26])
            if key not in residue_map:
                residue_map[key] = next_res
                next_res += 1
            line = f"{line[:22]}{residue_map[key]:4d} {line[27:]}"
        out_lines.append(line)
    if not out_lines or not any(line.startswith('ATOM') for line in out_lines):
        raise ValueError(f'No ATOM records found while reindexing template: {src}')
    dst.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
    return dst


def write_boltz_template_cif(src: Path, dst: Path, target_sequence: str, chain_id: str = 'A') -> Path:
    """Write a Boltz2-readable mmCIF template with explicit full sequence.

    Gemmi can infer coordinates from the seed PDB, but these extracted PDBs do
    not carry a full polymer sequence. Boltz2 template parsing then receives an
    empty sequence and fails. Setting the entity sequence before mmCIF export
    makes the template parser align coordinates to the intended target sequence.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_pdb = dst.with_suffix('.reindexed.pdb')
    reindex_pdb_residues(src, tmp_pdb, chain_id=chain_id)
    structure = gemmi.read_structure(str(tmp_pdb))
    structure.setup_entities()
    seq3 = [AA1_TO_AA3.get(res.upper(), 'ALA') for res in target_sequence]
    if not structure.entities:
        raise ValueError(f'No polymer entity found while creating template CIF: {src}')
    for entity in structure.entities:
        if entity.entity_type == gemmi.EntityType.Polymer:
            entity.full_sequence = seq3
    doc = structure.make_mmcif_document()
    doc.write_file(str(dst))
    return dst


def reference_anchor_constraints(
    sample: dict[str, Any],
    state_idx: int,
    target_len: int,
    binder_len: int,
    max_constraints: int,
) -> list[tuple[int, int]]:
    """Extract 1-based contact anchors from the accepted reference complex."""
    if max_constraints <= 0:
        return []
    paths = sample.get('predicted_complex_paths') or []
    if state_idx >= len(paths) or not paths[state_idx]:
        return []
    ref_path = REPO / paths[state_idx]
    chains = list_chain_ids(ref_path)
    target_chain, binder_chain = normalize_chain_ids(chains)
    ref_binder_len = int(sample['binder_seq_mask'].bool().sum().item())
    target_atom = chain_to_atom37_any(ref_path, target_chain, target_len)
    binder_atom = chain_to_atom37_any(ref_path, binder_chain, ref_binder_len)
    labels = interface_labels(target_atom, binder_atom)
    contacts = labels['contact_labels'] > 0.5
    distances = labels['distance_labels']
    candidates: list[tuple[float, int, int]] = []
    for pair in torch.nonzero(contacts, as_tuple=False):
        ti = int(pair[0])
        bi = int(pair[1])
        if ti >= target_len or bi >= binder_len:
            continue
        candidates.append((float(distances[ti, bi].item()), ti + 1, bi + 1))
    candidates.sort(key=lambda x: x[0])
    selected: list[tuple[int, int]] = []
    used_target: set[int] = set()
    used_binder: set[int] = set()
    for _, ti, bi in candidates:
        if ti in used_target or bi in used_binder:
            continue
        selected.append((ti, bi))
        used_target.add(ti)
        used_binder.add(bi)
        if len(selected) >= max_constraints:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--sampling-summary', type=Path, default=DEFAULT_SMOKE)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--msa-cache-dir', type=Path, default=DEFAULT_MSA_CACHE)
    parser.add_argument('--recycling-steps', type=int, default=1)
    parser.add_argument('--sampling-steps', type=int, default=5)
    parser.add_argument('--max-anchor-constraints', type=int, default=8)
    parser.add_argument('--timeout-sec', type=int, default=1800)
    args = parser.parse_args()

    smoke = json.loads(args.sampling_summary.read_text(encoding='utf-8'))
    sample = load_sample(args.dataset, smoke['sample_id'])
    valid_k = int(sample['state_present_mask'].bool().sum().item())
    sequences = {
        'B1_strategy01': smoke['pred_sequence'],
        'B2_reference': smoke['reference_sequence'],
    }
    adapter = BoltzAdapter(BoltzRunConfig(
        recycling_steps=args.recycling_steps,
        sampling_steps=args.sampling_steps,
        diffusion_samples=1,
        use_msa_server=True,
        use_no_kernels=True,
        use_potentials=args.max_anchor_constraints > 0,
        num_workers=0,
    ))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {'status': 'running', 'sample_id': smoke['sample_id'], 'target_id': smoke.get('target_id'), 'valid_states': valid_k, 'sequences': sequences, 'constraint_mode': 'reference_anchor_constraints' if args.max_anchor_constraints > 0 else 'unconstrained', 'max_anchor_constraints': args.max_anchor_constraints, 'per_mode': {}, 'started_at': time.time(), 'boltz_environment': adapter.environment_status()}
    for mode, binder_seq in sequences.items():
        rows = []
        mode_dir = args.out_dir / mode
        for state_idx in range(valid_k):
            target_pdb = REPO / sample['target_state_paths'][state_idx]
            t_seq = target_seq(sample, state_idx)
            sample_id = f"{smoke['sample_id']}__{mode}"
            try:
                template_pdb = write_boltz_template_cif(
                    target_pdb,
                    mode_dir / 'templates' / f"{sample_id}_state{state_idx:02d}_target_template.cif",
                    t_seq,
                    chain_id='A',
                )
                msa_spec = {
                    'A': str(BoltzAdapter.cached_msa_csv(t_seq, args.msa_cache_dir, 'A') or ''),
                    'B': str(BoltzAdapter.cached_msa_csv(binder_seq, args.msa_cache_dir, 'B') or ''),
                }
                constraints = reference_anchor_constraints(sample, state_idx, len(t_seq), len(binder_seq), args.max_anchor_constraints)
                pred = adapter.predict(
                    target_pdb=target_pdb,
                    binder_sequence=binder_seq,
                    out_dir=mode_dir,
                    sample_id=sample_id,
                    state_index=state_idx,
                    target_sequence=t_seq,
                    target_template_pdb=template_pdb,
                    use_template=True,
                    force_template=True,
                    template_threshold=2.0,
                    msa=msa_spec,
                    contact_constraints=constraints,
                    timeout_sec=args.timeout_sec,
                    target_len=len(t_seq),
                    binder_len=len(binder_seq),
                )
                input_stem = f"{sample_id}_state{state_idx:02d}_boltz"
                for chain_label, seq in (('A', t_seq), ('B', binder_seq)):
                    msa_csv = BoltzAdapter.generated_msa_csvs(mode_dir, input_stem).get(chain_label)
                    if msa_csv is not None and msa_csv.exists():
                        BoltzAdapter.cache_msa_csv(msa_csv, seq, args.msa_cache_dir, chain_label)
                chains = list_chain_ids(Path(pred.complex_path))
                target_chain, binder_chain = normalize_chain_ids(chains)
                target_atom = chain_to_atom37_any(Path(pred.complex_path), target_chain, len(t_seq))
                binder_atom = chain_to_atom37_any(Path(pred.complex_path), binder_chain, len(binder_seq))
                labels = interface_labels(target_atom, binder_atom)
                geom = labels['metrics']
                row = {'state_index': state_idx, 'complex_path': str(pred.complex_path), **pred.metrics, **geom}
                row['n_anchor_constraints'] = len(constraints)
                row['bronze_like'] = int(row.get('contact_count', 0)) >= 8 and not bool(row.get('severe_clash', True)) and (row.get('interchain_pae_A') is None or float(row['interchain_pae_A']) <= 15.0) and (row.get('protein_iptm') is None or float(row['protein_iptm']) >= 0.20) and (row.get('complex_plddt_norm') is None or float(row['complex_plddt_norm']) >= 0.55)
            except Exception as exc:
                row = {'state_index': state_idx, 'error': repr(exc), 'bronze_like': False}
            rows.append(row)
            all_results['per_mode'][mode] = {'states': rows, 'summary': summarize_metrics(rows)}
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding='utf-8')
    all_results['status'] = 'passed' if all(not any('error' in r for r in v['states']) for v in all_results['per_mode'].values()) else 'partial_failed'
    all_results['elapsed_sec'] = time.time() - all_results['started_at']
    args.report.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
