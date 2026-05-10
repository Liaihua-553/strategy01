#!/usr/bin/env python3
"""Stage17 target-only multiseed de-novo candidate probe.

This is a diagnostic, not a production oracle selector.  It separates two cases:
(1) the current flow never samples acceptable shared-sequence candidates; or
(2) some seeds are better, so Strategy01 needs a non-leaking candidate ranker / test-time scaling.

Selection proxies are label-free: sequence entropy and cross-state sequence disagreement.
Reference identity is reported only as a post-hoc diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / 'src') not in sys.path:
    sys.path.insert(0, str(REPO / 'src'))

import scripts.strategy01.stage04_real_complex_loss_debug as s4  # noqa: E402
import scripts.strategy01.stage10_pose_init_training as s10  # noqa: E402
import scripts.strategy01.stage12_de_novo_multistate_training as s12  # noqa: E402
import scripts.strategy01.stage12c_de_novo_smoke as s12c  # noqa: E402


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s4.jsonable(obj), ensure_ascii=False, indent=2), encoding='utf-8')


def final_diag(smoke: dict[str, Any]) -> dict[str, float | None]:
    diagnostics = smoke.get('diagnostics') or []
    if not diagnostics:
        return {'shared_entropy': None, 'state_seq_disagreement': None}
    last = diagnostics[-1]
    return {
        'shared_entropy': last.get('shared_entropy'),
        'state_seq_disagreement': last.get('state_seq_disagreement'),
    }


def no_leak_score(row: dict[str, Any]) -> float:
    # Lower is better. Uses only target coordinates and model confidence/agreement, never reference binder labels.
    ent = 10.0 if row.get('shared_entropy') is None else float(row['shared_entropy'])
    dis = 10.0 if row.get('state_seq_disagreement') is None else float(row['state_seq_disagreement'])
    clash = float(row.get('target_severe_clash_rate') or 0.0)
    contact = float(row.get('target_contact_count_mean') or 0.0)
    hotspot = float(row.get('target_hotspot_contact_count_mean') or 0.0)
    min_dist = row.get('target_min_distance_nm_mean')
    min_dist = 2.0 if min_dist is None else float(min_dist)
    contact_deficit = max(0.0, 8.0 - contact) / 8.0
    hotspot_deficit = max(0.0, 2.0 - hotspot) / 2.0
    distance_penalty = max(0.0, min_dist - 1.0)
    return ent + 10.0 * dis + 10.0 * clash + 2.0 * contact_deficit + 2.0 * hotspot_deficit + distance_penalty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=s12.DEFAULT_DATASET)
    parser.add_argument('--ae-ckpt', type=Path, default=s12.DEFAULT_AE_CKPT)
    parser.add_argument('--sampling-config', type=Path, default=s12.DEFAULT_SAMPLING_CONFIG)
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--max-samples', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=0, help='If >0, evaluate selected samples in chunks to avoid GPU OOM.')
    parser.add_argument('--nsteps', type=int, default=16)
    parser.add_argument('--seeds', default='1207,1211,1217,1223,1231,1237,1249,1259')
    parser.add_argument('--stage13-native-state-path', action='store_true')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--report-json', type=Path, default=REPO / 'reports/strategy01/probes/stage17_multiseed_denovo_probe.json')
    args = parser.parse_args()

    device = s12.choose_device(args.device)
    samples, manifest = s10.load_dataset(args.dataset)
    selected = [s for s in samples if s.get('split') == args.split][: args.max_samples]
    if not selected:
        raise RuntimeError(f'No samples selected for split={args.split}')

    stack_args = argparse.Namespace(
        init_ckpt=args.checkpoint,
        ae_ckpt=args.ae_ckpt,
        target_center_noise_scale_nm=0.20,
        latent_state_residual_noise_scale=0.05,
        lambda_ae_seq=1.5,
        lambda_seq_ae_consistency=0.15,
        lambda_flow_gate_reg=0.03,
        ae_seq_hard_state_alpha=0.0,
        ae_seq_hard_state_gamma=0.0,
    )
    model, fm, ae, model_meta, ckpt_meta, loss_cfg = s12.build_model_stack(device, stack_args)
    sampling_cfg = s12.load_sampling_cfg(args.sampling_config)
    seeds = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]

    chunk_size = int(args.chunk_size or 0)
    chunks = [selected] if chunk_size <= 0 else [selected[i : i + chunk_size] for i in range(0, len(selected), chunk_size)]
    rows = []
    for seed in seeds:
        chunk_rows = []
        for chunk_index, chunk in enumerate(chunks):
            smoke = s12c.rollout_final(
                model=model,
                ae=ae,
                fm=fm,
                samples=chunk,
                device=device,
                sampling_cfg=sampling_cfg,
                nsteps=args.nsteps,
                seed=seed + chunk_index * 100003,
                stage13_native_state_path=args.stage13_native_state_path,
            )
            diag = final_diag(smoke)
            final_x = smoke.get('final_x_identity') or {}
            chunk_rows.append({
                'chunk_index': chunk_index,
                'sample_count': len(chunk),
                'shared_identity_mean_posthoc': final_x.get('final_x_ae_shared_identity_mean'),
                'ae_state_identity_mean_posthoc': final_x.get('final_x_ae_state_identity_mean'),
                **diag,
            })
        denom = max(1, sum(int(r['sample_count']) for r in chunk_rows))
        def wavg(key: str):
            vals = [(float(r[key]), int(r['sample_count'])) for r in chunk_rows if r.get(key) is not None]
            if not vals:
                return None
            return sum(v * w for v, w in vals) / max(1, sum(w for _, w in vals))
        row = {
            'seed': seed,
            'nsteps': args.nsteps,
            'sample_count': len(selected),
            'chunk_size': chunk_size or len(selected),
            'chunk_count': len(chunks),
            'shared_identity_mean_posthoc': wavg('shared_identity_mean_posthoc'),
            'ae_state_identity_mean_posthoc': wavg('ae_state_identity_mean_posthoc'),
            'shared_entropy': wavg('shared_entropy'),
            'state_seq_disagreement': wavg('state_seq_disagreement'),
            'target_min_distance_nm_mean': wavg('target_min_distance_nm_mean'),
            'target_min_distance_nm_min': wavg('target_min_distance_nm_min'),
            'target_contact_count_mean': wavg('target_contact_count_mean'),
            'target_hotspot_contact_count_mean': wavg('target_hotspot_contact_count_mean'),
            'target_severe_clash_rate': wavg('target_severe_clash_rate'),
            'chunks': chunk_rows,
        }
        row['no_leak_score'] = no_leak_score(row)
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k != 'chunks'}, ensure_ascii=False), flush=True)

    rows_by_proxy = sorted(rows, key=lambda x: x['no_leak_score'])
    rows_by_identity = sorted(rows, key=lambda x: float(x.get('shared_identity_mean_posthoc') or -1.0), reverse=True)
    selected_proxy = rows_by_proxy[0] if rows_by_proxy else None
    oracle = rows_by_identity[0] if rows_by_identity else None
    result = {
        'stage': 'stage17_multiseed_denovo_probe',
        'status': 'passed',
        'checkpoint': str(args.checkpoint),
        'dataset': str(args.dataset),
        'split': args.split,
        'sample_count': len(selected),
        'chunk_size': chunk_size or len(selected),
        'nsteps': args.nsteps,
        'seeds': seeds,
        'contract': {
            'target_only_de_novo': True,
            'production_selection_uses_reference_labels': False,
            'posthoc_identity_is_diagnostic_only': True,
            'stage13_native_state_path': bool(args.stage13_native_state_path),
        },
        'model_meta': model_meta,
        'checkpoint_meta': ckpt_meta,
        'loss_cfg': loss_cfg,
        'rows': rows,
        'selected_by_no_leak_proxy': selected_proxy,
        'oracle_best_by_reference_identity': oracle,
        'diagnosis': {
            'candidate_diversity_identity_range': [
                min(float(r.get('shared_identity_mean_posthoc') or 0.0) for r in rows),
                max(float(r.get('shared_identity_mean_posthoc') or 0.0) for r in rows),
            ] if rows else None,
            'proxy_matches_oracle_seed': (selected_proxy or {}).get('seed') == (oracle or {}).get('seed') if rows else None,
        },
    }
    write_json(args.report_json, result)
    print(json.dumps({'status': 'passed', 'report': str(args.report_json)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
