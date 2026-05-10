#!/usr/bin/env python3
"""Stage17 per-sample target-only multiseed candidate probe."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for x in [REPO, REPO / 'src']:
    if str(x) not in sys.path:
        sys.path.insert(0, str(x))

import scripts.strategy01.stage04_real_complex_loss_debug as s4  # noqa: E402
import scripts.strategy01.stage10_pose_init_training as s10  # noqa: E402
import scripts.strategy01.stage12_de_novo_multistate_training as s12  # noqa: E402
import scripts.strategy01.stage12c_de_novo_smoke as s12c  # noqa: E402


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s4.jsonable(obj), ensure_ascii=False, indent=2), encoding='utf-8')


def final_diag(smoke: dict[str, Any]) -> dict[str, float | None]:
    diagnostics = smoke.get('diagnostics') or []
    last = diagnostics[-1] if diagnostics else {}
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


def mean(xs):
    xs=[float(x) for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--dataset', type=Path, default=s12.DEFAULT_DATASET)
    ap.add_argument('--ae-ckpt', type=Path, default=s12.DEFAULT_AE_CKPT)
    ap.add_argument('--sampling-config', type=Path, default=s12.DEFAULT_SAMPLING_CONFIG)
    ap.add_argument('--split', choices=['train','val'], default='val')
    ap.add_argument('--max-samples', type=int, default=12)
    ap.add_argument('--nsteps', type=int, default=16)
    ap.add_argument('--seeds', default='1207,1211,1217,1223,1231,1237,1249,1259')
    ap.add_argument('--stage13-native-state-path', action='store_true')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--report-json', type=Path, default=REPO/'reports/strategy01/probes/stage17_per_sample_multiseed_probe.json')
    args=ap.parse_args()

    device=s12.choose_device(args.device)
    samples, manifest=s10.load_dataset(args.dataset)
    selected=[s for s in samples if s.get('split')==args.split][:args.max_samples]
    seeds=[int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    stack_args=argparse.Namespace(init_ckpt=args.checkpoint, ae_ckpt=args.ae_ckpt, target_center_noise_scale_nm=0.20, latent_state_residual_noise_scale=0.05, lambda_ae_seq=1.5, lambda_seq_ae_consistency=0.15, lambda_flow_gate_reg=0.03, ae_seq_hard_state_alpha=0.0, ae_seq_hard_state_gamma=0.0)
    model,fm,ae,model_meta,ckpt_meta,loss_cfg=s12.build_model_stack(device, stack_args)
    sampling_cfg=s12.load_sampling_cfg(args.sampling_config)

    sample_rows=[]
    for sample_idx, sample in enumerate(selected):
        cand=[]
        for seed in seeds:
            smoke=s12c.rollout_final(model, ae, fm, [sample], device, sampling_cfg, args.nsteps, seed + sample_idx*100003, stage13_native_state_path=args.stage13_native_state_path)
            final_x=smoke.get('final_x_identity') or {}
            row={'seed':seed,'sample_idx':sample_idx,'sample_id':sample.get('sample_id'), 'target_id':sample.get('target_id'), 'shared_identity_mean_posthoc':final_x.get('final_x_ae_shared_identity_mean'), 'ae_state_identity_mean_posthoc':final_x.get('final_x_ae_state_identity_mean'), **final_diag(smoke), **(smoke.get('target_geometry_proxy') or {})}
            row['no_leak_score']=no_leak_score(row)
            cand.append(row)
        by_proxy=sorted(cand, key=lambda r:r['no_leak_score'])[0]
        by_oracle=sorted(cand, key=lambda r:float(r.get('shared_identity_mean_posthoc') or -1), reverse=True)[0]
        first=next(r for r in cand if r['seed']==seeds[0])
        out={'sample_idx':sample_idx,'sample_id':sample.get('sample_id'),'target_id':sample.get('target_id'),'first_seed':first,'selected_by_proxy':by_proxy,'oracle_best':by_oracle,'proxy_matches_oracle':by_proxy['seed']==by_oracle['seed'],'identity_range':[min(float(r.get('shared_identity_mean_posthoc') or 0) for r in cand), max(float(r.get('shared_identity_mean_posthoc') or 0) for r in cand)],'candidates':cand}
        sample_rows.append(out)
        print(json.dumps({k:v for k,v in out.items() if k!='candidates'}, ensure_ascii=False), flush=True)

    first_vals=[r['first_seed']['shared_identity_mean_posthoc'] for r in sample_rows]
    proxy_vals=[r['selected_by_proxy']['shared_identity_mean_posthoc'] for r in sample_rows]
    oracle_vals=[r['oracle_best']['shared_identity_mean_posthoc'] for r in sample_rows]
    result={'stage':'stage17_per_sample_multiseed_probe','status':'passed','checkpoint':str(args.checkpoint),'split':args.split,'sample_count':len(selected),'nsteps':args.nsteps,'seeds':seeds,'contract':{'target_only_de_novo':True,'production_selection_uses_reference_labels':False,'posthoc_identity_is_diagnostic_only':True},'summary':{'first_seed_identity_mean':mean(first_vals),'proxy_selected_identity_mean':mean(proxy_vals),'oracle_identity_mean':mean(oracle_vals),'proxy_matches_oracle_rate':sum(1 for r in sample_rows if r['proxy_matches_oracle'])/max(1,len(sample_rows))},'samples':sample_rows,'model_meta':model_meta,'checkpoint_meta':ckpt_meta,'loss_cfg':loss_cfg}
    write_json(args.report_json, result)
    print(json.dumps({'status':'passed','report':str(args.report_json),'summary':result['summary']}, ensure_ascii=False, indent=2))

if __name__=='__main__':
    main()
