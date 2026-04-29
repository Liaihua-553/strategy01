#!/usr/bin/env python
"""Stage07 state-specific multistate sampling smoke.

This script is intentionally separate from proteinfoundation.generate. The stock
sampler consumes the legacy averaged `bb_ca/local_latents` outputs, while
Strategy01 needs each target state to maintain its own trajectory.  Here we keep
one shared binder sequence head, but update K state-specific flow states using
`bb_ca_states/local_latents_states` at every step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from openfold.np import residue_constants

REPO = Path('/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / 'src') not in sys.path:
    sys.path.insert(0, str(REPO / 'src'))

from proteinfoundation.proteina import Proteina  # noqa: E402
import scripts.strategy01.stage04_real_complex_loss_debug as s4  # noqa: E402

DEFAULT_CKPT = REPO / 'ckpts/stage07_sequence_consensus/runs/stage07_v23_wave1_pilot_structfix5/mini_final_lightning.ckpt'
DEFAULT_AE = REPO / 'ckpts/stage03_multistate_loss/complexa_ae_init_readonly_copy.ckpt'
DEFAULT_DATASET = REPO / 'data/strategy01/stage07_production_v23_wave1/stage06_predictor_pilot_samples_materialized_ae_latents.pt'
DEFAULT_SAMPLING = REPO / 'configs/pipeline/model_sampling.yaml'
DEFAULT_OUT = REPO / 'results/strategy01/stage07_sampling_smoke'
DEFAULT_REPORT = REPO / 'reports/strategy01/probes/stage07_multistate_sampling_smoke_summary.json'
AA = residue_constants.restypes


def tensor_to_json(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): tensor_to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tensor_to_json(v) for v in x]
    return x


def choose_sample(samples: list[dict[str, Any]], split: str, min_states: int) -> tuple[int, dict[str, Any]]:
    for i, s in enumerate(samples):
        if s.get('split') == split and int(s['state_present_mask'].bool().sum().item()) >= min_states:
            return i, s
    for i, s in enumerate(samples):
        if int(s['state_present_mask'].bool().sum().item()) >= min_states:
            return i, s
    raise RuntimeError(f'No sample with >= {min_states} states')


def weighted_average_states(x_states: dict[str, torch.Tensor], weights: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    out = {}
    for dm, x in x_states.items():
        out[dm] = (x * weights[:, :, None, None]).sum(dim=1) * mask[..., None]
    return out


def normalize_weights(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    present = batch['state_present_mask'].bool().to(device)
    weights = batch['target_state_weights'].float().to(device) * present.float()
    fallback = present.float() / present.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    denom = weights.sum(dim=1, keepdim=True)
    return torch.where(denom > 1e-8, weights / denom.clamp_min(1e-8), fallback)


def kabsch_rmsd(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.bool()
    if int(valid.sum()) < 3:
        return float('nan')
    x = x[valid].float()
    y = y[valid].float()
    x0 = x - x.mean(dim=0, keepdim=True)
    y0 = y - y.mean(dim=0, keepdim=True)
    cov = x0.T @ y0
    u, _, vh = torch.linalg.svd(cov)
    d = torch.sign(torch.det(vh.T @ u.T))
    corr = torch.diag(torch.tensor([1.0, 1.0, float(d)], device=x.device))
    rot = vh.T @ corr @ u.T
    xa = x0 @ rot.T
    return float(torch.sqrt(((xa - y0) ** 2).sum(dim=-1).mean()).detach().cpu().item())


def seq_from_logits(logits: torch.Tensor, mask: torch.Tensor) -> str:
    idx = logits.argmax(dim=-1)[0]
    m = mask[0].bool()
    return ''.join(AA[int(i)] if int(i) < len(AA) else 'X' for i in idx[m].detach().cpu())


def seq_from_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> str:
    return ''.join(AA[int(i)] if int(i) < len(AA) else 'X' for i, ok in zip(tokens.detach().cpu(), mask.detach().cpu()) if bool(ok))


def write_ca_pdb(path: Path, ca_nm: torch.Tensor, seq: str, chain: str = 'B') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    ca_a = ca_nm.detach().cpu().float() * 10.0
    for i, xyz in enumerate(ca_a, start=1):
        aa = seq[i-1] if i-1 < len(seq) else 'X'
        res3 = residue_constants.restype_1to3.get(aa, 'UNK')
        x, y, z = [float(v) for v in xyz]
        lines.append(f"ATOM  {i:5d}  CA  {res3:>3s} {chain}{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
    lines.append('TER')
    lines.append('END')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def state_specific_sample(model: Proteina, batch: dict[str, Any], nsteps: int, sampling_cfg: Any, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    fm = model.fm
    model.eval()
    mask = batch['mask'].bool().to(device)
    state_mask = batch['state_mask'].bool().to(device)
    b, k, n = state_mask.shape
    weights = normalize_weights(batch, device)
    x_states = {}
    for dm in fm.data_modes:
        dim = int(batch['x_1_states'][dm].shape[-1])
        base = fm.base_flow_matchers[dm]
        flat_mask = state_mask.reshape(b * k, n)
        x0 = base.sample_noise(n=n, shape=(b * k,), device=device, mask=flat_mask, training=False)
        x_states[dm] = x0.reshape(b, k, n, dim) * state_mask[..., None]

    ts, gt = fm.sample_schedule(nsteps=nsteps, sampling_model_args=sampling_cfg)
    ts = {dm: val.to(device) for dm, val in ts.items()}
    gt = {dm: val.to(device) for dm, val in gt.items()}
    last_nn_out = None
    for step in range(nsteps):
        batch['x_t'] = weighted_average_states(x_states, weights, mask)
        batch['t'] = {dm: ts[dm][step].expand(b).to(device) for dm in fm.data_modes}
        batch['x_sc'] = {dm: torch.zeros_like(batch['x_t'][dm]) for dm in fm.data_modes}
        with torch.no_grad():
            nn_out = model.nn(batch)
        last_nn_out = nn_out
        updated = {}
        for dm in fm.data_modes:
            state_key = f'{dm}_states'
            param = model.cfg_exp.nn.output_parameterization[dm]
            flat_x = x_states[dm].reshape(b * k, n, x_states[dm].shape[-1])
            flat_mask = state_mask.reshape(b * k, n)
            flat_t = ts[dm][step].expand(b * k).to(device)
            dt = ts[dm][step + 1] - ts[dm][step]
            flat_nn = {param: nn_out[state_key].reshape(b * k, n, x_states[dm].shape[-1])}
            base = fm.base_flow_matchers[dm]
            flat_nn = base.nn_out_add_clean_sample_prediction(flat_x, flat_t, flat_mask, flat_nn)
            flat_nn = base.nn_out_add_simulation_tensor(flat_x, flat_t, flat_mask, flat_nn)
            flat_nn = base.nn_out_add_guided_simulation_tensor(flat_nn, None, None, guidance_w=1.0, ag_ratio=0.0)
            flat_next = base.simulation_step(
                x_t=flat_x,
                nn_out=flat_nn,
                t=flat_t,
                dt=dt,
                gt=gt[dm][step],
                mask=flat_mask,
                simulation_step_params=sampling_cfg[dm]['simulation_step_params'],
            )
            updated[dm] = flat_next.reshape(b, k, n, x_states[dm].shape[-1]) * state_mask[..., None]
        x_states = updated

    batch['x_t'] = weighted_average_states(x_states, weights, mask)
    batch['t'] = {dm: torch.ones(b, device=device) for dm in fm.data_modes}
    batch['x_sc'] = {dm: torch.zeros_like(batch['x_t'][dm]) for dm in fm.data_modes}
    with torch.no_grad():
        final_nn_out = model.nn(batch)
    return x_states, final_nn_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--autoencoder-ckpt', type=Path, default=DEFAULT_AE)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--sampling-config', type=Path, default=DEFAULT_SAMPLING)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--split', default='val')
    parser.add_argument('--min-states', type=int, default=3)
    parser.add_argument('--nsteps', type=int, default=24)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    data = torch.load(args.dataset, map_location='cpu', weights_only=False)
    samples = data['samples']
    sample_idx, sample = choose_sample(samples, args.split, args.min_states)
    model = Proteina.load_from_checkpoint(str(args.ckpt), strict=False, autoencoder_ckpt_path=str(args.autoencoder_ckpt), map_location='cpu')
    model.to(device)
    batch = s4.collate_samples([sample], device)
    sampling_cfg = OmegaConf.to_container(OmegaConf.load(args.sampling_config).model, resolve=True)
    x_states, nn_out = state_specific_sample(model, batch, args.nsteps, sampling_cfg, device)

    pred_seq = seq_from_logits(nn_out['seq_logits_shared'], batch['mask'])
    ref_seq = sample.get('shared_binder_sequence') or seq_from_tokens(sample['binder_seq_shared'], sample['binder_seq_mask'])
    seq_identity = sum(a == b for a, b in zip(pred_seq, ref_seq)) / max(1, min(len(pred_seq), len(ref_seq)))
    sample_out = args.out_dir / str(sample.get('sample_id', f'sample_{sample_idx}'))
    sample_out.mkdir(parents=True, exist_ok=True)
    (sample_out / 'shared_sequence.fasta').write_text(f">stage07_b1_smoke|{sample.get('sample_id')}\n{pred_seq}\n", encoding='utf-8')
    (sample_out / 'reference_sequence.fasta').write_text(f">reference|{sample.get('sample_id')}\n{ref_seq}\n", encoding='utf-8')

    state_summaries = []
    b0 = batch['x_1_states']['bb_ca'][0].detach().cpu()
    sm = batch['state_mask'][0].detach().cpu()
    pred_ca = x_states['bb_ca'][0].detach().cpu()
    valid_k = int(sample['state_present_mask'].bool().sum().item())
    for k in range(valid_k):
        write_ca_pdb(sample_out / f'state{k:02d}_binder_ca.pdb', pred_ca[k, sm[k]], pred_seq)
        rmsd = kabsch_rmsd(pred_ca[k], b0[k], sm[k])
        state_summaries.append({'state_index': k, 'ca_rmsd_to_label_nm': rmsd})

    pairwise = []
    for i in range(valid_k):
        for j in range(i+1, valid_k):
            common = sm[i] & sm[j]
            pairwise.append({'state_i': i, 'state_j': j, 'ca_rmsd_between_generated_states_nm': kabsch_rmsd(pred_ca[i], pred_ca[j], common)})

    torch.save({'sample_id': sample.get('sample_id'), 'pred_seq': pred_seq, 'ref_seq': ref_seq, 'x_states': {k: v.detach().cpu() for k, v in x_states.items()}}, sample_out / 'state_specific_outputs.pt')
    summary = {
        'status': 'passed',
        'sample_index': sample_idx,
        'sample_id': sample.get('sample_id'),
        'split': sample.get('split'),
        'target_id': sample.get('target_id'),
        'nsteps': args.nsteps,
        'valid_states': valid_k,
        'pred_sequence': pred_seq,
        'reference_sequence': ref_seq,
        'sequence_identity_to_reference': seq_identity,
        'state_specific_output_shapes': {k: list(v.shape) for k, v in x_states.items()},
        'has_shared_seq_logits': 'seq_logits_shared' in nn_out,
        'has_state_seq_logits': 'state_seq_logits' in nn_out,
        'state_summaries': state_summaries,
        'generated_state_pairwise': pairwise,
        'out_dir': str(sample_out),
        'ckpt': str(args.ckpt),
        'dataset': str(args.dataset),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(tensor_to_json(summary), indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(tensor_to_json(summary), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
