#!/usr/bin/env python
"""Convert a Stage07 NN-only mini checkpoint into a minimal Lightning checkpoint.

The Stage07 fine-tune script saves only model.nn.state_dict() to keep pilot
artifacts small. proteinfoundation.generate expects Proteina.load_from_checkpoint,
which needs Lightning hyper_parameters and a state_dict with `nn.` prefixes.
This converter uses the stage baseline checkpoint only as a metadata template,
replaces cfg_exp.nn with the multistate architecture config, and writes a
minimal inference checkpoint without optimizer states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO = Path('/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase')
DEFAULT_BASE = REPO / 'ckpts' / 'stage03_multistate_loss' / 'complexa_init_readonly_copy.ckpt'
DEFAULT_MINI = REPO / 'ckpts' / 'stage07_sequence_consensus' / 'runs' / 'stage07_v23_wave1_pilot_structfix5' / 'mini_final.pt'
DEFAULT_AE = REPO / 'ckpts' / 'stage03_multistate_loss' / 'complexa_ae_init_readonly_copy.ckpt'
DEFAULT_NN = REPO / 'configs' / 'nn' / 'local_latents_score_nn_160M_multistate.yaml'
DEFAULT_OUT = REPO / 'ckpts' / 'stage07_sequence_consensus' / 'runs' / 'stage07_v23_wave1_pilot_structfix5' / 'mini_final_lightning.ckpt'
DEFAULT_REPORT = REPO / 'reports' / 'strategy01' / 'probes' / 'stage07_mini_lightning_ckpt_conversion_summary.json'


def jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-ckpt', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--mini-pt', type=Path, default=DEFAULT_MINI)
    parser.add_argument('--autoencoder-ckpt', type=Path, default=DEFAULT_AE)
    parser.add_argument('--nn-config', type=Path, default=DEFAULT_NN)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    for path in [args.base_ckpt, args.mini_pt, args.autoencoder_ckpt, args.nn_config]:
        if not path.exists():
            raise FileNotFoundError(path)

    base = torch.load(args.base_ckpt, map_location='cpu', weights_only=False)
    mini = torch.load(args.mini_pt, map_location='cpu', weights_only=False)
    mini_state = mini.get('model_state_dict')
    if not isinstance(mini_state, dict):
        raise ValueError(f'{args.mini_pt} does not contain model_state_dict')

    cfg_exp = base['hyper_parameters']['cfg_exp'].copy()
    OmegaConf.set_struct(cfg_exp, False)
    nn_cfg = OmegaConf.load(args.nn_config)
    cfg_exp.nn = nn_cfg
    cfg_exp.autoencoder_ckpt_path = str(args.autoencoder_ckpt)
    if 'product_flowmatcher' in cfg_exp and 'local_latents' in cfg_exp.product_flowmatcher:
        cfg_exp.product_flowmatcher.local_latents.dim = int(mini_state['local_latents_linear.1.weight'].shape[0])
    OmegaConf.set_struct(cfg_exp, True)

    state_dict = {f'nn.{k}': v.detach().cpu() for k, v in mini_state.items()}
    ckpt = {
        'state_dict': state_dict,
        'hyper_parameters': {
            'cfg_exp': cfg_exp,
            'store_dir': base.get('hyper_parameters', {}).get('store_dir', './tmp'),
            'autoencoder_ckpt_path': str(args.autoencoder_ckpt),
        },
        'pytorch-lightning_version': base.get('pytorch-lightning_version', '2.5.0'),
        'epoch': int(base.get('epoch', 0)),
        'global_step': int(base.get('global_step', 0)),
        'nflops': int(base.get('nflops', 0)),
        'nsamples_processed': int(base.get('nsamples_processed', 0)),
        'stage07_source': {
            'base_ckpt': str(args.base_ckpt),
            'mini_pt': str(args.mini_pt),
            'nn_config': str(args.nn_config),
            'phase': mini.get('phase'),
            'steps': mini.get('steps'),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output)
    report = {
        'status': 'passed',
        'output': str(args.output),
        'output_size_bytes': args.output.stat().st_size,
        'nn_keys_written': len(state_dict),
        'cfg_nn_name': str(cfg_exp.nn.name),
        'latent_dim': int(cfg_exp.product_flowmatcher.local_latents.dim),
        'autoencoder_ckpt_path': str(args.autoencoder_ckpt),
        'source_phase': mini.get('phase'),
        'source_steps': mini.get('steps'),
        'first_keys': list(state_dict.keys())[:20],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
