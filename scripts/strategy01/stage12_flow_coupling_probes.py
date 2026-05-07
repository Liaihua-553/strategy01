#!/usr/bin/env python3
"""Stage12 coupled de-novo multistate flow probes."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch
from omegaconf import OmegaConf

from proteinfoundation.flow_matching.multistate_loss import corrupt_multistate_batch
from proteinfoundation.nn.local_latents_transformer_multistate import LocalLatentsTransformerMultistate
from scripts.strategy01.probe_multistate_arch import (
    load_checkpoint_artifacts,
    make_binder_stub_batch,
    make_synthetic_target_states,
    summarize_forward,
    to_jsonable,
)

DEFAULT_CONFIG = REPO / "configs/training_local_latents_multistate_ckpt_align_probe.yaml"
DEFAULT_REPORT = REPO / "reports/strategy01/probes/stage12_flow_coupling_probe.json"


class _DummyBaseFlow:
    def __init__(self, dim: int):
        self.dim = dim

    def sample_noise(self, n, shape, device, mask, training=False):
        return torch.randn(*shape, n, self.dim, device=device) * mask[..., None].float()

    def interpolate(self, x_0, x_1, t, mask):
        while t.ndim < x_1.ndim:
            t = t.unsqueeze(-1)
        return (1.0 - t) * x_0 + t * x_1


class _DummyFlowMatcher:
    training = True
    data_modes = ("bb_ca", "local_latents")

    def __init__(self, latent_dim: int):
        self.base_flow_matchers = {"bb_ca": _DummyBaseFlow(3), "local_latents": _DummyBaseFlow(latent_dim)}
        self.cfg_exp = OmegaConf.create({"loss": {"multistate": {"de_novo_multistate": True}}})

    def sample_t(self, shape, device):
        return {dm: torch.full(shape, 0.5, device=device) for dm in self.data_modes}


def _permute_state_batch(batch: dict, perm: torch.Tensor) -> dict:
    state_keys = {
        "x_target_states",
        "target_mask_states",
        "seq_target_states",
        "target_hotspot_mask_states",
        "seq_target_mask_states",
        "target_state_weights",
        "target_state_roles",
        "state_present_mask",
    }
    out = {}
    for key, value in batch.items():
        out[key] = value.index_select(1, perm) if key in state_keys and isinstance(value, torch.Tensor) else value
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage12 CPU architecture probes")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--binder-len", type=int, default=32)
    parser.add_argument("--target-len", type=int, default=64)
    parser.add_argument("--nstates", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device("cpu")
    torch.manual_seed(1207)

    _, checkpoint_nn_cfg, latent_dim = load_checkpoint_artifacts(cfg.baseline_ckpt_path)
    nn_cfg = OmegaConf.to_container(OmegaConf.load(cfg.nn_config_path), resolve=True)
    nn_cfg["feats_seq"] = deepcopy(checkpoint_nn_cfg["feats_seq"])
    nn_cfg["feats_pair_repr"] = deepcopy(checkpoint_nn_cfg["feats_pair_repr"])
    model = LocalLatentsTransformerMultistate(**nn_cfg, latent_dim=latent_dim).to(device)
    model.eval()

    batch = {
        **make_binder_stub_batch(args.batch_size, args.binder_len, latent_dim, device),
        **make_synthetic_target_states(args.batch_size, args.nstates, args.target_len, device),
        "de_novo_multistate_mode": True,
    }
    with torch.no_grad():
        out = model(batch)

    assert list(out["bb_ca_states"].shape) == [args.batch_size, args.nstates, args.binder_len, 3]
    assert list(out["local_latents_states"].shape) == [args.batch_size, args.nstates, args.binder_len, latent_dim]
    assert list(out["seq_logits_shared"].shape) == [args.batch_size, args.binder_len, 20]
    assert list(out["shared_seq_tokens"].shape) == [args.batch_size, args.binder_len, model.token_dim]
    assert float(out["arch_debug"]["stage12_shared_sequence_token_norm_mean"]) > 0.0

    leaked_batch = dict(batch)
    leaked_batch["init_bb_ca_states"] = torch.zeros(args.batch_size, args.nstates, args.binder_len, 3)
    expected_no_leak_guard_passed = False
    try:
        model(leaked_batch)
    except ValueError as exc:
        expected_no_leak_guard_passed = "de_novo_multistate" in str(exc)
    assert expected_no_leak_guard_passed, "de_novo_multistate no-source guard did not fire"

    repair_batch = dict(batch)
    repair_batch["de_novo_multistate_mode"] = False
    repair_batch["init_bb_ca_states"] = torch.zeros(args.batch_size, args.nstates, args.binder_len, 3)
    with torch.no_grad():
        repair_out = model(repair_batch)
    assert list(repair_out["bb_ca_states"].shape) == [args.batch_size, args.nstates, args.binder_len, 3]

    perm = torch.tensor(list(reversed(range(args.nstates))), dtype=torch.long)
    with torch.no_grad():
        out_perm = model(_permute_state_batch(batch, perm))
    perm_delta = (out["seq_logits_shared"] - out_perm["seq_logits_shared"]).abs().max().item()
    assert perm_delta < 1.0e-4, f"shared sequence logits are not state-order invariant enough: {perm_delta}"

    x1_states = {
        "bb_ca": torch.randn(args.batch_size, args.nstates, args.binder_len, 3),
        "local_latents": torch.randn(args.batch_size, args.nstates, args.binder_len, latent_dim),
    }
    loss_batch = {
        **batch,
        "x_1_states": x1_states,
        "binder_seq_shared": torch.randint(0, 20, (args.batch_size, args.binder_len)),
        "binder_seq_mask": batch["mask"],
    }
    corrupt_multistate_batch(_DummyFlowMatcher(latent_dim), loss_batch)
    assert loss_batch["stage12_primary_state_tensors"] is True
    assert list(loss_batch["x_t_states"]["bb_ca"].shape) == [args.batch_size, args.nstates, args.binder_len, 3]
    assert list(loss_batch["x_t_states"]["local_latents"].shape) == [
        args.batch_size,
        args.nstates,
        args.binder_len,
        latent_dim,
    ]

    results = {
        "stage": "stage12_flow_coupling_probe",
        "status": "passed",
        "forward_shapes": summarize_forward(out),
        "state_permutation_max_abs_delta": perm_delta,
        "expected_no_leak_guard_passed": expected_no_leak_guard_passed,
        "repair_mode_allows_init_pose": True,
        "corrupt_multistate": {
            "stage12_primary_state_tensors": bool(loss_batch["stage12_primary_state_tensors"]),
            "bb_ca_state_delta_mean": float(
                (loss_batch["x_t_states"]["bb_ca"][:, 0] - loss_batch["x_t_states"]["bb_ca"][:, 1]).abs().mean()
            ),
            "local_latents_state_delta_mean": float(
                (loss_batch["x_t_states"]["local_latents"][:, 0] - loss_batch["x_t_states"]["local_latents"][:, 1])
                .abs()
                .mean()
            ),
        },
    }
    report = Path(args.report_json)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(to_jsonable(results), indent=2), encoding="utf-8")
    print(json.dumps(to_jsonable(results), indent=2))


if __name__ == "__main__":
    main()
