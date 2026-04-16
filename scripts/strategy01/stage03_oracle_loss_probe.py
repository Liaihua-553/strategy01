import json
import sys
from pathlib import Path

import torch

REPO = Path("/data/kfliao/general_model/strategy_repos/Strategy01_complexa_multistate_benchmarkbase")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "strategy01"))

from stage03_multistate_loss_debug import build_debug_samples, build_model, collate_samples, ensure_stage_ckpt_copies


def main():
    ensure_stage_ckpt_copies()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, fm, meta = build_model(device)
    samples, _ = build_debug_samples(2, 24, 48, int(meta["latent_dim"]))
    batch = collate_samples(samples[:2], device)
    torch.manual_seed(777)
    batch = fm.corrupt_multistate_batch(batch)
    b, k, n = batch["x_1_states"]["bb_ca"].shape[:3]
    nn_out = {}
    for dm, out_key in [("bb_ca", "bb_ca_states"), ("local_latents", "local_latents_states")]:
        t = batch["t_states"][dm][:, :, None, None]
        nn_out[out_key] = (batch["x_1_states"][dm] - batch["x_t_states"][dm]) / (1.0 - t + 1e-5)
    logits = torch.full((b, n, 20), -20.0, device=device)
    logits.scatter_(2, batch["binder_seq_shared"].long().unsqueeze(-1), 20.0)
    nn_out["seq_logits_shared"] = logits
    losses = fm.compute_multistate_loss(batch, nn_out)
    result = {k: float(v.detach().mean().cpu()) for k, v in losses.items()}
    out = REPO / "reports" / "strategy01" / "probes" / "stage03_oracle_loss_probe.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
