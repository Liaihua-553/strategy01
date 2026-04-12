import torch
from torch import nn

from proteinfoundation.nn.feature_factory.target_feats import TargetConcatSeqFeat
from proteinfoundation.nn.modules.cross_state_fusion import CrossStateFusion, GlobalResidueRefinement


class EnsembleTargetEncoder(nn.Module):
    """Encode a target conformational ensemble into a shared target memory.

    This module keeps the original single-state target feature extraction path
    and adds an explicit state axis followed by residue-aligned fusion.
    """

    def __init__(
        self,
        token_dim: int,
        cross_state_nheads: int = 8,
        cross_state_dropout: float = 0.0,
        global_refine_layers: int = 2,
        global_refine_nheads: int = 8,
        state_role_num_embeddings: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.single_state_feature_extractor = TargetConcatSeqFeat(**kwargs)
        self.single_state_feat_dim = self.single_state_feature_extractor.dim
        self.target_feature_projection = nn.Sequential(
            nn.LayerNorm(self.single_state_feat_dim),
            nn.Linear(self.single_state_feat_dim, token_dim),
        )
        self.cross_state_fusion = CrossStateFusion(
            dim=token_dim,
            nheads=cross_state_nheads,
            role_vocab_size=state_role_num_embeddings,
            dropout=cross_state_dropout,
        )
        self.global_refinement = GlobalResidueRefinement(
            dim=token_dim,
            nheads=global_refine_nheads,
            nlayers=global_refine_layers,
            dropout=cross_state_dropout,
        )
        self.state_summary_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def _extract_state_axis_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = [
            "x_target_states",
            "target_mask_states",
            "seq_target_states",
            "target_hotspot_mask_states",
        ]
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                "EnsembleTargetEncoder currently requires explicit state-axis target tensors. "
                f"Missing keys: {missing}"
            )

        x_target_states = batch["x_target_states"]
        target_mask_states = batch["target_mask_states"]
        seq_target_states = batch["seq_target_states"]
        hotspot_states = batch["target_hotspot_mask_states"]
        batch_size, nstates, nres = seq_target_states.shape
        device = x_target_states.device

        state_present_mask = batch.get(
            "state_present_mask",
            torch.ones(batch_size, nstates, dtype=torch.bool, device=device),
        )
        target_state_weights = batch.get(
            "target_state_weights",
            torch.ones(batch_size, nstates, dtype=x_target_states.dtype, device=device),
        )
        target_state_roles = batch.get(
            "target_state_roles",
            torch.zeros(batch_size, nstates, dtype=torch.long, device=device),
        )
        seq_target_mask_states = batch.get(
            "seq_target_mask_states",
            target_mask_states.any(dim=-1),
        ).bool()

        return {
            "x_target_states": x_target_states,
            "target_mask_states": target_mask_states,
            "seq_target_states": seq_target_states,
            "target_hotspot_mask_states": hotspot_states,
            "seq_target_mask_states": seq_target_mask_states,
            "target_state_weights": target_state_weights,
            "target_state_roles": target_state_roles.long(),
            "state_present_mask": state_present_mask.bool(),
        }

    def _encode_single_state_targets(self, state_batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x_target_states = state_batch["x_target_states"]
        target_mask_states = state_batch["target_mask_states"]
        seq_target_states = state_batch["seq_target_states"]
        hotspot_states = state_batch["target_hotspot_mask_states"]
        seq_target_mask_states = state_batch["seq_target_mask_states"]

        batch_size, nstates, nres = seq_target_states.shape
        flat_batch = {
            "x_target": x_target_states.reshape(batch_size * nstates, nres, 37, 3),
            "target_mask": target_mask_states.reshape(batch_size * nstates, nres, 37),
            "seq_target": seq_target_states.reshape(batch_size * nstates, nres),
            "target_hotspot_mask": hotspot_states.reshape(batch_size * nstates, nres),
            "seq_target_mask": seq_target_mask_states.reshape(batch_size * nstates, nres),
            "mask": seq_target_mask_states.reshape(batch_size * nstates, nres),
        }
        target_feats, target_mask = self.single_state_feature_extractor(flat_batch)
        target_feats = self.target_feature_projection(target_feats) * target_mask[..., None]
        target_feats = target_feats.reshape(batch_size, nstates, nres, self.token_dim)
        target_mask = target_mask.reshape(batch_size, nstates, nres)
        return target_feats, target_mask.bool()

    def _build_state_summary(self, token_states: torch.Tensor, state_residue_mask: torch.Tensor) -> torch.Tensor:
        mask = state_residue_mask.float()
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        state_summary = (token_states * mask[..., None]).sum(dim=2) / denom
        return self.state_summary_projection(state_summary)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state_batch = self._extract_state_axis_batch(batch)
        token_states, state_feature_mask = self._encode_single_state_targets(state_batch)

        state_present_mask = state_batch["state_present_mask"]
        target_state_weights = state_batch["target_state_weights"]
        target_state_roles = state_batch["target_state_roles"]
        seq_target_mask_states = state_batch["seq_target_mask_states"]
        state_residue_mask = state_feature_mask & seq_target_mask_states & state_present_mask[:, :, None]

        fusion = self.cross_state_fusion(
            token_states,
            residue_mask=state_residue_mask,
            state_weights=target_state_weights,
            state_roles=target_state_roles,
            state_present_mask=state_present_mask,
        )

        ensemble_target_mask = state_residue_mask.any(dim=1)
        ensemble_target_memory = self.global_refinement(fusion["fused_residue_repr"], ensemble_target_mask)
        state_summary_tokens = self._build_state_summary(token_states, state_residue_mask)

        return {
            "state_target_tokens": token_states,
            "state_target_mask": state_residue_mask,
            "ensemble_target_memory": ensemble_target_memory,
            "ensemble_target_mask": ensemble_target_mask,
            "state_summary_tokens": state_summary_tokens,
            "state_present_mask": state_present_mask,
            "target_state_weights": fusion["normalized_state_weights"],
            "target_state_roles": target_state_roles,
            "cross_state_attention_mean": fusion["attention_mean"],
            "attention_pooled_repr": fusion["attention_pooled_repr"],
            "weighted_mean_repr": fusion["weighted_mean_repr"],
            "max_repr": fusion["max_repr"],
        }
