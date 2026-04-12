import math

import torch
from torch import nn


class CrossStateFusion(nn.Module):
    """Fuse per-state residue features into a single residue representation.

    The module follows the planned hybrid design:
    1. residue-aligned cross-state self-attention
    2. gated residual fusion that mixes attention-pooled, weighted-mean and max features
    3. explicit state-weight and state-role conditioning
    """

    def __init__(
        self,
        dim: int,
        nheads: int = 8,
        role_vocab_size: int = 8,
        dropout: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert dim % nheads == 0, f"dim ({dim}) must be divisible by nheads ({nheads})"
        self.dim = dim
        self.nheads = nheads
        self.head_dim = dim // nheads
        self.scale = self.head_dim ** -0.5
        self.eps = eps

        self.role_embedding = nn.Embedding(role_vocab_size, dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def _normalize_state_weights(self, state_weights: torch.Tensor, state_present_mask: torch.Tensor) -> torch.Tensor:
        weights = state_weights.float() * state_present_mask.float()
        denom = weights.sum(dim=1, keepdim=True)
        fallback = state_present_mask.float()
        fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
        normalized = torch.where(denom > self.eps, weights / denom.clamp_min(self.eps), fallback)
        return normalized

    def forward(
        self,
        state_repr: torch.Tensor,
        residue_mask: torch.Tensor,
        state_weights: torch.Tensor | None = None,
        state_roles: torch.Tensor | None = None,
        state_present_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Fuse state-aware target representations.

        Args:
            state_repr: [B, K, L, D]
            residue_mask: [B, K, L]
            state_weights: [B, K]
            state_roles: [B, K]
            state_present_mask: [B, K]
        Returns:
            dict with fused residue features and debug tensors.
        """
        batch_size, nstates, nres, dim = state_repr.shape
        assert dim == self.dim

        device = state_repr.device
        if state_present_mask is None:
            state_present_mask = torch.ones(batch_size, nstates, dtype=torch.bool, device=device)
        if state_weights is None:
            state_weights = torch.ones(batch_size, nstates, dtype=state_repr.dtype, device=device)
        if state_roles is None:
            state_roles = torch.zeros(batch_size, nstates, dtype=torch.long, device=device)

        state_roles = state_roles.clamp_min(0)
        state_role_emb = self.role_embedding(state_roles)[:, :, None, :]  # [B, K, 1, D]

        valid_mask = residue_mask.bool() & state_present_mask[:, :, None].bool()  # [B, K, L]
        normed_state_weights = self._normalize_state_weights(state_weights, state_present_mask)  # [B, K]

        x = state_repr + state_role_emb
        x = x * valid_mask[..., None]
        x = x.transpose(1, 2)  # [B, L, K, D]
        valid_mask_rlk = valid_mask.transpose(1, 2)  # [B, L, K]

        q = self.q_proj(x).view(batch_size, nres, nstates, self.nheads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(x).view(batch_size, nres, nstates, self.nheads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = self.v_proj(x).view(batch_size, nres, nstates, self.nheads, self.head_dim).permute(0, 1, 3, 2, 4)

        attn_logits = torch.einsum("blhqd,blhkd->blhqk", q, k) * self.scale  # [B, L, H, K, K]
        key_weight_bias = torch.log(normed_state_weights.clamp_min(self.eps))[:, None, None, None, :]  # [B,1,1,1,K]
        attn_logits = attn_logits + key_weight_bias

        key_mask = valid_mask_rlk[:, :, None, None, :]  # [B, L, 1, 1, K]
        attn_logits = attn_logits.masked_fill(~key_mask, float("-inf"))
        attn = torch.softmax(attn_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.dropout(attn)

        attn_out = torch.einsum("blhqk,blhkd->blhqd", attn, v)  # [B, L, H, K, d]
        attn_out = attn_out.permute(0, 1, 3, 2, 4).reshape(batch_size, nres, nstates, dim)
        attn_out = self.out_proj(attn_out)
        attn_out = attn_out * valid_mask_rlk[..., None]

        state_weight_rlk = normed_state_weights[:, None, :].expand(-1, nres, -1) * valid_mask_rlk.float()  # [B,L,K]
        state_weight_rlk = state_weight_rlk / state_weight_rlk.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        weighted_mean = torch.sum(x * state_weight_rlk[..., None], dim=2)  # [B, L, D]
        attn_pooled = torch.sum(attn_out * state_weight_rlk[..., None], dim=2)  # [B, L, D]

        max_input = x.masked_fill(~valid_mask_rlk[..., None], -1e9)
        max_feat = max_input.max(dim=2).values
        has_any_valid = valid_mask_rlk.any(dim=2, keepdim=True)
        max_feat = torch.where(has_any_valid, max_feat, torch.zeros_like(max_feat))

        fusion_base = 0.5 * (attn_pooled + weighted_mean)
        gate_in = torch.cat([attn_pooled, weighted_mean, max_feat], dim=-1)
        gate = self.gate_proj(gate_in)
        fused = gate * fusion_base + (1.0 - gate) * max_feat
        fused = self.norm(fused + weighted_mean)

        attn_mean = attn.mean(dim=(2, 3))  # [B, L, K]
        return {
            "fused_residue_repr": fused,
            "attention_pooled_repr": attn_pooled,
            "weighted_mean_repr": weighted_mean,
            "max_repr": max_feat,
            "attention_mean": attn_mean,
            "normalized_state_weights": normed_state_weights,
        }


class GlobalResidueRefinement(nn.Module):
    """Lightweight global residue refinement after cross-state fusion."""

    def __init__(self, dim: int, nheads: int = 8, nlayers: int = 2, dropout: float = 0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nheads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(self, residue_repr: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~residue_mask.bool()
        refined = self.encoder(residue_repr, src_key_padding_mask=key_padding_mask)
        return refined * residue_mask[..., None]
