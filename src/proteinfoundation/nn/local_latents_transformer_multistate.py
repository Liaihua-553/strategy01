import torch
from torch import nn

from proteinfoundation.nn.ensemble_target_encoder import EnsembleTargetEncoder
from proteinfoundation.nn.feature_factory.feature_factory import FeatureFactory
from proteinfoundation.nn.modules.attn_n_transition import MultiheadAttnAndTransition, MultiheadCrossAttnAndTransition
from proteinfoundation.nn.modules.pair_update import PairReprUpdate
from proteinfoundation.nn.modules.seq_transition_af3 import Transition
from proteinfoundation.nn.protein_transformer import PairReprBuilder


class LocalLatentsTransformerMultistate(nn.Module):
    """Multistate binder generator with one shared sequence head and K state-specific structure heads."""

    def __init__(self, **kwargs):
        super().__init__()
        self.nlayers = kwargs["nlayers"]
        self.token_dim = kwargs["token_dim"]
        self.pair_repr_dim = kwargs["pair_repr_dim"]
        self.update_pair_repr = kwargs["update_pair_repr"]
        self.update_pair_repr_every_n = kwargs["update_pair_repr_every_n"]
        self.use_tri_mult = kwargs["use_tri_mult"]
        self.use_tri_attn = kwargs.get("use_tri_attn", False)
        self.use_qkln = kwargs["use_qkln"]
        self.output_param = kwargs["output_parameterization"]
        self.latent_dim = kwargs["latent_dim"]

        self.init_repr_factory = FeatureFactory(
            feats=kwargs["feats_seq"],
            dim_feats_out=kwargs["token_dim"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )
        self.cond_factory = FeatureFactory(
            feats=kwargs["feats_cond_seq"],
            dim_feats_out=kwargs["dim_cond"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )
        self.transition_c_1 = Transition(kwargs["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["dim_cond"], expansion_factor=2)

        # MODIFIED 2026-04-12: avoid passing duplicated keyword arguments when bootstrapping the
        # ensemble encoder from the baseline-compatible nn config.
        # ORIGINAL:
        # self.ensemble_target_encoder = EnsembleTargetEncoder(token_dim=kwargs["token_dim"], ..., **kwargs)
        self.ensemble_target_encoder = EnsembleTargetEncoder(**kwargs)
        self.target2binder_cross_attention_layer = nn.ModuleList(
            [
                MultiheadCrossAttnAndTransition(
                    dim_token_a=kwargs["token_dim"],
                    dim_token_b=kwargs["token_dim"],
                    nheads=kwargs["nheads"],
                    dim_cond=kwargs["dim_cond"],
                    residual_mha=True,
                    residual_transition=True,
                    use_qkln=self.use_qkln,
                )
                for _ in range(self.nlayers)
            ]
        )

        self.pair_repr_builder = PairReprBuilder(
            feats_repr=kwargs["feats_pair_repr"],
            feats_cond=kwargs["feats_pair_cond"],
            dim_feats_out=kwargs["pair_repr_dim"],
            dim_cond_pair=kwargs["dim_cond"],
            **kwargs,
        )
        self.transformer_layers = nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=self.token_dim,
                    dim_pair=self.pair_repr_dim,
                    nheads=kwargs["nheads"],
                    dim_cond=kwargs["dim_cond"],
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=True,
                    use_qkln=self.use_qkln,
                )
                for _ in range(self.nlayers)
            ]
        )

        if self.update_pair_repr:
            self.pair_update_layers = nn.ModuleList(
                [
                    (
                        PairReprUpdate(
                            token_dim=kwargs["token_dim"],
                            pair_dim=kwargs["pair_repr_dim"],
                            use_tri_mult=self.use_tri_mult,
                            use_tri_attn=self.use_tri_attn,
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )

        self.local_latents_linear = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, kwargs["latent_dim"], bias=False),
        )
        self.ca_linear = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 3, bias=False),
        )
        self.shared_seq_head = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 20, bias=False),
        )
        # MODIFIED 2026-04-25 Stage07:
        # Each state-specific binder trajectory now proposes its own sequence
        # preference.  The final shared sequence head is a robust consensus of
        # the baseline shared logits and these K state-conditioned logits, so
        # difficult states can influence the one exported binder sequence.
        self.state_seq_head = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 20, bias=False),
        )
        self.shared_seq_consensus_gate = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 1),
            nn.Sigmoid(),
        )
        # MODIFIED 2026-05-06 Stage11:
        # Feed the model's own robust shared sequence distribution back into
        # the state-specific structure/latent heads.  The input is a soft AA
        # distribution from model logits, not the ground-truth sequence, so this
        # keeps sampling-time behavior aligned with training and avoids label
        # leakage.
        self.soft_sequence_feedback_projector = nn.Sequential(
            nn.LayerNorm(20),
            nn.Linear(20, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.flow_gate_head = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 1),
            nn.Sigmoid(),
        )
        self.state_condition_projector = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.state_seq_condition_projector = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.state_token_norm = nn.LayerNorm(self.token_dim)
        # MODIFIED 2026-05-05 Stage10:
        # Optional transferred source-pose initialization is encoded as a
        # state-specific token bias.  This lets B1 refine plausible source-pose
        # transfers instead of generating each state pose from a weak averaged
        # representation.
        self.init_pose_projector = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        # MODIFIED 2026-04-19: Stage04 adds a light quality-proxy head.
        # It learns offline complex-confidence labels such as iPAE/pLDDT/ipTM
        # proxies without calling an external predictor during backpropagation.
        self.interface_quality_head = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 5, bias=False),
        )

    def forward(self, input: dict) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        mask = input["mask"]
        c = self.cond_factory(input)
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)

        seq_f_repr = self.init_repr_factory(input)
        seqs = seq_f_repr * mask[..., None]
        pair_rep = self.pair_repr_builder(input)

        target_bundle = self.ensemble_target_encoder(input)
        target_rep = target_bundle["ensemble_target_memory"]
        target_mask = target_bundle["ensemble_target_mask"]

        for i in range(self.nlayers):
            seqs = self.target2binder_cross_attention_layer[i](seqs, target_rep, c, mask, target_mask)
            seqs = self.transformer_layers[i](seqs, pair_rep, c, mask)
            if self.update_pair_repr and i < self.nlayers - 1 and self.pair_update_layers[i] is not None:
                pair_rep = self.pair_update_layers[i](seqs, pair_rep, mask)

        base_shared_seq_logits = self.shared_seq_head(seqs) * mask[..., None]

        batch_size, nbinder, _ = seqs.shape
        nstates = target_bundle["state_summary_tokens"].shape[1]
        state_bias = self.state_condition_projector(target_bundle["state_summary_tokens"])[:, :, None, :]
        state_tokens = self.state_token_norm(seqs[:, None, :, :] + state_bias)
        state_seq_bias = self.state_seq_condition_projector(target_bundle["state_summary_tokens"])[:, :, None, :]
        state_seq_tokens = self.state_token_norm(seqs[:, None, :, :] + state_seq_bias)
        init_pose = input.get("init_bb_ca_states")
        if init_pose is not None:
            init_pose = init_pose.to(device=seqs.device, dtype=seqs.dtype)
            if init_pose.shape[:3] != (batch_size, nstates, nbinder):
                raise ValueError(
                    f"init_bb_ca_states shape {tuple(init_pose.shape)} is incompatible with "
                    f"(B,K,Nb)=({batch_size},{nstates},{nbinder})"
                )
            init_tokens = self.init_pose_projector(init_pose)
            init_tokens = init_tokens * mask[:, None, :, None] * target_bundle["state_present_mask"][:, :, None, None]
            state_tokens = self.state_token_norm(state_tokens + init_tokens)
            state_seq_tokens = self.state_token_norm(state_seq_tokens + 0.5 * init_tokens)
        flat_state_tokens = state_tokens.reshape(batch_size * nstates, nbinder, self.token_dim)
        flat_state_seq_tokens = state_seq_tokens.reshape(batch_size * nstates, nbinder, self.token_dim)
        flat_state_mask = mask[:, None, :].expand(-1, nstates, -1).reshape(batch_size * nstates, nbinder)

        state_weights = target_bundle["target_state_weights"].float() * target_bundle["state_present_mask"].float()
        state_weights = state_weights / state_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        state_binder_summary = (state_tokens * mask[:, None, :, None]).sum(dim=2) / mask.float().sum(dim=1).clamp_min(1.0)[:, None, None]
        interface_quality_logits = self.interface_quality_head(
            state_binder_summary + target_bundle["state_summary_tokens"]
        ) * target_bundle["state_present_mask"][:, :, None]

        # The quality proxy is intentionally detached for consensus weighting:
        # sequence gradients flow through state_seq_logits/state_tokens, while
        # the quality head is trained by its own offline labels.
        state_quality = torch.sigmoid(interface_quality_logits.detach()).mean(dim=-1)
        state_uncertainty = (1.0 - state_quality) * target_bundle["state_present_mask"].float()
        robust_state_weights = state_weights * (1.0 + 0.5 * state_uncertainty)
        robust_state_weights = robust_state_weights / robust_state_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        # First sequence readout: estimate a shared soft sequence and feed it
        # back into state-conditioned tokens before final structure/latent
        # readout.  This is the Stage11 Complexa-native coupling: K state
        # trajectories jointly form one sequence signal, and that signal then
        # influences the local latent flow that the frozen AE decoder reads.
        flat_state_seq_logits_pre = self.state_seq_head(flat_state_seq_tokens) * flat_state_mask[..., None]
        state_seq_logits_pre = flat_state_seq_logits_pre.reshape(batch_size, nstates, nbinder, 20)
        state_present_mask = target_bundle["state_present_mask"][:, :, None, None]
        state_seq_logits_pre = state_seq_logits_pre * state_present_mask
        consensus_logits_pre = (state_seq_logits_pre * robust_state_weights[:, :, None, None]).sum(dim=1)
        consensus_gate_pre = self.shared_seq_consensus_gate(seqs)
        shared_seq_logits_pre = (base_shared_seq_logits + consensus_gate_pre * consensus_logits_pre) * mask[..., None]
        shared_seq_soft = torch.softmax(shared_seq_logits_pre.float(), dim=-1).to(dtype=seqs.dtype) * mask[..., None]
        seq_feedback_tokens = self.soft_sequence_feedback_projector(shared_seq_soft) * mask[..., None]
        state_tokens = self.state_token_norm(state_tokens + 0.25 * seq_feedback_tokens[:, None, :, :])
        state_seq_tokens = self.state_token_norm(state_seq_tokens + 0.50 * seq_feedback_tokens[:, None, :, :])

        flat_state_tokens = state_tokens.reshape(batch_size * nstates, nbinder, self.token_dim)
        flat_state_seq_tokens = state_seq_tokens.reshape(batch_size * nstates, nbinder, self.token_dim)
        flat_local_latents = self.local_latents_linear(flat_state_tokens) * flat_state_mask[..., None]
        flat_bb_ca = self.ca_linear(flat_state_tokens) * flat_state_mask[..., None]
        flat_state_seq_logits = self.state_seq_head(flat_state_seq_tokens) * flat_state_mask[..., None]
        flat_flow_gate = self.flow_gate_head(flat_state_tokens) * flat_state_mask[..., None]
        local_latents_states = flat_local_latents.reshape(batch_size, nstates, nbinder, self.latent_dim)
        bb_ca_states = flat_bb_ca.reshape(batch_size, nstates, nbinder, 3)
        state_seq_logits = flat_state_seq_logits.reshape(batch_size, nstates, nbinder, 20)
        flow_gate = flat_flow_gate.reshape(batch_size, nstates, nbinder, 1)

        local_latents_states = local_latents_states * state_present_mask
        bb_ca_states = bb_ca_states * state_present_mask
        state_seq_logits = state_seq_logits * state_present_mask
        flow_gate = flow_gate * state_present_mask

        state_weight_view = state_weights[:, :, None, None]
        shared_local_latents = (local_latents_states * state_weight_view).sum(dim=1) * mask[..., None]
        shared_bb_ca = (bb_ca_states * state_weight_view).sum(dim=1) * mask[..., None]
        consensus_logits = (state_seq_logits * robust_state_weights[:, :, None, None]).sum(dim=1)
        consensus_gate = self.shared_seq_consensus_gate(seqs)
        shared_seq_logits = (base_shared_seq_logits + consensus_gate * consensus_logits) * mask[..., None]

        return {
            "bb_ca": {self.output_param["bb_ca"]: shared_bb_ca},
            "local_latents": {self.output_param["local_latents"]: shared_local_latents},
            "seq_logits_shared": shared_seq_logits,
            "seq_logits_base_shared": base_shared_seq_logits,
            "state_seq_logits": state_seq_logits,
            "bb_ca_states": bb_ca_states,
            "local_latents_states": local_latents_states,
            "flow_gate": flow_gate,
            "seq_feedback_tokens": seq_feedback_tokens,
            "seq_logits_pre_feedback": shared_seq_logits_pre,
            "interface_quality_logits": interface_quality_logits,
            "arch_debug": {
                "ensemble_target_memory": target_bundle["ensemble_target_memory"],
                "ensemble_target_mask": target_bundle["ensemble_target_mask"],
                "state_summary_tokens": target_bundle["state_summary_tokens"],
                "state_target_tokens": target_bundle["state_target_tokens"],
                "state_target_mask": target_bundle["state_target_mask"],
                "target_state_weights": target_bundle["target_state_weights"],
                "target_state_roles": target_bundle["target_state_roles"],
                "sampling_state_weights": state_weights,
                "sequence_consensus_state_weights": robust_state_weights,
                "sequence_consensus_gate_mean": consensus_gate.detach().mean(),
                "cross_state_attention_mean": target_bundle["cross_state_attention_mean"],
                "sequence_feedback_norm_mean": seq_feedback_tokens.detach().norm(dim=-1).mean(),
                "flow_gate_mean": flow_gate.detach().mean(),
                "flow_gate_std": flow_gate.detach().std(),
            },
        }
