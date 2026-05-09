import torch
from torch import nn

from proteinfoundation.nn.ensemble_target_encoder import EnsembleTargetEncoder
from proteinfoundation.nn.feature_factory.concat_feature_factory import ConcatFeaturesFactory
from proteinfoundation.nn.feature_factory.concat_pair_feature_factory import ConcatPairFeaturesFactory
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
        # Stage13: preserve the original Complexa target-concat denoiser path.
        # Stage12's ensemble cross-attention path is useful for multistate
        # research, but it bypasses the checkpoint-trained concat target
        # placement prior.  The native-state path below reuses these modules
        # with their original parameter names so `complexa.ckpt` can initialize
        # single-state target-conditioned de-novo generation.
        concat_config = kwargs.get("concat_features", {})
        self.use_concat = (
            concat_config.get("enable_motif", False)
            or concat_config.get("enable_target", False)
            or concat_config.get("enable_ligand", False)
        )
        if self.use_concat:
            self.concat_factory = ConcatFeaturesFactory(
                enable_motif=concat_config.get("enable_motif", False),
                enable_target=concat_config.get("enable_target", False),
                enable_ligand=concat_config.get("enable_ligand", False),
                dim_feats_out=kwargs["token_dim"],
                use_ln_out=False,
                **kwargs,
            )
            self.use_advanced_pair = (
                (concat_config.get("enable_motif", False) and concat_config.get("motif_pair_features", False))
                or (concat_config.get("enable_target", False) and concat_config.get("target_pair_features", False))
                or (concat_config.get("enable_ligand", False) and concat_config.get("ligand_pair_features", False))
            )
            if self.use_advanced_pair:
                self.concat_pair_factory = ConcatPairFeaturesFactory(
                    enable_motif=concat_config.get("enable_motif", False),
                    enable_target=concat_config.get("enable_target", False),
                    enable_ligand=concat_config.get("enable_ligand", False),
                    **kwargs,
                )
        else:
            self.concat_factory = None
            self.use_advanced_pair = False
        self.use_target_cross_attn = kwargs.get("use_target_cross_attn", False)
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
        # MODIFIED 2026-05-07 Stage12:
        # A single AE decoder does not by itself force K state-specific latents
        # to decode to one sequence. Stage12 therefore adds an explicit shared
        # sequence token path: state-specific binder tokens for the same residue
        # attend across conformational states, produce one shared sequence
        # representation, and feed that representation back into every state's
        # bb_ca/local-latent flow heads.
        self.cross_state_shared_seq_attention = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=kwargs["nheads"],
            dropout=kwargs.get("dropout", 0.0),
            batch_first=True,
        )
        self.shared_seq_token_norm = nn.LayerNorm(self.token_dim)
        self.shared_seq_token_update = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.shared_seq_to_state_projector = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.shared_seq_token_head = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 20, bias=False),
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
        # MODIFIED 2026-05-08 Stage12D:
        # The Stage12C short-rollout probes showed that sampled local latents
        # drift away from the frozen AE sequence manifold.  This light residual
        # repair path lets the shared sequence signal directly adjust the
        # state-specific local latent readout, instead of hoping the final
        # sequence head alone can fix an off-manifold z.
        self.latent_sequence_repair_projector = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.latent_dim, bias=False),
        )
        self.latent_sequence_repair_gate = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, 1),
            nn.Sigmoid(),
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
        # MODIFIED 2026-05-08 Stage12B:
        # The coupled de-novo multistate model must condition each state's
        # denoising head on that state's current flow variables, not only on a
        # shared binder token plus state bias. These projectors inject
        # x_t_states['bb_ca'] and x_t_states['local_latents'] into the
        # state-specific tokens. They are generated/noisy flow variables, not
        # source poses or true labels, so they preserve the target-only
        # de-novo contract while making K state trajectories first-class.
        self.state_xt_bb_ca_projector = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.state_xt_latent_projector = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.token_dim),
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

    def _native_state_flatten_input(self, input: dict) -> tuple[dict, int, int, int]:
        """Flatten [B,K] state tensors into the original Complexa single-state schema."""
        mask = input["mask"]
        b, nbinder = mask.shape
        if "state_present_mask" not in input:
            raise ValueError("stage13 native-state path requires state_present_mask")
        k = input["state_present_mask"].shape[1]
        device = mask.device
        flat: dict = dict(input)
        flat["mask"] = mask[:, None, :].expand(b, k, nbinder).reshape(b * k, nbinder)
        if "x_t_states" not in input or "t_states" not in input:
            raise ValueError("stage13 native-state path requires x_t_states and t_states")
        flat["x_t"] = {
            dm: value.reshape(b * k, nbinder, value.shape[-1])
            for dm, value in input["x_t_states"].items()
        }
        flat["t"] = {
            dm: value.reshape(b * k)
            for dm, value in input["t_states"].items()
        }
        flat["x_sc"] = {dm: torch.zeros_like(value) for dm, value in flat["x_t"].items()}

        if "x_target_states" not in input or "target_mask_states" not in input or "seq_target_states" not in input:
            raise ValueError("stage13 native-state path requires explicit target state tensors")
        ntarget = input["x_target_states"].shape[2]
        flat["x_target"] = input["x_target_states"].reshape(b * k, ntarget, 37, 3)
        flat["target_mask"] = input["target_mask_states"].reshape(b * k, ntarget, 37)
        flat["seq_target"] = input["seq_target_states"].reshape(b * k, ntarget)
        flat["seq_target_mask"] = flat["target_mask"][..., 1].bool()
        if "target_hotspot_mask_states" in input:
            flat["target_hotspot_mask"] = input["target_hotspot_mask_states"].reshape(b * k, ntarget).bool()
        else:
            flat["target_hotspot_mask"] = torch.zeros(b * k, ntarget, dtype=torch.bool, device=device)
        target_idx = torch.arange(1, ntarget + 1, device=device, dtype=torch.float32).expand(b * k, -1)
        flat["target_pdb_idx"] = target_idx
        flat["target_chains"] = torch.zeros(b * k, ntarget, device=device, dtype=torch.long)
        flat.setdefault("residue_pdb_idx", torch.arange(1, nbinder + 1, device=device, dtype=torch.float32).expand(b * k, -1))
        if flat["residue_pdb_idx"].shape[0] == b:
            flat["residue_pdb_idx"] = flat["residue_pdb_idx"][:, None, :].expand(b, k, nbinder).reshape(b * k, nbinder)
        flat.setdefault("chains", torch.zeros(b * k, nbinder, device=device, dtype=torch.long))
        if flat["chains"].shape[0] == b:
            flat["chains"] = flat["chains"][:, None, :].expand(b, k, nbinder).reshape(b * k, nbinder)
        flat.setdefault("hotspot_mask", torch.zeros(b * k, nbinder, dtype=torch.bool, device=device))
        if flat["hotspot_mask"].shape[0] == b:
            flat["hotspot_mask"] = flat["hotspot_mask"][:, None, :].expand(b, k, nbinder).reshape(b * k, nbinder)
        return flat, b, k, nbinder

    def _native_single_state_forward(self, flat: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the original Complexa target-concat denoiser on flattened states."""
        mask = flat["mask"]
        orig_mask = mask.clone()
        c = self.cond_factory(flat)
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)
        seqs = self.init_repr_factory(flat) * mask[..., None]
        b, n_orig, _ = seqs.shape
        if self.use_concat:
            seqs, mask = self.concat_factory(flat, seqs, mask)
            n_extended = seqs.shape[1]
            n_concat = n_extended - n_orig
            if n_concat > 0:
                zero_cond = torch.zeros(b, n_concat, c.shape[-1], device=seqs.device, dtype=c.dtype)
                c = torch.cat([c, zero_cond], dim=1)
        else:
            n_concat = 0
        if self.use_concat and self.use_advanced_pair:
            pair_rep = self.pair_repr_builder(flat)
            pair_rep = self.concat_pair_factory(flat, pair_rep, orig_mask)
        else:
            pair_rep = self.pair_repr_builder(flat)
            if n_concat > 0:
                dim_pair = pair_rep.shape[-1]
                zero_pad_1 = torch.zeros(b, n_concat, n_orig, dim_pair, device=seqs.device, dtype=pair_rep.dtype)
                pair_rep = torch.cat([pair_rep, zero_pad_1], dim=1)
                zero_pad_2 = torch.zeros(b, pair_rep.shape[1], n_concat, dim_pair, device=seqs.device, dtype=pair_rep.dtype)
                pair_rep = torch.cat([pair_rep, zero_pad_2], dim=2)
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](seqs, pair_rep, c, mask)
            if self.update_pair_repr and i < self.nlayers - 1 and self.pair_update_layers[i] is not None:
                pair_rep = self.pair_update_layers[i](seqs, pair_rep, mask)
        local_latents_out = self.local_latents_linear(seqs) * mask[..., None]
        ca_nm_out = self.ca_linear(seqs) * mask[..., None]
        seq_tokens = seqs
        if n_concat > 0:
            local_latents_out = local_latents_out[:, :n_orig, :] * orig_mask[:, :, None]
            ca_nm_out = ca_nm_out[:, :n_orig, :] * orig_mask[:, :, None]
            seq_tokens = seq_tokens[:, :n_orig, :] * orig_mask[:, :, None]
        return ca_nm_out, local_latents_out, seq_tokens, orig_mask

    def forward(self, input: dict) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        mask = input["mask"]
        # MODIFIED 2026-05-07 Stage12:
        # In de_novo_multistate mode the model must start from target-only
        # conditioning. Pose/source fields are valid for repair baselines but
        # would change the scientific task here into inverse folding or pose
        # refinement, so fail loudly when they appear.
        if bool(input.get("de_novo_multistate_mode", False)):
            forbidden = [
                "init_bb_ca_states",
                "source_bb_ca_states",
                "source_complex_paths",
                "source_interface_contact_labels",
                "source_interface_distance_labels",
            ]
            leaked = [key for key in forbidden if key in input]
            if leaked:
                raise ValueError(
                    "de_novo_multistate mode is target-only and forbids source/pose inputs: "
                    f"{leaked}"
                )
            # Stage12C: optional CA/residue-type features are valid for repair
            # baselines but leak true binder geometry/sequence in target-only
            # de-novo mode. Self-conditioning must use x_t_states, not these
            # baseline optional feature switches.
            if bool(input.get("use_ca_coors_nm_feature", False)):
                raise ValueError("de_novo_multistate mode forbids optional CA coordinate features")
            if bool(input.get("use_residue_type_feature", False)):
                raise ValueError("de_novo_multistate mode forbids optional residue-type features")
        if bool(input.get("stage13_native_state_path", False)):
            flat, batch_size, nstates, nbinder = self._native_state_flatten_input(input)
            flat_bb_ca, flat_local_latents, flat_seq_tokens, flat_mask = self._native_single_state_forward(flat)
            bb_ca_states = flat_bb_ca.reshape(batch_size, nstates, nbinder, 3)
            local_latents_states = flat_local_latents.reshape(batch_size, nstates, nbinder, self.latent_dim)
            state_tokens = flat_seq_tokens.reshape(batch_size, nstates, nbinder, self.token_dim)
            state_seq_logits = self.state_seq_head(flat_seq_tokens).reshape(batch_size, nstates, nbinder, 20)
            state_present = input["state_present_mask"].to(device=bb_ca_states.device).bool()
            state_mask = input.get("state_mask", mask[:, None, :].expand(-1, nstates, -1)).to(device=bb_ca_states.device).bool()
            state_view = state_present[:, :, None, None]
            bb_ca_states = bb_ca_states * state_view
            local_latents_states = local_latents_states * state_view
            state_seq_logits = state_seq_logits * state_view
            weights = input.get("target_state_weights")
            if weights is None:
                weights = state_present.float()
            else:
                weights = weights.to(device=bb_ca_states.device, dtype=torch.float32) * state_present.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            shared_bb_ca = (bb_ca_states * weights[:, :, None, None]).sum(dim=1) * mask[..., None]
            shared_local_latents = (local_latents_states * weights[:, :, None, None]).sum(dim=1) * mask[..., None]
            shared_tokens = (state_tokens * weights[:, :, None, None]).sum(dim=1) * mask[..., None]
            shared_seq_logits = self.shared_seq_head(shared_tokens) * mask[..., None]
            consensus_logits = (state_seq_logits * weights[:, :, None, None]).sum(dim=1) * mask[..., None]
            shared_seq_logits = shared_seq_logits + consensus_logits
            flow_gate = torch.ones(batch_size, nstates, nbinder, 1, device=bb_ca_states.device, dtype=bb_ca_states.dtype)
            return {
                "bb_ca": {self.output_param["bb_ca"]: shared_bb_ca},
                "local_latents": {self.output_param["local_latents"]: shared_local_latents},
                "seq_logits_shared": shared_seq_logits,
                "seq_logits_base_shared": shared_seq_logits,
                "seq_logits_shared_tokens": shared_seq_logits,
                "shared_seq_tokens": shared_tokens,
                "state_seq_logits": state_seq_logits,
                "bb_ca_states": bb_ca_states,
                "local_latents_states": local_latents_states,
                "flow_gate": flow_gate * state_mask[..., None],
                "interface_quality_logits": torch.zeros(batch_size, nstates, 5, device=bb_ca_states.device, dtype=bb_ca_states.dtype),
                "arch_debug": {
                    "stage13_native_state_path": torch.tensor(True, device=bb_ca_states.device),
                    "target_state_weights": weights,
                    "sampling_state_weights": weights,
                    "flow_gate_mean": flow_gate.detach().mean(),
                },
            }
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
        x_t_states = input.get("x_t_states")
        if x_t_states is not None:
            xt_update = 0.0
            if "bb_ca" in x_t_states:
                xt_bb = x_t_states["bb_ca"].to(device=seqs.device, dtype=seqs.dtype)
                if xt_bb.shape[:3] != (batch_size, nstates, nbinder):
                    raise ValueError(
                        f"x_t_states['bb_ca'] shape {tuple(xt_bb.shape)} is incompatible with "
                        f"(B,K,Nb)=({batch_size},{nstates},{nbinder})"
                    )
                xt_update = xt_update + self.state_xt_bb_ca_projector(xt_bb)
            if "local_latents" in x_t_states:
                xt_lat = x_t_states["local_latents"].to(device=seqs.device, dtype=seqs.dtype)
                if xt_lat.shape[:3] != (batch_size, nstates, nbinder):
                    raise ValueError(
                        f"x_t_states['local_latents'] shape {tuple(xt_lat.shape)} is incompatible with "
                        f"(B,K,Nb)=({batch_size},{nstates},{nbinder})"
                    )
                xt_update = xt_update + self.state_xt_latent_projector(xt_lat)
            if not isinstance(xt_update, float):
                xt_update = xt_update * mask[:, None, :, None] * target_bundle["state_present_mask"][:, :, None, None]
                state_tokens = self.state_token_norm(state_tokens + xt_update)
                state_seq_tokens = self.state_token_norm(state_seq_tokens + 0.5 * xt_update)
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
        state_weights = target_bundle["target_state_weights"].float() * target_bundle["state_present_mask"].float()
        state_weights = state_weights / state_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        # Stage12 cross-state shared sequence tokens. For each binder residue,
        # attend over its K state-specific tokens, then pool back to one token.
        # This is the key difference from post-hoc K-state sequence voting: the
        # shared token is injected back before bb_ca/local_latents readout.
        state_present_bool = target_bundle["state_present_mask"].bool()
        state_tokens_by_residue = state_tokens.permute(0, 2, 1, 3).reshape(batch_size * nbinder, nstates, self.token_dim)
        state_key_padding = ~state_present_bool[:, None, :].expand(batch_size, nbinder, nstates).reshape(
            batch_size * nbinder, nstates
        )
        attended_by_residue, _ = self.cross_state_shared_seq_attention(
            state_tokens_by_residue,
            state_tokens_by_residue,
            state_tokens_by_residue,
            key_padding_mask=state_key_padding,
            need_weights=False,
        )
        attended_states = attended_by_residue.reshape(batch_size, nbinder, nstates, self.token_dim).permute(0, 2, 1, 3)
        state_weight_view = state_weights[:, :, None, None]
        shared_seq_tokens_seed = (state_tokens * state_weight_view).sum(dim=1)
        shared_seq_tokens_attn = (attended_states * state_weight_view).sum(dim=1)
        shared_seq_tokens = self.shared_seq_token_norm(
            shared_seq_tokens_seed + self.shared_seq_token_update(shared_seq_tokens_attn)
        ) * mask[..., None]
        shared_seq_token_logits = self.shared_seq_token_head(shared_seq_tokens) * mask[..., None]
        shared_seq_state_feedback = self.shared_seq_to_state_projector(shared_seq_tokens) * mask[..., None]
        state_tokens = self.state_token_norm(state_tokens + shared_seq_state_feedback[:, None, :, :])
        state_seq_tokens = self.state_token_norm(state_seq_tokens + 0.5 * shared_seq_state_feedback[:, None, :, :])

        # Stage12D: inject a token-only shared sequence signal before the first
        # state sequence readout.  Stage12C already fed back the post-consensus
        # soft sequence, but only after state logits had been computed once.
        # This earlier feedback makes the flow heads sequence-aware even when
        # sampled x_t states are still far from the training interpolant.
        early_shared_seq_logits = (base_shared_seq_logits + shared_seq_token_logits) * mask[..., None]
        early_shared_seq_soft = torch.softmax(early_shared_seq_logits.float(), dim=-1).to(dtype=seqs.dtype) * mask[..., None]
        early_seq_feedback_tokens = self.soft_sequence_feedback_projector(early_shared_seq_soft) * mask[..., None]
        state_tokens = self.state_token_norm(state_tokens + 0.15 * early_seq_feedback_tokens[:, None, :, :])
        state_seq_tokens = self.state_token_norm(state_seq_tokens + 0.25 * early_seq_feedback_tokens[:, None, :, :])

        flat_state_tokens = state_tokens.reshape(batch_size * nstates, nbinder, self.token_dim)
        flat_state_seq_tokens = state_seq_tokens.reshape(batch_size * nstates, nbinder, self.token_dim)
        flat_state_mask = mask[:, None, :].expand(-1, nstates, -1).reshape(batch_size * nstates, nbinder)

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
        shared_seq_logits_pre = (base_shared_seq_logits + shared_seq_token_logits + consensus_gate_pre * consensus_logits_pre) * mask[..., None]
        shared_seq_soft = torch.softmax(shared_seq_logits_pre.float(), dim=-1).to(dtype=seqs.dtype) * mask[..., None]
        seq_feedback_tokens = self.soft_sequence_feedback_projector(shared_seq_soft) * mask[..., None]
        state_tokens = self.state_token_norm(state_tokens + 0.25 * seq_feedback_tokens[:, None, :, :])
        state_seq_tokens = self.state_token_norm(state_seq_tokens + 0.50 * seq_feedback_tokens[:, None, :, :])

        flat_state_tokens = state_tokens.reshape(batch_size * nstates, nbinder, self.token_dim)
        flat_state_seq_tokens = state_seq_tokens.reshape(batch_size * nstates, nbinder, self.token_dim)
        flat_local_latents_base = self.local_latents_linear(flat_state_tokens)
        flat_latent_repair = self.latent_sequence_repair_projector(flat_state_seq_tokens)
        flat_latent_repair_gate = self.latent_sequence_repair_gate(flat_state_seq_tokens)
        flat_local_latents = (
            flat_local_latents_base + 0.25 * flat_latent_repair_gate * flat_latent_repair
        ) * flat_state_mask[..., None]
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
        shared_seq_logits = (base_shared_seq_logits + shared_seq_token_logits + consensus_gate * consensus_logits) * mask[..., None]

        return {
            "bb_ca": {self.output_param["bb_ca"]: shared_bb_ca},
            "local_latents": {self.output_param["local_latents"]: shared_local_latents},
            "seq_logits_shared": shared_seq_logits,
            "seq_logits_base_shared": base_shared_seq_logits,
            "seq_logits_shared_tokens": shared_seq_token_logits,
            "shared_seq_tokens": shared_seq_tokens,
            "state_seq_logits": state_seq_logits,
            "bb_ca_states": bb_ca_states,
            "local_latents_states": local_latents_states,
            "flow_gate": flow_gate,
            "latent_sequence_repair_gate": flat_latent_repair_gate.reshape(batch_size, nstates, nbinder, 1)
            * state_present_mask,
            "seq_feedback_tokens": seq_feedback_tokens,
            "early_seq_feedback_tokens": early_seq_feedback_tokens,
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
                "stage12_shared_sequence_token_norm_mean": shared_seq_tokens.detach().norm(dim=-1).mean(),
                "stage12_shared_sequence_token_logits_norm_mean": shared_seq_token_logits.detach().norm(dim=-1).mean(),
                "cross_state_attention_mean": target_bundle["cross_state_attention_mean"],
                "sequence_feedback_norm_mean": seq_feedback_tokens.detach().norm(dim=-1).mean(),
                "early_sequence_feedback_norm_mean": early_seq_feedback_tokens.detach().norm(dim=-1).mean(),
                "latent_sequence_repair_gate_mean": flat_latent_repair_gate.detach().mean(),
                "latent_sequence_repair_norm_mean": flat_latent_repair.detach().norm(dim=-1).mean(),
                "flow_gate_mean": flow_gate.detach().mean(),
                "flow_gate_std": flow_gate.detach().std(),
            },
        }
