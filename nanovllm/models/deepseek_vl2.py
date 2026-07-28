
from typing import Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention, store_kvcache
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear, RowParallelLinear, ReplicatedLinear,
    MergedColumnParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.fused_moe import FusedMoE
from nanovllm.utils.context import get_context
from nanovllm.multimodal_runtime import (
    MultimodalContractError,
    VL2EmbeddingCache,
)


# ============================================================================
# 1. MLA (Multi-head Latent Attention)
# ============================================================================


#

#   x [seq, 2048]
#     ├─ q_proj → q [seq, 16, 192]        # 192 = nope(128) + rope(64)


#     │
#     └─ kv_a_proj_with_mqa → [seq, 576]   # 576 = kv_lora_rank(512) + qk_rope(64)

#         │   └─ kv_a_layernorm
#         │       └─ kv_b_proj → [seq, 16, 256]   # 256 = nope(128) + v(128)
#         │           ├─ k_nope [seq, 16, 128]
#         │           └─ v [seq, 16, 128]

#
#   Q_final = concat(q_nope, q_rope)  [seq, 16, 192]

#   V       = v                        [seq, 16, 128]
#
#   attention(Q, K, V) → [seq, 16, 128]
#   o_proj → [seq, 2048]
# ============================================================================

class DeepseekV2MLA(nn.Module):
    """Multi-head Latent Attention for DeepSeek-V2"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: int = 512,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict] = None,
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-06,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 192
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.scaling = self.qk_head_dim ** -0.5

        tp_size = dist.get_world_size()
        assert num_heads % tp_size == 0, (
            f"num_heads {num_heads} must be divisible by tp_size {tp_size}")
        self.num_local_heads = num_heads // tp_size

        
        if q_lora_rank is None:
            
            # q_proj: [hidden_size → num_heads × qk_head_dim]
            #   Small: [2048 → 16 × 192 = 3072]
            self.q_proj = ColumnParallelLinear(
                hidden_size,
                num_heads * self.qk_head_dim,
                bias=False,
            )
            self.q_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
        else:
            
            self.q_a_proj = ReplicatedLinear(hidden_size, q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                num_heads * self.qk_head_dim,
                bias=False,
            )
            self.q_proj = None

        
        # kv_a_proj_with_mqa: [hidden → kv_lora_rank + qk_rope_head_dim]
        #   Small: [2048 → 512 + 64 = 576]
        
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            bias=False,
        )

        
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)

        
        # kv_b_proj: [kv_lora_rank → num_heads × (qk_nope + v_head_dim)]
        #   Small: [512 → 16 × (128 + 128) = 4096]
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        
        # o_proj: [num_heads × v_head_dim → hidden_size]
        #   Small: [16 × 128 = 2048 → 2048]
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
        )

        
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            # The official DeepSeek-VL2 checkpoint was trained with adjacent
            # rotary pairs.  Its reference implementation expresses this as
            # an even/odd permutation followed by rotate_half; nano-vLLM's
            # equivalent kernel layout is the non-NeoX (interleaved) mode.
            is_neox_style=False,
        )

        # ---- Attention kernel ----
        
        
        
        
        
        self.attn = Attention(
            self.num_local_heads,
            self.qk_head_dim,  
            self.scaling,
            self.num_local_heads,  
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        bsz_seqlen, _ = hidden_states.shape  # flatten batch × seq

        
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q_a = self.q_a_proj(hidden_states)
            q_a = self.q_a_layernorm(q_a)
            q = self.q_b_proj(q_a)

        q = q.view(bsz_seqlen, self.num_local_heads, self.qk_head_dim)
        
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        
        kv_a = self.kv_a_proj_with_mqa(hidden_states)  # [seq, 576]
        
        # k_pe:          [seq, 64]
        compressed_kv, k_pe = torch.split(
            kv_a, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)

        
        kv = self.kv_b_proj(compressed_kv)
        # [seq, 16, 256] = [seq, 16, nope(128) + v(128)]
        kv = kv.view(bsz_seqlen, self.num_local_heads,
                     self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # ========== 4. RoPE ==========
        
        # q_pe: [seq, num_local_heads, qk_rope_head_dim] → [seq, num_local_heads*qk_rope_head_dim]
        
        q_pe_flat = q_pe.reshape(bsz_seqlen, self.num_local_heads * self.qk_rope_head_dim).contiguous()
        
        k_pe_flat = k_pe.contiguous()  # [seq, 64]
        q_pe_flat, k_pe_flat = self.rotary_emb(positions, q_pe_flat, k_pe_flat)
        
        q_pe = q_pe_flat.view(bsz_seqlen, self.num_local_heads, self.qk_rope_head_dim)
        
        k_pe = k_pe_flat.view(bsz_seqlen, 1, self.qk_rope_head_dim) \
                        .expand(-1, self.num_local_heads, -1)

        
        # Q_final = [q_nope, q_pe]  [seq, 16, 192]
        q_final = torch.cat([q_nope, q_pe], dim=-1)
        k_final = torch.cat([k_nope, k_pe], dim=-1)  # [seq, 16, 192]

        
        pad_size = self.qk_head_dim - self.v_head_dim  # 192 - 128 = 64
        v_padded = F.pad(v, (0, pad_size))  # [seq, 16, 192]

        # ========== 6. Attention ==========
        
        q_flat = q_final.view(bsz_seqlen, -1)
        k_flat = k_final.view(bsz_seqlen, -1)
        v_flat = v_padded.view(bsz_seqlen, -1)

        attn_output = self.attn(q_flat, k_flat, v_flat)
        # attn_output: [seq, num_heads * qk_head_dim] = [seq, 16 × 192]

        
        attn_output = attn_output.view(
            bsz_seqlen, self.num_local_heads, self.qk_head_dim
        )
        attn_output = attn_output[..., :self.v_head_dim]  # [seq, 16, 128]
        attn_output = attn_output.reshape(bsz_seqlen, -1)  # [seq, 2048]

        
        output = self.o_proj(attn_output)
        return output


# ============================================================================

# ============================================================================

class DeepseekV2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ):
        super().__init__()

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu", f"Unsupported activation: {hidden_act}"
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        out = self.down_proj(x)
        x = out[0] if isinstance(out, tuple) else out
        return x


# ============================================================================
# 3. DeepseekV2MoE (MoE block with shared + routed experts)
# ============================================================================
#




# ============================================================================

class DeepseekV2MoE(nn.Module):
    """DeepSeek-V2 MoE block with shared experts"""

    def __init__(self, config):
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', 1.0)
        self.norm_topk_prob = getattr(config, 'norm_topk_prob', False)
        self.scoring_func = getattr(config, 'scoring_func', 'softmax')
        self.topk_method = getattr(config, 'topk_method', 'greedy')
        self.use_grouped_topk = self.topk_method == 'noaux_tc'
        self.num_expert_group = getattr(config, 'n_group', None)
        self.topk_group = getattr(config, 'topk_group', None)

        assert self.tp_size <= self.num_experts, (
            f"TP size {self.tp_size} > num_experts {self.num_experts}")

        
        correction_bias = None
        if self.use_grouped_topk:
            correction_bias = nn.Parameter(torch.zeros(self.num_experts))
        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=self.moe_intermediate_size,
            reduce_results=False,  
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            scoring_func=self.scoring_func,
            e_score_correction_bias=correction_bias,
        )

        
        self.gate = ReplicatedLinear(
            self.hidden_size,
            self.num_experts,
            bias=False,
        )

        # weight shape: [num_experts, hidden_size] = [64, 2048]

        # ---- Shared experts ----
        
        #   Small: 2 × 1408 = 2816
        n_shared = config.n_shared_experts
        if n_shared is not None and n_shared > 0:
            shared_intermediate = self.moe_intermediate_size * n_shared
            self.shared_experts = DeepseekV2MLP(
                hidden_size=self.hidden_size,
                intermediate_size=shared_intermediate,
                hidden_act=config.hidden_act,
            )
        else:
            self.shared_experts = None

        
        self._layer_idx = -1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _exact_overlap = bool(
            getattr(self.experts, 'hover_exact_h2d_overlap', False)
        )
        if _exact_overlap:
            router_logits = self.gate(hidden_states)
            if self.shared_experts is None:
                # Generic fallback for MoE variants without a shared branch:
                # resident-expert routes cover true-miss H2D after exact gate
                # selection.  No predictor is involved.
                routed_output = (
                    self.experts.forward_with_route_split_overlap(
                        hidden_states=hidden_states,
                        router_logits=router_logits,
                    )
                )
                shared_output = None
            else:
                # Shared compute remains the lower-overhead preferred anchor
                # when the architecture provides it.
                routed_output, shared_output = (
                    self.experts.forward_with_shared_expert_overlap(
                        hidden_states=hidden_states,
                        router_logits=router_logits,
                        shared_forward=lambda: self.shared_experts(hidden_states),
                    )
                )
        else:
            # Preserve the established flag-off execution and TP collective
            # order byte-for-byte at the operation level: shared first, then
            # route selection/routed experts.
            shared_output = None
            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)

            router_logits = self.gate(hidden_states)  # [seq, num_experts]
            routed_output = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        # ---- routed_scaling_factor ----
        if self.routed_scaling_factor != 1.0:
            routed_output = routed_output * self.routed_scaling_factor

        # Routed experts are sharded across TP ranks because FusedMoE was
        # constructed with reduce_results=False.  Shared experts, however,
        # end in RowParallelLinear, whose forward already all-reduces and
        # therefore returns the same complete tensor on every rank.  Reducing
        # their sum would count the shared branch tp_size times.
        if self.tp_size > 1:
            routed_output = self.experts.maybe_all_reduce_tensor_model_parallel(
                routed_output
            )

        if shared_output is not None:
            final_hidden_states = routed_output + shared_output
        else:
            final_hidden_states = routed_output

        return final_hidden_states


# ============================================================================

# ============================================================================

class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # ---- Attention (MLA) ----
        self.self_attn = DeepseekV2MLA(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=getattr(config, 'qk_nope_head_dim', 128),
            qk_rope_head_dim=getattr(config, 'qk_rope_head_dim', 64),
            v_head_dim=getattr(config, 'v_head_dim', 128),
            q_lora_rank=getattr(config, 'q_lora_rank', None),
            kv_lora_rank=getattr(config, 'kv_lora_rank', 512),
            rope_theta=getattr(config, 'rope_theta', 10000),
            rope_scaling=getattr(config, 'rope_scaling', None),
            max_position_embeddings=getattr(config, 'max_position_embeddings', 4096),
            rms_norm_eps=config.rms_norm_eps,
        )

        
        first_k_dense_replace = getattr(config, 'first_k_dense_replace', 0)
        if layer_idx < first_k_dense_replace:
            # Dense FFN
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        else:
            # MoE
            self.mlp = DeepseekV2MoE(config)
            self.mlp._layer_idx = layer_idx
            self.mlp.experts._moe_layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# ============================================================================

# ============================================================================

class DeepseekV2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            DeepseekV2DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.is_graph_captured = False
        self.num_stages = -1

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        intermediate_outputs=None,
    ):
        from nanovllm.utils.context import get_context

        ctx = get_context()

        
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        
        hover_stage_layers = ctx.prefill_compute_layers if ctx else None

        
        try:
            from nanovllm.engine.model_runner import _GLOBAL_MODEL_RUNNER as _gr
            if _gr is not None and hasattr(_gr, "trigger_prefetch_for_next_stage") and hover_stage_layers:
                _gr.trigger_prefetch_for_next_stage(hover_stage_layers, len(self.layers))
        except Exception:
            pass

        if not hover_stage_layers:
            
            residual = None
            for layer in self.layers:
                hidden_states, residual = layer(positions, hidden_states, residual)
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states

        
        
        
        
        active_layers = sorted({int(layer_idx) for layer_idx in hover_stage_layers})
        first_layer = active_layers[0]
        last_layer = active_layers[-1]
        num_layers = len(self.layers)
        if active_layers != list(range(first_layer, last_layer + 1)):
            raise ValueError(
                f"HoVer staged prefill requires contiguous layers, got {active_layers}"
            )
        if first_layer < 0 or last_layer >= num_layers:
            raise ValueError(
                f"HoVer staged prefill layers out of range: {active_layers}, num_layers={num_layers}"
            )

        num_prefill = int(getattr(ctx, "len_prefill", 0) or 0)
        num_rows = int(hidden_states.size(0))
        if num_prefill < 0 or num_prefill > num_rows:
            raise ValueError(
                f"invalid len_prefill={num_prefill} for {num_rows} input rows"
            )
        num_decode = num_rows - num_prefill

        residual = hidden_states.clone()
        i_hidden = i_residual = None
        if intermediate_outputs is not None:
            i_hidden, i_residual = intermediate_outputs

        if first_layer > 0 and num_prefill > 0:
            if i_hidden is None or i_residual is None:
                raise ValueError(
                    "missing HoVer intermediate outputs for a non-first stage"
                )
            if i_hidden.size(0) != num_prefill or i_residual.size(0) != num_prefill:
                raise ValueError(
                    "HoVer intermediate rows must exactly match len_prefill "
                    f"({i_hidden.size(0)}, {i_residual.size(0)} != {num_prefill})"
                )
            hidden_states = torch.cat([i_hidden, hidden_states[num_prefill:]], dim=0)
            residual = torch.cat([i_residual, residual[num_prefill:]], dim=0)
        elif i_hidden is not None or i_residual is not None:
            if i_hidden is None or i_residual is None:
                raise ValueError("HoVer hidden/residual intermediates must be paired")
            if i_hidden.size(0) != num_prefill or i_residual.size(0) != num_prefill:
                raise ValueError(
                    "HoVer intermediate rows must exactly match len_prefill "
                    f"({i_hidden.size(0)}, {i_residual.size(0)} != {num_prefill})"
                )
            hidden_states = torch.cat([i_hidden, hidden_states[num_prefill:]], dim=0)
            residual = torch.cat([i_residual, residual[num_prefill:]], dim=0)

        def _decode_rows(value):
            if value is None:
                return None
            return value[num_prefill:]

        def _run_decode_layers(
            layer_indices, decode_hidden, decode_residual, apply_norm=False
        ):
            """Run a decode-only slice while keeping all row metadata aligned."""
            if num_decode == 0 or (not layer_indices and not apply_norm):
                return decode_hidden, decode_residual

            row_fields = (
                "slot_mapping",
                "images_seq_mask",
                "token_modalities",
                "token_phase",
                "token_seqid",
            )
            saved_is_prefill = ctx.is_prefill
            saved_len_prefill = ctx.len_prefill
            saved_rows = {name: getattr(ctx, name, None) for name in row_fields}
            try:
                ctx.is_prefill = False
                ctx.len_prefill = 0
                for name, value in saved_rows.items():
                    setattr(ctx, name, _decode_rows(value))
                for layer_idx in layer_indices:
                    decode_hidden, decode_residual = self.layers[layer_idx](
                        positions[num_prefill:], decode_hidden, decode_residual
                    )
                if apply_norm:
                    decode_hidden, _ = self.norm(decode_hidden, decode_residual)
                return decode_hidden, decode_residual
            finally:
                ctx.is_prefill = saved_is_prefill
                ctx.len_prefill = saved_len_prefill
                for name, value in saved_rows.items():
                    setattr(ctx, name, value)

        # Decode rows first catch up through the layers before the prefill stage.
        decode_hidden = hidden_states[num_prefill:]
        decode_residual = None
        decode_hidden, decode_residual = _run_decode_layers(
            range(0, first_layer), decode_hidden, decode_residual
        )
        if first_layer > 0 and num_decode > 0:
            hidden_states = torch.cat(
                [hidden_states[:num_prefill], decode_hidden], dim=0
            )
            residual = torch.cat(
                [residual[:num_prefill], decode_residual], dim=0
            )

        # Current stage runs on both prefill and decode rows.
        current_residual = residual if first_layer > 0 else None
        for layer_idx in active_layers:
            hidden_states, current_residual = self.layers[layer_idx](
                positions, hidden_states, current_residual
            )
        residual = current_residual

        # Decode rows finish the layers after the prefill stage.
        decode_hidden = hidden_states[num_prefill:]
        decode_residual = residual[num_prefill:]
        decode_hidden, decode_residual = _run_decode_layers(
            range(last_layer + 1, num_layers), decode_hidden, decode_residual
        )
        if last_layer < num_layers - 1 and num_decode > 0:
            hidden_states = torch.cat(
                [hidden_states[:num_prefill], decode_hidden], dim=0
            )
            residual = torch.cat(
                [residual[:num_prefill], decode_residual], dim=0
            )

        if last_layer == num_layers - 1:
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states

        # Middle prefill stages must retain unnormalised intermediates, while
        # decode rows are already full-depth and therefore need final norm now.
        if num_decode > 0:
            decode_hidden, _ = _run_decode_layers(
                (),
                hidden_states[num_prefill:],
                residual[num_prefill:],
                apply_norm=True,
            )
            hidden_states = torch.cat(
                [hidden_states[:num_prefill], decode_hidden], dim=0
            )

        if num_prefill > 0:
            return (hidden_states, residual)
        return hidden_states


# ============================================================================

# ============================================================================

class DeepseekV2ForCausalLM(nn.Module):

    
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DeepseekV2Model(config)

        tie = getattr(config, 'tie_word_embeddings', False)
        if not tie:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        else:
            self.lm_head = None

    def forward(self, input_ids, positions, intermediate_outputs=None):
        out = self.model(input_ids, positions, intermediate_outputs=intermediate_outputs)
        
        
        if isinstance(out, tuple):
            return out  
        return out, None  

    def compute_logits(self, hidden_states):
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        return logits


# ============================================================================

# ============================================================================



def _import_vl2_components():
    import sys
    from pathlib import Path
    candidates = [
        Path("multimodal_migration/models/DeepSeek-VL2-main/DeepSeek-VL2-main"),
        Path("../multimodal_migration/models/DeepSeek-VL2-main/DeepSeek-VL2-main"),
        Path.cwd() / "multimodal_migration" / "models" / "DeepSeek-VL2-main" / "DeepSeek-VL2-main",
    ]
    for cand in candidates:
        if cand.is_dir():
            cand_str = str(cand.resolve())
            if cand_str not in sys.path:
                sys.path.insert(0, cand_str)
            break
    from deepseek_vl2.models.siglip_vit import VisionTransformer as _VT
    from deepseek_vl2.models.modeling_deepseek_vl_v2 import (
        MlpProjector as _MP,
        MlpProjectorConfig as _MPCfg,
    )
    return _VT, _MP, _MPCfg


class VisionTower(nn.Module):

    def __init__(self, vision_config):
        super().__init__()
        _VT, _, _ = _import_vl2_components()

        
        def _get(cfg, key, default=None):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        self.vit = _VT(
            img_size=_get(vision_config, "image_size", 384),
            patch_size=_get(vision_config, "patch_size", 14),
            embed_dim=_get(vision_config, "width", 1152),
            depth=_get(vision_config, "layers", 27),
            num_heads=_get(vision_config, "heads", 16),
            mlp_ratio=_get(vision_config, "mlp_ratio", 3.7362),
            class_token=_get(vision_config, "class_token", False),
            global_pool=_get(vision_config, "global_pool", "map"),
            ignore_head=_get(vision_config, "ignore_head", True),
            weight_init=_get(vision_config, "weight_init", "skip"),
            num_classes=0,
            deterministic=_get(vision_config, "deterministic", False),
            num_recomputing_layers=_get(vision_config, "num_recomputing_layers", 0),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim == 5:
            b, t, c, h, w = pixel_values.shape
            pixel_values = pixel_values.view(b * t, c, h, w)
        return self.vit(pixel_values)


class MlpProjector(nn.Module):

    def __init__(self, projector_config, vision_hidden=None, llm_hidden=None):
        super().__init__()
        _, _MP, _MPCfg = _import_vl2_components()

        
        if isinstance(projector_config, _MPCfg):
            cfg = projector_config
        else:
            if isinstance(projector_config, dict):
                kwargs = dict(projector_config)
            elif hasattr(projector_config, "to_dict"):
                kwargs = projector_config.to_dict()
            else:
                kwargs = dict(projector_config.__dict__) if hasattr(projector_config, "__dict__") else {}

            
            kwargs.setdefault("projector_type", "downsample_mlp_gelu")
            kwargs.setdefault("input_dim", vision_hidden or 1152)
            kwargs.setdefault("n_embed", llm_hidden or 2048)
            kwargs.setdefault("depth", 2)
            kwargs.setdefault("mlp_ratio", 1)
            kwargs.setdefault("downsample_ratio", 2)
            kwargs.setdefault("token_pooling", False)

            
            try:
                allowed = _MPCfg().to_dict().keys()
            except Exception:
                allowed = list(kwargs.keys())
            clean = {k: v for k, v in kwargs.items() if k in allowed}
            cfg = _MPCfg(**clean)

        self.cfg = cfg
        self.proj = _MP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ============================================================================

# ============================================================================

class DeepseekVLV2ForCausalLM(nn.Module):

    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        self.config = config

        vision_cfg = config.vision_config
        language_cfg = config.language_config

        
        if isinstance(language_cfg, dict):
            from types import SimpleNamespace
            language_cfg = SimpleNamespace(**language_cfg)

        
        self.vision = VisionTower(vision_cfg)

        
        if isinstance(vision_cfg, dict):
            vision_hidden = vision_cfg.get("width") or vision_cfg.get("hidden_size", 1152)
        else:
            vision_hidden = getattr(vision_cfg, "width", None) or getattr(vision_cfg, "hidden_size", 1152)

        
        llm_hidden = getattr(language_cfg, "hidden_size", 2048)

        
        projector_cfg = getattr(config, "projector_config", {})
        self.projector = MlpProjector(projector_cfg, vision_hidden, llm_hidden)

        
        
        self.tile_tag = getattr(config, "tile_tag", "2D")
        self.global_view_pos = getattr(config, "global_view_pos", "head")
        if self.tile_tag != "2D":
            raise MultimodalContractError(
                f"transactional VL2 runtime supports tile_tag='2D', got {self.tile_tag!r}"
            )
        embed_std = 1.0 / (llm_hidden ** 0.5)
        if self.tile_tag == "2D":
            self.image_newline = nn.Parameter(torch.randn(llm_hidden) * embed_std)
            self.view_seperator = nn.Parameter(torch.randn(llm_hidden) * embed_std)
        elif self.tile_tag == "1D":
            candidate_resolutions = getattr(config, "candidate_resolutions", [])
            tile_variants_num = max(len(candidate_resolutions), 1)
            self.tile_indicators = nn.Parameter(
                torch.randn(tile_variants_num + 1, llm_hidden) * embed_std
            )

        
        self.language = DeepseekV2ForCausalLM(language_cfg)
        self._visual_cache = VL2EmbeddingCache(
            self.image_newline,
            self.view_seperator,
            self.global_view_pos,
            limit=int(getattr(config, "max_visual_cache_entries", 512)),
        )
        self._pending_visual_transactions = ()

    def prepare_multimodal_embeddings(self, input_ids, ctx):
        """Scatter official formatted visual slots for this scheduling round."""
        text_emb = self.language.model.embed_tokens(input_ids)
        ordinals = getattr(ctx, "image_slot_ordinals", None)
        seqids = getattr(ctx, "token_seqid", None)
        payloads = getattr(ctx, "multimodal_payloads", None) or {}
        self._pending_visual_transactions = ()
        if ordinals is None or seqids is None or not any(value >= 0 for value in ordinals):
            return text_emb

        def encode(images):
            target = next(self.vision.parameters())
            images = images.to(
                device=target.device,
                dtype=target.dtype,
                non_blocking=bool(images.device.type == "cpu" and images.is_pinned()),
            )
            if images.device != target.device or images.dtype != target.dtype:
                raise MultimodalContractError("vision input device/dtype conversion failed")
            with torch.no_grad():
                return self.projector(self.vision(images))



        result, telemetry = self._visual_cache.scatter(
            text_emb,
            seqids,
            ordinals,
            payloads,
            encode,
        )
        self._pending_visual_transactions = telemetry.transactions
        return result

    def has_pending_multimodal_forward(self):
        return bool(self._pending_visual_transactions)

    def commit_multimodal_forward(self):
        """Commit prepared cache progress only after the language forward."""
        released = self._visual_cache.commit(self._pending_visual_transactions)
        self._pending_visual_transactions = ()
        return released

    def forward(self, input_ids, positions, intermediate_outputs=None):
        ctx = get_context()
        inputs_embeds = self.prepare_multimodal_embeddings(input_ids, ctx)

        
        out = self.language.model(
            input_ids, positions,
            inputs_embeds=inputs_embeds,
            intermediate_outputs=intermediate_outputs,
        )

        
        if isinstance(out, tuple):
            return out  
        return out, None  

    def compute_logits(self, hidden_states):
        return self.language.compute_logits(hidden_states)
