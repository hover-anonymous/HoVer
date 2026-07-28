"""NanoVLLM adapter for moonshotai/Kimi-VL-A3B-Instruct."""

from __future__ import annotations

import logging
import sys

import torch
from torch import nn
from torch.nn import functional as F
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from nanovllm.models.deepseek_vl2 import DeepseekV2ForCausalLM
from nanovllm.multimodal_runtime import (
    DirectEmbeddingCache,
    MultimodalContractError,
)
from nanovllm.utils.context import get_context


logger = logging.getLogger(__name__)

KIMI_VISION_ATTENTION_PROTOCOL = "segmented-sdpa-v1"


def segmented_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: torch.Tensor | None = None,
    k_cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply exact block-diagonal vision attention without a global LxL mask.

    MoonViT packs all images into one token axis and supplies cumulative image
    boundaries.  The upstream eager path materializes attention for the full
    packed axis, including pairs of tokens belonging to different images, and
    then masks those pairs.  A 12-page request therefore tries to allocate a
    tens-of-GiB matrix even though every image is an independent attention
    block.  Running SDPA once per image preserves those block semantics while
    keeping peak memory proportional to the largest image instead of the sum
    of all images squared.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Kimi vision q/k/v must have shape [tokens, heads, dim]")
    if k.shape != v.shape:
        raise ValueError("Kimi vision k and v shapes must match")
    if q.shape[1:] != k.shape[1:]:
        raise ValueError("Kimi vision q/k head shapes must match")

    def boundaries(value, token_count, label):
        if value is None:
            return [0, token_count]
        if value.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        points = [int(item) for item in value.detach().cpu().tolist()]
        if len(points) < 2 or points[0] != 0 or points[-1] != token_count:
            raise ValueError(
                f"{label} must start at 0 and end at {token_count}: {points}"
            )
        if any(left >= right for left, right in zip(points, points[1:])):
            raise ValueError(f"{label} contains an empty or reversed segment")
        return points

    q_points = boundaries(q_cu_seqlens, q.shape[0], "q_cu_seqlens")
    k_points = boundaries(k_cu_seqlens, k.shape[0], "k_cu_seqlens")
    if len(q_points) != len(k_points):
        raise ValueError("Kimi vision q/k segment counts must match")

    outputs = []
    for q_start, q_end, k_start, k_end in zip(
        q_points, q_points[1:], k_points, k_points[1:]
    ):
        q_block = q[q_start:q_end].transpose(0, 1)
        k_block = k[k_start:k_end].transpose(0, 1)
        v_block = v[k_start:k_end].transpose(0, 1)
        block = F.scaled_dot_product_attention(
            q_block,
            k_block,
            v_block,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        outputs.append(block.transpose(0, 1).reshape(q_end - q_start, -1))
    return torch.cat(outputs, dim=0)


def _install_segmented_vision_attention(vision_cls) -> None:
    module = sys.modules.get(vision_cls.__module__)
    functions = getattr(module, "VL_VISION_ATTENTION_FUNCTIONS", None)
    if not isinstance(functions, dict) or "sdpa" not in functions:
        raise RuntimeError(
            "Kimi remote model does not expose the expected vision attention registry"
        )
    functions["sdpa"] = segmented_sdpa_attention


def _remote_class(config, class_name: str):
    model_path = getattr(config, "_name_or_path", None)
    if not model_path:
        raise ValueError("Kimi-VL config is missing _name_or_path")
    return get_class_from_dynamic_module(
        f"modeling_kimi_vl.{class_name}",
        model_path,
    )


class KimiVLForConditionalGeneration(nn.Module):
    """MoonViT + Kimi projector + NanoVLLM DeepSeek-V3 text runtime."""

    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        text_config = config.text_config
        if isinstance(text_config, dict):
            from types import SimpleNamespace
            text_config = SimpleNamespace(**text_config)

        vision_config = config.vision_config
        vision_cls = _remote_class(config, "MoonVitPretrainedModel")
        _install_segmented_vision_attention(vision_cls)
        vision_config._attn_implementation = "sdpa"
        projector_cls = _remote_class(config, "KimiVLMultiModalProjector")
        self.vision_tower = vision_cls(vision_config)
        self.vision_attention_protocol = KIMI_VISION_ATTENTION_PROTOCOL
        logger.info(
            "Kimi MoonViT attention protocol: %s",
            self.vision_attention_protocol,
        )
        self.multi_modal_projector = projector_cls(config)
        self.language = DeepseekV2ForCausalLM(text_config)

        self.media_placeholder_token_id = int(
            config.media_placeholder_token_id
        )
        self._visual_cache = DirectEmbeddingCache(
            limit=int(getattr(config, "max_visual_cache_entries", 512))
        )
        self._pending_visual_transactions = ()

    def prepare_multimodal_embeddings(self, input_ids, ctx):
        text_embeddings = self.language.model.embed_tokens(input_ids)
        ordinals = getattr(ctx, "image_slot_ordinals", None)
        seqids = getattr(ctx, "token_seqid", None)
        payloads = getattr(ctx, "multimodal_payloads", None) or {}
        self._pending_visual_transactions = ()
        if ordinals is None or seqids is None or not any(
            value >= 0 for value in ordinals
        ):
            return text_embeddings

        def encode(pixel_values, image_grid_hws):
            target = next(self.vision_tower.parameters())
            pixel_values = pixel_values.to(
                device=target.device,
                dtype=target.dtype,
                non_blocking=bool(
                    pixel_values.device.type == "cpu"
                    and pixel_values.is_pinned()
                ),
            )
            image_grid_hws = image_grid_hws.to(
                device=target.device,
                dtype=torch.long,
                non_blocking=bool(
                    image_grid_hws.device.type == "cpu"
                    and image_grid_hws.is_pinned()
                ),
            )
            with torch.no_grad():
                features = self.vision_tower(
                    pixel_values, image_grid_hws
                )
                return self.multi_modal_projector(features)



        result, telemetry = self._visual_cache.scatter(
            text_embeddings,
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
        released = self._visual_cache.commit(
            self._pending_visual_transactions
        )
        self._pending_visual_transactions = ()
        return released

    def clear_multimodal_cache(self, seq_id):
        """Release projected visual embeddings after full prompt completion."""
        return self._visual_cache.clear(seq_id)

    def forward(self, input_ids, positions, intermediate_outputs=None):
        ctx = get_context()
        inputs_embeds = self.prepare_multimodal_embeddings(input_ids, ctx)
        out = self.language.model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            intermediate_outputs=intermediate_outputs,
        )
        if isinstance(out, tuple):
            return out
        return out, None

    def compute_logits(self, hidden_states):
        return self.language.compute_logits(hidden_states)
