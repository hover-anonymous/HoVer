from __future__ import annotations

from collections import Counter
import logging
import math

import numpy as np

from nanovllm.engine.modality_aware_partitioner import (
    ModalityAwarePartitioner,
)

logger = logging.getLogger(__name__)


class HorizontalScheduler:
    """Construct decode-first PD batches under token and disturbance budgets."""

    def __init__(self, config, partitioner: ModalityAwarePartitioner):
        self.config = config
        self.partitioner = partitioner

    def select_prefill(
        self,
        candidates: list,
        token_budget: int,
        now: float,
        model_runner=None,
        base_chunk: int = 512,
        jmax: int = 0,
        max_selected: int | None = None,
        token_caps_by_id: dict[int, int] | None = None,
    ) -> list:
        if max_selected is not None:
            max_selected = max(0, int(max_selected))
            if max_selected == 0:
                return []
        base_chunk = int(base_chunk)
        jmax = int(jmax)
        if base_chunk <= 0:
            raise ValueError("HoVer base chunk must be positive")
        top_n = max(1, int(getattr(self.config, "hover_top_n_prefill", 16)))
        deadline_guard = float(
            getattr(self.config, "hover_deadline_guard_s", 1.0)
        )
        token_caps_by_id = token_caps_by_id or {}
        forced = []
        scored = []
        for seq in candidates:
            remaining = len(seq) - seq.num_processed_tokens
            if remaining <= 0:
                continue
            block_tokens = self.partitioner.budget_limited_length(
                remaining, token_budget, base_chunk, jmax
            )
            if id(seq) in token_caps_by_id:
                block_tokens = min(
                    block_tokens,
                    max(0, int(token_caps_by_id[id(seq)])),
                )
            if block_tokens <= 0:
                continue
            progress = block_tokens / (remaining + 1e-9)
            urgency = float(getattr(seq, "urgency_score", 0.0))
            gain = urgency * (1.0 - math.exp(-progress))
            density = gain / max(block_tokens, 1)
            deadline = getattr(seq, "ttft_deadline", None)
            slack = deadline - now if deadline is not None else float("inf")
            row = (slack if slack <= deadline_guard else density,
                   block_tokens, seq)
            (forced if slack <= deadline_guard else scored).append(row)
        if not forced and not scored:
            return []
        forced.sort(key=lambda row: row[0])
        scored.sort(key=lambda row: row[0], reverse=True)
        scored = scored[:top_n]
        anchor = forced[0][2] if forced else scored[0][2]
        anchor_modality = self.partitioner.next_modality(anchor)[0]
        scored.sort(
            key=lambda row: (
                0
                if self.partitioner.next_modality(row[2])[0]
                == anchor_modality
                else 1
            )
        )
        try:
            transfer_budget, transfer_cost, per_admission = (
                self._disturbance_budget(
                    candidates, model_runner, self.config
                )
            )
        except Exception as exc:
            logger.warning("HoVer disturbance estimation failed: %s", exc)
            transfer_budget, transfer_cost, per_admission = (
                float("inf"), {}, False
            )
        selected = []
        used_tokens = 0
        used_transfer = 0.0
        for _priority, block_tokens, seq in forced + scored:
            if max_selected is not None and len(selected) >= max_selected:
                break
            candidate_cost = transfer_cost.get(id(seq), 1.0)
            charged_cost = (
                candidate_cost
                if per_admission
                else candidate_cost * block_tokens
            )
            if (
                used_tokens + block_tokens <= token_budget
                and used_transfer + charged_cost <= transfer_budget
            ):
                selected.append(seq)
                used_tokens += block_tokens
                used_transfer += charged_cost
        if not selected:
            pool = forced if forced else scored
            if pool:
                selected = [pool[0][2]]
        return selected

    @staticmethod
    def _modality_weights(seq) -> dict:
        token_modalities = getattr(seq, "token_modalities", None)
        if not token_modalities:
            return {0: 1.0}
        remaining = token_modalities[seq.num_processed_tokens:]
        if not remaining:
            return {0: 1.0}
        counts = Counter(int(modality) for modality in remaining)
        total = sum(counts.values()) or 1
        return {
            modality: count / total
            for modality, count in counts.items()
        }

    def _disturbance_budget(self, candidates: list, model_runner, config):
        predictor = (
            getattr(model_runner, "history_aware_prediction", None)
            if model_runner is not None else None
        )
        if predictor is None or not getattr(predictor, "warm", False):
            return (float("inf"), {}, False)
        num_experts = predictor.E
        resident = {}
        total_transfer_ms = 0.0
        total_loads = 0
        top_k = 0
        indexed = getattr(model_runner, "_hover_moe_by_layer", None)
        if indexed:
            modules = indexed.items()
        else:
            from nanovllm.layers.fused_moe import FusedMoE
            modules = (
                (getattr(module, "_moe_layer_idx", -1), module)
                for _name, module in model_runner.model.named_modules()
                if isinstance(module, FusedMoE)
            )
        for layer_idx, module in modules:
            layer_idx = int(layer_idx)
            cache = getattr(module, 'expert_cache', None)
            if layer_idx >= 0 and cache is not None:
                resident[layer_idx] = {
                    int(expert)
                    for expert in cache.pinned
                    if 0 <= int(expert) < num_experts
                }
                total_transfer_ms += float(
                    getattr(cache, "transfer_time_ms", 0.0) or 0.0
                )
                total_loads += int(
                    getattr(cache, "miss_count", 0) or 0
                )
            if top_k == 0:
                top_k = int(
                    getattr(module, "top_k", 0)
                    or getattr(module, "topk", 0)
                    or 0
                )
        if not resident:
            return (float("inf"), {}, False)
        if top_k <= 0:
            top_k = int(
                getattr(
                    getattr(config, "hf_config", None),
                    "num_experts_per_tok",
                    6,
                )
                or 6
            )
        decode_demand = {}
        for layer_idx in resident:
            demand = predictor.get_layer_demand(layer_idx)
            if demand is not None:
                decode_demand[layer_idx] = demand / (
                    demand.sum() + 1e-9
                )
        ms_per_expert = (
            total_transfer_ms / total_loads
            if total_loads > 0
            else float(
                getattr(config, "hover_c2_ms_per_expert", 0.5) or 0.5
            )
        )
        transfer_budget = float(
            getattr(config, "hover_c2_budget_ms", 0.0) or 0.0
        )
        if transfer_budget <= 0.0:
            transfer_budget = 0.5 * float(
                getattr(config, "tbt_slo_ms", 100.0) or 100.0
            )
        transfer_cost = {}
        for seq in candidates:
            modality_weights = self._modality_weights(seq)
            positive_nonresident_mass = 0.0
            for layer_idx, demand in decode_demand.items():
                prefill_prior = predictor.modality_prior(
                    layer_idx, modality_weights
                )
                delta = prefill_prior - demand
                cached = resident[layer_idx]
                if cached:
                    delta = delta.copy()
                    delta[np.fromiter(cached, dtype=np.int64)] = 0.0
                positive_nonresident_mass += float(
                    np.clip(delta, 0.0, None).sum()
                )
            transfer_cost[id(seq)] = (
                ms_per_expert * top_k * positive_nonresident_mass
            )
        return (transfer_budget, transfer_cost, True)
