# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, Optional, Union, overload

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parameter import UninitializedParameter

from nanovllm.layers.custom_all_reduce import tensor_model_parallel_all_reduce
# from nanovllm.layers.nccl_communicator import tensor_model_parallel_all_reduce
from nanovllm.layers.for_moe.utils import set_weight_attrs
from nanovllm.layers.for_moe.fused_moe import fused_experts


fused_moe_pallas = None  # type: ignore


@dataclass
class _ExactH2DOverlapState:
    """Opaque state joining one exact routing decision to one cache ticket."""

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    required_experts: list[int]
    cache_ticket: Any

# ============================================================================

# ============================================================================
























try:
    from nanovllm.layers.resident_expert_prefetcher import (
        RouteExactExpertTransferOverlap as _RouteExactExpertTransferOverlap,
    )
except Exception:
    _RouteExactExpertTransferOverlap = None
try:
    from nanovllm.layers.physical_expert_cache import (
        PhysicalExpertCache as _PhysicalExpertCache,
    )
except Exception:
    _PhysicalExpertCache = None



# ModelRunner sets this process-local construction policy immediately before
# instantiating a model.  Each TP rank owns one process, so no cross-rank or
# cross-model mutable state is shared.  The policy is cleared before checkpoint
# loading starts.
_PHYSICAL_EXPERT_BUILD_CONFIG = None


def configure_physical_expert_build(
    *, enabled: bool, capacity: int = -1, host_pinned: bool = True
) -> None:
    """Select CPU-first expert allocation for the next model construction."""
    global _PHYSICAL_EXPERT_BUILD_CONFIG
    if not enabled:
        _PHYSICAL_EXPERT_BUILD_CONFIG = None
        return
    if int(capacity) <= 0:
        raise ValueError("physical expert build capacity must be positive")
    _PHYSICAL_EXPERT_BUILD_CONFIG = {
        "capacity": int(capacity),
        "host_pinned": bool(host_pinned),
    }


def _pack_hover_phase_experts(topk_ids: torch.Tensor, len_prefill: int):
    """Copy only compact HoVer routing metadata from device to host.

    ModelRunner lays rows out as ``[prefill rows, decode rows]``.  The old
    path copied every row to NumPy and then copied ``torch.unique`` again for
    residency.  For long multimodal prefills that introduced a device sync
    and a host transfer proportional to the full prompt at every MoE layer.

    This helper compacts the large prefill phase on device, but deliberately
    does not run ``torch.unique`` for the small decode phase.  Decode rows must
    already be copied for the history-aware demand update, so deriving their unique set on
    the host avoids one synchronising GPU kernel per MoE layer.  The compact
    prefill unique values and decode rows share one host transfer, and their
    union is reused for residency.
    """
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be 2-D, got shape={tuple(topk_ids.shape)}")
    num_rows = int(topk_ids.shape[0])
    top_k = int(topk_ids.shape[1])
    if top_k <= 0:
        raise ValueError("topk_ids must contain at least one expert per row")
    len_prefill = int(len_prefill)
    if len_prefill < 0 or len_prefill > num_rows:
        raise ValueError(
            f"invalid len_prefill={len_prefill} for {num_rows} routing rows"
        )

    pre_rows = topk_ids[:len_prefill]
    dec_rows = topk_ids[len_prefill:]
    pre_unique = (
        torch.unique(pre_rows)
        if len_prefill > 0
        else topk_ids.new_empty((0,))
    )
    dec_flat = dec_rows.reshape(-1)
    packed = torch.cat((pre_unique, dec_flat), dim=0)
    host = packed.detach().cpu().tolist()

    n_pre_unique = int(pre_unique.numel())
    pre_values = host[:n_pre_unique]
    dec_flat_values = host[n_pre_unique:]
    # ``torch.unique`` returned sorted values.  Preserve that exact order so
    # RouteExactExpertTransferOverlap sees the same ensure/touch sequence as the old path.
    dec_values = sorted(set(dec_flat_values))
    decode_rows = (
        [dec_flat_values[i:i + top_k]
         for i in range(0, len(dec_flat_values), top_k)]
        if top_k > 0 else []
    )
    # The pre-HoVer cache path used ``torch.unique(topk_ids)``, which returns
    # one globally sorted vector across both phases.  Sorting only within the
    # prefill/decode partitions changes LRU touch/admission order whenever a
    # smaller decode expert follows a larger prefill expert.  Restore the
    # exact global cache order while retaining the single packed D2H copy.
    all_unique = sorted(set(pre_values).union(dec_values))
    return set(pre_values), set(dec_values), decode_rows, all_unique

# ============================================================================

# ============================================================================











import os




# ============================================================================

# ============================================================================

@dataclass
class FusedMoEParallelConfig:
    tp_size: int
    tp_rank: int

    @staticmethod
    def make():
        tp_size = dist.get_world_size()
        tp_rank = dist.get_rank()
        return FusedMoEParallelConfig(tp_size=tp_size, tp_rank=tp_rank)

@dataclass
class MoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int

    num_local_experts: int
    moe_parallel_config: FusedMoEParallelConfig

    in_dtype: torch.dtype  
    quant_dtype: torch.dtype = None

    
    block_size: int = 128

    max_num_tokens: int = 256

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


class FusedMoEMethodBase(ABC):

    moe: MoEConfig

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        raise NotImplementedError

    def init_prepare_finalize(self, moe: MoEConfig):
        self.moe = moe
        self.topk_indices_dtype = None

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
        preselected_topk: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        exact_overlap_ticket: Any = None,
        exact_overlap_experts: Optional[list[int]] = None,
        exact_route_split: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError


# count_tensor: torch.Tensor
# count: torch.Tensor

class UnquantizedFusedMoEMethod(FusedMoEMethodBase):

    def __init__(self, moe: MoEConfig):
        super().__init__()
        self.fused_experts = fused_experts  
        self.topk_indices_dtype = None  
        self.moe = moe

        
        self.rocm_aiter_fused_experts = None  # type: ignore

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype,
                       has_bias: bool = False,
                       **extra_weight_attrs):
        physical_build = _PHYSICAL_EXPERT_BUILD_CONFIG
        gpu_num_experts = num_experts
        if physical_build is not None:
            gpu_num_experts = min(
                int(num_experts), int(physical_build["capacity"])
            )

        
        
        w13_weight = torch.nn.Parameter(torch.empty(
            gpu_num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        
        if has_bias:
            w13_bias = torch.nn.Parameter(torch.zeros(
                gpu_num_experts,
                2 * intermediate_size_per_partition,
                dtype=params_dtype),
                                          requires_grad=False)
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

        
        w2_weight = torch.nn.Parameter(torch.empty(
            gpu_num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        
        if has_bias:
            w2_bias = torch.nn.Parameter(torch.zeros(gpu_num_experts,
                                                     hidden_size,
                                                     dtype=params_dtype),
                                         requires_grad=False)
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

        if physical_build is not None:
            def _cpu_master(shape):
                kwargs = {
                    "dtype": params_dtype,
                    "device": "cpu",
                    "pin_memory": bool(physical_build["host_pinned"]),
                }
                try:
                    return torch.empty(shape, **kwargs)
                except RuntimeError:
                    # Pageable masters remain correct for the synchronous v2
                    # baseline.  Telemetry exposes whether pinning succeeded.
                    kwargs["pin_memory"] = False
                    return torch.empty(shape, **kwargs)

            layer._physical_host_load_enabled = True
            layer._physical_num_experts = int(num_experts)
            layer._physical_host_pinned_requested = bool(
                physical_build["host_pinned"]
            )
            layer._physical_loaded_expert_shards = {
                "w1": set(), "w2": set(), "w3": set()
            }
            layer._physical_w13_host = _cpu_master((
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
            ))
            layer._physical_w2_host = _cpu_master((
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
            ))
            layer._physical_w13_bias_host = (
                _cpu_master((num_experts, 2 * intermediate_size_per_partition))
                if has_bias else None
            )
            layer._physical_w2_bias_host = (
                _cpu_master((num_experts, hidden_size))
                if has_bias else None
            )

        
        # global count_tensor, count
        # count_tensor = torch.zeros(num_experts, dtype=torch.int32).cuda()
        # count = torch.zeros(num_experts, dtype=torch.int32).cuda()

    def _apply_exact_route_split(
        self,
        *,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        cache,
        ticket,
        activation: str,
        apply_router_weight_on_input: bool,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute ready routes during exact H2D, then blocked routes.

        The fused kernel consumes a rectangular ``[tokens, top_k]`` route
        matrix.  Flattening it into two ``[route, 1]`` sub-batches lets the
        ready routes perform useful work while true misses are copied.  Every
        original route is evaluated once, then reduced in its original order.
        """

        def _run(
            route_x: torch.Tensor,
            route_weights: torch.Tensor,
            route_ids: torch.Tensor,
            *,
            inplace: bool,
        ) -> torch.Tensor:
            return self.fused_experts(
                hidden_states=route_x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                w1_bias=(
                    layer.w13_bias if hasattr(layer, "w13_bias") else None
                ),
                w2_bias=(
                    layer.w2_bias if hasattr(layer, "w2_bias") else None
                ),
                topk_weights=route_weights,
                topk_ids=route_ids,
                inplace=inplace,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )

        waited = False
        try:
            ready_experts, blocked_experts = (
                cache.partition_forward_exact_async(ticket)
            )
            flat_ids = topk_ids.reshape(-1)
            flat_weights = topk_weights.reshape(-1)
            total_routes = int(flat_ids.numel())

            ready_lookup = torch.zeros(
                cache.num_experts,
                dtype=torch.bool,
                device=topk_ids.device,
            )
            if ready_experts:
                ready_lookup[
                    torch.tensor(
                        ready_experts,
                        dtype=torch.long,
                        device=topk_ids.device,
                    )
                ] = True
            ready_mask = ready_lookup[flat_ids.to(dtype=torch.long)]
            ready_indices = torch.nonzero(
                ready_mask, as_tuple=False
            ).reshape(-1)
            blocked_indices = torch.nonzero(
                ~ready_mask, as_tuple=False
            ).reshape(-1)
            ready_routes = int(ready_indices.numel())
            blocked_routes = int(blocked_indices.numel())

            min_ready = max(
                1,
                int(
                    getattr(
                        layer,
                        "hover_route_split_min_ready_routes",
                        8,
                    )
                ),
            )
            max_routes = max(
                1,
                int(
                    getattr(
                        layer,
                        "hover_route_split_max_routes",
                        8192,
                    )
                ),
            )
            fallback_reason = None
            if ticket.legacy_fallback:
                fallback_reason = "legacy_copy_fallback"
            elif not blocked_experts or blocked_routes == 0:
                fallback_reason = "no_blocked_routes"
            elif ready_routes == 0:
                fallback_reason = "no_ready_routes"
            elif ready_routes < min_ready:
                fallback_reason = "ready_below_threshold"
            elif total_routes > max_routes:
                fallback_reason = "route_limit"

            if fallback_reason is not None:
                cache.record_exact_route_split(
                    used=False,
                    ready_routes=ready_routes,
                    blocked_routes=blocked_routes,
                    fallback_reason=fallback_reason,
                )
                cache.wait_forward_exact_async(ticket)
                waited = True
                return _run(
                    x,
                    topk_weights,
                    topk_ids,
                    inplace=True,
                )

            cache.record_exact_route_split(
                used=True,
                ready_routes=ready_routes,
                blocked_routes=blocked_routes,
            )
            top_k = int(topk_ids.shape[1])
            token_for_route = torch.arange(
                int(topk_ids.shape[0]),
                dtype=torch.long,
                device=topk_ids.device,
            ).repeat_interleave(top_k)
            route_outputs = torch.empty(
                (total_routes, int(x.shape[-1])),
                dtype=x.dtype,
                device=x.device,
            )

            ready_tokens = token_for_route.index_select(0, ready_indices)
            ready_output = _run(
                x.index_select(0, ready_tokens).contiguous(),
                flat_weights.index_select(0, ready_indices)
                .reshape(-1, 1)
                .contiguous(),
                flat_ids.index_select(0, ready_indices)
                .reshape(-1, 1)
                .contiguous(),
                inplace=False,
            )
            route_outputs.index_copy_(0, ready_indices, ready_output)

            cache.wait_forward_exact_async(ticket)
            waited = True

            blocked_tokens = token_for_route.index_select(
                0, blocked_indices
            )
            blocked_output = _run(
                x.index_select(0, blocked_tokens).contiguous(),
                flat_weights.index_select(0, blocked_indices)
                .reshape(-1, 1)
                .contiguous(),
                flat_ids.index_select(0, blocked_indices)
                .reshape(-1, 1)
                .contiguous(),
                inplace=False,
            )
            route_outputs.index_copy_(
                0, blocked_indices, blocked_output
            )
            return route_outputs.view(
                int(topk_ids.shape[0]), top_k, int(x.shape[-1])
            ).sum(dim=1)
        except BaseException:
            if not waited:
                cache.abort_forward_exact_async(ticket)
            raise
        finally:
            if waited:
                cache.post_forward_exact_async(ticket)

    def _apply_physical_cache(
        self,
        *,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        required_experts: list[int],
        cache,
        activation: str,
        apply_router_weight_on_input: bool,
        expert_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run one compact-slot kernel or route waves when capacity is tight."""
        if expert_map is not None:
            raise RuntimeError(
                "physical-cpu-first does not yet support expert parallel maps"
            )
        flat_ids = topk_ids.reshape(-1)
        required_set = set(int(eid) for eid in required_experts)
        # Consume current hits before miss waves are allowed to evict them.
        # The remaining IDs are sorted for deterministic TP-rank behavior.
        resident_required = [
            eid for eid in cache.gpu_resident if eid in required_set
        ]
        required = resident_required + sorted(
            required_set.difference(resident_required)
        )
        capacity = int(cache.capacity)
        num_waves = (len(required) + capacity - 1) // capacity
        cache.note_forward_waves(num_waves)

        def _run(
            route_x: torch.Tensor,
            route_weights: torch.Tensor,
            route_ids: torch.Tensor,
            *,
            inplace: bool,
        ) -> torch.Tensor:
            return self.fused_experts(
                hidden_states=route_x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                w1_bias=(
                    layer.w13_bias if hasattr(layer, "w13_bias") else None
                ),
                w2_bias=(
                    layer.w2_bias if hasattr(layer, "w2_bias") else None
                ),
                topk_weights=route_weights,
                topk_ids=route_ids,
                inplace=inplace,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                # The kernel sees physical slots, not logical experts.
                global_num_experts=capacity,
                expert_map=None,
            )

        if num_waves <= 1:
            try:
                mapping = cache.prepare_wave(required)
                physical_ids = cache.remap_ids(topk_ids, mapping).contiguous()
                return _run(
                    x, topk_weights, physical_ids, inplace=True
                )
            finally:
                cache.finish_forward()

        top_k = int(topk_ids.shape[1])
        total_routes = int(flat_ids.numel())
        token_for_route = torch.arange(
            int(topk_ids.shape[0]),
            dtype=torch.long,
            device=topk_ids.device,
        ).repeat_interleave(top_k)
        route_outputs = torch.empty(
            (total_routes, int(x.shape[-1])),
            dtype=x.dtype,
            device=x.device,
        )
        try:
            for start in range(0, len(required), capacity):
                wave = required[start:start + capacity]
                mapping = cache.prepare_wave(wave)
                wave_lookup = torch.zeros(
                    cache.num_experts,
                    dtype=torch.bool,
                    device=topk_ids.device,
                )
                wave_lookup[
                    torch.tensor(
                        wave, dtype=torch.long, device=topk_ids.device
                    )
                ] = True
                route_indices = torch.nonzero(
                    wave_lookup[flat_ids.to(dtype=torch.long)], as_tuple=False
                ).reshape(-1)
                logical_route_ids = flat_ids.index_select(0, route_indices)
                physical_route_ids = cache.remap_ids(
                    logical_route_ids, mapping
                ).reshape(-1, 1).contiguous()
                wave_output = _run(
                    x.index_select(
                        0, token_for_route.index_select(0, route_indices)
                    ).contiguous(),
                    topk_weights.reshape(-1)
                    .index_select(0, route_indices)
                    .reshape(-1, 1)
                    .contiguous(),
                    physical_route_ids,
                    inplace=False,
                )
                route_outputs.index_copy_(0, route_indices, wave_output)
            return route_outputs.view(
                int(topk_ids.shape[0]), top_k, int(x.shape[-1])
            ).sum(dim=1)
        finally:
            cache.finish_forward()

    def _apply_logical_cache_strict_waves(
        self,
        *,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        required_experts: list[int],
        cache,
        activation: str,
        apply_router_weight_on_input: bool,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Execute logical expert routes in waves that respect cache capacity.

        The legacy logical cache keeps full expert tensors allocated, but a
        fused call can transiently mark every routed expert resident.  This
        path preserves the full tensors while splitting routed work so each
        cache transaction contains at most ``capacity`` resident experts,
        including pinned experts that occupy persistent slots.
        """
        if expert_map is not None:
            raise RuntimeError(
                "logical-strict-wave does not support expert parallel maps"
            )
        capacity = int(cache.capacity)
        if capacity <= 0:
            raise RuntimeError(
                "logical-strict-wave requires a positive cache capacity"
            )

        required_set = set(int(eid) for eid in required_experts)
        pinned = [
            int(eid) for eid in cache.pinned
            if 0 <= int(eid) < int(cache.num_experts)
        ]
        pinned_set = set(pinned)
        if not pinned_set.issubset(set(cache.gpu_resident)):
            raise RuntimeError(
                "logical-strict-wave requires every pinned expert to be "
                "resident before route execution"
            )
        available = capacity - len(pinned_set)
        required_pinned = [eid for eid in pinned if eid in required_set]
        resident_nonpinned = [
            int(eid) for eid in cache.gpu_resident
            if int(eid) in required_set and int(eid) not in pinned_set
        ]
        missing_nonpinned = sorted(
            required_set.difference(pinned_set).difference(
                resident_nonpinned
            )
        )
        nonpinned = resident_nonpinned + missing_nonpinned
        if nonpinned and available <= 0:
            raise RuntimeError(
                "logical-strict-wave has no working slot because pinned "
                "experts fill the entire cache"
            )

        route_waves: list[list[int]] = []
        if nonpinned:
            for start in range(0, len(nonpinned), available):
                route_waves.append(nonpinned[start:start + available])
        else:
            route_waves.append([])
        # Required pinned routes are computed exactly once.  Pinned experts
        # still occupy cache slots in every later transaction automatically.
        route_waves[0] = required_pinned + route_waves[0]
        route_waves = [wave for wave in route_waves if wave]
        if not route_waves:
            raise RuntimeError("logical-strict-wave received no routes")

        num_waves = len(route_waves)
        cache.strict_wave_forward_count = int(
            getattr(cache, "strict_wave_forward_count", 0)
        ) + 1
        cache.strict_total_waves = int(
            getattr(cache, "strict_total_waves", 0)
        ) + num_waves
        cache.strict_max_waves_per_forward = max(
            int(getattr(cache, "strict_max_waves_per_forward", 0)),
            num_waves,
        )
        if num_waves > 1:
            cache.strict_multi_wave_forward_count = int(
                getattr(cache, "strict_multi_wave_forward_count", 0)
            ) + 1

        def _run(
            route_x: torch.Tensor,
            route_weights: torch.Tensor,
            route_ids: torch.Tensor,
            *,
            inplace: bool,
        ) -> torch.Tensor:
            return self.fused_experts(
                hidden_states=route_x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                w1_bias=(
                    layer.w13_bias if hasattr(layer, "w13_bias") else None
                ),
                w2_bias=(
                    layer.w2_bias if hasattr(layer, "w2_bias") else None
                ),
                topk_weights=route_weights,
                topk_ids=route_ids,
                inplace=inplace,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=None,
            )

        if num_waves == 1:
            opened = False
            try:
                cache.begin_forward(route_waves[0])
                opened = True
                return _run(x, topk_weights, topk_ids, inplace=True)
            finally:
                if opened:
                    cache.post_forward(route_waves[0])

        flat_ids = topk_ids.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        top_k = int(topk_ids.shape[1])
        total_routes = int(flat_ids.numel())
        token_for_route = torch.arange(
            int(topk_ids.shape[0]),
            dtype=torch.long,
            device=topk_ids.device,
        ).repeat_interleave(top_k)
        route_outputs = torch.empty(
            (total_routes, int(x.shape[-1])),
            dtype=x.dtype,
            device=x.device,
        )
        for wave in route_waves:
            opened = False
            try:
                cache.begin_forward(wave)
                opened = True
                wave_lookup = torch.zeros(
                    cache.num_experts,
                    dtype=torch.bool,
                    device=topk_ids.device,
                )
                wave_lookup[
                    torch.tensor(
                        wave, dtype=torch.long, device=topk_ids.device
                    )
                ] = True
                route_indices = torch.nonzero(
                    wave_lookup[flat_ids.to(dtype=torch.long)],
                    as_tuple=False,
                ).reshape(-1)
                route_tokens = token_for_route.index_select(
                    0, route_indices
                )
                wave_output = _run(
                    x.index_select(0, route_tokens).contiguous(),
                    flat_weights.index_select(0, route_indices)
                    .reshape(-1, 1)
                    .contiguous(),
                    flat_ids.index_select(0, route_indices)
                    .reshape(-1, 1)
                    .contiguous(),
                    inplace=False,
                )
                route_outputs.index_copy_(0, route_indices, wave_output)
            finally:
                if opened:
                    cache.post_forward(wave)
        return route_outputs.view(
            int(topk_ids.shape[0]), top_k, int(x.shape[-1])
        ).sum(dim=1)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
        preselected_topk: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        exact_overlap_ticket: Any = None,
        exact_overlap_experts: Optional[list[int]] = None,
        exact_route_split: bool = False,
    ) -> torch.Tensor:

        
        
        if preselected_topk is None:
            topk_weights, topk_ids = FusedMoE.select_experts(
                hidden_states=x,
                router_logits=router_logits,
                use_grouped_topk=use_grouped_topk,
                top_k=top_k,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
                indices_type=self.topk_indices_dtype)
        else:
            topk_weights, topk_ids = preselected_topk

        
        
        

        
        layer_idx = getattr(layer, "_moe_layer_idx", -1)

        _ctx = None
        try:
            from nanovllm.utils.context import get_context as _get_ctx
            _ctx = _get_ctx()
        except Exception:
            pass

        
        _cache = getattr(layer, 'expert_cache', None)

        
        
        _tphase = getattr(_ctx, 'token_phase', None) if _ctx is not None else None
        _tsid = getattr(_ctx, 'token_seqid', None) if _ctx is not None else None
        try:
            from nanovllm.engine.model_runner import _GLOBAL_MODEL_RUNNER as _global_runner
        except Exception:
            _global_runner = None
        # Prefer forward-scoped deferral, but retain immediate recording if a
        # rolling deployment temporarily pairs this layer file with an older
        # ModelRunner.  Silently dropping predictor/TTL history is not safe.
        _record_decode = (
            getattr(_global_runner, 'defer_decode_batch', None)
            or getattr(_global_runner, 'record_decode_batch', None)
        ) if _global_runner is not None else None
        _decode_record_due = True
        if _global_runner is not None:
            _record_due_fn = getattr(_global_runner, '_decode_record_due', None)
            if callable(_record_due_fn):
                _decode_record_due = bool(
                    _record_due_fn(int(getattr(_global_runner, '_hover_round', 0)))
                )
        _resident_experts = None
        if (_record_decode is not None
                and getattr(_global_runner, 'history_aware_prediction', None) is not None
                and _tphase is not None and len(_tphase) == int(topk_ids.shape[0])):
            try:
                _lp = int(getattr(_ctx, 'len_prefill', 0) or 0)
                _layout_ok = (
                    0 <= _lp <= len(_tphase)
                    and (_lp == 0 or (
                        _tphase[0] == 'P' and _tphase[_lp - 1] == 'P'
                    ))
                    and (_lp == len(_tphase) or (
                        _tphase[_lp] == 'D' and _tphase[-1] == 'D'
                    ))
                )
                if _layout_ok:
                    if not _decode_record_due and _lp == 0:
                        # Stable pure-decode fast path: the cache and phase
                        # counters need only the expert union.  Avoid copying
                        # every per-request routing row to Python when the
                        # control plane intentionally does not consume it.
                        _resident_experts = (
                            torch.unique(topk_ids).detach().cpu().tolist()
                        )
                        if _cache is not None:
                            _cache.record_phase_stats(
                                set(_resident_experts), set()
                            )
                    else:
                        _pre, _dec, _drows, _resident_experts = (
                            _pack_hover_phase_experts(topk_ids, _lp)
                        )
                        if _cache is not None:
                            _cache.record_phase_stats(_dec, _pre)
                        if _drows and _decode_record_due:
                            _drids = (
                                list(_tsid[_lp:_lp + len(_drows)])
                                if _tsid is not None and len(_tsid) >= _lp + len(_drows)
                                else list(range(_lp, _lp + len(_drows)))
                            )
                            _record_decode(
                                layer_idx,
                                _drids,
                                _drows,
                                union_experts=list(_dec),
                            )
                else:
                    # Defensive compatibility for any future model that does
                    # not preserve the documented P-first/D-last row layout.
                    import numpy as _np
                    _rows_np = topk_ids.detach().cpu().numpy()
                    _ph_np = _np.asarray(_tphase)
                    _dmask = (_ph_np == 'D')
                    _dec = set(_rows_np[_dmask].reshape(-1).tolist()) if _dmask.any() else set()
                    _pre = set(_rows_np[~_dmask].reshape(-1).tolist()) if (~_dmask).any() else set()
                    # Match the globally sorted order returned by the former
                    # ``torch.unique(topk_ids)`` cache path even on a defensive
                    # non P-first/D-last layout.
                    _resident_experts = sorted(_pre | _dec)
                    if _cache is not None:
                        _cache.record_phase_stats(_dec, _pre)
                    if _dmask.any():
                        _didx = _np.nonzero(_dmask)[0]
                        _drids = [_tsid[i] for i in _didx] if _tsid else _didx.tolist()
                        _drows = _rows_np[_didx].tolist()
                        _record_decode(
                            layer_idx,
                            _drids,
                            _drows,
                            union_experts=list(_dec),
                        )
            except Exception:
                pass

        _cache_forward_experts = None
        if _cache is not None:
            if exact_overlap_experts is not None:
                _cache_forward_experts = list(exact_overlap_experts)
                if (
                    _resident_experts is not None
                    and set(_resident_experts) != set(_cache_forward_experts)
                ):
                    raise RuntimeError(
                        "exact-overlap routing/cache expert sets diverged"
                    )
            else:
                _cache_forward_experts = (
                    _resident_experts if _resident_experts is not None
                    else torch.unique(topk_ids).detach().cpu().tolist()
                )

        

        
        
        
        if getattr(_cache, "is_physical", False):
            if exact_overlap_ticket is not None or exact_route_split:
                raise RuntimeError(
                    "physical-cpu-first is incompatible with exact overlap"
                )
            return self._apply_physical_cache(
                layer=layer,
                x=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                required_experts=_cache_forward_experts,
                cache=_cache,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                expert_map=expert_map,
            )

        if (
            _cache is not None
            and getattr(layer, "strict_logical_expert_capacity", False)
        ):
            if exact_overlap_ticket is not None or exact_route_split:
                raise RuntimeError(
                    "logical-strict-wave is incompatible with exact overlap"
                )
            return self._apply_logical_cache_strict_waves(
                layer=layer,
                x=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                required_experts=_cache_forward_experts,
                cache=_cache,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )

        if exact_overlap_ticket is not None and exact_route_split:
            if _cache is None or _cache_forward_experts is None:
                raise RuntimeError(
                    "route-split overlap requires an RouteExactExpertTransferOverlap and exact experts"
                )
            return self._apply_exact_route_split(
                layer=layer,
                x=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                cache=_cache,
                ticket=exact_overlap_ticket,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )

        _cache_forward_opened = False
        _cache_exact_opened = False
        if exact_overlap_ticket is not None:
            if _cache is None or _cache_forward_experts is None:
                raise RuntimeError(
                    "exact-overlap ticket requires an RouteExactExpertTransferOverlap and exact experts"
                )
            try:
                _cache.wait_forward_exact_async(exact_overlap_ticket)
                _cache_exact_opened = True
            except BaseException:
                _cache.abort_forward_exact_async(exact_overlap_ticket)
                raise
        elif _cache is not None and _cache_forward_experts is not None:
            # HoVer forward setup. begin_forward also
            # performs touch; copy/touch failures clean themselves and never
            # open the transaction.  No fallible work sits between a
            # successful begin and the try/finally below.
            _cache.begin_forward(_cache_forward_experts)
            _cache_forward_opened = True
        try:
            return self.fused_experts(
                    hidden_states=x,
                    w1=layer.w13_weight,  
                    w2=layer.w2_weight,   
                    w1_bias=layer.w13_bias if hasattr(layer, "w13_bias") else None,
                    w2_bias=layer.w2_bias if hasattr(layer, "w2_bias") else None,
                    topk_weights=topk_weights,  
                    topk_ids=topk_ids,          
                    inplace=True,               
                    activation=activation,      
                    apply_router_weight_on_input=apply_router_weight_on_input,
                    global_num_experts=global_num_experts,
                    expert_map=expert_map,
                )
        finally:
            # The kernel is asynchronous.  post_forward records a compute-
            # stream event before trimming metadata, so later copy-stream DMA
            # cannot race the kernel.  The finally path also covers a kernel
            # launch failure.
            if _cache_exact_opened:
                _cache.post_forward_exact_async(exact_overlap_ticket)
            elif _cache_forward_opened:
                _cache.post_forward(_cache_forward_experts)

class FusedMoE(torch.nn.Module):

    def __init__(
        self,
        num_experts: int,  
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        num_redundant_experts: int = 0,
        has_bias=False,
    ):
        super().__init__()
        
        self._moe_layer_idx = -1

        # OFFLOAD
        self.offload_strategy: str = "none"
        self.gpu_expert_cache_slots: int = -1
        self.expert_cache = None  
        self.hover_exact_h2d_overlap = False
        self.strict_logical_expert_capacity = False
        self.hover_route_split_min_ready_routes = 8
        self.hover_route_split_max_routes = 8192
        self._w13_host = None
        self._w2_host = None

        
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        
        self.moe_parallel_config: FusedMoEParallelConfig = (
            FusedMoEParallelConfig.make())

        
        
        self.global_num_experts = num_experts + num_redundant_experts
        # Logical expert count must never be inferred from compact GPU weight
        # shapes.  ModelRunner predictors consume this stable public field.
        self.num_experts = self.global_num_experts

        
        self.layer_name = prefix

        
        self.expert_load_view: Optional[torch.Tensor] = None
        self.logical_to_physical_map: Optional[torch.Tensor] = None
        self.logical_replica_count: Optional[torch.Tensor] = None

        
        
        self.local_num_experts, self.expert_map = (self.global_num_experts,
                                                       None)

        
        self.top_k = top_k

        
        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        
        self.intermediate_size_per_partition = intermediate_size // self.tp_size

        
        self.reduce_results = reduce_results
        
        self.renormalize = renormalize

        
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group

        
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation

        
        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")

        
        quant_dtype = params_dtype

        
        moe = MoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=params_dtype,
            quant_dtype=quant_dtype,
            max_num_tokens=256,
        )
        self.moe_config = moe

        
        
        
        quant_method = UnquantizedFusedMoEMethod(moe)
        self.quant_method = quant_method

        
        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition":
            self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
            "has_bias": has_bias,
        }

        
        self.quant_method.create_weights(layer=self, **moe_quant_params)

        
        self.batched_hidden_states: Optional[torch.Tensor] = None
        self.batched_router_logits: Optional[torch.Tensor] = None

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    def _load_per_tensor_weight_scale(self, shard_id: str,
                                      param: torch.nn.Parameter,
                                      loaded_weight: torch.Tensor,
                                      expert_id: int):
        param_data = param.data
        
        if shard_id in ("w1", "w3"):
            
            
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_model_weight_or_group_weight_scale(self,
                                                 shard_dim: int,
                                                 expert_data: torch.Tensor,
                                                 shard_id: str,
                                                 loaded_weight: torch.Tensor,
                                                 tp_rank: int,
                                                 load_full_w2: bool = False):
        if expert_data.ndim != loaded_weight.ndim:
            loaded_weight = loaded_weight.reshape(*loaded_weight.shape[:-2], -1)

        if shard_id == "w2":
            # In the case where we have actorder/g_idx, we do not partition the
            # w2 scales, as indicated by `load_full` argument, for all tp cases
            self._load_w2(shard_dim=shard_dim,
                          loaded_weight=loaded_weight,
                          expert_data=expert_data,
                          tp_rank=tp_rank,
                          load_full=load_full_w2)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)
        elif shard_id in ("w13",):
            # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
            shard_size = expert_data.shape[1]
            loaded_w = loaded_weight.narrow(1, shard_size * tp_rank, shard_size)
            expert_data.narrow(1, 0, shard_size).copy_(loaded_w)

    def _load_per_channel_weight_scale(self, expert_data: torch.Tensor,
                                       shard_dim: int, shard_id: str,
                                       loaded_weight: torch.Tensor,
                                       tp_rank: int):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)

    def _load_w13(self, expert_data: torch.Tensor, shard_dim: int,
                  shard_id: str, loaded_weight: torch.Tensor, tp_rank: int):

        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        shard_size = expert_data.shape[shard_dim] // 2
        loaded_weight = loaded_weight.narrow(shard_dim, shard_size * tp_rank,
                                             shard_size)
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(self,
                 expert_data: torch.Tensor,
                 shard_dim: int,
                 loaded_weight: torch.Tensor,
                 tp_rank: int,
                 load_full: bool = False):

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(shard_dim,
                                                 shard_size * tp_rank,
                                                 shard_size)
        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _load_single_value(self, param: torch.nn.Parameter,
                           loaded_weight: torch.Tensor, expert_id: int):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(self, shard_id: str, expert_data: torch.Tensor,
                    shard_dim: int, loaded_weight: torch.Tensor, tp_rank: int):

        if shard_id == "w2":
            self._load_w2(shard_dim=shard_dim,
                          loaded_weight=loaded_weight,
                          expert_data=expert_data,
                          tp_rank=tp_rank)
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

    @overload
    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int,
                      return_success: Literal[False]) -> None:
        ...

    @overload
    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int,
                      return_success: Literal[True]) -> bool:
        ...

    def weight_loader(self,
                      param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor,
                      weight_name: str,
                      shard_id: str,
                      expert_id: int,
                      return_success: bool = False) -> Optional[bool]:
        if shard_id == "all":
            # (FIXME) for gpt-oss all experts are combined
            if "bias" in weight_name:
                param.data.copy_(loaded_weight)
            else:
                param_shape = param.data.shape
                loaded_weight = loaded_weight.reshape(*param_shape[:-1], -1)
                param.data.copy_(loaded_weight)
            return True if return_success else None

        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if expert_id == -1:
            # Failed to load this param since it's not local to this rank
            return False if return_success else None
        # Hereafter, `expert_id` is local physical id

        quant_method_name = self.quant_method.__class__.__name__

        physical_host_data = None
        if getattr(self, "_physical_host_load_enabled", False):
            if quant_method_name != "UnquantizedFusedMoEMethod":
                raise RuntimeError(
                    "physical-cpu-first supports unquantized FusedMoE only"
                )
            if "bias" in weight_name:
                raise RuntimeError(
                    "physical-cpu-first does not yet support expert bias"
                )
            if param is self.w13_weight:
                physical_host_data = self._physical_w13_host
            elif param is self.w2_weight:
                physical_host_data = self._physical_w2_host
            else:
                raise RuntimeError(
                    "physical expert loader received an unknown expert parameter"
                )

        if shard_id not in ("w1", "w2", "w3", "all", "w13"):
            raise ValueError(f"shard_id must be ['w1','w2','w3', 'all'] but "
                             f"got {shard_id}.")

        WEIGHT_SCALE_SUPPORTED = [
            e.value for e in FusedMoeWeightScaleSupported
        ]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0, "w13": 1}

        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()
            param.data.copy_(loaded_weight)
            return True if return_success else None

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        full_load = len(loaded_weight.shape) == 3 or "bias" in weight_name or "block" in weight_name
        if full_load:
            shard_dim += 1

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            final_shape = list(loaded_weight.shape)
            if shard_id in ["w1", "w3"]:
                final_shape[1] *= 2
            final_shape[shard_dim] = final_shape[shard_dim] // self.tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        target_data = (
            physical_host_data
            if physical_host_data is not None else param.data
        )
        expert_data = target_data if full_load else target_data[expert_id]

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if ("compressed" in quant_method_name.lower()
                    and param.data[expert_id] != 1
                    and (param.data[expert_id] - loaded_weight).abs() > 1e-5):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}")

            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return True if return_success else None

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(shard_dim=0,
                             shard_id=shard_id,
                             loaded_weight=loaded_weight,
                             expert_data=expert_data,
                             tp_rank=self.tp_rank)
            return True if return_success else None

        # TODO @dsikka: ModelOpt should follow the proper MoE loading pattern
        if "ModelOpt" in quant_method_name:
            if ('weight_scale_2' in weight_name
                    or 'input_scale' in weight_name):
                self._load_per_tensor_weight_scale(shard_id=shard_id,
                                                   param=param,
                                                   loaded_weight=loaded_weight,
                                                   expert_id=expert_id)
            elif "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank)
            return True if return_success else None

        # Case weight scales, zero_points and offset, weight/input global scales
        if ("scale" in weight_name or "zero" in weight_name
                or "offset" in weight_name):
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank)
            elif quant_method in [
                    FusedMoeWeightScaleSupported.GROUP.value,
                    FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full_w2=getattr(param, "load_full_w2", False))
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(shard_id=shard_id,
                                                   param=param,
                                                   loaded_weight=loaded_weight,
                                                   expert_id=expert_id)
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}")
            return True if return_success else None

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return True if return_success else None

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank)
            if physical_host_data is not None:
                loaded_ids = tuple(
                    range(self.local_num_experts)
                    if full_load else (int(expert_id),)
                )
                if shard_id == "w13":
                    self._physical_loaded_expert_shards["w1"].update(
                        loaded_ids
                    )
                    self._physical_loaded_expert_shards["w3"].update(
                        loaded_ids
                    )
                else:
                    self._physical_loaded_expert_shards[shard_id].update(
                        loaded_ids
                    )
            return True if return_success else None

        # Case model weights
        if "block" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=-1,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank)
            return True if return_success else None

        if "bias" in weight_name:
            if shard_id == "w2":
                if self.tp_rank == 0:
                    expert_data.copy_(loaded_weight)
            else:
                loaded_weight = loaded_weight.narrow(-1, self.tp_rank * loaded_weight.size(-1) // self.tp_size, loaded_weight.size(-1) // self.tp_size)
                expert_data.copy_(loaded_weight)
            return True if return_success else None

        raise ValueError(f"Unrecognized weight_name {weight_name}.")

        return False if return_success else None

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        weights = list(self.named_parameters())
        
        assert all(weight.is_contiguous() for _, weight in weights)

        
        
        NON_EXPERT_WEIGHTS = {
            "e_score_correction_bias",
        }

        return [
            weight.view(self.local_num_experts, -1) for name, weight in weights
            if name not in NON_EXPERT_WEIGHTS
        ]

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.expert_load_view = expert_load_view[moe_layer_idx]
        self.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
        self.logical_replica_count = logical_replica_count[moe_layer_idx]

    @staticmethod
    def select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        indices_type: Optional[torch.dtype] = None,
        expert_map: Optional[torch.Tensor] = None,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from nanovllm.layers.for_moe.fused_moe import fused_topk

        
        if custom_routing_function is None:
            
            topk_weights, topk_ids, token_expert_indices = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
            )
        else:
            
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)
            
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)

        return topk_weights, topk_ids

    def must_reduce_shared_expert_outputs(self) -> bool:
        return False

    def maybe_all_reduce_tensor_model_parallel(
            self, final_hidden_states: torch.Tensor):
        return tensor_model_parallel_all_reduce(final_hidden_states)

    def _apply_quant_method(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        preselected_topk: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        exact_overlap_ticket: Any = None,
        exact_overlap_experts: Optional[list[int]] = None,
        exact_route_split: bool = False,
    ) -> torch.Tensor:
        assert self.quant_method is not None
        return self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            expert_load_view=self.expert_load_view,
            logical_to_physical_map=self.logical_to_physical_map,
            logical_replica_count=self.logical_replica_count,
            preselected_topk=preselected_topk,
            exact_overlap_ticket=exact_overlap_ticket,
            exact_overlap_experts=exact_overlap_experts,
            exact_route_split=exact_route_split,
        )

    def _maybe_reduce_output(self, output: torch.Tensor) -> torch.Tensor:
        if self.reduce_results and self.tp_size > 1:
            return self.maybe_all_reduce_tensor_model_parallel(output)
        return output

    def forward_with_shared_expert_overlap(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_forward: Callable[[], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Overlap exact current-layer H2D copies with shared-expert work.

        The route is selected exactly once.  No predicted expert is copied,
        and the routed kernel is not launched until its cache ticket has
        inserted a dependency on the current compute stream.  Any exception
        before or during the routed kernel closes the cache transaction.
        """
        if not self.hover_exact_h2d_overlap:
            raise RuntimeError("route-exact overlap was called while disabled")
        cache = self.expert_cache
        if cache is None:
            raise RuntimeError("route-exact overlap requires RouteExactExpertTransferOverlap")

        topk_weights, topk_ids = self.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=self.use_grouped_topk,
            top_k=self.top_k,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            indices_type=self.quant_method.topk_indices_dtype,
        )
        required_experts = (
            torch.unique(topk_ids).detach().cpu().tolist()
        )
        state = _ExactH2DOverlapState(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            required_experts=required_experts,
            cache_ticket=cache.begin_forward_exact_async(required_experts),
        )
        try:
            # DeepSeek's shared branch includes its RowParallel all-reduce.
            # Keeping it on the default stream preserves TP collective order;
            # only H2D copies use the cache stream.
            shared_output = shared_forward()
            routed_output = self._apply_quant_method(
                hidden_states,
                router_logits,
                preselected_topk=(state.topk_weights, state.topk_ids),
                exact_overlap_ticket=state.cache_ticket,
                exact_overlap_experts=state.required_experts,
            )
            return self._maybe_reduce_output(routed_output), shared_output
        except BaseException:
            cache.abort_forward_exact_async(state.cache_ticket)
            raise

    def forward_with_route_split_overlap(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Overlap true-miss H2D with useful resident-expert route work.

        This path is model-agnostic for the unquantized ``FusedMoE`` runtime:
        it needs no shared-expert branch and uses the same exact routing result
        for transfer admission and both route sub-batches.
        """
        if not self.hover_exact_h2d_overlap:
            raise RuntimeError("route-exact overlap was called while disabled")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "route-exact overlap requires eager execution; CUDA Graph "
                "capture cannot record dynamic route/cache transactions"
            )
        cache = self.expert_cache
        if cache is None:
            raise RuntimeError("route-split overlap requires RouteExactExpertTransferOverlap")
        if not isinstance(self.quant_method, UnquantizedFusedMoEMethod):
            raise RuntimeError(
                "route-split overlap currently supports unquantized FusedMoE only"
            )

        topk_weights, topk_ids = self.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=self.use_grouped_topk,
            top_k=self.top_k,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            indices_type=self.quant_method.topk_indices_dtype,
        )
        required_experts = (
            torch.unique(topk_ids).detach().cpu().tolist()
        )
        ticket = cache.begin_forward_exact_async(required_experts)
        try:
            output = self._apply_quant_method(
                hidden_states,
                router_logits,
                preselected_topk=(topk_weights, topk_ids),
                exact_overlap_ticket=ticket,
                exact_overlap_experts=required_experts,
                exact_route_split=True,
            )
            return self._maybe_reduce_output(output)
        except BaseException:
            cache.abort_forward_exact_async(ticket)
            raise

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor):
        if self.hover_exact_h2d_overlap:
            return self.forward_with_route_split_overlap(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        
        final_hidden_states = self._apply_quant_method(
            hidden_states, router_logits
        )

        
        # if hidden_states.shape[0] <= 10:
        #     print(self.tp_rank, final_hidden_states)

        
        final_hidden_states = self._maybe_reduce_output(final_hidden_states)

        
        # if hidden_states.shape[0] <= 10:
        #     print(self.tp_rank, final_hidden_states)
        #     exit(0)

        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(
            cls,
            ckpt_gate_proj_name: str,
            ckpt_down_proj_name: str,
            ckpt_up_proj_name: str,
            num_experts: int,
            num_redundant_experts: int = 0) -> list[tuple[str, str, int, str]]:

        num_physical_experts = num_experts + num_redundant_experts

        
        
        
        global_physical_to_logical_map = list(range(num_experts))
        global_physical_to_logical_map += [
            i % num_experts for i in range(num_redundant_experts)
        ]
        physical_to_logical_map = global_physical_to_logical_map

        return [
            # (param_name, weight_name, expert_id, shard_id)
            ("experts.w13_" if weight_name
             in [ckpt_gate_proj_name, ckpt_up_proj_name] else "experts.w2_",
             f"experts.{physical_to_logical_map[expert_id]}.{weight_name}.",
             expert_id, shard_id) for expert_id in range(num_physical_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def extra_repr(self) -> str:

        s = (
            f"global_num_experts={self.global_num_experts}, "
            f"local_num_experts={self.local_num_experts}, "
            f"top_k={self.top_k}, "
            f"intermediate_size_per_partition={self.intermediate_size_per_partition}, "  # noqa: E501
            f"tp_size={self.tp_size},\n"
            f"reduce_results={self.reduce_results}, "
            f"renormalize={self.renormalize}, "
            f"use_grouped_topk={self.use_grouped_topk}")

        s += f", scoring_func='{self.scoring_func}', activation='{self.activation}'"  # noqa: E501

        return s

# ============================================================================

# ============================================================================

def configure_moe_offload(
    moe_layer,
    w13_host: torch.Tensor,
    w2_host: torch.Tensor,
    strategy: str = "lru",
    capacity: int = -1,
    physical: bool = False,
    w13_bias_host: Optional[torch.Tensor] = None,
    w2_bias_host: Optional[torch.Tensor] = None,
):
    if strategy == "none":
        return
    num_experts = int(w13_host.shape[0])
    if capacity is None or capacity < 0:
        capacity = num_experts
    capacity = min(int(capacity), num_experts)
    if physical:
        if _PhysicalExpertCache is None:
            raise RuntimeError("PhysicalExpertCache could not be imported")
        if not isinstance(moe_layer.quant_method, UnquantizedFusedMoEMethod):
            raise RuntimeError(
                "physical-cpu-first supports unquantized FusedMoE only"
            )
        if getattr(moe_layer, "expert_map", None) is not None:
            raise RuntimeError(
                "physical-cpu-first does not support expert parallel maps"
            )
        if capacity <= 0:
            raise ValueError("physical expert cache capacity must be positive")
        if not getattr(moe_layer, "_physical_host_load_enabled", False):
            raise RuntimeError(
                "physical-cpu-first requires compact construction before "
                "checkpoint loading"
            )
        if hasattr(moe_layer, "w13_bias") or hasattr(moe_layer, "w2_bias"):
            raise RuntimeError(
                "physical-cpu-first does not yet support expert bias"
            )
        if (
            int(moe_layer.w13_weight.shape[0]) != capacity
            or int(moe_layer.w2_weight.shape[0]) != capacity
        ):
            raise RuntimeError(
                "CPU-first compact parameter shape does not match capacity"
            )
        if (
            w13_host is not moe_layer._physical_w13_host
            or w2_host is not moe_layer._physical_w2_host
        ):
            raise RuntimeError("CPU-first host master identity mismatch")

        full_gpu_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (w13_host, w2_host)
        )
        # Checkpoint loading populated only the CPU masters.  Seed compact GPU
        # slots now; a full [num_experts, ...] GPU tensor never exists.
        for slot in range(capacity):
            moe_layer.w13_weight.data[slot].copy_(
                w13_host[slot], non_blocking=False
            )
            moe_layer.w2_weight.data[slot].copy_(
                w2_host[slot], non_blocking=False
            )

        cache = _PhysicalExpertCache(
            layer_idx=getattr(moe_layer, "_moe_layer_idx", -1),
            num_experts=num_experts,
            capacity=capacity,
            w13_gpu=moe_layer.w13_weight.data,
            w2_gpu=moe_layer.w2_weight.data,
            w13_host=w13_host,
            w2_host=w2_host,
            strategy=strategy,
            w13_bias_gpu=(
                moe_layer.w13_bias.data
                if hasattr(moe_layer, "w13_bias") else None
            ),
            w2_bias_gpu=(
                moe_layer.w2_bias.data
                if hasattr(moe_layer, "w2_bias") else None
            ),
            w13_bias_host=w13_bias_host,
            w2_bias_host=w2_bias_host,
            initial_experts=range(capacity),
            full_gpu_bytes=full_gpu_bytes,
        )
        moe_layer.physical_expert_cache = True
        moe_layer.physical_expert_load_contract = "cpu_first_checkpoint"
    else:
        if _RouteExactExpertTransferOverlap is None:
            raise RuntimeError("RouteExactExpertTransferOverlap could not be imported")
        cache = _RouteExactExpertTransferOverlap(
            layer_idx=getattr(moe_layer, "_moe_layer_idx", -1),
            num_experts=num_experts,
            capacity=capacity,
            w13_gpu=moe_layer.w13_weight.data,
            w2_gpu=moe_layer.w2_weight.data,
            w13_host=w13_host,
            w2_host=w2_host,
            strategy=strategy,
        )
    moe_layer.offload_strategy = strategy
    moe_layer.gpu_expert_cache_slots = capacity
    moe_layer.expert_cache = cache
    moe_layer._w13_host = w13_host
    moe_layer._w2_host = w2_host
