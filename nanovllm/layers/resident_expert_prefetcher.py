"""Resident Expert Prefetcher from the HoVer architecture.

This module contains history-aware prediction, locality-aware selection,
and route-exact transfer overlap as one paper-level component.
"""

import threading
from collections import defaultdict, deque, OrderedDict
from typing import Dict, List, Optional, Iterable

import numpy as np


class HistoryAwareExpertDemandPredictor:
    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        theta: int = 5,
        gamma: float = 0.95,          
        beta: float = 0.01,           
        kh: int = 4,                  
        warmup_decode_tokens: int = 200,
        max_tracked_requests: int = 256,
    ):
        self.L = num_layers
        self.E = num_experts
        self.theta = max(1, theta)
        self.gamma = gamma
        self.beta = beta
        self.kh = max(1, kh)
        self.warmup_decode_tokens = warmup_decode_tokens
        self.max_tracked = max_tracked_requests
        
        self.N = np.zeros((self.L, self.E, self.E), dtype=np.float64)
        self.dN = np.zeros((self.L, self.E, self.E), dtype=np.float64)
        self.P = None   

        
        self.dH: Dict = defaultdict(self._zeros_e)
        self.pi: Dict = {}

        
        # summary[rid][ℓ] = deque[set]; last[rid][ℓ] = set
        self.summary: "OrderedDict[str, Dict[int, deque]]" = OrderedDict()
        self.last: Dict = {}
        self.req_modality: Dict = {}        # rid -> {m: weight}

        
        
        # Predicted layer-wise decode demand from Eqs. (2), (3), and (6).
        self.D_agg: Dict = {}

        self._decode_tokens_seen = 0
        self._lock = threading.Lock()
        self.refresh_count = 0
    def _zeros_e(self):
        return np.zeros(self.E, dtype=np.float64)

    @property
    def warm(self) -> bool:
        return self._decode_tokens_seen >= self.warmup_decode_tokens

    
    def record_decode(
        self,
        rid: str,
        layer_idx: int,
        expert_ids,
        round_t: int,
        modality_weights: Optional[Dict[int, float]] = None,
    ):
        """Record one per-layer decode observation.

        This compatibility API has no forward boundary, so its warmup counter
        advances once for each accepted call. ModelRunner execution uses
        :meth:`record_decode_bulk`, whose counter is forward-scoped.
        """
        if layer_idx < 0 or layer_idx >= self.L:
            return
        eids = _clean(expert_ids, self.E)
        if not eids:
            return
        rid = str(rid)
        cur = set(eids)
        with self._lock:
            
            if rid not in self.summary:
                self.summary[rid] = {}
                self.last[rid] = {}
                if modality_weights:
                    self.req_modality[rid] = dict(modality_weights)
                self._evict_if_needed()
            self.summary.move_to_end(rid)

            
            prev = self.last[rid].get(layer_idx)
            if prev:
                pe = np.fromiter(prev, dtype=np.int64)
                ce = np.fromiter(cur, dtype=np.int64)
                self.dN[layer_idx][np.ix_(pe, ce)] += 1.0
            self.last[rid][layer_idx] = cur

            # summary
            dq = self.summary[rid].get(layer_idx)
            if dq is None:
                dq = deque(maxlen=self.kh)
                self.summary[rid][layer_idx] = dq
            dq.append(cur)

            
            mw = modality_weights or self.req_modality.get(rid) or {0: 1.0}
            for m, w in mw.items():
                h = self.dH[(layer_idx, int(m))]
                for e in cur:
                    h[e] += float(w)

            self._decode_tokens_seen += 1

    def _record_decode_batch_unlocked(
        self,
        layer_idx: int,
        rids,
        rows_experts,
        round_t: int,
        modality_map=None,
        count_decode_tokens: bool = True,
    ):
        """Apply one layer while ``self._lock`` is already held."""
        if layer_idx < 0 or layer_idx >= self.L or not rids:
            return
        mmap = modality_map or {}
        for rid, experts in zip(rids, rows_experts):
            eids = _clean(experts, self.E)
            if not eids:
                continue
            rid = str(rid)
            cur = set(eids)
            if rid not in self.summary:
                self.summary[rid] = {}
                self.last[rid] = {}
                mw0 = mmap.get(rid)
                if mw0:
                    self.req_modality[rid] = dict(mw0)
                self._evict_if_needed()
            self.summary.move_to_end(rid)
            prev = self.last[rid].get(layer_idx)
            if prev:
                pe = np.fromiter(prev, dtype=np.int64)
                ce = np.fromiter(cur, dtype=np.int64)
                self.dN[layer_idx][np.ix_(pe, ce)] += 1.0
            self.last[rid][layer_idx] = cur
            dq = self.summary[rid].get(layer_idx)
            if dq is None:
                dq = deque(maxlen=self.kh)
                self.summary[rid][layer_idx] = dq
            dq.append(cur)
            mw = mmap.get(rid) or self.req_modality.get(rid) or {0: 1.0}
            for m, w in mw.items():
                h = self.dH[(layer_idx, int(m))]
                for e in cur:
                    h[e] += float(w)
            if count_decode_tokens:
                self._decode_tokens_seen += 1

    def record_decode_batch(self, layer_idx: int, rids, rows_experts, round_t: int,
                            modality_map=None):
        """Record all decode observations for one layer call.

        ``rids[i]`` / ``rows_experts[i]`` is one decode request at this layer.
        Since this API cannot see the other layers or a completed-forward
        boundary, the warmup counter remains per accepted row *per call*.
        Use ``record_decode_bulk`` for exact forward-scoped token counting.
        """
        if layer_idx < 0 or layer_idx >= self.L or not rids:
            return
        with self._lock:
            self._record_decode_batch_unlocked(
                layer_idx, rids, rows_experts, round_t, modality_map
            )

    def record_decode_bulk(self, records):
        """Record all MoE layers from one completed forward under one lock.

        Each item is ``(layer_idx, rids, rows_experts, round_t)`` or
        ``(layer_idx, rids, rows_experts, round_t, modality_map)``.  A bad
        layer remains fail-open and does not block later layers.  The returned
        boolean list lets the caller preserve the rule that TTL is not
        updated when the predictor update for that layer failed.

        This call is the formal forward boundary.  Transition, summary, and
        modality statistics are still applied once per request per MoE layer,
        but ``_decode_tokens_seen`` advances only once per distinct request in
        the completed forward, independent of the number of MoE layers.
        """
        records = list(records)
        if not records:
            return []
        applied = []
        forward_rids = set()
        with self._lock:
            for record in records:
                try:
                    if len(record) == 4:
                        layer_idx, rids, rows_experts, round_t = record
                        modality_map = None
                    else:
                        (layer_idx, rids, rows_experts, round_t,
                         modality_map) = record
                    # ``rids`` is the actual decode batch for this forward.
                    # Materialize it once so iterable inputs are not consumed
                    # by accounting before the per-layer state update.
                    rids = list(rids)
                    forward_rids.update(str(rid) for rid in rids)
                    self._record_decode_batch_unlocked(
                        layer_idx,
                        rids,
                        rows_experts,
                        round_t,
                        modality_map,
                        count_decode_tokens=False,
                    )
                    applied.append(True)
                except Exception:
                    applied.append(False)
            self._decode_tokens_seen += len(forward_rids)
        return applied

    def _evict_if_needed(self):
        while len(self.summary) > self.max_tracked:
            old_rid, _ = self.summary.popitem(last=False)
            self.last.pop(old_rid, None)
            self.req_modality.pop(old_rid, None)

    def drop_request(self, rid: str):
        rid = str(rid)
        with self._lock:
            self.summary.pop(rid, None)
            self.last.pop(rid, None)
            self.req_modality.pop(rid, None)

    
    def refresh(self, round_t: int, active_decode_rids: Optional[Iterable[str]] = None):
        with self._lock:
            
            self.N = self.gamma * self.N + self.dN
            self.dN[:] = 0.0
            row = self.N.sum(axis=2, keepdims=True) + self.beta * self.E
            self.P = (self.N + self.beta) / row   # (L,E,E)

            
            
            self.pi = {}
            for (li, m), h in self.dH.items():
                s = h.sum() + self.beta * self.E
                self.pi[(li, m)] = (h + self.beta) / s
            
            for k in self.dH:
                self.dH[k] *= self.gamma

            
            # Eq. (3): average request-level reuse profiles for each layer.
            # Eq. (6): propagate that layer-wise profile through the learned
            # first-order expert-transition matrix.
            rids = list(active_decode_rids) if active_decode_rids is not None else list(self.summary.keys())
            rids = [str(r) for r in rids if str(r) in self.summary]
            self.D_agg = {}
            for li in range(self.L):
                rho_agg = np.zeros(self.E, dtype=np.float64)
                for rid in rids:
                    dq = self.summary[rid].get(li)
                    if dq:
                        cnt = np.zeros(self.E, dtype=np.float64)
                        for s in dq:
                            for e in s:
                                cnt[e] += 1.0
                        # Eq. (2): request-level reuse profile.
                        rho = cnt / (cnt.sum() + 1e-9)
                        rho_agg += rho

                # Eq. (3): layer-wise reuse profile.
                rho_agg /= float(len(rids)) + 1e-9

                # Eq. (6): one-step Markov prediction.
                self.D_agg[li] = rho_agg @ self.P[li]
            self.refresh_count += 1

    
    def get_layer_demand(self, layer_idx: int) -> Optional[np.ndarray]:
        return self.D_agg.get(layer_idx)

    def modality_prior(self, layer_idx: int, modality_weights: Dict[int, float]) -> np.ndarray:
        out = np.zeros(self.E, dtype=np.float64)
        tw = sum(modality_weights.values()) or 1.0
        for m, w in modality_weights.items():
            pi = self.pi.get((int(layer_idx), int(m)))
            if pi is not None:
                out += (w / tw) * pi
        if out.sum() == 0:
            out = np.full(self.E, 1.0 / self.E)
        return out

    def stats(self) -> dict:
        return {
            "type": "history_aware_expert_demand_prediction",
            "num_layers": self.L,
            "num_experts": self.E,
            "theta": self.theta,
            "refresh_count": self.refresh_count,
            "tracked_requests": len(self.summary),
            "decode_tokens_seen": self._decode_tokens_seen,
            "decode_counter_bulk_unit": "unique_request_per_forward",
            "decode_counter_unit": "accepted_per_layer_observation",
            "warm": self.warm,
        }


def _clean(expert_ids, E) -> List[int]:
    # Compact HoVer tracking hands us ``list[int]`` produced by
    # ``Tensor.tolist()``.  Avoid constructing a NumPy array and sorting it for
    # every decode row at every MoE layer. The fallback retains the compatible
    # behavior for tensors, ndarrays, floats and third-party sequence types.
    if isinstance(expert_ids, (list, tuple)) and all(
            isinstance(x, (int, np.integer))
            and not isinstance(x, (bool, np.bool_))
            for x in expert_ids):
        return sorted({int(x) for x in expert_ids if 0 <= int(x) < E})
    if hasattr(expert_ids, "detach"):
        arr = expert_ids.detach().reshape(-1).cpu().numpy()
    else:
        arr = np.asarray(expert_ids).reshape(-1)
    if arr.size == 0:
        return []
    arr = arr[(arr >= 0) & (arr < E)]
    return [int(x) for x in np.unique(arr)]


import threading
from typing import Dict, List, Optional, Iterable

import numpy as np


class LocalityAwareResidentSelector:
    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        capacity: int,                 
        predictor,                     # HistoryAwareExpertDemandPredictor
        theta: int = 5,
        ttl_max: int = 5,
        pin_ratio: float = 0.5,        
        enabled: bool = True,
    ):
        self.L = num_layers
        self.E = num_experts
        self.capacity = max(1, capacity) if capacity > 0 else num_experts
        self.predictor = predictor
        self.theta = max(1, theta)
        self.ttl_max = ttl_max
        self.pin_ratio = pin_ratio
        self.enabled = enabled

        
        self.ttl = np.zeros((self.L, self.E), dtype=np.int32)
        self.active_window = np.zeros((self.L, self.E), dtype=bool)

        
        self.k = max(1, int(self.capacity * pin_ratio))

        self._lock = threading.Lock()
        self._round = 0
        self.update_count = 0
        self.stable_skip_count = 0

    
    def _clean_expert_ids(self, expert_ids):
        # Normal compact-tracking input is a Python list of integer expert
        # IDs.  Keep tensors/ndarrays on the historical NumPy path.
        if isinstance(expert_ids, (list, tuple)) and all(
                isinstance(x, (int, np.integer))
                and not isinstance(x, (bool, np.bool_))
                for x in expert_ids):
            return [int(x) for x in expert_ids if 0 <= int(x) < self.E]
        if hasattr(expert_ids, "detach"):
            arr = expert_ids.detach().reshape(-1).cpu().numpy()
        else:
            arr = np.asarray(expert_ids).reshape(-1)
        arr = arr[(arr >= 0) & (arr < self.E)]
        return arr

    def _record_decode_unlocked(self, layer_idx: int, expert_ids):
        if not self.enabled or layer_idx < 0 or layer_idx >= self.L:
            return
        arr = self._clean_expert_ids(expert_ids)
        if len(arr) == 0:
            return
        self.active_window[layer_idx, arr] = True

    def record_decode(self, layer_idx: int, expert_ids):
        if not self.enabled or layer_idx < 0 or layer_idx >= self.L:
            return
        with self._lock:
            self._record_decode_unlocked(layer_idx, expert_ids)

    def record_decode_bulk(self, records):
        """Record ``(layer_idx, expert_ids)`` for all layers under one lock.

        Invalid layer payloads remain isolated so one malformed layer does not
        prevent TTL accounting for later layers.
        """
        records = list(records)
        if not self.enabled or not records:
            return []
        applied = []
        with self._lock:
            for record in records:
                try:
                    layer_idx, expert_ids = record
                    self._record_decode_unlocked(layer_idx, expert_ids)
                    applied.append(True)
                except Exception:
                    applied.append(False)
        return applied

    
    def step(
        self,
        active_decode_rids: Optional[Iterable[str]] = None,
        skip_selection: bool = False,
    ) -> Optional[Dict[int, List[int]]]:
        """Advance one round and optionally suppress a stable-window reselection.

        ``skip_selection`` is used only after the caller has observed an
        unchanged pure-decode batch and no cache misses for a full ``theta``
        window.  TTL accounting still advances and consumes the accumulated
        activation window, while the O(L * E log E) demand ranking and cache
        resident-set writes are omitted.
        """
        if not self.enabled:
            return None
        with self._lock:
            self._round += 1
            if self._round % self.theta != 0:
                return None

            
            decayed = np.maximum(0, self.ttl - 1)
            self.ttl = np.minimum(self.ttl_max, decayed + self.active_window.astype(np.int32))
            self.active_window.fill(False)

            if skip_selection:
                self.stable_skip_count += 1
                return None

            
            result: Dict[int, List[int]] = {}
            for li in range(self.L):
                D = self.predictor.get_layer_demand(li) if self.predictor is not None else None
                if D is None:
                    D = np.zeros(self.E, dtype=np.float64)
                
                
                
                
                key = D * (self.ttl_max + 1) + self.ttl[li].astype(np.float64)
                order = np.argsort(-key)[: self.k]
                result[li] = order.tolist()
            self.update_count += 1
            return result

    def stats(self) -> dict:
        return {
            "type": "locality_aware_selection",
            "num_layers": self.L,
            "num_experts": self.E,
            "capacity": self.capacity,
            "k_per_layer": self.k,
            "theta": self.theta,
            "round": self._round,
            "update_count": self.update_count,
            "stable_skip_count": self.stable_skip_count,
            "ttl_mean": float(self.ttl.mean()),
        }


"""Expert cache with a strict *persistent metadata* capacity contract.

``w13_gpu`` and ``w2_gpu`` keep their full ``[num_experts, ...]`` shapes.  The
capacity in this class therefore limits which expert weights may be reused
without another host-to-device copy; it does not reduce allocated GPU tensor
memory.  A fused MoE invocation may temporarily register every expert needed
by the active batch, even when that set is larger than ``capacity``.  The
common forward hook must call :meth:`post_forward` in a ``finally`` block, at
which point reusable metadata is trimmed back to the configured capacity.
"""

from collections import OrderedDict
from typing import List

import torch


CAPACITY_CONTRACT_VERSION = "logical-persistent-v1"


class _ExactForwardTicket:
    """Opaque state for one route-exact asynchronous forward transaction."""

    __slots__ = (
        "_owner_token",
        "required",
        "initial_missing",
        "ready_experts",
        "blocked_experts",
        "scheduled",
        "events",
        "compute_event",
        "copy_start",
        "copy_end",
        "wait_start",
        "wait_end",
        "copy_bytes",
        "legacy_fallback",
        "fallback_reason",
        "state",
        "timing_harvested",
    )

    def __init__(self, owner_token, required):
        self._owner_token = owner_token
        self.required = tuple(required)
        self.initial_missing = ()
        self.ready_experts = ()
        self.blocked_experts = ()
        self.scheduled = ()
        self.events = ()
        self.compute_event = None
        self.copy_start = None
        self.copy_end = None
        self.wait_start = None
        self.wait_end = None
        self.copy_bytes = 0
        self.legacy_fallback = False
        self.fallback_reason = None
        self.state = "begun"
        self.timing_harvested = False


class RouteExactExpertTransferOverlap:
    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        capacity: int,
        w13_gpu: torch.Tensor,
        w2_gpu: torch.Tensor,
        w13_host: torch.Tensor,
        w2_host: torch.Tensor,
        strategy: str = "lru",
    ):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.capacity = (
            max(0, min(capacity, num_experts))
            if capacity >= 0 else num_experts
        )
        self.w13_gpu = w13_gpu
        self.w2_gpu = w2_gpu
        self.w13_host = w13_host
        self.w2_host = w2_host
        self.strategy = strategy

        # Values are intentionally None.  Membership and OrderedDict order are
        # the logical validity/reuse policy; expert IDs index fixed GPU slices.
        self.gpu_resident: "OrderedDict[int, None]" = OrderedDict()
        self.pinned: set[int] = set()

        self._working_frequency = {}
        self._frequency_updates = 0
        self._frequency_decay_interval = 64
        self._last_prefill_experts = None

        self.hit_count = 0
        self.miss_count = 0
        self.transfer_bytes = 0
        self.transfer_time_ms = 0.0
        self.eviction_count = 0
        self.calls = 0
        self._bytes_per_expert = (
            self.w13_host[0].numel() * self.w13_host[0].element_size()
            + self.w2_host[0].numel() * self.w2_host[0].element_size()
        )

        self._prefetch_stream = None
        self._prefetch_events = {}
        self._pending_transfer_timings = []
        # Latest compute-stream fence.  Trimming metadata does not cancel the
        # fused kernel that just consumed these fixed expert slices.  Any copy
        # stream reuse must wait until that read has completed.
        self._last_compute_event = None
        self.resident_update_count = 0
        self.resident_update_skip_count = 0
        self.resident_async_copy_count = 0
        # Synchronous demand copies must wait only for their own current-
        # stream completion event.  A device-wide barrier would also drain
        # unrelated resident-copy streams from every other MoE layer and
        # destroy the intended overlap without adding a data dependency.
        self.sync_copy_event_wait_count = 0
        self.sync_copy_stream_wait_count = 0

        # Route-exact current-layer copy transactions.  This path is enabled explicitly;
        # the existing begin_forward/ensure_resident behavior is unchanged.
        self._exact_owner_token = object()
        self._exact_active_ticket = None
        self._exact_timing_tickets = []
        self.exact_begin_calls = 0
        self.exact_wait_count = 0
        self.exact_post_count = 0
        self.exact_abort_count = 0
        self.exact_begin_exception_count = 0
        self.exact_fallback_count = 0
        self._exact_fallback_reasons = {}
        self.exact_wait_timing_fallback_count = 0
        self.exact_required_expert_count = 0
        self.exact_hit_count = 0
        self.exact_miss_count = 0
        self.exact_async_copy_count = 0
        self.exact_transfer_bytes = 0
        self.exact_raw_dma_ms = 0.0
        self.exact_exposed_wait_ms = 0.0
        self.exact_estimated_hidden_dma_ms = 0.0
        self.exact_aborted_raw_dma_ms = 0.0
        self.exact_timed_ticket_count = 0
        self.exact_route_split_attempt_count = 0
        self.exact_route_split_used_count = 0
        self.exact_route_split_ready_routes = 0
        self.exact_route_split_blocked_routes = 0
        self._exact_route_split_fallback_reasons = {}

        # Capacity-contract telemetry.  ``final`` means after a fail-closed
        # boundary (forward, prefetch/resident update, or error cleanup).
        self._active_forward_depth = 0
        self._max_active_forward_depth = 0
        self._max_transient_resident = 0
        self._max_final_resident = 0
        self._last_final_resident = 0
        self._transient_over_capacity_count = 0
        self._persistent_trim_count = 0
        self._prefetch_drop_count = 0
        self._forward_finalize_count = 0
        self._exception_finalize_count = 0
        self.strict_logical_wave_enabled = False
        self.strict_wave_forward_count = 0
        self.strict_multi_wave_forward_count = 0
        self.strict_total_waves = 0
        self.strict_max_waves_per_forward = 0

        if self.capacity >= self.num_experts and self.strategy != "naive":
            self._warmup_all()
            self._finalize_persistent("warmup")

    @staticmethod
    def _stable_unique(values):
        return list(dict.fromkeys(int(value) for value in values))

    def _valid_ids(self, values):
        return [
            eid for eid in self._stable_unique(values)
            if 0 <= eid < self.num_experts
        ]

    def _required_ids(self, values):
        required = self._stable_unique(values)
        invalid = [eid for eid in required if not 0 <= eid < self.num_experts]
        if invalid:
            raise ValueError(f"invalid required expert IDs: {invalid}")
        return required

    def _observe_transient(self):
        size = len(self.gpu_resident)
        self._max_transient_resident = max(self._max_transient_resident, size)
        if size > self.capacity:
            self._transient_over_capacity_count += 1

    def _warmup_all(self):
        try:
            for eid in range(self.num_experts):
                self.w13_gpu[eid].copy_(
                    self.w13_host[eid], non_blocking=False
                )
                self.w2_gpu[eid].copy_(
                    self.w2_host[eid], non_blocking=False
                )
                self.gpu_resident[eid] = None
            torch.cuda.synchronize()
            self._observe_transient()
        except BaseException:
            self.gpu_resident.clear()
            self.pinned.clear()
            self._exception_finalize_count += 1
            self._finalize_persistent("warmup_exception")
            raise

    def _ensure_prefetch_stream(self):
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream()
        return self._prefetch_stream

    def _wait_for_last_compute(self, stream):
        if self._last_compute_event is not None:
            stream.wait_event(self._last_compute_event)

    def _record_compute_fence(self):
        try:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            self._last_compute_event = event
        except Exception:
            # Losing the dependency would permit an async H2D write to race a
            # still-running fused kernel.  The rare fallback is fail-closed.
            torch.cuda.synchronize()
            self._last_compute_event = None

    @staticmethod
    def _event_complete(event) -> bool:
        try:
            return bool(event.query())
        except Exception:
            return False

    def _reap_completed_prefetch_events(self):
        for eid, event in list(self._prefetch_events.items()):
            if (
                self._event_complete(event)
                and self._prefetch_events.get(eid) is event
            ):
                self._prefetch_events.pop(eid, None)

    def _harvest_transfer_timings(self):
        pending = []
        for start, end in self._pending_transfer_timings:
            if not self._event_complete(end):
                pending.append((start, end))
                continue
            try:
                self.transfer_time_ms += start.elapsed_time(end)
            except Exception:
                pass
        self._pending_transfer_timings = pending

    def _evict_one(self, protected=()) -> bool:
        protected = set(protected)
        candidates = [
            eid for eid in self.gpu_resident
            if eid not in self.pinned and eid not in protected
        ]
        if not candidates:
            return False
        if self.strategy == "dlfu":
            eid = min(
                candidates,
                key=lambda value: self._working_frequency.get(value, 0.0),
            )
        else:
            eid = candidates[0]
        self.gpu_resident.pop(eid, None)
        self._working_frequency.pop(eid, None)
        # Keep an unfinished event: a later reload must wait for the older DMA.
        self.eviction_count += 1
        return True

    def _finalize_persistent(self, reason: str):
        """Restore the idle/reusable metadata invariant without a CUDA sync."""
        # Failed asynchronous loads can leave a requested ID pinned but not
        # valid.  Such an ID is not a persistent resident and must not block
        # eviction.
        self.pinned.intersection_update(self.gpu_resident.keys())

        if len(self.pinned) > self.capacity:
            # This only repairs legacy/corrupt state.  Keep the most recently
            # used pinned entries because OrderedDict is oldest -> newest.
            ordered = [eid for eid in self.gpu_resident if eid in self.pinned]
            self.pinned = set(ordered[-self.capacity:]) if self.capacity else set()

        before = len(self.gpu_resident)
        if self.capacity <= 0:
            self.gpu_resident.clear()
            self._working_frequency.clear()
            self.pinned.clear()
        else:
            while len(self.gpu_resident) > self.capacity:
                if not self._evict_one():
                    # ``pinned <= capacity`` means this is unreachable unless
                    # metadata was concurrently corrupted.  Fail closed by
                    # unpinning the oldest entry and retrying.
                    oldest = next(iter(self.gpu_resident))
                    self.pinned.discard(oldest)
                    if not self._evict_one():
                        raise RuntimeError(
                            "RouteExactExpertTransferOverlap could not restore persistent capacity"
                        )
        if len(self.gpu_resident) < before:
            self._persistent_trim_count += before - len(self.gpu_resident)

        final = len(self.gpu_resident)
        if final > self.capacity or len(self.pinned) > self.capacity:
            raise RuntimeError("RouteExactExpertTransferOverlap persistent capacity invariant failed")
        self._last_final_resident = final
        self._max_final_resident = max(self._max_final_resident, final)

    def _enqueue_async_copies(self, expert_ids: List[int], source: str) -> List[int]:
        """Best-effort prefetch that never leaves persistent over-capacity state."""
        if self.capacity <= 0:
            return []
        track_timing = source == "resident"
        if track_timing:
            self._reap_completed_prefetch_events()
            self._harvest_transfer_timings()

        needed = self._required_ids(expert_ids)
        existing_requested = {eid for eid in needed if eid in self.gpu_resident}
        missing = [eid for eid in needed if eid not in self.gpu_resident]
        if not missing:
            self._finalize_persistent(f"{source}_noop")
            return []

        # Predictions are optional.  Preserve pinned and higher-priority
        # requested experts, admit in caller order, and drop the tail rather
        # than granting a persistent cache larger than capacity.
        protected = set(self.pinned) | existing_requested
        admitted = []
        for eid in missing:
            while len(self.gpu_resident) + len(admitted) >= self.capacity:
                if not self._evict_one(protected | set(admitted)):
                    break
            if len(self.gpu_resident) + len(admitted) >= self.capacity:
                self._prefetch_drop_count += 1
                continue
            admitted.append(eid)

        if not admitted:
            self._finalize_persistent(f"{source}_full")
            return []

        stream = self._ensure_prefetch_stream()
        scheduled = []
        timing_start = None
        if track_timing:
            try:
                timing_start = torch.cuda.Event(enable_timing=True)
            except Exception:
                pass

        try:
            with torch.cuda.stream(stream):
                self._wait_for_last_compute(stream)
                if timing_start is not None:
                    try:
                        timing_start.record(stream)
                    except Exception:
                        timing_start = None
                try:
                    for eid in admitted:
                        if eid in self.gpu_resident:
                            continue
                        completion = torch.cuda.Event()
                        try:
                            self.w13_gpu[eid].copy_(
                                self.w13_host[eid], non_blocking=True
                            )
                            self.w2_gpu[eid].copy_(
                                self.w2_host[eid], non_blocking=True
                            )
                            completion.record(stream)
                        except BaseException:
                            try:
                                completion.record(stream)
                                self._prefetch_events[eid] = completion
                            except Exception:
                                pass
                            raise
                        self._prefetch_events[eid] = completion
                        self.gpu_resident[eid] = None
                        self.transfer_bytes += self._bytes_per_expert
                        scheduled.append(eid)
                        if track_timing:
                            self.resident_async_copy_count += 1
                finally:
                    if timing_start is not None and scheduled:
                        try:
                            timing_end = torch.cuda.Event(enable_timing=True)
                            timing_end.record(stream)
                            self._pending_transfer_timings.append(
                                (timing_start, timing_end)
                            )
                        except Exception:
                            pass
        except BaseException:
            self._exception_finalize_count += 1
            self._finalize_persistent(f"{source}_exception")
            raise

        self._observe_transient()
        self._finalize_persistent(source)
        return scheduled

    def record_phase_stats(self, decode_experts, prefill_experts):
        self._last_prefill_experts = set(
            int(value) for value in (prefill_experts or [])
        )

    def _validate_exact_ticket(self, ticket):
        if not isinstance(ticket, _ExactForwardTicket):
            raise TypeError("invalid route-exact forward ticket")
        if ticket._owner_token is not self._exact_owner_token:
            raise ValueError("route-exact ticket belongs to another cache")

    def _record_exact_fallback(self, reason: str):
        self.exact_fallback_count += 1
        self._exact_fallback_reasons[reason] = (
            self._exact_fallback_reasons.get(reason, 0) + 1
        )

    def _exact_host_buffers_are_pinned(self) -> bool:
        for tensor in (self.w13_host, self.w2_host):
            is_pinned = getattr(tensor, "is_pinned", None)
            if not callable(is_pinned):
                return False
            try:
                if not bool(is_pinned()):
                    return False
            except Exception:
                return False
        return True

    def _begin_forward_exact_fallback(
        self, required, initial_missing, reason: str
    ):
        """Open a synchronous transaction when true asynchronous DMA is unsafe.

        Copy/wait behavior is the legacy path, but touch is deliberately
        deferred to wait_forward_exact_async.  The exact integration records
        phase attribution after begin (while shared experts run); touching in
        begin would consume the previous phase marker and leave the current
        one stale.
        """
        self._record_exact_fallback(reason)
        ticket = _ExactForwardTicket(self._exact_owner_token, required)
        ticket.initial_missing = tuple(initial_missing)
        ticket.ready_experts = tuple(required)
        ticket.blocked_experts = ()
        ticket.legacy_fallback = True
        ticket.fallback_reason = reason
        self._active_forward_depth += 1
        self._max_active_forward_depth = max(
            self._max_active_forward_depth, self._active_forward_depth
        )
        try:
            self._wait_for_last_compute(torch.cuda.current_stream())
            self.ensure_resident(required)
        except BaseException:
            self._active_forward_depth -= 1
            self._exception_finalize_count += 1
            self._finalize_persistent("exact_fallback_exception")
            raise
        self._exact_active_ticket = ticket
        return ticket

    def _harvest_exact_timings(self):
        """Harvest completed CUDA timings without synchronizing the device."""
        pending = []
        for ticket in self._exact_timing_tickets:
            if ticket.timing_harvested:
                continue
            if ticket.state not in ("posted", "aborted"):
                pending.append(ticket)
                continue
            if (
                ticket.copy_end is not None
                and not self._event_complete(ticket.copy_end)
            ):
                pending.append(ticket)
                continue
            if (
                ticket.wait_end is not None
                and not self._event_complete(ticket.wait_end)
            ):
                pending.append(ticket)
                continue

            raw_ms = 0.0
            exposed_ms = 0.0
            try:
                if ticket.copy_start is not None and ticket.copy_end is not None:
                    raw_ms = max(
                        0.0, ticket.copy_start.elapsed_time(ticket.copy_end)
                    )
            except Exception:
                raw_ms = 0.0
            try:
                if ticket.wait_start is not None and ticket.wait_end is not None:
                    exposed_ms = max(
                        0.0, ticket.wait_start.elapsed_time(ticket.wait_end)
                    )
            except Exception:
                exposed_ms = 0.0

            self.transfer_time_ms += raw_ms
            self.exact_raw_dma_ms += raw_ms
            self.exact_exposed_wait_ms += exposed_ms
            if ticket.state == "posted":
                # This is deliberately labelled an estimate.  A ticket can
                # also inherit a pending predictive-prefetch event, so the
                # exposed wait is a conservative upper bound for exact DMA.
                self.exact_estimated_hidden_dma_ms += max(
                    0.0, raw_ms - exposed_ms
                )
            else:
                self.exact_aborted_raw_dma_ms += raw_ms
            self.exact_timed_ticket_count += 1
            ticket.timing_harvested = True
        self._exact_timing_tickets = pending

    def begin_forward_exact_async(self, required):
        """Open a route-exact H2D transaction on the cache copy stream.

        Unlike best-effort prefetch, every missing expert required by this
        forward is scheduled, even when the active set exceeds persistent
        ``capacity``.  The returned ticket must be waited and posted, or
        aborted on every exception path.
        """
        self.exact_begin_calls += 1
        if self._exact_active_ticket is not None or self._active_forward_depth:
            raise RuntimeError("an RouteExactExpertTransferOverlap forward transaction is active")

        required = self._required_ids(required)
        hits = [eid for eid in required if eid in self.gpu_resident]
        missing = [eid for eid in required if eid not in self.gpu_resident]
        self.exact_required_expert_count += len(required)
        self.exact_hit_count += len(hits)
        self.exact_miss_count += len(missing)

        if missing and not self._exact_host_buffers_are_pinned():
            return self._begin_forward_exact_fallback(
                required, missing, "host_not_pinned"
            )

        stream = None
        completion_events = {}
        copy_start = copy_end = None
        if missing:
            try:
                stream = self._ensure_prefetch_stream()
                completion_events = {
                    eid: torch.cuda.Event() for eid in missing
                }
                copy_start = torch.cuda.Event(enable_timing=True)
                copy_end = torch.cuda.Event(enable_timing=True)
            except Exception:
                return self._begin_forward_exact_fallback(
                    required, missing, "cuda_preflight_unavailable"
                )

        ticket = _ExactForwardTicket(self._exact_owner_token, required)
        ticket.initial_missing = tuple(missing)
        ticket.compute_event = self._last_compute_event
        inherited_events = {
            eid: self._prefetch_events[eid]
            for eid in required
            if eid in self._prefetch_events
        }

        self._active_forward_depth += 1
        self._max_active_forward_depth = max(
            self._max_active_forward_depth, self._active_forward_depth
        )
        self._exact_active_ticket = ticket
        scheduled = []
        copy_started = False
        try:
            if self.capacity > 0:
                protected = set(required) | self.pinned
                while len(self.gpu_resident) + len(missing) > self.capacity:
                    if not self._evict_one(protected):
                        break
            else:
                self.gpu_resident.clear()
                self.pinned.clear()

            if missing:
                with torch.cuda.stream(stream):
                    self._wait_for_last_compute(stream)
                    copy_start.record(stream)
                    copy_started = True
                    for eid in missing:
                        completion = completion_events[eid]
                        self.w13_gpu[eid].copy_(
                            self.w13_host[eid], non_blocking=True
                        )
                        self.w2_gpu[eid].copy_(
                            self.w2_host[eid], non_blocking=True
                        )
                        completion.record(stream)
                        self._prefetch_events[eid] = completion
                        if self.capacity > 0:
                            self.gpu_resident[eid] = None
                        self.transfer_bytes += self._bytes_per_expert
                        self.exact_transfer_bytes += self._bytes_per_expert
                        self.exact_async_copy_count += 1
                        scheduled.append(eid)
                    copy_end.record(stream)

            inherited_events.update(
                (eid, completion_events[eid]) for eid in scheduled
            )
            ticket.scheduled = tuple(scheduled)
            ticket.events = tuple(inherited_events.items())
            # Metadata residency is not enough for route-ready work: an expert
            # with a pending prefetch event is still unsafe to read.  Every
            # ticket dependency, including newly scheduled true misses, is
            # blocked; only the remainder may execute during DMA.
            blocked = set(inherited_events)
            ticket.ready_experts = tuple(
                eid for eid in required if eid not in blocked
            )
            ticket.blocked_experts = tuple(
                eid for eid in required if eid in blocked
            )
            ticket.copy_start = copy_start if scheduled else None
            ticket.copy_end = copy_end if scheduled else None
            ticket.copy_bytes = self._bytes_per_expert * len(scheduled)

            self.calls += 1
            self.hit_count += len(hits)
            self.miss_count += len(missing)
            self._observe_transient()
            self._exact_timing_tickets.append(ticket)
            return ticket
        except BaseException:
            # If event recording or a copy launch failed, drain this cache's
            # copy stream only.  No untracked write may survive into a later
            # routed kernel.  The normal path never synchronizes here.
            if copy_started and stream is not None:
                try:
                    stream.synchronize()
                except Exception:
                    torch.cuda.synchronize()
            ticket.state = "aborted"
            self._exact_active_ticket = None
            if self._active_forward_depth > 0:
                self._active_forward_depth -= 1
            self.exact_begin_exception_count += 1
            self._exception_finalize_count += 1
            self._finalize_persistent("exact_begin_exception")
            raise

    def partition_forward_exact_async(self, ticket):
        """Return the admission-time ready/blocked expert partition."""
        self._validate_exact_ticket(ticket)
        if self._exact_active_ticket is not ticket:
            raise RuntimeError("route-exact ticket is not active")
        if ticket.state != "begun":
            raise RuntimeError(
                f"route-exact ticket cannot be partitioned from state={ticket.state}"
            )
        return ticket.ready_experts, ticket.blocked_experts

    def record_exact_route_split(
        self,
        *,
        used: bool,
        ready_routes: int,
        blocked_routes: int,
        fallback_reason=None,
    ):
        """Record one gate-time adaptive route-split decision."""
        self.exact_route_split_attempt_count += 1
        self.exact_route_split_ready_routes += max(0, int(ready_routes))
        self.exact_route_split_blocked_routes += max(0, int(blocked_routes))
        if used:
            self.exact_route_split_used_count += 1
            return
        reason = str(fallback_reason or "unspecified")
        self._exact_route_split_fallback_reasons[reason] = (
            self._exact_route_split_fallback_reasons.get(reason, 0) + 1
        )

    def wait_forward_exact_async(self, ticket):
        """Fence only dependencies captured by ``ticket`` on the compute stream."""
        self._validate_exact_ticket(ticket)
        if self._exact_active_ticket is not ticket:
            raise RuntimeError("route-exact ticket is not active")
        if ticket.state != "begun":
            raise RuntimeError(
                f"route-exact ticket cannot wait from state={ticket.state}"
            )

        if ticket.legacy_fallback:
            # The fallback begin copied and waited synchronously, but touch is
            # kept here so phase/LRU/DLFU ordering matches the async path.
            self.touch(ticket.required)
            ticket.state = "waited"
            self.exact_wait_count += 1
            return ticket

        current = torch.cuda.current_stream()
        if ticket.compute_event is not None:
            current.wait_event(ticket.compute_event)

        unique_events = []
        seen_events = set()
        for _, event in ticket.events:
            marker = id(event)
            if marker not in seen_events:
                unique_events.append(event)
                seen_events.add(marker)

        wait_start = wait_end = None
        if unique_events:
            try:
                wait_start = torch.cuda.Event(enable_timing=True)
                wait_end = torch.cuda.Event(enable_timing=True)
                wait_start.record(current)
            except Exception:
                wait_start = wait_end = None
                self.exact_wait_timing_fallback_count += 1

        for event in unique_events:
            current.wait_event(event)

        if wait_start is not None and wait_end is not None:
            try:
                wait_end.record(current)
                ticket.wait_start = wait_start
                ticket.wait_end = wait_end
            except Exception:
                ticket.wait_start = None
                ticket.wait_end = None
                self.exact_wait_timing_fallback_count += 1

        # Remove only the same event instances consumed by this ticket.  An
        # unrelated or newer prefetch for another ID is never drained.
        for eid, event in ticket.events:
            if self._prefetch_events.get(eid) is event:
                self._prefetch_events.pop(eid, None)

        self.touch(ticket.required)
        ticket.state = "waited"
        self.exact_wait_count += 1
        return ticket

    def _finish_exact_ticket_state(self, ticket, final_state: str):
        """Fail-closed bookkeeping after post_forward, including rare errors."""
        if self._active_forward_depth > 0:
            self._active_forward_depth -= 1
            if self._active_forward_depth == 0:
                self._finalize_persistent(f"exact_{final_state}_recovery")
        self._exact_active_ticket = None
        ticket.state = final_state
        self._harvest_exact_timings()

    def post_forward_exact_async(self, ticket):
        """Close a waited ticket after the routed fused kernel was launched."""
        self._validate_exact_ticket(ticket)
        if self._exact_active_ticket is not ticket:
            raise RuntimeError("route-exact ticket is not active")
        if ticket.state != "waited":
            raise RuntimeError(
                f"route-exact ticket cannot post from state={ticket.state}"
            )
        try:
            self.post_forward(ticket.required)
        finally:
            self.exact_post_count += 1
            self._finish_exact_ticket_state(ticket, "posted")
        return ticket

    def abort_forward_exact_async(self, ticket) -> bool:
        """Idempotently clean up a begun/waited ticket after an exception."""
        self._validate_exact_ticket(ticket)
        if ticket.state in ("posted", "aborted"):
            return False
        if self._exact_active_ticket is not ticket:
            raise RuntimeError("route-exact ticket is not active")
        if ticket.state not in ("begun", "waited"):
            raise RuntimeError(
                f"route-exact ticket cannot abort from state={ticket.state}"
            )
        try:
            # Even on abort, a shared-branch kernel may already have been
            # queued on the current stream.  Reuse the normal compute fence so
            # future copy-stream writes cannot overtake it.
            self.post_forward(ticket.required)
        finally:
            self.exact_abort_count += 1
            self._finish_exact_ticket_state(ticket, "aborted")
        return True

    def begin_forward(self, expert_ids: List[int]) -> float:
        """Open a transient-capacity transaction for one fused MoE call."""
        self._active_forward_depth += 1
        self._max_active_forward_depth = max(
            self._max_active_forward_depth, self._active_forward_depth
        )
        try:
            self._wait_for_last_compute(torch.cuda.current_stream())
            transfer_ms = self.ensure_resident(expert_ids)
            self.touch(expert_ids)
            return transfer_ms
        except BaseException:
            self._active_forward_depth -= 1
            self._exception_finalize_count += 1
            self._finalize_persistent("ensure_exception")
            raise

    def ensure_resident(self, expert_ids: List[int]) -> float:
        """Load all required experts; over-capacity is transient while active."""
        self.calls += 1
        needed = self._valid_ids(expert_ids)
        if self._prefetch_events:
            self.wait_prefetch(needed)

        missing = [eid for eid in needed if eid not in self.gpu_resident]
        self.hit_count += len(needed) - len(missing)
        self.miss_count += len(missing)
        if not missing:
            if self._active_forward_depth == 0:
                self._finalize_persistent("standalone_ensure_hit")
            return 0.0

        if self.capacity > 0:
            protected = set(needed) | self.pinned
            while len(self.gpu_resident) + len(missing) > self.capacity:
                if not self._evict_one(protected):
                    break
        else:
            self.gpu_resident.clear()
            self.pinned.clear()

        loaded = []
        timing_start = timing_end = None
        try:
            try:
                timing_start = torch.cuda.Event(enable_timing=True)
                timing_end = torch.cuda.Event(enable_timing=True)
                timing_start.record()
            except Exception:
                timing_start = timing_end = None

            for eid in missing:
                self.w13_gpu[eid].copy_(
                    self.w13_host[eid], non_blocking=False
                )
                self.w2_gpu[eid].copy_(
                    self.w2_host[eid], non_blocking=False
                )
                loaded.append(eid)
                if self.capacity > 0:
                    self.gpu_resident[eid] = None

            if timing_end is not None:
                timing_end.record()
                # Wait for this demand-copy batch only.  Do not drain the
                # per-layer resident/prefetch streams on the whole device.
                timing_end.synchronize()
                self.sync_copy_event_wait_count += 1
                elapsed = (
                    timing_start.elapsed_time(timing_end)
                    if timing_start is not None else 0.0
                )
            else:
                # Event creation is best-effort telemetry, but correctness
                # still requires the current stream's two H2D writes to be
                # complete before the routed kernel reads their fixed slices.
                torch.cuda.current_stream().synchronize()
                self.sync_copy_stream_wait_count += 1
                elapsed = 0.0
        except BaseException:
            # Do not claim validity if either tensor copy or its completion
            # barrier failed.  A future attempt must reload both slices.
            for eid in loaded:
                self.gpu_resident.pop(eid, None)
            self._exception_finalize_count += 1
            self._finalize_persistent("sync_copy_exception")
            raise

        self.transfer_time_ms += elapsed
        self.transfer_bytes += self._bytes_per_expert * len(missing)
        self._observe_transient()
        if self._active_forward_depth == 0:
            self._finalize_persistent("standalone_ensure")
        return elapsed

    def touch(self, expert_ids: List[int]):
        if self.capacity == 0:
            return
        touched_order = self._required_ids(expert_ids)
        touched = set(touched_order)
        frequency_touched = (
            touched & self._last_prefill_experts
            if self._last_prefill_experts is not None else set()
        )
        self._last_prefill_experts = None
        if self.strategy == "dlfu":
            self._frequency_updates += 1
            if self._frequency_updates >= self._frequency_decay_interval:
                for eid in list(self._working_frequency):
                    self._working_frequency[eid] *= 0.5
                    if self._working_frequency[eid] < 1e-3:
                        self._working_frequency.pop(eid, None)
                self._frequency_updates = 0
        for eid in touched_order:
            if eid in self.gpu_resident:
                self.gpu_resident.move_to_end(eid)
                if (
                    self.strategy == "dlfu"
                    and eid not in self.pinned
                    and eid in frequency_touched
                ):
                    self._working_frequency[eid] = (
                        self._working_frequency.get(eid, 0.0) + 1.0
                    )

    def post_forward(self, expert_ids: List[int]):
        """Close a fused call, including kernel-exception cleanup."""
        fence_error = None
        try:
            self._record_compute_fence()
        except BaseException as exc:
            fence_error = exc
        try:
            if self.strategy == "naive":
                removed = len(self.gpu_resident)
                self.gpu_resident.clear()
                self.pinned.clear()
                self.eviction_count += removed
            if self._active_forward_depth > 0:
                self._active_forward_depth -= 1
            if self._active_forward_depth == 0:
                self._finalize_persistent("post_forward")
                self._forward_finalize_count += 1
        finally:
            if fence_error is not None:
                raise fence_error

    def prefetch_async(self, expert_ids: List[int]):
        return self._enqueue_async_copies(expert_ids, source="prefetch")

    def wait_prefetch(self, expert_ids: List[int]):
        for eid in set(self._valid_ids(expert_ids)):
            event = self._prefetch_events.get(eid)
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
                self._prefetch_events.pop(eid, None)

    def set_resident(self, expert_ids: List[int]):
        if self.capacity <= 0:
            self.pinned.clear()
            self._finalize_persistent("resident_disabled")
            return
        ordered_target = self._valid_ids(expert_ids)[: self.capacity]
        target = set(ordered_target)
        if target == self.pinned and all(
            eid in self.gpu_resident for eid in target
        ):
            self.resident_update_skip_count += 1
            self._reap_completed_prefetch_events()
            self._harvest_transfer_timings()
            self._finalize_persistent("resident_noop")
            return

        old_pinned = set(self.pinned)
        self.pinned = target
        if target != old_pinned:
            self.resident_update_count += 1
            if self.strategy == "dlfu":
                for eid in old_pinned | target:
                    self._working_frequency.pop(eid, None)
        try:
            missing = [eid for eid in ordered_target if eid not in self.gpu_resident]
            if missing:
                self._enqueue_async_copies(missing, source="resident")
        except BaseException:
            # The enqueue helper already trimmed; retain only valid pins.
            self.pinned.intersection_update(self.gpu_resident.keys())
            self._finalize_persistent("resident_exception")
            raise
        for eid in ordered_target:
            if eid in self.gpu_resident:
                self.gpu_resident.move_to_end(eid)
        self._finalize_persistent("resident")

    def stats(self) -> dict:
        self._reap_completed_prefetch_events()
        self._harvest_transfer_timings()
        self._harvest_exact_timings()
        total = self.hit_count + self.miss_count
        return {
            "layer_idx": self.layer_idx,
            "capacity": self.capacity,
            "num_experts": self.num_experts,
            "strategy": self.strategy,
            "calls": self.calls,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / total if total else 0.0,
            "eviction_count": self.eviction_count,
            "transfer_bytes": self.transfer_bytes,
            "transfer_time_ms": self.transfer_time_ms,
            "pending_prefetch_events": len(self._prefetch_events),
            "pending_transfer_timings": len(self._pending_transfer_timings),
            "compute_fence_pending": (
                self._last_compute_event is not None
                and not self._event_complete(self._last_compute_event)
            ),
            "resident_update_count": self.resident_update_count,
            "resident_update_skip_count": self.resident_update_skip_count,
            "resident_async_copy_count": self.resident_async_copy_count,
            "sync_copy_event_wait_count": self.sync_copy_event_wait_count,
            "sync_copy_stream_wait_count": self.sync_copy_stream_wait_count,
            "exact_begin_calls": self.exact_begin_calls,
            "exact_wait_count": self.exact_wait_count,
            "exact_post_count": self.exact_post_count,
            "exact_abort_count": self.exact_abort_count,
            "exact_begin_exception_count": self.exact_begin_exception_count,
            "exact_fallback_count": self.exact_fallback_count,
            "exact_fallback_reasons": dict(self._exact_fallback_reasons),
            "exact_wait_timing_fallback_count": (
                self.exact_wait_timing_fallback_count
            ),
            "exact_required_expert_count": self.exact_required_expert_count,
            "exact_hit_count": self.exact_hit_count,
            "exact_miss_count": self.exact_miss_count,
            "exact_async_copy_count": self.exact_async_copy_count,
            "exact_transfer_bytes": self.exact_transfer_bytes,
            "exact_raw_dma_ms": self.exact_raw_dma_ms,
            "exact_exposed_wait_ms": self.exact_exposed_wait_ms,
            "exact_estimated_hidden_dma_ms": (
                self.exact_estimated_hidden_dma_ms
            ),
            "exact_aborted_raw_dma_ms": self.exact_aborted_raw_dma_ms,
            "exact_timed_ticket_count": self.exact_timed_ticket_count,
            "exact_route_split_attempt_count": self.exact_route_split_attempt_count,
            "exact_route_split_used_count": self.exact_route_split_used_count,
            "exact_route_split_ready_routes": self.exact_route_split_ready_routes,
            "exact_route_split_blocked_routes": self.exact_route_split_blocked_routes,
            "exact_route_split_fallback_reasons": dict(
                self._exact_route_split_fallback_reasons
            ),
            "exact_timing_pending": len(self._exact_timing_tickets),
            "exact_active": self._exact_active_ticket is not None,
            "currently_resident": len(self.gpu_resident),
            "working_frequency_mean": (
                sum(self._working_frequency.values()) / len(self._working_frequency)
                if self._working_frequency else 0.0
            ),
            "working_frequency_max": max(
                self._working_frequency.values(), default=0.0
            ),
            "pinned_count": len(self.pinned),
            # Strict-capacity contract and fail-closed telemetry.
            "capacity_contract_version": (
                "logical-strict-wave"
                if self.strict_logical_wave_enabled
                else CAPACITY_CONTRACT_VERSION
            ),
            "capacity_semantics": (
                "logical_strict_per_wave_residency"
                if self.strict_logical_wave_enabled
                else "logical_persistent_reuse_metadata"
            ),
            "strict_wave_forward_count": self.strict_wave_forward_count,
            "strict_multi_wave_forward_count": (
                self.strict_multi_wave_forward_count
            ),
            "strict_total_waves": self.strict_total_waves,
            "strict_max_waves_per_forward": (
                self.strict_max_waves_per_forward
            ),
            "active_forward_depth": self._active_forward_depth,
            "max_active_forward_depth": self._max_active_forward_depth,
            "max_transient_resident": self._max_transient_resident,
            "max_transient_over_capacity": max(
                0, self._max_transient_resident - self.capacity
            ),
            "transient_over_capacity_count": (
                self._transient_over_capacity_count
            ),
            "final_resident": self._last_final_resident,
            "max_final_resident": self._max_final_resident,
            "persistent_trim_count": self._persistent_trim_count,
            "prefetch_drop_count": self._prefetch_drop_count,
            "forward_finalize_count": self._forward_finalize_count,
            "exception_finalize_count": self._exception_finalize_count,
            "pinned_nonresident_count": len(
                self.pinned.difference(self.gpu_resident.keys())
            ),
        }

    def reset_stats(self):
        if self._exact_active_ticket is not None:
            raise RuntimeError("cannot reset stats during a route-exact forward")
        self._harvest_exact_timings()
        self.hit_count = 0
        self.miss_count = 0
        self.transfer_bytes = 0
        self.transfer_time_ms = 0.0
        self.eviction_count = 0
        self.calls = 0
        self.strict_wave_forward_count = 0
        self.strict_multi_wave_forward_count = 0
        self.strict_total_waves = 0
        self.strict_max_waves_per_forward = 0
        self._pending_transfer_timings = []
        self.resident_update_count = 0
        self.resident_update_skip_count = 0
        self.resident_async_copy_count = 0
        self.sync_copy_event_wait_count = 0
        self.sync_copy_stream_wait_count = 0
        self._exact_timing_tickets = []
        self.exact_begin_calls = 0
        self.exact_wait_count = 0
        self.exact_post_count = 0
        self.exact_abort_count = 0
        self.exact_begin_exception_count = 0
        self.exact_fallback_count = 0
        self._exact_fallback_reasons = {}
        self.exact_wait_timing_fallback_count = 0
        self.exact_required_expert_count = 0
        self.exact_hit_count = 0
        self.exact_miss_count = 0
        self.exact_async_copy_count = 0
        self.exact_transfer_bytes = 0
        self.exact_raw_dma_ms = 0.0
        self.exact_exposed_wait_ms = 0.0
        self.exact_estimated_hidden_dma_ms = 0.0
        self.exact_aborted_raw_dma_ms = 0.0
        self.exact_timed_ticket_count = 0
        self.exact_route_split_attempt_count = 0
        self.exact_route_split_used_count = 0
        self.exact_route_split_ready_routes = 0
        self.exact_route_split_blocked_routes = 0
        self._exact_route_split_fallback_reasons = {}
        current = len(self.gpu_resident)
        self._max_active_forward_depth = self._active_forward_depth
        self._max_transient_resident = current
        self._max_final_resident = current if self._active_forward_depth == 0 else 0
        self._last_final_resident = (
            current if self._active_forward_depth == 0 else self._last_final_resident
        )
        self._transient_over_capacity_count = 0
        self._persistent_trim_count = 0
        self._prefetch_drop_count = 0
        self._forward_finalize_count = 0
        self._exception_finalize_count = 0


def _zone_total(stats):
    return stats["pinned_hit"] + stats["working_hit"] + stats["miss"]
