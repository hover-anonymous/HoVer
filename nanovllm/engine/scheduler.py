from __future__ import annotations
from collections import deque
import logging
import math
import time
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.horizontal_scheduler import HorizontalScheduler
from nanovllm.engine.modality_aware_partitioner import ModalityAwarePartitioner
from nanovllm.engine.vertical_scheduler import VerticalScheduler
from nanovllm.utils.utils import setup_file_logger
logger = setup_file_logger('scheduler', level=logging.INFO)

class Scheduler:
    _DEPTH_MIN_ESTIMATOR_SAMPLES = 3
    _DEPTH_CALIBRATION_SAMPLES = 4
    _DEPTH_GUARD_MARGIN_MS = 5.0
    _DEPTH_CALIBRATION_MAD_MULTIPLIER = 6.0
    _DEPTH_CALIBRATION_RELATIVE_HEADROOM = 2.0

    def __init__(self, config: Config):
        self.config = config
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.prefilling: deque[Sequence] = deque()
        self.decoding: deque[Sequence] = deque()
        self.max_model_len = config.max_model_len
        if int(getattr(config, 'hover_chunk_size', 0)) <= 0:
            raise ValueError('hover_chunk_size must be positive')
        self.modality_aware_partitioner = ModalityAwarePartitioner(
            getattr(config, 'hover_modality_alpha', 0.5)
        )
        self.horizontal_scheduler = HorizontalScheduler(
            config, self.modality_aware_partitioner
        )
        self.vertical_scheduler = VerticalScheduler(
            num_stages=config.num_stages,
            stage_policy=getattr(
                config, 'vertical_stage_policy', 'threshold_v1'
            ),
            group_tokens=getattr(config, 'vertical_group_tokens', 512),
        )
        self.model_runner = None
        self.model_num_layers = 0
        self.round_cost_ewma_ms = 0.0
        self.last_round_cost_ms = 0.0
        self.round_cost_sample_count = 0
        self.prefill_work_cost_samples = deque(maxlen=64)
        self.prefill_service_cost_samples = deque(maxlen=64)
        self.prefill_cost_calibration_observations = deque(maxlen=self._DEPTH_CALIBRATION_SAMPLES)
        self.prefill_cost_estimator_calibrated = False
        self.prefill_cost_calibration_seen = 0
        self.prefill_cost_quarantined_samples = 0
        self.last_prefill_service_ms = 0.0
        self.last_prefill_layer_work = 0
        self.last_prefill_combined_work = 0
        self.last_prefill_decode_count = 0
        self.dynamic_num_stages: int = self.vertical_scheduler.num_stages
        self.batch_prefill_tokens: int = 0
        self.stage_queue: list[deque[Sequence]] = [
            deque() for _ in range(self.vertical_scheduler.num_stages)
        ]
        self.current_stage = -1
        self.dynamic_num_stages = self.vertical_scheduler.num_stages
        self.batch_prefill_tokens: int = 0
        self.batch_dynamic_slot_reason: str = 'disabled'
        self.batch_dynamic_slot_predicted_ms: float = 0.0
        self.batch_dynamic_slot_pack_enabled: int = 0
        self._hover_dynamic_rescue_target_ids: tuple[int, ...] = ()
        self._hover_no_prefill_progress_rounds: int = 0

    def is_finished(self):
        return not self.waiting and (not self.prefilling) and (not self.decoding)

    def add(self, seq: Sequence):
        if len(seq) > self.max_model_len:
            raise ValueError(f'Sequence length {len(seq)} exceeds max model length {self.max_model_len}.')
        import time as _t
        now = _t.time()
        arrival_time = getattr(seq, 'arrival_time', None)
        if not isinstance(arrival_time, (int, float)) or arrival_time <= 0:
            arrival_time = now
            seq.arrival_time = arrival_time
        if getattr(seq, 'ttft_deadline', None) is None:
            ttft_slo_s = getattr(self.config, 'ttft_slo_ms', 500.0) / 1000.0
            seq.ttft_deadline = arrival_time + ttft_slo_s
        if not getattr(seq, 'tbt_slo_s', None):
            seq.tbt_slo_s = getattr(self.config, 'tbt_slo_ms', 80.0) / 1000.0
        self.waiting.append(seq)

    def schedule(self) -> list[Sequence]:
        return self.hover_schedule()

    def _reset_round_token_state(self) -> None:
        self.batch_prefill_tokens = 0

    def _commit_round_token_state(self, prefill_scheduled_seqs: list[Sequence], _decode_scheduled_seqs: list[Sequence]) -> None:
        prefill_tokens = sum((max(0, int(getattr(seq, 'num_tokens_to_process', 0) or 0)) for seq in prefill_scheduled_seqs))
        self.batch_prefill_tokens = prefill_tokens

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.stage = -1
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        assert len(seqs) == len(token_ids), f'Number of sequences and token IDs must match. ({len(seqs)} != {len(token_ids)})'
        for (seq, token_id) in zip(seqs, token_ids):
            if seq.status == SequenceStatus.PREFILLING:
                if seq.stage == seq.num_stages - 1:
                    seq.num_processed_tokens += seq.num_tokens_to_process
                    if seq.num_processed_tokens >= len(seq):
                        seq.status = SequenceStatus.DECODING
                        if seq in self.stage_queue[seq.stage]:
                            self.stage_queue[seq.stage].remove(seq)
                        seq.stage += 1
                        self.decoding.append(seq)
                        self.prefilling.remove(seq)
                    else:
                        seq.status = SequenceStatus.PREFILLING
                        if seq in self.stage_queue[seq.stage]:
                            self.stage_queue[seq.stage].remove(seq)
            if seq.status == SequenceStatus.DECODING:
                import time as _time
                token_time = _time.time()
                token_monotonic = _time.monotonic()
                if seq.first_token_time is None:
                    seq.first_token_time = token_time
                    seq.first_token_monotonic = token_monotonic
                seq.last_token_time = token_time
                seq.last_token_monotonic = token_monotonic
                seq.append_token(token_id)
                if not seq.ignore_eos and token_id == self.eos or seq.num_completion_tokens >= seq.max_tokens or len(seq) >= self.max_model_len:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.decoding.remove(seq)
        if hasattr(self, 'stage_queue') and (not any(self.stage_queue)):
            self.current_stage = -1

    @staticmethod
    def _hover_decode_tbt_slack_ms(seq, mono_now: float, wall_now: float):
        tbt_slo_s = float(getattr(seq, 'tbt_slo_s', 0.0) or 0.0)
        if tbt_slo_s <= 0.0:
            return None
        last_mono = getattr(seq, 'last_token_monotonic', None)
        if last_mono is not None:
            age_s = max(0.0, mono_now - float(last_mono))
        else:
            last_wall = getattr(seq, 'last_token_time', None) or getattr(seq, 'first_token_time', None)
            if last_wall is None:
                return None
            age_s = max(0.0, wall_now - float(last_wall))
        return 1000.0 * (tbt_slo_s - age_s)

    @staticmethod
    def _hover_decode_slot_cap(seq_limit: int, inflight_prefill_width: int, fresh_prefill_reserve: int) -> int:
        return max(0, int(seq_limit) - max(0, int(inflight_prefill_width)) - max(0, int(fresh_prefill_reserve)))

    @staticmethod
    def _hover_should_apply_tbt_floor(fresh_prefill_reserve: int, dynamic_slot_reason: str) -> bool:
        return int(fresh_prefill_reserve) > 0 and dynamic_slot_reason == 'trigger_slo_safe_one_slot'

    @classmethod
    def _hover_bounded_fallback_stages(cls, proposed_stages: int, base_stages: int, predicted_mixed_ms: float, nominal_tbt_slo_ms: float) -> int:
        proposed = max(1, int(proposed_stages))
        base = max(proposed, int(base_stages))
        tbt_ms = float(nominal_tbt_slo_ms or 0.0)
        predicted_ms = float(predicted_mixed_ms or 0.0)
        if tbt_ms <= 0.0 or predicted_ms <= 0.0:
            return proposed
        target = int(math.ceil((predicted_ms + cls._DEPTH_GUARD_MARGIN_MS) / tbt_ms))
        return max(proposed, min(base, max(1, target)))

    def _hover_dynamic_prefill_reserve(self, pending_prefill: list, decode_candidates: list, seq_limit: int, decode_token_cap: int, now: float) -> int:
        cfg = self.config
        reserve_max = max(0, int(getattr(cfg, 'hover_prefill_reserve_max', 0) or 0))
        reserve_max = min(reserve_max, max(0, int(seq_limit) - 1))
        self.batch_dynamic_slot_reason = 'disabled'
        self.batch_dynamic_slot_predicted_ms = 0.0
        self._hover_dynamic_rescue_target_ids = ()
        no_progress_rounds = int(getattr(self, '_hover_no_prefill_progress_rounds', 0) or 0)
        if reserve_max <= 0:
            return 0
        if not pending_prefill:
            self.batch_dynamic_slot_reason = 'no_pending_prefill'
            return 0
        if any(self.stage_queue):
            self.batch_dynamic_slot_reason = 'stage_continuation'
            return 0
        if decode_token_cap < seq_limit or len(decode_candidates) < seq_limit:
            self.batch_dynamic_slot_reason = 'decode_not_slot_full'
            return 0
        quantum_ms = max(float(getattr(self, 'round_cost_ewma_ms', 0.0) or 0.0), float(getattr(self, 'last_round_cost_ms', 0.0) or 0.0))
        if quantum_ms <= 0.0:
            self.batch_dynamic_slot_reason = 'cost_estimator_cold'
            return 0
        ttft_rows = [(1000.0 * (float(deadline) - float(now)), float(deadline), seq) for seq in pending_prefill for deadline in [getattr(seq, 'ttft_deadline', None)] if deadline is not None]
        ttft_slacks = [row[0] for row in ttft_rows]
        if not ttft_slacks:
            self.batch_dynamic_slot_reason = 'missing_ttft_deadline'
            return 0
        hol_slack_ms = min(ttft_slacks)
        if hol_slack_ms > quantum_ms:
            self.batch_dynamic_slot_reason = 'ttft_can_wait_one_quantum'
            return 0
        mono_now = time.monotonic()
        tbt_slacks = [self._hover_decode_tbt_slack_ms(seq, mono_now, now) for seq in decode_candidates]
        if any((value is None for value in tbt_slacks)):
            self.batch_dynamic_slot_reason = 'missing_decode_timing'
            return 0
        min_tbt_slack_ms = min(tbt_slacks) if tbt_slacks else 1e309
        prefill_samples = [float(service_ms) for (service_ms, _work) in self._prefill_cost_observations()]
        prefill_p95_ms = self._depth_nearest_rank(prefill_samples, 95) if prefill_samples else 0.0
        predicted_ms = max(quantum_ms, prefill_p95_ms)
        self.batch_dynamic_slot_predicted_ms = predicted_ms
        endangered_rows = sorted((row for row in ttft_rows if row[0] <= predicted_ms + self._DEPTH_GUARD_MARGIN_MS), key=lambda row: (row[1], float(getattr(row[2], 'arrival_time', now) or now)))
        endangered_prefills = len(endangered_rows)
        if predicted_ms + self._DEPTH_GUARD_MARGIN_MS > min_tbt_slack_ms:
            decode_only_already_infeasible = quantum_ms + self._DEPTH_GUARD_MARGIN_MS > min_tbt_slack_ms
            admission_rho = self.vertical_scheduler.pressure_ratio(
                pending_prefill, decode_candidates, now, normalized=True
            )
            if no_progress_rounds >= 2 and (decode_only_already_infeasible or hol_slack_ms <= predicted_ms + self._DEPTH_GUARD_MARGIN_MS) and (admission_rho >= 0.5):
                target_reserve = min(reserve_max, len(pending_prefill), max(1, endangered_prefills))
                self._hover_dynamic_rescue_target_ids = tuple((id(row[2]) for row in endangered_rows[:target_reserve]))
                self.batch_dynamic_slot_reason = 'trigger_starvation_batched_fallback' if target_reserve > 1 else 'trigger_starvation_fallback'
                return target_reserve
            self.batch_dynamic_slot_reason = 'tbt_slack_insufficient'
            return 0
        self.batch_dynamic_slot_reason = 'trigger_slo_safe_one_slot'
        return min(1, reserve_max)

    @staticmethod
    def _hover_max_min_token_caps(candidates: list, token_budget: int) -> dict[int, int]:
        budget_left = max(0, int(token_budget))
        rows = [[seq, max(0, int(len(seq)) - int(getattr(seq, 'num_processed_tokens', 0) or 0))] for seq in candidates]
        rows = [row for row in rows if row[1] > 0]
        caps = {id(row[0]): 0 for row in rows}
        active = rows
        while active and budget_left > 0:
            share = budget_left // len(active)
            if share <= 0:
                for (seq, _remaining) in active[:budget_left]:
                    caps[id(seq)] += 1
                budget_left = 0
                break
            short = [row for row in active if row[1] <= share]
            if short:
                short_ids = {id(row[0]) for row in short}
                for (seq, remaining) in short:
                    caps[id(seq)] += remaining
                    budget_left -= remaining
                active = [row for row in active if id(row[0]) not in short_ids]
                continue
            for (seq, _remaining) in active:
                caps[id(seq)] += share
            budget_left -= share * len(active)
            for (seq, _remaining) in active[:budget_left]:
                caps[id(seq)] += 1
            budget_left = 0
        return caps

    def _hover_tbt_safe_stage_floor(self, proposed_stages: int, base_stages: int, decode_scheduled: list, now: float, target_ratio: float=0.0) -> int:
        proposed_stages = max(1, int(proposed_stages))
        base_stages = max(1, int(base_stages))
        if not decode_scheduled:
            return proposed_stages
        max_stages = max(proposed_stages, int(getattr(self.config, 'num_stages', base_stages) or base_stages))
        observations = self._prefill_cost_observations()
        if len(observations) < self._DEPTH_MIN_ESTIMATOR_SAMPLES:
            return max(proposed_stages, base_stages)
        model_layers = int(getattr(self, 'model_num_layers', 0) or 0)
        prefill_tokens = int(getattr(self, 'batch_prefill_tokens', 0) or 0)
        if model_layers <= 0 or prefill_tokens <= 0:
            return max(proposed_stages, base_stages)
        mono_now = time.monotonic()
        slacks = [self._hover_decode_tbt_slack_ms(seq, mono_now, now) for seq in decode_scheduled]
        if any((value is None for value in slacks)):
            return max(proposed_stages, base_stages)
        min_slack_ms = min(slacks)
        target_ratio = float(target_ratio or 0.0)
        if target_ratio > 0.0:
            nominal_tbt_slos_ms = [1000.0 * float(getattr(seq, 'tbt_slo_s', 0.0) or 0.0) for seq in decode_scheduled]
            nominal_tbt_slos_ms = [value for value in nominal_tbt_slos_ms if value > 0.0]
            if not nominal_tbt_slos_ms:
                return max(proposed_stages, base_stages)
            min_slack_ms = min(min_slack_ms, target_ratio * min(nominal_tbt_slos_ms))
        rates = [service_ms / float(work) for (service_ms, work) in observations]
        rate_p95 = self._depth_nearest_rank(rates, 95)
        rate_p25 = self._depth_nearest_rank(rates, 25)
        residuals = [max(0.0, service_ms - rate_p25 * float(work)) for (service_ms, work) in observations]
        positive_residuals = [value for value in residuals if value > 0.0]
        residual_p50_ms = self._depth_nearest_rank(positive_residuals, 50)
        for candidate_stages in range(proposed_stages, max_stages + 1):
            active_layers = (model_layers + candidate_stages - 1) // candidate_stages
            work = prefill_tokens * active_layers + len(decode_scheduled) * model_layers
            predicted_ms = self._depth_hybrid_cost_bound_ms(work, rate_p95, rate_p25, residual_p50_ms)
            if predicted_ms + self._DEPTH_GUARD_MARGIN_MS <= min_slack_ms:
                return candidate_stages
        return max_stages

    def hover_schedule(self) -> list[Sequence]:
        import time as _t
        now = _t.time()
        cfg = self.config
        self._reset_round_token_state()
        prefill_scheduled_seqs: list[Sequence] = []
        decode_scheduled_seqs: list[Sequence] = []
        num_seqs = 0
        num_batched_tokens = 0
        B_total = int(self.max_num_batched_tokens)
        inflight_prefill_tokens = 0
        inflight_prefill_width = 0
        if 0 <= self.current_stage < len(self.stage_queue):
            inflight_prefill_width = len(self.stage_queue[self.current_stage])
            inflight_prefill_tokens = sum((max(0, int(getattr(seq, 'num_tokens_to_process', 0) or 0)) for seq in self.stage_queue[self.current_stage]))
        decode_token_cap = max(0, B_total - inflight_prefill_tokens)
        seq_limit = max(1, int(self.max_num_seqs))
        self.batch_prefill_tokens = 0
        self.batch_dynamic_slot_reason = 'disabled'
        self.batch_dynamic_slot_predicted_ms = 0.0
        self.batch_dynamic_slot_pack_enabled = 0
        self._hover_dynamic_rescue_target_ids = ()
        decode_cands = list(self.decoding)
        pending_prefill = list(self.prefilling) + list(self.waiting)
        reserve = self._hover_dynamic_prefill_reserve(pending_prefill, decode_cands, seq_limit, decode_token_cap, now)
        decode_slot_cap = self._hover_decode_slot_cap(seq_limit, inflight_prefill_width, reserve)
        decode_limit = min(decode_slot_cap, decode_token_cap)
        for s in decode_cands:
            if hasattr(s, 'update_urgency'):
                s.update_urgency(now)
        decode_cands.sort(key=lambda s: getattr(s, 'urgency_score', 0.0), reverse=True)
        self.decoding = deque(decode_cands)
        while self.decoding and num_seqs < decode_limit:
            seq = self.decoding.popleft()
            while not self.block_manager.can_append(seq):
                if self.decoding:
                    self.preempt(self.decoding.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                decode_scheduled_seqs.append(seq)
        D = len(decode_scheduled_seqs)
        B_rem = max(0, B_total - D)
        selection_budget = B_rem
        if self.current_stage >= 0:
            while self.stage_queue[self.current_stage]:
                seq = self.stage_queue[self.current_stage][0]
                self.stage_queue[self.current_stage].popleft()
                self.prefilling.remove(seq)
                prefill_scheduled_seqs.append(seq)
                num_seqs += 1
        free_slots = max(0, seq_limit - num_seqs)
        if not prefill_scheduled_seqs and B_rem > 0 and (free_slots > 0):
            candidates = list(self.prefilling) + list(self.waiting)
            for s in candidates:
                if hasattr(s, 'update_urgency'):
                    s.update_urgency(now)
            rescue_target_ids = tuple(getattr(self, '_hover_dynamic_rescue_target_ids', ()) or ())
            rescue_id_rank = {seq_id: rank for (rank, seq_id) in enumerate(rescue_target_ids)}
            if self.batch_dynamic_slot_reason == 'trigger_starvation_batched_fallback' and len(rescue_target_ids) > 1:
                rescue_candidates = [seq for seq in candidates if id(seq) in rescue_id_rank]
                rescue_candidates.sort(key=lambda seq: rescue_id_rank[id(seq)])
                if len(rescue_candidates) == len(rescue_target_ids):
                    candidates = rescue_candidates
                    self.batch_dynamic_slot_pack_enabled = 1
            base_chunk = int(getattr(cfg, 'hover_chunk_size', 512) or 0)
            jmax = int(getattr(cfg, 'hover_jmax', 0))
            selection_budget = B_rem
            rescue_token_caps: dict[int, int] = {}
            if self.batch_dynamic_slot_pack_enabled:
                rescue_remaining = sum((max(0, int(len(seq)) - int(getattr(seq, 'num_processed_tokens', 0) or 0)) for seq in candidates))
                if rescue_remaining > selection_budget:
                    rescue_token_caps = self._hover_max_min_token_caps(candidates, selection_budget)
            selected = self.horizontal_scheduler.select_prefill(
                candidates,
                selection_budget,
                now,
                model_runner=self.model_runner,
                base_chunk=base_chunk,
                jmax=jmax,
                max_selected=free_slots,
                token_caps_by_id=rescue_token_caps,
            )
            ntp = 0
            for seq in selected:
                if num_seqs >= seq_limit or num_batched_tokens >= self.max_num_batched_tokens:
                    break
                is_from_prefilling = seq in self.prefilling
                if not is_from_prefilling:
                    if not self.block_manager.can_allocate(seq):
                        continue
                    self.block_manager.allocate(seq)
                    self.waiting.remove(seq)
                else:
                    self.prefilling.remove(seq)
                seq.stage = -1
                seq.num_stages = -1
                remaining = len(seq) - seq.num_processed_tokens
                available = max(0, min(selection_budget - num_batched_tokens, self.max_num_batched_tokens - D - num_batched_tokens))
                cap = self.modality_aware_partitioner.budget_limited_length(
                    remaining, available, base_chunk, jmax
                )
                if id(seq) in rescue_token_caps:
                    cap = min(cap, rescue_token_caps[id(seq)])
                ntp = cap
                if ntp < len(seq) - seq.num_processed_tokens:
                    ntp = self.modality_aware_partitioner.align_to_boundary(
                        seq, ntp
                    )
                if ntp <= 0:
                    if is_from_prefilling:
                        self.prefilling.append(seq)
                    else:
                        self.block_manager.deallocate(seq)
                        self.waiting.appendleft(seq)
                    continue
                num_seqs += 1
                seq.num_tokens_to_process = ntp
                num_batched_tokens += ntp
                self.batch_prefill_tokens += ntp
                prefill_scheduled_seqs.append(seq)
            if prefill_scheduled_seqs:
                ns_base = self.vertical_scheduler.base_advancement_span(
                    num_batched_tokens
                )
                rho = self.vertical_scheduler.pressure_ratio(
                    prefill_scheduled_seqs, decode_scheduled_seqs, now
                )
                ns = self.vertical_scheduler.dynamic_advancement_span(
                    ns_base, rho
                )
                if self._hover_should_apply_tbt_floor(reserve, self.batch_dynamic_slot_reason):
                    ns = self._hover_tbt_safe_stage_floor(ns, ns_base, decode_scheduled_seqs, now)
                elif reserve > 0:
                    nominal_tbt_slos = [1000.0 * float(getattr(seq, 'tbt_slo_s', 0.0) or 0.0) for seq in decode_scheduled_seqs]
                    nominal_tbt_slos = [value for value in nominal_tbt_slos if value > 0.0]
                    nominal_tbt_slo_ms = min(nominal_tbt_slos) if nominal_tbt_slos else 0.0
                    ns = self._hover_bounded_fallback_stages(ns, ns_base, self.batch_dynamic_slot_predicted_ms, nominal_tbt_slo_ms)
                for seq in prefill_scheduled_seqs:
                    seq.num_stages = ns
                self.dynamic_num_stages = ns
        if prefill_scheduled_seqs:
            for seq in prefill_scheduled_seqs:
                seq.status = SequenceStatus.PREFILLING
                seq.stage += 1
                if seq.stage < seq.num_stages:
                    self.stage_queue[seq.stage].append(seq)
                    self.current_stage = seq.stage
                else:
                    self.current_stage = -1
                self.prefilling.append(seq)
        if decode_scheduled_seqs:
            self.decoding.extendleft(reversed(decode_scheduled_seqs))
        if prefill_scheduled_seqs:
            self._hover_no_prefill_progress_rounds = 0
        elif pending_prefill and len(decode_cands) >= seq_limit:
            self._hover_no_prefill_progress_rounds = min(1000000, int(getattr(self, '_hover_no_prefill_progress_rounds', 0) or 0) + 1)
        else:
            self._hover_no_prefill_progress_rounds = 0
        self._commit_round_token_state(prefill_scheduled_seqs, decode_scheduled_seqs)
        return prefill_scheduled_seqs + decode_scheduled_seqs

    @staticmethod
    def _depth_nearest_rank(values, percentile: int) -> float:
        ordered = sorted((float(value) for value in values))
        if not ordered:
            return 0.0
        rank = (int(percentile) * len(ordered) + 99) // 100
        return ordered[max(0, min(len(ordered) - 1, rank - 1))]

    @staticmethod
    def _depth_hybrid_cost_bound_ms(work: int, rate_p95: float, rate_p25: float, intercept_p50_ms: float) -> float:
        work = max(0, int(work))
        return max(float(rate_p95) * work, float(rate_p25) * work + float(intercept_p50_ms))

    def _prefill_cost_observations(self) -> list[tuple[float, int]]:
        explicit = getattr(self, 'prefill_cost_estimator_observations', None)
        if explicit is not None:
            return [(float(service_ms), int(work)) for (service_ms, work) in explicit if float(service_ms) > 0.0 and int(work) > 0]
        rates = list(getattr(self, 'prefill_work_cost_samples', ()))
        services = list(getattr(self, 'prefill_service_cost_samples', ()))
        observations = []
        for (rate, service_ms) in zip(rates, services):
            rate = float(rate)
            service_ms = float(service_ms)
            if rate <= 0.0 or service_ms <= 0.0:
                continue
            observations.append((service_ms, max(1, int(round(service_ms / rate)))))
        return observations

    def _observe_prefill_cost_sample(self, service_ms: float, combined_work: int) -> None:
        observations = getattr(self, 'prefill_cost_estimator_observations', None)
        if observations is None:
            observations = deque(maxlen=64)
        calibration = getattr(self, 'prefill_cost_calibration_observations', None)
        if calibration is None:
            calibration = deque(maxlen=self._DEPTH_CALIBRATION_SAMPLES)
            self.prefill_cost_calibration_observations = calibration
        if bool(getattr(self, 'prefill_cost_estimator_calibrated', False)):
            observations.append((float(service_ms), int(combined_work)))
            return
        calibration.append((float(service_ms), int(combined_work)))
        self.prefill_cost_calibration_seen = int(getattr(self, 'prefill_cost_calibration_seen', 0)) + 1
        if len(calibration) < self._DEPTH_CALIBRATION_SAMPLES:
            return
        rates = [sample_ms / float(sample_work) for (sample_ms, sample_work) in calibration]
        median_rate = self._depth_nearest_rank(rates, 50)
        mad_rate = self._depth_nearest_rank([abs(rate - median_rate) for rate in rates], 50)
        upper_rate = median_rate + max(self._DEPTH_CALIBRATION_MAD_MULTIPLIER * mad_rate, self._DEPTH_CALIBRATION_RELATIVE_HEADROOM * median_rate)
        accepted_mask = [rate <= upper_rate for rate in rates]
        accepted = [observation for (observation, keep) in zip(calibration, accepted_mask) if keep]
        observations.extend(accepted)
        self.prefill_cost_quarantined_samples = int(getattr(self, 'prefill_cost_quarantined_samples', 0)) + len(calibration) - len(accepted)
        self.prefill_cost_estimator_calibrated = True

    def observe_round_service(self, service_ms: float, prefill_layer_work: int, decode_count: int, model_layers: int, prefill_tokens: int=0, active_layers: int=0, active_stage: int=-1, num_stages: int=-1) -> None:
        service_ms = max(0.0, float(service_ms))
        model_layers = max(0, int(model_layers))
        previous = float(getattr(self, 'round_cost_ewma_ms', 0.0) or 0.0)
        self.round_cost_ewma_ms = service_ms if previous <= 0.0 else 0.8 * previous + 0.2 * service_ms
        self.last_round_cost_ms = service_ms
        self.round_cost_sample_count = int(getattr(self, 'round_cost_sample_count', 0)) + 1
        combined_work = max(0, int(prefill_layer_work)) + max(0, int(decode_count)) * model_layers
        if prefill_layer_work > 0 and combined_work > 0:
            self.last_prefill_service_ms = service_ms
            self.last_prefill_layer_work = max(0, int(prefill_layer_work))
            self.last_prefill_combined_work = combined_work
            self.last_prefill_decode_count = max(0, int(decode_count))
            samples = getattr(self, 'prefill_work_cost_samples', None)
            if samples is None:
                samples = deque(maxlen=64)
                self.prefill_work_cost_samples = samples
            samples.append(service_ms / float(combined_work))
            service_samples = getattr(self, 'prefill_service_cost_samples', None)
            if service_samples is None:
                service_samples = deque(maxlen=64)
                self.prefill_service_cost_samples = service_samples
            service_samples.append(service_ms)
            self._observe_prefill_cost_sample(service_ms, combined_work)
