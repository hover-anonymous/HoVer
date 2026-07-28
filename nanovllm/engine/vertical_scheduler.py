from __future__ import annotations

import math


class VerticalScheduler:
    """Control static stage grouping and dynamic model-depth advancement."""

    SUPPORTED_STAGE_COUNTS = {4, 12, 16, 24}

    def __init__(
        self,
        num_stages: int,
        stage_policy: str = "threshold_v1",
        group_tokens: int = 512,
    ):
        self.num_stages = int(num_stages)
        self.stage_policy = str(stage_policy)
        self.group_tokens = int(group_tokens)
        if self.num_stages not in self.SUPPORTED_STAGE_COUNTS:
            raise ValueError(
                f"unsupported HoVer stage count: {self.num_stages}"
            )
        if self.stage_policy not in {"threshold_v1", "ceil_group_v1"}:
            raise ValueError(
                "vertical stage policy must be threshold_v1 or ceil_group_v1"
            )
        if self.group_tokens <= 0:
            raise ValueError("vertical group_tokens must be positive")

    def base_advancement_span(self, prefill_tokens: int) -> int:
        prefill_tokens = max(0, int(prefill_tokens))
        if self.stage_policy == "threshold_v1":
            if prefill_tokens <= 512:
                target = 1
            elif prefill_tokens <= 1024:
                target = 2
            elif prefill_tokens <= 2048:
                target = 4
            elif prefill_tokens <= 4096:
                target = (
                    6 if self.num_stages == 12
                    else min(8, self.num_stages)
                )
            else:
                target = self.num_stages
        else:
            target = max(1, math.ceil(prefill_tokens / self.group_tokens))
        return min(self.num_stages, target)

    def dynamic_advancement_span(
        self,
        base_span: int,
        prefill_pressure_ratio: float,
    ) -> int:
        pressure = min(1.0, max(0.0, float(prefill_pressure_ratio)))
        span = int(round(int(base_span) * (1.5 - pressure)))
        return max(1, min(self.num_stages, span))

    @staticmethod
    def pressure_ratio(
        prefill_seqs: list,
        decode_seqs: list,
        now: float,
        normalized: bool = False,
    ) -> float:
        eps = 0.001
        if normalized:
            def risk(seq) -> float:
                try:
                    return max(0.0, float(seq.update_urgency(now)))
                except Exception:
                    return max(
                        0.0,
                        float(getattr(seq, "urgency_score", 0.0) or 0.0),
                    )

            prefill_risk = [risk(seq) for seq in prefill_seqs]
            decode_risk = [risk(seq) for seq in decode_seqs]
            prefill_pressure = (
                sum(prefill_risk) / len(prefill_risk)
                if prefill_risk else 0.0
            )
            decode_pressure = (
                sum(decode_risk) / len(decode_risk)
                if decode_risk else 0.0
            )
        else:
            def slack(seq) -> float:
                token_time = (
                    getattr(seq, "last_token_time", None)
                    or getattr(seq, "first_token_time", None)
                )
                if token_time is not None:
                    slo = max(
                        getattr(seq, "tbt_slo_s", 0.08) or 0.08,
                        eps,
                    )
                    return max(slo - (now - token_time), 0.0)
                deadline = getattr(seq, "ttft_deadline", None)
                return 1.0 if deadline is None else max(deadline - now, 0.0)

            prefill_pressure = sum(
                1.0 / (slack(seq) + eps) for seq in prefill_seqs
            )
            decode_pressure = sum(
                1.0 / (slack(seq) + eps) for seq in decode_seqs
            )
        total = prefill_pressure + decode_pressure
        if total <= 0.0:
            return 0.5
        return float(prefill_pressure / total)
