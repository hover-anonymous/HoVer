from __future__ import annotations


class ModalityAwarePartitioner:
    """Generate budget-feasible prefill segments with soft modality alignment."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = float(alpha)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("modality-aware partitioning alpha must be in (0, 1]")

    @staticmethod
    def next_modality(seq) -> tuple[int, int]:
        remaining = len(seq) - seq.num_processed_tokens
        token_modalities = getattr(seq, "token_modalities", None)
        start = seq.num_processed_tokens
        if not token_modalities or start >= len(token_modalities):
            return (0, remaining)
        modality = token_modalities[start]
        end = min(len(token_modalities), len(seq))
        cursor = start
        while cursor < end and token_modalities[cursor] == modality:
            cursor += 1
        return (int(modality), max(1, cursor - start))

    @staticmethod
    def budget_limited_length(
        remaining: int,
        batch_remaining: int,
        base_chunk: int,
        jmax: int,
    ) -> int:
        remaining = int(remaining)
        batch_remaining = int(batch_remaining)
        base_chunk = int(base_chunk)
        jmax = int(jmax)
        if remaining <= 0 or batch_remaining <= 0:
            return 0
        if base_chunk <= 0:
            raise ValueError("base_chunk must be positive")
        chunk_length = min(remaining, batch_remaining)
        if jmax > 0:
            chunk_length = min(chunk_length, base_chunk * jmax)
        return max(0, chunk_length)

    def align_to_boundary(self, seq, chunk_length: int) -> int:
        token_modalities = getattr(seq, "token_modalities", None)
        start = seq.num_processed_tokens
        if (
            not token_modalities
            or start >= len(token_modalities)
            or chunk_length <= 1
        ):
            return chunk_length
        end = min(len(token_modalities), len(seq))
        last_boundary = 0
        for offset in range(1, min(chunk_length, end - start)):
            if (
                token_modalities[start + offset]
                != token_modalities[start + offset - 1]
            ):
                last_boundary = offset
        if last_boundary >= chunk_length * self.alpha:
            return last_boundary
        return chunk_length
