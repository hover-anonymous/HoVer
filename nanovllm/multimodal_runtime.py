"""Fail-closed DeepSeek-VL2 processor, chunk and embedding runtime.

This module mirrors the official ``prepare_inputs_embeds`` 2-D image layout
while making visual replacement transactional across HoVer prefill stages.
It deliberately accepts the official processor fields ``images`` and
``images_spatial_crop``; ``pixel_values`` is not a DeepSeek-VL2 output field.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Any, Callable, Mapping, Optional, Sequence

import torch


CONTRACT_VERSION = "deepseek-vl2-images-spatial-crop-v1"
EMBEDDING_PROTOCOL = "deepseek-vl2-official-2d-chunk-offset-v1"
KIMI_CONTRACT_VERSION = "kimi-vl-pixel-grid-v1"
KIMI_EMBEDDING_PROTOCOL = "kimi-vl-moonvit-direct-chunk-offset-v1"
SUPPORTED_CONTRACTS = frozenset((CONTRACT_VERSION, KIMI_CONTRACT_VERSION))
SUPPORTED_EMBEDDING_PROTOCOLS = frozenset(
    (EMBEDDING_PROTOCOL, KIMI_EMBEDDING_PROTOCOL)
)


class MultimodalContractError(ValueError):
    pass


def _plain_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MultimodalContractError(f"{name} must be an integer")
    return value


def _tensor(name: str, value: Any, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
        raise MultimodalContractError(
            f"processor {name} must be a rank-{ndim} tensor, got {shape}"
        )
    return value


def build_user_content(prompt: str, image_count: int) -> str:
    if not isinstance(prompt, str):
        raise MultimodalContractError("prompt must be a string")
    image_count = _plain_int("image_count", image_count)
    if image_count < 0:
        raise MultimodalContractError("image_count cannot be negative")
    if "<image>" in prompt:
        raise MultimodalContractError(
            "prompt must not contain <image>; images are supplied separately"
        )
    return "<image>\n" * image_count + prompt


@dataclass(frozen=True)
class ProcessorPayload:
    token_ids: list[int]
    images_seq_mask: list[bool]
    token_modalities: list[int]
    images: torch.Tensor
    images_spatial_crop: torch.Tensor
    num_visual_tokens: int
    num_image_tiles: int
    image_bytes: int
    contract: str = CONTRACT_VERSION
    embedding_protocol: str = EMBEDDING_PROTOCOL
    transport_dtype: str = "bfloat16"

    def protocol_fields(self) -> dict[str, Any]:
        return {
            "server_multimodal_contract": self.contract,
            "server_multimodal_embedding_protocol": self.embedding_protocol,
            "server_multimodal_transport_dtype": self.transport_dtype,
            "server_multimodal_visual_tokens": self.num_visual_tokens,
            "server_multimodal_image_tiles": self.num_image_tiles,
            "server_multimodal_image_bytes": self.image_bytes,
            "server_multimodal_spatial_crop": self.images_spatial_crop.tolist(),
        }


def payload_from_processor_output(
    processed: Any,
    *,
    image_count: int,
    max_visual_tokens: int,
) -> ProcessorPayload:
    image_count = _plain_int("image_count", image_count)
    max_visual_tokens = _plain_int("max_visual_tokens", max_visual_tokens)
    if image_count < 0 or max_visual_tokens <= 0:
        raise MultimodalContractError("invalid image/token limit")
    ids = _tensor("input_ids", getattr(processed, "input_ids", None), 2)
    mask_t = _tensor(
        "images_seq_mask", getattr(processed, "images_seq_mask", None), 2
    )
    images = _tensor("images", getattr(processed, "images", None), 5)
    crop = _tensor(
        "images_spatial_crop",
        getattr(processed, "images_spatial_crop", None),
        3,
    )
    if ids.dtype != torch.long or mask_t.dtype != torch.bool:
        raise MultimodalContractError("input_ids/mask must be Long/Bool")
    if crop.dtype != torch.long:
        raise MultimodalContractError("images_spatial_crop must be Long")
    if not images.is_floating_point() or images.shape[2] != 3:
        raise MultimodalContractError("images must be floating-point RGB")
    if images.shape[3] <= 0 or images.shape[4] <= 0:
        raise MultimodalContractError("images have an empty spatial axis")
    if not bool(torch.isfinite(images).all().item()):
        raise MultimodalContractError("images contain non-finite values")
    if ids.shape[0] != 1 or mask_t.shape != ids.shape:
        raise MultimodalContractError("endpoint requires one aligned request")
    if images.shape[0] != 1 or crop.shape[0] != 1:
        raise MultimodalContractError("image tensors must have batch size one")

    token_ids = ids[0].tolist()
    mask = mask_t[0].tolist()
    visual_tokens = sum(mask)
    if visual_tokens > max_visual_tokens:
        raise MultimodalContractError(
            f"visual token count {visual_tokens} exceeds {max_visual_tokens}"
        )
    crop_rows: list[tuple[int, int]] = []
    saw_padding = False
    for width, height in crop[0].tolist():
        if width == 0 and height == 0:
            saw_padding = True
            continue
        if saw_padding or width <= 0 or height <= 0:
            raise MultimodalContractError("invalid/padded crop row ordering")
        crop_rows.append((width, height))
    if len(crop_rows) != image_count:
        raise MultimodalContractError(
            f"processor images={len(crop_rows)}, request images={image_count}"
        )
    tiles = sum(1 + width * height for width, height in crop_rows)
    if image_count:
        if visual_tokens <= 0 or images.shape[1] != tiles:
            raise MultimodalContractError(
                f"visual slots/tiles inconsistent: slots={visual_tokens}, "
                f"tiles={images.shape[1]}, expected_tiles={tiles}"
            )
    elif visual_tokens or tiles:
        raise MultimodalContractError("text request contains visual state")

    canonical_images = images[0, :tiles].detach().to(
        device="cpu", dtype=torch.bfloat16
    ).contiguous()
    canonical_crop = crop[0, :image_count].detach().cpu().contiguous()
    return ProcessorPayload(
        token_ids,
        mask,
        [1 if value else 0 for value in mask],
        canonical_images,
        canonical_crop,
        visual_tokens,
        tiles,
        canonical_images.numel() * canonical_images.element_size(),
    )


def payload_from_kimi_processor_output(
    processed: Any,
    *,
    media_placeholder_token_id: int,
    image_count: int,
    max_visual_tokens: int,
) -> ProcessorPayload:
    """Canonicalize the official Kimi-VL processor output for transport."""
    image_count = _plain_int("image_count", image_count)
    max_visual_tokens = _plain_int("max_visual_tokens", max_visual_tokens)
    media_placeholder_token_id = _plain_int(
        "media_placeholder_token_id", media_placeholder_token_id
    )
    if image_count < 0 or max_visual_tokens <= 0:
        raise MultimodalContractError("invalid image/token limit")

    ids = _tensor("input_ids", getattr(processed, "input_ids", None), 2)
    pixels = _tensor("pixel_values", getattr(processed, "pixel_values", None), 4)
    grid = _tensor(
        "image_grid_hws", getattr(processed, "image_grid_hws", None), 2
    )
    if ids.dtype != torch.long or grid.dtype != torch.long:
        raise MultimodalContractError("Kimi input_ids/image_grid_hws must be Long")
    if ids.shape[0] != 1:
        raise MultimodalContractError("endpoint requires one aligned request")
    if grid.shape != (image_count, 2):
        raise MultimodalContractError(
            f"Kimi processor grids={grid.shape[0]}, request images={image_count}"
        )
    if not pixels.is_floating_point() or pixels.shape[1] != 3:
        raise MultimodalContractError(
            "Kimi pixel_values must be floating-point [patches,3,H,W]"
        )
    if not bool(torch.isfinite(pixels).all().item()):
        raise MultimodalContractError("Kimi pixel_values contain non-finite values")

    token_ids = ids[0].tolist()
    mask = [token == media_placeholder_token_id for token in token_ids]
    visual_tokens = sum(mask)
    if visual_tokens > max_visual_tokens:
        raise MultimodalContractError(
            f"visual token count {visual_tokens} exceeds {max_visual_tokens}"
        )
    merge_kernel = 4
    expected_visual_tokens = sum(
        int(height) * int(width) // merge_kernel for height, width in grid.tolist()
    )
    if visual_tokens != expected_visual_tokens:
        raise MultimodalContractError(
            f"Kimi visual slots={visual_tokens}, grids imply={expected_visual_tokens}"
        )
    expected_patches = sum(int(height) * int(width) for height, width in grid.tolist())
    if int(pixels.shape[0]) != expected_patches:
        raise MultimodalContractError(
            f"Kimi patches={pixels.shape[0]}, grids imply={expected_patches}"
        )

    canonical_pixels = pixels.detach().to(
        device="cpu", dtype=torch.bfloat16
    ).contiguous()
    canonical_grid = grid.detach().cpu().contiguous()
    return ProcessorPayload(
        token_ids,
        mask,
        [1 if value else 0 for value in mask],
        canonical_pixels,
        canonical_grid,
        visual_tokens,
        int(grid.shape[0]),
        canonical_pixels.numel() * canonical_pixels.element_size(),
        contract=KIMI_CONTRACT_VERSION,
        embedding_protocol=KIMI_EMBEDDING_PROTOCOL,
    )


def image_slot_ordinals(
    full_mask: Sequence[bool], start: int, end: int, inject: bool
) -> list[int]:
    if any(type(value) is not bool for value in full_mask):
        raise MultimodalContractError("images_seq_mask entries must be bool")
    start, end = _plain_int("start", start), _plain_int("end", end)
    if start < 0 or end < start or end > len(full_mask):
        raise MultimodalContractError("chunk lies outside images_seq_mask")
    if not inject:
        return [-1] * (end - start)
    ordinal = sum(full_mask[:start])
    result = []
    for value in full_mask[start:end]:
        result.append(ordinal if value else -1)
        ordinal += int(value)
    return result


def _expected_tiles(crop: torch.Tensor) -> int:
    if (
        not isinstance(crop, torch.Tensor)
        or crop.dtype != torch.long
        or crop.ndim != 2
        or crop.shape[1] != 2
    ):
        raise MultimodalContractError("crop must be Long[n_images,2]")
    total = 0
    for width, height in crop.tolist():
        if width <= 0 or height <= 0:
            raise MultimodalContractError("crop rows must be positive")
        total += 1 + width * height
    return total


def format_projected_tiles_2d(
    projected: torch.Tensor,
    crop: torch.Tensor,
    newline: torch.Tensor,
    separator: torch.Tensor,
    global_view_pos: str,
) -> torch.Tensor:
    if projected.ndim != 3:
        raise MultimodalContractError("projected tiles must be [tiles,hw,hidden]")
    num_tiles, hw, hidden = map(int, projected.shape)
    side = isqrt(hw)
    if side * side != hw or num_tiles != _expected_tiles(crop):
        raise MultimodalContractError("projector shape does not match crop plan")
    if newline.shape != (hidden,) or separator.shape != (hidden,):
        raise MultimodalContractError("newline/separator hidden size mismatch")
    if global_view_pos not in ("head", "tail"):
        raise MultimodalContractError("only head/tail global view is supported")
    output, tile = [], 0
    for width, height in crop.tolist():
        local_count = width * height
        global_view = projected[tile].view(side, side, hidden)
        global_view = torch.cat(
            (global_view, newline.view(1, 1, hidden).expand(side, 1, hidden)),
            dim=1,
        ).reshape(-1, hidden)
        local = projected[tile + 1:tile + 1 + local_count]
        local = local.view(height, width, side, side, hidden)
        local = local.permute(0, 2, 1, 3, 4).reshape(
            height * side, width * side, hidden
        )
        local = torch.cat(
            (
                local,
                newline.view(1, 1, hidden).expand(height * side, 1, hidden),
            ),
            dim=1,
        ).reshape(-1, hidden)
        sep = separator.view(1, hidden)
        views = (global_view, sep, local) if global_view_pos == "head" else (
            local, sep, global_view
        )
        output.append(torch.cat(views, dim=0))
        tile += 1 + local_count
    return torch.cat(output, dim=0) if output else projected.new_empty((0, hidden))


@dataclass(frozen=True)
class RequestVisualPayload:
    images: Optional[torch.Tensor]
    images_spatial_crop: Optional[torch.Tensor]
    expected_slots: int


@dataclass
class _CacheEntry:
    embeddings: torch.Tensor
    committed_next_ordinal: int = 0


@dataclass(frozen=True)
class VisualTransaction:
    seq_id: Any
    expected_start: int
    next_ordinal: int
    release: bool


@dataclass(frozen=True)
class ScatterTelemetry:
    transactions: tuple[VisualTransaction, ...] = ()


class VL2EmbeddingCache:
    """Rank-local cache whose progress changes only after forward succeeds."""

    def __init__(self, newline, separator, global_view_pos="head", limit=64):
        self.newline = newline
        self.separator = separator
        self.global_view_pos = global_view_pos
        self.limit = _plain_int("cache limit", limit)
        if self.limit <= 0:
            raise MultimodalContractError("cache limit must be positive")
        self._cache: dict[Any, _CacheEntry] = {}

    @property
    def keys(self):
        return tuple(self._cache)

    def clear(self, seq_id: Any) -> bool:
        return self._cache.pop(seq_id, None) is not None

    def _encode(self, seq_id, payload, encoder, on_encoded=None):
        if payload.images is None or payload.images_spatial_crop is None:
            raise MultimodalContractError(
                f"cache miss for {seq_id!r} without canonical images/crop"
            )
        images, crop = payload.images, payload.images_spatial_crop
        if (
            isinstance(payload.expected_slots, bool)
            or not isinstance(payload.expected_slots, int)
            or payload.expected_slots <= 0
        ):
            raise MultimodalContractError("expected_slots must be a positive integer")
        if (
            images.ndim != 4
            or images.shape[1] != 3
            or images.shape[2] <= 0
            or images.shape[3] <= 0
            or not images.is_floating_point()
            or not bool(torch.isfinite(images).all().item())
        ):
            raise MultimodalContractError("request images must be finite [tiles,3,H,W]")
        if len(self._cache) >= self.limit:
            raise MultimodalContractError("visual cache full; refusing silent eviction")
        projected = encoder(images)
        formatted = format_projected_tiles_2d(
            projected, crop, self.newline, self.separator,
            self.global_view_pos,
        )
        if formatted.shape[0] != payload.expected_slots:
            raise MultimodalContractError(
                f"formatted slots={formatted.shape[0]}, mask={payload.expected_slots}"
            )
        self._cache[seq_id] = _CacheEntry(formatted)
        if on_encoded is not None:
            on_encoded(seq_id, formatted, isqrt(int(projected.shape[1])))
        return self._cache[seq_id]

    def scatter(
        self,
        text_embeddings: torch.Tensor,
        token_seqids: Sequence[Any],
        ordinals: Sequence[int],
        payloads: Mapping[Any, RequestVisualPayload],
        encoder: Callable[[torch.Tensor], torch.Tensor],
        on_encoded: Optional[Callable[[Any, torch.Tensor, int], None]] = None,
    ) -> tuple[torch.Tensor, ScatterTelemetry]:
        if text_embeddings.ndim != 2:
            raise MultimodalContractError("text embeddings must be [rows,hidden]")
        ordinals = list(ordinals)
        if len(token_seqids) != text_embeddings.shape[0] or len(ordinals) != len(token_seqids):
            raise MultimodalContractError("multimodal row metadata is misaligned")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in ordinals):
            raise MultimodalContractError("visual ordinals must be integers")
        groups: dict[Any, list[tuple[int, int]]] = {}
        for row, (seq_id, ordinal) in enumerate(zip(token_seqids, ordinals)):
            if ordinal < -1:
                raise MultimodalContractError("visual ordinal cannot be below -1")
            if ordinal >= 0:
                groups.setdefault(seq_id, []).append((row, ordinal))
        if not groups:
            return text_embeddings, ScatterTelemetry()

        result = text_embeddings.clone()
        transactions = []
        for seq_id, rows in groups.items():
            payload = payloads.get(seq_id)
            entry = self._cache.get(seq_id)
            if entry is None:
                if payload is None:
                    raise MultimodalContractError(f"missing payload/cache for {seq_id!r}")
                entry = self._encode(
                    seq_id, payload, encoder, on_encoded=on_encoded
                )
            else:
                if payload is not None and payload.expected_slots != entry.embeddings.shape[0]:
                    raise MultimodalContractError("cached slot count changed")
            row_ids = [row for row, _ in rows]
            slots = [slot for _, slot in rows]
            start = entry.committed_next_ordinal
            if slots != list(range(start, start + len(slots))):
                raise MultimodalContractError(
                    f"slots for {seq_id!r} must continue at {start}, got {slots}"
                )
            if slots[-1] >= entry.embeddings.shape[0]:
                raise MultimodalContractError("visual slot exceeds formatted embeddings")
            result[torch.tensor(row_ids, device=result.device)] = entry.embeddings[
                torch.tensor(slots, device=entry.embeddings.device)
            ].to(device=result.device, dtype=result.dtype)
            next_ordinal = start + len(slots)
            transactions.append(
                VisualTransaction(
                    seq_id, start, next_ordinal,
                    next_ordinal == entry.embeddings.shape[0],
                )
            )
        return result, ScatterTelemetry(tuple(transactions))

    def commit(self, transactions: Sequence[VisualTransaction]) -> int:
        transactions = tuple(transactions)
        seen = set()
        for txn in transactions:
            if txn.seq_id in seen:
                raise MultimodalContractError("duplicate transaction")
            seen.add(txn.seq_id)
            entry = self._cache.get(txn.seq_id)
            if entry is None or entry.committed_next_ordinal != txn.expected_start:
                raise MultimodalContractError("stale/missing visual transaction")
            if txn.next_ordinal <= txn.expected_start or txn.next_ordinal > entry.embeddings.shape[0]:
                raise MultimodalContractError("invalid visual transaction progress")
            if txn.release and txn.next_ordinal != entry.embeddings.shape[0]:
                raise MultimodalContractError("early visual cache release")
        for txn in transactions:
            self._cache[txn.seq_id].committed_next_ordinal = txn.next_ordinal
        released = 0
        for txn in transactions:
            if txn.release:
                released += int(self.clear(txn.seq_id))
        return released


class DirectEmbeddingCache:
    """Transactional cache for models whose projector already returns slots."""

    def __init__(self, limit=64):
        self.limit = _plain_int("cache limit", limit)
        if self.limit <= 0:
            raise MultimodalContractError("cache limit must be positive")
        self._cache: dict[Any, _CacheEntry] = {}

    def clear(self, seq_id: Any) -> bool:
        return self._cache.pop(seq_id, None) is not None

    def scatter(
        self,
        text_embeddings: torch.Tensor,
        token_seqids: Sequence[Any],
        ordinals: Sequence[int],
        payloads: Mapping[Any, RequestVisualPayload],
        encoder: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        on_encoded: Optional[Callable[[Any, torch.Tensor, int], None]] = None,
    ) -> tuple[torch.Tensor, ScatterTelemetry]:
        if text_embeddings.ndim != 2:
            raise MultimodalContractError("text embeddings must be [rows,hidden]")
        ordinals = list(ordinals)
        if len(token_seqids) != text_embeddings.shape[0] or len(ordinals) != len(token_seqids):
            raise MultimodalContractError("multimodal row metadata is misaligned")
        groups: dict[Any, list[tuple[int, int]]] = {}
        for row, (seq_id, ordinal) in enumerate(zip(token_seqids, ordinals)):
            if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < -1:
                raise MultimodalContractError("visual ordinals must be integers >= -1")
            if ordinal >= 0:
                groups.setdefault(seq_id, []).append((row, ordinal))
        if not groups:
            return text_embeddings, ScatterTelemetry()

        result = text_embeddings.clone()
        transactions = []
        for seq_id, rows in groups.items():
            payload = payloads.get(seq_id)
            entry = self._cache.get(seq_id)
            if entry is None:
                if payload is None or payload.images is None or payload.images_spatial_crop is None:
                    raise MultimodalContractError(
                        f"cache miss for {seq_id!r} without Kimi pixels/grid"
                    )
                if len(self._cache) >= self.limit:
                    raise MultimodalContractError(
                        "visual cache full; refusing silent eviction"
                    )
                embeddings = encoder(
                    payload.images, payload.images_spatial_crop
                )
                if embeddings.ndim != 2 or embeddings.shape[0] != payload.expected_slots:
                    raise MultimodalContractError(
                        f"Kimi projected slots={embeddings.shape[0]}, "
                        f"mask={payload.expected_slots}"
                    )
                entry = _CacheEntry(embeddings)
                self._cache[seq_id] = entry
                if on_encoded is not None:
                    on_encoded(seq_id, embeddings, 0)
            else:
                if payload is not None and payload.expected_slots != entry.embeddings.shape[0]:
                    raise MultimodalContractError("cached slot count changed")

            row_ids = [row for row, _ in rows]
            slots = [slot for _, slot in rows]
            start = entry.committed_next_ordinal
            if slots != list(range(slots[0], slots[0] + len(slots))):
                raise MultimodalContractError(
                    f"slots for {seq_id!r} must be contiguous, got {slots}"
                )
            if slots[0] > start:
                raise MultimodalContractError(
                    f"slots for {seq_id!r} skip committed ordinal {start}: {slots}"
                )
            if slots[-1] >= entry.embeddings.shape[0]:
                raise MultimodalContractError("visual slot exceeds projected embeddings")
            result[torch.tensor(row_ids, device=result.device)] = entry.embeddings[
                torch.tensor(slots, device=entry.embeddings.device)
            ].to(device=result.device, dtype=result.dtype)
            # Preemption can roll back the processed-token cursor and replay
            # already committed visual slots. Scatter cached embeddings
            # idempotently and advance only for previously unseen ordinals.
            next_ordinal = max(start, slots[-1] + 1)
            transactions.append(
                VisualTransaction(
                    seq_id,
                    start,
                    next_ordinal,
                    False,
                )
            )
        return result, ScatterTelemetry(tuple(transactions))

    def commit(self, transactions: Sequence[VisualTransaction]) -> int:
        transactions = tuple(transactions)
        seen = set()
        for txn in transactions:
            if txn.seq_id in seen:
                raise MultimodalContractError("duplicate transaction")
            seen.add(txn.seq_id)
            entry = self._cache.get(txn.seq_id)
            if entry is None or entry.committed_next_ordinal != txn.expected_start:
                raise MultimodalContractError("stale/missing visual transaction")
            if txn.next_ordinal < txn.expected_start or txn.next_ordinal > entry.embeddings.shape[0]:
                raise MultimodalContractError("invalid visual transaction progress")
        for txn in transactions:
            self._cache[txn.seq_id].committed_next_ordinal = txn.next_ordinal
        # ModelRunner releases the entry only after the whole prompt completes
        # its final model stage, which is safe against later prefill replay.
        return 0
