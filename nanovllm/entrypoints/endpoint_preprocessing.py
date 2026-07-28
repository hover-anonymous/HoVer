"""Non-blocking, ordered preprocessing for the benchmark API endpoint.

The FastAPI endpoint is an asyncio coroutine.  Image decoding and the
DeepSeek-VL2 processor are synchronous CPU work and must not execute on the
uvicorn event-loop thread: doing so pauses every active StreamingResponse.

One dedicated worker deliberately preserves arrival order and avoids assuming
that the third-party processor is thread-safe.  This is an endpoint transport
fix shared by every scheduler, not a scheduling-policy optimization.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import torch

from nanovllm.multimodal_runtime import (
    ProcessorPayload,
    build_user_content,
    payload_from_kimi_processor_output,
    payload_from_processor_output,
)


def _normalize_kimi_media_slots(processed: Any, tokenizer: Any) -> None:
    """Repair fragmented adjacent ``<|media_pad|>`` tokens in Kimi output.

    Kimi's remote processor expands every image placeholder by concatenating
    the media-pad string many times and then tokenizes the resulting text.  The
    fast tokenizer occasionally leaves one of those adjacent special-token
    strings split into ordinary sub-tokens for multi-image prompts.  That makes
    the number of media slots disagree with ``image_grid_hws`` and, if ignored,
    would misalign visual embeddings with language positions.

    Rebuild only the media spans, using the processor's own grid-derived slot
    counts.  Text outside ``media_start``/``media_end`` is preserved exactly.
    """
    ids = processed["input_ids"]
    grid = processed["image_grid_hws"]
    if ids.ndim != 2 or ids.shape[0] != 1 or grid.ndim != 2:
        return

    token_ids = ids[0].tolist()
    pad_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
    start_id = tokenizer.convert_tokens_to_ids("<|media_start|>")
    content_id = tokenizer.convert_tokens_to_ids("<|media_content|>")
    end_id = tokenizer.convert_tokens_to_ids("<|media_end|>")
    expected = [int(h) * int(w) // 4 for h, w in grid.tolist()]
    if token_ids.count(pad_id) == sum(expected):
        return

    rebuilt: list[int] = []
    cursor = 0
    for image_index, slot_count in enumerate(expected):
        try:
            start = token_ids.index(start_id, cursor)
            end = token_ids.index(end_id, start + 1)
        except ValueError as exc:
            raise ValueError(
                f"Kimi image {image_index} is missing media boundary tokens"
            ) from exc
        span = token_ids[start + 1 : end]
        try:
            content_offset = span.index(content_id)
        except ValueError as exc:
            raise ValueError(
                f"Kimi image {image_index} is missing media_content"
            ) from exc
        rebuilt.extend(token_ids[cursor : start + 1])
        rebuilt.extend(span[: content_offset + 1])
        rebuilt.extend([pad_id] * slot_count)
        rebuilt.append(end_id)
        cursor = end + 1
    rebuilt.extend(token_ids[cursor:])

    if rebuilt.count(start_id) != len(expected) or rebuilt.count(end_id) != len(expected):
        raise ValueError("Kimi media boundary count does not match image grids")
    processed["input_ids"] = torch.tensor(
        [rebuilt], dtype=ids.dtype, device=ids.device
    )
    if "attention_mask" in processed:
        processed["attention_mask"] = torch.ones(
            (1, len(rebuilt)),
            dtype=processed["attention_mask"].dtype,
            device=processed["attention_mask"].device,
        )


@dataclass(frozen=True)
class MultimodalPreprocessResult:
    token_ids: list[int]
    images_seq_mask: list[bool]
    token_modalities: list[int]
    images: torch.Tensor
    images_spatial_crop: torch.Tensor
    num_visual_tokens: int
    num_image_tiles: int
    image_bytes: int
    queue_ms: float
    compute_ms: float
    event_loop_return_ms: float
    total_ms: float
    contract: str
    embedding_protocol: str
    transport_dtype: str

    def timing_fields(self) -> dict[str, float]:
        """Return stable JSON field names for per-request audit telemetry."""
        return {
            "server_preprocess_queue_ms": round(self.queue_ms, 3),
            "server_preprocess_compute_ms": round(self.compute_ms, 3),
            "server_preprocess_event_loop_return_ms": round(
                self.event_loop_return_ms, 3
            ),
            "server_preprocess_total_ms": round(self.total_ms, 3),
        }

    def protocol_fields(self) -> dict[str, Any]:
        payload = ProcessorPayload(
            self.token_ids,
            self.images_seq_mask,
            self.token_modalities,
            self.images,
            self.images_spatial_crop,
            self.num_visual_tokens,
            self.num_image_tiles,
            self.image_bytes,
            contract=self.contract,
            embedding_protocol=self.embedding_protocol,
            transport_dtype=self.transport_dtype,
        )
        return payload.protocol_fields()


class OrderedMultimodalPreprocessor:
    """Run synchronous VL preprocessing away from the asyncio event loop.

    ``processor_loader`` is called inside the same dedicated worker as
    ``processor(...)``.  Consequently lazy model/processor initialization also
    cannot freeze token streaming.  ``max_workers=1`` is intentional: request
    preprocessing remains FIFO and the processor need not be thread-safe.
    """

    def __init__(
        self,
        processor_loader: Callable[[str], Any],
        *,
        executor: Optional[Executor] = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._processor_loader = processor_loader
        self._clock_ns = clock_ns
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="vl-endpoint-preprocess",
        )

    async def preprocess(
        self,
        model_path: str,
        prompt: str,
        image_paths: Iterable[str],
        *,
        max_visual_tokens: int,
    ) -> MultimodalPreprocessResult:
        submitted_ns = self._clock_ns()
        loop = asyncio.get_running_loop()
        payload, worker_start_ns, worker_end_ns = await loop.run_in_executor(
            self._executor,
            self._preprocess_sync,
            model_path,
            prompt,
            tuple(image_paths),
            max_visual_tokens,
        )
        returned_ns = self._clock_ns()
        return MultimodalPreprocessResult(
            token_ids=payload.token_ids,
            images_seq_mask=payload.images_seq_mask,
            token_modalities=payload.token_modalities,
            images=payload.images,
            images_spatial_crop=payload.images_spatial_crop,
            num_visual_tokens=payload.num_visual_tokens,
            num_image_tiles=payload.num_image_tiles,
            image_bytes=payload.image_bytes,
            queue_ms=max(0, worker_start_ns - submitted_ns) / 1e6,
            compute_ms=max(0, worker_end_ns - worker_start_ns) / 1e6,
            event_loop_return_ms=max(0, returned_ns - worker_end_ns) / 1e6,
            total_ms=max(0, returned_ns - submitted_ns) / 1e6,
            contract=payload.contract,
            embedding_protocol=payload.embedding_protocol,
            transport_dtype=payload.transport_dtype,
        )

    def _preprocess_sync(
        self,
        model_path: str,
        prompt: str,
        image_paths: tuple[str, ...],
        max_visual_tokens: int,
    ) -> tuple[ProcessorPayload, int, int]:
        worker_start_ns = self._clock_ns()
        processor = self._processor_loader(model_path)
        if processor.__class__.__name__ == "KimiVLProcessor":
            payload = self._preprocess_kimi(
                processor,
                prompt,
                image_paths,
                max_visual_tokens,
            )
            return payload, worker_start_ns, self._clock_ns()

        conversation = [
            {
                "role": "<|User|>",
                "content": build_user_content(prompt, len(image_paths)),
                "images": list(image_paths),
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = [self._load_rgb_image(path) for path in image_paths]
        try:
            processed = processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt="",
            )
        finally:
            for image in pil_images:
                close = getattr(image, "close", None)
                if close is not None:
                    close()

        payload = payload_from_processor_output(
            processed,
            image_count=len(image_paths),
            max_visual_tokens=max_visual_tokens,
        )
        return payload, worker_start_ns, self._clock_ns()

    def _preprocess_kimi(
        self,
        processor: Any,
        prompt: str,
        image_paths: tuple[str, ...],
        max_visual_tokens: int,
    ) -> ProcessorPayload:
        pil_images = [self._load_rgb_image(path) for path in image_paths]
        content = [
            {"type": "image", "image": path}
            for path in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        try:
            processed = processor(
                images=pil_images if len(pil_images) != 1 else pil_images[0],
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            _normalize_kimi_media_slots(processed, processor.tokenizer)
        finally:
            for image in pil_images:
                close = getattr(image, "close", None)
                if close is not None:
                    close()
        media_token_id = processor.tokenizer.convert_tokens_to_ids(
            "<|media_pad|>"
        )
        return payload_from_kimi_processor_output(
            processed,
            media_placeholder_token_id=media_token_id,
            image_count=len(image_paths),
            max_visual_tokens=max_visual_tokens,
        )

    @staticmethod
    def _load_rgb_image(path: str):
        from PIL import Image

        if isinstance(path, str) and path.startswith("data:image"):
            _, encoded = path.split(",", 1)
            source = io.BytesIO(base64.b64decode(encoded))
            with source, Image.open(source) as image:
                return image.convert("RGB")
        if isinstance(path, str) and path.startswith("file://"):
            path = path[len("file://") :]
        with Image.open(path) as image:
            return image.convert("RGB")

    def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=True, cancel_futures=False)
