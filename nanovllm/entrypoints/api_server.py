"""Minimal HTTP API for the HoVer research inference engine.

This server is intended for research evaluation and is not hardened for
production deployment.
"""
import json
import os
import sys
import uuid
import time
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


_deepseek_source = os.environ.get("DEEPSEEK_VL2_PATH")
if _deepseek_source:
    _p = Path(_deepseek_source).expanduser().resolve()
    if not _p.is_dir():
        raise RuntimeError(f"DEEPSEEK_VL2_PATH is not a directory: {_p}")
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

from nanovllm.sampling_params import SamplingParams
from nanovllm.config import PAPER_BASE_DEFAULTS
from nanovllm.engine.async_llm_engine import AsyncLLMEngine
from nanovllm.entrypoints.config import APIServerConfig
from nanovllm.entrypoints.endpoint_preprocessing import (
    OrderedMultimodalPreprocessor,
)
TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None

_VL_PROCESSOR = None


def _get_or_load_vl_processor(model_path: str):
    global _VL_PROCESSOR
    if _VL_PROCESSOR is None:
        from transformers import AutoConfig, AutoProcessor
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if getattr(config, "model_type", None) == "kimi_vl":
            _VL_PROCESSOR = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
        else:
            from deepseek_vl2.models import DeepseekVLV2Processor
            _VL_PROCESSOR = DeepseekVLV2Processor.from_pretrained(model_path)
    return _VL_PROCESSOR


# A single dedicated worker preserves preprocessing order and avoids assuming
# the third-party VL processor is thread-safe.  Most importantly, synchronous
# image/processor work no longer freezes every active StreamingResponse.
_VL_ENDPOINT_PREPROCESSOR = OrderedMultimodalPreprocessor(
    _get_or_load_vl_processor
)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def _request_timing_metadata(
    server_receive_wall_s,
    server_receive_wall_ns,
    server_receive_monotonic_ns,
    ttft_deadline_ms=None,
    tbt_slo_ms=None,
):
    """Build timing metadata carried with the request to ``_AsyncLLMEngine``.

    The wall-time arrival/deadline are applied to ``Sequence``.  Monotonic time
    remains audit-only in endpoint responses so the client can reconstruct
    server receive-to-first-token latency without mixing clocks.
    """
    metadata = {
        "_arrival_time": float(server_receive_wall_s),
        "_server_receive_wall_ns": int(server_receive_wall_ns),
        "_server_receive_monotonic_ns": int(server_receive_monotonic_ns),
        "_ttft_anchor_protocol": "server-endpoint-entry-v1",
    }
    if ttft_deadline_ms is not None:
        metadata["_ttft_deadline"] = (
            float(server_receive_wall_s) + float(ttft_deadline_ms) / 1000.0
        )
    if tbt_slo_ms is not None:
        metadata["_tbt_slo_s"] = float(tbt_slo_ms) / 1000.0
    return metadata


# ============================================================================

# ============================================================================

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)




@app.post("/generate")
async def generate(request: Request) -> Response:
    server_receive_wall_ns = time.time_ns()
    server_receive_wall_s = server_receive_wall_ns / 1e9
    server_receive_monotonic_ns = time.monotonic_ns()
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    request_dict.pop("model", None)
    
    ttft_deadline_ms = request_dict.pop("ttft_deadline_ms", None)
    tbt_slo_ms = request_dict.pop("tbt_slo_ms", None)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    # Always propagate the receive anchor, including requests that rely on the
    # server's default TTFT SLO.
    multimodal = _request_timing_metadata(
        server_receive_wall_s,
        server_receive_wall_ns,
        server_receive_monotonic_ns,
        ttft_deadline_ms,
        tbt_slo_ms,
    )

    assert engine is not None
    results_generator = engine.generate(request_id, prompt, sampling_params, multimodal=multimodal)

    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            text_outputs = request_output[1]
            token_ids = request_output[2]
            if isinstance(token_ids, np.ndarray):
                token_ids = token_ids.tolist()
            ret = {
                "generated_text": text_outputs,
                "output_tokens": token_ids,
                "server_monotonic_ns": time.monotonic_ns(),
                "server_receive_wall_ns": server_receive_wall_ns,
                "server_receive_monotonic_ns": server_receive_monotonic_ns,
                "server_request_id": request_id,
            }
            yield (json.dumps(ret) + "\n").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    final_output = None
    text_parts = []
    async for request_output in results_generator:
        if await request.is_disconnected():
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output
        text_parts.append(request_output[1])

    assert final_output is not None
    text_outputs = prompt + "".join(text_parts)
    return JSONResponse({
        "text": text_outputs,
        "server_request_id": request_id,
        "server_receive_wall_ns": server_receive_wall_ns,
        "server_receive_monotonic_ns": server_receive_monotonic_ns,
        "server_response_monotonic_ns": time.monotonic_ns(),
    })


# ============================================================================

# ============================================================================

@app.post("/generate_multimodal")
async def generate_multimodal(request: Request) -> Response:
    server_receive_wall_ns = time.time_ns()
    server_receive_wall_s = server_receive_wall_ns / 1e9
    server_receive_monotonic_ns = time.monotonic_ns()
    request_dict = await request.json()
    prompt = request_dict.pop("prompt", "")
    image_paths = request_dict.pop("images", []) or []
    stream = request_dict.pop("stream", False)
    request_dict.pop("model", None)
    ttft_deadline_ms = request_dict.pop("ttft_deadline_ms", None)
    tbt_slo_ms = request_dict.pop("tbt_slo_ms", None)

    sampling_params = SamplingParams(**request_dict)

    assert engine is not None
    model_path = engine.config.model
    max_visual_tokens = int(os.environ.get(
        "NANOVLLM_MAX_VISUAL_TOKENS",
        getattr(engine.config, "max_model_len", 4096),
    ))
    if max_visual_tokens <= 0:
        raise ValueError("NANOVLLM_MAX_VISUAL_TOKENS must be positive")
    preprocessed = await _VL_ENDPOINT_PREPROCESSOR.preprocess(
        model_path,
        prompt,
        image_paths,
        max_visual_tokens=max_visual_tokens,
    )
    token_ids = preprocessed.token_ids
    if len(token_ids) > int(getattr(engine.config, "max_model_len", 4096)):
        raise ValueError(
            f"processor token count {len(token_ids)} exceeds max_model_len "
            f"{engine.config.max_model_len}"
        )
    images_seq_mask = preprocessed.images_seq_mask
    token_modalities = preprocessed.token_modalities
    images = preprocessed.images
    images_spatial_crop = preprocessed.images_spatial_crop
    preprocess_timing = preprocessed.timing_fields()
    multimodal_protocol = preprocessed.protocol_fields()

    multimodal = {
        "images": images,
        "images_spatial_crop": images_spatial_crop,
        "token_modalities": token_modalities,
        "images_seq_mask": images_seq_mask,
        "num_visual_tokens": preprocessed.num_visual_tokens,
        "multimodal_contract": preprocessed.contract,
        "multimodal_embedding_protocol": preprocessed.embedding_protocol,
    }
    multimodal.update(_request_timing_metadata(
        server_receive_wall_s,
        server_receive_wall_ns,
        server_receive_monotonic_ns,
        ttft_deadline_ms,
        tbt_slo_ms,
    ))

    request_id = random_uuid()
    results_generator = engine.generate(
        request_id, token_ids, sampling_params, multimodal=multimodal
    )

    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            text_outputs = request_output[1]
            tokens = request_output[2]
            if isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            ret = {
                "generated_text": text_outputs,
                "output_tokens": tokens,
                "server_monotonic_ns": time.monotonic_ns(),
                "server_receive_wall_ns": server_receive_wall_ns,
                "server_receive_monotonic_ns": server_receive_monotonic_ns,
                "server_request_id": request_id,
            }
            ret.update(preprocess_timing)
            ret.update(multimodal_protocol)
            yield (json.dumps(ret) + "\n").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    final_output = None
    text_parts = []
    async for request_output in results_generator:
        if await request.is_disconnected():
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output
        text_parts.append(request_output[1])

    assert final_output is not None
    text_outputs = "".join(text_parts)
    response_payload = {
        "text": text_outputs,
        "server_request_id": request_id,
        "server_receive_wall_ns": server_receive_wall_ns,
        "server_receive_monotonic_ns": server_receive_monotonic_ns,
        "server_response_monotonic_ns": time.monotonic_ns(),
    }
    response_payload.update(preprocess_timing)
    response_payload.update(multimodal_protocol)
    return JSONResponse(response_payload)


def parse_args() -> APIServerConfig:
    """Parse command line arguments and return the config."""
    import argparse

    parser = argparse.ArgumentParser(description="API server for AsyncLLMEngine.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model directory.")
    parser.add_argument(
        "--max-num-batched-tokens", type=int,
        default=PAPER_BASE_DEFAULTS["max_num_batched_tokens"],
    )
    parser.add_argument(
        "--max-num-seqs", type=int,
        default=PAPER_BASE_DEFAULTS["max_num_seqs"],
    )
    parser.add_argument(
        "--max-model-len", type=int,
        default=PAPER_BASE_DEFAULTS["max_model_len"],
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float,
        default=PAPER_BASE_DEFAULTS["gpu_memory_utilization"],
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int,
        default=PAPER_BASE_DEFAULTS["tensor_parallel_size"],
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--log-level", type=str, default="debug")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--nccl-port", type=int, default=2333)
    parser.add_argument(
        "--num-stages", type=int,
        default=PAPER_BASE_DEFAULTS["num_stages"],
    )
    parser.add_argument(
        "--vertical-stage-policy",
        choices=["threshold_v1", "ceil_group_v1"],
        default=PAPER_BASE_DEFAULTS["vertical_stage_policy"],
        help="HoVer depth mapping policy",
    )
    parser.add_argument(
        "--vertical-group-tokens",
        type=int,
        default=PAPER_BASE_DEFAULTS["vertical_group_tokens"],
        help=(
            "Tokens per target stage when "
            "--vertical-stage-policy=ceil_group_v1"
        ),
    )


    # MULTIMODAL
    parser.add_argument("--trust-remote-code", action="store_true", default=False)

    # OFFLOAD
    parser.add_argument(
        "--offload-strategy", type=str,
        default=PAPER_BASE_DEFAULTS["offload_strategy"],
                        choices=["none", "naive", "lru", "dlfu", "predicted"])
    parser.add_argument(
        "--gpu-expert-cache-slots", type=int,
        default=PAPER_BASE_DEFAULTS["gpu_expert_cache_slots"],
    )
    parser.add_argument("--offload-host-pinned", action="store_true", default=True)
    parser.add_argument(
        "--strict-logical-expert-capacity",
        action="store_true",
        default=False,
        help=(
            "Keep the full logical expert tensor allocated, but execute each "
            "MoE forward in route waves that never make more than "
            "--gpu-expert-cache-slots experts logically resident"
        ),
    )

    # SLO-aware HoVer controls
    parser.add_argument(
        "--ttft-slo-ms", type=float,
        default=PAPER_BASE_DEFAULTS["ttft_slo_ms"],
    )
    parser.add_argument(
        "--tbt-slo-ms", type=float,
        default=PAPER_BASE_DEFAULTS["tbt_slo_ms"],
    )
    parser.add_argument(
        "--ttl-max", type=int, default=PAPER_BASE_DEFAULTS["ttl_max"],
    )
    parser.add_argument(
        "--theta", type=int, default=PAPER_BASE_DEFAULTS["theta"],
    )
    parser.add_argument(
        "--expert-transition-decay", type=float,
        default=PAPER_BASE_DEFAULTS["expert_transition_decay"],
                        help="Expert-transition decay from Paper Eq. (4)")

    # HoVer controls
    parser.add_argument(
        "--hover-kh", type=int, default=PAPER_BASE_DEFAULTS["hover_kh"],
    )
    parser.add_argument(
        "--hover-pin-ratio", type=float,
        default=PAPER_BASE_DEFAULTS["hover_pin_ratio"],
    )
    parser.add_argument(
        "--hover-warmup-decode-tokens", type=int,
        default=PAPER_BASE_DEFAULTS["hover_warmup_decode_tokens"],
    )
    parser.add_argument(
        "--hover-top-n-prefill", type=int,
        default=PAPER_BASE_DEFAULTS["hover_top_n_prefill"],
    )
    parser.add_argument(
        "--hover-c2-budget-ms", type=float,
        default=PAPER_BASE_DEFAULTS["hover_c2_budget_ms"],
    )
    parser.add_argument(
        "--hover-chunk-size", type=int,
        default=PAPER_BASE_DEFAULTS["hover_chunk_size"],
        help="HoVer C3 base prefix-block size; not the global batch token budget",
    )
    parser.add_argument(
        "--hover-deadline-guard-s", type=float,
        default=PAPER_BASE_DEFAULTS["hover_deadline_guard_s"],
    )
    parser.add_argument(
        "--hover-exact-h2d-overlap",
        dest="hover_exact_h2d_overlap",
        action=argparse.BooleanOptionalAction,
        default=PAPER_BASE_DEFAULTS["hover_exact_h2d_overlap"],
        help=(
            "After exact top-k routing, overlap only the current layer's required "
            "expert H2D copies with shared-expert compute when available, or "
            "resident ready-route compute otherwise; enabling this mode "
            "automatically disables CUDA Graph capture"
        ),
    )
    parser.add_argument(
        "--hover-route-split-min-ready-routes",
        type=int,
        default=PAPER_BASE_DEFAULTS["hover_route_split_min_ready_routes"],
        help="Minimum immediately-readable routes required for generic overlap",
    )
    parser.add_argument(
        "--hover-route-split-max-routes",
        type=int,
        default=PAPER_BASE_DEFAULTS["hover_route_split_max_routes"],
        help="Maximum routes expanded by generic overlap before safe fallback",
    )
    parser.add_argument(
        "--hover-modality-alpha", type=float,
        default=PAPER_BASE_DEFAULTS["hover_modality_alpha"],
                        help="Soft modality boundary threshold alpha in Paper Eq. (1), in (0, 1]")
    parser.add_argument(
        "--hover-jmax", type=int, default=PAPER_BASE_DEFAULTS["hover_jmax"],
        help=">0 caps one request at Jmax consecutive base chunks per batch; <=0 allows multiple chunks up to remaining global budget",
    )
    parser.add_argument(
        "--hover-prefill-reserve-max", type=int,
        default=PAPER_BASE_DEFAULTS["hover_prefill_reserve_max"],
                        help="Maximum dynamic sequence-slot reserve for prefill; 0 disables it")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()


    config = APIServerConfig(
        model=args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        log_level=args.log_level,
        host=args.host,
        port=args.port,
        nccl_port=args.nccl_port,
        num_stages=args.num_stages,
        vertical_stage_policy=getattr(
            args,
            "vertical_stage_policy",
            PAPER_BASE_DEFAULTS["vertical_stage_policy"],
        ),
        vertical_group_tokens=getattr(
            args,
            "vertical_group_tokens",
            PAPER_BASE_DEFAULTS["vertical_group_tokens"],
        ),
        trust_remote_code=getattr(args, "trust_remote_code", False),
        offload_strategy=getattr(
            args, "offload_strategy", PAPER_BASE_DEFAULTS["offload_strategy"]
        ),
        gpu_expert_cache_slots=getattr(
            args,
            "gpu_expert_cache_slots",
            PAPER_BASE_DEFAULTS["gpu_expert_cache_slots"],
        ),
        offload_host_pinned=getattr(args, "offload_host_pinned", True),
        strict_logical_expert_capacity=getattr(
            args, "strict_logical_expert_capacity", False
        ),
        ttft_slo_ms=getattr(
            args, "ttft_slo_ms", PAPER_BASE_DEFAULTS["ttft_slo_ms"]
        ),
        tbt_slo_ms=getattr(
            args, "tbt_slo_ms", PAPER_BASE_DEFAULTS["tbt_slo_ms"]
        ),
        ttl_max=getattr(args, "ttl_max", PAPER_BASE_DEFAULTS["ttl_max"]),
        theta=getattr(args, "theta", PAPER_BASE_DEFAULTS["theta"]),
        expert_transition_decay=getattr(
            args,
            "expert_transition_decay",
            PAPER_BASE_DEFAULTS["expert_transition_decay"],
        ),
        # HoVer
        hover_kh=getattr(args, "hover_kh", PAPER_BASE_DEFAULTS["hover_kh"]),
        hover_pin_ratio=getattr(
            args, "hover_pin_ratio", PAPER_BASE_DEFAULTS["hover_pin_ratio"]
        ),
        hover_warmup_decode_tokens=getattr(
            args,
            "hover_warmup_decode_tokens",
            PAPER_BASE_DEFAULTS["hover_warmup_decode_tokens"],
        ),
        hover_top_n_prefill=getattr(
            args,
            "hover_top_n_prefill",
            PAPER_BASE_DEFAULTS["hover_top_n_prefill"],
        ),
        hover_c2_budget_ms=getattr(
            args,
            "hover_c2_budget_ms",
            PAPER_BASE_DEFAULTS["hover_c2_budget_ms"],
        ),
        hover_chunk_size=getattr(
            args, "hover_chunk_size", PAPER_BASE_DEFAULTS["hover_chunk_size"]
        ),
        hover_deadline_guard_s=getattr(
            args,
            "hover_deadline_guard_s",
            PAPER_BASE_DEFAULTS["hover_deadline_guard_s"],
        ),
        hover_exact_h2d_overlap=getattr(
            args,
            "hover_exact_h2d_overlap",
            PAPER_BASE_DEFAULTS["hover_exact_h2d_overlap"],
        ),
        hover_route_split_min_ready_routes=getattr(
            args,
            "hover_route_split_min_ready_routes",
            PAPER_BASE_DEFAULTS["hover_route_split_min_ready_routes"],
        ),
        hover_route_split_max_routes=getattr(
            args,
            "hover_route_split_max_routes",
            PAPER_BASE_DEFAULTS["hover_route_split_max_routes"],
        ),
        hover_modality_alpha=getattr(
            args,
            "hover_modality_alpha",
            PAPER_BASE_DEFAULTS["hover_modality_alpha"],
        ),
        hover_jmax=getattr(
            args, "hover_jmax", PAPER_BASE_DEFAULTS["hover_jmax"]
        ),
        hover_prefill_reserve_max=getattr(
            args,
            "hover_prefill_reserve_max",
            PAPER_BASE_DEFAULTS["hover_prefill_reserve_max"],
        ),
    )

    engine = AsyncLLMEngine(config)

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        workers=1,
    )
