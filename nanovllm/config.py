from __future__ import annotations

import os

from dataclasses import dataclass

import torch
from transformers import AutoConfig


# Default server profile used by the paper's DS-VL2-S/VisionArena base and
# multi-rate experiments.  Model- and workload-specific evaluation profiles
# override the corresponding fields documented in README.md.
PAPER_BASE_DEFAULTS = {
    "max_num_batched_tokens": 4096,
    "max_num_seqs": 40,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.85,
    "tensor_parallel_size": 2,
    "num_stages": 4,
    "vertical_stage_policy": "threshold_v1",
    "vertical_group_tokens": 512,
    "offload_strategy": "lru",
    "gpu_expert_cache_slots": 32,
    "ttft_slo_ms": 10000.0,
    "tbt_slo_ms": 1000.0,
    "ttl_max": 5,
    "theta": 5,
    "expert_transition_decay": 0.95,
    "hover_kh": 4,
    "hover_pin_ratio": 0.875,
    "hover_warmup_decode_tokens": 200,
    "hover_top_n_prefill": 16,
    "hover_c2_budget_ms": 0.0,
    "hover_c2_ms_per_expert": 0.5,
    "hover_chunk_size": 512,
    "hover_deadline_guard_s": 1.0,
    "hover_exact_h2d_overlap": True,
    "hover_route_split_min_ready_routes": 8,
    "hover_route_split_max_routes": 8192,
    "hover_prefill_reserve_max": 0,
    "hover_modality_alpha": 0.5,
    "hover_jmax": 0,
}


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = PAPER_BASE_DEFAULTS["max_num_batched_tokens"]
    max_num_seqs: int = PAPER_BASE_DEFAULTS["max_num_seqs"]
    max_model_len: int = PAPER_BASE_DEFAULTS["max_model_len"]
    gpu_memory_utilization: float = PAPER_BASE_DEFAULTS["gpu_memory_utilization"]
    tensor_parallel_size: int = PAPER_BASE_DEFAULTS["tensor_parallel_size"]
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    nccl_port: int = 2333
    kvcache_block_size: int = 16
    num_kvcache_blocks: int = -1
    num_stages: int = PAPER_BASE_DEFAULTS["num_stages"]
    # Vertical Scheduler stage policy. ``ceil_group_v1`` uses
    # ceil(prefill_tokens / vertical_group_tokens), capped by ``num_stages``.
    vertical_stage_policy: str = PAPER_BASE_DEFAULTS["vertical_stage_policy"]
    vertical_group_tokens: int = PAPER_BASE_DEFAULTS["vertical_group_tokens"]
    rpc_base_path: str = "/tmp"


    
    trust_remote_code: bool = False              

    
    offload_strategy: str = PAPER_BASE_DEFAULTS["offload_strategy"]
    gpu_expert_cache_slots: int = PAPER_BASE_DEFAULTS["gpu_expert_cache_slots"]
    offload_host_pinned: bool = True             
    strict_logical_expert_capacity: bool = False  # True = logical cache executes routed experts in <=capacity waves

    
    ttft_slo_ms: float = PAPER_BASE_DEFAULTS["ttft_slo_ms"]
    tbt_slo_ms: float = PAPER_BASE_DEFAULTS["tbt_slo_ms"]

    
    
    ttl_max: int = PAPER_BASE_DEFAULTS["ttl_max"]
    theta: int = PAPER_BASE_DEFAULTS["theta"]
    expert_transition_decay: float = PAPER_BASE_DEFAULTS["expert_transition_decay"]
    hover_kh: int = PAPER_BASE_DEFAULTS["hover_kh"]
    hover_pin_ratio: float = PAPER_BASE_DEFAULTS["hover_pin_ratio"]
    hover_warmup_decode_tokens: int = PAPER_BASE_DEFAULTS["hover_warmup_decode_tokens"]
    hover_top_n_prefill: int = PAPER_BASE_DEFAULTS["hover_top_n_prefill"]
    hover_c2_budget_ms: float = PAPER_BASE_DEFAULTS["hover_c2_budget_ms"]
    hover_c2_ms_per_expert: float = PAPER_BASE_DEFAULTS["hover_c2_ms_per_expert"]
    hover_chunk_size: int = PAPER_BASE_DEFAULTS["hover_chunk_size"]
    hover_deadline_guard_s: float = PAPER_BASE_DEFAULTS["hover_deadline_guard_s"]
    hover_exact_h2d_overlap: bool = PAPER_BASE_DEFAULTS["hover_exact_h2d_overlap"]
    hover_route_split_min_ready_routes: int = PAPER_BASE_DEFAULTS[
        "hover_route_split_min_ready_routes"
    ]
    hover_route_split_max_routes: int = PAPER_BASE_DEFAULTS[
        "hover_route_split_max_routes"
    ]
    hover_prefill_reserve_max: int = PAPER_BASE_DEFAULTS[
        "hover_prefill_reserve_max"
    ]
    hover_modality_alpha: float = PAPER_BASE_DEFAULTS["hover_modality_alpha"]
    hover_jmax: int = PAPER_BASE_DEFAULTS["hover_jmax"]

    def __post_init__(self):
        # assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 16 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if (
            isinstance(self.vertical_group_tokens, bool)
            or not isinstance(self.vertical_group_tokens, int)
            or self.vertical_group_tokens <= 0
        ):
            raise ValueError(
                "vertical_group_tokens must be a positive integer"
            )
        if self.vertical_stage_policy not in {
            "threshold_v1",
            "ceil_group_v1",
        }:
            raise ValueError(
                "vertical_stage_policy must be threshold_v1 or "
                "ceil_group_v1"
            )
        if self.hover_chunk_size <= 0:
            raise ValueError("hover_chunk_size must be positive")
        if self.strict_logical_expert_capacity:
            if self.offload_strategy == "none":
                raise ValueError(
                    "strict_logical_expert_capacity requires expert offload/cache"
                )
            if self.gpu_expert_cache_slots <= 0:
                raise ValueError(
                    "strict_logical_expert_capacity requires "
                    "gpu_expert_cache_slots > 0"
                )
            if self.hover_exact_h2d_overlap:
                raise ValueError(
                    "strict logical expert waves do not support exact H2D overlap"
                )
            if not self.enforce_eager:
                self.enforce_eager = True
                print(
                    "[Config] strict logical expert capacity forces eager "
                    "execution (route waves are input-dependent)",
                    flush=True,
                )
        if self.hover_exact_h2d_overlap:
            # Exact cache admission depends on the current layer's dynamic
            # top-k IDs and opens CPU-managed H2D event transactions.  Those
            # operations cannot be recorded or replayed by a CUDA Graph.
            # Keep the public flag self-contained: callers that enable exact
            # overlap automatically get the only correct execution mode.
            if not self.enforce_eager:
                self.enforce_eager = True
                print(
                    "[Config] HoVer exact H2D overlap forces eager execution "
                    "(dynamic route/cache transactions are not CUDA-Graph safe)",
                    flush=True,
                )
            if not self.offload_host_pinned:
                raise ValueError(
                    "route-exact H2D overlap requires pinned host expert weights"
                )
            if self.offload_strategy == "none":
                raise ValueError(
                    "route-exact H2D overlap requires expert offload/cache to be enabled"
                )
            if self.hover_route_split_min_ready_routes < 1:
                raise ValueError(
                    "hover_route_split_min_ready_routes must be positive"
                )
            if self.hover_route_split_max_routes < 1:
                raise ValueError(
                    "hover_route_split_max_routes must be positive"
                )
        if self.trust_remote_code:
            import sys
            from pathlib import Path

            source = os.environ.get("DEEPSEEK_VL2_PATH")
            if source:
                path = Path(source).expanduser().resolve()
                if not path.is_dir():
                    raise ValueError(
                        f"DEEPSEEK_VL2_PATH is not a directory: {path}"
                    )
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            try:
                import deepseek_vl2.models  # noqa: F401
            except ImportError:
                pass

        self.hf_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )
        
        
        if not hasattr(self.hf_config, "max_position_embeddings"):
            lang_cfg = getattr(self.hf_config, "language_config", None)
            if lang_cfg is None:
                lang_cfg = getattr(self.hf_config, "text_config", None)
            if lang_cfg is not None:
                
                max_pos = lang_cfg.get("max_position_embeddings") if isinstance(lang_cfg, dict) \
                          else getattr(lang_cfg, "max_position_embeddings", None)
                if max_pos is not None:
                    
                    self.hf_config.max_position_embeddings = max_pos
        
        max_pos_emb = getattr(self.hf_config, "max_position_embeddings", self.max_model_len)
        if not getattr(self.hf_config, "torch_dtype", None):
            self.hf_config.torch_dtype = torch.bfloat16
        self.max_model_len = min(self.max_model_len, max_pos_emb)
        # assert self.max_num_batched_tokens >= self.max_model_len
