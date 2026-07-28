
import gc  
import pickle  
import numpy as np  
import torch  
import torch.distributed as dist  
import nvtx  
from multiprocessing.synchronize import Event  
from multiprocessing.shared_memory import SharedMemory  

from nanovllm.config import Config  
from nanovllm.engine.sequence import Sequence, SequenceStatus  



from nanovllm.layers.sampler import Sampler  
from nanovllm.utils.context import set_context, reset_context  
from nanovllm.utils.loader import load_model, setup_offload_for_model  
from nanovllm.utils.utils import disable_gc, setup_file_logger  
from nanovllm.multimodal_runtime import (
    MultimodalContractError,
    RequestVisualPayload,
    SUPPORTED_CONTRACTS,
    image_slot_ordinals,
)



logger = setup_file_logger("model_runner", None)


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        
        self.config = config
        hf_config = config.hf_config  
        self.block_size = config.kvcache_block_size  
        self.enforce_eager = config.enforce_eager  
        self.world_size = config.tensor_parallel_size  
        self.rank = rank  
        self.event = event  

        
        
        
        dist.init_process_group("nccl", f"tcp://localhost:{config.nccl_port}", world_size=self.world_size, rank=rank)

        
        
        torch.cuda.set_device(rank)

        
        
        default_dtype = torch.get_default_dtype()
        
        torch.set_default_dtype(hf_config.torch_dtype if hf_config.torch_dtype else torch.bfloat16)
        
        torch.set_default_device("cuda")

        
        
        
        arch = None
        archs = getattr(hf_config, "architectures", None)
        if archs:
            arch = archs[0]
        if arch is None:
            mt = getattr(hf_config, "model_type", None)
            if mt == "deepseek_vl_v2":
                arch = "DeepseekVLV2ForCausalLM"
            elif mt == "deepseek_v2":
                arch = "DeepseekV2ForCausalLM"
            else:
                
                lang = getattr(hf_config, "language_config", None)
                if lang is not None:
                    if isinstance(lang, dict):
                        a = lang.get("architectures")
                        if a:
                            arch = a[0]
                    else:
                        a = getattr(lang, "architectures", None)
                        if a:
                            arch = a[0]
        if arch is None:
            raise ValueError(f"Cannot determine architecture from hf_config: model_type={getattr(hf_config,'model_type',None)}")
        logger.info(f"Detected model architecture: {arch}")

        if arch == "DeepseekVLV2ForCausalLM":
            
            from nanovllm.models.deepseek_vl2 import DeepseekVLV2ForCausalLM
            self.model = DeepseekVLV2ForCausalLM(hf_config)
        elif arch == "DeepseekV2ForCausalLM":
            
            from nanovllm.models.deepseek_vl2 import DeepseekV2ForCausalLM
            self.model = DeepseekV2ForCausalLM(hf_config)
        elif arch == "KimiVLForConditionalGeneration":
            from nanovllm.models.kimi_vl import KimiVLForConditionalGeneration
            self.model = KimiVLForConditionalGeneration(hf_config)
        else:
            raise ValueError(f"Unsupported model architecture: {arch}")

        
        
        
        load_model(self.model, config.model)

        
        if getattr(config, 'offload_strategy', 'none') != 'none':
            setup_offload_for_model(
                self.model,
                offload_strategy=config.offload_strategy,
                gpu_expert_cache_slots=getattr(config, 'gpu_expert_cache_slots', -1),
                host_pinned=getattr(config, 'offload_host_pinned', True),
            )

        # Route-exact overlap is attached after offload setup so every
        # enabled FusedMoE has a concrete RouteExactExpertTransferOverlap.
        _exact_h2d_overlap = bool(
            getattr(config, 'hover_exact_h2d_overlap', False)
        )
        _strict_logical_capacity = bool(
            getattr(config, 'strict_logical_expert_capacity', False)
        )
        try:
            from nanovllm.layers.fused_moe import (
                FusedMoE as _FusedMoE,
                UnquantizedFusedMoEMethod as _UnquantizedFusedMoEMethod,
            )
            _exact_moe_layers = 0
            _exact_cache_layers = 0
            for _module in self.model.modules():
                if isinstance(_module, _FusedMoE):
                    _module.hover_exact_h2d_overlap = _exact_h2d_overlap
                    _module.strict_logical_expert_capacity = (
                        _strict_logical_capacity
                    )
                    _module.hover_route_split_min_ready_routes = int(
                        getattr(
                            config,
                            'hover_route_split_min_ready_routes',
                            8,
                        )
                    )
                    _module.hover_route_split_max_routes = int(
                        getattr(
                            config,
                            'hover_route_split_max_routes',
                            8192,
                        )
                    )
                    _exact_moe_layers += 1
                    _cache = getattr(_module, 'expert_cache', None)
                    if _cache is not None:
                        _exact_cache_layers += 1
                        _cache.strict_logical_wave_enabled = (
                            _strict_logical_capacity
                        )
                        if _exact_h2d_overlap:
                            _pinned_check = getattr(
                                _cache, '_exact_host_buffers_are_pinned', None
                            )
                            if not callable(_pinned_check) or not _pinned_check():
                                raise RuntimeError(
                                    "route-exact H2D overlap requires successfully pinned "
                                    "host weights on every MoE layer"
                                )
                            if not isinstance(
                                _module.quant_method,
                                _UnquantizedFusedMoEMethod,
                            ):
                                raise RuntimeError(
                                    "adaptive route-exact overlap currently requires "
                                    "unquantized FusedMoE"
                                )
            if _exact_h2d_overlap:
                if _exact_moe_layers == 0 or _exact_cache_layers != _exact_moe_layers:
                    raise RuntimeError(
                        "route-exact H2D overlap requires "
                        "RouteExactExpertTransferOverlap on every MoE layer "
                        f"(moe={_exact_moe_layers}, cache={_exact_cache_layers})"
                    )
                logger.info(
                    "HoVer adaptive route-exact overlap enabled on %d MoE layers "
                    "(route_split_min=%d, route_split_max=%d)",
                    _exact_cache_layers,
                    int(getattr(config, 'hover_route_split_min_ready_routes', 8)),
                    int(getattr(config, 'hover_route_split_max_routes', 8192)),
                )
        except ImportError:
            if _exact_h2d_overlap:
                raise

        
        
        self.history_aware_prediction = None
        self.locality_aware_selection = None
        self._hover_round = 0
        # Decode routing metadata is produced once per MoE layer while the
        # CUDA forward is still being launched.  Keep the predictor/TTL CPU
        # bookkeeping out of that layer-by-layer critical path and apply the
        # records, in their original order, immediately after forward.
        self._defer_decode_records_active = False
        self._pending_decode_records = []
        self._hover_moe_by_layer = {}
        self._hover_moe_layer_ids = []
        # Stable pure-decode windows do not need a full demand refresh on
        # every control epoch.  These counters are updated after ``prepare``
        # has exposed the current batch, so a newly admitted request disables
        # the fast path before its routing observations are recorded.
        self._hover_tail_fastpath = False
        self._hover_tail_stable_rounds = 0
        self._hover_tail_last_rids = ()
        self._hover_tail_last_miss_total = None
        self._hover_tail_activation_count = 0
        self._hover_tail_refresh_skip_count = 0
        try:
            from nanovllm.layers.resident_expert_prefetcher import (
                HistoryAwareExpertDemandPredictor,
                LocalityAwareResidentSelector,
            )
            from nanovllm.layers.fused_moe import FusedMoE
            ne, nl, max_moe_layer_idx = None, 0, -1
            for _n, _m in self.model.named_modules():
                if isinstance(_m, FusedMoE):
                    ne = int(_m.global_num_experts)
                    nl += 1
                    layer_id = int(getattr(_m, '_moe_layer_idx', nl - 1))
                    if layer_id < 0:
                        logger.warning("Ignoring HoVer MoE module with invalid layer id %s", layer_id)
                        continue
                    if layer_id in self._hover_moe_by_layer:
                        logger.warning("Ignoring duplicate HoVer MoE layer id %s", layer_id)
                        continue
                    max_moe_layer_idx = max(max_moe_layer_idx, layer_id)
                    self._hover_moe_by_layer[layer_id] = _m
            self._hover_moe_layer_ids = sorted(self._hover_moe_by_layer)
            if ne and nl > 0:
                predictor_layers = max(nl, max_moe_layer_idx + 1)
                effective_pin_ratio = getattr(
                    config, 'hover_pin_ratio', 0.5
                )
                self.history_aware_prediction = HistoryAwareExpertDemandPredictor(
                    num_layers=predictor_layers, num_experts=ne,
                    theta=getattr(config, 'theta', 5),
                    gamma=getattr(
                        config, 'expert_transition_decay', 0.95
                    ),
                    kh=getattr(config, 'hover_kh', 4),
                    warmup_decode_tokens=getattr(config, 'hover_warmup_decode_tokens', 200),
                )
                self.locality_aware_selection = LocalityAwareResidentSelector(
                    num_layers=predictor_layers, num_experts=ne,
                    capacity=max(1, getattr(config, 'gpu_expert_cache_slots', -1) or 1),
                    predictor=self.history_aware_prediction,
                    theta=getattr(config, 'theta', 5),
                    ttl_max=getattr(config, 'ttl_max', 5),
                    pin_ratio=effective_pin_ratio,
                )
                logger.info(
                    "Resident Expert Prefetcher policy=%s configured_pin_ratio=%s "
                    "effective_pin_ratio=%s",
                    "locality_aware_topk",
                    getattr(config, 'hover_pin_ratio', 0.5),
                    effective_pin_ratio,
                )
                logger.info(
                    f"HoVer predictor+resident initialized: "
                    f"{self.history_aware_prediction.stats()} | "
                    f"{self.locality_aware_selection.stats()}"
                )
        except Exception as e:
            logger.warning(f"HoVer init failed: {e}")

        
        _register_global_runner(self)

        
        
        
        self.sampler = Sampler()

        
        
        
        # key: seq_id, value: (hidden_states, residual, input_ids, positions)
        self.intermediate_outputs = dict()

        
        
        
        self.warmup_model()

        
        
        
        self.allocate_kv_cache()

        
        
        gc.collect()
        
        
        gc.freeze()

        
        
        
        if not self.enforce_eager:
            self.capture_cudagraph()

        
        
        torch.set_default_device("cpu")
        
        torch.set_default_dtype(default_dtype)

        
        
        gc.collect()
        gc.freeze()

        
        
        if self.world_size > 1:
            if rank == 0:
                
                try:
                    
                    SharedMemory('nanovllm', create=False).unlink()
                except FileNotFoundError:
                    
                    print("No existing shared memory to unlink.")
                    pass

                
                
                
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**27)  # 128MB

                
                dist.barrier()
            else:
                
                dist.barrier()
                
                self.shm = SharedMemory(name="nanovllm")
                
                self.loop()

    def exit(self):
        if self.world_size > 1:
            
            self.shm.close()
            
            dist.barrier()
            if self.rank == 0:
                
                self.shm.unlink()

        
        torch.cuda.synchronize()
        
        dist.destroy_process_group()

    def loop(self):
        while True:
            
            method_name, args = self.read_shm()
            
            self.call(method_name, *args)
            
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        
        self.event.wait()

        
        n = int.from_bytes(self.shm.buf[0:4], "little")
        
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        
        data = pickle.dumps([method_name, *args], protocol=pickle.HIGHEST_PROTOCOL)
        n = len(data)
        capacity = len(self.shm.buf) - 4
        if n > capacity:
            raise MultimodalContractError(
                f"TP command payload {n} bytes exceeds shared-memory capacity "
                f"{capacity}; refusing truncated multimodal transport"
            )

        
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        
        self.shm.buf[4:n+4] = data

        
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            
            self.write_shm(method_name, *args)

        
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        
        torch.cuda.empty_cache()
        
        torch.cuda.reset_peak_memory_stats()

        
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_lens = []

        
        
        for _ in range(self.config.max_num_seqs):
            
            seq_len = min(max_num_batched_tokens, max_model_len)
            seq_lens.append(seq_len)
            max_num_batched_tokens -= seq_len
            
            if max_num_batched_tokens <= 0:
                break

        
        
        seqs = [Sequence([0] * seq_len) for seq_len in seq_lens]
        for seq in seqs:
            seq.status = SequenceStatus.PREFILLING  
            seq.num_tokens_to_process = seq.num_prompt_tokens  
        
        self.run(seqs)

        
        
        seqs = [Sequence([0]) for seq_len in seq_lens]
        for seq in seqs:
            seq.status = SequenceStatus.PREFILLING
            seq.num_tokens_to_process = seq.num_prompt_tokens
        self.run(seqs)

        
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config

        
        free, total = torch.cuda.mem_get_info()  
        used = total - free  
        
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        
        
        def _get_lang_attr(cfg, attr, default=None):
            if hasattr(cfg, attr):
                v = getattr(cfg, attr)
                if v is not None:
                    return v
            for nested_name in ("language_config", "text_config"):
                lang = getattr(cfg, nested_name, None)
                if lang is not None:
                    if isinstance(lang, dict):
                        v = lang.get(attr, None)
                    else:
                        v = getattr(lang, attr, None)
                    if v is not None:
                        return v
            return default

        num_attention_heads = _get_lang_attr(hf_config, "num_attention_heads", 16)
        num_kv_heads_raw = _get_lang_attr(hf_config, "num_key_value_heads", num_attention_heads)
        num_kv_heads = num_kv_heads_raw // self.world_size
        num_hidden_layers = _get_lang_attr(hf_config, "num_hidden_layers", 27)
        hidden_size = _get_lang_attr(hf_config, "hidden_size", 2048)

        
        
        qk_nope_head_dim = _get_lang_attr(hf_config, "qk_nope_head_dim", None)
        qk_rope_head_dim = _get_lang_attr(hf_config, "qk_rope_head_dim", None)
        if qk_nope_head_dim is not None and qk_rope_head_dim is not None:
            head_dim = qk_nope_head_dim + qk_rope_head_dim
        else:
            head_dim = _get_lang_attr(hf_config, "head_dim", hidden_size // num_attention_heads)

        torch_dtype = hf_config.torch_dtype if hf_config.torch_dtype else torch.bfloat16

        logger.info(f"KV cache config: num_layers={num_hidden_layers}, "
                    f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, dtype={torch_dtype}")

        
        block_bytes = 2 * num_hidden_layers * self.block_size * num_kv_heads * head_dim * torch_dtype.itemsize

        
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0, "Insufficient GPU memory to allocate the KV cache"

        
        self.kv_cache = torch.zeros(2, num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)

        
        layer_id = 0
        for module in self.model.modules():
            
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                
                module.k_cache = self.kv_cache[0, layer_id]  
                module.v_cache = self.kv_cache[1, layer_id]  
                layer_id += 1
        
        self.num_layers = layer_id

    def prepare_block_tables(self, seqs: list[Sequence]):
        if not seqs:
            return None

        
        max_len = max(len(s.block_table) for s in seqs)
        n = len(seqs)

        
        
        block_tables = torch.full(
            (n, max_len), -1, dtype=torch.int32, pin_memory=True
        )

        
        arr = block_tables.numpy()  # zero-copy view
        for i, seq in enumerate(seqs):
            block_table = seq.block_table
            
            arr[i, :len(block_table)] = block_table

        
        return block_tables.cuda(non_blocking=True)

    
    

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        
        self.model.eval()

        with torch.no_grad():
            
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                embed_layer = self.model.model.embed_tokens
            elif hasattr(self.model, 'embed_tokens'):
                embed_layer = self.model.embed_tokens
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
                embed_layer = self.model.transformer.wte
            else:
                raise NotImplementedError(
                    f"Could not find the embedding layer for model {type(self.model).__name__}."
                    f" Add support for this architecture explicitly."
                )

            
            
            
            if input_ids.dim() == 1:
                
                embeddings = embed_layer(input_ids)
            else:
                
                batch_embeddings = []
                for i in range(input_ids.size(0)):
                    single_ids = input_ids[i]  # shape: [seq_len]
                    single_emb = embed_layer(single_ids)  # shape: [seq_len, embed_dim]
                    batch_embeddings.append(single_emb)
                embeddings = torch.stack(batch_embeddings)  # shape: [batch, seq_len, embed_dim]

        return embeddings
    
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @nvtx.annotate("ModelRunner::prepare")
    def prepare(self, seqs: list[Sequence]):
        
        input_ids = np.array([], dtype=np.int64)  
        positions = np.array([], dtype=np.int64)  
        cu_seqlens_q = [0]  
        cu_seqlens_k = [0]  
        max_seqlen_q = 0  
        max_seqlen_k = 0  
        slot_mapping = []  
        context_lens = []  

        
        prefill_seqs = []
        decode_seqs = []
        
        prefill_compute_layers = None  

        
        all_token_modalities = []
        all_images_seq_mask = []
        all_image_slot_ordinals = []
        multimodal_payloads = {}
        multimodal_stage_by_seqid = {}
        self._visual_round_payload_seqids = set()
        
        all_token_phase = []
        all_token_seqid = []
        inter_hidden_states = []  
        inter_residual = []  
        inter_input_ids = []  
        inter_positions = []  

        
        for seq in seqs:
            if seq.status == SequenceStatus.PREFILLING:
                
                prefill_seqs.append(seq)
                
                seqlen = seq.num_processed_tokens + seq.num_tokens_to_process

                
                input_ids = np.concatenate((input_ids, seq[seq.num_processed_tokens:seqlen]))
                
                positions = np.concatenate((positions, np.arange(seq.num_processed_tokens, seqlen)))

                # Official VL2 request state stays per request.  Only absolute
                # visual-slot ordinals are flattened with token rows.
                s_idx = seq.num_processed_tokens
                e_idx = seq.num_processed_tokens + seq.num_tokens_to_process
                full_mask = getattr(seq, "images_seq_mask", None)
                modalities = getattr(seq, "token_modalities", None)
                if full_mask is None:
                    full_mask = [False] * seq.num_prompt_tokens
                if len(full_mask) != seq.num_prompt_tokens:
                    raise MultimodalContractError(
                        f"mask length mismatch for {seq.seq_id!r}"
                    )
                if modalities is None:
                    modalities = [1 if value else 0 for value in full_mask]
                if len(modalities) != seq.num_prompt_tokens:
                    raise MultimodalContractError(
                        f"modality length mismatch for {seq.seq_id!r}"
                    )
                if any(type(value) is not bool for value in full_mask):
                    raise MultimodalContractError("images_seq_mask must contain bools")
                # KV preemption can move a decoding request back to prefill.
                # Its replay stream then contains the original prompt plus
                # already generated tokens, while multimodal metadata is
                # intentionally prompt-sized.  Generated tokens are text-only,
                # so extend both row metadata arrays with explicit non-visual
                # entries for the replay suffix.
                prompt_mask = list(full_mask)
                prompt_modalities = list(modalities)
                replay_len = len(seq)
                if replay_len < seq.num_prompt_tokens or e_idx > replay_len:
                    raise MultimodalContractError(
                        f"invalid replay span for {seq.seq_id!r}: "
                        f"prompt={seq.num_prompt_tokens}, replay={replay_len}, "
                        f"chunk=[{s_idx},{e_idx})"
                    )
                suffix_len = replay_len - seq.num_prompt_tokens
                full_mask = prompt_mask + [False] * suffix_len
                modalities = prompt_modalities + [0] * suffix_len
                visual_count = sum(prompt_mask)
                if visual_count != int(getattr(seq, "num_visual_tokens", 0)):
                    raise MultimodalContractError(
                        f"visual token count changed for {seq.seq_id!r}"
                    )
                inject = int(seq.stage) <= 0
                ordinals = image_slot_ordinals(full_mask, s_idx, e_idx, inject)
                all_image_slot_ordinals.extend(ordinals)
                all_token_modalities.extend(modalities[s_idx:e_idx])
                all_images_seq_mask.extend(full_mask[s_idx:e_idx])
                if visual_count:
                    contract = getattr(seq, "multimodal_contract", None)
                    if contract not in SUPPORTED_CONTRACTS:
                        raise MultimodalContractError(
                            f"unsupported multimodal Sequence contract {contract!r}"
                        )
                    crop = getattr(seq, "images_spatial_crop", None)
                    images = getattr(seq, "images", None)
                    multimodal_stage_by_seqid[seq.seq_id] = int(seq.stage)
                    if any(value >= 0 for value in ordinals):
                        multimodal_payloads[seq.seq_id] = RequestVisualPayload(
                            images=images,
                            images_spatial_crop=crop,
                            expected_slots=visual_count,
                        )
                        if images is not None:
                            self._visual_round_payload_seqids.add(seq.seq_id)
                
                all_token_phase.extend(['P'] * seq.num_tokens_to_process)
                all_token_seqid.extend([seq.seq_id] * seq.num_tokens_to_process)

                seqlen_q = seq.num_tokens_to_process  
                seqlen_k = seqlen  
                cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
                cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
                max_seqlen_q = max(seqlen_q, max_seqlen_q)
                max_seqlen_k = max(seqlen_k, max_seqlen_k)

                
                if not seq.block_table:
                    continue

                
                
                seq_slot_mapping = []
                start_idx = seq.num_processed_tokens
                end_idx = seq.num_processed_tokens + seq.num_tokens_to_process
                
                for i in range(start_idx // seq.block_size, (end_idx - 1) // seq.block_size + 1):
                    
                    start = seq.block_table[i] * self.block_size
                    
                    if i != seq.num_blocks - 1:
                        
                        end = (seq.block_table[i] + 1) * self.block_size
                    else:
                        
                        end = start + seq.last_block_num_tokens
                    
                    if i * self.block_size < start_idx:
                        
                        start += start_idx - i * self.block_size
                    if (i + 1) * self.block_size > end_idx:
                        
                        end = min(end, end_idx - i * self.block_size + start)
                    
                    seq_slot_mapping.append(np.arange(start, end))
                
                seq_slot_mapping = np.concatenate(seq_slot_mapping)
                slot_mapping.extend(seq_slot_mapping)

                
                
                
                if seq.stage != -1:
                    current_layers = set(
                        np.array_split(
                            np.arange(self.num_layers), seq.num_stages
                        )[seq.stage].tolist()
                    )
                    
                    if prefill_compute_layers is not None:
                        if prefill_compute_layers != current_layers:
                            raise ValueError("All sequences in a HoVer batch must be at the same stage")
                    else:
                        prefill_compute_layers = current_layers

                    
                    i_hidden_states, i_residual, i_input_ids, i_positions = self.intermediate_outputs.get(seq.seq_id, (None, None, None, None))
                    
                    if seq.seq_id in self.intermediate_outputs:
                        del self.intermediate_outputs[seq.seq_id]
                    
                    if i_hidden_states is not None:
                        inter_hidden_states.append(i_hidden_states)
                    if i_residual is not None:
                        inter_residual.append(i_residual)
                    if i_input_ids is not None:
                        inter_input_ids.append(i_input_ids)
                    if i_positions is not None:
                        inter_positions.append(i_positions)

            elif seq.status == SequenceStatus.DECODING:
                
                decode_seqs.append(seq)
                
                all_token_modalities.append(0)
                all_images_seq_mask.append(False)
                all_image_slot_ordinals.append(-1)
                
                all_token_phase.append('D')
                all_token_seqid.append(seq.seq_id)
                
                input_ids = np.concatenate((input_ids, np.array([seq.last_token], dtype=np.int64)))
                
                positions = np.concatenate((positions, np.array([len(seq) - 1], dtype=np.int64)))
                
                context_lens.append(len(seq))

                
                slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
            else:
                raise ValueError(f"Invalid sequence status: {seq.status}")

        
        is_prefill = len(prefill_seqs) > 0
        prefill_block_tables = None

        
        
        
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # partial HoVer prefill
            prefill_block_tables = self.prepare_block_tables(prefill_seqs)

        
        decode_block_tables = self.prepare_block_tables(decode_seqs)
        max_seqlen_k_dec = max(context_lens) if context_lens else 0

        
        len_prefill = int(cu_seqlens_q[-1]) if is_prefill else 0
        
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        
        inter_hidden_states = torch.cat(inter_hidden_states, dim=0) if inter_hidden_states else None
        inter_residual = torch.cat(inter_residual, dim=0) if inter_residual else None
        inter_input_ids = torch.cat(inter_input_ids, dim=0) if inter_input_ids else None
        inter_positions = torch.cat(inter_positions, dim=0) if inter_positions else None

        
        
        len_inter = inter_input_ids.size(0) if inter_input_ids is not None else 0
        input_ids = torch.from_numpy(input_ids[len_inter:]).cuda(non_blocking=True)
        positions = torch.from_numpy(positions[len_inter:]).cuda(non_blocking=True)
        input_ids = torch.cat([inter_input_ids, input_ids], dim=0) if inter_input_ids is not None else input_ids
        positions = torch.cat([inter_positions, positions], dim=0) if inter_positions is not None else positions

        
        if all_token_modalities:
            token_modalities_tensor = torch.tensor(
                all_token_modalities, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
            
            from collections import Counter as _C
            mod_counter = _C(all_token_modalities)
            self._batch_dominant_modality = mod_counter.most_common(1)[0][0]
            self._batch_modality_dist = dict(mod_counter)
        else:
            token_modalities_tensor = None
            self._batch_dominant_modality = 0
            self._batch_modality_dist = {0: 1}

        if all_images_seq_mask:
            images_seq_mask_tensor = torch.tensor(
                all_images_seq_mask, dtype=torch.bool, pin_memory=True
            ).cuda(non_blocking=True)
        else:
            images_seq_mask_tensor = None

        
        set_context(
            is_prefill,
            len_prefill,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            max_seqlen_k_dec,
            slot_mapping,
            context_lens,
            prefill_block_tables,
            decode_block_tables,
            prefill_compute_layers,
            images_seq_mask=images_seq_mask_tensor,
            token_modalities=token_modalities_tensor,
            image_slot_ordinals=all_image_slot_ordinals,
            multimodal_payloads=multimodal_payloads,
            multimodal_stage_by_seqid=multimodal_stage_by_seqid,
            token_phase=all_token_phase if all_token_phase else None,
            token_seqid=all_token_seqid if all_token_seqid else None,
        )
        
        self._active_decode_rids = [s.seq_id for s in decode_seqs]
        self._update_hover_tail_fastpath(
            self._active_decode_rids,
            has_prefill=bool(prefill_seqs),
        )
        return input_ids, positions, (inter_hidden_states, inter_residual)

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @nvtx.annotate("ModelRunner::run_model")
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, intermediate_outputs = None):
        hidden_states, residual = self.model(input_ids, positions, intermediate_outputs)
        return hidden_states, residual

    def _complete_multimodal_forward(self, seqs: list[Sequence]):
        """Audit and commit after every TP rank succeeds."""
        has_pending = bool(
            getattr(self.model, "has_pending_multimodal_forward", lambda: False)()
        )

        # A visual forward is considered successful only when every rank has
        # returned from the language model.  The second barrier ensures every
        # rank-local cache committed before rank 0 marks the payload encoded.
        if has_pending and self.world_size > 1:
            dist.barrier()

        if has_pending:
            self.model.commit_multimodal_forward()

        if has_pending and self.world_size > 1:
            dist.barrier()

        if self.rank == 0:
            for seq in seqs:
                if seq.seq_id not in self._visual_round_payload_seqids:
                    continue
                if seq.images is None or seq.images_spatial_crop is None:
                    raise MultimodalContractError(
                        "visual payload disappeared before TP commit"
                    )
                seq.images = None
                seq.images_spatial_crop = None
                seq.multimodal_payload_encoded = True


    def release_multimodal_cache(self, seq_ids) -> int:
        """Release projected visual state only after requests are finished.

        A decode sequence can be KV-preempted after its first prompt pass.  In
        that case BlockManager resets its processed-token cursor and the full
        prompt, including visual slots, must be replayed.  Prompt completion is
        therefore too early to release these embeddings; EngineCore invokes
        this method after scheduler.postprocess marks the request FINISHED.
        """
        clear_multimodal_cache = getattr(
            self.model, "clear_multimodal_cache", None
        )
        released = 0
        if callable(clear_multimodal_cache):
            for seq_id in seq_ids:
                released += int(clear_multimodal_cache(seq_id))
        if self.world_size > 1:
            dist.barrier()
        return released

    @nvtx.annotate("ModelRunner::run")
    @disable_gc()
    def run(self, seqs: list[Sequence]) -> list[int]:
        
        

        
        

        
        try:
            self.update_resident_expert_prefetcher()
        except Exception as _e:
            logger.warning(
                f"resident expert prefetcher update failed: {_e}"
            )

        
        
        input_ids, positions, intermediate_outputs = self.prepare(seqs)

        
        
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        
        
        
        # The MoE hooks enqueue decode predictor/TTL records during forward.
        # Flush them before any post-forward work and, critically, before the
        # next resident expert prefetcher update. If forward fails, discard the incomplete
        # round so it can never leak into a later round.
        self.begin_decode_records()
        try:
            hidden_states, residual = self.run_model(
                input_ids, positions, intermediate_outputs
            )
        except BaseException:
            self.discard_decode_records()
            raise
        self.flush_decode_records()

        
        with nvtx.annotate("ModelRunner::store_intermediate"):
            start_idx = 0
            for seq in seqs:
                end_idx = start_idx + seq.num_tokens_to_process

                
                
                if seq.stage != -1 and seq.status == SequenceStatus.PREFILLING:
                    
                    i_hidden_states = hidden_states
                    if hidden_states is not None:
                        i_hidden_states = hidden_states[start_idx:end_idx]
                    
                    i_residual = residual
                    if residual is not None:
                        i_residual = residual[start_idx:end_idx]
                    
                    i_input_ids = input_ids[start_idx:end_idx]
                    i_positions = positions[start_idx:end_idx]
                    
                    self.intermediate_outputs[seq.seq_id] = (i_hidden_states, i_residual, i_input_ids, i_positions)

                
                
                if seq.stage >= seq.num_stages - 1:
                    if seq.seq_id in self.intermediate_outputs:
                        del self.intermediate_outputs[seq.seq_id]

                
                
                
                
                start_idx = end_idx

        
        with torch.inference_mode():
            
            logits = self.model.compute_logits(hidden_states)

        
        with nvtx.annotate("ModelRunner::sample"):
            
            token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        

        # Commit visual progress only after logits, sampling and every other
        # post-forward step succeeded.  Any earlier exception leaves both the
        # cache ordinal and Sequence transport intact for an exact retry.
        self._complete_multimodal_forward(seqs)

        
        
        reset_context()
        return token_ids
    
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config

        
        if hasattr(self.model, "capture_cudagraph_layers"):
            
            
            self.model.capture_cudagraph_layers(
                max_num_batched_tokens=config.max_num_batched_tokens,
                schedule_mode="hover",
            )

    # ========================================================================
    
    # ========================================================================
    def update_resident_expert_prefetcher(self):
        if (
            self.history_aware_prediction is None
            or self.locality_aware_selection is None
        ):
            return
        self._hover_round += 1
        t = self._hover_round
        rids = getattr(self, '_active_decode_rids', [])
        theta = self.locality_aware_selection.theta
        stable_tail = bool(self._hover_tail_fastpath)
        if t % theta == 0 and not stable_tail:
            
            self.history_aware_prediction.refresh(
                round_t=t, active_decode_rids=rids
            )
        elif t % theta == 0:
            self._hover_tail_refresh_skip_count += 1
        C = self.locality_aware_selection.step(
            active_decode_rids=rids,
            skip_selection=stable_tail,
        )   
        if C is not None:
            self._apply_locality_aware_selection(C)

    def _hover_cache_miss_total(self) -> int:
        """Read the cumulative miss counters without materializing stats."""
        total = 0
        for module in self._hover_moe_by_layer.values():
            cache = getattr(module, 'expert_cache', None)
            if cache is not None:
                total += int(getattr(cache, "miss_count", 0))
        return total

    def _update_hover_tail_fastpath(self, active_rids, has_prefill: bool):
        """Enter a low-overhead control path after a miss-free stable window.

        A request-set change, a prefill batch, or any new expert-cache miss
        disables the path immediately.  Stability must then be observed for a
        complete control epoch before refresh suppression and 2x routing
        record sampling are enabled again.
        """
        if (
            self.history_aware_prediction is None
            or self.locality_aware_selection is None
        ):
            return
        rids = tuple(sorted(str(rid) for rid in active_rids))
        misses = self._hover_cache_miss_total()
        same_rids = bool(rids) and rids == self._hover_tail_last_rids
        miss_free = (
            self._hover_tail_last_miss_total is not None
            and misses == self._hover_tail_last_miss_total
        )
        if not has_prefill and same_rids and miss_free:
            self._hover_tail_stable_rounds += 1
        else:
            self._hover_tail_stable_rounds = 0
        threshold = max(1, int(self.locality_aware_selection.theta))
        enabled = self._hover_tail_stable_rounds >= threshold
        if enabled and not self._hover_tail_fastpath:
            self._hover_tail_activation_count += 1
        self._hover_tail_fastpath = enabled
        self._hover_tail_last_rids = rids
        self._hover_tail_last_miss_total = misses

    def _apply_locality_aware_selection(self, resident: dict):
        modules = getattr(self, '_hover_moe_by_layer', {})
        for _li, experts in resident.items():
            _mod = modules.get(int(_li))
            if _mod is None:
                continue
            _cache = getattr(_mod, 'expert_cache', None)
            if _cache is not None:
                _cache.set_resident(experts)

    def record_decode(self, layer_idx: int, rid, expert_ids, modality_weights=None):
        if self.history_aware_prediction is None:
            return
        try:
            self.history_aware_prediction.record_decode(
                str(rid), layer_idx, expert_ids,
                round_t=self._hover_round, modality_weights=modality_weights,
            )
            if self.locality_aware_selection is not None:
                self.locality_aware_selection.record_decode(
                    layer_idx, expert_ids
                )
        except Exception:
            pass

    def _decode_record_due(self, round_t: int) -> bool:
        """Return whether this round participates in predictor recording."""
        if getattr(self, "_hover_tail_fastpath", False):
            return False
        return True

    @staticmethod
    def _freeze_decode_record(
        layer_idx: int, rids, rows_experts, union_experts, round_t: int
    ):
        """Snapshot a layer record so caller mutation cannot change history."""
        # Use ordinary lists, matching the predictor's existing public input,
        # and copy only once.  The queue owns these containers after return.
        frozen_rids = list(rids)
        frozen_rows = [list(row) for row in rows_experts]
        if union_experts is None:
            frozen_union = [e for row in frozen_rows for e in row]
        else:
            frozen_union = list(union_experts)
        return (
            int(layer_idx), frozen_rids, frozen_rows, frozen_union, int(round_t)
        )

    def _apply_decode_record(self, record):
        """Apply one frozen layer record with the legacy failure boundary."""
        layer_idx, rids, rows_experts, union_experts, round_t = record
        if self.history_aware_prediction is None or not rids:
            return
        try:
            self.history_aware_prediction.record_decode_batch(
                layer_idx, rids, rows_experts, round_t=round_t
            )
            if self.locality_aware_selection is not None:
                self.locality_aware_selection.record_decode(
                    layer_idx, union_experts
                )
        except Exception:
            # Match record_decode_batch's historical fail-open behavior and
            # isolate a bad layer without dropping later layer records.
            pass

    def record_decode_batch(self, layer_idx: int, rids, rows_experts, union_experts=None):
        if self.history_aware_prediction is None or not rids:
            return
        round_t = int(self._hover_round)
        if not self._decode_record_due(round_t):
            return
        try:
            self.history_aware_prediction.record_decode_batch(
                layer_idx, rids, rows_experts, round_t=round_t
            )
            if self.locality_aware_selection is not None:
                self.locality_aware_selection.record_decode(
                    layer_idx, union_experts if union_experts is not None
                    else [e for row in rows_experts for e in row]
                )
        except Exception:
            pass

    def begin_decode_records(self):
        """Open one forward-scoped decode-record transaction."""
        # A stale transaction can only be an interrupted/failed forward.  It
        # is unsafe to attribute those partial records to the new round.
        self._pending_decode_records = []
        self._defer_decode_records_active = True

    def defer_decode_batch(
        self, layer_idx: int, rids, rows_experts, union_experts=None
    ):
        """Queue one layer's decode record for the post-forward flush.

        Calls outside ModelRunner.run() retain the old immediate behavior;
        this matters for warmup, direct model calls and future graph tooling.
        """
        if self.history_aware_prediction is None or not rids:
            return
        if not getattr(self, '_defer_decode_records_active', False):
            self.record_decode_batch(
                layer_idx, rids, rows_experts, union_experts=union_experts
            )
            return
        round_t = int(self._hover_round)
        if not self._decode_record_due(round_t):
            return
        record = self._freeze_decode_record(
            layer_idx, rids, rows_experts, union_experts, round_t
        )
        self._pending_decode_records.append(record)

    def flush_decode_records(self) -> int:
        """Apply a completed forward with one lock per state manager."""
        pending = getattr(self, '_pending_decode_records', [])
        # Close and detach first so failures or re-entrant hooks cannot replay
        # an already-consumed record on the next forward.
        self._pending_decode_records = []
        self._defer_decode_records_active = False
        if not pending:
            return 0

        predictor_bulk = getattr(
            self.history_aware_prediction, 'record_decode_bulk', None
        ) if self.history_aware_prediction is not None else None
        if not callable(predictor_bulk):
            # Rolling-deployment fallback for the old predictor/resident pair.
            for record in pending:
                self._apply_decode_record(record)
            return len(pending)

        predictor_records = [
            (layer_idx, rids, rows_experts, round_t)
            for (layer_idx, rids, rows_experts, _union_experts, round_t)
            in pending
        ]
        try:
            applied = predictor_bulk(predictor_records)
        except Exception:
            # Do not retry: a third-party bulk implementation may have applied
            # a prefix before raising, and replay would double-count it.
            return len(pending)
        if applied is None:
            applied = [True] * len(pending)
        else:
            try:
                applied = list(applied)
            except Exception:
                return len(pending)
        if len(applied) < len(pending):
            applied.extend([False] * (len(pending) - len(applied)))

        if self.locality_aware_selection is not None:
            resident_records = [
                (record[0], record[3])
                for record, ok in zip(pending, applied)
                if ok
            ]
            resident_bulk = getattr(
                self.locality_aware_selection,
                'record_decode_bulk',
                None,
            )
            if callable(resident_bulk):
                try:
                    resident_bulk(resident_records)
                except Exception:
                    pass
            else:
                for layer_idx, expert_ids in resident_records:
                    try:
                        self.locality_aware_selection.record_decode(
                            layer_idx, expert_ids
                        )
                    except Exception:
                        pass
        return len(pending)

    def discard_decode_records(self) -> int:
        """Drop records from an incomplete forward and close the transaction."""
        count = len(getattr(self, '_pending_decode_records', []))
        self._pending_decode_records = []
        self._defer_decode_records_active = False
        return count


# ============================================================================

# ============================================================================
_GLOBAL_MODEL_RUNNER = None

def _register_global_runner(runner):
    global _GLOBAL_MODEL_RUNNER
    _GLOBAL_MODEL_RUNNER = runner
