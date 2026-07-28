from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    len_prefill: int = 0
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    max_seqlen_k_dec: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    prefill_block_tables: torch.Tensor | None = None
    decode_block_tables: torch.Tensor | None = None
    prefill_compute_layers: list[int] | None = None

    
    images_seq_mask: torch.Tensor | None = None      # [num_tokens] bool, True=image token
    token_modalities: torch.Tensor | None = None     # [num_tokens] int32, 0=text, 1=image, 2=video, 3=audio
    image_slot_ordinals: list[int] | None = None     # absolute per-request slot; -1=text/continuation
    multimodal_payloads: dict | None = None          # seq_id -> RequestVisualPayload
    multimodal_stage_by_seqid: dict | None = None

    
    token_phase: list | None = None                   
    token_seqid: list | None = None                   # [num_tokens] rid

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
        is_prefill=False,
        len_prefill=0,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=0,
        max_seqlen_k=0,
        max_seqlen_k_dec=0,
        slot_mapping=None,
        context_lens=None,
        prefill_block_tables=None,
        decode_block_tables=None,
        prefill_compute_layers=None,
        # MULTIMODAL
        images_seq_mask=None,
        token_modalities=None,
        image_slot_ordinals=None,
        multimodal_payloads=None,
        multimodal_stage_by_seqid=None,
        token_phase=None,
        token_seqid=None,
        ):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        len_prefill=len_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        max_seqlen_k_dec=max_seqlen_k_dec,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        prefill_block_tables=prefill_block_tables,
        decode_block_tables=decode_block_tables,
        prefill_compute_layers=prefill_compute_layers,
        images_seq_mask=images_seq_mask,
        token_modalities=token_modalities,
        image_slot_ordinals=image_slot_ordinals,
        multimodal_payloads=multimodal_payloads,
        multimodal_stage_by_seqid=multimodal_stage_by_seqid,
        token_phase=token_phase,
        token_seqid=token_seqid,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
