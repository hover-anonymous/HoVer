
from __future__ import annotations

from copy import copy
from enum import Enum, auto
from itertools import count

import numpy as np

from nanovllm.sampling_params import SamplingParams
from nanovllm.multimodal_runtime import CONTRACT_VERSION


class SequenceStatus(Enum):
    WAITING = auto()      
    PREFILLING = auto()   
    DECODING = auto()     
    FINISHED = auto()     
    ERROR = auto()        


class Sequence:
    block_size = 16  
    counter = count()  

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams(), seq_id: str = None):
        if seq_id is None:
            seq_id = next(Sequence.counter)
        self.seq_id = seq_id  

        
        self.status = SequenceStatus.WAITING  

        
        self.token_ids = np.array(copy(token_ids), dtype=np.int64)  
        self.last_token = copy(token_ids[-1])  
        self.num_tokens = len(self.token_ids)  
        self.num_prompt_tokens = len(self.token_ids)  

        
        self.num_processed_tokens = 0  
        self.num_tokens_to_process = 0  

        
        self.block_table = []  

        
        self.temperature = sampling_params.temperature  
        self.max_tokens = sampling_params.max_tokens  
        self.ignore_eos = sampling_params.ignore_eos  

        
        self.stage = -1  
        self.num_stages = -1  

        
        self.intermediate_outputs = None  

        
        self.error_message = ""  
        
        
        


        # ``pixel_values`` is retained only to unpickle the historical 4-field
        # state.  It is never reinterpreted as VL2 images because that state
        # lacks the required spatial crop plan.
        self.pixel_values = None
        self.images = None                 # [tiles, 3, H, W], CPU bfloat16
        self.images_spatial_crop = None    # [n_images, 2], CPU long
        self.token_modalities = None       
        self.images_seq_mask = None        
        self.num_visual_tokens = 0         
        self.multimodal_payload_encoded = False
        self.multimodal_contract = CONTRACT_VERSION

        
        import time as _time
        self.arrival_time: float = _time.time()
        self.ttft_deadline: float | None = None
        self.tbt_slo_s: float = 0.080
        self.urgency_score: float = 0.0
        self.first_token_time: float | None = None
        self.last_token_time: float | None = None

    def update_urgency(self, current_time: float = None):
        import time as _time
        if current_time is None:
            current_time = _time.time()

        eps = 1e-3
        
        token_time = self.last_token_time or self.first_token_time
        if token_time is not None:
            S_r = max(self.tbt_slo_s, eps)
            elapsed = max(current_time - token_time, 0.0)
            s_r = max(S_r - elapsed, 0.0)
            if s_r > 0.0:
                self.urgency_score = S_r / max(s_r, eps)
            else:
                # Keep overdue decode requests ordered by lateness instead of saturating.
                self.urgency_score = min(100.0, 1.0 + (elapsed - S_r) / S_r)
            return self.urgency_score

        
        if self.ttft_deadline is None:
            self.urgency_score = 0.0
            return self.urgency_score
        S_r = max(self.ttft_deadline - (self.arrival_time or current_time), eps)
        s_r = max(self.ttft_deadline - current_time, 0.0)
        self.urgency_score = S_r / (s_r + eps)
        return self.urgency_score

    def slo_priority_score(self, current_time: float = None) -> float:
        import time as _time
        if current_time is None:
            current_time = _time.time()
        if self.ttft_deadline is not None:
            slack = self.ttft_deadline - current_time
            deadline_score = -slack
        else:
            deadline_score = 0.0
        wait = current_time - (self.arrival_time or current_time)
        progress = (self.num_processed_tokens / max(self.num_prompt_tokens, 1)) if self.num_prompt_tokens > 0 else 0.0
        return 10.0 * deadline_score + 1.0 * wait + 5.0 * progress

    def set_error(self, message: str):
        self.status = SequenceStatus.ERROR
        self.error_message = message
        self.last_token = -1  

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_processed_blocks(self):
        return self.num_processed_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids = np.append(self.token_ids, [token_id])
        self.last_token = token_id
        self.num_tokens += 1

    def _restore_multimodal_state(self, state):
        """Restore new state and accept old four-field pickles fail-closed."""
        self.pixel_values = None
        self.images = None
        self.images_spatial_crop = None
        self.token_modalities = None
        self.images_seq_mask = None
        self.num_visual_tokens = 0
        self.multimodal_payload_encoded = False
        self.multimodal_contract = CONTRACT_VERSION
        if state is None:
            return
        if isinstance(state, tuple) and len(state) == 7:
            (
                contract,
                self.images,
                self.images_spatial_crop,
                self.token_modalities,
                self.images_seq_mask,
                self.num_visual_tokens,
                self.multimodal_payload_encoded,
            ) = state
            if contract != CONTRACT_VERSION:
                raise ValueError(f"unsupported multimodal contract {contract!r}")
            self.multimodal_contract = contract
            return
        if isinstance(state, tuple) and len(state) == 4:
            # Deserialization remains compatible, but this legacy state is not
            # guessed into official tiles: it has no spatial crop plan.
            (
                self.pixel_values,
                self.token_modalities,
                self.images_seq_mask,
                self.num_visual_tokens,
            ) = state
            self.multimodal_contract = "legacy-pixel-values-v0"
            return
        raise ValueError("unsupported Sequence multimodal state")

    def __getstate__(self):
        
        base_state = (self.status, self.stage, self.num_stages, self.seq_id, self.num_tokens,
                      self.num_prompt_tokens, self.num_processed_tokens, self.num_tokens_to_process,
                      self.temperature, self.ignore_eos, self.max_tokens, self.block_table, self.error_message)

        
        
        scheduling_state = ()

        multimodal_state = (
            CONTRACT_VERSION,
            self.images,
            self.images_spatial_crop,
            self.token_modalities,
            self.images_seq_mask,
            self.num_visual_tokens,
            self.multimodal_payload_encoded,
        )

        
        slo_state = (self.arrival_time, self.ttft_deadline,
                     self.tbt_slo_s, self.urgency_score, self.first_token_time,
                     self.last_token_time)

        
        token_data = self.token_ids if self.status in [SequenceStatus.WAITING, SequenceStatus.PREFILLING] else self.last_token

        return (base_state, scheduling_state, token_data, multimodal_state, slo_state)

    def __setstate__(self, state):
        
        
        
        
        is_slo_format = (
            isinstance(state, tuple) and len(state) == 5 and isinstance(state[0], tuple)
        )
        is_multimodal_format = (
            isinstance(state, tuple) and len(state) == 4 and isinstance(state[0], tuple)
        )
        is_legacy_scheduling_format = (
            isinstance(state, tuple) and len(state) == 3 and isinstance(state[0], tuple)
        )

        if is_slo_format:
            base_state, scheduling_state, token_data, multimodal_state, slo_state = state
            self._restore_multimodal_state(multimodal_state)
            if len(slo_state) >= 6:
                (self.arrival_time, self.ttft_deadline, self.tbt_slo_s,
                 self.urgency_score, self.first_token_time,
                 self.last_token_time) = slo_state
            else:
                (self.arrival_time, self.ttft_deadline, self.tbt_slo_s,
                 self.urgency_score, self.first_token_time) = slo_state
                self.last_token_time = self.first_token_time
        elif is_multimodal_format:
            base_state, scheduling_state, token_data, multimodal_state = state
            self._restore_multimodal_state(multimodal_state)
            self.arrival_time = 0.0
            self.ttft_deadline = None
            self.tbt_slo_s = 0.080
            self.urgency_score = 0.0
            self.first_token_time = None
            self.last_token_time = None
        elif is_legacy_scheduling_format:
            base_state, scheduling_state, token_data = state
            self._restore_multimodal_state(None)
            self.arrival_time = 0.0
            self.ttft_deadline = None
            self.tbt_slo_s = 0.080
            self.urgency_score = 0.0
            self.first_token_time = None
            self.last_token_time = None

        if is_slo_format or is_multimodal_format or is_legacy_scheduling_format:
            
            (self.status, self.stage, self.num_stages, self.seq_id, self.num_tokens,
             self.num_prompt_tokens, self.num_processed_tokens, self.num_tokens_to_process,
             self.temperature, self.ignore_eos, self.max_tokens, self.block_table, self.error_message) = base_state

            
            if self.status in [SequenceStatus.WAITING, SequenceStatus.PREFILLING]:
                self.token_ids = token_data
                self.last_token = -1
            elif self.status in [SequenceStatus.ERROR]:
                self.last_token = -1
            else:
                self.last_token = token_data
        else:
            
            self._restore_multimodal_state(None)
            self.arrival_time = 0.0
            self.ttft_deadline = None
            self.tbt_slo_s = 0.080
            self.urgency_score = 0.0
            self.first_token_time = None
            self.last_token_time = None

            self.status, self.stage, self.num_stages, self.seq_id, self.num_tokens, self.num_prompt_tokens, self.num_processed_tokens, self.num_tokens_to_process, self.temperature, self.ignore_eos, self.max_tokens, self.block_table, self.error_message = state[:-1]

            if self.status in [SequenceStatus.WAITING, SequenceStatus.PREFILLING]:
                self.token_ids = state[-1]
                self.last_token = -1
            elif self.status in [SequenceStatus.ERROR]:
                self.last_token = -1
            else:
                self.last_token = state[-1]

            

    def __repr__(self):
        return (f"Sequence(seq_id={self.seq_id}, status={self.status}, num_tokens={self.num_tokens}, "
                f"num_prompt_tokens={self.num_prompt_tokens}, num_processed_tokens={self.num_processed_tokens}, "
                f"num_tokens_to_process={self.num_tokens_to_process}, block_table={self.block_table}, "
                f"num_completion_tokens={self.num_completion_tokens}, last_token={self.last_token})")

    def __str__(self):
        return self.__repr__()

