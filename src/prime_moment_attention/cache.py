"""
PRIME Constant O(1) Cache Container for Hugging Face Transformers
==================================================================

Replaces standard DynamicCache (which stores all past keys and values)
with a bounded recurrent moment container storing only:
    (S0, S1, S2, K0, K1, K2) per layer.
Memory footprint is strictly independent of sequence length L.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers.cache_utils import Cache

class PrimeMomentCache(Cache):
    def __init__(self):
        super().__init__()
        self.prime_states: Dict[int, Tuple[torch.Tensor, ...]] = {}
        self._seen_tokens = 0

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_max_length(self) -> Optional[int]:
        return None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return current states without storing history
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return key_states, value_states

    def get_state(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, ...]]:
        return self.prime_states.get(layer_idx, None)

    def set_state(self, layer_idx: int, state: Tuple[torch.Tensor, ...]):
        self.prime_states[layer_idx] = state

    def reset(self):
        self.prime_states.clear()
        self._seen_tokens = 0
