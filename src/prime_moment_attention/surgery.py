"""
Model Surgery: In-Place Attention Layer Transplantation
=======================================================

Provides zero-shot and hybrid attention transplanting for pretrained Hugging Face models.
Replaces quadratic Softmax attention with PRIME Moment Attention while preserving
original weights, biases, and rotary positional embeddings (RoPE).
"""

from typing import Set, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

class PrimeTransplantedAttention(nn.Module):
    def __init__(self, original_attn: nn.Module, layer_idx: int, decay: float = 0.9995):
        super().__init__()
        self.config = original_attn.config
        self.layer_idx = layer_idx
        self.hidden_size = original_attn.config.hidden_size
        self.num_heads = original_attn.config.num_attention_heads
        self.head_dim = getattr(original_attn, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = original_attn.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = getattr(original_attn, "scaling", 1.0 / (self.head_dim ** 0.5))
        self.decay = decay

        # Borrow exact projection weights and biases
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor = None,
        past_key_values: Cache = None,
        **kwargs
    ) -> Tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        B, L = input_shape
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # GQA repeat for Key and Value
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Scale queries
        query_states = query_states * self.scaling

        # Cast to float32 for recurrence
        q_f32 = query_states.to(torch.float32)
        kt_f32 = key_states.to(torch.float32)
        vt_f32 = value_states.to(torch.float32)

        state = None
        if past_key_values is not None:
            if not hasattr(past_key_values, 'prime_states'):
                past_key_values.prime_states = {}
            state = past_key_values.prime_states.get(self.layer_idx, None)

        if state is not None:
            S0, S1, S2, K0, K1, K2 = state
        else:
            S0 = torch.zeros(B, self.num_heads, self.head_dim, device=query_states.device, dtype=torch.float32)
            S1 = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=query_states.device, dtype=torch.float32)
            S2 = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=query_states.device, dtype=torch.float32)
            K0 = torch.zeros(B, self.num_heads, 1, device=query_states.device, dtype=torch.float32)
            K1 = torch.zeros(B, self.num_heads, self.head_dim, device=query_states.device, dtype=torch.float32)
            K2 = torch.zeros(B, self.num_heads, self.head_dim, device=query_states.device, dtype=torch.float32)

        # Fast Autoregressive Token Generation (L == 1)
        if L == 1:
            qt = q_f32[:, :, 0]
            kt = kt_f32[:, :, 0]
            vt = vt_f32[:, :, 0]

            S0 = self.decay * S0 + vt
            S1 = self.decay * S1 + torch.einsum('bhd,bhe->bhde', kt, vt)
            S2 = self.decay * S2 + torch.einsum('bhd,bhe->bhde', kt**2, vt)

            K0 = self.decay * K0 + 1.0
            K1 = self.decay * K1 + kt
            K2 = self.decay * K2 + (kt**2)

            num = S0 + torch.einsum('bhd,bhde->bhe', qt, S1) + 0.5 * torch.einsum('bhd,bhde->bhe', qt**2, S2)
            den = (K0 + torch.sum(qt * K1, dim=-1, keepdim=True) + 0.5 * torch.sum((qt**2) * K2, dim=-1, keepdim=True)).clamp(min=1e-3)

            attn_output = (num / den).unsqueeze(2)

            if past_key_values is not None:
                past_key_values.prime_states[self.layer_idx] = (S0, S1, S2, K0, K1, K2)

        # Prefill Mode (L > 1)
        else:
            attn_outs = []
            for t in range(L):
                qt = q_f32[:, :, t]
                kt = kt_f32[:, :, t]
                vt = vt_f32[:, :, t]

                S0 = self.decay * S0 + vt
                S1 = self.decay * S1 + torch.einsum('bhd,bhe->bhde', kt, vt)
                S2 = self.decay * S2 + torch.einsum('bhd,bhe->bhde', kt**2, vt)

                K0 = self.decay * K0 + 1.0
                K1 = self.decay * K1 + kt
                K2 = self.decay * K2 + (kt**2)

                num = S0 + torch.einsum('bhd,bhde->bhe', qt, S1) + 0.5 * torch.einsum('bhd,bhde->bhe', qt**2, S2)
                den = (K0 + torch.sum(qt * K1, dim=-1, keepdim=True) + 0.5 * torch.sum((qt**2) * K2, dim=-1, keepdim=True)).clamp(min=1e-3)

                attn_outs.append(num / den)

            attn_output = torch.stack(attn_outs, dim=2)

            if past_key_values is not None:
                past_key_values.prime_states[self.layer_idx] = (S0, S1, S2, K0, K1, K2)

        attn_output = attn_output.to(hidden_states.dtype)
        attn_output = attn_output.transpose(1, 2).contiguous().view(*input_shape, -1)
        return self.o_proj(attn_output), None

def convert_transformer_to_prime(
    model: PreTrainedModel,
    hybrid_ratio: float = 1.0,
    decay: float = 0.9995
) -> Tuple[PreTrainedModel, Set[int]]:
    """
    Transplants PRIME Moment Attention into a Hugging Face Transformer model.
    hybrid_ratio:
      1.0 = 100% layers converted to PRIME Moment Attention.
      0.5 = 50% layers converted (retains boundary layers for exact local attention).
    """
    layers = model.model.layers
    num_layers = len(layers)

    if hybrid_ratio >= 1.0:
        layers_to_convert = set(range(num_layers))
    else:
        boundary = max(1, int(num_layers * (1.0 - hybrid_ratio) / 2))
        layers_to_convert = set(range(boundary, num_layers - boundary))

    for idx in layers_to_convert:
        orig = layers[idx].self_attn
        prime_layer = PrimeTransplantedAttention(orig, layer_idx=idx, decay=decay)
        layers[idx].self_attn = prime_layer

    return model, layers_to_convert
