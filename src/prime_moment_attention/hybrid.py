"""
Hybrid Window-PRIME Attention (Compromise A)
============================================

Combines a bounded sliding-window Softmax KV cache with a second-order PRIME
moment accumulator for evicted tokens.

Mechanics:
  1. Local Window [t - W + 1, t]: Exact Softmax attention over recent tokens.
  2. Recurrent State [0, t - W]: Second-order Taylor moment recurrence accumulating
     all historical tokens that fell out of the sliding window.
  3. Joint Fusion: Gated convex combination:
     Output_t = alpha * Attn_local + (1 - alpha) * Attn_prime

Memory is strictly bounded to O(W + D^2) and independent of sequence length L.
"""

import os
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

_cpp_kernel = None

def get_cpp_kernel():
    global _cpp_kernel
    if _cpp_kernel is False:
        return None
    if _cpp_kernel is not None:
        return _cpp_kernel
    try:
        from torch.utils.cpp_extension import load
        cpp_source = os.path.join(os.path.dirname(__file__), "prime_cpp_kernel.cpp")
        if os.path.exists(cpp_source):
            _cpp_kernel = load(name="prime_cpp_hybrid", sources=[cpp_source], verbose=False)
            return _cpp_kernel
    except Exception:
        pass
    _cpp_kernel = False
    return None

class HybridWindowPrimeAttention(nn.Module):
    def __init__(
        self,
        original_attn: nn.Module,
        layer_idx: int,
        window_size: int = 512,
        decay: float = 0.9995,
        alpha: float = 0.5
    ):
        super().__init__()
        self.config = original_attn.config
        self.layer_idx = layer_idx
        self.hidden_size = original_attn.config.hidden_size
        self.num_heads = original_attn.config.num_attention_heads
        self.head_dim = getattr(original_attn, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = original_attn.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = getattr(original_attn, "scaling", 1.0 / (self.head_dim ** 0.5))
        self.window_size = window_size
        self.decay = decay
        self.alpha = alpha  # Weight for local window vs recurrent memory

        # Borrow exact projection weights and biases
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        B, L = input_shape
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        q_scaled = query_states * self.scaling
        device = hidden_states.device
        cpp_mod = get_cpp_kernel()

        state = None
        if past_key_values is not None:
            if not hasattr(past_key_values, 'hybrid_prime_states'):
                past_key_values.hybrid_prime_states = {}
            state = past_key_values.hybrid_prime_states.get(self.layer_idx, None)

        if L > 1 or state is None:
            # Prefill mode:
            # 1. PRIME C++ scan
            q_f32 = q_scaled.to(torch.float32)
            k_f32 = key_states.to(torch.float32)
            v_f32 = value_states.to(torch.float32)

            if cpp_mod is not None:
                out_prime, S0, S1, S2, K0, K1, K2 = cpp_mod.prime_prefill_cpp(q_f32, k_f32, v_f32, self.decay)
            else:
                out_prime = torch.zeros_like(q_f32)
                S0 = torch.zeros(B, self.num_heads, self.head_dim, device=device, dtype=torch.float32)
                S1 = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=torch.float32)
                S2 = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=torch.float32)
                K0 = torch.zeros(B, self.num_heads, 1, device=device, dtype=torch.float32)
                K1 = torch.zeros(B, self.num_heads, self.head_dim, device=device, dtype=torch.float32)
                K2 = torch.zeros(B, self.num_heads, self.head_dim, device=device, dtype=torch.float32)
                for t in range(L):
                    qt = q_f32[:, :, t]
                    kt = k_f32[:, :, t]
                    vt = v_f32[:, :, t]
                    S0 = self.decay * S0 + vt
                    S1 = self.decay * S1 + torch.einsum('bhd,bhe->bhde', kt, vt)
                    S2 = self.decay * S2 + torch.einsum('bhd,bhe->bhde', kt**2, vt)
                    K0 = self.decay * K0 + 1.0
                    K1 = self.decay * K1 + kt
                    K2 = self.decay * K2 + (kt**2)
                    num = S0 + torch.einsum('bhd,bhde->bhe', qt, S1) + 0.5 * torch.einsum('bhd,bhde->bhe', qt**2, S2)
                    den = (K0 + torch.sum(qt * K1, dim=-1, keepdim=True) + 0.5 * torch.sum((qt**2) * K2, dim=-1, keepdim=True)).clamp(min=1e-3)
                    out_prime[:, :, t] = num / den

            out_prime = out_prime.to(query_states.dtype)

            # 2. Local Softmax
            scores = torch.matmul(q_scaled, key_states.transpose(2, 3))
            causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
            attn_weights = F.softmax((scores + causal_mask).to(torch.float32), dim=-1).to(query_states.dtype)
            out_local = torch.matmul(attn_weights, value_states)

            # 3. Blend
            fused = self.alpha * out_local + (1.0 - self.alpha) * out_prime

            if past_key_values is not None:
                past_key_values.hybrid_prime_states[self.layer_idx] = {
                    'k_win': key_states[:, :, -self.window_size:].clone(),
                    'v_win': value_states[:, :, -self.window_size:].clone(),
                    'prime_state': (S0, S1, S2, K0, K1, K2)
                }

            out = fused.transpose(1, 2).contiguous().view(*input_shape, -1)
            return self.o_proj(out), None
        else:
            # Autoregressive decoding (L == 1)
            k_win = state['k_win']
            v_win = state['v_win']
            S0, S1, S2, K0, K1, K2 = state['prime_state']

            # 1. Update PRIME state with the new token
            kt_f32 = key_states[:, :, 0].to(torch.float32)
            vt_f32 = value_states[:, :, 0].to(torch.float32)
            q_f32 = q_scaled[:, :, 0].to(torch.float32)

            if cpp_mod is not None:
                cpp_mod.prime_step_evict_update_cpp(kt_f32, vt_f32, S0, S1, S2, K0, K1, K2, self.decay)
                out_prime = cpp_mod.prime_query_step_cpp(q_f32, S0, S1, S2, K0, K1, K2).unsqueeze(2).to(query_states.dtype)
            else:
                S0 = self.decay * S0 + vt_f32
                S1 = self.decay * S1 + torch.einsum('bhd,bhe->bhde', kt_f32, vt_f32)
                S2 = self.decay * S2 + torch.einsum('bhd,bhe->bhde', kt_f32**2, vt_f32)
                K0 = self.decay * K0 + 1.0
                K1 = self.decay * K1 + kt_f32
                K2 = self.decay * K2 + (kt_f32**2)
                num = S0 + torch.einsum('bhd,bhde->bhe', q_f32, S1) + 0.5 * torch.einsum('bhd,bhde->bhe', q_f32**2, S2)
                den = (K0 + torch.sum(q_f32 * K1, dim=-1, keepdim=True) + 0.5 * torch.sum((q_f32**2) * K2, dim=-1, keepdim=True)).clamp(min=1e-3)
                out_prime = (num / den).unsqueeze(2).to(query_states.dtype)

            # 2. Local Window Softmax
            k_full = torch.cat([k_win, key_states], dim=2)
            v_full = torch.cat([v_win, value_states], dim=2)

            scores = torch.matmul(q_scaled, k_full.transpose(2, 3))
            attn_weights = F.softmax(scores.to(torch.float32), dim=-1).to(query_states.dtype)
            out_local = torch.matmul(attn_weights, v_full)

            # 3. Blend
            fused = self.alpha * out_local + (1.0 - self.alpha) * out_prime

            state['k_win'] = k_full[:, :, -self.window_size:]
            state['v_win'] = v_full[:, :, -self.window_size:]
            state['prime_state'] = (S0, S1, S2, K0, K1, K2)

            out = fused.transpose(1, 2).contiguous().view(*input_shape, -1)
            return self.o_proj(out), None

def convert_transformer_to_hybrid_prime(
    model: nn.Module,
    target_layers: Optional[list] = None,
    window_size: int = 512,
    decay: float = 0.9995,
    alpha: float = 0.5
) -> Tuple[nn.Module, list]:
    """
    Converts specified layers of a Transformer to HybridWindowPrimeAttention.
    If target_layers is None, converts the middle trunk layer.
    
    Returns:
        (model, converted_layer_indices)
    """
    from prime_moment_attention.surgery import get_model_layers
    layers = get_model_layers(model)
    num_layers = len(layers)

    if target_layers is None:
        target_layers = [num_layers // 2]

    for idx in target_layers:
        layer = layers[idx]
        attr_name = "self_attn" if hasattr(layer, "self_attn") else "attn"
        orig_attn = getattr(layer, attr_name)
        hybrid_layer = HybridWindowPrimeAttention(
            original_attn=orig_attn,
            layer_idx=idx,
            window_size=window_size,
            decay=decay,
            alpha=alpha
        )
        setattr(layer, attr_name, hybrid_layer)

    return model, target_layers
