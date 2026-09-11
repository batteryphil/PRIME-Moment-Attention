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

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

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

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # GQA repeat for Key and Value
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Scale query
        query_states = query_states * self.scaling

        device = query_states.device
        H, D = self.num_heads, self.head_dim

        # ----------------------------------------------------------------------
        # MODE 1: Prefill (L > 1)
        # ----------------------------------------------------------------------
        if L > 1:
            # 1. Local Window Attention (Tokens attend to the most recent min(t, W) tokens)
            scores = torch.matmul(query_states, key_states.transpose(2, 3))  # [B, H, L, L]

            # Create causal + sliding window mask
            causal_mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)
            window_mask = torch.tril(torch.full((L, L), float("-inf"), device=device), diagonal=-self.window_size)
            combined_mask = causal_mask + window_mask

            attn_weights = F.softmax((scores + combined_mask.unsqueeze(0).unsqueeze(0)).to(torch.float32), dim=-1).to(query_states.dtype)
            local_attn_out = torch.matmul(attn_weights, value_states)  # [B, H, L, D]

            # 2. Accumulate tokens that have fallen out of the final window into PRIME state
            num_evicted = max(0, L - self.window_size)
            
            S0 = torch.zeros(B, H, D, device=device, dtype=torch.float32)
            S1 = torch.zeros(B, H, D, D, device=device, dtype=torch.float32)
            S2 = torch.zeros(B, H, D, D, device=device, dtype=torch.float32)
            K0 = torch.zeros(B, H, 1, device=device, dtype=torch.float32)
            K1 = torch.zeros(B, H, D, device=device, dtype=torch.float32)
            K2 = torch.zeros(B, H, D, device=device, dtype=torch.float32)

            if num_evicted > 0:
                k_evict = key_states[:, :, :num_evicted].to(torch.float32)
                v_evict = value_states[:, :, :num_evicted].to(torch.float32)
                
                # Recurrent accumulation over evicted tokens
                for t in range(num_evicted):
                    kt = k_evict[:, :, t]
                    vt = v_evict[:, :, t]
                    S0 = self.decay * S0 + vt
                    S1 = self.decay * S1 + torch.einsum('bhd,bhe->bhde', kt, vt)
                    S2 = self.decay * S2 + torch.einsum('bhd,bhe->bhde', kt**2, vt)
                    K0 = self.decay * K0 + 1.0
                    K1 = self.decay * K1 + kt
                    K2 = self.decay * K2 + (kt**2)

                # Fuse PRIME distant attention for the final token that initiates generation
                q_last = query_states[:, :, -1].to(torch.float32)
                num_prime = S0 + torch.einsum('bhd,bhde->bhe', q_last, S1) + 0.5 * torch.einsum('bhd,bhde->bhe', q_last**2, S2)
                den_prime = (K0 + torch.sum(q_last * K1, dim=-1, keepdim=True) + 0.5 * torch.sum((q_last**2) * K2, dim=-1, keepdim=True)).clamp(min=1e-3)
                prime_last_out = (num_prime / den_prime).to(query_states.dtype)

                # Fuse for the final prefill token
                local_attn_out[:, :, -1] = self.alpha * local_attn_out[:, :, -1] + (1.0 - self.alpha) * prime_last_out

            # 3. Store local window buffer and PRIME recurrent state in past_key_values
            if past_key_values is not None:
                if not hasattr(past_key_values, 'hybrid_window_states'):
                    past_key_values.hybrid_window_states = {}
                
                # Window buffer stores the last W tokens
                w_start = max(0, L - self.window_size)
                k_win = key_states[:, :, w_start:].clone()
                v_win = value_states[:, :, w_start:].clone()

                past_key_values.hybrid_window_states[self.layer_idx] = {
                    "k_window": k_win,
                    "v_window": v_win,
                    "prime_state": (S0, S1, S2, K0, K1, K2),
                    "num_evicted": num_evicted
                }

            attn_output = local_attn_out.transpose(1, 2).contiguous().view(*input_shape, -1)
            return self.o_proj(attn_output), None

        # ----------------------------------------------------------------------
        # MODE 2: Autoregressive Decoding (L == 1)
        # ----------------------------------------------------------------------
        state = None
        if past_key_values is not None and hasattr(past_key_values, 'hybrid_window_states'):
            state = past_key_values.hybrid_window_states.get(self.layer_idx, None)

        if state is None:
            # Fallback if no cache
            scores = torch.matmul(query_states, key_states.transpose(2, 3))
            local_weights = F.softmax(scores.to(torch.float32), dim=-1).to(query_states.dtype)
            local_out = torch.matmul(local_weights, value_states)
            attn_output = local_out.transpose(1, 2).contiguous().view(*input_shape, -1)
            return self.o_proj(attn_output), None

        k_win = state["k_window"]
        v_win = state["v_window"]
        S0, S1, S2, K0, K1, K2 = state["prime_state"]

        # Step 2a: Local Softmax Attention over the current window
        k_full = torch.cat([k_win, key_states], dim=2)
        v_full = torch.cat([v_win, value_states], dim=2)

        local_scores = torch.matmul(query_states, k_full.transpose(2, 3))  # [B, H, 1, WinLen+1]
        local_weights = F.softmax(local_scores.to(torch.float32), dim=-1).to(query_states.dtype)
        local_attn_out = torch.matmul(local_weights, v_full)  # [B, H, 1, D]

        # Step 2b: Distant PRIME Attention from Recurrent State
        q_f32 = query_states[:, :, 0].to(torch.float32)  # [B, H, D]
        has_prime_history = (state["num_evicted"] > 0)

        if has_prime_history:
            num_prime = S0 + torch.einsum('bhd,bhde->bhe', q_f32, S1) + 0.5 * torch.einsum('bhd,bhde->bhe', q_f32**2, S2)
            den_prime = (K0 + torch.sum(q_f32 * K1, dim=-1, keepdim=True) + 0.5 * torch.sum((q_f32**2) * K2, dim=-1, keepdim=True)).clamp(min=1e-3)
            prime_attn_out = (num_prime / den_prime).unsqueeze(2).to(query_states.dtype)  # [B, H, 1, D]

            # Step 2c: Joint Convex Fusion
            fused_out = self.alpha * local_attn_out + (1.0 - self.alpha) * prime_attn_out
        else:
            # All tokens still fit in local window, pure local softmax is exact
            fused_out = local_attn_out

        # Step 2d: Window Eviction & PRIME Update
        if k_full.size(2) > self.window_size:
            kt_evict = k_full[:, :, 0].to(torch.float32)
            vt_evict = v_full[:, :, 0].to(torch.float32)

            S0 = self.decay * S0 + vt_evict
            S1 = self.decay * S1 + torch.einsum('bhd,bhe->bhde', kt_evict, vt_evict)
            S2 = self.decay * S2 + torch.einsum('bhd,bhe->bhde', kt_evict**2, vt_evict)
            K0 = self.decay * K0 + 1.0
            K1 = self.decay * K1 + kt_evict
            K2 = self.decay * K2 + (kt_evict**2)

            k_win = k_full[:, :, 1:]
            v_win = v_full[:, :, 1:]
            state["num_evicted"] += 1
        else:
            k_win = k_full
            v_win = v_full

        state["k_window"] = k_win
        state["v_window"] = v_win
        state["prime_state"] = (S0, S1, S2, K0, K1, K2)

        attn_output = fused_out.transpose(1, 2).contiguous().view(*input_shape, -1)
        return self.o_proj(attn_output), None

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
    layers = getattr(model, "model", model).layers
    num_layers = len(layers)

    if target_layers is None:
        target_layers = [num_layers // 2]

    for idx in target_layers:
        orig_attn = layers[idx].self_attn
        layers[idx].self_attn = HybridWindowPrimeAttention(
            original_attn=orig_attn,
            layer_idx=idx,
            window_size=window_size,
            decay=decay,
            alpha=alpha
        )

    return model, target_layers
