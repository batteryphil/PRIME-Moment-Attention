"""
Model Surgery: In-Place Attention Layer Transplantation
=======================================================

Provides zero-shot and hybrid attention transplanting for pretrained Hugging Face models.
Replaces quadratic Softmax attention with PRIME Moment Attention (with optional Chunked SSD
acceleration and Lyapunov contractive bounding) while preserving original weights, biases,
and rotary positional embeddings (RoPE).
"""

import math
from typing import Set, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


class PrimeTransplantedAttention(nn.Module):
    def __init__(
        self,
        original_attn: nn.Module,
        layer_idx: int,
        decay: float = 0.9995,
        use_chunked_prefill: bool = True,
        chunk_size: int = 64,
        lyapunov_bound: float = 25.0,
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
        self.decay = decay
        self.use_chunked_prefill = use_chunked_prefill
        self.chunk_size = chunk_size
        self.lyapunov_bound = lyapunov_bound

        # Borrow exact projection weights and biases
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: torch.Tensor = None,
        past_key_values: Cache = None,
        **kwargs
    ) -> Tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        B, L = input_shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # GQA repeat for Key and Value
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Scale queries
        query_states = query_states * self.scaling

        # Cast to float32 for recurrence/accumulation
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
            S0 = torch.zeros(B, self.num_heads, self.head_dim, device=device, dtype=torch.float32)
            S1 = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=torch.float32)
            S2 = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=torch.float32)
            K0 = torch.zeros(B, self.num_heads, 1, device=device, dtype=torch.float32)
            K1 = torch.zeros(B, self.num_heads, self.head_dim, device=device, dtype=torch.float32)
            K2 = torch.zeros(B, self.num_heads, self.head_dim, device=device, dtype=torch.float32)

        # -------------------------------------------------------------
        # Fast Autoregressive Token Generation (L == 1)
        # -------------------------------------------------------------
        if L == 1:
            qt = q_f32[:, :, 0]
            kt = kt_f32[:, :, 0]
            vt = vt_f32[:, :, 0]

            S0 = self.decay * S0 + vt
            S1 = self.decay * S1 + torch.einsum('bhd,bhe->bhde', kt, vt)
            S2 = self.decay * S2 + torch.einsum('bhd,bhe->bhde', kt**2, vt)

            # Lyapunov bounding
            if self.lyapunov_bound > 0.0:
                s1_norm = torch.linalg.norm(S1, dim=(-2, -1), keepdim=True)
                S1 = S1 * torch.clamp(self.lyapunov_bound / (s1_norm + 1e-6), max=1.0)
                s2_norm = torch.linalg.norm(S2, dim=(-2, -1), keepdim=True)
                S2 = S2 * torch.clamp(self.lyapunov_bound / (s2_norm + 1e-6), max=1.0)

            K0 = self.decay * K0 + 1.0
            K1 = self.decay * K1 + kt
            K2 = self.decay * K2 + (kt**2)

            num = S0 + torch.einsum('bhd,bhde->bhe', qt, S1) + 0.5 * torch.einsum('bhd,bhde->bhe', qt**2, S2)
            den = (K0 + torch.sum(qt * K1, dim=-1, keepdim=True) + 0.5 * torch.sum((qt**2) * K2, dim=-1, keepdim=True)).clamp(min=1e-3)

            attn_output = (num / den).unsqueeze(2)

            if past_key_values is not None:
                past_key_values.prime_states[self.layer_idx] = (S0, S1, S2, K0, K1, K2)

        # -------------------------------------------------------------
        # Prefill Mode (L > 1): Chunked SSD or Sequential Loop
        # -------------------------------------------------------------
        elif self.use_chunked_prefill and L >= self.chunk_size:
            pad_len = (self.chunk_size - (L % self.chunk_size)) % self.chunk_size
            if pad_len > 0:
                q_p = F.pad(q_f32, (0, 0, 0, pad_len))
                k_p = F.pad(kt_f32, (0, 0, 0, pad_len))
                v_p = F.pad(vt_f32, (0, 0, 0, pad_len))
            else:
                q_p, k_p, v_p = q_f32, kt_f32, vt_f32

            L_padded = L + pad_len
            num_chunks = L_padded // self.chunk_size
            C = self.chunk_size

            q_chunks = q_p.view(B, self.num_heads, num_chunks, C, self.head_dim)
            k_chunks = k_p.view(B, self.num_heads, num_chunks, C, self.head_dim)
            v_chunks = v_p.view(B, self.num_heads, num_chunks, C, self.head_dim)

            idx = torch.arange(C, device=device)
            diff = idx.unsqueeze(1) - idx.unsqueeze(0)
            causal_mask = diff >= 0
            decay_mask = torch.where(
                causal_mask,
                torch.pow(self.decay, diff.float()),
                torch.zeros(C, C, device=device)
            ).view(1, 1, 1, C, C)

            # Intra-chunk GEMM
            dot1 = torch.matmul(q_chunks, k_chunks.transpose(-1, -2))
            dot2 = 0.5 * torch.matmul(q_chunks**2, (k_chunks**2).transpose(-1, -2))
            intra_kernel = (dot1 + dot2) * decay_mask
            y_intra = torch.matmul(intra_kernel, v_chunks)

            # Inter-chunk state propagation
            decay_vec = torch.pow(self.decay, (C - 1 - idx).float()).view(1, 1, 1, C, 1)
            k_decayed = k_chunks * decay_vec
            k2_decayed = (k_chunks**2) * decay_vec

            chunk_S1 = torch.matmul(k_decayed.transpose(-1, -2), v_chunks)
            chunk_S2 = torch.matmul(k2_decayed.transpose(-1, -2), v_chunks)

            chunk_decay = math.pow(self.decay, C)
            inter_S1_list = []
            inter_S2_list = []
            cur_S1 = S1
            cur_S2 = S2

            for c in range(num_chunks):
                inter_S1_list.append(cur_S1)
                inter_S2_list.append(cur_S2)
                cur_S1 = cur_S1 * chunk_decay + chunk_S1[:, :, c]
                cur_S2 = cur_S2 * chunk_decay + chunk_S2[:, :, c]
                if self.lyapunov_bound > 0.0:
                    s1_norm = torch.linalg.norm(cur_S1, dim=(-2, -1), keepdim=True)
                    cur_S1 = cur_S1 * torch.clamp(self.lyapunov_bound / (s1_norm + 1e-6), max=1.0)
                    s2_norm = torch.linalg.norm(cur_S2, dim=(-2, -1), keepdim=True)
                    cur_S2 = cur_S2 * torch.clamp(self.lyapunov_bound / (s2_norm + 1e-6), max=1.0)

            inter_S1 = torch.stack(inter_S1_list, dim=2)
            inter_S2 = torch.stack(inter_S2_list, dim=2)

            decay_into = torch.pow(self.decay, (idx + 1).float()).view(1, 1, 1, C, 1)
            q_decayed = q_chunks * decay_into
            q2_decayed = (q_chunks**2) * decay_into

            y_inter = torch.matmul(q_decayed, inter_S1) + 0.5 * torch.matmul(q2_decayed, inter_S2)
            y = y_intra + y_inter
            y = y.view(B, self.num_heads, L_padded, self.head_dim)
            if pad_len > 0:
                y = y[:, :, :L, :]

            attn_output = y
            if past_key_values is not None:
                past_key_values.prime_states[self.layer_idx] = (S0, cur_S1, cur_S2, K0, K1, K2)

        else:
            attn_outs = []
            for t in range(L):
                qt = q_f32[:, :, t]
                kt = kt_f32[:, :, t]
                vt = vt_f32[:, :, t]

                S0 = self.decay * S0 + vt
                S1 = self.decay * S1 + torch.einsum('bhd,bhe->bhde', kt, vt)
                S2 = self.decay * S2 + torch.einsum('bhd,bhe->bhde', kt**2, vt)

                if self.lyapunov_bound > 0.0:
                    s1_norm = torch.linalg.norm(S1, dim=(-2, -1), keepdim=True)
                    S1 = S1 * torch.clamp(self.lyapunov_bound / (s1_norm + 1e-6), max=1.0)
                    s2_norm = torch.linalg.norm(S2, dim=(-2, -1), keepdim=True)
                    S2 = S2 * torch.clamp(self.lyapunov_bound / (s2_norm + 1e-6), max=1.0)

                K0 = self.decay * K0 + 1.0
                K1 = self.decay * K1 + kt
                K2 = self.decay * K2 + (kt**2)

                num = S0 + torch.einsum('bhd,bhde->bhe', qt, S1) + 0.5 * torch.einsum('bhd,bhde->bhe', qt**2, S2)
                den = (K0 + torch.sum(qt * K1, dim=-1, keepdim=True) + 0.5 * torch.sum((qt**2) * K2, dim=-1, keepdim=True)).clamp(min=1e-3)

                attn_outs.append(num / den)

            attn_output = torch.stack(attn_outs, dim=2)
            if past_key_values is not None:
                past_key_values.prime_states[self.layer_idx] = (S0, S1, S2, K0, K1, K2)

        attn_output = attn_output.to(dtype)
        attn_output = attn_output.transpose(1, 2).contiguous().view(*input_shape, -1)
        return self.o_proj(attn_output), None


def get_model_layers(model: nn.Module):
    """
    Universally locates the decoder layer list across Hugging Face and PyTorch models.
    Supports LLaMA, Qwen, Mistral, Gemma, DeepSeek, GPT-2, Falcon, and custom modules.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "layers"):
        return model.layers
    for _, module in model.named_modules():
        if isinstance(module, (nn.ModuleList, list)) and len(module) > 1:
            if hasattr(module[0], "self_attn") or hasattr(module[0], "attn"):
                return module
    raise AttributeError("Could not automatically locate Transformer layers in model.")


def convert_transformer_to_prime(
    model: PreTrainedModel,
    hybrid_ratio: float = 1.0,
    decay: float = 0.9995,
    use_chunked_prefill: bool = True,
    chunk_size: int = 64,
    lyapunov_bound: float = 25.0,
    target_layers: Optional[Set[int]] = None
) -> Tuple[PreTrainedModel, Set[int]]:
    """
    Transplants PRIME Moment Attention into a Hugging Face Transformer model.
    target_layers:
      Specific set/list of layer indices to convert. If provided, overrides hybrid_ratio.
    hybrid_ratio:
      1.0 = 100% layers converted to PRIME Moment Attention.
      0.5 = 50% layers converted (retains boundary layers for exact local attention).
    """
    layers = get_model_layers(model)
    num_layers = len(layers)

    if target_layers is not None:
        layers_to_convert = set(target_layers)
    elif hybrid_ratio >= 1.0:
        layers_to_convert = set(range(num_layers))
    else:
        boundary = max(1, int(num_layers * (1.0 - hybrid_ratio) / 2))
        layers_to_convert = set(range(boundary, num_layers - boundary))

    for idx in layers_to_convert:
        layer = layers[idx]
        attr_name = "self_attn" if hasattr(layer, "self_attn") else "attn"
        orig = getattr(layer, attr_name)
        prime_layer = PrimeTransplantedAttention(
            orig,
            layer_idx=idx,
            decay=decay,
            use_chunked_prefill=use_chunked_prefill,
            chunk_size=chunk_size,
            lyapunov_bound=lyapunov_bound
        )
        setattr(layer, attr_name, prime_layer)

    return model, layers_to_convert
