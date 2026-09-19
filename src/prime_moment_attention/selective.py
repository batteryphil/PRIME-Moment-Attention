"""
PRIME-Selective Attention: Constant-State Recurrent Attention with Dynamic Gating & Contrast Scaling
===================================================================================================

Combines:
  1. Learnable Inverse Temperature beta_h in [1.5, 12.0] per head (expands contrast resolution to >24:1,
     resolving the Cosine Plateau / Rank Deficit).
  2. Input-Dependent Selective Gating Delta_t = softplus(W_delta x_t + b_delta) (dynamically arrests
     state decay lambda_t -> 1.0 for salient entity retention, resolving the 32k+ Amnesia Horizon).
  3. Hybrid Mixed-Precision FP32 Accumulator with bfloat16 Linear Projections (bypasses the BF16 machine
     epsilon wall at long horizons tau > 1000 tokens).
  4. Vectorized O(1) Prefill Handoff and Clamped Prefix-Sum Parallel Sequence Training.
"""

import math
from typing import Optional, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


class PrimeSelectiveAttention(nn.Module):
    """
    PRIME-Selective Attention Module.
    Replaces quadratic causal softmax attention with temperature-scaled, selective recurrent attention.
    """
    def __init__(
        self,
        original_attn: nn.Module,
        layer_idx: int,
        min_tau: float = 2.0,
        max_tau: float = 1000.0,
        init_beta: float = 4.0
    ):
        super().__init__()
        self.config = original_attn.config
        self.layer_idx = layer_idx
        self.hidden_size = original_attn.config.hidden_size
        self.num_heads = original_attn.config.num_attention_heads
        self.head_dim = getattr(original_attn, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = original_attn.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        device = original_attn.q_proj.weight.device
        dtype = original_attn.q_proj.weight.dtype

        # Borrow original projection matrices
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

        # 1. Output normalization per-head prior to W_o projection (constrains residual stream energy)
        self.head_norm = nn.RMSNorm(self.head_dim, eps=1e-6, device=device, dtype=dtype)

        # 2. Multiscale timescale bank tau_h: biased toward long horizons for code induction
        if self.num_heads == 12:
            tau_schedule = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 384.0, 512.0, 768.0, 1024.0, 1500.0]
            init_log_tau = torch.log(torch.tensor(tau_schedule, device=device, dtype=torch.float32))
        else:
            init_log_tau = torch.linspace(math.log(min_tau), math.log(max_tau), self.num_heads, device=device)
        self.log_tau = nn.Parameter(init_log_tau)

        # 3. Learnable inverse temperature beta_h per head (init at 4.0 for ~24.5:1 contrast)
        inv_softplus_beta = math.log(math.exp(init_beta - 1.0) - 1.0)
        self.beta_param = nn.Parameter(torch.full((self.num_heads,), inv_softplus_beta, device=device))

        # 4. Input-dependent selective gating projection W_delta: [hidden_size -> num_heads]
        self.delta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True, device=device, dtype=dtype)
        nn.init.constant_(self.delta_proj.bias, 0.5413) # softplus(0.5413) ~ 1.0
        nn.init.normal_(self.delta_proj.weight, std=0.01)

    def get_tau(self, device: torch.device) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_tau.to(device)), min=1.5, max=2000.0)

    def get_beta(self, device: torch.device) -> torch.Tensor:
        return torch.clamp(F.softplus(self.beta_param.to(device)) + 1.0, min=1.5, max=12.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        B, L = input_shape
        hidden_shape = (*input_shape, -1, self.head_dim)
        device = hidden_states.device

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # L2-normalization for Taylor polynomial stability
        q_norm = F.normalize(query_states.float(), p=2, dim=-1)
        k_norm = F.normalize(key_states.float(), p=2, dim=-1)
        v_f = value_states.float()

        tau = self.get_tau(device)
        beta = self.get_beta(device)

        # Compute data-dependent selective step size: Delta_t in (0, inf)
        delta = F.softplus(self.delta_proj(hidden_states)) # [B, L, H]

        # ----------------------------------------------------------------------
        # Autoregressive Generation (L == 1) with Selective FP32 Recurrence
        # ----------------------------------------------------------------------
        if L == 1 and past_key_values is not None:
            if not hasattr(past_key_values, "prime_states"):
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

            dt = delta[:, 0, :].view(B, self.num_heads, 1).float()
            lam_vec = torch.exp(-dt / tau.view(1, self.num_heads, 1)).to(torch.float32)
            lam_mat = lam_vec.unsqueeze(-1)

            b_vec = beta.view(1, self.num_heads, 1).float()
            b_sq = (0.5 * (beta ** 2)).view(1, self.num_heads, 1).float()

            qt = q_norm[:, :, 0]
            kt = k_norm[:, :, 0]
            vt = v_f[:, :, 0]

            # FP32 Selective Accumulator Update
            S0 = lam_vec * S0 + vt
            S1 = lam_mat * S1 + b_vec.unsqueeze(-1) * torch.matmul(kt.unsqueeze(-1), vt.unsqueeze(-2))
            S2 = lam_mat * S2 + b_sq.unsqueeze(-1) * torch.matmul((kt**2).unsqueeze(-1), vt.unsqueeze(-2))

            K0 = lam_vec * K0 + 1.0
            K1 = lam_vec * K1 + b_vec * kt
            K2 = lam_vec * K2 + b_sq * (kt**2)

            num = S0 + torch.matmul(qt.unsqueeze(-2), S1).squeeze(-2) + torch.matmul((qt**2).unsqueeze(-2), S2).squeeze(-2)
            den = K0 + (qt * K1).sum(dim=-1, keepdim=True) + ((qt**2) * K2).sum(dim=-1, keepdim=True)
            den = den.clamp(min=1e-5)

            past_key_values.prime_states[self.layer_idx] = (S0, S1, S2, K0, K1, K2)
            attn_output = (num / den).to(hidden_states.dtype).unsqueeze(2)

        # ----------------------------------------------------------------------
        # Parallel Sequence Mode (Training & Vectorized Prefill)
        # ----------------------------------------------------------------------
        else:
            C = torch.cumsum(delta.float(), dim=1) # [B, L, H]
            decay_diff = (C.unsqueeze(2) - C.unsqueeze(1)).clamp(min=0.0) # [B, L, L, H]
            decay_matrix = torch.exp(-decay_diff / tau.view(1, 1, 1, self.num_heads)).permute(0, 3, 1, 2) # [B, H, L, L]

            causal_mask = torch.tril(torch.ones(L, L, device=device)).view(1, 1, L, L)
            decay_matrix = decay_matrix * causal_mask

            b_mat = beta.view(1, self.num_heads, 1, 1).float()
            sim1 = b_mat * torch.matmul(q_norm, k_norm.transpose(-1, -2))
            sim2 = 0.5 * (b_mat ** 2) * torch.matmul(q_norm**2, (k_norm**2).transpose(-1, -2))
            p_weights = torch.clamp(1.0 + sim1 + sim2, min=0.0) * decay_matrix

            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    p_weights = p_weights * attention_mask
                else:
                    p_weights = p_weights * (attention_mask >= 0.0)

            p_denom = p_weights.sum(dim=-1, keepdim=True).clamp(min=1e-5)
            normalized_weights = p_weights / p_denom

            attn_output = torch.matmul(normalized_weights, v_f).to(hidden_states.dtype)

            # Vectorized O(1) state handoff for subsequent autoregressive tokens
            if past_key_values is not None:
                if not hasattr(past_key_values, "prime_states"):
                    past_key_values.prime_states = {}
                C_last = C[:, -1:, :]
                decay_to_end = torch.exp(-(C_last - C).clamp(min=0.0) / tau.view(1, 1, self.num_heads)).permute(0, 2, 1).unsqueeze(-1)

                b_v = beta.view(1, self.num_heads, 1, 1).float()
                b_v2 = (0.5 * (beta ** 2)).view(1, self.num_heads, 1, 1).float()

                S0 = (v_f * decay_to_end).sum(dim=2)
                S1 = b_v * torch.matmul((k_norm * decay_to_end).transpose(-2, -1), v_f)
                S2 = b_v2 * torch.matmul(((k_norm**2) * decay_to_end).transpose(-2, -1), v_f)
                K0 = decay_to_end.sum(dim=2)
                K1 = beta.view(1, self.num_heads, 1).float() * (k_norm * decay_to_end).sum(dim=2)
                K2 = (0.5 * (beta ** 2)).view(1, self.num_heads, 1).float() * ((k_norm**2) * decay_to_end).sum(dim=2)
                past_key_values.prime_states[self.layer_idx] = (S0, S1, S2, K0, K1, K2)

        # Per-head RMSNorm across head_dim to constrain residual stream energy to ~1.0
        attn_output = self.head_norm(attn_output)
        attn_output = attn_output.transpose(1, 2).contiguous().view(*input_shape, -1)
        return self.o_proj(attn_output), None


def convert_transformer_to_prime_selective(
    model: PreTrainedModel,
    hybrid_ratio: float = 1.0,
    min_tau: float = 2.0,
    max_tau: float = 1000.0,
    init_beta: float = 4.0,
    hybrid_pattern: str = "sandwich",
    interleaved_interval: int = 4,
    softmax_layers: Optional[Set[int]] = None,
    layers_to_convert: Optional[Set[int]] = None
) -> Tuple[PreTrainedModel, Set[int]]:
    """
    Transplants PRIME-Selective Attention into a Hugging Face Transformer model.
    hybrid_ratio:
      1.0 = 100% layers converted to PRIME-Selective Attention.
      0.75 = 75% layers converted (e.g. 21 PRIME layers, 7 Softmax layers).
    hybrid_pattern:
      - 'sandwich': Softmax at boundaries (first/last), PRIME in the middle.
      - 'interleaved': Preserves 1 Softmax layer every `interleaved_interval` layers (e.g. layers 0, 4, 8, ...).
      - 'custom': Uses `layers_to_convert` or `softmax_layers`.
    """
    layers = model.model.layers
    num_layers = len(layers)

    if layers_to_convert is not None:
        target_layers = set(layers_to_convert)
    elif softmax_layers is not None:
        target_layers = set(range(num_layers)) - set(softmax_layers)
    elif hybrid_pattern == "interleaved":
        # Keep every interleaved_interval layer as Softmax (e.g., 0, 4, 8, 12, ...)
        softmax_set = set(range(0, num_layers, interleaved_interval))
        target_layers = set(range(num_layers)) - softmax_set
    elif hybrid_ratio >= 1.0:
        target_layers = set(range(num_layers))
    else:
        # Sandwich pattern
        boundary = max(1, int(num_layers * (1.0 - hybrid_ratio) / 2))
        target_layers = set(range(boundary, num_layers - boundary))

    for idx in target_layers:
        orig = layers[idx].self_attn
        prime_layer = PrimeSelectiveAttention(
            orig,
            layer_idx=idx,
            min_tau=min_tau,
            max_tau=max_tau,
            init_beta=init_beta
        )
        layers[idx].self_attn = prime_layer

    return model, target_layers


class PrimeSelectiveMomentAttention(nn.Module):
    """
    Native PRIME-Selective Attention module for PrimeForCausalLM.
    
    Combines:
      - 2nd-order Taylor polynomial moment recurrence.
      - Input-dependent selective step size Delta_t = softplus(W_delta x_t + b_delta).
      - Multiscale learnable timescale bank tau_h per head.
      - Learnable inverse temperature beta_h per head.
      - Identity-preserving initialization (W_delta=0, b_delta calibrated to lambda_base, beta=1.0).
      - Exact mathematical equivalence between parallel sequence mode (training) and O(1) recurrent step decoding.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        decay: float = 0.995,
        use_qk_norm: bool = True,
        eps: float = 1.0,
        min_tau: float = 2.0,
        max_tau: float = 2000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.decay = decay
        self.scaling = 1.0 / math.sqrt(head_dim)
        self.eps = eps
        self.use_qk_norm = use_qk_norm
        self.min_tau = min_tau
        self.max_tau = max_tau

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(head_dim)
            self.k_norm = nn.LayerNorm(head_dim)

        # Selective Gating: W_delta projects [hidden_size] -> [num_heads]
        self.delta_proj = nn.Linear(hidden_size, num_heads, bias=True)

        # Timescale bank tau_h per head
        base_tau = -1.0 / math.log(max(1e-5, min(0.9999, decay)))
        self.log_tau = nn.Parameter(torch.full((num_heads,), math.log(base_tau), dtype=torch.float32))

        # Learnable temperature beta_h per head (exp(log_beta) with init 0.0 -> beta=1.0)
        self.log_beta = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))

        # Identity-Preserving Initialization
        self.init_selective_weights()

    def init_selective_weights(self):
        """Initializes selective parameters to exact mathematical identity of base PRIME."""
        with torch.no_grad():
            self.delta_proj.weight.zero_()
            inv_sp_1 = math.log(math.exp(1.0) - 1.0)
            self.delta_proj.bias.fill_(inv_sp_1)
            base_tau = -1.0 / math.log(max(1e-5, min(0.9999, self.decay)))
            self.log_tau.fill_(math.log(base_tau))
            self.log_beta.zero_()

    def get_tau(self, device: torch.device) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_tau.to(device)), min=self.min_tau, max=self.max_tau)

    def get_beta(self, device: torch.device) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_beta.to(device)), min=0.1, max=15.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        B, L, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        device = hidden_states.device

        # Project Q, K, V
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q * self.scaling
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        tau = self.get_tau(device)
        beta = self.get_beta(device)
        delta = F.softplus(self.delta_proj(hidden_states)).clamp(min=1e-4, max=50.0) # [B, L, H]

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
        # Autoregressive Step Decoding Mode (L == 1)
        # -------------------------------------------------------------
        if L == 1:
            dt = delta[:, 0, :] # [B, H]
            lam_vec = torch.exp(-dt / tau.view(1, self.num_heads)).unsqueeze(-1) # [B, H, 1]
            lam_mat = lam_vec.unsqueeze(-1) # [B, H, 1, 1]
            b_vec = beta.view(1, self.num_heads, 1)
            b_sq = (0.5 * (beta ** 2)).view(1, self.num_heads, 1)

            qt = q_f32[:, :, 0]
            kt = k_f32[:, :, 0]
            vt = v_f32[:, :, 0]

            S0 = lam_vec * S0 + vt
            S1 = lam_mat * S1 + b_vec.unsqueeze(-1) * torch.einsum("bhd,bhe->bhde", kt, vt)
            S2 = lam_mat * S2 + b_sq.unsqueeze(-1) * torch.einsum("bhd,bhe->bhde", kt**2, vt)
            K0 = lam_vec * K0 + 1.0
            K1 = lam_vec * K1 + b_vec * kt
            K2 = lam_vec * K2 + b_sq * (kt**2)

            term1_num = torch.einsum("bhd,bhde->bhe", qt, S1)
            term2_num = torch.einsum("bhd,bhde->bhe", qt**2, S2)
            num = S0 + term1_num + term2_num
            term1_den = torch.sum(qt * K1, dim=-1, keepdim=True)
            term2_den = torch.sum((qt**2) * K2, dim=-1, keepdim=True)
            den = (K0 + term1_den + term2_den).clamp(min=self.eps)
            y = (num / den).unsqueeze(2)
            next_state = (S0, S1, S2, K0, K1, K2) if return_state else None

        # -------------------------------------------------------------
        # Parallel Sequence Mode (L > 1) with Prefix-Sum Cumulative Decay
        # -------------------------------------------------------------
        else:
            C = torch.cumsum(delta.float(), dim=1) # [B, L, H]
            decay_diff = (C.unsqueeze(2) - C.unsqueeze(1)).clamp(min=0.0) # [B, L, L, H]
            decay_matrix = torch.exp(-decay_diff / tau.view(1, 1, 1, self.num_heads)).permute(0, 3, 1, 2) # [B, H, L, L]
            causal_mask = torch.tril(torch.ones(L, L, device=device)).view(1, 1, L, L)
            decay_matrix = decay_matrix * causal_mask

            b_mat = beta.view(1, self.num_heads, 1, 1).float()
            sim1 = b_mat * torch.matmul(q_f32, k_f32.transpose(-1, -2))
            sim2 = 0.5 * (b_mat ** 2) * torch.matmul(q_f32**2, (k_f32**2).transpose(-1, -2))
            p_weights = (1.0 + sim1 + sim2) * decay_matrix
            p_denom = p_weights.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            y = torch.matmul(p_weights / p_denom, v_f32) # [B, H, L, D]

            if return_state:
                decay_to_end = torch.exp(-(C[:, -1:, :] - C).clamp(min=0.0) / tau.view(1, 1, self.num_heads)).permute(0, 2, 1).unsqueeze(-1)
                b_v = beta.view(1, self.num_heads, 1, 1).float()
                b_v2 = (0.5 * (beta ** 2)).view(1, self.num_heads, 1, 1).float()
                S0 = (v_f32 * decay_to_end).sum(dim=2)
                S1 = b_v * torch.matmul((k_f32 * decay_to_end).transpose(-2, -1), v_f32)
                S2 = b_v2 * torch.matmul(((k_f32**2) * decay_to_end).transpose(-2, -1), v_f32)
                K0 = decay_to_end.sum(dim=2)
                K1 = beta.view(1, self.num_heads, 1).float() * (k_f32 * decay_to_end).sum(dim=2)
                K2 = (0.5 * (beta ** 2)).view(1, self.num_heads, 1).float() * ((k_f32**2) * decay_to_end).sum(dim=2)
                next_state = (S0, S1, S2, K0, K1, K2)
            else:
                next_state = None

        y = y.to(orig_dtype).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)
        output = self.o_proj(y)
        return output, next_state
