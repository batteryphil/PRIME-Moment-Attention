"""
PyTorch PRIME (Selective Moment Attention) Causal Language Model.
Compatible with Hugging Face Transformers AutoModelForCausalLM (trust_remote_code=True).
"""

import math
from typing import Optional, Tuple, List, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig, AutoModelForCausalLM

try:
    from .configuration_prime import PrimeConfig
except (ImportError, ValueError):
    from configuration_prime import PrimeConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class PrimeMLP(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PrimeSelectiveMomentAttention(nn.Module):
    """
    PRIME-Selective Attention: Constant-State Recurrent Attention with Dynamic Gating & Contrast Scaling.
    Uses 2nd-order Taylor polynomial recurrence with input-dependent decay rates.
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

        self.delta_proj = nn.Linear(hidden_size, num_heads, bias=True)
        base_tau = -1.0 / math.log(max(1e-5, min(0.9999, decay)))
        self.log_tau = nn.Parameter(torch.full((num_heads,), math.log(base_tau), dtype=torch.float32))
        self.log_beta = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))

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
        delta = F.softplus(self.delta_proj(hidden_states)).clamp(min=1e-4, max=50.0)

        if state is not None:
            S0, S1, S2, K0, K1, K2 = state
        else:
            S0 = torch.zeros(B, self.num_heads, self.head_dim, device=device, dtype=torch.float32)
            S1 = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=torch.float32)
            S2 = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=torch.float32)
            K0 = torch.zeros(B, self.num_heads, 1, device=device, dtype=torch.float32)
            K1 = torch.zeros(B, self.num_heads, self.head_dim, device=device, dtype=torch.float32)
            K2 = torch.zeros(B, self.num_heads, self.head_dim, device=device, dtype=torch.float32)

        if L == 1:
            # Autoregressive single-step decoding mode (O(1) state)
            dt = delta[:, 0, :]
            lam_vec = torch.exp(-dt / tau.view(1, self.num_heads)).unsqueeze(-1)
            lam_mat = lam_vec.unsqueeze(-1)
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
        else:
            # Parallel sequence mode (training & prompt prefill)
            C = torch.cumsum(delta.float(), dim=1)
            decay_diff = (C.unsqueeze(2) - C.unsqueeze(1)).clamp(min=0.0)
            decay_matrix = torch.exp(-decay_diff / tau.view(1, 1, 1, self.num_heads)).permute(0, 3, 1, 2)
            causal_mask = torch.tril(torch.ones(L, L, device=device)).view(1, 1, L, L)
            decay_matrix = decay_matrix * causal_mask

            b_mat = beta.view(1, self.num_heads, 1, 1).float()
            p_weights = torch.matmul(q_f32, k_f32.transpose(-1, -2))
            p_weights.mul_(b_mat)
            sim2 = torch.matmul(q_f32**2, (k_f32**2).transpose(-1, -2))
            sim2.mul_(0.5 * (b_mat ** 2))
            p_weights.add_(sim2)
            del sim2
            p_weights.add_(1.0)
            p_weights.mul_(decay_matrix)
            del decay_matrix
            p_denom = p_weights.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            y = torch.matmul(p_weights / p_denom, v_f32)
            del p_weights, p_denom

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


class PrimeBlock(nn.Module):
    def __init__(self, config: PrimeConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = PrimeSelectiveMomentAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            num_kv_heads=config.num_kv_heads,
            decay=config.decay,
            use_qk_norm=config.use_qk_norm,
            eps=config.eps,
            min_tau=config.min_tau,
            max_tau=config.max_tau,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = PrimeMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, next_state = self.self_attn(
            hidden_states,
            state=state,
            return_state=return_state
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, next_state


class PrimePreTrainedModel(PreTrainedModel):
    config_class = PrimeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()


class PrimeForCausalLM(PrimePreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}

    def __init__(self, config: PrimeConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([PrimeBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        states: Optional[List[Tuple[torch.Tensor, ...]]] = None,
        return_states: bool = False,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[CausalLMOutputWithPast, Dict[str, Any]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embed_tokens(input_ids)
        next_states = [] if return_states else None

        for i, layer in enumerate(self.layers):
            layer_state = states[i] if states is not None else None
            hidden_states, next_s = layer(
                hidden_states,
                state=layer_state,
                return_state=return_states
            )
            if return_states:
                next_states.append(next_s)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().to(torch.float32)
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            return (loss, logits, next_states) if loss is not None else (logits, next_states)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_states,
            hidden_states=hidden_states,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k: int = 40,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Fast O(1) constant-memory recurrent step decoding.
        Bypasses quadratic attention KV caches entirely.
        """
        self.eval()
        B, L = input_ids.shape

        out = self.forward(input_ids, return_states=True)
        states = out.past_key_values if hasattr(out, "past_key_values") else out["states"]
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        next_token_logits = logits[:, -1, :]

        generated = input_ids

        for _ in range(max_new_tokens):
            next_token_logits = torch.nan_to_num(next_token_logits, nan=0.0, posinf=50.0, neginf=-50.0)
            if temperature > 0.0:
                l_scale = next_token_logits / max(temperature, 1e-4)
                if top_k > 0:
                    v, _ = torch.topk(l_scale, min(top_k, l_scale.size(-1)))
                    l_scale[l_scale < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(l_scale, dim=-1)
                probs = torch.nan_to_num(probs, nan=1.0 / self.config.vocab_size)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            step_out = self.forward(next_token, states=states, return_states=True)
            states = step_out.past_key_values if hasattr(step_out, "past_key_values") else step_out["states"]
            step_logits = step_out.logits if hasattr(step_out, "logits") else step_out["logits"]
            next_token_logits = step_logits[:, -1, :]

        return generated

    def generate_with_primenet(
        self,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 350,
        temperature: float = 0.6,
        top_k: int = 40,
        verbose: bool = False,
    ) -> str:
        """
        Generates text using the PRIME-Net Neuro-Symbolic Co-Thinker during the <think> phase.
        """
        try:
            from src.prime_moment_attention.primenet import generate_with_primenet_cothinker
        except ImportError:
            from primenet import generate_with_primenet_cothinker
        return generate_with_primenet_cothinker(
            model=self,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            verbose=verbose,
        )


# Register custom classes with Hugging Face Auto classes
AutoConfig.register("prime", PrimeConfig)
AutoModelForCausalLM.register(PrimeConfig, PrimeForCausalLM)
