"""
PRIME Causal Language Model (PrimeForCausalLM)
==============================================
A full decoder-only language model built natively with PRIME Moment Attention.
Supports training from scratch, standard cross-entropy loss, and O(1) constant-memory
autoregressive generation.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import PrimeMomentAttention


@dataclass
class PrimeConfig:
    """Configuration for PrimeForCausalLM."""
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 64
    num_kv_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    decay: float = 0.9995
    use_qk_norm: bool = True
    rms_norm_eps: float = 1e-6
    eps: float = 1.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.intermediate_size is None:
            self.intermediate_size = int(2 * (4 * self.hidden_size) / 3)


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


class PrimeBlock(nn.Module):
    """A single Transformer block powered by PRIME Moment Attention."""
    def __init__(self, config: PrimeConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = PrimeMomentAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            num_kv_heads=config.num_kv_heads,
            decay=config.decay,
            use_qk_norm=config.use_qk_norm,
            eps=config.eps,
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


class PrimeForCausalLM(nn.Module):
    """
    Full Causal Language Model with PRIME Moment Attention.
    
    Can be trained from scratch with standard autoregressive cross-entropy loss.
    Supports O(1) constant-memory recurrent step decoding during inference.
    """
    def __init__(self, config: PrimeConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([PrimeBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

        # Scale output projections once by 1/sqrt(2 * num_layers) for residual variance stability
        scale = self.config.initializer_range / math.sqrt(2 * self.config.num_layers)
        for layer in self.layers:
            layer.self_attn.o_proj.weight.data.normal_(mean=0.0, std=scale)
            layer.mlp.down_proj.weight.data.normal_(mean=0.0, std=scale)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        states: Optional[List[Tuple[torch.Tensor, ...]]] = None,
        return_states: bool = False,
    ) -> Dict[str, Any]:
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

        return {
            "loss": loss,
            "logits": logits,
            "states": next_states,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        B, L = input_ids.shape

        out = self.forward(input_ids, return_states=True)
        states = out["states"]
        next_token_logits = out["logits"][:, -1, :]

        generated = input_ids

        for _ in range(max_new_tokens):
            next_token_logits = torch.nan_to_num(next_token_logits, nan=0.0, posinf=50.0, neginf=-50.0)
            if temperature > 0.0:
                logits = next_token_logits / max(temperature, 1e-4)
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=1.0 / self.config.vocab_size)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            step_out = self.forward(next_token, states=states, return_states=True)
            states = step_out["states"]
            next_token_logits = step_out["logits"][:, -1, :]

        return generated
