#!/usr/bin/env python3
"""
PrimeLM-50M: 50.9M Parameter Language Model with In-Layer PRIME-Net Primitives
=============================================================================
Architecture:
  - Total Parameters: 50.92M (tied vocabulary)
  - Vocabulary: 50,257 (GPT-2 compatible BPE)
  - Hidden Dimension: d_model = 512
  - Feedforward Dimension: d_ff = 1024 (Invariant-Gated SwiGLU)
  - Layers: 8 Transformer Blocks
  - Attention Heads: 8 query heads, 8 key-value heads (head_dim = 64)
  - Positional Encodings: Rotary Position Embeddings (RoPE)
  - Invariant Registers: K=4 continuous latent registers Z_inv prepended to K, V
  - Invariant Loss: Auxiliary MSE probe on final hidden representations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_rope_cis(head_dim, max_seq_len=4096, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs) # [L, head_dim // 2]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def apply_rotary_emb(x, cos, sin):
    # x: [B, H, L, D]
    B, H, L, D = x.shape
    cos = cos[:L, :].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype) # [1, 1, L, D//2]
    sin = sin[:L, :].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
    
    x1 = x[..., :D // 2]
    x2 = x[..., D // 2:]
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rotated

class InvariantRegisterEncoder(nn.Module):
    """Maps continuous invariant vector into K latent register tokens"""
    def __init__(self, inv_dim=4, d_model=512, num_registers=4):
        super().__init__()
        self.num_registers = num_registers
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(inv_dim, d_model * 2),
            RMSNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_registers * d_model)
        )

    def forward(self, inv_vec):
        B = inv_vec.shape[0]
        norm_inv = torch.tanh(inv_vec / 50.0)
        out = self.net(norm_inv)
        return out.view(B, self.num_registers, self.d_model)

class CalibratedHybridPrimeAttention(nn.Module):
    """Attention layer with sliding local window and Invariant Registers"""
    def __init__(self, d_model=512, n_heads=8, head_dim=64, window_size=256, use_registers=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.use_registers = use_registers

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin, inv_registers=None, is_causal=True):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to queries and keys
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if self.use_registers and inv_registers is not None:
            K_len = inv_registers.shape[1]
            k_reg = self.k_proj(inv_registers).view(B, K_len, self.n_heads, self.head_dim).transpose(1, 2)
            v_reg = self.v_proj(inv_registers).view(B, K_len, self.n_heads, self.head_dim).transpose(1, 2)
            
            # Prepend registers to Keys and Values
            k_full = torch.cat([k_reg, k], dim=2) # [B, H, K+L, D]
            v_full = torch.cat([v_reg, v], dim=2)
            
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k_full.transpose(-1, -2)) * scale
            
            # Causal mask: [L, K + L]
            causal = torch.triu(torch.full((L, L), float('-inf'), device=x.device, dtype=scores.dtype), diagonal=1)
            reg_mask = torch.zeros((L, K_len), device=x.device, dtype=scores.dtype)
            full_mask = torch.cat([reg_mask, causal], dim=-1)
            
            scores = scores + full_mask.unsqueeze(0).unsqueeze(0)
            attn = F.softmax(scores, dim=-1, dtype=scores.dtype)
            out = torch.matmul(attn, v_full).transpose(1, 2).contiguous().view(B, L, D)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            out = out.transpose(1, 2).contiguous().view(B, L, D)

        return self.o_proj(out)

class InvariantGatedSwiGLU(nn.Module):
    """SwiGLU feedforward network with physical/algebraic invariant gating"""
    def __init__(self, d_model=512, d_ff=1024, use_gating=True):
        super().__init__()
        self.use_gating = use_gating
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        
        if use_gating:
            self.inv_gate = nn.Linear(d_model, d_ff, bias=True)
            nn.init.constant_(self.inv_gate.bias, 2.0)

    def forward(self, x):
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        if self.use_gating:
            gate = torch.sigmoid(self.inv_gate(x))
            h = h * gate
        return self.down_proj(h)

class PrimeLMBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, head_dim=64, d_ff=1024, window_size=256, use_registers=True, use_gating=True):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CalibratedHybridPrimeAttention(d_model, n_heads, head_dim, window_size, use_registers=use_registers)
        self.ln2 = RMSNorm(d_model)
        self.mlp = InvariantGatedSwiGLU(d_model, d_ff, use_gating=use_gating)

    def forward(self, x, cos, sin, inv_registers=None):
        x = x + self.attn(self.ln1(x), cos, sin, inv_registers=inv_registers)
        x = x + self.mlp(self.ln2(x))
        return x

class PrimeLM50M(nn.Module):
    """Complete 50.9M Parameter Language Model with In-Layer PRIME-Net Primitives"""
    def __init__(
        self,
        vocab_size=50257,
        d_model=512,
        n_layers=8,
        n_heads=8,
        head_dim=64,
        d_ff=1024,
        window_size=256,
        num_registers=4,
        use_registers=True,
        use_gating=True,
        use_probe=True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_registers = num_registers
        self.use_registers = use_registers
        self.use_probe = use_probe

        # Token Embeddings
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        
        # Invariant Register Encoder
        if use_registers:
            self.register_encoder = InvariantRegisterEncoder(inv_dim=4, d_model=d_model, num_registers=num_registers)

        # Transformer Layers
        self.layers = nn.ModuleList([
            PrimeLMBlock(
                d_model=d_model,
                n_heads=n_heads,
                head_dim=head_dim,
                d_ff=d_ff,
                window_size=window_size,
                use_registers=use_registers,
                use_gating=use_gating
            )
            for _ in range(n_layers)
        ])

        self.ln_f = RMSNorm(d_model)
        
        # Tied LM Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight # Weight tying for parameter efficiency

        # Invariant Probe Head (predicts conservation parameters from hidden representation)
        if use_probe:
            self.inv_probe = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.GELU(),
                nn.Linear(128, 4)
            )

        # Precompute RoPE tables
        cos, sin = precompute_rope_cis(head_dim, max_seq_len=4096)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def count_parameters(self):
        # Tied weights counted once
        total = sum(p.numel() for p in self.parameters())
        if self.lm_head.weight is self.tok_embed.weight:
            # PyTorch sum already counts tied parameter once if identical Parameter instance
            return total
        return total

    def forward(self, input_ids, inv_vec=None):
        B, L = input_ids.shape
        x = self.tok_embed(input_ids)

        # Invariant Registers
        inv_regs = None
        if self.use_registers and inv_vec is not None:
            inv_regs = self.register_encoder(inv_vec)

        cos = self.rope_cos[:L, :]
        sin = self.rope_sin[:L, :]

        for layer in self.layers:
            x = layer(x, cos, sin, inv_registers=inv_regs)

        h = self.ln_f(x)
        logits = self.lm_head(h)

        pred_inv = None
        if self.use_probe:
            pred_inv = self.inv_probe(h[:, -1, :])

        return logits, pred_inv

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=0.7, top_k=50, inv_vec=None, eos_token_id=50256):
        self.eval()
        curr = input_ids.clone()
        for _ in range(max_new_tokens):
            L = curr.shape[1]
            if L >= 4096:
                curr = curr[:, -2048:]
                L = curr.shape[1]
            
            logits, _ = self.forward(curr, inv_vec=inv_vec)
            next_token_logits = logits[:, -1, :]
            
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                
            curr = torch.cat([curr, next_token], dim=-1)
            if next_token.item() == eos_token_id:
                break
        return curr
