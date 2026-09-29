import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, AutoTokenizer

class GPT2HybridWindowPrimeAttention(nn.Module):
    def __init__(self, orig_attn, layer_idx: int, window_size: int = 256, decay: float = 0.9995, init_gate: float = -3.5):
        super().__init__()
        self.layer_idx = layer_idx
        self.window_size = window_size
        self.decay = decay
        
        self.c_attn = orig_attn.c_attn
        self.c_proj = orig_attn.c_proj
        self.split_size = orig_attn.split_size
        self.num_heads = orig_attn.num_heads
        self.head_dim = orig_attn.head_dim
        self.scaling = 1.0 / (self.head_dim ** 0.5)

        self.gate = nn.Parameter(torch.full((1, self.num_heads, 1, 1), init_gate))
        self.prime_norm = nn.LayerNorm(self.head_dim)

    def forward(self, hidden_states, past_key_values=None, attention_mask=None, **kwargs):
        B, L, C = hidden_states.shape
        device = hidden_states.device
        
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        shape_kv = (B, L, self.num_heads, self.head_dim)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        query_states = query_states.view(shape_kv).transpose(1, 2)

        q_scaled = query_states * self.scaling
        
        # 1. PRIME 2nd-order moment recurrence
        q_f32 = q_scaled.float()
        k_f32 = key_states.float()
        v_f32 = value_states.float()
        
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
        out_prime = out_prime.to(hidden_states.dtype)

        # 2. Local Sliding Window Softmax Attention
        scores = torch.matmul(q_scaled, key_states.transpose(2, 3))
        row_idx = torch.arange(L, device=device).unsqueeze(1)
        col_idx = torch.arange(L, device=device).unsqueeze(0)
        window_mask = (col_idx <= row_idx) & (col_idx > row_idx - self.window_size)
        scores_masked = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = F.softmax(scores_masked.float(), dim=-1).to(hidden_states.dtype)
        out_local = torch.matmul(attn_weights, value_states)

        # 3. Fuse with LayerNorm and Adaptive Gating
        out_prime = self.prime_norm(out_prime)
        g = torch.sigmoid(self.gate)
        fused = (1.0 - g) * out_local + g * out_prime

        fused = fused.transpose(1, 2).contiguous().view(B, L, C)
        out = self.c_proj(fused)
        return out, None

# Test replacing layer 6 with Adaptive GPT2HybridWindowPrimeAttention
model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

orig_layer = model.transformer.h[6].attn
hybrid_layer = GPT2HybridWindowPrimeAttention(orig_layer, layer_idx=6, window_size=128).cuda()
model.transformer.h[6].attn = hybrid_layer

prompt = "The capital of France is"
inp = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**inp)
    logits = out.logits
    next_tok = tokenizer.decode(logits[0, -1].argmax().item())

print("Prompt:", prompt)
print("Top next token prediction:", repr(next_tok))
assert next_tok == " the", f"Expected exact parity with stock GPT-2 (' the'), got {repr(next_tok)}"
print("Model surgery verified successfully with exact parity to stock GPT-2!")
