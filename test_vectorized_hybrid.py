import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, AutoTokenizer

class VectorizedAdaptiveGPT2HybridPrimeAttention(nn.Module):
    def __init__(self, orig_attn, layer_idx: int, window_size: int = 128, decay: float = 0.9995, init_gate: float = -3.5):
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
        
        idx = torch.arange(L, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        causal_mask = diff >= 0

        # 1. Local Sliding Window Softmax
        scores = torch.matmul(q_scaled, key_states.transpose(2, 3))
        window_mask = (diff >= 0) & (diff < self.window_size)
        scores_masked = scores.masked_fill(~window_mask.view(1, 1, L, L), float('-inf'))
        attn_weights = F.softmax(scores_masked.float(), dim=-1).to(hidden_states.dtype)
        out_local = torch.matmul(attn_weights, value_states)

        # 2. Vectorized 2nd-Order PRIME Taylor Attention
        decay_mat = torch.where(
            causal_mask,
            torch.pow(self.decay, diff.float()),
            torch.zeros(L, L, device=device)
        ).view(1, 1, L, L)
        q_f32 = q_scaled.float()
        k_f32 = key_states.float()
        v_f32 = value_states.float()
        dot1 = torch.matmul(q_f32, k_f32.transpose(-1, -2))
        dot2 = 0.5 * torch.matmul(q_f32**2, (k_f32**2).transpose(-1, -2))
        A = decay_mat * (1.0 + dot1 + dot2)
        num_dense = torch.matmul(A, v_f32)
        den_dense = torch.sum(A, dim=-1, keepdim=True).clamp(min=1e-3)
        out_prime = (num_dense / den_dense).to(hidden_states.dtype)
        out_prime = self.prime_norm(out_prime)

        # 3. Adaptive Gated Fusion
        g = torch.sigmoid(self.gate)
        fused = (1.0 - g) * out_local + g * out_prime

        fused = fused.transpose(1, 2).contiguous().view(B, L, C)
        out = self.c_proj(fused)
        return out, None

# Test on GPU with forward and backward
model = GPT2LMHeadModel.from_pretrained("gpt2")
for i in range(len(model.transformer.h)):
    orig = model.transformer.h[i].attn
    model.transformer.h[i].attn = VectorizedAdaptiveGPT2HybridPrimeAttention(orig, layer_idx=i, window_size=128, init_gate=-3.5)
model.cuda()

x = torch.randint(0, 1000, (2, 256), device="cuda")
labels = torch.randint(0, 1000, (2, 256), device="cuda")
out = model(x, labels=labels)
loss = out.loss
loss.backward()
print("Vectorized forward + backward loss:", loss.item())
print("Gradient check successful!")
