import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
import math
import time

class HybridQwen2PrimeAttention(nn.Module):
    def __init__(self, orig_attn: Qwen2Attention, decay=0.995):
        super().__init__()
        self.orig_attn = orig_attn
        self.decay = decay
        self.num_heads = orig_attn.config.num_attention_heads
        self.num_kv_heads = orig_attn.config.num_key_value_heads
        self.head_dim = orig_attn.config.hidden_size // self.num_heads
        self.gate = nn.Parameter(torch.full((1, self.num_heads, 1, 1), -3.5, device=orig_attn.q_proj.weight.device, dtype=torch.bfloat16))
        self.prime_norm = nn.LayerNorm(orig_attn.config.hidden_size, dtype=torch.bfloat16).to(orig_attn.q_proj.weight.device)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, cache_position=None, **kwargs):
        orig_out = self.orig_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs
        )
        attn_output = orig_out[0]

        B, L, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        q = self.orig_attn.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.orig_attn.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.orig_attn.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        num_kv_groups = self.num_heads // self.num_kv_heads
        if num_kv_groups > 1:
            k = k.repeat_interleave(num_kv_groups, dim=1)
            v = v.repeat_interleave(num_kv_groups, dim=1)

        q = q * (1.0 / math.sqrt(self.head_dim))
        q_pos = F.elu(q.float()) + 1.0
        k_pos = F.elu(k.float()) + 1.0
        v_f32 = v.float()

        idx = torch.arange(L, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        causal_mask = diff >= 0
        decay_mat = torch.where(
            causal_mask,
            torch.pow(self.decay, diff.float()),
            torch.zeros(L, L, device=device)
        ).view(1, 1, L, L)

        A = decay_mat * torch.matmul(q_pos, k_pos.transpose(-1, -2))
        num = torch.matmul(A, v_f32)
        den = torch.sum(A, dim=-1, keepdim=True).clamp(min=1e-3)
        
        prime_out = (num / den).to(dtype).transpose(1, 2).contiguous().view(B, L, -1)
        prime_out = self.orig_attn.o_proj(prime_out)
        prime_out = self.prime_norm(prime_out)

        g = torch.sigmoid(self.gate)
        orig_out_head = attn_output.view(B, L, self.num_heads, self.head_dim)
        prime_out_head = prime_out.view(B, L, self.num_heads, self.head_dim)
        
        g_view = g.view(1, 1, self.num_heads, 1)
        fused = (1.0 - g_view) * orig_out_head + g_view * prime_out_head
        fused = fused.view(B, L, -1)

        return (fused,) + orig_out[1:]

def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"[*] Loading {model_name} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    
    print("[*] Model loaded. Transplanting Hybrid PRIME Attention...")
    for i, layer in enumerate(model.model.layers):
        layer.self_attn = HybridQwen2PrimeAttention(layer.self_attn, decay=0.995)
    
    print("[*] Freezing trunk, setting gates and norms to trainable...")
    for p in model.parameters():
        p.requires_grad = False
    
    trainable_params = []
    for layer in model.model.layers:
        layer.self_attn.gate.requires_grad = True
        layer.self_attn.prime_norm.weight.requires_grad = True
        layer.self_attn.prime_norm.bias.requires_grad = True
        trainable_params.extend([
            layer.self_attn.gate,
            layer.self_attn.prime_norm.weight,
            layer.self_attn.prime_norm.bias
        ])
    
    print(f"[*] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    
    text = "Here is a mathematical proof. We know that 2 + 2 = 4. Therefore, if we take the integral of x, we get x^2 / 2. " * 5
    tokens = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    batch = tokens.repeat(2, 1)
    
    print(f"[*] Starting short training loop (sequence length: {batch.shape[1]})...")
    
    model.train()
    for step in range(10):
        optimizer.zero_grad()
        outputs = model(input_ids=batch[:, :-1], labels=batch[:, 1:])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            avg_gate = sum(torch.sigmoid(l.self_attn.gate).mean().item() for l in model.model.layers) / len(model.model.layers)
            
        print(f"Step {step+1:2d} | Loss: {loss.item():.4f} | Avg Gate g (amount of PRIME used): {avg_gate:.4f}")
        
    print("[*] Let's test what happens if we FORCE the model to use ONLY PRIME attention (gate=1.0)...")
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.gate.fill_(10.0) 
        
        outputs_prime_only = model(input_ids=batch[:, :-1], labels=batch[:, 1:])
        print(f"[*] Loss with ONLY PRIME Attention: {outputs_prime_only.loss.item():.4f} (Compare to {loss.item():.4f} above)")
        
    print("[*] Script complete!")

if __name__ == "__main__":
    main()
