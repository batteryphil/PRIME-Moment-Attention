import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaAttention
import sys
sys.path.insert(0, 'src')
from prime_moment_attention.attention import PrimeMomentAttention

class PureLlamaPrimeAttention(nn.Module):
    def __init__(self, orig_attn: LlamaAttention, decay=0.99, use_qk_norm=True):
        super().__init__()
        self.prime_attn = PrimeMomentAttention(
            hidden_size=orig_attn.config.hidden_size,
            num_heads=orig_attn.config.num_attention_heads,
            head_dim=orig_attn.config.hidden_size // orig_attn.config.num_attention_heads,
            num_kv_heads=orig_attn.config.num_key_value_heads,
            decay=decay,
            use_qk_norm=use_qk_norm,
            chunk_threshold=10000 
        )
        self.prime_attn.q_proj.weight.data.copy_(orig_attn.q_proj.weight.data)
        self.prime_attn.k_proj.weight.data.copy_(orig_attn.k_proj.weight.data)
        self.prime_attn.v_proj.weight.data.copy_(orig_attn.v_proj.weight.data)
        self.prime_attn.o_proj.weight.data.copy_(orig_attn.o_proj.weight.data)
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, cache_position=None, **kwargs):
        # We manually call prime_attn, but we have to enforce the L > 1 parallel path manually if chunked_ssd isn't used
        # Wait, the prime_attn.forward uses chunked_prime_ssd_core, which we don't have installed right now (C++ extension).
        # We need the pure python O(L^2) materialization fallback in the wrapper.
        
        B, L, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        q = self.prime_attn.q_proj(hidden_states).view(B, L, self.prime_attn.num_heads, self.prime_attn.head_dim).transpose(1, 2)
        k = self.prime_attn.k_proj(hidden_states).view(B, L, self.prime_attn.num_kv_heads, self.prime_attn.head_dim).transpose(1, 2)
        v = self.prime_attn.v_proj(hidden_states).view(B, L, self.prime_attn.num_kv_heads, self.prime_attn.head_dim).transpose(1, 2)

        if self.prime_attn.num_kv_groups > 1:
            k = k.repeat_interleave(self.prime_attn.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.prime_attn.num_kv_groups, dim=1)

        if self.prime_attn.use_qk_norm:
            q = self.prime_attn.q_norm(q)
            k = self.prime_attn.k_norm(k)

        q = q * self.prime_attn.scaling
        
        import torch.nn.functional as F
        q_pos = F.elu(q) + 1.0
        k_pos = F.elu(k) + 1.0

        idx = torch.arange(L, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        causal_mask = diff >= 0
        decay_mat = torch.where(
            causal_mask,
            torch.pow(self.prime_attn.decay, diff.float()),
            torch.zeros(L, L, device=device)
        ).view(1, 1, L, L)

        A_1 = decay_mat * torch.matmul(q_pos, k_pos.transpose(-1, -2))
        q_pos2 = q_pos ** 2
        k_pos2 = k_pos ** 2
        A_2 = decay_mat * torch.matmul(q_pos2, k_pos2.transpose(-1, -2))
        
        decay_mat_sum = decay_mat.sum(dim=-1, keepdim=True)
        num = torch.matmul(decay_mat, v) + torch.matmul(A_1, v) + 0.5 * torch.matmul(A_2, v)
        den = (decay_mat_sum + torch.sum(A_1, dim=-1, keepdim=True) + 0.5 * torch.sum(A_2, dim=-1, keepdim=True)).clamp(min=1e-3)
        
        prime_out = (num / den).to(dtype).transpose(1, 2).contiguous().view(B, L, -1)
        prime_out = self.prime_attn.o_proj(prime_out)
        return (prime_out, None)

def generate_associative_recall_batch(batch_size, num_pairs, vocab_size, device):
    keys = torch.randint(100, vocab_size, (batch_size, num_pairs), device=device)
    vals = torch.randint(100, vocab_size, (batch_size, num_pairs), device=device)
    seq = torch.zeros(batch_size, num_pairs * 2 + 2, dtype=torch.long, device=device)
    seq[:, 0::2][:, :-1] = keys
    seq[:, 1::2][:, :-1] = vals
    target_idx = torch.randint(0, num_pairs, (batch_size,), device=device)
    query_keys = keys[torch.arange(batch_size), target_idx]
    target_vals = vals[torch.arange(batch_size), target_idx]
    seq[:, -2] = query_keys
    seq[:, -1] = target_vals
    x = seq[:, :-1]
    y = torch.full_like(seq[:, 1:], -100)
    y[:, -1] = target_vals 
    return x, y

def main():
    model_name = "HuggingFaceTB/SmolLM-135M"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    
    for i, layer in enumerate(model.model.layers):
        layer.self_attn = PureLlamaPrimeAttention(layer.self_attn, decay=1.0, use_qk_norm=True)
        
    model.to("cuda:0")
    
    # Enable QK Norm training explicitly
    trainable_params = []
    for name, param in model.named_parameters():
        if "prime_attn.q_norm" in name or "prime_attn.k_norm" in name:
            param.requires_grad = True
        elif "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name:
            param.requires_grad = True # train projections to learn mapping
        else:
            param.requires_grad = False # Freeze MLP and embeddings to focus entirely on attention logic

    # We need to include parameters in optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    print("[*] Starting Associative Recall test with QK Norm...")
    vocab_size = model.config.vocab_size
    batch_size = 32
    num_pairs = 10 
    
    model.train()
    for step in range(300):
        optimizer.zero_grad()
        x, y = generate_associative_recall_batch(batch_size, num_pairs, vocab_size, "cuda:0")
        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 200
    with torch.no_grad():
        x, y = generate_associative_recall_batch(total, num_pairs, vocab_size, "cuda:0")
        outputs = model(input_ids=x)
        logits = outputs.logits[:, -1, :] 
        preds = logits.argmax(dim=-1)
        targets = y[:, -1]
        correct = (preds == targets).sum().item()
        
    print(f"[*] Associative Recall Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    main()
