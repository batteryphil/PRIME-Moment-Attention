import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.models.llama.modeling_llama import LlamaAttention
from datasets import load_dataset
import sys
import os
sys.path.insert(0, 'src')
from prime_moment_attention.attention import PrimeMomentAttention

class PureLlamaPrimeAttention(nn.Module):
    def __init__(self, orig_attn: LlamaAttention, decay=0.999, use_qk_norm=True):
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

def main():
    model_name = "HuggingFaceTB/SmolLM-135M"
    print(f"[*] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cuda:0")
    
    print("[*] Grafting PRIME Moment Attention...")
    for i, layer in enumerate(model.model.layers):
        layer.self_attn = PureLlamaPrimeAttention(layer.self_attn, decay=0.999, use_qk_norm=True)
    
    model.to("cuda:0")
    
    # We unfreeze EVERYTHING for a real pre-training run
    for param in model.parameters():
        param.requires_grad = True

    print(f"[*] Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Dataset: A tiny chunk of wikitext for testing the pipeline
    print("[*] Loading dataset...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5%]")
    except Exception as e:
        print(f"[!] Failed to download dataset (network?): {e}")
        print("[*] Falling back to a synthetic dataset for pipeline verification...")
        from datasets import Dataset
        dataset = Dataset.from_dict({"text": ["This is a synthetic document for pipeline testing. " * 50] * 100})

    def tokenize_func(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")

    tokenizer.pad_token = tokenizer.eos_token
    tokenized_ds = dataset.map(tokenize_func, batched=True, remove_columns=["text"])
    
    # Labels for CausalLM are identical to input_ids (shifted internally by the model)
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    tokenized_ds = tokenized_ds.map(add_labels, batched=True)

    training_args = TrainingArguments(
        output_dir="checkpoints/prime-smollm",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=1,
        save_steps=500,
        bf16=True, # Critical for memory, but we use fp32 inside the actual recurrence loop for stability
        optim="adamw_torch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
    )

    print("[*] Starting formal Training Pipeline...")
    trainer.train()

    print("[*] Saving model to checkpoints/prime-smollm/final...")
    trainer.save_model("checkpoints/prime-smollm/final")
    print("[*] Pipeline Complete!")

if __name__ == "__main__":
    main()
