import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import time
import sys
import argparse

sys.path.insert(0, 'src')
from prime_moment_attention.hybrid import HybridWindowPrimeAttention
from prime_moment_attention.thought_reconstruction_map import GenerativeThoughtReconstructionLayer

class ZeroInitGistWrapper(nn.Module):
    def __init__(self, orig_layer, d_map=32, decay=0.995):
        super().__init__()
        self.orig_layer = orig_layer
        self.gist = GenerativeThoughtReconstructionLayer(
            d_model=orig_layer.hidden_size, d_map=d_map, decay=decay
        )
        nn.init.zeros_(self.gist.recon_proj.weight)
        if self.gist.recon_proj.bias is not None:
            nn.init.zeros_(self.gist.recon_proj.bias)
        self.gist.to(device=orig_layer.self_attn.q_proj.weight.device, dtype=torch.bfloat16)
        
    def forward(self, hidden_states, *args, **kwargs):
        layer_outputs = self.orig_layer(hidden_states, *args, **kwargs)
        orig_hidden = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        gist_out, _ = self.gist(orig_hidden, return_state=False)
        return (gist_out,) + layer_outputs[1:] if isinstance(layer_outputs, tuple) else gist_out

def generate_passkey_task(tokenizer, seq_len):
    passkey = random.randint(10000, 99999)
    prefix = f"There is important hidden information in this text. "
    filler = "The quick brown fox jumps over the lazy dog. "
    suffix = "\nThe passkey is "
    
    prefix_tokens = tokenizer(prefix, return_tensors="pt").input_ids[0]
    filler_tokens = tokenizer(filler, return_tensors="pt").input_ids[0]
    passkey_str = f"The passkey is {passkey}. Remember it. "
    passkey_tokens = tokenizer(passkey_str, return_tensors="pt").input_ids[0]
    suffix_tokens = tokenizer(suffix, return_tensors="pt").input_ids[0]
    target_tokens = tokenizer(str(passkey), return_tensors="pt").input_ids[0]
    
    num_fillers = (seq_len - len(prefix_tokens) - len(passkey_tokens) - len(suffix_tokens) - len(target_tokens)) // len(filler_tokens)
    if num_fillers < 0: num_fillers = 1
    
    # Place passkey randomly in the first half of the sequence
    insert_pos = random.randint(0, num_fillers // 2)
    
    seq = prefix_tokens.tolist()
    for i in range(num_fillers):
        if i == insert_pos:
            seq.extend(passkey_tokens.tolist())
        seq.extend(filler_tokens.tolist())
    seq.extend(suffix_tokens.tolist())
    
    input_ids = torch.tensor([seq])
    labels = torch.tensor([target_tokens.tolist()])
    return input_ids, labels, str(passkey)

def evaluate_model(model, tokenizer, lengths):
    model.eval()
    results = {}
    
    for length in lengths:
        print(f"  Testing Context Length: {length}")
        correct = 0
        trials = 5
        
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            for _ in range(trials):
                input_ids, target_ids, passkey_str = generate_passkey_task(tokenizer, length)
                input_ids = input_ids.to(model.device)
                target_ids = target_ids.to(model.device)
                
                with torch.no_grad():
                    # We just need to generate a few tokens
                    out_tokens = model.generate(
                        input_ids,
                        max_new_tokens=6,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False
                    )
                
                gen_text = tokenizer.decode(out_tokens[0][input_ids.shape[1]:])
                if passkey_str in gen_text:
                    correct += 1
                    
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            acc = correct / trials
            results[length] = {"acc": acc, "mem": peak_mem}
            print(f"    -> Acc: {acc*100:.0f}% | VRAM: {peak_mem:.2f}GB")
            
        except torch.cuda.OutOfMemoryError:
            print(f"    -> OOM! VRAM exceeded 16GB.")
            results[length] = {"acc": 0.0, "mem": 16.0}
    return results

def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"[1] Loading Baseline {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    
    lengths = [1000, 2000, 4000, 8000, 16000]
    
    print("\n[2] Evaluating Baseline Softmax Context Limits...")
    base_results = evaluate_model(model, tokenizer, lengths)
    
    print("\n[3] Converting to Hybrid PRIME (Layer 7 = PRIME, Layer 14 = Gist)...")
    model.model.layers[7].self_attn = HybridWindowPrimeAttention(
        original_attn=model.model.layers[7].self_attn, layer_idx=7, window_size=512, alpha=1.0
    )
    model.model.layers[14] = ZeroInitGistWrapper(model.model.layers[14])
    
    print("\n[4] Commencing Continuous Training on Long Context Passkey...")
    # Train the injected layers to utilize their recurrent state
    for p in model.parameters():
        p.requires_grad = False
    for p in model.model.layers[7].self_attn.parameters():
        p.requires_grad = True
    for p in model.model.layers[14].gist.parameters():
        p.requires_grad = True
        
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    
    model.train()
    # Curriculum: start short, get longer
    train_lengths = [1000]*5 + [2000]*5
    for i, seq_len in enumerate(train_lengths):
        optimizer.zero_grad()
        input_ids, target_ids, _ = generate_passkey_task(tokenizer, seq_len)
        input_ids = input_ids.to(model.device)
        target_ids = target_ids.to(model.device)
        
        # Teacher forcing
        full_seq = torch.cat([input_ids, target_ids], dim=1)
        labels = torch.full_like(full_seq, -100)
        labels[:, -target_ids.shape[1]:] = target_ids
        
        outputs = model(input_ids=full_seq[:, :-1], labels=labels[:, 1:])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"  Train Step {i+1}/{len(train_lengths)} | Len: {seq_len} | Loss: {loss.item():.4f}")

    print("\n[5] Evaluating Trained Hybrid PRIME Context Limits...")
    model.gradient_checkpointing_disable()
    hybrid_results = evaluate_model(model, tokenizer, lengths)
    
    print("\n================ SUMMARY ================")
    print("Length | Baseline Acc (Mem) | Hybrid Acc (Mem)")
    for l in lengths:
        b_acc = base_results[l]['acc']*100; b_mem = base_results[l]['mem']
        h_acc = hybrid_results[l]['acc']*100; h_mem = hybrid_results[l]['mem']
        print(f"{l:5d}  | {b_acc:3.0f}% ({b_mem:4.1f} GB)    | {h_acc:3.0f}% ({h_mem:4.1f} GB)")

if __name__ == "__main__":
    main()
