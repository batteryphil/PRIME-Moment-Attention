#!/usr/bin/env python3
"""
Diagnostic & Experiment: Multi-Layer PRIME Surgery vs Retention Window & Generation Coherence
Target: Qwen/Qwen2.5-1.5B-Instruct (28 Layers total)
Testing layer surgery configurations:
  Config A: 0 PRIME layers (Baseline Softmax)
  Config B: 1 PRIME layer (Layer 14 - Middle Trunk)
  Config C: 2 PRIME layers (Layers 13, 14)
  Config D: 4 PRIME layers (Layers 12, 13, 14, 15)
  Config E: 7 PRIME layers (Layers 10 to 16 - 25% of model)
  Config F: 14 PRIME layers (Layers 7 to 20 - 50% of model)

For each config, measure:
  1. Text Generation Coherence & Perplexity (does it loop, babble, or stay coherent?)
  2. Information Retention over long context (retrieval cosine similarity at 500, 1000, 2000, 4000 tokens)
"""

import sys, os, time
sys.path.insert(0, '/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/src')
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from prime_moment_attention.surgery import PrimeTransplantedAttention

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print("=" * 80)
print("EXPERIMENT: MULTI-LAYER PRIME CONVERSION & RETENTION SCALING")
print(f"Model: {MODEL_ID} (28 layers)")
print("=" * 80)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def evaluate_config(name, prime_layer_indices, decay=0.9995):
    print(f"\n>>> Testing Config: {name} (Layers: {prime_layer_indices})")
    
    # Fresh model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    
    # Apply surgery
    for idx in prime_layer_indices:
        orig = model.model.layers[idx].self_attn
        model.model.layers[idx].self_attn = PrimeTransplantedAttention(orig, layer_idx=idx, decay=decay)
        
    # 1. Test Text Generation Coherence
    prompt = "The key to building reliable long-context memory in artificial intelligence is"
    inputs = tok(prompt, return_tensors="pt").to("cuda:0")
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id
        )
    gen_text = tok.decode(out[0], skip_special_tokens=True)
    
    # Detect repeating loop / degradation
    words = gen_text.split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    is_looping = unique_ratio < 0.5
    
    print(f"  [Text Sample]: {gen_text}")
    print(f"  [Coherence]: Unique ratio = {unique_ratio:.2f} | Looping/Collapsed: {is_looping}")
    
    del model
    torch.cuda.empty_cache()
    
    return {
        "name": name,
        "layers": prime_layer_indices,
        "unique_ratio": round(unique_ratio, 2),
        "coherent": not is_looping,
        "sample": gen_text
    }

configs = [
    ("Baseline (0 layers)", []),
    ("1 Layer (Layer 14)", [14]),
    ("2 Layers (Layers 13, 14)", [13, 14]),
    ("4 Layers (Layers 12-15)", [12, 13, 14, 15]),
    ("7 Layers (Layers 10-16)", list(range(10, 17))),
    ("14 Layers (Layers 7-20)", list(range(7, 21))),
]

summary = []
for name, layers in configs:
    res = evaluate_config(name, layers)
    summary.append(res)

print("\n" + "=" * 80)
print("MULTI-LAYER CONVERSION COHERENCE SUMMARY")
print("=" * 80)
print(f"{'Configuration':<30} | {'Coherent?':<12} | {'Unique Word Ratio':<20}")
print("-" * 70)
for s in summary:
    print(f"{s['name']:<30} | {str(s['coherent']):<12} | {s['unique_ratio']:<20}")
print("=" * 80)
