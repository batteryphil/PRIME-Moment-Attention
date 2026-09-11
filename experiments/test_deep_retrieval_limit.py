#!/usr/bin/env python3
"""
Stress Test: Testing the true long-context retrieval boundary beyond 4,000 tokens.
Comparing:
  1. Softmax Baseline
  2. 1 Layer PRIME (Layer 14, decay 0.9995)
  3. Spaced 2 Layers PRIME (Layers 10, 18, decay 0.9995)
  4. Spaced 2 Layers with High Retention Decay (Layers 10, 18, decay 0.9999)

Testing Gaps: 4,000 | 6,000 | 8,000 | 12,000 tokens
"""

import sys, os, time
sys.path.insert(0, '/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/src')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prime_moment_attention.surgery import PrimeTransplantedAttention

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

needle_prompt = "The secret code of Project Obsidian is 94812. "
distractor_sentence = "The weather today in the northern sector remains cold with intermittent solar flares. "
query_prompt = "Question: What is the secret code of Project Obsidian? Answer: The secret code of Project Obsidian is"

def build_context(gap_tokens):
    base_distractor = distractor_sentence * (gap_tokens // 14 + 1)
    distractor_ids = tok(base_distractor, return_tensors="pt")["input_ids"][0][:gap_tokens]
    distractor_text = tok.decode(distractor_ids, skip_special_tokens=True)
    return needle_prompt + distractor_text + "\n\n" + query_prompt

def test_config(name, layers, decay=0.9995):
    print(f"\n--- Testing {name} (Layers: {layers}, Decay: {decay}) ---")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0")
    for idx in layers:
        orig = model.model.layers[idx].self_attn
        model.model.layers[idx].self_attn = PrimeTransplantedAttention(orig, layer_idx=idx, decay=decay)
        
    res = {}
    for g in [4000, 6000, 8000]:
        prompt = build_context(g)
        inputs = tok(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        success = "94812" in gen
        res[g] = (success, gen)
        print(f"  Gap {g:<5} tokens | Success: {str(success):<5} | Output: '{gen}'")
        
    del model
    torch.cuda.empty_cache()
    return res

results = {
    "Softmax Baseline": test_config("Softmax Baseline", []),
    "1 Layer (14, d=0.9995)": test_config("1 Layer (14, d=0.9995)", [14], decay=0.9995),
    "Spaced 2 Layers (10, 18, d=0.9995)": test_config("Spaced 2 Layers (10, 18, d=0.9995)", [10, 18], decay=0.9995),
    "Spaced 2 Layers (10, 18, d=0.9999)": test_config("Spaced 2 Layers (10, 18, d=0.9999)", [10, 18], decay=0.9999),
}

print("\n" + "=" * 80)
print("DEEP CONTEXT RETRIEVAL MATRIX (Beyond 4K Tokens)")
print("=" * 80)
print(f"{'Config':<35} | Gap 4000  | Gap 6000  | Gap 8000")
print("-" * 80)
for name, row in results.items():
    statuses = " | ".join([f"{'PASS' if row[g][0] else 'FAIL':<9}" for g in [4000, 6000, 8000]])
    print(f"{name:<35} | {statuses}")
print("=" * 80)
