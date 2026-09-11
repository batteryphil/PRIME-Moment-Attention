#!/usr/bin/env python3
"""
Diagnostic & Experiment: Does Multi-Layer PRIME or Multi-Scale Decay Extend the Retention Window?
Target: Qwen/Qwen2.5-1.5B-Instruct
Methodology:
  Plant a distinct needle token sequence at position 0.
  Feed N distractor tokens (gaps: 500, 1000, 2000, 4000, 8000).
  Query the model to retrieve the needle fact.
  Compare:
    1. Baseline Softmax (Upper bound)
    2. Single Trunk Layer 14 (decay = 0.9995)
    3. Multi-Trunk 7 Layers (decay = 0.9995)
    4. Multi-Scale Decay Trunk (Decay pyramid: 0.9995 to 0.99995 across heads/layers)
"""

import sys, os, time
sys.path.insert(0, '/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/src')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prime_moment_attention.surgery import PrimeTransplantedAttention

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print("=" * 80)
print("EXPERIMENT: NEEDLE RETRIEVAL VS GAP LENGTH ACROSS LAYER CONFIGURATIONS")
print("=" * 80)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

needle_prompt = "The secret code of Project Obsidian is 94812. "
distractor_sentence = "The weather today in the northern sector remains cold with intermittent solar flares. "
query_prompt = "Question: What is the secret code of Project Obsidian? Answer: The secret code of Project Obsidian is"

def build_context(gap_tokens):
    # Construct prompt with needle at start, then gap_tokens of distractors, then query
    base_distractor = distractor_sentence * (gap_tokens // 14 + 1)
    distractor_ids = tok(base_distractor, return_tensors="pt")["input_ids"][0][:gap_tokens]
    distractor_text = tok.decode(distractor_ids, skip_special_tokens=True)
    full_prompt = needle_prompt + distractor_text + "\n\n" + query_prompt
    return full_prompt

def test_retrieval(model, full_prompt):
    inputs = tok(full_prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=tok.pad_token_id
        )
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    success = "94812" in gen
    return gen, success

gaps = [200, 500, 1000, 2000, 4000]

def run_suite(name, prime_layers, decay=0.9995):
    print(f"\n--- Testing {name} ---")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0")
    
    for idx in prime_layers:
        orig = model.model.layers[idx].self_attn
        # If multi-scale, assign decay based on layer
        d = decay if isinstance(decay, float) else decay.get(idx, 0.9995)
        model.model.layers[idx].self_attn = PrimeTransplantedAttention(orig, layer_idx=idx, decay=d)
        
    res = {}
    for g in gaps:
        prompt = build_context(g)
        gen, success = test_retrieval(model, prompt)
        res[g] = (success, gen)
        print(f"  Gap {g:<5} tokens | Success: {str(success):<5} | Output: '{gen}'")
        
    del model
    torch.cuda.empty_cache()
    return res

results = {}
# 1. Baseline Softmax
results["Softmax Baseline"] = run_suite("Softmax Baseline (All Layers Softmax)", [])

# 2. 1 Layer PRIME (Layer 14, decay 0.9995)
results["1 Layer (Layer 14, d=0.9995)"] = run_suite("1 Layer PRIME (Layer 14, d=0.9995)", [14], decay=0.9995)

# 3. 7 Layers PRIME (Layers 10-16, d=0.9995)
results["7 Layers (Layers 10-16, d=0.9995)"] = run_suite("7 Layers PRIME (Layers 10-16, d=0.9995)", list(range(10, 17)), decay=0.9995)

# 4. Multi-Scale Decay Pyramid (Higher layers retain longer)
# Layer 10: 0.9990, Layer 11: 0.9992, Layer 12: 0.9994, Layer 13: 0.9996, Layer 14: 0.9998, Layer 15: 0.9999, Layer 16: 0.99995
pyramid_decay = {
    10: 0.9990,
    11: 0.9992,
    12: 0.9994,
    13: 0.9996,
    14: 0.9998,
    15: 0.9999,
    16: 0.99995
}
results["7 Layers (Multi-Scale Decay Pyramid)"] = run_suite("7 Layers PRIME (Multi-Scale Decay Pyramid)", list(range(10, 17)), decay=pyramid_decay)

print("\n" + "=" * 85)
print("FINAL RETRIEVAL COMPARISON MATRIX (Success / Passkey Found)")
print("=" * 85)
print(f"{'Architecture Configuration':<40} | " + " | ".join([f"Gap {g:<5}" for g in gaps]))
print("-" * 85)
for name, row in results.items():
    statuses = " | ".join([f"{'PASS' if row[g][0] else 'FAIL':<9}" for g in gaps])
    print(f"{name:<40} | {statuses}")
print("=" * 85)
