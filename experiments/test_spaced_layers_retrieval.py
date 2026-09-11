#!/usr/bin/env python3
"""
Diagnostic: Consecutive vs Spaced Multi-Layer Surgery
Hypothesis: Stacking consecutive PRIME layers (e.g. 13 + 14) compounds Taylor error P(P(x)).
What happens if we SPACE them apart so intermediate Softmax layers re-normalize the representations?
  Test 1: Layer 14 only (Baseline)
  Test 2: Spaced 2 Layers (Layer 10 and Layer 18 - separated by 7 Softmax layers)
  Test 3: Spaced 3 Layers (Layer 8, Layer 14, Layer 20 - separated by 5 Softmax layers)
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

def test_config(name, layers):
    print(f"\nTesting {name} (Layers: {layers})...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0")
    for idx in layers:
        orig = model.model.layers[idx].self_attn
        model.model.layers[idx].self_attn = PrimeTransplantedAttention(orig, layer_idx=idx, decay=0.9995)
        
    res = {}
    for g in [500, 1000, 2000, 4000]:
        prompt = build_context(g)
        inputs = tok(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        success = "94812" in gen
        res[g] = (success, gen)
        print(f"  Gap {g:<5} | Success: {str(success):<5} | Output: '{gen}'")
        
    del model
    torch.cuda.empty_cache()
    return res

results = {
    "1 Layer (14)": test_config("1 Layer (Layer 14)", [14]),
    "Spaced 2 Layers (10, 18)": test_config("Spaced 2 Layers (10, 18)", [10, 18]),
    "Spaced 3 Layers (8, 14, 20)": test_config("Spaced 3 Layers (8, 14, 20)", [8, 14, 20]),
}

print("\n" + "=" * 75)
print("SPACED SURGERY THRESHOLD MATRIX")
print("=" * 75)
print(f"{'Config':<30} | Gap 500   | Gap 1000  | Gap 2000  | Gap 4000")
print("-" * 75)
for name, row in results.items():
    statuses = " | ".join([f"{'PASS' if row[g][0] else 'FAIL':<9}" for g in [500, 1000, 2000, 4000]])
    print(f"{name:<30} | {statuses}")
print("=" * 75)
