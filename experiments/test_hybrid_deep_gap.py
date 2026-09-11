#!/usr/bin/env python3
"""
Deep Gap Stress Test: Pure Sliding Window vs Pure PRIME vs Hybrid Window + PRIME
Testing at Gap 8,000 and Gap 12,000 tokens
"""

import sys, os, time
sys.path.insert(0, '/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/src')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prime_moment_attention.surgery import PrimeTransplantedAttention
from prime_moment_attention.hybrid import HybridWindowPrimeAttention

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda:0"
LAYER_IDX = 14
WINDOW_SIZE = 512

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

def test_model(config_name, config_type, gap):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE)
    orig_attn = model.model.layers[LAYER_IDX].self_attn

    if config_type == "pure_sliding_window":
        model.model.layers[LAYER_IDX].self_attn = HybridWindowPrimeAttention(
            orig_attn, layer_idx=LAYER_IDX, window_size=WINDOW_SIZE, decay=0.99999, alpha=1.0
        )
    elif config_type == "pure_prime":
        model.model.layers[LAYER_IDX].self_attn = PrimeTransplantedAttention(
            orig_attn, layer_idx=LAYER_IDX, decay=0.99999
        )
    elif config_type == "hybrid_window_prime":
        model.model.layers[LAYER_IDX].self_attn = HybridWindowPrimeAttention(
            orig_attn, layer_idx=LAYER_IDX, window_size=WINDOW_SIZE, decay=0.99999, alpha=0.5
        )

    prompt = build_context(gap)
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
    elapsed = time.time() - t0

    gen = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    success = "94812" in gen
    del model
    torch.cuda.empty_cache()
    return success, gen, elapsed

print("=" * 80)
print("DEEP GAP STRESS TEST (8,000 and 12,000 Tokens, decay=0.99999)")
print("=" * 80)

for gap in [8000, 12000]:
    print(f"\n--- Testing Gap: {gap} Tokens ---")
    for name, ctype in [
        ("Pure Sliding Window (W=512)", "pure_sliding_window"),
        ("Pure PRIME (decay=0.99999)", "pure_prime"),
        ("Hybrid Window + PRIME", "hybrid_window_prime")
    ]:
        succ, text, t = test_model(name, ctype, gap)
        status = "PASS [94812]" if succ else f"FAIL ('{text}')"
        print(f"  {name:<30} | {status:<25} | Time: {t:.2f}s")
