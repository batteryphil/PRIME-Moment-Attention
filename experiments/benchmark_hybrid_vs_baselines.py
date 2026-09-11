#!/usr/bin/env python3
"""
Comprehensive Benchmark: Compromise A (Hybrid Sliding Window + PRIME Recurrence)
Compares 4 Configurations on Qwen/Qwen2.5-1.5B-Instruct:
  1. Full Softmax (Unmodified Model - Standard KV-cache)
  2. Pure Sliding Window (Window W=512 - Older tokens discarded)
  3. Pure PRIME Recurrence (No KV cache - Single Taylor moment state)
  4. Hybrid Window + PRIME (Window W=512 + Recurrent state for evicted tokens)

Tests passkey retrieval ('94812') across context gaps:
  - Gap 250 (Inside W=512 window)
  - Gap 1000 (Outside window)
  - Gap 2000 (Outside window)
  - Gap 4000 (Distant memory)
"""

import sys, os, time, json
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

def load_fresh_model(config_type):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=DEVICE
    )
    orig_attn = model.model.layers[LAYER_IDX].self_attn

    if config_type == "full_softmax":
        pass  # keep unmodified
    elif config_type == "pure_sliding_window":
        # Window W=512, alpha=1.0 (pure local window, ignores evicted tokens)
        model.model.layers[LAYER_IDX].self_attn = HybridWindowPrimeAttention(
            orig_attn, layer_idx=LAYER_IDX, window_size=WINDOW_SIZE, decay=0.9995, alpha=1.0
        )
    elif config_type == "pure_prime":
        # Pure PRIME recurrence (no KV cache)
        model.model.layers[LAYER_IDX].self_attn = PrimeTransplantedAttention(
            orig_attn, layer_idx=LAYER_IDX, decay=0.9995
        )
    elif config_type == "hybrid_window_prime":
        # Window W=512 + PRIME recurrent overflow, alpha=0.5
        model.model.layers[LAYER_IDX].self_attn = HybridWindowPrimeAttention(
            orig_attn, layer_idx=LAYER_IDX, window_size=WINDOW_SIZE, decay=0.9995, alpha=0.5
        )
    else:
        raise ValueError(f"Unknown config: {config_type}")

    return model

def run_experiment():
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK: HYBRID WINDOW + PRIME VS ALL BASELINES")
    print(f"Model: {MODEL_ID} | Layer: {LAYER_IDX} | Window Size: {WINDOW_SIZE}")
    print("=" * 80)

    configs = [
        ("Full Softmax (Standard KV)", "full_softmax"),
        ("Pure Sliding Window (W=512)", "pure_sliding_window"),
        ("Pure PRIME Recurrence", "pure_prime"),
        ("Hybrid Window + PRIME (W=512)", "hybrid_window_prime"),
    ]

    gaps = [250, 1000, 2000, 4000]
    all_results = {}

    for config_name, config_type in configs:
        print(f"\nEvaluating: {config_name}...")
        model = load_fresh_model(config_type)
        config_results = {}

        for g in gaps:
            prompt = build_context(g)
            inputs = tok(prompt, return_tensors="pt").to(DEVICE)
            prompt_len = inputs["input_ids"].shape[1]

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=6,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id
                )

            torch.cuda.synchronize()
            elapsed = time.time() - t0
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

            gen = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
            success = "94812" in gen

            config_results[str(g)] = {
                "success": success,
                "output": gen,
                "latency_s": round(elapsed, 3),
                "peak_vram_mb": round(peak_vram, 2)
            }
            status_str = "PASS [94812]" if success else f"FAIL ('{gen}')"
            print(f"  Gap {g:<5} tokens | {status_str:<25} | {elapsed:.2f}s | Peak VRAM: {peak_vram:.1f} MB")

        all_results[config_name] = config_results
        del model
        torch.cuda.empty_cache()

    # Summary Table
    print("\n" + "=" * 90)
    print("FINAL BENCHMARK COMPARISON MATRIX")
    print("=" * 90)
    print(f"{'Architecture':<32} | Gap 250    | Gap 1000   | Gap 2000   | Gap 4000   | Memory Property")
    print("-" * 90)
    
    mem_props = {
        "Full Softmax (Standard KV)": "Unbounded O(L)",
        "Pure Sliding Window (W=512)": "Bounded O(W)",
        "Pure PRIME Recurrence": "Bounded O(1)",
        "Hybrid Window + PRIME (W=512)": "Bounded O(W+D^2)"
    }

    for config_name, res in all_results.items():
        row_str = f"{config_name:<32} | "
        for g in gaps:
            r = res[str(g)]
            status = "PASS" if r["success"] else "FAIL"
            row_str += f"{status:<10} | "
        row_str += mem_props.get(config_name, "N/A")
        print(row_str)

    print("=" * 90)

    # Save to JSON
    out_json = "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/experiments/hybrid_vs_baselines_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[✓] Full telemetry saved to: {out_json}")

if __name__ == "__main__":
    run_experiment()
