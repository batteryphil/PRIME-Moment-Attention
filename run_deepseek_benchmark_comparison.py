#!/usr/bin/env python3
"""
DeepSeek Benchmark Comparison: Original vs PRIME + Gist
======================================================
Runs the standard reasoning benchmarks on:
1. Original DeepSeek-R1-Distill-Qwen-7B (Softmax Baseline)
2. Hybrid DeepSeek-R1-Distill-Qwen-7B (PRIME Attention @ L14, Gist Memory @ L20)

Reports accuracy, reasoning traces, token throughput, and VRAM footprint.
"""

import os
import sys
import time
import re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

sys.path.insert(0, 'src')
from prime_moment_attention.hybrid import HybridWindowPrimeAttention
from prime_moment_attention.thought_reconstruction_map import GenerativeThoughtReconstructionLayer

class ZeroInitGistWrapper(nn.Module):
    def __init__(self, orig_layer: Qwen2DecoderLayer, d_map=32, decay=0.995):
        super().__init__()
        self.orig_layer = orig_layer
        self.gist = GenerativeThoughtReconstructionLayer(
            d_model=orig_layer.hidden_size,
            d_map=d_map,
            decay=decay
        )
        nn.init.zeros_(self.gist.recon_proj.weight)
        if self.gist.recon_proj.bias is not None:
            nn.init.zeros_(self.gist.recon_proj.bias)
        device = orig_layer.self_attn.q_proj.weight.device
        self.gist.to(device=device, dtype=torch.bfloat16)

    def forward(self, hidden_states, *args, **kwargs):
        layer_outputs = self.orig_layer(hidden_states, *args, **kwargs)
        orig_hidden = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        gist_out, _ = self.gist(orig_hidden, return_state=False)
        return (gist_out,) + layer_outputs[1:] if isinstance(layer_outputs, tuple) else gist_out

BENCHMARK_SUITE = [
    {
        "id": "logic_boxes",
        "name": "Multi-Step Logic (Three Boxes)",
        "prompt": "You have three closed boxes labeled 'Apples', 'Oranges', and 'Apples and Oranges'. All three labels are incorrect. You can pick only one fruit from one box. How can you determine the correct labels for all three boxes? Name the box you must pick from.",
        "ground_truth_keywords": ["apples and oranges", "both", "mixed"],
    },
    {
        "id": "gsm8k_natalia",
        "name": "GSM8K Arithmetic (Natalia Clips)",
        "prompt": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? State the final number clearly.",
        "ground_truth_keywords": ["72"],
    },
    {
        "id": "prob_hypergeometric",
        "name": "Hypergeometric Probability (10 Balls)",
        "prompt": "A box contains 5 red balls and 5 blue balls. If you draw 4 balls at random without replacement, what is the exact probability that you draw more red balls than blue balls? Express your answer as a simplified fraction.",
        "ground_truth_keywords": ["11/42", "0.2619"],
    },
    {
        "id": "modular_arithmetic",
        "name": "Number Theory (2^2026 + 3^2026 mod 7)",
        "prompt": "What is the remainder when 2^2026 + 3^2026 is divided by 7? State the final remainder as an integer.",
        "ground_truth_keywords": ["6"],
    },
    {
        "id": "bayes_coin",
        "name": "Bayesian Probability (Two-Headed Coin)",
        "prompt": "A bag contains 2 fair coins and 1 two-headed coin (heads on both sides). A coin is chosen at random and flipped 3 times, landing on heads every single time. What is the conditional probability that the chosen coin was the two-headed coin? Express your answer as a simplified fraction.",
        "ground_truth_keywords": ["4/5", "0.8"],
    },
]

def check_answer(response: str, keywords: list) -> bool:
    resp_lower = response.lower()
    for kw in keywords:
        if kw.lower() in resp_lower:
            return True
    return False

def evaluate_model_on_suite(model, tokenizer, mode_name: str, device="cuda:0"):
    print(f"\n{'='*80}")
    print(f"  EVALUATING: {mode_name}")
    print(f"{'='*80}")
    
    results = []
    model.eval()

    for idx, test in enumerate(BENCHMARK_SUITE, 1):
        print(f"\n--- [Problem {idx}/5: {test['name']}] ---")
        prompt_formatted = f"<｜User｜>{test['prompt']}<｜Assistant｜><think>\n"
        input_ids = tokenizer.encode(prompt_formatted, return_tensors="pt").to(device)
        input_len = input_ids.shape[1]

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=350,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
            )
        
        gen_time = time.time() - t0
        gen_tokens = output_ids.shape[1] - input_len
        tok_s = gen_tokens / max(gen_time, 1e-4)
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

        decoded = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=False)
        
        # Parse think trace vs answer
        if "</think>" in decoded:
            parts = decoded.split("</think>", 1)
            think_part = parts[0].strip()
            ans_part = parts[1].strip()
        else:
            think_part = decoded[:200]
            ans_part = decoded

        passed = check_answer(ans_part, test["ground_truth_keywords"]) or check_answer(decoded, test["ground_truth_keywords"])

        print(f"  Throughput: {gen_tokens} tokens in {gen_time:.1f}s ({tok_s:.1f} tok/s) | Peak VRAM: {peak_vram:.2f} GB")
        print(f"  Answer Snippet: {ans_part[:150].replace(chr(10), ' ')}...")
        print(f"  Verdict: {'[PASS]' if passed else '[FAIL]'}")

        results.append({
            "name": test["name"],
            "passed": passed,
            "tokens": gen_tokens,
            "tok_s": tok_s,
            "vram_gb": peak_vram,
            "answer": ans_part[:120].strip(),
        })

    return results

def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    device = "cuda:0"
    
    print(f"[*] Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"[*] Loading {model_name} in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # 1. Evaluate Original Baseline
    baseline_results = evaluate_model_on_suite(
        model, tokenizer, "Original DeepSeek-R1-Distill-Qwen-7B (Baseline Softmax)"
    )

    # 2. Inject PRIME Attention & GIST Layer
    print("\n[*] Transplanting Hybrid PRIME (Layer 14) and Gist Memory (Layer 20)...")
    orig_attn = model.model.layers[14].self_attn
    hybrid_layer = HybridWindowPrimeAttention(
        original_attn=orig_attn,
        layer_idx=14,
        window_size=512,
        decay=0.999,
        alpha=1.0,
    )
    model.model.layers[14].self_attn = hybrid_layer
    model.model.layers[20] = ZeroInitGistWrapper(model.model.layers[20])

    # 3. Evaluate Hybrid Model
    hybrid_results = evaluate_model_on_suite(
        model, tokenizer, "DeepSeek-7B + PRIME Attention (L14) + GIST Memory (L20)"
    )

    # 4. Final Scorecard
    print("\n" + "=" * 95)
    print("  FINAL COMPARISON SCORECARD: ORIGINAL VS. HYBRID PRIME + GIST")
    print("=" * 95)
    print(f"{'Problem Name':<35} | {'Original Baseline':<20} | {'PRIME + Gist':<20}")
    print("-" * 95)

    base_passes = sum(1 for r in baseline_results if r["passed"])
    hybrid_passes = sum(1 for r in hybrid_results if r["passed"])
    base_tok_s = sum(r["tok_s"] for r in baseline_results) / len(baseline_results)
    hybrid_tok_s = sum(r["tok_s"] for r in hybrid_results) / len(hybrid_results)
    base_vram = max(r["vram_gb"] for r in baseline_results)
    hybrid_vram = max(r["vram_gb"] for r in hybrid_results)

    for b, h in zip(baseline_results, hybrid_results):
        b_str = f"{'[PASS]' if b['passed'] else '[FAIL]'} ({b['tok_s']:.1f} tok/s)"
        h_str = f"{'[PASS]' if h['passed'] else '[FAIL]'} ({h['tok_s']:.1f} tok/s)"
        print(f"{b['name']:<35} | {b_str:<20} | {h_str:<20}")

    print("-" * 95)
    print(f"Total Accuracy: Baseline = {base_passes}/5 ({base_passes/5*100:.0f}%) | "
          f"Hybrid PRIME+Gist = {hybrid_passes}/5 ({hybrid_passes/5*100:.0f}%)")
    print(f"Avg Speed:      Baseline = {base_tok_s:.1f} tok/s | Hybrid = {hybrid_tok_s:.1f} tok/s")
    print(f"Peak VRAM:      Baseline = {base_vram:.2f} GB | Hybrid = {hybrid_vram:.2f} GB")
    print("=" * 95 + "\n")

if __name__ == "__main__":
    main()
