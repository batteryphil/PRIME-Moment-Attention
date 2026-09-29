#!/usr/bin/env python3
"""
DeepSeek-R1-Distill-Qwen-7B Baseline Evaluation
==============================================
Profiles memory footprint, inference throughput, and reasoning traces
on local AMD ROCm GPU (17.1 GB VRAM).
"""

import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/data/models/DeepSeek-R1-Distill-Qwen-7B"

def get_vram_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved
    return 0.0, 0.0

def run_baseline_eval():
    print("=" * 80)
    print("  DEEPSEEK-R1-DISTILL-QWEN-7B: BASELINE ROCm GPU BENCHMARK")
    print("=" * 80)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model path not found: {MODEL_PATH}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    vram_before, _ = get_vram_usage()
    print(f"[*] VRAM Allocated Before Load: {vram_before:.2f} GB")

    print("[*] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[*] Loading model weights in bfloat16...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    load_time = time.time() - t0
    vram_loaded, vram_reserved = get_vram_usage()

    print(f"[+] Model loaded in {load_time:.2f}s")
    print(f"[+] VRAM Allocated After Load: {vram_loaded:.2f} GB (Reserved: {vram_reserved:.2f} GB)")

    test_prompts = [
        {
            "name": "Multi-Step Logic (Coin & Boxes)",
            "prompt": "You have three closed boxes labeled 'Apples', 'Oranges', and 'Apples and Oranges'. All three labels are incorrect. You can pick only one fruit from one box. How can you determine the correct labels for all three boxes?",
        },
        {
            "name": "Mathematical Reasoning (GSM8K)",
            "prompt": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        },
    ]

    print("\n" + "=" * 80)
    print("  RUNNING REASONING TRACE EVALUATIONS")
    print("=" * 80)

    model.eval()
    for idx, test in enumerate(test_prompts):
        print(f"\n--- [Test {idx+1}: {test['name']}] ---")
        prompt_text = test["prompt"]
        formatted = f"<｜User｜>{prompt_text}<｜Assistant｜><think>\n"
        
        input_ids = tokenizer.encode(formatted, return_tensors="pt").to(device)
        input_len = input_ids.shape[1]

        t_start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=250,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_time = time.time() - t_start

        gen_tokens = output_ids.shape[1] - input_len
        tok_per_sec = gen_tokens / max(gen_time, 1e-4)

        full_output = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=False)

        # Separate <think> trace from final answer
        has_think = "</think>" in full_output
        if has_think:
            parts = full_output.split("</think>", 1)
            think_trace = parts[0].strip()
            answer_text = parts[1].strip()
        else:
            think_trace = full_output
            answer_text = "[No explicit closing tag]"

        print(f"Generated {gen_tokens} tokens in {gen_time:.2f}s ({tok_per_sec:.1f} tok/s)")
        print(f"\n[Reasoning Trace (<think>...)]:\n{think_trace[:400]}...\n")
        print(f"[Final Answer]:\n{answer_text[:300]}...")

    vram_peak, vram_peak_reserved = get_vram_usage()
    print("\n" + "=" * 80)
    print("  RESOURCE & MEMORY FOOTPRINT SUMMARY")
    print("=" * 80)
    print(f"Model: DeepSeek-R1-Distill-Qwen-7B (bfloat16)")
    print(f"VRAM Allocated: {vram_loaded:.2f} GB / 17.1 GB ({(vram_loaded / 17.1) * 100:.1f}%)")
    print(f"Peak VRAM During Generation: {vram_peak:.2f} GB (Reserved: {vram_peak_reserved:.2f} GB)")
    print(f"Free VRAM Available: {17.1 - vram_peak_reserved:.2f} GB")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_baseline_eval()
