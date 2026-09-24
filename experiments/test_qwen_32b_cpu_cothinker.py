#!/usr/bin/env python3
"""
Test Qwen2.5-32B-Instruct on CPU + PRIME-Net Symbolic Co-Thinker
==============================================================
Runs the 32.5-Billion parameter foundation model entirely on CPU RAM (124 GB system)
with multi-threaded acceleration (16 cores) and attaches the PRIME-Net SymPy bridge.
"""

import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prime_moment_attention.primenet_cothinker_bridge import PrimeNetCoThinker

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"

# Configure multi-core CPU parallelism
num_threads = min(16, os.cpu_count() or 8)
torch.set_num_threads(num_threads)
print(f"[*] Configured PyTorch CPU threads: {num_threads}")

print(f"[*] Initializing Tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"[*] Loading 32.5B Parameter Weights on CPU (bfloat16)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True
)
model.eval()
load_time = time.time() - t0

total_params = sum(p.numel() for p in model.parameters())
print(f"[+] Loaded {total_params:,} parameters into CPU memory in {load_time:.2f}s!")

cothinker = PrimeNetCoThinker()

test_queries = [
    ("Hydraulic Power & Equipment Engineering",
     "A heavy excavator operates at a main relief pressure of 4500 PSI with two travel motors flowing a combined 80 GPM. Calculate the hydraulic horsepower using HP = (PSI * GPM) / 1714, and convert to kW (1 HP = 0.7457 kW). Show your exact reasoning inside <think> and write your calculation tags like <<expression>> so our symbolic engine verifies it."),

    ("Advanced Diesel & Hydraulic Troubleshooting",
     "A 35-ton crawler excavator experiences a sudden loss of hydraulic power on both travel tracks and boom functions simultaneously when the hydraulic oil reaches 180°F, but operates normally when cold. As an expert field mechanic, diagnose the most likely root causes."),

    ("Exact Multi-Digit Arithmetic Check",
     "Calculate the total linear thrust of four 6.5-inch diameter hydraulic cylinders operating at 3850 PSI. The area of each cylinder is A = pi * (d/2)^2. Use calculation tags like <<...>> for every step.")
]

print("\n" + "=" * 80)
print("         QWEN2.5-32B-INSTRUCT (CPU) + PRIME-NET CO-THINKER BENCHMARK")
print("=" * 80 + "\n")

for name, query in test_queries:
    print(f"\n--- [{name.upper()}] ---")
    print(f"Technician Query: {query}")

    messages = [
        {"role": "system", "content": "You are a master heavy equipment mechanical engineer and diagnostic specialist. When doing physical equations or math, write your derivation inside <think>...</think> and format every calculation inside <<expression>> so the PRIME-Net symbolic solver can execute the exact math."},
        {"role": "user", "content": query}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    print(f"[*] Generating response on CPU ({num_threads} threads)...")
    gen_start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            top_p=0.9,
            do_sample=True
        )
    gen_duration = time.time() - gen_start
    gen_tokens = out.shape[1] - inputs.input_ids.shape[1]
    tok_per_sec = gen_tokens / max(0.01, gen_duration)

    raw_response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    verified_response, injections = cothinker.intercept_and_solve(raw_response)

    print(f"\nQwen2.5-32B Response ({tok_per_sec:.2f} tok/s | {gen_tokens} tokens in {gen_duration:.1f}s):")
    print(verified_response)
    if injections:
        print(f"\n  [PRIME-Net Exact Symbolic Verifications ({len(injections)})]:")
        for inj in injections:
            print(f"    - {inj['expr']} = {inj['result']}")
    print("-" * 80)

