#!/usr/bin/env python3
"""
Test Falcon3-10B-Instruct-1.58bit + PRIME-Net Symbolic Co-Thinker
=================================================================
Evaluates the 10-Billion parameter 1.58-bit model on:
  1. Hydraulic Cylinder Force (F = P * A)
  2. Hydraulic Pump Cavitation & Troubleshooting
  3. Horsepower from Torque (HP = Torque * RPM / 5252)
  4. Complex Multi-Step Mechanic Algebra
"""

import os
import sys

scratch_dir = "/home/phil/.gemini/antigravity/scratch/.triton"
os.environ["TRITON_HOME"] = scratch_dir
os.environ["TRITON_CACHE_DIR"] = os.path.join(scratch_dir, "cache")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(scratch_dir, "inductor")
os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(scratch_dir, "extensions")
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prime_moment_attention.primenet_cothinker_bridge import PrimeNetCoThinker

MODEL_ID = "tiiuae/Falcon3-10B-Instruct-1.58bit"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"[*] Initializing Tokenizer & Model for {MODEL_ID} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    trust_remote_code=True
)
model.eval()

cothinker = PrimeNetCoThinker()
vram = torch.cuda.memory_allocated() / (1024**3)
print(f"[+] Loaded Falcon3-10B (1.58-bit) successfully! VRAM: {vram:.2f} GB")

test_cases = [
    ("Hydraulic Cylinder Force",
     "A hydraulic cylinder on an excavator has a piston area of 5 square inches operating at 2000 PSI. Calculate the force produced using F = P * A. Show your work inside <think> and write your calculation tags like <<pressure * area>> so our symbolic engine verifies it."),
     
    ("Mechanical Troubleshooting",
     "An excavator hydraulic pump is making a loud whining noise and the boom moves sluggishly. As an expert heavy equipment mechanic, what are the top 3 diagnostic checks you would perform and why?"),
     
    ("Horsepower Physics Calculation",
     "A diesel engine produces 350 ft-lbs of torque at 2100 RPM. Calculate the horsepower using HP = (Torque * RPM) / 5252. Show your exact calculation steps inside <think> with <<...>> tags.")
]

print("\n" + "=" * 80)
print("         FALCON3-10B-1.58BIT + PRIME-NET CO-THINKER BENCHMARK")
print("=" * 80 + "\n")

for name, query in test_cases:
    print(f"\n--- [{name.upper()}] ---")
    print(f"User Query: {query}")
    
    messages = [
        {"role": "system", "content": "You are an expert heavy equipment master technician and diagnostic assistant. When solving physical equations or arithmetic, reason inside <think>...</think> and use calculation tags like <<expression>> so the PRIME-Net symbolic solver can execute the exact mathematics."},
        {"role": "user", "content": query}
    ]
    
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"System: Master Technician Assistant\nUser: {query}\nAssistant: "
        
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.3,
            top_p=0.9,
            do_sample=True
        )
        
    raw_output = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    verified_output, injections = cothinker.intercept_and_solve(raw_output)
    
    print(f"\nFalcon3-10B Output:")
    print(verified_output)
    if injections:
        print(f"\n  [PRIME-Net Symbolic Injections Verified]: {injections}")
    print("-" * 80)

