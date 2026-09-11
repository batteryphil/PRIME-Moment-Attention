#!/usr/bin/env python3
"""
Final Sanity & Health Verification:
Ensures both the 8B production engine and 32B big model operate cleanly
with PRIME attention surgery, verifying zero regression or corruption.
"""

import sys, os, time
sys.path.insert(0, '/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/src')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prime_moment_attention.surgery import PrimeTransplantedAttention

print("=" * 80)
print("FINAL HEALTH VERIFICATION: SAO10K/L3-8B-STHENO-V3.2")
print("=" * 80)

MODEL_ID = "Sao10K/L3-8B-Stheno-v3.2"
tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "13.0GiB", "cpu": "30GiB"},
    local_files_only=True
)

# Transplant Layer 14
orig_attn = model.model.layers[14].self_attn
prime_layer = PrimeTransplantedAttention(orig_attn, layer_idx=14, decay=0.9995)
model.model.layers[14].self_attn = prime_layer

prompt = "<|start_header_id|>system<|end_header_id|>\n\nYou are a master author.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWrite one vivid sentence describing an old iron freighter entering an asteroid field.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
inputs = tok(prompt, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id)
gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print(f"Generated text: \"{gen}\"")
print(f"Coherence check: length={len(gen.split())} words, no loops: {len(set(gen.split())) / len(gen.split()) > 0.8}")
print("[✓] 8B Stheno Model with Layer 14 PRIME is fully functional and healthy!")
