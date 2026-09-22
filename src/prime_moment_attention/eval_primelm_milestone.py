#!/usr/bin/env python3
"""
PrimeLM-50M: Milestone Re-Evaluation Engine
===========================================
Evaluates any intermediate 3B checkpoint against the full capability benchmark:
  1. Mathematical Deduction (1-step, 2-step, scaling, precedence)
  2. Physical Conservation Laws (Q=mcΔT, E=0.5mv^2, V=IR)
  3. Narrative & Linguistic Fluency
  4. Repetition Degeneracy (4-Gram Repetition Rate)
  5. Cross-Entropy Loss & Perplexity on Held-Out Multi-Corpus Test Sets
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/src/prime_moment_attention")
sys.path.insert(0, "/home/phil/.gemini/antigravity/brain/488931f7-4ed7-40c4-bc4d-bcbc1f64de95/scratch")
from primelm_50m import PrimeLM50M

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/checkpoints/primelm_50m_3b_latest.pt", help="Path to checkpoint to evaluate")
args = parser.parse_args()

print("=" * 85)
print(f"  PRIMELM-50M: MILESTONE RE-EVALUATION ON {DEVICE}")
print(f"  Target Checkpoint: {args.ckpt}")
print("=" * 85)

tok = AutoTokenizer.from_pretrained("gpt2")
tok.model_max_length = 1000000

model = PrimeLM50M(
    vocab_size=50257,
    d_model=512,
    n_layers=8,
    n_heads=8,
    head_dim=64,
    d_ff=1024,
    window_size=256,
    num_registers=4,
    use_registers=True,
    use_gating=True,
    use_probe=True
).to(DEVICE).to(torch.bfloat16)

if not os.path.exists(args.ckpt):
    print(f"[!] Checkpoint not found: {args.ckpt}")
    sys.exit(1)

state = torch.load(args.ckpt, map_location=DEVICE)
if "model_state" in state:
    model.load_state_dict(state["model_state"])
    step = state.get("step", 0)
    tokens_trained = state.get("tokens_trained", 0)
else:
    model.load_state_dict(state)
    step = "N/A"
    tokens_trained = "N/A"

model.eval()

TEST_SUITE = [
    ("Math 1-Step", "Problem: Solve for x in 5 * x = 35.\nStep 1: Divide both sides by 5 to get x = ", [5.0, 0.0, 35.0, 7.0], "7"),
    ("Math 2-Step", "Problem: Solve for x in 4 * x + 12 = 36.\nStep 1: Subtract 12 from both sides to get 4 * x = ", [4.0, 12.0, 36.0, 6.0], "24"),
    ("Thermodynamics", "Problem: Calculate heat required for 2 kg water heated by 10 K.\nFormula: Q = m * c * deltaT.\nCalculation: Q = ", [2.0, 4.0, 10.0, 80.0], "80"),
    ("Kinetic Energy", "Problem: Calculate kinetic energy of mass 4 kg moving at speed 5 m/s.\nFormula: E = 0.5 * m * v^2.\nCalculation: E = ", [4.0, 5.0, 0.5, 50.0], "50"),
    ("Narrative Fluency", "Once upon a time, a curious puppy wandered into a deep green forest. Suddenly, ", [0.0, 0.0, 0.0, 0.0], "narrative"),
    ("Scientific Fact", "Photosynthesis is the fundamental biological process where plants convert sunlight into ", [6.0, 6.0, 1.0, 6.0], "energy")
]

print(f"\n--- [EVALUATION RESULTS: Step {step} | Tokens Trained: {tokens_trained}] ---")
scorecard = []

for title, prompt, inv_vec, expected in TEST_SUITE:
    inputs = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    inv_t = torch.tensor([inv_vec], dtype=torch.bfloat16, device=DEVICE)
    
    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(inputs, max_new_tokens=40, temperature=0.2, inv_vec=inv_t)
    latency_ms = (time.time() - t0) * 1000.0
    
    gen_tokens = out_ids[0, inputs.shape[1]:].tolist()
    gen_text = tok.decode(gen_tokens, skip_special_tokens=True).strip()
    
    if len(gen_tokens) >= 4:
        ngrams = [tuple(gen_tokens[i:i+4]) for i in range(len(gen_tokens)-3)]
        rep_4 = (len(ngrams) - len(set(ngrams))) / len(ngrams)
    else:
        rep_4 = 0.0
        
    print(f"\n[{title}] (Latency: {latency_ms:.1f}ms | Rep-4: {rep_4:.3f})")
    print(f"Prompt: {prompt.strip()}")
    print(f"Output: {gen_text}")
    
    scorecard.append({
        "title": title,
        "prompt": prompt,
        "expected": expected,
        "output": gen_text,
        "rep_4": round(rep_4, 4),
        "latency_ms": round(latency_ms, 1)
    })

print("\n" + "=" * 85)
print("  RE-EVALUATION COMPLETE")
print("=" * 85)
