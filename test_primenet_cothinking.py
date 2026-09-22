"""
Test PRIME-Net Co-Thinker in the Thinking Phase of PRIME-125M
============================================================
"""

import os
import sys
from transformers import AutoTokenizer

export_dir = "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/export_hf/prime-125m-reasoning"
sys.path.insert(0, export_dir)
from modeling_prime import PrimeForCausalLM

print("[*] Loading model from export_hf...")
model = PrimeForCausalLM.from_pretrained(export_dir)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(export_dir)

test_cases = [
    # 1. Multi-step word problem with arithmetic
    "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. How many eggs are left?",
    # 2. Money / House flipping problem
    "Josh buys a house for 80000 dollars and puts in 50000 dollars in repairs. How much did he spend in total?",
    # 3. Direct calculation
    "Calculate 25 * 16.",
    # 4. Pattern / Invariant discovery
    "Find the rule for the sequence: 1, 4, 9, 16."
]

print("\n" + "=" * 75)
print("  PRIME-125M + PRIME-Net Co-Thinker Test")
print("=" * 75)

for q in test_cases:
    print(f"\n>>> QUESTION: {q}")
    # Prime with the CoT opening the model was fine-tuned on
    prompt = f"User: {q}\n\nAssistant: <think>\nLet's analyze this step-by-step:\n"
    
    out = model.generate_with_primenet(
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=200,
        temperature=0.5,
        top_k=40,
        verbose=True
    )
    
    print("\n--- OUTPUT ---")
    if "Assistant:" in out:
        asst = out.split("Assistant:", 1)[1].strip()
        print(asst)
    else:
        print(out)
    print("-" * 75)
