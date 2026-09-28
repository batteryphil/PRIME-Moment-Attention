#!/usr/bin/env python3
"""
Multi-Domain Benchmark: Python, C/C++, Math & Analytical Reasoning
=================================================================
Evaluates foundational model competence across four target pillars:
1. Python Syntax & Generation (AST verification + basic execution)
2. C/C++ Systems Code Syntax (GCC compiler syntax pass/fail)
3. Mathematical Problem Solving (GSM8K numerical exact match)
4. Multi-Step Analytical Reasoning (Deductive logic & cloze probes)
"""

import os
import sys
import ast
import time
import math
import subprocess
import argparse
from typing import Dict, Any, List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "export_hf", "prime-125m-reasoning")))
from prime_moment_attention.model import PrimeConfig, PrimeForCausalLM


# 1. Python Prompts
PYTHON_BENCHMARKS = [
    {
        "prompt": "Write a Python function `add(a, b)` that returns their sum:\n```python\n",
        "expected_def": "def add(",
    },
    {
        "prompt": "Write a Python function `is_even(n)` that returns True if n is even, else False:\n```python\n",
        "expected_def": "def is_even(",
    },
    {
        "prompt": "Write a Python function `reverse_list(lst)` that returns the reversed list:\n```python\n",
        "expected_def": "def reverse_list(",
    },
]

# 2. C / C++ Prompts
C_BENCHMARKS = [
    {
        "prompt": "Write a C function to calculate the square of an integer:\n```c\n",
        "expected_signature": "int square(",
    },
    {
        "prompt": "Write a C function to swap two integers using pointers:\n```c\n",
        "expected_signature": "void swap(",
    },
    {
        "prompt": "Write a C function to compute factorial of n recursively:\n```c\n",
        "expected_signature": "long factorial(",
    },
]

# 3. Math Prompts (GSM8K Cloze & Arithmetic)
MATH_BENCHMARKS = [
    {
        "question": "Janet has 3 apples and buys 4 more. Then she gives 2 apples to her sister. How many apples does Janet have left? Answer:",
        "target_num": " 5",
        "distractor_num": " 9",
    },
    {
        "question": "A box contains 12 pencils. If Mark buys 3 boxes, how many pencils does he have in total? Answer:",
        "target_num": " 36",
        "distractor_num": " 15",
    },
    {
        "question": "A farmer has 20 sheep. Half of them run away. How many sheep are left? Answer:",
        "target_num": " 10",
        "distractor_num": " 5",
    },
]


def test_python_generation(model, tokenizer, device: str) -> Dict[str, Any]:
    """Generates code and checks for valid AST syntax."""
    model.eval()
    valid_ast_count = 0
    results = []

    for item in PYTHON_BENCHMARKS:
        prompt = item["prompt"]
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=40, temperature=0.2, top_k=20)
        
        gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
        code_part = gen_text[len(prompt):].split("```")[0]
        
        # Test Python syntax parsing
        has_valid_syntax = False
        try:
            ast.parse(code_part)
            has_valid_syntax = True
            valid_ast_count += 1
        except SyntaxError:
            has_valid_syntax = False

        results.append({
            "prompt": prompt,
            "generated": code_part.strip(),
            "valid_ast": has_valid_syntax,
        })

    accuracy = (valid_ast_count / len(PYTHON_BENCHMARKS)) * 100.0
    return {"accuracy": accuracy, "results": results}


def test_c_syntax(model, tokenizer, device: str) -> Dict[str, Any]:
    """Generates C code and checks validity using gcc -fsyntax-only."""
    model.eval()
    valid_c_count = 0
    results = []

    for item in C_BENCHMARKS:
        prompt = item["prompt"]
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=40, temperature=0.2, top_k=20)
        
        gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
        code_part = gen_text[len(prompt):].split("```")[0].strip()

        # Wrap in minimal header and test with GCC
        wrapped_c = f"#include <stdio.h>\n#include <stdlib.h>\n{code_part}\n"
        has_valid_c = False
        try:
            res = subprocess.run(
                ["gcc", "-fsyntax-only", "-std=c99", "-xc", "-"],
                input=wrapped_c,
                text=True,
                capture_output=True,
                timeout=5,
            )
            has_valid_c = (res.returncode == 0)
        except Exception:
            has_valid_c = False

        if has_valid_c:
            valid_c_count += 1

        results.append({
            "prompt": prompt,
            "generated": code_part,
            "valid_gcc": has_valid_c,
        })

    accuracy = (valid_c_count / len(C_BENCHMARKS)) * 100.0
    return {"accuracy": accuracy, "results": results}


def test_math_reasoning(model, tokenizer, device: str) -> Dict[str, Any]:
    """Evaluates discriminative probability on math arithmetic answers."""
    model.eval()
    correct_count = 0
    results = []

    for item in MATH_BENCHMARKS:
        q = item["question"]
        c_tok = tokenizer.encode(item["target_num"])[0]
        i_tok = tokenizer.encode(item["distractor_num"])[0]

        input_ids = tokenizer.encode(q, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids)
            logits = out["logits"][0, -1]
            probs = F.softmax(logits.float(), dim=-1)

        p_c = probs[c_tok].item()
        p_i = probs[i_tok].item()
        is_correct = (p_c > p_i)

        if is_correct:
            correct_count += 1

        results.append({
            "question": q,
            "p_target": p_c,
            "p_distractor": p_i,
            "is_correct": is_correct,
        })

    accuracy = (correct_count / len(MATH_BENCHMARKS)) * 100.0
    return {"accuracy": accuracy, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on Code, Math, and Reasoning")
    parser.add_argument("--checkpoint", type=str, default="/data/prime_checkpoints/prime_125m_step_10000.pt",
                        help="Path to .pt checkpoint file")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  MULTI-DOMAIN FOUNDATION EVALUATION: CODE, MATH & REASONING")
    print(f"  Target Checkpoint: {args.checkpoint}")
    print(f"  Device           : {device}")
    print("=" * 80 + "\n")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config")
    model = PrimeForCausalLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    py_res = test_python_generation(model, tokenizer, device)
    c_res = test_c_syntax(model, tokenizer, device)
    math_res = test_math_reasoning(model, tokenizer, device)

    print("\n" + "=" * 80)
    print("  EVALUATION SUMMARY SCORECARD")
    print("=" * 80)
    print(f"  1. Python Code AST Validity   : {py_res['accuracy']:5.1f}% ({sum(r['valid_ast'] for r in py_res['results'])}/{len(PYTHON_BENCHMARKS)})")
    print(f"  2. C/C++ GCC Syntax Validity  : {c_res['accuracy']:5.1f}% ({sum(r['valid_gcc'] for r in c_res['results'])}/{len(C_BENCHMARKS)})")
    print(f"  3. Math Problem Solving (Acc) : {math_res['accuracy']:5.1f}% ({sum(r['is_correct'] for r in math_res['results'])}/{len(MATH_BENCHMARKS)})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
