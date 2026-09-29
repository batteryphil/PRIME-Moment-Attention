#!/usr/bin/env python3
"""
PRIME-Net Neuro-Symbolic Math Harness Benchmark
===============================================
Evaluates raw unassisted language model generation vs. PRIME-Net assisted generation
across multi-domain mathematical reasoning challenges:
1. Arithmetic Word Problems (GSM8K)
2. Percentages & Rate Calculations
3. Algebraic Equation Solving
4. Multi-Item Unit Pricing
5. Closed-Form Sequence Invariant Discovery
"""

import os
import sys
import time
import re
import argparse
from typing import Dict, List, Any, Tuple
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

# Ensure src is on python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from prime_moment_attention.primenet_harness import PrimeNetMathHarness


# Benchmark Challenge Suite
MATH_BENCHMARK_SUITE = [
    # 1. Multi-Step Arithmetic (GSM8K style)
    {
        "category": "Arithmetic Word Problems",
        "prompt": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "ground_truth": "72",
        "formula": "48 + 48 / 2",
    },
    {
        "category": "Arithmetic Word Problems",
        "prompt": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "ground_truth": "10",
        "formula": "12 * 50 / 60",
    },
    {
        "category": "Arithmetic Word Problems",
        "prompt": "Betty picked 16 apples. Her brother picked 8 apples. They gave 4 apples to their neighbor. How many apples do they have left?",
        "ground_truth": "20",
        "formula": "16 + 8 - 4",
    },
    {
        "category": "Arithmetic Word Problems",
        "prompt": "A bakery had 80 cakes. They sold 35 cakes in the morning and 25 cakes in the afternoon. How many cakes are remaining?",
        "ground_truth": "20",
        "formula": "80 - 35 - 25",
    },

    # 2. Percentages & Discounts
    {
        "category": "Percentages & Rates",
        "prompt": "What is 20 percent of 150?",
        "ground_truth": "30",
        "formula": "0.20 * 150",
    },
    {
        "category": "Percentages & Rates",
        "prompt": "What is 25 percent of 200?",
        "ground_truth": "50",
        "formula": "0.25 * 200",
    },
    {
        "category": "Percentages & Rates",
        "prompt": "What is 15 percent of 80?",
        "ground_truth": "12",
        "formula": "0.15 * 80",
    },
    {
        "category": "Percentages & Rates",
        "prompt": "A car travels 180 miles in 3 hours. Calculate 180 / 3 for speed in miles per hour.",
        "ground_truth": "60",
        "formula": "180 / 3",
    },

    # 3. Algebraic Equation Solving
    {
        "category": "Algebraic Equations",
        "prompt": "Solve for x: 3 * x + 7 = 22",
        "ground_truth": "5",
        "formula": "(22 - 7) / 3",
    },
    {
        "category": "Algebraic Equations",
        "prompt": "Solve for x: 2 * x + 10 = 30",
        "ground_truth": "10",
        "formula": "(30 - 10) / 2",
    },
    {
        "category": "Algebraic Equations",
        "prompt": "Solve for x: 5 * x - 15 = 35",
        "ground_truth": "10",
        "formula": "(35 + 15) / 5",
    },
    {
        "category": "Algebraic Equations",
        "prompt": "Solve for x: 4 * x - 8 = 24",
        "ground_truth": "8",
        "formula": "(24 + 8) / 4",
    },

    # 4. Multi-Item Unit Pricing
    {
        "category": "Multi-Item Pricing",
        "prompt": "A customer bought 3 books for 15 dollars each and 2 pens for 4 dollars each. What is the total cost?",
        "ground_truth": "53",
        "formula": "3 * 15 + 2 * 4",
    },
    {
        "category": "Multi-Item Pricing",
        "prompt": "James bought 4 sandwiches at $6 each and 3 sodas at $2 each. How much did he spend in total?",
        "ground_truth": "30",
        "formula": "4 * 6 + 3 * 2",
    },

    # 5. Sequence Invariants
    {
        "category": "Sequence Invariants",
        "prompt": "Identify the mathematical rule for the sequence: 1, 4, 9, 16, 25.",
        "ground_truth": "n**2",
        "formula": "n**2",
    },
    {
        "category": "Sequence Invariants",
        "prompt": "Identify the mathematical rule for the sequence: 2, 4, 6, 8, 10.",
        "ground_truth": "2*n",
        "formula": "2*n",
    },
]


def extract_numbers_or_symbols(text: str) -> List[str]:
    """Extract numbers and mathematical variable expressions from generated text."""
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    # Also look for algebraic targets like n**2 or x = 5
    rules = re.findall(r"n\*\*\d+|x\s*=\s*\d+|\d+\*n", text)
    return nums + rules


def evaluate_raw_model(model, tokenizer, prompt: str, device: str) -> str:
    """Generate response using raw model without any symbolic assistance."""
    input_ids = tokenizer.encode(f"Question: {prompt}\nAnswer:", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=60,
            temperature=0.2,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


def evaluate_harness_model(harness: PrimeNetMathHarness, model, tokenizer, prompt: str, device: str) -> Tuple[str, float]:
    """Generate response using PRIME-Net Math Harness with live symbolic interception."""
    t0 = time.perf_counter()
    full_prompt = f"Question: {prompt}\nAnswer:"
    text = harness.generate_with_harness(
        model=model,
        tokenizer=tokenizer,
        prompt=full_prompt,
        max_new_tokens=80,
        temperature=0.2,
        top_k=20,
        device=device,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return text, latency_ms


def check_answer_correct(text: str, target: str) -> bool:
    """Checks if target ground truth appears in the output reasoning or final answer."""
    clean_text = text.replace(",", "").replace("$", "")
    target_clean = target.replace(",", "").replace("$", "").strip()

    # Exact rule match (e.g. n**2)
    if "**" in target_clean or "*n" in target_clean:
        return target_clean in clean_text

    # Number match
    tokens = re.findall(r"[-+]?\d*\.\d+|\d+", clean_text)
    return target_clean in tokens


def run_benchmark(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    print("=" * 95)
    print("  PRIME-Net Neuro-Symbolic Math Harness Benchmark")
    print(f"  Model: Stock GPT-2 (124M) | Device: {device}")
    print("  Testing Raw Unassisted Generation vs. PRIME-Net Symbolic Co-Thinking")
    print("=" * 95 + "\n")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    harness = PrimeNetMathHarness(verbose=False)

    results = []
    category_stats = {}

    total_harness_latency = 0.0

    for idx, tc in enumerate(MATH_BENCHMARK_SUITE):
        cat = tc["category"]
        prompt = tc["prompt"]
        target = tc["ground_truth"]

        if cat not in category_stats:
            category_stats[cat] = {"raw_pass": 0, "harness_pass": 0, "total": 0}
        category_stats[cat]["total"] += 1

        # 1. Raw Generation
        raw_out = evaluate_raw_model(model, tokenizer, prompt, device)
        raw_pass = check_answer_correct(raw_out, target)
        if raw_pass:
            category_stats[cat]["raw_pass"] += 1

        # 2. PRIME-Net Assisted Generation
        harness_out, lat_ms = evaluate_harness_model(harness, model, tokenizer, prompt, device)
        total_harness_latency += lat_ms
        harness_pass = check_answer_correct(harness_out, target)
        if harness_pass:
            category_stats[cat]["harness_pass"] += 1

        results.append({
            "idx": idx + 1,
            "category": cat,
            "target": target,
            "raw_pass": raw_pass,
            "harness_pass": harness_pass,
            "latency_ms": lat_ms,
        })

        status_str = f"PASS ({target})" if harness_pass else f"FAIL (want {target})"
        raw_str = "PASS" if raw_pass else "FAIL"
        print(f"[{idx+1:02d}/{len(MATH_BENCHMARK_SUITE):02d}] {cat:<24} | Raw: {raw_str:<4} | PRIME-Net: {status_str:<18} | Latency: {lat_ms:.1f}ms")

    # Summary Scorecard
    total_q = len(MATH_BENCHMARK_SUITE)
    total_raw_passed = sum(1 for r in results if r["raw_pass"])
    total_harness_passed = sum(1 for r in results if r["harness_pass"])

    raw_acc = (total_raw_passed / total_q) * 100.0
    harness_acc = (total_harness_passed / total_q) * 100.0
    avg_lat = total_harness_latency / total_q

    print("\n" + "=" * 95)
    print("  FINAL COMPARATIVE ACCURACY SCORECARD")
    print("=" * 95)
    print(f"{'Category':<28} | {'Problems':<10} | {'Raw 124M Accuracy':<20} | {'PRIME-Net Assisted':<20}")
    print("-" * 95)
    for cat, stats in category_stats.items():
        r_acc = (stats["raw_pass"] / stats["total"]) * 100.0
        h_acc = (stats["harness_pass"] / stats["total"]) * 100.0
        print(f"{cat:<28} | {stats['total']:<10} | {stats['raw_pass']}/{stats['total']} ({r_acc:.1f}%)        | {stats['harness_pass']}/{stats['total']} ({h_acc:.1f}%)")
    print("-" * 95)
    print(f"{'OVERALL TOTAL':<28} | {total_q:<10} | {total_raw_passed}/{total_q} ({raw_acc:.1f}%)        | {total_harness_passed}/{total_q} ({harness_acc:.1f}%)")
    print(f"Average Generation + Symbolic Verification Latency: {avg_lat:.2f} ms / problem")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRIME-Net Math Harness Benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_benchmark(device=args.device)
