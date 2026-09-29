#!/usr/bin/env python3
"""
DeepSeek-R1-Distill-Qwen-7B: Shootout Benchmark
===============================================
Compares:
1. Natural Thinking (min_think_tokens=0, standard single-shot deliberation)
2. Extended Deliberation + Early-Exit Consensus + PRIME-Net Harness
   (min_think_tokens=800, N=3 rollouts with early exit at majority)

Across 3 challenging competition-level mathematical problems:
- Problem 1: Hypergeometric Probability (10 balls, draw 4 without replacement)
- Problem 2: Modular Arithmetic / Number Theory (2^2026 + 3^2026 mod 7)
- Problem 3: Bayesian Conditional Probability (Fair coins vs Two-headed coin)
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Ensure PRIME-Moment-Attention root and src are on PYTHONPATH
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from benchmarks.eval_deepseek_7b_consensus import (
    MODEL_PATH,
    generate_reasoning_rollout,
    cluster_votes,
    symbolic_equal,
    ForcedThinkingBudgetLogitsProcessor,
)
from prime_moment_attention.primenet_harness import PrimeNetMathHarness


BENCHMARK_PROBLEMS = [
    {
        "id": "prob_1_hypergeometric",
        "name": "Hypergeometric Probability (10 Balls)",
        "prompt": (
            "A box contains 5 red balls and 5 blue balls. If you draw 4 balls at random without replacement, "
            "what is the exact probability that you draw more red balls than blue balls? "
            "Express your answer as a simplified fraction."
        ),
        "ground_truth": "11/42",
    },
    {
        "id": "prob_2_modular_arithmetic",
        "name": "Modular Arithmetic (2^2026 + 3^2026 mod 7)",
        "prompt": (
            "What is the remainder when 2^2026 + 3^2026 is divided by 7? "
            "Show your work and state the final remainder as an integer."
        ),
        "ground_truth": "6",
    },
    {
        "id": "prob_3_bayes_coin",
        "name": "Bayesian Coin Selection (2 Fair, 1 Two-Headed)",
        "prompt": (
            "A bag contains 2 fair coins and 1 two-headed coin (with heads on both sides). "
            "A coin is chosen at random and flipped 3 times, landing on heads every single time. "
            "What is the exact conditional probability that the coin chosen was the two-headed coin? "
            "Express your answer as a simplified fraction."
        ),
        "ground_truth": "4/5",
    },
]


def run_shootout(device: str = "cuda", max_rollouts: int = 3):
    print("=" * 95)
    print("  DEEPSEEK-R1-DISTILL-QWEN-7B: NATURAL THINKING VS. EXTENDED DELIBERATION SHOOTOUT")
    print(f"  Target Device: {device} | 4-bit NF4 Quantization | Max Rollouts: {max_rollouts}")
    print("=" * 95)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[*] Loading model in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": device},
    )
    model.eval()

    harness = PrimeNetMathHarness()
    results = []

    for idx, item in enumerate(BENCHMARK_PROBLEMS, 1):
        pid = item["id"]
        name = item["name"]
        prompt = item["prompt"]
        gt = item["ground_truth"]

        print("\n" + "#" * 95)
        print(f"  PROBLEM {idx}/{len(BENCHMARK_PROBLEMS)}: {name}")
        print(f"  Expected Ground Truth: {gt}")
        print("#" * 95)

        # -----------------------------------------------------------------
        # 1. Natural Thinking (Unforced: min_think_tokens = 0)
        # -----------------------------------------------------------------
        print(f"\n>>> [Mode A: Natural Thinking (min_think_tokens=0)] Running single-shot...")
        t0 = time.time()
        res_natural = generate_reasoning_rollout(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            min_think_tokens=0,
            max_think_tokens=1000,
            max_answer_tokens=400,
            temperature=0.6,
            device=device,
        )
        time_natural = time.time() - t0
        ans_natural = res_natural["extracted_answer"]
        pass_natural = symbolic_equal(ans_natural, gt)
        print(f"  • Thinking Tokens: {res_natural['think_tokens']} tokens in {time_natural:.1f}s")
        print(f"  • Extracted Answer: {ans_natural}")
        print(f"  • Pass/Fail: {'[PASS]' if pass_natural else '[FAIL]'}")

        # -----------------------------------------------------------------
        # 2. Extended Deliberation + Early-Exit Consensus + PRIME-Net
        # -----------------------------------------------------------------
        print(f"\n>>> [Mode B: Extended Deliberation (min_think_tokens=800) + Early-Exit Consensus]...")
        # Check PRIME-Net Invariants
        injected = harness.extract_prompt_invariants(prompt)
        eval_prompt = f"{injected}\n{prompt}" if injected else prompt
        if injected:
            print(f"  [*] PRIME-Net Invariant Injected: {injected}")

        votes_extended = []
        extended_rollouts = []
        t0_ext = time.time()
        early_exited = False
        majority_threshold = (max_rollouts // 2) + 1

        for r in range(max_rollouts):
            print(f"  --- Rollout {r+1}/{max_rollouts} (Forced Deliberation >= 800 tokens)... ---")
            res_ext = generate_reasoning_rollout(
                model=model,
                tokenizer=tokenizer,
                prompt=eval_prompt,
                min_think_tokens=800,
                max_think_tokens=1200,
                max_answer_tokens=400,
                temperature=0.7,
                device=device,
            )
            extended_rollouts.append(res_ext)
            ans_ext = res_ext["extracted_answer"]
            votes_extended.append(ans_ext)
            print(f"      Think Tokens: {res_ext['think_tokens']} | Extracted: {ans_ext}")
            if res_ext.get("answer_text"):
                print(f"      Answer Snippet: {res_ext['answer_text'][:120].strip()}...")

            # Early-Exit Check
            if len(votes_extended) >= 2:
                clusters = cluster_votes(votes_extended)
                top_cand, top_cnt = clusters.most_common(1)[0]
                if top_cand is not None and top_cnt >= majority_threshold:
                    print(f"      >>> [EARLY EXIT]: Candidate '{top_cand}' reached majority ({top_cnt}/{max_rollouts})!")
                    early_exited = True
                    break

        time_extended = time.time() - t0_ext
        vote_clusters = cluster_votes(votes_extended)
        winner, win_count = vote_clusters.most_common(1)[0]
        pass_extended = symbolic_equal(winner, gt)
        print(f"  • Consensus Decision: {winner} ({win_count}/{len(votes_extended)} votes)")
        print(f"  • Pass/Fail: {'[PASS]' if pass_extended else '[FAIL]'}")
        print(f"  • Rollouts used: {len(votes_extended)}/{max_rollouts} | Total Time: {time_extended:.1f}s")

        results.append({
            "name": name,
            "ground_truth": gt,
            "natural": {
                "answer": ans_natural,
                "pass": pass_natural,
                "think_tokens": res_natural["think_tokens"],
                "time_sec": time_natural,
            },
            "extended": {
                "answer": winner,
                "pass": pass_extended,
                "rollouts": len(votes_extended),
                "early_exited": early_exited,
                "time_sec": time_extended,
                "votes": dict(vote_clusters),
            },
        })

    # -----------------------------------------------------------------
    # Summary Shootout Scorecard
    # -----------------------------------------------------------------
    print("\n" + "=" * 95)
    print("  FINAL SHOOTOUT SCORECARD: NATURAL THINKING VS. EXTENDED DELIBERATION")
    print("=" * 95)
    print(f"{'Problem Name':<35} | {'Ground Truth':<12} | {'Natural (0 Budget)':<20} | {'Extended (Consensus)':<22}")
    print("-" * 95)

    nat_correct = 0
    ext_correct = 0
    for r in results:
        nat_str = f"{str(r['natural']['answer']):<8} {'[PASS]' if r['natural']['pass'] else '[FAIL]'}"
        ext_str = f"{str(r['extended']['answer']):<8} {'[PASS]' if r['extended']['pass'] else '[FAIL]'}"
        if r['extended']['early_exited']:
            ext_str += f" (N={r['extended']['rollouts']})"
        print(f"{r['name']:<35} | {r['ground_truth']:<12} | {nat_str:<20} | {ext_str:<22}")
        if r['natural']['pass']:
            nat_correct += 1
        if r['extended']['pass']:
            ext_correct += 1

    print("-" * 95)
    print(f"FINAL ACCURACY: Natural = {nat_correct}/{len(results)} ({nat_correct/len(results)*100:.1f}%) | "
          f"Extended Deliberation = {ext_correct}/{len(results)} ({ext_correct/len(results)*100:.1f}%)")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, default=3, help="Max voting consensus rollouts")
    args = parser.parse_args()

    run_shootout(max_rollouts=args.rollouts)
