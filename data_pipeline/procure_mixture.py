#!/usr/bin/env python3
"""
Dataset Procurement Pipeline: Python, C, C++, Math & Reasoning Mixture
======================================================================
Procures, filters, and standardizes a high-quality multi-domain training mixture:
1. Python Programming: Instructions, algorithms, and implementations.
2. C & C++ Systems Programming: Low-level C99/C++ code, pointers, data structures.
3. Mathematical Problem Solving: GSM8K with step-by-step arithmetic and logic traces.
4. Multi-Step Analytical Reasoning: SmolTalk Magpie Ultra chain-of-thought dialogues.

Outputs clean JSONL splits (train.jsonl and val.jsonl) to the target directory.
"""

import os
import sys
import json
import random
import argparse
from typing import Dict, Any, List, Generator
from datasets import load_dataset
from tqdm import tqdm


def format_qa_sample(instruction: str, response: str, category: str) -> Dict[str, Any]:
    """Standardizes prompt and response into unified format."""
    inst = instruction.strip()
    resp = response.strip()
    formatted_text = f"User: {inst}\n\nAssistant: {resp}"
    return {
        "text": formatted_text,
        "instruction": inst,
        "response": resp,
        "category": category,
    }


def stream_gsm8k(max_samples: int = 10000) -> List[Dict[str, Any]]:
    """Fetches GSM8K math reasoning problems."""
    print("[1/4] Procuring Mathematical Reasoning (GSM8K)...")
    samples = []
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        for item in ds:
            q = item["question"]
            a = item["answer"]
            samples.append(format_qa_sample(q, a, "math"))
            if len(samples) >= max_samples:
                break
    except Exception as e:
        print(f"  Warning loading GSM8k: {e}")
    print(f"  Extracted {len(samples)} GSM8K math reasoning samples.")
    return samples


def stream_python_alpaca(max_samples: int = 15000) -> List[Dict[str, Any]]:
    """Fetches Python code instructions."""
    print("[2/4] Procuring Python Instructions (iamtarun/python_code_instructions_18k_alpaca)...")
    samples = []
    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        for item in ds:
            inst = item.get("instruction", "")
            inp = item.get("input", "")
            out = item.get("output", "")
            if inp and len(inp.strip()) > 0:
                full_inst = f"{inst}\n\nContext / Input:\n{inp}"
            else:
                full_inst = inst
            if full_inst and out:
                samples.append(format_qa_sample(full_inst, out, "python"))
            if len(samples) >= max_samples:
                break
    except Exception as e:
        print(f"  Warning loading Python Alpaca: {e}")
    print(f"  Extracted {len(samples)} Python instruction samples.")
    return samples


def stream_c_and_cpp(max_samples: int = 15000) -> List[Dict[str, Any]]:
    """Fetches C and C++ programming data from CodeFeedback."""
    print("[3/4] Procuring C & C++ Systems Code (m-a-p/CodeFeedback-Filtered-Instruction)...")
    samples = []
    try:
        ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)
        for item in ds:
            lang = (item.get("lang") or "").lower()
            if lang in ["c", "cpp"]:
                q = item.get("query", "")
                a = item.get("answer", "")
                if q and a:
                    samples.append(format_qa_sample(q, a, f"c_cpp_{lang}"))
                if len(samples) >= max_samples:
                    break
    except Exception as e:
        print(f"  Warning loading C/C++ CodeFeedback: {e}")
    print(f"  Extracted {len(samples)} C/C++ programming samples.")
    return samples


def stream_smoltalk_reasoning(max_samples: int = 10000) -> List[Dict[str, Any]]:
    """Fetches multi-turn analytical reasoning dialogues from SmolTalk Magpie Ultra."""
    print("[4/4] Procuring Multi-Step Analytical Reasoning (SmolTalk Magpie Ultra)...")
    samples = []
    try:
        ds = load_dataset("HuggingFaceTB/smoltalk", "smol-magpie-ultra", split="train")
        for item in ds:
            msgs = item.get("messages", [])
            if len(msgs) >= 2:
                user_msg = ""
                assistant_msg = ""
                for m in msgs:
                    if m.get("role") == "user" and not user_msg:
                        user_msg = m.get("content", "")
                    elif m.get("role") == "assistant" and not assistant_msg:
                        assistant_msg = m.get("content", "")
                if user_msg and assistant_msg:
                    samples.append(format_qa_sample(user_msg, assistant_msg, "reasoning"))
                if len(samples) >= max_samples:
                    break
    except Exception as e:
        print(f"  Warning loading SmolTalk reasoning: {e}")
    print(f"  Extracted {len(samples)} analytical reasoning samples.")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Procure and format multi-domain dataset")
    parser.add_argument("--output-dir", type=str, default="/data/datasets/prime_code_math_reasoning",
                        help="Target output directory on disk")
    parser.add_argument("--val-samples", type=int, default=2000,
                        help="Number of held-out validation samples")
    parser.add_argument("--max-per-category", type=int, default=15000,
                        help="Maximum samples per category")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic splitting")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    print("  PROCURING FOUNDATION CODE, MATH & REASONING DATASET")
    print(f"  Destination Directory : {args.output_dir}")
    print(f"  Max per Category      : {args.max_per_category:,}")
    print("=" * 80 + "\n")

    # Procure all 4 pillars
    math_samples = stream_gsm8k(max_samples=args.max_per_category)
    py_samples = stream_python_alpaca(max_samples=args.max_per_category)
    c_samples = stream_c_and_cpp(max_samples=args.max_per_category)
    reasoning_samples = stream_smoltalk_reasoning(max_samples=args.max_per_category)

    all_samples = math_samples + py_samples + c_samples + reasoning_samples
    print(f"\n[+] Total raw procured samples: {len(all_samples):,}")

    # Shuffle to interleave categories uniformly
    print("[*] Shuffling dataset mixture...")
    random.shuffle(all_samples)

    # Split into train and validation
    val_count = min(args.val_samples, int(len(all_samples) * 0.1))
    val_samples = all_samples[:val_count]
    train_samples = all_samples[val_count:]

    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    stats_path = os.path.join(args.output_dir, "dataset_stats.json")

    print(f"[*] Writing {len(train_samples):,} training records to: {train_path}...")
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[*] Writing {len(val_samples):,} validation records to: {val_path}...")
    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    category_counts = {}
    for s in all_samples:
        c = s["category"]
        category_counts[c] = category_counts.get(c, 0) + 1

    stats = {
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "categories": category_counts,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 80)
    print("  DATASET PROCUREMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"  Total Samples      : {len(all_samples):,}")
    print(f"  Training Split     : {len(train_samples):,}")
    print(f"  Validation Split   : {len(val_samples):,}")
    print(f"  Category Breakdown :")
    for cat, count in category_counts.items():
        pct = (count / len(all_samples)) * 100.0
        print(f"    - {cat:<18}: {count:,} ({pct:5.1f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
