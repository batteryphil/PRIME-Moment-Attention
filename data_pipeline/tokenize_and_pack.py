#!/usr/bin/env python3
"""
Tokenization & Binary Packing Utility
====================================
Tokenizes JSONL training and validation splits into memory-mapped uint16 binary files.
Enables zero-overhead, high-throughput streaming training directly into PyTorch tensors.
"""

import os
import sys
import json
import argparse
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm


def pack_jsonl_to_bin(jsonl_path: str, bin_path: str, tokenizer, eos_id: int):
    """Tokenizes each JSONL record, appends EOS token, and writes to uint16 binary file."""
    print(f"[*] Processing {jsonl_path} -> {bin_path}...")
    
    total_tokens = 0
    record_count = 0

    # First pass to count tokens
    with open(jsonl_path, "r", encoding="utf-8") as f:
        with open(bin_path, "wb") as out_f:
            for line in tqdm(f, desc=f"Tokenizing {os.path.basename(jsonl_path)}"):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text", "")
                if not text:
                    continue
                
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                token_ids.append(eos_id)
                arr = np.array(token_ids, dtype=np.uint16)
                out_f.write(arr.tobytes())
                
                total_tokens += len(token_ids)
                record_count += 1

    file_size_mb = os.path.getsize(bin_path) / (1024 * 1024)
    print(f"[+] Done: {record_count:,} records | {total_tokens:,} tokens | {file_size_mb:.2f} MB")
    return total_tokens, record_count


def main():
    parser = argparse.ArgumentParser(description="Tokenize and pack JSONL to binary files")
    parser.add_argument("--input-dir", type=str, default="/data/datasets/prime_code_math_reasoning",
                        help="Directory containing train.jsonl and val.jsonl")
    args = parser.parse_args()

    train_jsonl = os.path.join(args.input_dir, "train.jsonl")
    val_jsonl = os.path.join(args.input_dir, "val.jsonl")
    train_bin = os.path.join(args.input_dir, "train.bin")
    val_bin = os.path.join(args.input_dir, "val.bin")

    if not os.path.exists(train_jsonl):
        print(f"Error: {train_jsonl} does not exist. Run procure_mixture.py first.")
        sys.exit(1)

    print("=" * 80)
    print("  TOKENIZING AND PACKING DATASET INTO CONTINUOUS BINARY ARRAYS")
    print(f"  Input Directory : {args.input_dir}")
    print("=" * 80 + "\n")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eos_id = tokenizer.eos_token_id

    train_tokens, train_records = pack_jsonl_to_bin(train_jsonl, train_bin, tokenizer, eos_id)
    val_tokens, val_records = pack_jsonl_to_bin(val_jsonl, val_bin, tokenizer, eos_id)

    manifest = {
        "train_tokens": train_tokens,
        "train_records": train_records,
        "val_tokens": val_tokens,
        "val_records": val_records,
        "total_tokens": train_tokens + val_tokens,
        "dtype": "uint16",
        "vocab_size": len(tokenizer),
    }

    manifest_path = os.path.join(args.input_dir, "packed_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("  PACKING COMPLETE")
    print("=" * 80)
    print(f"  Total Tokens Packed : {manifest['total_tokens']:,}")
    print(f"  Training Tokens     : {train_tokens:,}")
    print(f"  Validation Tokens   : {val_tokens:,}")
    print(f"  Binary Manifest     : {manifest_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
