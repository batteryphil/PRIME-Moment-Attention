#!/usr/bin/env python3
"""
Grounded Perplexity & Throughput Evaluation Benchmark
=====================================================
Evaluates empirical cross-entropy loss, perplexity, and token processing throughput
on held-out reference data for the PRIME 125M model architecture.
"""

import os
import sys
import time
import math
import argparse
from typing import Tuple, List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "export_hf", "prime-125m-reasoning")))
from prime_moment_attention import PrimeForCausalLM, PrimeConfig


# Built-in held-out corpus of diverse linguistic passages for offline/consistent evaluation
REFERENCE_EVAL_PASSAGES = [
    "Once upon a time, there was a little boy named Tim. Tim loved playing in the garden with his yellow toy truck. One day, he found a tiny silver key under a flat stone. Tim picked up the key and wondered what it opened. He walked to the garden shed and found an old wooden box covered in ivy. The key fit smoothly into the lock and clicked.",
    "Inside the wooden box was a collection of glass marbles and an old brass compass. Tim held the compass in his hand and noticed the needle pointing toward the tall apple tree. He ran across the lawn to the tree and sat down on the grass, feeling the cool autumn breeze on his face. His sister Sarah joined him with two warm biscuits.",
    "A carpenter works with cedar and pine boards, measuring twice and cutting once with a sharp hand saw. Every joint requires careful chiseling to ensure that the mortise and tenon fit snugly without wobbling. After dry fitting the frame, a thin layer of hide glue is applied and clamped tightly until cure.",
    "During winter storms, coastal lighthouses maintain continuous vigilance over treacherous shoals and offshore reefs. The rotating prism amplifies the beam, casting alternating flashes of light across breaking swells. Ships navigate by these timed intervals, matching charts against the visual signature of the beacon.",
    "Scientific observation depends on systematic calibration of measuring instruments. When recording barometric pressure, temperature corrections must be applied to mercury columns to account for thermal expansion of both the liquid metal and the brass scale.",
    "In botanical taxonomy, plants are categorized by morphological characteristics including leaf venation, floral structure, and seed coat development. Modern genetic sequencing has refined these phylogenetic classifications, clarifying ancestral divergences.",
]


def load_model_and_tokenizer(model_path: str, device: str):
    """Loads either a HuggingFace directory or a raw PyTorch .pt checkpoint."""
    print(f"Loading model from: {model_path} on {device}...")
    if os.path.isdir(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    else:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config")
        model = PrimeForCausalLM(config)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model.to(device).eval()
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model.to(dtype)
    return model, tokenizer, dtype


def evaluate_perplexity_on_texts(
    model,
    tokenizer,
    device: str,
    dtype: torch.dtype,
    texts: List[str],
    seq_len: int = 256,
) -> Tuple[float, float, int, float]:
    """
    Computes cross-entropy loss, perplexity, total evaluated tokens, and tokens/sec throughput.
    """
    total_nll = 0.0
    total_tokens = 0

    t0 = time.time()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            L = input_ids.shape[1]
            if L < 4:
                continue

            # Process in chunks of seq_len
            for i in range(0, L - 1, seq_len):
                chunk = input_ids[:, i : i + seq_len + 1]
                if chunk.shape[1] < 2:
                    continue

                x = chunk[:, :-1]
                y = chunk[:, 1:]

                with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=dtype):
                    out = model(x)
                    logits = out.logits if hasattr(out, "logits") else out["logits"]
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")

                total_nll += loss.item()
                total_tokens += y.numel()

    if "cuda" in device:
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    throughput = total_tokens / max(elapsed, 1e-6)

    mean_loss = total_nll / max(total_tokens, 1)
    ppl = math.exp(min(mean_loss, 20.0))  # cap to prevent overflow if untrained
    return mean_loss, ppl, total_tokens, throughput


def main():
    parser = argparse.ArgumentParser(description="Grounded PRIME 125M Perplexity Benchmark")
    parser.add_argument("--model-path", type=str, default="export_hf/prime-125m-reasoning",
                        help="Path to HF model directory or .pt checkpoint")
    parser.add_argument("--use-tinystories", action="store_true",
                        help="Attempt to stream validation split from roneneldan/TinyStories")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Evaluation sequence chunk length")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device (cuda or cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, dtype = load_model_and_tokenizer(args.model_path, device)

    texts = []
    if args.use_tinystories:
        try:
            from datasets import load_dataset
            print("Streaming validation split from roneneldan/TinyStories...")
            ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
            for sample in ds:
                t = sample.get("text", "").strip()
                if t:
                    texts.append(t)
                if len(texts) >= args.num_samples:
                    break
            print(f"Loaded {len(texts)} samples from TinyStories validation split.")
        except Exception as e:
            print(f"Could not load TinyStories dataset ({e}). Falling back to reference eval corpus.")
            texts = REFERENCE_EVAL_PASSAGES * (args.num_samples // len(REFERENCE_EVAL_PASSAGES) + 1)
            texts = texts[:args.num_samples]
    else:
        texts = REFERENCE_EVAL_PASSAGES * (args.num_samples // len(REFERENCE_EVAL_PASSAGES) + 1)
        texts = texts[:args.num_samples]

    print("\n" + "=" * 80)
    print("  GROUNDED PRIME-125M PERPLEXITY & THROUGHPUT EVALUATION")
    print(f"  Target: {args.model_path}")
    print(f"  Device: {device} | Dtype: {dtype} | Samples: {len(texts)} | Chunk Len: {args.seq_len}")
    print("=" * 80 + "\n")

    mean_loss, ppl, total_tok, throughput = evaluate_perplexity_on_texts(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        texts=texts,
        seq_len=args.seq_len,
    )

    print("=" * 80)
    print("  EVALUATION RESULTS")
    print("=" * 80)
    print(f"  Total Tokens Evaluated : {total_tok:,}")
    print(f"  Mean Cross-Entropy Loss: {mean_loss:.4f} nats/token")
    print(f"  Perplexity (PPL)       : {ppl:.2f}")
    print(f"  Inference Throughput   : {throughput:.1f} tokens/sec")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
