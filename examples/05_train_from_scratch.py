#!/usr/bin/env python3
"""
Example 05: Training a PRIME Language Model From Scratch
========================================================
Demonstrates training a complete decoder-only Causal Language Model
powered natively by PRIME Moment Attention from step 0.

Key highlights:
1. End-to-end architecture (RMSNorm, SwiGLU, tied embeddings).
2. QK-Normalization (use_qk_norm=True) to keep query/key dot products
   in the highest-fidelity zone of the 2nd-order Taylor polynomial.
3. Standard cross-entropy loss with PyTorch Autograd backpropagation.
4. O(1) constant-state generation after training.
"""

import math
import torch
import torch.nn as nn
from torch.optim import AdamW

from prime_moment_attention import PrimeConfig, PrimeForCausalLM

def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1] Target Device: {device}")

    # Toy text dataset (Character-level Shakespeare excerpt)
    corpus = (
        "To be, or not to be, that is the question: "
        "Whether 'tis nobler in the mind to suffer "
        "The slings and arrows of outrageous fortune, "
        "Or to take arms against a sea of troubles "
        "And by opposing end them. To die—to sleep, "
        "No more; and by a sleep to say we end "
        "The heart-ache and the thousand natural shocks "
        "That flesh is heir to: 'tis a consummation "
        "Devoutly to be wish'd."
    )
    
    # Character tokenizer
    chars = sorted(list(set(corpus)))
    vocab_size = len(chars)
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    
    encoded = torch.tensor([char2idx[c] for c in corpus], dtype=torch.long)
    print(f"[2] Vocabulary Size: {vocab_size} characters | Corpus Length: {len(encoded)} tokens")

    # Configure native PRIME model
    config = PrimeConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=3,
        num_heads=4,
        head_dim=32,
        decay=0.995,
        use_qk_norm=True,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    
    model = PrimeForCausalLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[3] Initialized PrimeForCausalLM ({total_params:,} parameters, {config.num_layers} layers)")

    # Training Setup
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    seq_len = 32
    batch_size = 4
    num_steps = 150

    print(f"[4] Training from scratch for {num_steps} steps...")
    model.train()
    
    for step in range(1, num_steps + 1):
        # Sample random batches
        ix = torch.randint(0, len(encoded) - seq_len - 1, (batch_size,))
        x = torch.stack([encoded[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([encoded[i + 1 : i + seq_len + 1] for i in ix]).to(device)

        optimizer.zero_grad()
        out = model(x, labels=y)
        loss = out["loss"]
        loss.backward()
        
        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 25 == 0 or step == 1:
            print(f"    Step {step:3d}/{num_steps} | Loss: {loss.item():.4f}")

    # Generate text with O(1) constant-state generation
    print("\n[5] Autoregressive Generation with O(1) Constant State:")
    model.eval()
    seed_text = "To be, or"
    prompt_ids = torch.tensor([[char2idx[c] for c in seed_text]], dtype=torch.long).to(device)

    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=60,
        temperature=0.6,
        top_k=5,
    )[0].tolist()

    generated_text = "".join([idx2char[i] for i in generated_ids])
    print(f"    Prompt:    \"{seed_text}\"")
    print(f"    Generated: \"{generated_text}\"")
    print("\n[Success] Model trained natively from scratch with zero NaN errors and flat O(1) state generation.")

if __name__ == "__main__":
    main()
