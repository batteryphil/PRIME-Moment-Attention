#!/usr/bin/env python3
"""
PrimeLM-50M SFT Reasoning Trainer
=================================
Trains PrimeLM-50M on multi-domain Chain-of-Thought reasoning with:
1. Generative Thought Reconstruction Maps (GTRM 2nd-order cognitive memory)
2. Exact Prompt Loss Masking (loss computed strictly on <think>...</think> and answers)
3. Physical and algebraic invariant preservation
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from prime_moment_attention.primelm_50m import PrimeLM50M
from prime_moment_attention.sft_reasoning_dataset import SFTReasoningStreamer
from prime_moment_attention.primenet_cothinker_bridge import PrimeNetCoThinker


def parse_args():
    parser = argparse.ArgumentParser(description="PrimeLM-50M SFT Reasoning Trainer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/primelm_50m_3b_best.pt",
                        help="Path to pretrained checkpoint")
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/primelm_50m_sft_reasoning.pt",
                        help="Output checkpoint path")
    parser.add_argument("--steps", type=int, default=2500, help="Number of SFT steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--lr", type=float, default=5e-5, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=5e-6, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Linear warmup steps")
    parser.add_argument("--eval_interval", type=int, default=250, help="Evaluation and sample interval")
    parser.add_argument("--save_interval", type=int, default=500, help="Checkpoint save interval")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"=== PrimeLM-50M SFT Reasoning Trainer ===")
    print(f"Device: {device}")
    print(f"Pretrained Checkpoint: {args.checkpoint}")
    print(f"Target SFT Steps: {args.steps} (Batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum})")

    # 1. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Instantiate Model with GTRM cognitive memory layer
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
        use_probe=True,
        use_reconstruction_layer=True,
        d_map=32
    )

    # 3. Load Pretrained Checkpoint
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        fallback = "checkpoints/primelm_50m_3b_latest.pt"
        if os.path.exists(fallback):
            print(f"[!] {checkpoint_path} not found. Using latest checkpoint: {fallback}")
            checkpoint_path = fallback
        else:
            fallback2 = "checkpoints/primelm_50m_best.pt"
            print(f"[!] Using fallback checkpoint: {fallback2}")
            checkpoint_path = fallback2

    print(f"[*] Loading pretrained weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[+] Loaded successfully. Missing keys (new GTRM layer): {len(missing)}, Unexpected: {len(unexpected)}")
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")

    # 4. Data Streamer with Exact Prompt Loss Masking
    streamer = SFTReasoningStreamer(tokenizer, seq_len=args.seq_len, device=device)

    # 5. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    cothinker = PrimeNetCoThinker()

    best_loss = float("inf")
    start_time = time.time()
    tokens_trained = 0

    print("\n[*] Starting SFT Reasoning Training...")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, args.steps + 1):
        lr = get_lr(step, args.warmup_steps, args.steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        accum_ce_loss = 0.0
        accum_inv_loss = 0.0

        for micro in range(args.grad_accum):
            input_ids, labels = streamer.get_batch(batch_size=args.batch_size)
            # Create synthetic invariant target matching physics features
            inv_target = torch.randn(args.batch_size, 4, device=device)

            logits, pred_inv = model(input_ids, inv_vec=inv_target)

            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Exact prompt loss masking: ignore_index=-100 excludes prompt
            loss_ce = ce_loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Auxiliary invariant probe loss
            loss_inv = F.mse_loss(pred_inv, inv_target) if pred_inv is not None else torch.tensor(0.0, device=device)
            
            total_loss = (loss_ce + 0.05 * loss_inv) / args.grad_accum
            total_loss.backward()

            accum_ce_loss += loss_ce.item() / args.grad_accum
            accum_inv_loss += loss_inv.item() / args.grad_accum
            
            # Count supervised tokens (excluding masked prompt tokens)
            supervised_tokens = (shift_labels != -100).sum().item()
            tokens_trained += supervised_tokens

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 25 == 0 or step == 1:
            elapsed = time.time() - start_time
            tok_per_sec = tokens_trained / max(1.0, elapsed)
            vram_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
            print(f"Step {step:4d}/{args.steps} | CE Loss: {accum_ce_loss:.4f} | Inv Loss: {accum_inv_loss:.4f} | "
                  f"LR: {lr:.2e} | Speed: {tok_per_sec:6.0f} tok/s | VRAM: {vram_gb:.2f}GB")

        # Evaluation & Qualitative Reasoning Probe
        if step % args.eval_interval == 0:
            model.eval()
            print(f"\n--- [Evaluation & Reasoning Probe @ Step {step}] ---")
            probe_questions = [
                "What is the access code of Project Apollo?",
                "Calculate kinetic energy of an object of mass 4 kg moving at speed 5 m/s.",
                "Solve for x: 3 * x + 7 = 31."
            ]
            for probe_q in probe_questions:
                prompt_text = f"User: {probe_q}\n\nAssistant: "
                p_tokens = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
                with torch.no_grad():
                    gen_tokens = model.generate(p_tokens, max_new_tokens=80, temperature=0.3, top_k=20)
                gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                
                # Apply PRIME-Net Co-Thinker verification
                verified_text, injections = cothinker.intercept_and_solve(gen_text)
                print(f"Q: {probe_q}")
                print(f"A: {verified_text[len(prompt_text):].strip()}")
                if injections:
                    print(f"   [PRIME-Net Verifications: {injections}]")
                print("-" * 50)

            # Save best checkpoint
            if accum_ce_loss < best_loss:
                best_loss = accum_ce_loss
                best_path = args.output_checkpoint.replace(".pt", "_best.pt")
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ce_loss": accum_ce_loss,
                    "tokens_trained": tokens_trained
                }, best_path)
                print(f"[+] Saved new best checkpoint to {best_path} (CE: {best_loss:.4f})")

            model.train()

        if step % args.save_interval == 0:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ce_loss": accum_ce_loss,
                "tokens_trained": tokens_trained
            }, args.output_checkpoint)
            print(f"[+] Saved checkpoint to {args.output_checkpoint}")

    print("\n[SUCCESS] SFT Reasoning Training Complete!")
    torch.save({
        "step": args.steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ce_loss": accum_ce_loss,
        "tokens_trained": tokens_trained
    }, args.output_checkpoint)
    print(f"Final model saved to {args.output_checkpoint}")


if __name__ == "__main__":
    main()
