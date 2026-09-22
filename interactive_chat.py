#!/usr/bin/env python3
"""
Interactive CLI for PRIME-125M Reasoning & Thinking Model

Streams responses with separate color-coded thought traces (<think>...</think>)
and final answers.
"""

import os
import sys
import re
import argparse
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.prime_moment_attention.model import PrimeForCausalLM


CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(description="PRIME-125M Interactive Reasoning Chat with PRIME-Net")
    parser.add_argument("--checkpoint", type=str, default="/data/prime_checkpoints/prime_125m_reasoning_sft_final.pt",
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--max-tokens", type=int, default=350, help="Max response tokens")
    parser.add_argument("--temp", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--no-primenet", action="store_true", help="Disable PRIME-Net symbolic co-thinker")
    args = parser.parse_args()

    use_primenet = not args.no_primenet

    # Fallback to latest checkpoint if final doesn't exist yet
    if not os.path.exists(args.checkpoint):
        ckpts = [os.path.join("/data/prime_checkpoints", f) for f in os.listdir("/data/prime_checkpoints")
                 if "prime_125m_reasoning_sft" in f and f.endswith(".pt")]
        if ckpts:
            args.checkpoint = sorted(ckpts, key=os.path.getmtime)[-1]
        else:
            args.checkpoint = "/data/prime_checkpoints/prime_125m_base_pretrained.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"  {BOLD}PRIME-125M Neuro-Symbolic Reasoning Console{RESET}")
    print(f"  Model Checkpoint: {CYAN}{args.checkpoint}{RESET}")
    print(f"  Device:           {device}")
    if use_primenet:
        print(f"  PRIME-Net Engine: {MAGENTA}{BOLD}ACTIVE (Symbolic Invariant & Thought Verifier){RESET}")
    else:
        print(f"  PRIME-Net Engine: {DIM}DISABLED (Pure Neural Mode){RESET}")
    print(f"  Type 'exit', 'quit', or 'q' to end.")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PrimeForCausalLM(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    while True:
        try:
            user_input = input(f"\n{BOLD}You:{RESET} ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            prompt = f"User: {user_input}\n\nAssistant: <think>\n"

            print(f"\n{DIM}{YELLOW}Thinking Process:{RESET}")
            
            with torch.no_grad():
                if use_primenet:
                    full_output = model.generate_with_primenet(
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temp,
                        top_k=args.top_k,
                        verbose=False,
                    )
                else:
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temp,
                        top_k=args.top_k,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            if "Assistant:" in full_output:
                asst_part = full_output.split("Assistant:", 1)[1].strip()
            else:
                asst_part = full_output

            if "<think>" in asst_part and "</think>" in asst_part:
                parts = asst_part.split("</think>", 1)
                think_text = parts[0].replace("<think>", "").strip()
                ans_text = parts[1].strip()

                # Highlight PRIME-Net interventions in magenta
                highlighted_think = re.sub(
                    r'(\[PRIME-Net[^\]]*\])',
                    f'{BOLD}{MAGENTA}\\1{RESET}{DIM}{CYAN}',
                    think_text
                )
                print(f"{DIM}{CYAN}{highlighted_think}{RESET}\n")
                print(f"{BOLD}{GREEN}Answer:{RESET}")
                print(f"{ans_text}")
            elif "</think>" in asst_part:
                parts = asst_part.split("</think>", 1)
                think_text = parts[0].replace("<think>", "").strip()
                ans_text = parts[1].strip()

                highlighted_think = re.sub(
                    r'(\[PRIME-Net[^\]]*\])',
                    f'{BOLD}{MAGENTA}\\1{RESET}{DIM}{CYAN}',
                    think_text
                )
                print(f"{DIM}{CYAN}{highlighted_think}{RESET}\n")
                print(f"{BOLD}{GREEN}Answer:{RESET}")
                print(f"{ans_text}")
            else:
                print(f"{asst_part}")

        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
