#!/usr/bin/env python3
"""
Interactive Reasoning Chat with PRIME-Net Co-Thinking & 2nd-Order Memory
========================================================================
Features:
- Live streaming <think>...</think> chain-of-thought generation
- Real-time PRIME-Net Co-Thinker symbolic validation ([PRIME-Net: ...])
- 2nd-Order Generative Thought Reconstruction Map (GTRM, 64 KB constant O(1) memory)
- Interactive CLI with ANSI color syntax highlighting
- Direct test benchmark evaluation mode (--eval)
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from prime_moment_attention.primelm_50m import PrimeLM50M
from prime_moment_attention.primenet_cothinker_bridge import PrimeNetCoThinker


# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


class PrimeReasoningChat:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        print(f"{DIM}[*] Loading tokenizer (GPT-2 BPE)...{RESET}")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"{DIM}[*] Instantiating PrimeLM-50M with Generative Thought Reconstruction Layer...{RESET}")
        self.model = PrimeLM50M(
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

        # Check for available checkpoint
        found_ckpt = None
        for cand in [checkpoint_path, "checkpoints/primelm_50m_sft_reasoning_best.pt",
                     "checkpoints/primelm_50m_sft_reasoning.pt", "checkpoints/primelm_50m_3b_best.pt",
                     "checkpoints/primelm_50m_3b_latest.pt", "checkpoints/primelm_50m_best.pt"]:
            if os.path.exists(cand):
                found_ckpt = cand
                break

        if found_ckpt:
            print(f"{GREEN}[+] Loading checkpoint: {found_ckpt}{RESET}")
            ckpt = torch.load(found_ckpt, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model", ckpt)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"{DIM}[+] Checkpoint loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}{RESET}")
        else:
            print(f"{YELLOW}[!] Warning: No checkpoint found, running with initialized weights.{RESET}")

        self.model.to(self.device)
        self.model.eval()

        self.cothinker = PrimeNetCoThinker()
        # Initialize persistent 2nd-order memory state M^(2) in R^(1 x 32 x 512) (64 KB)
        self.memory_state = torch.zeros(1, 32, 512, device=self.device)
        self.total_memories_consolidated = 0

    def generate_response(self, prompt: str, max_tokens: int = 150, temperature: float = 0.4, top_k: int = 25):
        """
        Generates response while streaming tokens, displaying thoughts in real time,
        and invoking PRIME-Net symbolic harness for exact math calculations.
        """
        formatted_prompt = f"User: {prompt.strip()}\n\nAssistant: "
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)

        # Prefill prompt through model and update persistent memory state
        with torch.no_grad():
            logits, _, next_mem = self.model(input_ids, return_gtrm_state=True, gtrm_state=self.memory_state)
            if next_mem is not None:
                self.memory_state = next_mem
                self.total_memories_consolidated += input_ids.shape[1]

        curr = input_ids.clone()
        generated_tokens = []
        in_think_block = False
        has_opened_think = False
        has_closed_think = False
        think_buffer = ""
        full_output = ""

        sys.stdout.write(f"\n{BOLD}Assistant:{RESET} ")
        sys.stdout.flush()

        for step in range(max_tokens):
            L = curr.shape[1]
            if L >= 4096:
                curr = curr[:, -2048:]

            with torch.no_grad():
                logits, _ = self.model(curr)
                next_token_logits = logits[:, -1, :]

            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            curr = torch.cat([curr, next_token], dim=-1)
            tok_id = next_token.item()
            generated_tokens.append(tok_id)

            if tok_id == self.tokenizer.eos_token_id:
                break

            tok_text = self.tokenizer.decode([tok_id])
            full_output += tok_text

            # Detect <think> state transitions
            if "<think>" in full_output and not has_opened_think:
                has_opened_think = True
                in_think_block = True
                sys.stdout.write(f"\n{CYAN}{DIM}─── [Reconstructive Thought Trace] ───\n")
                sys.stdout.flush()

            if in_think_block:
                think_buffer += tok_text
                # Check for arithmetic or invariant expressions to verify via PRIME-Net
                if "<<" in think_buffer and ">>" in think_buffer:
                    verified_buf, injections = self.cothinker.intercept_and_solve(think_buffer)
                    if injections:
                        for inj in injections:
                            sys.stdout.write(f"\n{GREEN}{BOLD}[PRIME-Net Exact Math: {inj['expr']} = {inj['result']}]{RESET}{CYAN}{DIM} ")
                        think_buffer = ""
                
                # Check for explicit bracket calculation [calc: ...]
                if "[" in think_buffer and "]" in think_buffer:
                    verified_buf, injections = self.cothinker.intercept_and_solve(think_buffer)
                    if injections:
                        for inj in injections:
                            sys.stdout.write(f"\n{GREEN}{BOLD}[PRIME-Net Invariant Check: {inj['expr']} = {inj['result']}]{RESET}{CYAN}{DIM} ")
                        think_buffer = ""

                sys.stdout.write(f"{CYAN}{DIM}{tok_text}{RESET}")
                sys.stdout.flush()

                if "</think>" in full_output and not has_closed_think:
                    in_think_block = False
                    has_closed_think = True
                    sys.stdout.write(f"\n{CYAN}{DIM}───────────────────────────────────────{RESET}\n\n")
                    sys.stdout.flush()
            else:
                sys.stdout.write(tok_text)
                sys.stdout.flush()

        sys.stdout.write("\n\n")
        sys.stdout.flush()

    def run_eval_suite(self):
        """Runs automated evaluation on reasoning, math, and associative memory."""
        print(f"\n{BOLD}=== Running PrimeLM Reasoning & Co-Thinking Benchmark ==={RESET}\n")
        benchmarks = [
            ("Kinetic Energy Invariant", "Calculate kinetic energy of an object of mass 6 kg moving at speed 4 m/s."),
            ("Linear Algebra Deduction", "Solve for x: 5 * x + 15 = 45."),
            ("GSM8K Arithmetic Decomposition", "A store has 24 boxes of pens with 15 pens each. If 80 pens are sold, how many remain?"),
            ("Associative Memory Recall", "Record: The access code for Project Apollo is registered as 94812. What is the access code of Project Apollo?")
        ]

        for category, q in benchmarks:
            print(f"{YELLOW}{BOLD}[Benchmark: {category}]{RESET}")
            print(f"Prompt: {q}")
            self.generate_response(q, max_tokens=100, temperature=0.2)
            print("-" * 60)

    def interactive_loop(self):
        print(f"\n{BOLD}==================================================================={RESET}")
        print(f"{BOLD}  PrimeLM-50M: Co-Thinking Reasoning & Generative Reconstruction Map {RESET}")
        print(f"{BOLD}==================================================================={RESET}")
        print(f"{DIM}• Architecture: 50.9M Parameters | Hybrid PRIME Attention + GTRM Memory")
        print(f"• Biomimetic Episodic State: 64.0 KB Constant (M^(2) in R^(32x512))")
        print(f"• PRIME-Net Co-Thinker: Real-Time SymPy Invariant & Arithmetic Engine")
        print(f"• Commands: 'quit' or 'exit' to terminate, 'clear' to reset memory, 'eval' for benchmarks.{RESET}\n")

        while True:
            try:
                state_kb = (self.memory_state.numel() * 4) / 1024.0
                user_input = input(f"{BOLD}[Mem: {state_kb:.1f}KB | Toks: {self.total_memories_consolidated}]{RESET} >> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit"]:
                    print(f"{GREEN}Goodbye!{RESET}")
                    break
                elif user_input.lower() == "clear":
                    self.memory_state = torch.zeros(1, 32, 512, device=self.device)
                    self.total_memories_consolidated = 0
                    print(f"{YELLOW}[+] 2nd-order memory state reset to zero.{RESET}")
                    continue
                elif user_input.lower() == "eval":
                    self.run_eval_suite()
                    continue

                self.generate_response(user_input)

            except KeyboardInterrupt:
                print(f"\n{YELLOW}[!] Session interrupted by user.{RESET}")
                break


def main():
    parser = argparse.ArgumentParser(description="PrimeLM-50M Interactive Reasoning Chat")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/primelm_50m_sft_reasoning_best.pt")
    parser.add_argument("--eval", action="store_true", help="Run automated reasoning benchmark")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    chat = PrimeReasoningChat(args.checkpoint, device=args.device)
    if args.eval:
        chat.run_eval_suite()
    else:
        chat.interactive_loop()


if __name__ == "__main__":
    main()
