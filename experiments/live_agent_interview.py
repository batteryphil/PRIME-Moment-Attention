#!/usr/bin/env python3
"""
1-on-1 Multi-Topic Dialogue & Deep Evaluation of PrimeLM-50M
===========================================================
Conducts an interactive, multi-turn interview with PrimeLM-50M across:
  1. Mechanical Engineering & Hydraulic Pressure (Pascal's Law)
  2. Episodic Memory Consolidation & Retention (64 KB GTRM Recall)
  3. Mathematical Deduction & Symbolic PRIME-Net Co-Thinking
  4. Cognitive Self-Awareness & Architecture Reflection
  5. Practical Common Sense Reasoning

Maintains persistent 2nd-order memory M^(2) across all turns.
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from prime_moment_attention.primelm_50m import PrimeLM50M
from prime_moment_attention.primenet_cothinker_bridge import PrimeNetCoThinker


def run_interview():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing PrimeLM-50M Dialogue Engine on {device}...")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/primelm_50m_sft_v2.pt")
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--rep_penalty", type=float, default=1.2)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/primelm_50m_sft_v2_best.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/primelm_50m_sft_reasoning_best.pt"

    print(f"[*] Loading model from: {checkpoint_path}")
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

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    cothinker = PrimeNetCoThinker()
    # Persistent 64 KB memory state M^(2)
    memory_state = torch.zeros(1, 32, 512, device=device)

    # Multi-topic interview dialogue script
    conversation_turns = [
        # Turn 1: Equipment Mechanics / Hydraulic Physics
        ("Mechanics & Hydraulics",
         "A hydraulic cylinder on an excavator has a piston area of 5 square inches operating at 2000 PSI. Calculate the force produced using F = P * A."),

        # Turn 2: Planting an Episodic Memory (for testing organic recall later)
        ("Memory Planting",
         "Record this into your memory: The maintenance supervisor for Excavator Unit 7 is named Marcus, and the radio frequency is 462.55 MHz."),

        # Turn 3: Multi-Step Algebra with Variable Isolation
        ("Algebraic Deduction",
         "Solve this equation step-by-step: 4 * x + 16 = 40."),

        # Turn 4: Memory Recall across conversational horizon
        ("Episodic Memory Recall",
         "What is the maintenance supervisor for Excavator Unit 7, and what was the radio frequency?"),

        # Turn 5: Architecture & Metacognitive Reflection
        ("Metacognition & Purpose",
         "Explain how your thinking process works inside <think> and what your 2nd-order memory does.")
    ]

    print("\n" + "=" * 80)
    print("         1-ON-1 CONVERSATION & EVALUATION: ANTIGRAVITY <-> PRIMELM-50M")
    print("=" * 80 + "\n")

    transcript = []

    for i, (topic, user_query) in enumerate(conversation_turns, 1):
        print(f"\n--- [TURN {i}: {topic.upper()}] ---")
        print(f"Antigravity: {user_query}")

        prompt_text = f"User: {user_query}\n\nAssistant: "
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

        # Update persistent memory manifold from user prompt
        with torch.no_grad():
            _, _, next_mem = model(input_ids, return_gtrm_state=True, gtrm_state=memory_state)
            if next_mem is not None:
                memory_state = next_mem

        # Generate response while maintaining persistent 2nd-order memory conditioning
        curr = input_ids.clone()
        output_tokens = []
        for step in range(120):
            if curr.shape[1] >= 2048:
                curr = curr[:, -1024:]

            with torch.no_grad():
                logits, _, current_mem = model(curr, gtrm_state=memory_state, return_gtrm_state=True)
                next_logits = logits[:, -1, :].clone()
                
                # Apply repetition penalty to already generated tokens
                if args.rep_penalty > 1.0 and output_tokens:
                    unique_toks = set(output_tokens)
                    for tok_idx in unique_toks:
                        if next_logits[0, tok_idx] > 0:
                            next_logits[0, tok_idx] /= args.rep_penalty
                        else:
                            next_logits[0, tok_idx] *= args.rep_penalty
                
                if args.temp > 0:
                    next_logits = next_logits / args.temp
                    if args.top_k > 0:
                        v, _ = torch.topk(next_logits, min(args.top_k, next_logits.size(-1)))
                        next_logits[next_logits < v[:, [-1]]] = -float('Inf')
                    probs = F.softmax(next_logits, dim=-1)
                    next_tok = torch.multinomial(probs, 1)
                else:
                    next_tok = next_logits.argmax(dim=-1, keepdim=True)

            curr = torch.cat([curr, next_tok], dim=-1)
            tok_id = next_tok.item()
            output_tokens.append(tok_id)

            if tok_id == tokenizer.eos_token_id:
                break
        
        # After completing the turn, update memory state from the full generated turn
        if current_mem is not None:
            memory_state = current_mem

        raw_response = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

        # Intercept with PRIME-Net symbolic harness
        verified_response, injections = cothinker.intercept_and_solve(raw_response)

        # Separate think trace from final answer
        think_part = ""
        final_answer = verified_response
        if "<think>" in verified_response:
            parts = verified_response.split("</think>")
            think_part = parts[0].replace("<think>", "").strip()
            final_answer = parts[1].strip() if len(parts) > 1 else ""

        print(f"\nPrimeLM-50M:")
        if think_part:
            print(f"  [Reconstructive Thought Trace]:\n    {think_part}")
        if injections:
            print(f"  [PRIME-Net Exact Math Injected]: {injections}")
        print(f"  [Response]:\n    {final_answer if final_answer else verified_response}")

        mem_kb = (memory_state.numel() * 4) / 1024.0
        print(f"\n  [State Monitor: Episodic Memory Manifold = {mem_kb:.1f} KB Constant]")
        print("-" * 80)

        transcript.append({
            "turn": i,
            "topic": topic,
            "query": user_query,
            "think": think_part,
            "injections": injections,
            "answer": final_answer if final_answer else verified_response
        })

    return transcript


if __name__ == "__main__":
    run_interview()
