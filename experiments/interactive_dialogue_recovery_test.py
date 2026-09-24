#!/usr/bin/env python3
"""
Interactive Dialogue & Recovery Verification for PrimeLM-50M
===========================================================
Conducts a comprehensive conversation with PrimeLM-50M to assess:
  1. Practical Mechanical Diagnostics (Hydraulic pump cavitation / whining)
  2. Torque to Horsepower Calculation (HP = Torque * RPM / 5252)
  3. Declarative Memory Planting (Cat 320 oil filter 1R-0716)
  4. Memory Recall across Dialogue Turns
  5. General Conversational Greeting & Capability Scoping
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from prime_moment_attention.primelm_50m import PrimeLM50M
from prime_moment_attention.primenet_cothinker_bridge import PrimeNetCoThinker


def run_recovery_test(checkpoint_path="checkpoints/primelm_50m_sft_v3_best.pt", temp=0.5, top_k=25, rep_penalty=1.25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing PrimeLM-50M on {device} from {checkpoint_path}...")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    # 64 KB Persistent GTRM memory state
    memory_state = torch.zeros(1, 32, 512, device=device)

    dialogue_plan = [
        ("Greeting & Overview",
         "Hello! How can you help me as an equipment technician today?"),
        
        ("Mechanical Troubleshooting",
         "An excavator hydraulic pump is making a loud whining noise and the boom moves slowly. What should I inspect first?"),
         
        ("Horsepower Physics Calculation",
         "A diesel engine produces 350 ft-lbs of torque at 2100 RPM. Calculate the horsepower using HP = (Torque * RPM) / 5252."),
         
        ("Episodic Memory Planting",
         "Record this into your memory: The engine oil filter part number for our Cat 320 excavator is 1R-0716."),
         
        ("Memory Retrieval & Cross-Check",
         "What is the engine oil filter part number for our Cat 320 excavator?")
    ]

    print("\n" + "=" * 80)
    print("         PRIMELM-50M RECOVERY & USABILITY LIVE VERIFICATION")
    print("=" * 80 + "\n")

    for i, (topic, user_input) in enumerate(dialogue_plan, 1):
        print(f"\n--- [TURN {i}: {topic.upper()}] ---")
        print(f"Technician: {user_input}")

        prompt_text = f"User: {user_input}\n\nAssistant: "
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

        # Update persistent memory manifold from user prompt
        with torch.no_grad():
            _, _, next_mem = model(input_ids, return_gtrm_state=True, gtrm_state=memory_state)
            if next_mem is not None:
                memory_state = next_mem

        # Generate response
        curr = input_ids.clone()
        output_tokens = []
        for step in range(120):
            if curr.shape[1] >= 2048:
                curr = curr[:, -1024:]

            with torch.no_grad():
                logits, _, current_mem = model(curr, gtrm_state=memory_state, return_gtrm_state=True)
                next_logits = logits[:, -1, :].clone()
                
                # Repetition penalty
                if rep_penalty > 1.0 and output_tokens:
                    for tok_idx in set(output_tokens):
                        if next_logits[0, tok_idx] > 0:
                            next_logits[0, tok_idx] /= rep_penalty
                        else:
                            next_logits[0, tok_idx] *= rep_penalty
                
                if temp > 0:
                    next_logits = next_logits / temp
                    if top_k > 0:
                        v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
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

        if current_mem is not None:
            memory_state = current_mem

        raw_response = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        verified_response, injections = cothinker.intercept_and_solve(raw_response)

        think_part = ""
        answer_part = verified_response
        if "<think>" in verified_response:
            parts = verified_response.split("</think>")
            think_part = parts[0].replace("<think>", "").strip()
            answer_part = parts[1].strip() if len(parts) > 1 else ""

        print(f"\nPrimeLM-50M:")
        if think_part:
            print(f"  [Cognitive Thought Trace]:\n    {think_part}")
        if injections:
            print(f"  [PRIME-Net Symbolic Injections]: {injections}")
        print(f"  [Response]:\n    {answer_part if answer_part else verified_response}")
        print(f"  [GTRM State]: 64.0 KB Constant Memory Active")
        print("-" * 80)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/primelm_50m_sft_v3_best.pt"
    run_recovery_test(checkpoint_path=ckpt)
