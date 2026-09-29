#!/usr/bin/env python3
"""
DeepSeek-R1-Distill-Qwen-7B: Extended Reasoning Budget & Voting Consensus
========================================================================
Implements test-time compute scaling on AMD ROCm GPU:
1. Enforces a minimum thinking budget (e.g., 1,200 tokens) by suppressing the
   closing `</think>` token until deliberation depth is reached.
2. Generates N independent stochastic reasoning rollouts (Best-of-N).
3. Extracts candidate solutions and performs majority voting consensus.
"""

import os
import sys
import time
import re
import argparse
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
)

MODEL_PATH = "/data/models/DeepSeek-R1-Distill-Qwen-7B"
THINK_END_TOKEN_ID = 151649  # </think>


class ForcedThinkingBudgetLogitsProcessor(LogitsProcessor):
    """
    Suppresses the closing </think> token until a minimum number of thinking tokens have elapsed,
    forcing the model to continue verifying steps, considering alternative methods, and self-auditing.
    """
    def __init__(self, think_end_token_id: int, min_think_tokens: int, prompt_len: int):
        self.think_end_token_id = think_end_token_id
        self.min_think_tokens = min_think_tokens
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        gen_tokens = input_ids.shape[1] - self.prompt_len
        if gen_tokens < self.min_think_tokens:
            scores[:, self.think_end_token_id] = -float("inf")
        return scores


def extract_boxed_content(text: str) -> List[str]:
    """Extracts all content inside balanced \\boxed{...} braces."""
    results = []
    target = r"\boxed{"
    idx = 0
    while True:
        pos = text.find(target, idx)
        if pos == -1:
            break
        start = pos + len(target)
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start:i-1].strip())
        idx = start
    return results


def normalize_latex_answer(ans: str) -> str:
    """Normalizes fractions, removes LaTeX commands and whitespace."""
    ans = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", ans)
    ans = re.sub(r"[{}$]", "", ans)
    ans = re.sub(r"\\mathbf|\\mathrm|\\text", "", ans)
    return ans.strip().replace(" ", "")


def extract_boxed_or_final_answer(text: str) -> Optional[str]:
    """Extracts the final answer from boxed LaTeX, explicit statements, or conclusion lines."""
    # 1. LaTeX \boxed{answer} with balanced braces
    boxed_items = extract_boxed_content(text)
    if boxed_items:
        return normalize_latex_answer(boxed_items[-1])

    # 2. "The final answer is X" or "answer is: X"
    ans_match = re.findall(r"(?:final\s+answer\s+is|answer\s+is|is\s+equal\s+to)[:\s]*([^\n\.\,\$]+)", text, re.IGNORECASE)
    if ans_match:
        return normalize_latex_answer(ans_match[-1])

    # 3. GSM8K #### X
    hash_match = re.findall(r"####\s*([^\n]+)", text)
    if hash_match:
        return normalize_latex_answer(hash_match[-1])

    # 4. Fallback: look at last non-empty line
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        last = lines[-1].replace("$", "").replace("**", "").strip()
        nums = re.findall(r"[-+]?\d+(?:\/\d+)?(?:\.\d+)?", last)
        if nums:
            return nums[-1].strip()

    return None


def generate_reasoning_rollout(
    model,
    tokenizer,
    prompt: str,
    min_think_tokens: int = 800,
    max_think_tokens: int = 1200,
    max_answer_tokens: int = 400,
    temperature: float = 0.7,
    top_p: float = 0.95,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Generates a single reasoning rollout with enforced minimum thinking budget and guaranteed conclusion."""
    formatted = f"<｜User｜>{prompt}<｜Assistant｜><think>\n"
    input_ids = tokenizer.encode(formatted, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    processors = LogitsProcessorList()
    if min_think_tokens > 0:
        processors.append(ForcedThinkingBudgetLogitsProcessor(THINK_END_TOKEN_ID, min_think_tokens, prompt_len))

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_think_tokens,
            temperature=temperature,
            top_p=top_p,
            logits_processor=processors,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.03,
        )

    # Check if </think> was generated
    full_output = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=False)
    if "</think>" not in full_output:
        # Forcefully conclude thinking phase: append \n</think>\n and generate final answer
        forced_suffix = "\n</think>\n"
        suffix_ids = tokenizer.encode(forced_suffix, return_tensors="pt", add_special_tokens=False).to(device)
        extended_ids = torch.cat([output_ids, suffix_ids], dim=1)
        with torch.no_grad():
            final_output_ids = model.generate(
                extended_ids,
                max_new_tokens=max_answer_tokens,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.03,
            )
        gen_time = time.time() - t0
        total_tokens = final_output_ids.shape[1] - prompt_len
        full_output = tokenizer.decode(final_output_ids[0][prompt_len:], skip_special_tokens=False)
    else:
        # </think> was generated; ensure answer text has enough runway to produce a boxed answer or reach EOS
        parts = full_output.split("</think>", 1)
        ans_cand = parts[1].strip()
        has_boxed = "\\boxed" in ans_cand
        has_eos = any(tok in ans_cand for tok in ["<｜end of sentence｜>", "<|im_end|>", "</s>", "####"])
        if not has_boxed and not has_eos:
            with torch.no_grad():
                final_output_ids = model.generate(
                    output_ids,
                    max_new_tokens=max_answer_tokens,
                    temperature=0.6,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_time = time.time() - t0
            total_tokens = final_output_ids.shape[1] - prompt_len
            full_output = tokenizer.decode(final_output_ids[0][prompt_len:], skip_special_tokens=False)
        else:
            gen_time = time.time() - t0
            total_tokens = output_ids.shape[1] - prompt_len

    # Parse <think> tokens and answer
    if "</think>" in full_output:
        parts = full_output.split("</think>", 1)
        think_text = parts[0].strip()
        answer_text = parts[1].strip()
    else:
        think_text = full_output
        answer_text = ""

    think_tokens = len(tokenizer.encode(think_text, add_special_tokens=False))
    extracted = extract_boxed_or_final_answer(answer_text if answer_text else think_text)

    return {
        "think_text": think_text,
        "answer_text": answer_text,
        "think_tokens": think_tokens,
        "total_tokens": total_tokens,
        "gen_time": gen_time,
        "tok_per_sec": total_tokens / max(gen_time, 1e-4),
        "extracted_answer": extracted,
    }


def symbolic_equal(ans1: Optional[str], ans2: Optional[str]) -> bool:
    """Checks exact string or symbolic mathematical equivalence using SymPy."""
    if ans1 is None or ans2 is None:
        return False
    str1 = str(ans1).strip().replace("$", "")
    str2 = str(ans2).strip().replace("$", "")
    if str1 == str2:
        return True
    try:
        import sympy as sp
        sym1 = sp.sympify(str1)
        sym2 = sp.sympify(str2)
        if sp.simplify(sym1 - sym2) == 0:
            return True
    except Exception:
        pass
    return False


def cluster_votes(votes: List[Optional[str]]) -> Counter:
    """Clusters votes by symbolic mathematical equivalence."""
    clusters = Counter()
    canonical_map = {}
    for v in votes:
        if v is None:
            clusters[None] += 1
            continue
        matched_canonical = None
        for canonical in canonical_map:
            if symbolic_equal(v, canonical):
                matched_canonical = canonical
                break
        if matched_canonical is not None:
            clusters[matched_canonical] += 1
        else:
            canonical_map[v] = v
            clusters[v] += 1
    return clusters


def run_voting_consensus(
    prompt: str,
    ground_truth: str,
    num_rollouts: int = 3,
    min_think_tokens: int = 800,
    max_answer_tokens: int = 400,
    early_exit: bool = True,
    use_primenet: bool = True,
    device: str = "cuda",
):
    max_think_tokens = min_think_tokens + 400 if min_think_tokens > 0 else 1200
    majority_threshold = (num_rollouts // 2) + 1

    print("=" * 90)
    print(f"  DEEPSEEK-R1-DISTILL-QWEN-7B: EXTENDED REASONING & VOTING CONSENSUS")
    print(f"  Forced Thinking Budget: {min_think_tokens} tokens (Max Think: {max_think_tokens}) | Rollouts: N={num_rollouts}")
    print(f"  Early-Exit Threshold: >= {majority_threshold}/{num_rollouts} agreement | PRIME-Net: {'ENABLED' if use_primenet else 'DISABLED'}")
    print(f"  Target Device: {device} | 4-bit NF4 Quantization")
    print("=" * 90)

    # PRIME-Net prompt invariant extraction
    injected_invariant = None
    if use_primenet:
        try:
            from prime_moment_attention.primenet_harness import PrimeNetMathHarness
            harness = PrimeNetMathHarness()
            injected_invariant = harness.extract_prompt_invariants(prompt)
            if injected_invariant:
                print(f"[*] PRIME-Net Invariant Extracted: {injected_invariant}")
        except Exception as e:
            print(f"[!] PRIME-Net warning: {e}")

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

    eval_prompt = prompt
    if injected_invariant:
        eval_prompt = f"{injected_invariant}\n{prompt}"

    print(f"\n[Problem Prompt]:\n{eval_prompt}\n")
    print(f"Expected Ground Truth: {ground_truth}\n")

    rollouts = []
    votes = []
    early_exited = False

    for i in range(num_rollouts):
        print(f"--- [Rollout {i+1}/{num_rollouts}] (Thinking >= {min_think_tokens}, Cap {max_think_tokens} tokens)... ---")
        res = generate_reasoning_rollout(
            model=model,
            tokenizer=tokenizer,
            prompt=eval_prompt,
            min_think_tokens=min_think_tokens,
            max_think_tokens=max_think_tokens,
            max_answer_tokens=max_answer_tokens,
            temperature=0.7,
            top_p=0.95,
            device=device,
        )
        rollouts.append(res)
        ans = res["extracted_answer"]
        votes.append(ans)

        print(f"  • Thinking Tokens: {res['think_tokens']} tokens ({res['tok_per_sec']:.1f} tok/s in {res['gen_time']:.1f}s)")
        print(f"  • Extracted Candidate Answer: {ans}")
        print(f"  • Answer Text:\n    {res['answer_text'][:300].strip()}...\n")

        # Check early exit condition
        if early_exit and len(votes) >= 2:
            current_clusters = cluster_votes(votes)
            top_cand, top_count = current_clusters.most_common(1)[0]
            if top_cand is not None and top_count >= majority_threshold:
                print(f">>> [EARLY EXIT TRIGGERED]: Candidate '{top_cand}' secured majority ({top_count}/{num_rollouts})!")
                print(f">>> Skipping remaining {num_rollouts - (i + 1)} rollout(s). Compute saved: {((num_rollouts - (i + 1)) / num_rollouts) * 100:.1f}%")
                early_exited = True
                break

    # Tally votes with symbolic clustering
    vote_counter = cluster_votes(votes)
    winner, win_count = vote_counter.most_common(1)[0]
    total_evaluated = len(votes)
    confidence_pct = (win_count / total_evaluated) * 100.0
    matches_gt = symbolic_equal(winner, ground_truth)

    print("=" * 90)
    print("  CONSENSUS & DECISION REPORT")
    print("=" * 90)
    print(f"Vote Tally (Symbolic Clusters): {dict(vote_counter)}")
    print(f"Consensus Decision: {winner}")
    print(f"Consensus Confidence: {win_count}/{total_evaluated} ({confidence_pct:.1f}%)" + (" [EARLY-EXIT]" if early_exited else ""))
    print(f"Ground Truth Match: {'CORRECT [PASS]' if matches_gt else 'DISCREPANCY'}")
    print("=" * 90 + "\n")

    return {
        "decision": winner,
        "confidence_pct": confidence_pct,
        "rollouts_evaluated": total_evaluated,
        "early_exited": early_exited,
        "correct": matches_gt,
        "vote_tally": dict(vote_counter),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-think-tokens", type=int, default=800, help="Forced minimum thinking tokens")
    parser.add_argument("--max-answer-tokens", type=int, default=400, help="Maximum answer generation tokens")
    parser.add_argument("--rollouts", type=int, default=3, help="Number of voting consensus rollouts")
    parser.add_argument("--no-early-exit", action="store_true", help="Disable early exit majority stopping")
    parser.add_argument("--no-primenet", action="store_true", help="Disable PRIME-Net prompt invariant injection")
    args = parser.parse_args()

    # Challenging Combinatorial Probability Problem
    TEST_PROMPT = (
        "A box contains 5 red balls and 5 blue balls. If you draw 4 balls at random without replacement, "
        "what is the exact probability that you draw more red balls than blue balls? "
        "Express your answer as a simplified fraction."
    )
    GROUND_TRUTH = "11/42"

    run_voting_consensus(
        prompt=TEST_PROMPT,
        ground_truth=GROUND_TRUTH,
        num_rollouts=args.rollouts,
        min_think_tokens=args.min_think_tokens,
        max_answer_tokens=args.max_answer_tokens,
        early_exit=not args.no_early_exit,
        use_primenet=not args.no_primenet,
    )
