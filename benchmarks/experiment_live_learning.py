#!/usr/bin/env python3
"""
Grounded Empirical Study: Live Learning & Test-Time Training (TTT) on PRIME-125M
==============================================================================
Rigorously evaluates the plasticity vs. stability trade-off:
Can a 125M recurrent language model learn novel, out-of-distribution factual
documents during inference without destroying its core linguistic competence?

Compares 4 distinct experimental conditions:
1. Frozen Baseline (Zero-Shot Control)
2. Unconstrained Full-Model SGD (Catastrophic Forgetting Demonstration)
3. Restricted Projection Adaptation (v_proj Only)
4. Elastic Synaptic Plasticity (Fisher EWC Guardrails)
"""

import os
import sys
import copy
import json
import time
import math
import argparse
from typing import Dict, Any, List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from prime_moment_attention.model import PrimeConfig, PrimeForCausalLM
from prime_moment_attention.continual_ttt import OnlineTTTContinualLearner


# ─────────────────────────────────────────────────────────────────────────────
# 1. Novel Factual Document (In-Distribution Narrative with Novel Facts)
# ─────────────────────────────────────────────────────────────────────────────
NOVEL_NARRATIVE_DOCUMENT = (
    "Once upon a time in the quiet village of Oakhaven, a clever young inventor named Jasper constructed a "
    "magnificent mechanical bird made of polished brass. Jasper named his flying creation Zephyr. "
    "Zephyr was powered by a mysterious glowing blue crystal that Jasper had discovered deep inside Whispering Cave. "
    "Whenever Jasper turned the small silver key three times, Zephyr flapped its golden wings and soared into the sky. "
    "High above the village houses, Zephyr opened a secret compartment and dropped sweet butterscotch candies for the happy children. "
    "Every evening at sunset, Zephyr returned to rest on the tall cedar branch outside Jasper's workshop window."
)

# Cloze Probes testing exact factual recall of the novel narrative
NOVEL_NARRATIVE_PROBES = [
    {
        "id": "bird_name",
        "prompt": "Jasper named his magnificent mechanical brass bird",
        "target": " Ze",
        "distractor": " Lily",
    },
    {
        "id": "crystal_location",
        "prompt": "Jasper found the mysterious glowing blue crystal deep inside Whispering",
        "target": " Cave",
        "distractor": " House",
    },
    {
        "id": "key_turns",
        "prompt": "To make the bird soar into the sky, Jasper turned the small silver key",
        "target": " three",
        "distractor": " zero",
    },
    {
        "id": "dropped_item",
        "prompt": "High above the village houses, Zephyr opened a secret compartment and dropped sweet",
        "target": " butter",
        "distractor": " stones",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Foundational Anchor Battery (Grammar, Pronouns, Commonsense from TinyStories)
# ─────────────────────────────────────────────────────────────────────────────
ANCHOR_CALIBRATION_TEXT = (
    "Once upon a time, there was a little girl named Lily. Lily lived in a small house near the woods. "
    "Every morning, she ran to the garden to play with her friendly puppy. The puppy barked happily "
    "and chased a red ball across the green grass. Her mother called from the kitchen window, "
    "bringing two fresh apples for a morning snack. Lily smiled and shared the fruit with her brother Tim."
)

LINGUISTIC_PROBES = [
    # Subject-Verb Agreement
    {"context": "The happy little girl always", "correct": " plays", "incorrect": " play"},
    {"context": "Two little dogs", "correct": " run", "incorrect": " runs"},
    {"context": "The bird in the tree", "correct": " sings", "incorrect": " sing"},
    # Pronoun Binding
    {"context": "Lily lost her red shoe. When she looked around,", "correct": " she", "incorrect": " he"},
    {"context": "Tim wanted to play with his toy car. After lunch,", "correct": " he", "incorrect": " she"},
    {"context": "Lucy and her mother went to the store. Together,", "correct": " they", "incorrect": " he"},
    # Commonsense Semantics
    {"context": "Lily was very thirsty on a hot day, so she drank a glass of cold", "correct": " water", "incorrect": " sand"},
    {"context": "The hungry puppy was happy when his owner gave him a delicious", "correct": " bone", "incorrect": " stone"},
    {"context": "When it started to rain outside, Tim opened his big", "correct": " umbrella", "incorrect": " banana"},
    # Property Recall
    {"context": "Anna painted the wooden table bright blue. Now the table was", "correct": " blue", "incorrect": " green"},
    {"context": "The ice cream was left in the hot sun. Soon, the cold treat began to", "correct": " melt", "incorrect": " freeze"},
]


def evaluate_linguistic_battery(model: PrimeForCausalLM, tokenizer, device: str) -> float:
    """Evaluates discriminative linguistic accuracy on the 11 probe suite."""
    model.eval()
    correct_count = 0
    with torch.no_grad():
        for probe in LINGUISTIC_PROBES:
            ctx = probe["context"]
            c_tok = tokenizer.encode(probe["correct"])[0]
            i_tok = tokenizer.encode(probe["incorrect"])[0]

            input_ids = tokenizer.encode(ctx, return_tensors="pt").to(device)
            out = model(input_ids)
            logits = out["logits"][0, -1]
            probs = F.softmax(logits.float(), dim=-1)

            if probs[c_tok].item() > probs[i_tok].item():
                correct_count += 1

    return (correct_count / len(LINGUISTIC_PROBES)) * 100.0


def evaluate_factual_probes(model: PrimeForCausalLM, tokenizer, device: str) -> Dict[str, Any]:
    """Tests discriminative recall of facts from the novel narrative document."""
    model.eval()
    passed = 0
    details = []
    with torch.no_grad():
        for probe in NOVEL_NARRATIVE_PROBES:
            ctx = probe["prompt"]
            c_tok = tokenizer.encode(probe["target"])[0]
            i_tok = tokenizer.encode(probe["distractor"])[0]

            input_ids = tokenizer.encode(ctx, return_tensors="pt").to(device)
            out = model(input_ids)
            logits = out["logits"][0, -1]
            probs = F.softmax(logits.float(), dim=-1)

            p_target = probs[c_tok].item()
            p_distractor = probs[i_tok].item()
            top1_id = torch.argmax(probs).item()
            is_correct = p_target > p_distractor

            if is_correct:
                passed += 1

            details.append({
                "id": probe["id"],
                "p_target": p_target,
                "p_distractor": p_distractor,
                "is_correct": is_correct,
                "top1_predicted": tokenizer.decode([top1_id]),
            })

    accuracy = (passed / len(NOVEL_NARRATIVE_PROBES)) * 100.0
    return {"accuracy": accuracy, "details": details}


def run_condition_trial(
    condition_name: str,
    base_checkpoint_path: str,
    device: str,
    tokenizer,
    novel_input_ids: torch.Tensor,
    anchor_input_ids: torch.Tensor,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs a single live-learning condition from a fresh model instance."""
    print(f"\n[{condition_name.upper()}] Loading fresh model instance...")
    ckpt = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    model = PrimeForCausalLM(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Pre-adaptation evaluations
    pre_novel_loss = OnlineTTTContinualLearner(model).evaluate_loss(novel_input_ids)
    pre_anchor_loss = OnlineTTTContinualLearner(model).evaluate_loss(anchor_input_ids)
    pre_ling_acc = evaluate_linguistic_battery(model, tokenizer, device)
    pre_probe_res = evaluate_factual_probes(model, tokenizer, device)

    adapt_time = 0.0
    post_novel_loss = pre_novel_loss
    post_anchor_loss = pre_anchor_loss
    post_ling_acc = pre_ling_acc
    post_probe_res = pre_probe_res

    if config.get("enable_adaptation", False):
        learner = OnlineTTTContinualLearner(
            model=model,
            learning_rate=config["lr"],
            fisher_beta=0.99,
            elastic_lambda=config.get("elastic_lambda", 0.0),
            max_grad_norm=1.0,
            surprise_threshold=config.get("surprise_threshold", None),
            adapted_modules=config["adapted_modules"],
        )

        # Calibrate Fisher information if EWC is enabled
        if config.get("elastic_lambda", 0.0) > 0.0:
            print(f"  Calibrating Fisher information on foundational anchor...")
            learner.compute_fisher_initialization(anchor_input_ids, steps=15)

        t0 = time.time()
        print(f"  Executing {config['steps']} Test-Time Training (TTT) adaptation steps on novel text...")
        post_novel_loss = learner.adapt_on_sequence(
            novel_input_ids,
            steps=config["steps"],
            lr_scale=1.0,
        )
        if "cuda" in device:
            torch.cuda.synchronize()
        adapt_time = time.time() - t0

        post_anchor_loss = learner.evaluate_loss(anchor_input_ids)
        post_ling_acc = evaluate_linguistic_battery(model, tokenizer, device)
        post_probe_res = evaluate_factual_probes(model, tokenizer, device)

    anchor_drift = post_anchor_loss - pre_anchor_loss
    ling_delta = post_ling_acc - pre_ling_acc
    loss_reduction_pct = ((pre_novel_loss - post_novel_loss) / pre_novel_loss) * 100.0

    return {
        "condition": condition_name,
        "pre_novel_loss": pre_novel_loss,
        "post_novel_loss": post_novel_loss,
        "loss_reduction_pct": loss_reduction_pct,
        "pre_anchor_loss": pre_anchor_loss,
        "post_anchor_loss": post_anchor_loss,
        "anchor_drift": anchor_drift,
        "pre_ling_acc": pre_ling_acc,
        "post_ling_acc": post_ling_acc,
        "ling_delta": ling_delta,
        "pre_probe_acc": pre_probe_res["accuracy"],
        "post_probe_acc": post_probe_res["accuracy"],
        "adapt_time_s": adapt_time,
    }


def main():
    parser = argparse.ArgumentParser(description="PRIME 125M Live Learning Empirical Experiment")
    parser.add_argument("--checkpoint", type=str, default="/data/prime_checkpoints/prime_125m_step_10000.pt",
                        help="Path to base 125M checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  PRIME-125M LIVE LEARNING & TEST-TIME TRAINING (TTT) EMPIRICAL STUDY")
    print(f"  Target Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    novel_input_ids = tokenizer.encode(NOVEL_NARRATIVE_DOCUMENT, return_tensors="pt").to(device)
    anchor_input_ids = tokenizer.encode(ANCHOR_CALIBRATION_TEXT, return_tensors="pt").to(device)

    print(f"\nDocument Token Lengths:")
    print(f"  - Novel Narrative Text : {novel_input_ids.shape[1]} tokens")
    print(f"  - Foundational Anchor  : {anchor_input_ids.shape[1]} tokens\n")

    # Define the 4 experimental conditions
    conditions = {
        "1. Frozen Baseline": {
            "enable_adaptation": False,
        },
        "2. Unconstrained Full SGD": {
            "enable_adaptation": True,
            "lr": 5e-3,
            "steps": 30,
            "elastic_lambda": 0.0,
            "adapted_modules": "all",
        },
        "3. Restricted v_proj SGD": {
            "enable_adaptation": True,
            "lr": 5e-3,
            "steps": 30,
            "elastic_lambda": 0.0,
            "adapted_modules": ["v_proj"],
        },
        "4. Elastic Synaptic (EWC)": {
            "enable_adaptation": True,
            "lr": 5e-3,
            "steps": 30,
            "elastic_lambda": 300.0,
            "adapted_modules": ["v_proj"],
        },
    }

    results = []
    for cond_name, cfg in conditions.items():
        res = run_condition_trial(
            condition_name=cond_name,
            base_checkpoint_path=args.checkpoint,
            device=device,
            tokenizer=tokenizer,
            novel_input_ids=novel_input_ids,
            anchor_input_ids=anchor_input_ids,
            config=cfg,
        )
        results.append(res)

    print("\n" + "=" * 96)
    print("  EMPIRICAL RESEARCH FINDINGS: PLASTICITY VS. STABILITY SCORECARD")
    print("=" * 96)
    print(f"{'Condition':<26} | {'Novel Loss':<14} | {'Loss Red.':<10} | {'Anchor Drift':<13} | {'Ling. Acc':<12} | {'Probe Acc':<10}")
    print("-" * 96)

    for r in results:
        novel_str = f"{r['pre_novel_loss']:.2f} -> {r['post_novel_loss']:.2f}"
        drift_str = f"{r['anchor_drift']:+.4f}"
        ling_str = f"{r['post_ling_acc']:.1f}% ({r['ling_delta']:+.1f}%)"
        probe_str = f"{r['post_probe_acc']:.1f}%"
        loss_red_str = f"{r['loss_reduction_pct']:+.1f}%"
        print(f"{r['condition']:<26} | {novel_str:<14} | {loss_red_str:<10} | {drift_str:<13} | {ling_str:<12} | {probe_str:<10}")

    print("=" * 96 + "\n")

    # Save results to experiments
    out_path = "experiments/live_learning_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Full experimental telemetry saved to: {out_path}")


if __name__ == "__main__":
    main()
