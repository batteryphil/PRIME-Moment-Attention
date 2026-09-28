#!/usr/bin/env python3
"""
PRIME Autonomous Co-Learner Daemon (Student Model)
==================================================
Runs continuously alongside the DeepSeek-R1-Distill-14B Autonomous Researcher.
Streams in newly synthesized theoretical papers as they are formulated,
performing live Online Test-Time Training (TTT) with Elastic Synaptic Plasticity.

Memory Budget: ~750 MB VRAM (Fits cleanly within the 5.1 GB headroom).
"""

import os
import sys
import time
import json
import sqlite3
from typing import Dict, Any, List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, '/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention')
from src.prime_moment_attention.continual_ttt import OnlineTTTContinualLearner

DB_PATH = '/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/research_vault/research_vault.sqlite3'
STATUS_PATH = '/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/research_vault/co_learner_status.json'
MODEL_PATH = '/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/export_hf/prime-125m-reasoning'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ANCHOR_TEXTS = [
    "The fundamental laws of thermodynamics dictate that energy cannot be created or destroyed, only transformed.",
    "In sequence modeling, causal attention preserves autoregressive dependency across future tokens.",
    "A prime number is a natural number greater than 1 that cannot be formed by multiplying two smaller natural numbers.",
    "The quick brown fox jumps over the lazy dog while observing the surrounding landscape."
]


def update_status(status_dict: Dict[str, Any]):
    tmp_path = STATUS_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(status_dict, f, indent=2)
    os.replace(tmp_path, STATUS_PATH)


def main():
    print("=" * 80)
    print(" [*] STARTING PRIME AUTONOMOUS CO-LEARNER DAEMON (STUDENT)")
    print(f" [*] Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" [*] Model: {MODEL_PATH}")
    print("=" * 80)

    # 1. Load Tokenizer & Model
    print("[*] Loading PRIME-125M-Reasoning onto GPU...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(DEVICE)

    vram_init = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    print(f"[+] Model loaded! Initial VRAM: {vram_init:.1f} MB")

    # 2. Initialize Online TTT Continual Learner
    learner = OnlineTTTContinualLearner(
        model=model,
        learning_rate=1e-4,
        elastic_lambda=100.0,
        fisher_beta=0.98,
        eps=1e-5
    )
    print(f"[+] Online TTT Engine active! Monitoring {len(learner.adapted_params)} parameter matrices.")

    # 3. Calibrate Initial Fisher Information on Anchors
    print("[*] Calibrating foundational Fisher Information curvature on anchor corpus...")
    anchor_tokens_list = [tok(text, return_tensors='pt').input_ids.to(DEVICE) for text in ANCHOR_TEXTS]
    
    for tokens in anchor_tokens_list:
        learner.compute_fisher_initialization(tokens, steps=10)

    initial_anchor_losses = [learner.evaluate_loss(tokens) for tokens in anchor_tokens_list]
    baseline_anchor_loss = sum(initial_anchor_losses) / len(initial_anchor_losses)
    print(f"[+] Foundational calibration complete! Baseline Anchor Loss: {baseline_anchor_loss:.4f}")

    # 4. Determine Starting Cycle
    last_cycle_processed = 0
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT MAX(cycle) FROM theories")
        row = cur.fetchone()
        if row and row[0] is not None:
            last_cycle_processed = max(0, row[0] - 3)
        conn.close()
    except Exception as e:
        print(f"[-] Warning reading initial cycle: {e}")

    print(f"[*] Starting streaming ingestion from Cycle {last_cycle_processed + 1} onward...")

    total_ingested = 0
    start_time = time.time()

    # 5. Continuous Co-Learning Loop
    while True:
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            cur = conn.cursor()
            cur.execute("""
                SELECT cycle, domain, topic, summary, timestamp 
                FROM theories 
                WHERE cycle > ? 
                ORDER BY cycle ASC 
                LIMIT 1
            """, (last_cycle_processed,))
            new_row = cur.fetchone()
            conn.close()

            if new_row is None:
                vram_now = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
                status_dict = {
                    "status": "WAITING_FOR_NEXT_THEORY",
                    "uptime_seconds": int(time.time() - start_time),
                    "last_cycle_processed": last_cycle_processed,
                    "total_theories_ingested": total_ingested,
                    "baseline_anchor_loss": round(baseline_anchor_loss, 4),
                    "vram_mb": round(vram_now, 1),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                update_status(status_dict)
                time.sleep(10)
                continue

            cycle, domain, topic, summary, theory_ts = new_row
            print(f"\n[+] [{time.strftime('%H:%M:%S')}] >>> INGESTING NEW RESEARCH CYCLE {cycle}: {domain} <<<")

            tokens = tok(summary[:1200], return_tensors='pt', truncation=True, max_length=256).input_ids.to(DEVICE)
            if tokens.shape[1] < 8:
                last_cycle_processed = cycle
                continue

            # Pre-adaptation loss
            loss_before = learner.evaluate_loss(tokens)

            # Online TTT (3 elastic gradient steps)
            learner.adapt_on_sequence(tokens, steps=3, lr_scale=1.5)

            # Post-adaptation loss
            loss_after = learner.evaluate_loss(tokens)
            loss_reduction = loss_before - loss_after
            pct_improvement = (loss_reduction / loss_before) * 100.0

            # Catastrophic Forgetting Probe on Anchors
            cur_anchor_losses = [learner.evaluate_loss(toks) for toks in anchor_tokens_list]
            cur_anchor_loss = sum(cur_anchor_losses) / len(cur_anchor_losses)
            anchor_drift = abs(cur_anchor_loss - baseline_anchor_loss)

            vram_now = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
            total_ingested += 1
            last_cycle_processed = cycle

            print(f"    - Pre-Adaptation Loss:    {loss_before:.4f}")
            print(f"    - Post-Adaptation Loss:   {loss_after:.4f} (-{loss_reduction:.4f}, +{pct_improvement:.2f}%)")
            print(f"    - Anchor Loss & Drift:    {cur_anchor_loss:.4f} (drift: {anchor_drift:.5f})")
            print(f"    - Total Ingested:         {total_ingested} theories")
            print(f"    - Student VRAM Footprint: {vram_now:.1f} MB")

            status_dict = {
                "status": "ABSORBED_THEORY",
                "uptime_seconds": int(time.time() - start_time),
                "last_cycle_processed": cycle,
                "last_domain_learned": domain,
                "total_theories_ingested": total_ingested,
                "loss_before": round(loss_before, 4),
                "loss_after": round(loss_after, 4),
                "loss_reduction": round(loss_reduction, 4),
                "percent_improvement": round(pct_improvement, 2),
                "anchor_loss": round(cur_anchor_loss, 4),
                "anchor_drift": round(anchor_drift, 5),
                "vram_mb": round(vram_now, 1),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            update_status(status_dict)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time.sleep(2)

        except Exception as e:
            print(f"[-] Error in co-learning loop: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
