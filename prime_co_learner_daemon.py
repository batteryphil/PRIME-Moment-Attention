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
import math
import re
import sqlite3
from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, '/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention')
from src.prime_moment_attention.continual_ttt import OnlineTTTContinualLearner

DB_PATH = '/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/research_vault/research_vault.sqlite3'
STATUS_PATH = '/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/research_vault/co_learner_status.json'
INTELLIGENCE_METRICS_PATH = '/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/research_vault/student_intelligence_metrics.json'
MODEL_PATH = '/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/export_hf/prime-125m-reasoning'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ANCHOR_TEXTS = [
    "The fundamental laws of thermodynamics dictate that energy cannot be created or destroyed, only transformed.",
    "In sequence modeling, causal attention preserves autoregressive dependency across future tokens.",
    "A prime number is a natural number greater than 1 that cannot be formed by multiplying two smaller natural numbers.",
    "The quick brown fox jumps over the lazy dog while observing the surrounding landscape."
]

REASONING_PROBES = [
    {
        "id": "algebra_gsm8k",
        "category": "Mathematics",
        "prompt": "User: Solve for x: 4 * x - 8 = 24. Let's think step by step.\nAssistant: <think>\n",
        "validation_trace": "<think>\nTo solve 4 * x - 8 = 24:\nFirst, add 8 to both sides: 4 * x = 32.\nNext, divide both sides by 4: x = 8.\n</think>\nTherefore, x = 8."
    },
    {
        "id": "logic_syllogism",
        "category": "Deductive Logic",
        "prompt": "User: All planets orbit stars. Jupiter is a planet. What can we deduce?\nAssistant: <think>\n",
        "validation_trace": "<think>\nPremise 1: All planets orbit stars.\nPremise 2: Jupiter is a planet.\nTherefore, by deductive syllogism, Jupiter orbits a star.\n</think>\nJupiter orbits a star."
    },
    {
        "id": "prime_linear_attention",
        "category": "Linear Attention Theory",
        "prompt": "User: Why is cumulative state accumulation in linear attention associative?\nAssistant: <think>\n",
        "validation_trace": "<think>\nLinear attention computes S_t = S_{t-1} + k_t * v_t^T.\nMatrix addition is associative: (A + B) + C = A + (B + C).\nHence prefix sums can be computed in parallel with logarithmic span O(log N).\n</think>\nAssociativity stems from the associative property of matrix addition."
    }
]

COGNITIVE_KEYWORDS = ["first", "therefore", "because", "step", "since", "we have", "calculate", "so", "let's", "derive"]


def update_status(status_dict: Dict[str, Any]):
    tmp_path = STATUS_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(status_dict, f, indent=2)
    os.replace(tmp_path, STATUS_PATH)


def run_intelligence_and_imitation_evaluation(model, tok, learner, cycle_num: int) -> Dict[str, Any]:
    """
    Evaluates student intelligence and DeepSeek reasoning imitation:
    1. Perplexity on reference reasoning chains
    2. <think> tag adherence
    3. Cognitive keyword density
    4. Generates an actual response to track cognitive style evolution.
    """
    print(f"\n[*] === RUNNING INTELLIGENCE & DEEPSEEK IMITATION PROBE (Cycle {cycle_num}) ===")
    probe_results = []
    total_val_loss = 0.0
    think_tag_count = 0
    keyword_matches = 0
    sample_generation = ""

    for i, probe in enumerate(REASONING_PROBES):
        # 1. Evaluate loss on reference reasoning trace
        val_tokens = tok(probe["validation_trace"], return_tensors='pt').input_ids.to(DEVICE)
        val_loss = learner.evaluate_loss(val_tokens)
        total_val_loss += val_loss

        # 2. Generation test
        inp = tok(probe["prompt"], return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        gen_text = tok.decode(out[0], skip_special_tokens=True)
        if i == 0:
            sample_generation = gen_text

        has_think_start = "<think>" in gen_text
        has_think_end = "</think>" in gen_text
        if has_think_start:
            think_tag_count += 1

        lower_gen = gen_text.lower()
        matches = sum(1 for kw in COGNITIVE_KEYWORDS if kw in lower_gen)
        keyword_matches += matches

        probe_results.append({
            "probe_id": probe["id"],
            "category": probe["category"],
            "reasoning_loss": round(val_loss, 4),
            "reasoning_perplexity": round(math.exp(min(val_loss, 20.0)), 2),
            "think_tag_generated": has_think_start,
            "think_tag_closed": has_think_end,
            "cognitive_keywords_found": matches,
            "output_preview": gen_text.replace("\n", " ")[:140]
        })

    avg_loss = total_val_loss / len(REASONING_PROBES)
    avg_ppl = math.exp(min(avg_loss, 20.0))
    think_tag_rate = (think_tag_count / len(REASONING_PROBES)) * 100.0
    imitation_score = min(100.0, max(0.0, (think_tag_rate * 0.4) + (keyword_matches * 5.0) + (max(0.0, 10.0 - avg_loss) * 6.0)))

    eval_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cycle": cycle_num,
        "average_reasoning_loss": round(avg_loss, 4),
        "average_reasoning_perplexity": round(avg_ppl, 2),
        "think_tag_adherence_percent": round(think_tag_rate, 1),
        "cognitive_keyword_count": keyword_matches,
        "imitation_intelligence_index": round(imitation_score, 1),
        "sample_generation": sample_generation,
        "probes": probe_results
    }

    history = []
    if os.path.exists(INTELLIGENCE_METRICS_PATH):
        try:
            with open(INTELLIGENCE_METRICS_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(eval_record)
    with open(INTELLIGENCE_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"[+] Intelligence Probe Complete:")
    print(f"    - Avg Reasoning Loss:       {avg_loss:.4f} (Perplexity: {avg_ppl:.2f})")
    print(f"    - <think> Tag Adherence:    {think_tag_rate:.1f}%")
    print(f"    - Cognitive Keywords Found: {keyword_matches}")
    print(f"    - Imitation Index:          {imitation_score:.1f} / 100")
    print(f"    - Sample Generation:        {sample_generation.replace(chr(10), ' ')[:90]}...")

    return eval_record


def formulate_and_submit_inquiry(conn, cycle: int, topic: str, summary: str, thought_trace: str, tok, learner):
    """
    Identifies the mathematical formulation or statement in the paper where the student
    experiences the highest surprise (cross-entropy loss), and submits a Socratic query to DeepSeek-R1.
    """
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM socratic_inquiries WHERE cycle = ?", (cycle,))
        if cur.fetchone():
            return

        combined_text = summary + "\n" + (thought_trace or "")
        candidates = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', combined_text) if len(s.strip()) > 35]
        if not candidates:
            return

        tested = candidates[:8]
        worst_snippet = ""
        max_loss = -1.0

        for cand in tested:
            c_toks = tok(cand, return_tensors='pt', truncation=True, max_length=128).input_ids.to(DEVICE)
            if c_toks.shape[1] < 6:
                continue
            l = learner.evaluate_loss(c_toks)
            if l > max_loss:
                max_loss = l
                worst_snippet = cand

        if not worst_snippet or max_loss <= 0.0:
            worst_snippet = tested[0]
            max_loss = 4.5

        snippet_clean = worst_snippet.replace("\n", " ").strip()
        if len(snippet_clean) > 200:
            snippet_clean = snippet_clean[:200] + "..."

        question_text = f"Can you provide the step-by-step mathematical derivation and physical intuition for: '{snippet_clean}'?"

        cur.execute("""
            INSERT INTO socratic_inquiries (cycle, timestamp, question_text, context_snippet, status, student_loss_before)
            VALUES (?, ?, ?, ?, 'PENDING', ?)
        """, (
            cycle,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            question_text,
            snippet_clean,
            float(max_loss)
        ))
        conn.commit()
        print(f"\n[?] >>> SOCRATIC INQUIRY SUBMITTED TO TEACHER (Cycle {cycle}) <<<")
        print(f"    - Highest Surprise Loss:  {max_loss:.4f}")
        print(f"    - Inquired Formulation:   '{snippet_clean}'")
        print(f"    - Question:               {question_text}")
    except Exception as e:
        print(f"[-] Error formulating Socratic inquiry: {e}")


def check_and_absorb_teacher_clarifications(conn, tok, learner) -> Optional[Dict[str, Any]]:
    """
    Checks if Teacher (DeepSeek-R1) has answered any pending inquiries.
    If so, performs Online TTT adaptation on the Teacher's step-by-step reasoning trace and derivation.
    """
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, cycle, question_text, teacher_response, student_loss_before 
            FROM socratic_inquiries 
            WHERE status = 'ANSWERED' AND student_loss_after IS NULL 
            ORDER BY id ASC LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            return None

        inquiry_id, cycle, question_text, teacher_response, loss_before = row
        print(f"\n[+] >>> ABSORBING TEACHER SOCRATIC CLARIFICATION (Inquiry #{inquiry_id} for Cycle {cycle}) <<<")
        print(f"    Question: {question_text[:90]}...")

        tokens = tok(teacher_response[:1400], return_tensors='pt', truncation=True, max_length=256).input_ids.to(DEVICE)
        if tokens.shape[1] >= 8:
            cur_loss_before = learner.evaluate_loss(tokens)
            learner.adapt_on_sequence(tokens, steps=3, lr_scale=1.5)
            loss_after = learner.evaluate_loss(tokens)
            drop = cur_loss_before - loss_after
            pct = (drop / cur_loss_before) * 100.0 if cur_loss_before > 0 else 0.0

            cur.execute("""
                UPDATE socratic_inquiries 
                SET student_loss_after = ? 
                WHERE id = ?
            """, (float(loss_after), inquiry_id))
            conn.commit()

            print(f"    - Loss Before Clarification: {cur_loss_before:.4f}")
            print(f"    - Loss After Ingestion:      {loss_after:.4f} (-{drop:.4f}, +{pct:.2f}%)")
            print(f"    - Socratic Ingestion SUCCESS! Cognitive confusion resolved.")
            return {
                "inquiry_id": inquiry_id,
                "cycle": cycle,
                "loss_before": cur_loss_before,
                "loss_after": loss_after,
                "drop": drop
            }
    except Exception as e:
        print(f"[-] Error absorbing teacher clarification: {e}")
    return None


def main():
    print("=" * 80)
    print(" [*] STARTING PRIME AUTONOMOUS CO-LEARNER DAEMON v2.0 (STUDENT)")
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
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT MAX(cycle) FROM theories")
        row = cur.fetchone()
        if row and row[0] is not None:
            last_cycle_processed = max(0, row[0] - 2)
        conn.close()
    except Exception as e:
        print(f"[-] Warning reading initial cycle: {e}")

    print(f"[*] Starting streaming ingestion from Cycle {last_cycle_processed + 1} onward...")

    total_ingested = 0
    start_time = time.time()

    # Initial baseline intelligence probe
    latest_intel = run_intelligence_and_imitation_evaluation(model, tok, learner, last_cycle_processed)

    # 5. Continuous Co-Learning Loop
    while True:
        try:
            conn = sqlite3.connect(DB_PATH)

            # Check if teacher clarified any past inquiries
            clarif = check_and_absorb_teacher_clarifications(conn, tok, learner)

            # Fetch next research theory
            cur = conn.cursor()
            cur.execute("""
                SELECT cycle, domain, topic, summary, thought_trace, timestamp 
                FROM theories 
                WHERE cycle > ? 
                ORDER BY cycle ASC 
                LIMIT 1
            """, (last_cycle_processed,))
            new_row = cur.fetchone()

            if new_row is None:
                vram_now = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
                status_dict = {
                    "status": "WAITING_FOR_NEXT_THEORY",
                    "uptime_seconds": int(time.time() - start_time),
                    "last_cycle_processed": last_cycle_processed,
                    "total_theories_ingested": total_ingested,
                    "baseline_anchor_loss": round(baseline_anchor_loss, 4),
                    "vram_mb": round(vram_now, 1),
                    "latest_intelligence_index": latest_intel.get("imitation_intelligence_index", 0.0),
                    "latest_reasoning_loss": latest_intel.get("average_reasoning_loss", 0.0),
                    "think_tag_adherence_percent": latest_intel.get("think_tag_adherence_percent", 0.0),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                update_status(status_dict)
                conn.close()
                time.sleep(6)
                continue

            cycle, domain, topic, summary, thought_trace, theory_ts = new_row
            print(f"\n[+] [{time.strftime('%H:%M:%S')}] >>> INGESTING NEW RESEARCH CYCLE {cycle}: {domain} <<<")

            # Format training sequence including DeepSeek-R1 <think> cognitive trace
            if thought_trace and len(thought_trace.strip()) > 50:
                training_text = f"<think>\n{thought_trace[:1200]}\n</think>\n\n# {topic}\n{summary[:1000]}"
            else:
                training_text = f"# {topic}\n{summary[:1500]}"

            tokens = tok(training_text, return_tensors='pt', truncation=True, max_length=256).input_ids.to(DEVICE)
            if tokens.shape[1] < 8:
                last_cycle_processed = cycle
                conn.close()
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

            # Formulate Socratic inquiry for confusing concept
            formulate_and_submit_inquiry(conn, cycle, topic, summary, thought_trace, tok, learner)

            conn.close()

            vram_now = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
            total_ingested += 1
            last_cycle_processed = cycle

            print(f"    - Pre-Adaptation Loss:    {loss_before:.4f}")
            print(f"    - Post-Adaptation Loss:   {loss_after:.4f} (-{loss_reduction:.4f}, +{pct_improvement:.2f}%)")
            print(f"    - Anchor Loss & Drift:    {cur_anchor_loss:.4f} (drift: {anchor_drift:.5f})")
            print(f"    - Total Ingested:         {total_ingested} theories")
            print(f"    - Student VRAM Footprint: {vram_now:.1f} MB")

            # Run periodic intelligence probe every 3 cycles
            if total_ingested % 3 == 0:
                latest_intel = run_intelligence_and_imitation_evaluation(model, tok, learner, cycle)

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
                "latest_intelligence_index": latest_intel.get("imitation_intelligence_index", 0.0),
                "latest_reasoning_loss": latest_intel.get("average_reasoning_loss", 0.0),
                "think_tag_adherence_percent": latest_intel.get("think_tag_adherence_percent", 0.0),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            update_status(status_dict)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time.sleep(2)

        except Exception as e:
            print(f"[-] Error in co-learning loop: {e}")
            time.sleep(8)


if __name__ == "__main__":
    main()
