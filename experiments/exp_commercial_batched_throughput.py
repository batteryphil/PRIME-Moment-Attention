#!/usr/bin/env python3
"""
Crucible 1: The 'Server-Killer' Batched Throughput & OOM Horizon Benchmark
==========================================================================
Evaluates the enterprise server economics of PRIME by scaling batch concurrency
from B=1 to B=64 across long context lengths (L=8,192, 16,384).

Compares:
  1. Baseline Qwen2.5-Coder-1.5B (Standard Full Softmax Attention)
  2. Stage 7 Hybrid PRIME (25% Boundary Softmax, 75% Interior PRIME Trunk)

Measures:
  - Peak VRAM footprint (GB)
  - Exact OOM boundary point (maximum serving capacity per GPU)
  - Time-To-First-Token (TTFT in seconds)
  - Autoregressive Decode Throughput (tokens/sec across the batch)
"""

import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["MIOPEN_USER_DB_PATH"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_LOCK_FILE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_DEBUG_DISABLE_LOCK"] = "1"
os.environ["TRITON_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/.triton"
os.environ["PYTHONUNBUFFERED"] = "1"

import sys
sys.path.append("/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/src")

import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gumbel_qwen_distillation import convert_qwen_to_stage7_hybrid

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] Using {device}")

model_path = "/home/phil/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

CONTEXT_LENGTHS = [8192, 16384]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
GEN_STEPS = 6

def test_single_batch(model, B, L):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    input_ids = torch.randint(100, 10000, (B, L), dtype=torch.long, device=device)
    attention_mask = torch.ones((B, L), dtype=torch.long, device=device)
    
    try:
        # 1. Prefill / TTFT
        t0 = time.time()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ttft = time.time() - t0
        
        # 2. Decode Throughput
        t_gen_start = time.time()
        curr_token = next_token
        with torch.no_grad():
            for _ in range(GEN_STEPS):
                out_step = model(curr_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = out_step.past_key_values
                curr_token = out_step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        t_gen = time.time() - t_gen_start
        total_gen_tokens = B * GEN_STEPS
        tok_per_sec = total_gen_tokens / (t_gen + 1e-6)
        
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        
        del outputs, past_key_values, input_ids, curr_token
        torch.cuda.empty_cache()
        
        return {
            "status": "PASS",
            "batch_size": B,
            "context_length": L,
            "peak_vram_gb": round(peak_vram, 2),
            "ttft_sec": round(ttft, 3),
            "tok_per_sec_total": round(tok_per_sec, 1),
            "tok_per_sec_per_user": round(tok_per_sec / B, 1)
        }
        
    except (torch.OutOfMemoryError, RuntimeError) as e:
        err_msg = str(e)
        torch.cuda.empty_cache()
        return {
            "status": "OOM",
            "batch_size": B,
            "context_length": L,
            "peak_vram_gb": "OOM (>15.92 GB)",
            "ttft_sec": None,
            "tok_per_sec_total": 0.0,
            "tok_per_sec_per_user": 0.0
        }

out_dir = "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/dossier/telemetry"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "commercial_batched_throughput_results.json")

results = {
    "benchmark": "Server-Killer Batched Throughput & OOM Horizon",
    "gpu": "AMD Radeon Graphics (15.92 GB VRAM)",
    "data": {}
}

if os.path.exists(out_file):
    try:
        with open(out_file, "r") as f:
            existing = json.load(f)
            if "data" in existing:
                results["data"] = existing["data"]
                print("[+] Loaded existing benchmark telemetry.")
    except Exception as e:
        print(f"[!] Warning loading existing telemetry: {e}")

print("\n" + "="*85)
print("COMMENCING CRUCIBLE 1: BATCHED CONCURRENCY SWEEP")
print("=====================================================================================")

for L in CONTEXT_LENGTHS:
    print(f"\n[+] Testing Context Length L = {L} tokens...")
    if f"L_{L}" not in results["data"]:
        results["data"][f"L_{L}"] = {
            "baseline_softmax": [],
            "stage7_hybrid": []
        }
    
    # --- Condition 1: Baseline Softmax ---
    if not results["data"][f"L_{L}"].get("baseline_softmax"):
        print(f"\n--- Loading Baseline Softmax Model (L={L}) ---")
        model_baseline = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True
        ).to(device)
        model_baseline.eval()
        
        for B in BATCH_SIZES:
            print(f"[*] Baseline Softmax: B={B:2d}, L={L:5d}...", end="", flush=True)
            res = test_single_batch(model_baseline, B, L)
            print(f" -> {res['status']} | VRAM: {res['peak_vram_gb']} | TTFT: {res['ttft_sec']}s | Tok/s: {res['tok_per_sec_total']}")
            results["data"][f"L_{L}"]["baseline_softmax"].append(res)
            if res["status"] == "OOM":
                print(f"[!] Baseline Softmax hit hardware memory ceiling at B={B}.")
                for higher_B in [b for b in BATCH_SIZES if b > B]:
                    results["data"][f"L_{L}"]["baseline_softmax"].append({
                        "status": "OOM",
                        "batch_size": higher_B,
                        "context_length": L,
                        "peak_vram_gb": "OOM (>15.92 GB)",
                        "ttft_sec": None,
                        "tok_per_sec_total": 0.0,
                        "tok_per_sec_per_user": 0.0
                    })
                break
                
        del model_baseline
        torch.cuda.empty_cache()
    else:
        print(f"[+] Reusing measured Baseline Softmax telemetry for L={L}.")

    # Reset stage7_hybrid data to evaluate with chunked linear prefill
    results["data"][f"L_{L}"]["stage7_hybrid"] = []
    
    # --- Condition 2: Stage 7 Hybrid PRIME ---
    print(f"\n--- Loading Stage 7 Hybrid PRIME Model (L={L}) ---")
    model_hybrid = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to(device)
    model_hybrid, _, trunk = convert_qwen_to_stage7_hybrid(model_hybrid)
    for idx in trunk:
        model_hybrid.model.layers[idx].self_attn.router.freeze(
            fixed_orders=torch.ones(12, dtype=torch.long, device=device) # O1
        )
    model_hybrid.eval()
    
    for B in BATCH_SIZES:
        print(f"[*] Stage 7 Hybrid:  B={B:2d}, L={L:5d}...", end="", flush=True)
        res = test_single_batch(model_hybrid, B, L)
        print(f" -> {res['status']} | VRAM: {res['peak_vram_gb']} | TTFT: {res['ttft_sec']}s | Tok/s: {res['tok_per_sec_total']}")
        results["data"][f"L_{L}"]["stage7_hybrid"].append(res)
        if res["status"] == "OOM":
            print(f"[!] Stage 7 Hybrid hit hardware memory ceiling at B={B}.")
            for higher_B in [b for b in BATCH_SIZES if b > B]:
                results["data"][f"L_{L}"]["stage7_hybrid"].append({
                    "status": "OOM",
                    "batch_size": higher_B,
                    "context_length": L,
                    "peak_vram_gb": "OOM (>15.92 GB)",
                    "ttft_sec": None,
                    "tok_per_sec_total": 0.0,
                    "tok_per_sec_per_user": 0.0
                })
            break
            
    del model_hybrid
    torch.cuda.empty_cache()

out_dir = "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/dossier/telemetry"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "commercial_batched_throughput_results.json")
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*85)
print(f"[+] CRUCIBLE 1 COMPLETE! Telemetry saved to: {out_file}")
print("="*85)
