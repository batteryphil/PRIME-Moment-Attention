#!/usr/bin/env python3
"""
Execution of the 10,000-Step Code Formatting Distillation Curriculum
===================================================================
Distills knowledge from full Softmax Qwen2.5-Coder-1.5B into Stage 7 Hybrid PRIME
(25% Softmax, 75% Gumbel-Softmax STE PRIME Trunk) on an AMD Radeon GPU.
"""

import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["MIOPEN_USER_DB_PATH"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_LOCK_FILE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_DEBUG_DISABLE_LOCK"] = "1"
os.environ["TRITON_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/.triton"

import sys
sys.path.append("/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/src")

import time
import json
import copy
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from gumbel_qwen_distillation import (
    convert_qwen_to_stage7_hybrid,
    evaluate_indentation_fidelity
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] Using {device}")

model_path = "/home/phil/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a"

print("[1] Loading Tokenizer and Model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Teacher (Frozen)
print("[2] Initializing Teacher Model (Full Softmax Baseline)...")
teacher = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    local_files_only=True
).to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

# Load Student (Stage 7 Hybrid)
print("[3] Initializing Student Model (Stage 7 Hybrid PRIME)...")
student = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    local_files_only=True
).to(device)

student, boundary_layers, trunk_layers = convert_qwen_to_stage7_hybrid(
    student,
    tau_0=1.0,
    tau_min=0.05,
    anneal_steps=2000
)

# 4. Prepare Distillation Dataset (50% Algorithmic Python + 50% Dense Reasoning)
print("[4] Assembling Curriculum Corpus...")
code_parquet = "/home/phil/.cache/huggingface/hub/datasets--iamtarun--python_code_instructions_18k_alpaca/snapshots/7cae181e29701a8663a07a3ea43c8e105b663ba1/data/train-00000-of-00001-8b6e212f3e1ece96.parquet"
reasoning_parquet = "/home/phil/.cache/huggingface/hub/datasets--roneneldan--TinyStories/snapshots/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/data/train-00000-of-00004-2d5a1467fff1081b.parquet"

code_table = pq.read_table(code_parquet)
reasoning_table = pq.read_table(reasoning_parquet)

code_texts = [f"{ins}\n{out}" for ins, out in zip(code_table["instruction"][:500].to_pylist(), code_table["output"][:500].to_pylist())]
reasoning_texts = reasoning_table["text"][:500].to_pylist()

print(f"[+] Loaded {len(code_texts)} Python code samples and {len(reasoning_texts)} reasoning text samples.")

# Interleave 50/50
corpus = []
for c, r in zip(code_texts, reasoning_texts):
    corpus.append(c)
    corpus.append(r)

# 5. Baseline Evaluation before distillation
print("\n" + "="*80)
print("EVALUATING ZERO-SHOT STAGE 7 HYBRID BEFORE DISTILLATION")
print("="*80)
pre_eval = evaluate_indentation_fidelity(student, tokenizer, device=device)
print(f"[*] Pre-Distillation Mean Indentation Score: {pre_eval['mean_indent_score']:.4f}")
for d in pre_eval["details"]:
    print(f"    - {d['test_name']}: Score = {d['indent_score']:.4f}")

# 6. Distillation Engine Setup
# Train router projections in trunk in float32 to learn discrete Gumbel STE allocations
trainable_params = []
for idx in trunk_layers:
    student.model.layers[idx].self_attn.router.float()
    trainable_params.extend(list(student.model.layers[idx].self_attn.router.parameters()))

optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)

BATCH_SIZE = 2
SEQ_LEN = 128
TOTAL_STEPS = 100 # Verification run of curriculum steps with active telemetry
LOG_INTERVAL = 10

print(f"\n[5] Launching Distillation Run ({TOTAL_STEPS} curriculum steps, effective batch={BATCH_SIZE}, seq_len={SEQ_LEN})...")
student.train()

telemetry_log = []
t_start = time.time()

for step in range(1, TOTAL_STEPS + 1):
    # Sample batch
    batch_idx = ((step - 1) * BATCH_SIZE) % (len(corpus) - BATCH_SIZE)
    batch_texts = corpus[batch_idx: batch_idx + BATCH_SIZE]
    
    enc = tokenizer(
        batch_texts,
        max_length=SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    # Forward Teacher (No grad)
    with torch.no_grad():
        teacher_out = teacher(input_ids, attention_mask=attention_mask)
        teacher_logits = teacher_out.logits.detach()

    # Forward Student with Gumbel STE
    student_out = student(input_ids, attention_mask=attention_mask)
    student_logits = student_out.logits

    # Distillation Loss (KL Div on Logits + CrossEntropy on Next Token in float32)
    T_distill = 2.0
    s_log_probs = F.log_softmax(student_logits.float() / T_distill, dim=-1)
    t_log_probs = F.log_softmax(teacher_logits.float() / T_distill, dim=-1)
    
    # Token-level masked KD loss (normalized per active token)
    kd_elem = F.kl_div(s_log_probs[:, :-1, :], t_log_probs[:, :-1, :], log_target=True, reduction="none").sum(dim=-1) * (T_distill ** 2)
    pad_mask = attention_mask[:, 1:].float()
    loss_kd = (kd_elem * pad_mask).sum() / (pad_mask.sum() + 1e-6)

    labels = input_ids[:, 1:].contiguous()
    shift_logits = student_logits[:, :-1, :].contiguous().float()
    loss_lm = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), labels.view(-1), ignore_index=tokenizer.pad_token_id)

    total_loss = 0.7 * loss_kd + 0.3 * loss_lm

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()

    # Step annealing scheduler on all trunk routers
    active_tau = 1.0
    for idx in trunk_layers:
        active_tau = student.model.layers[idx].self_attn.router.scheduler.step()

    if step % LOG_INTERVAL == 0 or step == TOTAL_STEPS:
        # Measure head allocation distribution in mid-trunk layer (layer 12)
        sample_layer = student.model.layers[12].self_attn
        with torch.no_grad():
            sample_logits = sample_layer.router.router_proj.bias.view(12, 3)
            probs = F.softmax(sample_logits / max(active_tau, 0.05), dim=-1)
            o0_pct = float(probs[:, 0].mean().item() * 100)
            o1_pct = float(probs[:, 1].mean().item() * 100)
            o2_pct = float(probs[:, 2].mean().item() * 100)

        dt = time.time() - t_start
        print(f"Step {step:4d}/{TOTAL_STEPS} | Loss: {total_loss.item():.4f} (KD: {loss_kd.item():.4f}, LM: {loss_lm.item():.4f}) | Tau: {active_tau:.4f} | Layer 12 Routing: O0={o0_pct:.1f}%, O1={o1_pct:.1f}%, O2={o2_pct:.1f}% | Time: {dt:.1f}s")
        
        telemetry_log.append({
            "step": step,
            "total_loss": round(total_loss.item(), 4),
            "loss_kd": round(loss_kd.item(), 4),
            "loss_lm": round(loss_lm.item(), 4),
            "tau": round(active_tau, 4),
            "routing_distribution": {
                "O0_percent": round(o0_pct, 1),
                "O1_percent": round(o1_pct, 1),
                "O2_percent": round(o2_pct, 1)
            }
        })

# 7. Post-Distillation Evaluation
print("\n" + "="*80)
print("EVALUATING STAGE 7 HYBRID POST-DISTILLATION")
print("="*80)
post_eval = evaluate_indentation_fidelity(student, tokenizer, device=device)
print(f"[+] Post-Distillation Mean Indentation Score: {post_eval['mean_indent_score']:.4f}")
for d in post_eval["details"]:
    print(f"    - {d['test_name']}: Score = {d['indent_score']:.4f} | Snippet: {d['generated_snippet']}")

# 8. Save Telemetry Dossier
out_dir = "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/dossier/telemetry"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "gumbel_qwen_distillation_curriculum_results.json")
with open(out_file, "w") as f:
    json.dump({
        "model": "Qwen2.5-Coder-1.5B-Instruct",
        "hybrid_architecture": "25% Softmax (7 boundary), 75% PRIME (21 trunk)",
        "pre_distillation_indent_score": pre_eval["mean_indent_score"],
        "post_distillation_indent_score": post_eval["mean_indent_score"],
        "training_telemetry": telemetry_log,
        "indentation_eval_details": post_eval["details"]
    }, f, indent=2)

print(f"\n[+] Distillation curriculum telemetry saved to: {out_file}")
print("="*80)
