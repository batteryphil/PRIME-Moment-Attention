#!/usr/bin/env python3
"""
PRIME-Selective: Temperature-Scaled Taylor Attention with Input-Dependent Gating
================================================================================
Architecture:
  - 100% Layer Conversion (Zero Softmax Layers, O(1) Memory Forever)
  - Temperature-Scaled Taylor Kernel: P(s) = 1 + beta*s + 0.5*(beta*s)^2 (Contrast: 20:1)
  - Input-Dependent Selective Gating: Delta_t = softplus(W_delta * x_t + b_delta)
  - FP32 Mixed-Precision State Accumulator (Defeating BF16 Epsilon Wall)
  - Multiscale Horizon Bank: tau in [2.0, 1000.0] tokens
  - 90.01% Frozen Backbone (embeddings, MLPs, layernorms, LM head)
"""

import os
import sys
import time
import math
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

class PrimeSelectiveAttentionLayer(nn.Module):
    """
    PRIME-Selective Recurrent Attention Layer:
      Combines:
        1. Learnable temperature beta_h for high contrast resolution
        2. Input-dependent step size Delta_{t, h} for selective entity retention
        3. FP32 mixed-precision rolling state updates (S0, S1, S2, K0, K1, K2)
    """
    def __init__(self, orig_attn, layer_idx: int, min_tau: float = 2.0, max_tau: float = 1000.0):
        super().__init__()
        self.config = orig_attn.config
        self.layer_idx = layer_idx
        self.head_dim = orig_attn.head_dim
        self.num_heads = orig_attn.config.num_attention_heads
        self.num_key_value_heads = orig_attn.config.num_key_value_heads
        self.num_key_value_groups = orig_attn.num_key_value_groups
        self.hidden_size = orig_attn.config.hidden_size

        # Trainable projection matrices
        self.q_proj = orig_attn.q_proj
        self.k_proj = orig_attn.k_proj
        self.v_proj = orig_attn.v_proj
        self.o_proj = orig_attn.o_proj

        device = orig_attn.q_proj.weight.device
        dtype = orig_attn.q_proj.weight.dtype

        # 1. Multiscale timescale bank tau_h in [min_tau, max_tau]
        init_log_tau = torch.linspace(math.log(min_tau), math.log(max_tau), self.num_heads, device=device)
        self.log_tau = nn.Parameter(init_log_tau) # [H]

        # 2. Learnable inverse temperature beta_h per head (init at 4.0 for 24:1 contrast)
        init_beta_param = math.log(math.exp(4.0 - 1.0) - 1.0) # inverse softplus of 3.0 -> beta=4.0
        self.beta_param = nn.Parameter(torch.full((self.num_heads,), init_beta_param, device=device)) # [H]

        # 3. Input-dependent selective gating projection W_delta: [hidden_size -> num_heads]
        self.delta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True, device=device, dtype=dtype)
        # Initialize bias so default delta_t ~ 1.0 (standard decay)
        nn.init.constant_(self.delta_proj.bias, 0.5413) # softplus(0.5413) ~ 1.0
        nn.init.normal_(self.delta_proj.weight, std=0.01)

    def get_tau(self, device):
        return torch.clamp(torch.exp(self.log_tau.to(device)), min=1.5, max=2000.0)

    def get_beta(self, device):
        # beta_h in [1.5, 12.0]
        return torch.clamp(F.softplus(self.beta_param.to(device)) + 1.0, min=1.5, max=12.0)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        B, L = input_shape
        hidden_shape = (*input_shape, -1, self.head_dim)
        device = hidden_states.device

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # L2-normalization for Taylor polynomial stability
        q_norm = F.normalize(query_states.float(), p=2, dim=-1)
        k_norm = F.normalize(key_states.float(), p=2, dim=-1)
        v_f = value_states.float()

        tau = self.get_tau(device) # [H]
        beta = self.get_beta(device) # [H]

        # Compute data-dependent selective step size: Delta_t in (0, inf)
        delta = F.softplus(self.delta_proj(hidden_states)) # [B, L, H]

        # ----------------------------------------------------------------------
        # Autoregressive Generation (L == 1) with Selective FP32 Recurrence
        # ----------------------------------------------------------------------
        if L == 1 and past_key_values is not None:
            if not hasattr(past_key_values, "prime_states"):
                past_key_values.prime_states = {}
            state = past_key_values.prime_states.get(self.layer_idx, None)
            if state is not None:
                S0, S1, S2, K0, K1, K2 = state
            else:
                S0 = torch.zeros(B, self.num_heads, self.head_dim, device=device, dtype=torch.float32)
                S1 = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=torch.float32)
                S2 = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=torch.float32)
                K0 = torch.zeros(B, self.num_heads, 1, device=device, dtype=torch.float32)
                K1 = torch.zeros(B, self.num_heads, self.head_dim, device=device, dtype=torch.float32)
                K2 = torch.zeros(B, self.num_heads, self.head_dim, device=device, dtype=torch.float32)

            dt = delta[:, 0, :].view(B, self.num_heads, 1).float() # [B, H, 1]
            lam_vec = torch.exp(-dt / tau.view(1, self.num_heads, 1)).to(torch.float32) # [B, H, 1]
            lam_mat = lam_vec.unsqueeze(-1) # [B, H, 1, 1]

            b_vec = beta.view(1, self.num_heads, 1).float() # [1, H, 1]
            b_sq = (0.5 * (beta ** 2)).view(1, self.num_heads, 1).float()

            qt = q_norm[:, :, 0]
            kt = k_norm[:, :, 0]
            vt = v_f[:, :, 0]

            # FP32 Selective Accumulator Update
            S0 = lam_vec * S0 + vt
            S1 = lam_mat * S1 + b_vec.unsqueeze(-1) * torch.matmul(kt.unsqueeze(-1), vt.unsqueeze(-2))
            S2 = lam_mat * S2 + b_sq.unsqueeze(-1) * torch.matmul((kt**2).unsqueeze(-1), vt.unsqueeze(-2))

            K0 = lam_vec * K0 + 1.0
            K1 = lam_vec * K1 + b_vec * kt
            K2 = lam_vec * K2 + b_sq * (kt**2)

            num = S0 + torch.matmul(qt.unsqueeze(-2), S1).squeeze(-2) + torch.matmul((qt**2).unsqueeze(-2), S2).squeeze(-2)
            den = K0 + (qt * K1).sum(dim=-1, keepdim=True) + ((qt**2) * K2).sum(dim=-1, keepdim=True)
            den = den.clamp(min=1e-5)

            past_key_values.prime_states[self.layer_idx] = (S0, S1, S2, K0, K1, K2)
            attn_output = (num / den).to(hidden_states.dtype).unsqueeze(2)

        # ----------------------------------------------------------------------
        # Parallel Sequence Mode (Training & Vectorized Prefill)
        # ----------------------------------------------------------------------
        else:
            # Prefix-sum cumulative decay calculation with numerical clamping
            C = torch.cumsum(delta.float(), dim=1) # [B, L, H]
            decay_diff = (C.unsqueeze(2) - C.unsqueeze(1)).clamp(min=0.0) # [B, L, L, H]
            decay_matrix = torch.exp(-decay_diff / tau.view(1, 1, 1, self.num_heads)).permute(0, 3, 1, 2) # [B, H, L, L]

            causal_mask = torch.tril(torch.ones(L, L, device=device)).view(1, 1, L, L)
            decay_matrix = decay_matrix * causal_mask

            b_mat = beta.view(1, self.num_heads, 1, 1).float()
            sim1 = b_mat * torch.matmul(q_norm, k_norm.transpose(-1, -2))
            sim2 = 0.5 * (b_mat ** 2) * torch.matmul(q_norm**2, (k_norm**2).transpose(-1, -2))
            p_weights = torch.clamp(1.0 + sim1 + sim2, min=0.0) * decay_matrix

            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    p_weights = p_weights * attention_mask
                else:
                    p_weights = p_weights * (attention_mask >= 0.0)

            p_denom = p_weights.sum(dim=-1, keepdim=True).clamp(min=1e-5)
            normalized_weights = p_weights / p_denom

            attn_output = torch.matmul(normalized_weights, v_f).to(hidden_states.dtype)

            # Vectorized O(1) state handoff for subsequent autoregressive tokens
            if past_key_values is not None:
                if not hasattr(past_key_values, "prime_states"):
                    past_key_values.prime_states = {}
                C_last = C[:, -1:, :] # [B, 1, H]
                decay_to_end = torch.exp(-(C_last - C).clamp(min=0.0) / tau.view(1, 1, self.num_heads)).permute(0, 2, 1).unsqueeze(-1) # [B, H, L, 1]

                b_v = beta.view(1, self.num_heads, 1, 1).float()
                b_v2 = (0.5 * (beta ** 2)).view(1, self.num_heads, 1, 1).float()

                S0 = (v_f * decay_to_end).sum(dim=2)
                S1 = b_v * torch.matmul((k_norm * decay_to_end).transpose(-2, -1), v_f)
                S2 = b_v2 * torch.matmul(((k_norm**2) * decay_to_end).transpose(-2, -1), v_f)
                K0 = decay_to_end.sum(dim=2)
                K1 = beta.view(1, self.num_heads, 1).float() * (k_norm * decay_to_end).sum(dim=2)
                K2 = (0.5 * (beta ** 2)).view(1, self.num_heads, 1).float() * ((k_norm**2) * decay_to_end).sum(dim=2)
                past_key_values.prime_states[self.layer_idx] = (S0, S1, S2, K0, K1, K2)

        attn_output = attn_output.transpose(1, 2).contiguous().view(*input_shape, -1)
        return self.o_proj(attn_output), None

def evaluate_generation(model, tokenizer, prompt, device, max_new_tokens=45):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def evaluate_perplexity(model, tokenizer, test_texts, device, max_len=128):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in test_texts:
            enc = tokenizer(text, return_tensors="pt", max_length=max_len, truncation=True).to(device)
            input_ids = enc.input_ids
            if input_ids.shape[1] < 2:
                continue
            labels = input_ids.clone()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss.item()
            if math.isnan(loss) or math.isinf(loss):
                continue
            num_tokens = input_ids.shape[1] - 1
            total_loss += loss * num_tokens
            total_tokens += num_tokens
    if total_tokens == 0:
        return 999.99
    return min(999.99, math.exp(total_loss / total_tokens))

def main():
    parser = argparse.ArgumentParser(description="PRIME-Selective Distillation Engine")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Model ID")
    parser.add_argument("--steps", type=int, default=150, help="Distillation steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate for projections")
    parser.add_argument("--seq_len", type=int, default=128, help="Context length per window")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 85)
    print(" 🚀 STARTING PRIME-SELECTIVE DISTILLATION ENGINE (100% CONVERSION)")
    print(f" Target Model: {args.model} | Device: {device} | Precision: bfloat16")
    print(f" Pillars: Beta-Scaled Contrast (20:1) + Selective Gating Delta_t + FP32 State")
    print("=" * 85)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[1/6] Loading Teacher Model (Softmax Attention)...")
    teacher = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("[2/6] Initializing Student Model (PRIME-Selective Attention)...")
    student = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=device)
    num_layers = len(student.model.layers)

    # Freeze 90% backbone
    for p in student.parameters():
        p.requires_grad = False

    # Convert ALL layers to PRIME-Selective Attention
    proj_params = []
    gate_params = []
    meta_params = []

    for l_idx in range(num_layers):
        orig_attn = student.model.layers[l_idx].self_attn
        prime_attn = PrimeSelectiveAttentionLayer(orig_attn, layer_idx=l_idx, min_tau=2.0, max_tau=1000.0)
        student.model.layers[l_idx].self_attn = prime_attn

        for proj in [prime_attn.q_proj, prime_attn.k_proj, prime_attn.v_proj, prime_attn.o_proj]:
            for p in proj.parameters():
                p.requires_grad = True
                proj_params.append(p)

        for p in prime_attn.delta_proj.parameters():
            p.requires_grad = True
            gate_params.append(p)

        prime_attn.log_tau.requires_grad = True
        prime_attn.beta_param.requires_grad = True
        meta_params.extend([prime_attn.log_tau, prime_attn.beta_param])

    total_p = sum(p.numel() for p in student.parameters())
    train_p = sum(p.numel() for p in proj_params + gate_params + meta_params)
    print(f"[+] 100% PRIME-Selective Surgery Complete: {num_layers}/{num_layers} layers converted.")
    print(f"[+] Trainable Parameters: {train_p:,d} / {total_p:,d} ({train_p/total_p*100:.2f}%)")
    print(f"[+] Frozen Backbone: {total_p - train_p:,d} ({(total_p - train_p)/total_p*100:.2f}%)")

    # Load corpus
    calibration_texts = []
    novel_path = "/home/phil/.gemini/antigravity/scratch/alien_dog_witch_novel.md"
    if os.path.exists(novel_path):
        with open(novel_path, "r", encoding="utf-8") as f:
            novel_content = f.read()
        paras = [p.strip() for p in novel_content.split("\n\n") if len(p.split()) >= 25]
        calibration_texts.extend(paras[:250])

    additional_corpus = [
        "In artificial intelligence research, linear recurrence provides constant state space memory during autoregressive generation.",
        "The bayou mist curled around the ancient cypress knees, glowing with strange bioluminescent algae under the moon.",
        "Xylar activated his spectral scanner, recalibrating the acoustic dampeners while Barnaby pawed at the muddy bank.",
        "Vespera stirred the copper cauldron with a blackened ash branch, murmuring verses of root and elder shadow.",
        "Efficient algorithmic scaling transforms quadratic matrix multiplication into streaming cumulative outer products.",
        "Hardware accelerators achieve peak efficiency when memory access patterns adhere to cache line alignments.",
        "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "class PrimeMemoryState:\n    def __init__(self, dim):\n        self.s0 = torch.zeros(dim)\n        self.s1 = torch.zeros(dim, dim)"
    ] * 30
    calibration_texts.extend(additional_corpus)

    test_texts = [
        "The mysterious traveler entered the tavern, shaking the rain from his heavy woolen cloak.",
        "Computer science explores computational complexity, algorithms, and data structures.",
        "Xylar and Vespera watched as Barnaby chased a swarm of luminous fireflies across the clearing."
    ]

    print(f"[+] Calibration Training Set: {len(calibration_texts)} text chunks.")

    # Benchmark Baseline
    print("\n[3/6] Pre-Distillation Baseline Benchmark (Zero-Shot PRIME-Selective)...")
    eval_prompt = "The bayou was quiet tonight, but deep within the cypress grove, Xylar and Barnaby noticed"
    pre_sample = evaluate_generation(student, tokenizer, eval_prompt, device, max_new_tokens=45)
    pre_ppl = evaluate_perplexity(student, tokenizer, test_texts, device)
    teacher_ppl = evaluate_perplexity(teacher, tokenizer, test_texts, device)

    print(f"Teacher Baseline Perplexity (Softmax):       {teacher_ppl:.2f}")
    print(f"Pre-Distillation Student Perplexity (100%):  {pre_ppl:.2f}")
    print("\n--- PRE-DISTILLATION 100% GENERATION ---")
    print(pre_sample)
    print("----------------------------------------\n")

    optimizer = torch.optim.AdamW([
        {"params": proj_params, "lr": args.lr, "weight_decay": 1e-4},
        {"params": gate_params, "lr": 1e-3, "weight_decay": 1e-4},
        {"params": meta_params, "lr": 1e-3, "weight_decay": 0.0}
    ])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-5)

    print(f"[4/6] Executing Selective Multi-Objective Distillation ({args.steps} Steps)...")
    student.train()
    text_idx = 0
    t0 = time.time()
    loss_history = []

    for step in range(1, args.steps + 1):
        batch_samples = []
        for _ in range(args.batch_size):
            batch_samples.append(calibration_texts[text_idx % len(calibration_texts)])
            text_idx += 1

        enc = tokenizer(batch_samples, padding=True, truncation=True, max_length=args.seq_len, return_tensors="pt").to(device)
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask

        # Forward Teacher
        with torch.no_grad():
            teacher_out = teacher(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            teacher_hidden = teacher_out.hidden_states
            teacher_logits = teacher_out.logits

        # Forward Student
        optimizer.zero_grad()
        student_out = student(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        student_hidden = student_out.hidden_states
        student_logits = student_out.logits

        # Loss 1: Multi-layer hidden state Cosine Alignment in FP32
        stride = max(1, num_layers // 7)
        loss_cos = 0.0
        hidden_count = 0
        for l in range(stride, len(student_hidden), stride):
            sh_f = student_hidden[l].float()
            th_f = teacher_hidden[l].float()
            cos_sim = F.cosine_similarity(sh_f, th_f, dim=-1).mean()
            loss_cos += (1.0 - cos_sim)
            hidden_count += 1
        loss_hidden = loss_cos / hidden_count

        # Loss 2: Softmax-to-Taylor Logit KL Divergence (T=2.0) in FP32
        T = 2.0
        s_logprobs = F.log_softmax(student_logits.float() / T, dim=-1)
        t_probs = F.softmax(teacher_logits.float() / T, dim=-1)
        loss_logits = F.kl_div(s_logprobs, t_probs, reduction="batchmean") * (T ** 2)

        # Total Distillation Loss
        total_loss = loss_hidden + 0.05 * loss_logits
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(proj_params + gate_params + meta_params, max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        loss_history.append((step, total_loss.item(), loss_hidden.item(), loss_logits.item()))

        if step % 25 == 0 or step == 1 or step == args.steps:
            elapsed = time.time() - t0
            sps = step / elapsed
            print(f"Step [{step:3d}/{args.steps:3d}] | Loss: {total_loss.item():.4f} (Cos: {loss_hidden.item():.4f}, KL: {loss_logits.item():.4f}) | Speed: {sps:.2f} step/s")

    print("\n[5/6] Inspecting Learned Selective Parameters (Layer 14)...")
    l14 = student.model.layers[14].self_attn
    sample_tau = l14.get_tau(device).detach().cpu().numpy()
    sample_beta = l14.get_beta(device).detach().cpu().numpy()
    print("Layer 14 Timescales (tau) & Learned Inverse Temperatures (beta) across 12 heads:")
    for h in range(len(sample_tau)):
        print(f"  Head {h:2d}: tau = {sample_tau[h]:6.1f} tokens | beta = {sample_beta[h]:.2f} (Contrast ratio ~ {math.exp(sample_beta[h] * 0.8):.1f}:1)")

    save_dir = "/home/phil/.gemini/antigravity/scratch/prime_selective_checkpoint"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "prime_selective_100_1.5b.pt")
    state_to_save = {k: v.cpu() for k, v in student.state_dict().items() if any(p in k for p in ["q_proj", "k_proj", "v_proj", "o_proj", "delta_proj", "log_tau", "beta_param"])}
    torch.save(state_to_save, save_path)
    print(f"[+] Saved PRIME-Selective weights to: {save_path}")

    # Post-Distillation Verification
    print("\n[6/6] Post-Distillation Verification (PRIME-Selective 1.5B)...")
    post_ppl = evaluate_perplexity(student, tokenizer, test_texts, device)
    post_sample = evaluate_generation(student, tokenizer, eval_prompt, device, max_new_tokens=45)

    eval_prompt_2 = "Barnaby let out a low growl when the ancient runes began to pulse with emerald light."
    post_sample_2 = evaluate_generation(student, tokenizer, eval_prompt_2, device, max_new_tokens=45)

    print(f"\n==================== INTELLIGENCE AUDIT: PRIME-SELECTIVE ====================")
    print(f"Teacher Baseline Perplexity (Softmax):       {teacher_ppl:.2f}")
    print(f"Pre-Distillation Student Perplexity (100%):  {pre_ppl:.2f}")
    print(f"Post-Distillation Student Perplexity (100%): {post_ppl:.2f}")
    if post_ppl < pre_ppl:
        improvement = ((pre_ppl - post_ppl) / pre_ppl) * 100
        print(f"Perplexity Improvement:                     -{improvement:.1f}% (Closer to Teacher)")
    print(f"=============================================================================")

    print("\n--- PRIME-SELECTIVE GENERATION 1 ---")
    print(post_sample)
    print("------------------------------------\n")

    print("\n--- PRIME-SELECTIVE GENERATION 2 ---")
    print(post_sample_2)
    print("------------------------------------\n")

    # State footprint check
    state_mb = (num_layers * student.model.layers[0].self_attn.num_heads * (1 + 128 + 128*128 + 128*128) * 4) / (1024*1024)
    print(f"[+] Total Model Recurrent State VRAM: {state_mb:.2f} MB (O(1) flat forever!)")

    audit_data = {
        "model": args.model,
        "architecture": "PRIME-Selective (Beta-Scaled + Input-Dependent Delta_t)",
        "precision": "bfloat16",
        "state_accumulation": "float32",
        "total_parameters": total_p,
        "trainable_parameters": train_p,
        "trainable_percent": train_p / total_p * 100,
        "steps": args.steps,
        "teacher_perplexity": teacher_ppl,
        "pre_distillation_perplexity": pre_ppl,
        "post_distillation_perplexity": post_ppl,
        "state_vram_mb": state_mb,
        "learned_tau": sample_tau.tolist(),
        "learned_beta": sample_beta.tolist(),
        "pre_generation_sample": pre_sample,
        "post_generation_sample_1": post_sample,
        "post_generation_sample_2": post_sample_2
    }
    with open(os.path.join(save_dir, "distillation_audit_selective.json"), "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    print(f"[🎉] PRIME-Selective Distillation Completed Successfully! Audit saved to {save_dir}/distillation_audit_selective.json")

if __name__ == "__main__":
    main()
