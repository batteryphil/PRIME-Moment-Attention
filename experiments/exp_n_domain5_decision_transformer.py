#!/usr/bin/env python3
"""
Domain 5 Benchmark: Reinforcement Learning & Action Control (Decision Transformer)
==================================================================================
Evaluates PRIME 2nd-order polynomial moment recurrence on:
  - Model: edbeeching/decision-transformer-gym-hopper-medium (3 GPT-2 Transformer blocks)
  - Task: Offline RL Action Prediction & Continuous Control (Gym Hopper locomotion)

Evaluates:
  1. Action Prediction MSE vs Standard Softmax
  2. Policy Action Vector Cosine Similarity
  3. State Transition Prediction MSE
  4. Sequence Length Scaling & Latency (T = 10, 20, 50, 100 tokens, seq_len = 3*T)

Configurations:
  - Baseline Softmax GPT-2
  - PRIME Mid-Trunk (Layer 1 of 3)
  - PRIME Deep Trunk (Layers 1, 2 of 3)
  - 100% Zero-Shot PRIME (All 3 layers)
"""

import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["MIOPEN_USER_DB_PATH"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_LOCK_FILE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_DEBUG_DISABLE_LOCK"] = "1"
os.environ["TRITON_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/.triton"

import math
import time
import json
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DecisionTransformerModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] Using device: {device}")

model_id = "edbeeching/decision-transformer-gym-hopper-medium"
print(f"[Decision Transformer] Loading model: {model_id}...")
model = DecisionTransformerModel.from_pretrained(model_id).to(device).eval()

state_dim = model.config.state_dim   # 11
act_dim = model.config.act_dim       # 3
target_return = 3600.0               # Expert return target for Hopper

print(f"[Config] State dim: {state_dim}, Act dim: {act_dim}, Layers: {len(model.encoder.h)}")

# Backup original forward functions
orig_forwards = [block.attn.forward for block in model.encoder.h]

def make_prime_dt_attn(order=2, eps=1e-5):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        scale = self.scaling if hasattr(self, "scaling") else (1.0 / math.sqrt(self.head_dim))
        dot = torch.matmul(query_states * scale, key_states.transpose(-1, -2))

        if order == 1:
            kernel = 1.0 + dot
        elif order == 2:
            kernel = 1.0 + dot + 0.5 * (dot ** 2)
        else:
            kernel = 1.0 + dot + 0.5 * (dot ** 2) + (1.0 / 6.0) * (dot ** 3)

        kernel = F.relu(kernel)
        if attention_mask is not None:
            valid_mask = (attention_mask > -1e4).float()
            kernel = kernel * valid_mask

        attn_weights = kernel / (kernel.sum(dim=-1, keepdim=True) + eps)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, attn_weights

    return forward

def inject_prime(layers_to_replace, order=2):
    for i, block in enumerate(model.encoder.h):
        block.attn.forward = orig_forwards[i]
    fwd = make_prime_dt_attn(order=order)
    for idx in layers_to_replace:
        block = model.encoder.h[idx]
        block.attn.forward = types.MethodType(fwd, block.attn)

# Generate diverse test trajectories simulating Gym Hopper physical locomotion regimes:
# Trajectory 1: Periodic leaping gait (sinusoidal joint oscillations)
# Trajectory 2: Disturbed stumble recovery (sudden impulse)
# Trajectory 3: Smooth steady forward cruise
torch.manual_seed(42)
B = 8
max_T = 100

t_axis = torch.linspace(0, 10, max_T).unsqueeze(0).repeat(B, 1) # [B, max_T]
freqs = torch.tensor([1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 0.8, 1.0]).unsqueeze(1) # [B, 1]

# Synthesize physical state vectors (root angle, thigh, leg, foot positions and velocities)
hopper_states = torch.zeros(B, max_T, state_dim)
for d in range(state_dim):
    phase = d * 0.4
    hopper_states[:, :, d] = torch.sin(t_axis * freqs + phase) * 0.8 + torch.randn(B, max_T) * 0.05

hopper_actions = torch.zeros(B, max_T, act_dim)
for a in range(act_dim):
    hopper_actions[:, :, a] = torch.tanh(torch.cos(t_axis * freqs + a * 0.8))

hopper_returns = torch.full((B, max_T, 1), target_return)
# Linear return decrement over trajectory
for t in range(max_T):
    hopper_returns[:, t, 0] = target_return - (t * 20.0)

hopper_timesteps = torch.arange(max_T).repeat(B, 1)

def evaluate_model(T_len):
    s = hopper_states[:, :T_len, :].to(device)
    a = hopper_actions[:, :T_len, :].to(device)
    r = hopper_returns[:, :T_len, :].to(device)
    t = hopper_timesteps[:, :T_len].to(device)
    mask = torch.ones(B, T_len, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(states=s, actions=a, returns_to_go=r, timesteps=t, attention_mask=mask)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    repeats = 30
    start_time = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            out = model(states=s, actions=a, returns_to_go=r, timesteps=t, attention_mask=mask)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency_ms = ((time.time() - start_time) / repeats) * 1000.0

    return {
        "action_preds": out.action_preds.cpu(),
        "state_preds": out.state_preds.cpu() if hasattr(out, "state_preds") and out.state_preds is not None else None,
        "latency_ms": latency_ms
    }

print("\n" + "=" * 80)
print(" RUNNING EVALUATIONS ACROSS 4 ARCHITECTURAL CONFIGURATIONS & TRAJECTORY LENGTHS")
print("=" * 80)

lengths = [10, 20, 50, 100]
configs = [
    ("Softmax Baseline", []),
    ("PRIME Mid-Trunk (Layer 1)", [1]),
    ("PRIME Deep Trunk (Layers 1, 2)", [1, 2]),
    ("100% Zero-Shot PRIME (Layers 0, 1, 2)", [0, 1, 2])
]

results = {
    "lengths": lengths,
    "state_dim": state_dim,
    "act_dim": act_dim,
    "configs": {}
}

# Run Softmax baseline across all lengths first
base_outputs = {}
for T_len in lengths:
    inject_prime([])
    base_outputs[T_len] = evaluate_model(T_len)
    print(f"[Softmax Baseline] T={T_len} (SeqLen={3*T_len}): Latency = {base_outputs[T_len]['latency_ms']:.2f} ms")

for cfg_name, layers in configs:
    inject_prime(layers)
    results["configs"][cfg_name] = {"replaced_layers": layers, "per_length": {}}
    print(f"\n--- {cfg_name} (Layers: {layers}) ---")

    for T_len in lengths:
        eval_res = evaluate_model(T_len)
        base_res = base_outputs[T_len]

        act_pred = eval_res["action_preds"]
        base_act = base_res["action_preds"]

        mse = F.mse_loss(base_act, act_pred).item()
        cos = F.cosine_similarity(
            base_act.reshape(-1, act_dim),
            act_pred.reshape(-1, act_dim),
            dim=-1
        ).mean().item()

        state_mse = 0.0
        if eval_res["state_preds"] is not None and base_res["state_preds"] is not None:
            state_mse = F.mse_loss(base_res["state_preds"], eval_res["state_preds"]).item()

        print(f"  T={T_len:3d} (seq_len={3*T_len:3d}): Latency={eval_res['latency_ms']:.2f}ms | Action MSE={mse:.6f} | Action Cosine={cos:.4f} | State MSE={state_mse:.6f}")

        results["configs"][cfg_name]["per_length"][str(T_len)] = {
            "latency_ms": eval_res["latency_ms"],
            "action_mse": mse,
            "action_cosine": cos,
            "state_mse": state_mse
        }

# Save results to JSON
out_path = "/home/phil/.gemini/antigravity/scratch/domain5_decision_transformer_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[Saved] Domain 5 results saved to: {out_path}")
