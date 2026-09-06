#!/usr/bin/env python3
"""
The Brutal Control Experiment: Decision Transformer Layer 1
===========================================================
Tests whether Decision Transformer's high tolerance (>0.99 policy cosine)
is specific to PRIME's structured moment recurrence or if Layer 1 is trivial.

Conditions evaluated at Layer 1 (and 100% replacement across all 3 layers):
  1. Native Softmax Attention
  2. PRIME Order 1 ($S_0 + S_1$, Linear Kernel)
  3. PRIME Order 2 ($S_0 + S_1 + S_2$, Quadratic Moment Kernel)
  4. PRIME Order 0 ($S_0$, Uniform Mean Pooling Context)
  5. Control A: Random Causal Attention Weights (Dirichlet random routing)
  6. Control B: Identity / Zero-Attention (Output = 0, pure residual stream)
  7. Control C: Shuffled History Attention (Permuted value vectors)
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
print(f"[Decision Transformer] Loading {model_id}...")
model = DecisionTransformerModel.from_pretrained(model_id).to(device).eval()

state_dim = model.config.state_dim   # 11
act_dim = model.config.act_dim       # 3
target_return = 3600.0

orig_attn_forwards = [block.attn.forward for block in model.encoder.h]

# Generate standardized test trajectories
torch.manual_seed(42)
B = 8
max_T = 100

t_axis = torch.linspace(0, 10, max_T).unsqueeze(0).repeat(B, 1)
freqs = torch.tensor([1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 0.8, 1.0]).unsqueeze(1)

hopper_states = torch.zeros(B, max_T, state_dim)
for d in range(state_dim):
    hopper_states[:, :, d] = torch.sin(t_axis * freqs + d * 0.4) * 0.8 + torch.randn(B, max_T) * 0.05

hopper_actions = torch.zeros(B, max_T, act_dim)
for a in range(act_dim):
    hopper_actions[:, :, a] = torch.tanh(torch.cos(t_axis * freqs + a * 0.8))

hopper_returns = torch.full((B, max_T, 1), target_return)
for t in range(max_T):
    hopper_returns[:, t, 0] = target_return - (t * 20.0)

hopper_timesteps = torch.arange(max_T).repeat(B, 1)

def make_controlled_dt_attn(mode="prime_o1", eps=1e-5):
    """
    mode:
      - 'prime_o0': Uniform mean context
      - 'prime_o1': Linear kernel 1 + dot
      - 'prime_o2': Quadratic kernel 1 + dot + 0.5 dot^2
      - 'random_causal': Random Dirichlet causal weights
      - 'zero_attn': Attention output = 0 (pure residual)
      - 'shuffled': Permuted causal history
    """
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        valid_mask = 1.0
        if attention_mask is not None:
            valid_mask = (attention_mask > -1e4).float()

        if mode == "zero_attn":
            attn_output = torch.zeros_like(hidden_states)
            return attn_output, None

        elif mode == "random_causal":
            # Generate random attention weights masked causally
            rand_w = torch.rand(query_states.shape[0], query_states.shape[1], query_states.shape[2], key_states.shape[2], device=query_states.device)
            rand_w = rand_w * valid_mask
            attn_weights = rand_w / (rand_w.sum(dim=-1, keepdim=True) + eps)
            attn_output = torch.matmul(attn_weights, value_states)

        elif mode == "shuffled":
            # Permute tokens along sequence dimension
            perm = torch.randperm(key_states.shape[2])
            perm_values = value_states[:, :, perm, :]
            # Uniform causal weights over shuffled values
            kernel = torch.ones(query_states.shape[0], query_states.shape[1], query_states.shape[2], key_states.shape[2], device=query_states.device)
            kernel = kernel * valid_mask
            attn_weights = kernel / (kernel.sum(dim=-1, keepdim=True) + eps)
            attn_output = torch.matmul(attn_weights, perm_values)

        else:
            scale = self.scaling if hasattr(self, "scaling") else (1.0 / math.sqrt(self.head_dim))
            dot = torch.matmul(query_states * scale, key_states.transpose(-1, -2))

            if mode == "prime_o0":
                kernel = torch.ones_like(dot)
            elif mode == "prime_o1":
                kernel = 1.0 + dot
            elif mode == "prime_o2":
                kernel = 1.0 + dot + 0.5 * (dot ** 2)

            kernel = F.relu(kernel) * valid_mask
            attn_weights = kernel / (kernel.sum(dim=-1, keepdim=True) + eps)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, attn_weights

    return forward

def setup_dt(layers, mode="native"):
    for i, block in enumerate(model.encoder.h):
        block.attn.forward = orig_attn_forwards[i]
    if mode != "native":
        fwd = make_controlled_dt_attn(mode=mode)
        for l in layers:
            block = model.encoder.h[l]
            block.attn.forward = types.MethodType(fwd, block.attn)

def evaluate_dt():
    s = hopper_states.to(device)
    a = hopper_actions.to(device)
    r = hopper_returns.to(device)
    t = hopper_timesteps.to(device)
    mask = torch.ones(B, max_T, device=device)

    with torch.no_grad():
        out = model(states=s, actions=a, returns_to_go=r, timesteps=t, attention_mask=mask)
    return out.action_preds.cpu()

print("=" * 90)
print(" RUNNING THE BRUTAL CONTROL EXPERIMENT ON DECISION TRANSFORMER")
print("=" * 90)

# 1. Native Softmax Baseline
setup_dt([], mode="native")
base_act = evaluate_dt()
print("[Control 0] Native Softmax Baseline reference captured.")

modes = [
    ("PRIME Order 1 ($S_0+S_1$, Linear)", "prime_o1"),
    ("PRIME Order 2 ($S_0+S_1+S_2$, Quadratic)", "prime_o2"),
    ("PRIME Order 0 ($S_0$, Mean Context)", "prime_o0"),
    ("Control A: Random Causal Weights", "random_causal"),
    ("Control B: Zero Attention (Pure Residual)", "zero_attn"),
    ("Control C: Shuffled Token History", "shuffled")
]

test_scopes = [
    ("Single Trunk Substitution (Layer 1 only)", [1]),
    ("Complete 100% Substitution (All 3 Layers)", [0, 1, 2])
]

results = {}

for scope_label, layers in test_scopes:
    print(f"\n>>> Scope: {scope_label} (Layers: {layers})")
    results[scope_label] = {}

    for mode_label, mode in modes:
        setup_dt(layers, mode=mode)
        pred_act = evaluate_dt()

        mse = F.mse_loss(base_act, pred_act).item()
        cos = F.cosine_similarity(
            base_act.reshape(-1, act_dim),
            pred_act.reshape(-1, act_dim),
            dim=-1
        ).mean().item()

        results[scope_label][mode_label] = {"mse": mse, "cosine": cos}
        print(f"  {mode_label:42s}: Action MSE = {mse:.6f} | Policy Cosine = {cos:.4f}")

out_path = "/home/phil/.gemini/antigravity/scratch/dt_brutal_controls_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[Saved] DT controls results saved to: {out_path}")
