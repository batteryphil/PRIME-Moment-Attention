#!/usr/bin/env python3
"""
Systematic Trunk Substitution Matrix: Decision Transformer
==========================================================
Isolates:
  1. Layer Position: Layer 0 (Input), Layer 1 (Trunk), Layer 2 (Readout), and multi-layer combinations.
  2. Moment Order Ladder:
       - Order 0: Uniform / Mean Context ($S_0$)
       - Order 1: Linear Dot-Product Kernel ($S_0 + S_1$)
       - Order 2: Quadratic Polynomial Moment Kernel ($S_0 + S_1 + S_2$)
  3. Downstream Smoothing: Attention output cosine vs. Block output cosine vs. Final Action MSE.
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
orig_block_forwards = [block.forward for block in model.encoder.h]

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

# Instrumentation hooks to intercept attention outputs and block outputs
captured_attn_outputs = {}
captured_block_outputs = {}

def make_prime_dt_attn(order=2, layer_idx=0, eps=1e-5):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        scale = self.scaling if hasattr(self, "scaling") else (1.0 / math.sqrt(self.head_dim))
        dot = torch.matmul(query_states * scale, key_states.transpose(-1, -2))

        if order == 0:
            # Order 0: Uniform / Mean pooling over valid causal history
            kernel = torch.ones_like(dot)
        elif order == 1:
            # Order 1: Linear kernel
            kernel = 1.0 + dot
        else:
            # Order 2: Full quadratic polynomial kernel
            kernel = 1.0 + dot + 0.5 * (dot ** 2)

        kernel = F.relu(kernel)
        if attention_mask is not None:
            valid_mask = (attention_mask > -1e4).float()
            kernel = kernel * valid_mask

        attn_weights = kernel / (kernel.sum(dim=-1, keepdim=True) + eps)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        captured_attn_outputs[layer_idx] = attn_output.detach().clone()
        return attn_output, attn_weights

    return forward

def make_hooked_orig_attn(orig_fwd, layer_idx):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        out = orig_fwd(hidden_states, attention_mask=attention_mask, **kwargs)
        captured_attn_outputs[layer_idx] = out[0].detach().clone()
        return out
    return forward

def make_hooked_block(orig_block_fwd, layer_idx):
    def forward(self, *args, **kwargs):
        out = orig_block_fwd(*args, **kwargs)
        # Block output is a tuple with hidden_states as first element
        captured_block_outputs[layer_idx] = out[0].detach().clone()
        return out
    return forward

def setup_model(replaced_layers, order):
    captured_attn_outputs.clear()
    captured_block_outputs.clear()

    # Reset all blocks with hook wrappers
    for i, block in enumerate(model.encoder.h):
        block.forward = types.MethodType(make_hooked_block(orig_block_forwards[i], i), block)
        if i in replaced_layers:
            fwd = make_prime_dt_attn(order=order, layer_idx=i)
            block.attn.forward = types.MethodType(fwd, block.attn)
        else:
            block.attn.forward = types.MethodType(make_hooked_orig_attn(orig_attn_forwards[i], i), block.attn)

def run_rollout(T_len):
    s = hopper_states[:, :T_len, :].to(device)
    a = hopper_actions[:, :T_len, :].to(device)
    r = hopper_returns[:, :T_len, :].to(device)
    t = hopper_timesteps[:, :T_len].to(device)
    mask = torch.ones(B, T_len, device=device)

    with torch.no_grad():
        out = model(states=s, actions=a, returns_to_go=r, timesteps=t, attention_mask=mask)

    return {
        "action_preds": out.action_preds.cpu(),
        "state_preds": out.state_preds.cpu() if out.state_preds is not None else None,
        "attn_outputs": {k: v.cpu() for k, v in captured_attn_outputs.items()},
        "block_outputs": {k: v.cpu() for k, v in captured_block_outputs.items()}
    }

print("\n" + "=" * 80)
print(" RUNNING BASELINE SOFTMAX REFERENCE")
print("=" * 80)

setup_model([], order=2)
baseline_T20 = run_rollout(20)
base_attn_T20 = {k: v.clone() for k, v in baseline_T20["attn_outputs"].items()}
base_block_T20 = {k: v.clone() for k, v in baseline_T20["block_outputs"].items()}
base_act_T20 = baseline_T20["action_preds"].clone()

setup_model([], order=2)
baseline_T100 = run_rollout(100)
base_attn_T100 = {k: v.clone() for k, v in baseline_T100["attn_outputs"].items()}
base_block_T100 = {k: v.clone() for k, v in baseline_T100["block_outputs"].items()}
base_act_T100 = baseline_T100["action_preds"].clone()

print(f"Captured baselines for T=20 and T=100.")

# Define Matrix Grid
positions = [
    ("Layer 0 (Input Boundary)", [0]),
    ("Layer 1 (Interior Trunk)", [1]),
    ("Layer 2 (Readout Boundary)", [2]),
    ("Layers 0, 1 (Input + Trunk)", [0, 1]),
    ("Layers 1, 2 (Trunk + Readout)", [1, 2]),
    ("Layers 0, 1, 2 (100% Replacement)", [0, 1, 2])
]

moment_orders = [
    ("Order 0 ($S_0$, Mean)", 0),
    ("Order 1 ($S_0+S_1$, Linear)", 1),
    ("Order 2 ($S_0+S_1+S_2$, Quadratic)", 2)
]

matrix_results = {}

print("\n" + "=" * 80)
print(" EXECUTING THE SYSTEMATIC TRUNK SUBSTITUTION MATRIX (6 POSITIONS x 3 ORDERS)")
print("=" * 80)

for pos_label, layers in positions:
    matrix_results[pos_label] = {"layers": layers, "orders": {}}
    print(f"\n>>> Position: {pos_label} (Indices: {layers})")

    for order_label, order in moment_orders:
        setup_model(layers, order=order)
        res_T100 = run_rollout(100)

        # Compute Action Metrics at T=100
        pred_act = res_T100["action_preds"]
        action_mse = F.mse_loss(base_act_T100, pred_act).item()
        action_cosine = F.cosine_similarity(
            base_act_T100.reshape(-1, act_dim),
            pred_act.reshape(-1, act_dim),
            dim=-1
        ).mean().item()

        # Compute Per-Layer Attention Output Cosine and Block Output Cosine
        layer_metrics = {}
        for l_idx in range(3):
            attn_cos = F.cosine_similarity(
                base_attn_T100[l_idx].reshape(-1, model.config.hidden_size),
                res_T100["attn_outputs"][l_idx].reshape(-1, model.config.hidden_size),
                dim=-1
            ).mean().item()

            block_cos = F.cosine_similarity(
                base_block_T100[l_idx].reshape(-1, model.config.hidden_size),
                res_T100["block_outputs"][l_idx].reshape(-1, model.config.hidden_size),
                dim=-1
            ).mean().item()

            norm_ratio = (res_T100["block_outputs"][l_idx].norm(dim=-1).mean() /
                          base_block_T100[l_idx].norm(dim=-1).mean()).item()

            layer_metrics[f"layer_{l_idx}"] = {
                "attn_output_cosine": attn_cos,
                "block_output_cosine": block_cos,
                "norm_ratio": norm_ratio
            }

        matrix_results[pos_label]["orders"][order_label] = {
            "order": order,
            "action_mse": action_mse,
            "action_cosine": action_cosine,
            "layer_metrics": layer_metrics
        }

        print(f"  [{order_label:28s}] Action MSE: {action_mse:.6f} | Policy Cosine: {action_cosine:.4f} | "
              f"L0 BlkCos: {layer_metrics['layer_0']['block_output_cosine']:.4f} | "
              f"L1 BlkCos: {layer_metrics['layer_1']['block_output_cosine']:.4f} | "
              f"L2 BlkCos: {layer_metrics['layer_2']['block_output_cosine']:.4f}")

# Save results
out_path = "/home/phil/.gemini/antigravity/scratch/trunk_matrix_dt_results.json"
with open(out_path, "w") as f:
    json.dump(matrix_results, f, indent=2)
print(f"\n[Saved] Decision Transformer Substitution Matrix saved to: {out_path}")
