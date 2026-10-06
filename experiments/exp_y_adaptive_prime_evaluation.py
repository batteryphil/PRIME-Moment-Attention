#!/usr/bin/env python3
"""
Experiment Y: Adaptive PRIME Evaluation
=====================================
Evaluates the Adaptive PRIME architecture with continuous moment relaxation
and dynamic order routing across diverse physical regimes on Chronos.

Tests whether Adaptive PRIME can dynamically allocate contextual bandwidth:
    [g_0, g_1, g_2] -> alpha_1 = g_1 + g_2, alpha_2 = g_2
    K_ij = 1 + alpha_1 * dot_ij + 0.5 * alpha_2 * dot_ij^2
and match or exceed the performance of manually-tuned fixed orders!
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
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.adaptive_prime import EmpiricalBandwidthRouter

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] Using {device}")

model_id = "amazon/chronos-t5-mini"
print(f"[Chronos] Loading {model_id}...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, attn_implementation="eager").to(device).eval()

orig_encoder_forwards = [b.layer[0].SelfAttention.forward for b in model.encoder.block]

# Quantization utilities
bin_edges = torch.linspace(-15.0, 15.0, 4094).to(device)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
full_centers = torch.cat([torch.tensor([-15.0, -15.0], device=device), bin_centers, torch.tensor([15.0], device=device)])

def encode_series(series_tensor):
    series_tensor = series_tensor.to(device)
    scale = series_tensor.abs().mean() + 1e-5
    scaled = series_tensor / scale
    tokens = torch.bucketize(torch.clamp(scaled, -15.0, 15.0), bin_edges) + 2
    return tokens.unsqueeze(0).to(device), scale

def decode_tokens(pred_tokens, scale):
    token_indices = pred_tokens.clamp(0, len(full_centers) - 1)
    return full_centers[token_indices] * scale

empirical_router = EmpiricalBandwidthRouter().to(device)

def make_adaptive_t5_attn(mode="adaptive", fixed_order=2, eps=1e-5):
    """
    mode: 'fixed' (uses fixed_order in {0, 1, 2}) or 'adaptive' (uses empirical_router)
    """
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_values=None,
        output_attentions=False,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.key_value_proj_dim)

        query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        current_states = key_value_states if key_value_states is not None else hidden_states
        kv_shape = (*current_states.shape[:-1], -1, self.key_value_proj_dim)
        key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        value_states = self.v(current_states).view(kv_shape).transpose(1, 2)

        if position_bias is None:
            key_length = key_states.shape[-2]
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, input_shape[1], key_length),
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
            else:
                position_bias = self.compute_bias(
                    input_shape[1], key_length, device=query_states.device
                )

        valid_mask = 1.0
        if mask is not None:
            valid_mask = (mask > -1e4).float()

        dot = torch.matmul(query_states * self.scaling, key_states.transpose(-1, -2))
        if position_bias is not None:
            dot = dot + position_bias

        if mode == "fixed":
            if fixed_order == 0:
                kernel = torch.ones_like(dot)
            elif fixed_order == 1:
                kernel = 1.0 + dot
            else:
                kernel = 1.0 + dot + 0.5 * (dot ** 2)
            g_weights = torch.tensor([1.0 if i == fixed_order else 0.0 for i in range(3)], device=dot.device)
        else:
            # Adaptive mode: compute g, alpha_1, alpha_2 from current_states
            g, alpha_1, alpha_2 = empirical_router(current_states)
            kernel = 1.0 + alpha_1 * dot + 0.5 * alpha_2 * (dot ** 2)
            g_weights = g[0]
            self.last_g = g_weights.detach().cpu().numpy()
            self.last_alphas = (float(alpha_1[0,0,0,0].item()), float(alpha_2[0,0,0,0].item()))

        kernel = F.relu(kernel) * valid_mask
        attn_weights = kernel / (kernel.sum(dim=-1, keepdim=True) + eps)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o(attn_output)

        return attn_output, position_bias, attn_weights
    return forward

def setup_chronos(layers_to_replace, mode="fixed", fixed_order=2):
    for i, b in enumerate(model.encoder.block):
        b.layer[0].SelfAttention.forward = orig_encoder_forwards[i]
    if layers_to_replace:
        fwd = make_adaptive_t5_attn(mode=mode, fixed_order=fixed_order)
        for idx in layers_to_replace:
            sa = model.encoder.block[idx].layer[0].SelfAttention
            sa.forward = types.MethodType(fwd, sa)

def run_forecast(input_ids, forecast_len, scale):
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=forecast_len,
            min_new_tokens=forecast_len,
            do_sample=False
        )
    gen_tokens = out_ids[0, 1:forecast_len+1]
    pred = decode_tokens(gen_tokens, scale).cpu().numpy()
    return pred

# Benchmark Test Regimes
history_len = 96
forecast_len = 32
t_vals = torch.linspace(0, 16, history_len + forecast_len)

np.random.seed(42)
torch.manual_seed(42)

regimes = [
    {
        "name": "Clean Standard Oscillator (w=2.0, sig=0.0)",
        "omega": 2.0,
        "sigma": 0.0,
        "spike": False
    },
    {
        "name": "High Frequency Oscillator (w=3.0, sig=0.0)",
        "omega": 3.0,
        "sigma": 0.0,
        "spike": False
    },
    {
        "name": "Noisy Oscillator (w=2.0, sig=0.05)",
        "omega": 2.0,
        "sigma": 0.05,
        "spike": False
    },
    {
        "name": "Extreme High-Freq Noisy (w=10.0, sig=0.10)",
        "omega": 10.0,
        "sigma": 0.10,
        "spike": False
    },
    {
        "name": "Impulsive Shock at idx 50 (w=2.0)",
        "omega": 2.0,
        "sigma": 0.0,
        "spike": True
    }
]

evaluation_records = []

print("\n" + "="*85)
print("EXPERIMENT Y: ADAPTIVE PRIME EVALUATION & BANDWIDTH ROUTING")
print("="*85)

for reg in regimes:
    reg_name = reg["name"]
    omega = reg["omega"]
    sigma = reg["sigma"]
    
    clean_full = torch.exp(-0.05 * t_vals) * torch.cos(omega * t_vals)
    clean_hist = clean_full[:history_len]
    clean_future = clean_full[history_len:].cpu().numpy()
    
    if reg["spike"]:
        noise = torch.zeros_like(clean_hist)
        noise[50] = 0.50
    else:
        noise = sigma * torch.randn_like(clean_hist)
        
    obs_hist = clean_hist + noise
    input_ids, scale = encode_series(obs_hist)
    
    print(f"\n--- Regime: {reg_name} ---")
    
    # 1. Softmax Baseline
    setup_chronos([])
    pred_sm = run_forecast(input_ids, forecast_len, scale)
    mae_sm = float(np.mean(np.abs(pred_sm - clean_future)))
    r_sm = float(np.corrcoef(pred_sm, clean_future)[0, 1]) if np.std(pred_sm) > 1e-6 else 0.0
    
    # 2. Fixed Order 0
    setup_chronos([1], mode="fixed", fixed_order=0)
    pred_o0 = run_forecast(input_ids, forecast_len, scale)
    mae_o0 = float(np.mean(np.abs(pred_o0 - clean_future)))
    r_o0 = float(np.corrcoef(pred_o0, clean_future)[0, 1]) if np.std(pred_o0) > 1e-6 else 0.0
    
    # 3. Fixed Order 1
    setup_chronos([1], mode="fixed", fixed_order=1)
    pred_o1 = run_forecast(input_ids, forecast_len, scale)
    mae_o1 = float(np.mean(np.abs(pred_o1 - clean_future)))
    r_o1 = float(np.corrcoef(pred_o1, clean_future)[0, 1]) if np.std(pred_o1) > 1e-6 else 0.0
    
    # 4. Fixed Order 2
    setup_chronos([1], mode="fixed", fixed_order=2)
    pred_o2 = run_forecast(input_ids, forecast_len, scale)
    mae_o2 = float(np.mean(np.abs(pred_o2 - clean_future)))
    r_o2 = float(np.corrcoef(pred_o2, clean_future)[0, 1]) if np.std(pred_o2) > 1e-6 else 0.0
    
    # 5. Adaptive PRIME
    setup_chronos([1], mode="adaptive")
    pred_adapt = run_forecast(input_ids, forecast_len, scale)
    mae_adapt = float(np.mean(np.abs(pred_adapt - clean_future)))
    r_adapt = float(np.corrcoef(pred_adapt, clean_future)[0, 1]) if np.std(pred_adapt) > 1e-6 else 0.0
    
    sa_mod = model.encoder.block[1].layer[0].SelfAttention
    last_g = getattr(sa_mod, "last_g", [0.33, 0.33, 0.34])
    last_alphas = getattr(sa_mod, "last_alphas", (0.5, 0.5))
    
    print(f"  Softmax Baseline:  MAE = {mae_sm:6.4f} | r = {r_sm:+.4f}")
    print(f"  Fixed PRIME O0:    MAE = {mae_o0:6.4f} | r = {r_o0:+.4f}")
    print(f"  Fixed PRIME O1:    MAE = {mae_o1:6.4f} | r = {r_o1:+.4f}")
    print(f"  Fixed PRIME O2:    MAE = {mae_o2:6.4f} | r = {r_o2:+.4f}")
    print(f"  *Adaptive PRIME*:  MAE = {mae_adapt:6.4f} | r = {r_adapt:+.4f} | g=[{last_g[0]:.2f}, {last_g[1]:.2f}, {last_g[2]:.2f}] | alpha1={last_alphas[0]:.2f}, alpha2={last_alphas[1]:.2f}")
    
    entry = {
        "regime": reg_name,
        "omega": omega,
        "sigma": sigma,
        "softmax": {"mae": mae_sm, "r": r_sm},
        "fixed_o0": {"mae": mae_o0, "r": r_o0},
        "fixed_o1": {"mae": mae_o1, "r": r_o1},
        "fixed_o2": {"mae": mae_o2, "r": r_o2},
        "adaptive_prime": {
            "mae": mae_adapt,
            "r": r_adapt,
            "router_weights": [float(x) for x in last_g],
            "alpha_1": last_alphas[0],
            "alpha_2": last_alphas[1]
        }
    }
    evaluation_records.append(entry)

# Reset model forwards
setup_chronos([])

# Save results
out_dir = "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/dossier/telemetry"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "adaptive_prime_results.json")
with open(out_path, "w") as f:
    json.dump(evaluation_records, f, indent=2)

print("\n" + "="*85)
print(f"Telemetry saved to: {out_path}")
print("="*85)
