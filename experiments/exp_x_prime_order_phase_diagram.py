#!/usr/bin/env python3
"""
Experiment X: 42-Cell PRIME Order Response Surface & Phase Diagram
==================================================================
Maps the empirical phase boundary of the optimal moment order:
    O*(omega, sigma) = argmin_{O in {0, 1, 2}} MAE(O)
across a 2D parameter space:
    7 Frequencies: omega in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0] rad/s
    6 Noise levels: sigma in [0.0, 0.01, 0.025, 0.05, 0.10, 0.20] Gaussian noise

Tests the hypothesis:
    "Moment order acts as a controllable bandwidth / interaction-complexity parameter."
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
from transformers import AutoModelForSeq2SeqLM

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

def make_prime_t5_attn(order=2, eps=1e-5):
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

        if order == 0:
            kernel = torch.ones_like(dot)
        elif order == 1:
            kernel = 1.0 + dot
        else:
            kernel = 1.0 + dot + 0.5 * (dot ** 2)

        kernel = F.relu(kernel) * valid_mask
        attn_weights = kernel / (kernel.sum(dim=-1, keepdim=True) + eps)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o(attn_output)

        return attn_output, position_bias, attn_weights
    return forward

def setup_chronos(layers_to_replace, order=2):
    for i, b in enumerate(model.encoder.block):
        b.layer[0].SelfAttention.forward = orig_encoder_forwards[i]
    if layers_to_replace:
        fwd = make_prime_t5_attn(order=order)
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

# Define 2D Grid
frequencies = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
noise_levels = [0.0, 0.01, 0.025, 0.05, 0.10, 0.20]

history_len = 96
forecast_len = 32
t_vals = torch.linspace(0, 16, history_len + forecast_len)

np.random.seed(42)
torch.manual_seed(42)

surface_results = []
phase_map = {} # (omega, sigma) -> best_order

print("\n" + "="*85)
print(f"EXPERIMENT X: 42-CELL PRIME ORDER RESPONSE SURFACE (7 Freqs x 6 Noise Levels)")
print("="*85)

total_cells = len(frequencies) * len(noise_levels)
cell_idx = 0

for omega in frequencies:
    phase_map[omega] = {}
    for sigma in noise_levels:
        cell_idx += 1
        # Generate base physical oscillator: damped spring-mass
        clean_full = torch.exp(-0.05 * t_vals) * torch.cos(omega * t_vals)
        clean_hist = clean_full[:history_len]
        clean_future = clean_full[history_len:].cpu().numpy()
        
        # Additive Gaussian noise
        noise = sigma * torch.randn_like(clean_hist)
        observed_hist = clean_hist + noise
        
        input_ids, scale = encode_series(observed_hist)
        
        # 1. Softmax Baseline
        setup_chronos([])
        pred_sm = run_forecast(input_ids, forecast_len, scale)
        mae_sm = float(np.mean(np.abs(pred_sm - clean_future)))
        r_sm = float(np.corrcoef(pred_sm, clean_future)[0, 1]) if np.std(pred_sm) > 1e-6 else 0.0
        
        # 2. PRIME Order 0
        setup_chronos([1], order=0)
        pred_o0 = run_forecast(input_ids, forecast_len, scale)
        mae_o0 = float(np.mean(np.abs(pred_o0 - clean_future)))
        r_o0 = float(np.corrcoef(pred_o0, clean_future)[0, 1]) if np.std(pred_o0) > 1e-6 else 0.0
        
        # 3. PRIME Order 1
        setup_chronos([1], order=1)
        pred_o1 = run_forecast(input_ids, forecast_len, scale)
        mae_o1 = float(np.mean(np.abs(pred_o1 - clean_future)))
        r_o1 = float(np.corrcoef(pred_o1, clean_future)[0, 1]) if np.std(pred_o1) > 1e-6 else 0.0
        
        # 4. PRIME Order 2
        setup_chronos([1], order=2)
        pred_o2 = run_forecast(input_ids, forecast_len, scale)
        mae_o2 = float(np.mean(np.abs(pred_o2 - clean_future)))
        r_o2 = float(np.corrcoef(pred_o2, clean_future)[0, 1]) if np.std(pred_o2) > 1e-6 else 0.0
        
        # Determine best PRIME order based on lowest MAE
        prime_maes = {"O0": mae_o0, "O1": mae_o1, "O2": mae_o2}
        best_prime_order = min(prime_maes, key=prime_maes.get)
        best_prime_mae = prime_maes[best_prime_order]
        best_prime_r = {"O0": r_o0, "O1": r_o1, "O2": r_o2}[best_prime_order]
        
        beats_softmax = best_prime_mae < mae_sm
        mae_diff = mae_sm - best_prime_mae # positive means PRIME is better
        
        phase_map[omega][sigma] = best_prime_order
        
        entry = {
            "omega": omega,
            "sigma": sigma,
            "softmax": {"mae": mae_sm, "r": r_sm},
            "prime_o0": {"mae": mae_o0, "r": r_o0},
            "prime_o1": {"mae": mae_o1, "r": r_o1},
            "prime_o2": {"mae": mae_o2, "r": r_o2},
            "best_prime_order": best_prime_order,
            "best_prime_mae": best_prime_mae,
            "best_prime_r": best_prime_r,
            "beats_softmax": beats_softmax,
            "mae_improvement": mae_diff
        }
        surface_results.append(entry)
        
        status_marker = "PRIME > SM" if beats_softmax else "SM > PRIME"
        print(f"[{cell_idx:02d}/42] w={omega:5.2f}, sig={sigma:5.3f} | SM MAE: {mae_sm:6.4f} (r={r_sm:+.3f}) | Best: {best_prime_order} (MAE: {best_prime_mae:6.4f}, r={best_prime_r:+.3f}) [{status_marker}]")

# Reset model forwards
setup_chronos([])

# Print Summary Phase Map Heatmap
print("\n" + "="*85)
print("EMPIRICAL PHASE MAP: Best PRIME Order O*(omega, sigma)")
print("="*85)
header = "omega \\ sigma | " + " | ".join([f"s={s:<5}" for s in noise_levels])
print(header)
print("-" * len(header))
for omega in frequencies:
    row = f"w = {omega:<7.2f} | " + " | ".join([f" {phase_map[omega][s]:<4} " for s in noise_levels])
    print(row)
print("="*85)

# Save JSON
out_dir = "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/dossier/telemetry"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "prime_order_phase_surface_results.json")
with open(out_path, "w") as f:
    json.dump({
        "frequencies": frequencies,
        "noise_levels": noise_levels,
        "phase_map": {str(k): {str(s): v for s, v in inner.items()} for k, inner in phase_map.items()},
        "grid_results": surface_results
    }, f, indent=2)

print(f"\nTelemetry saved to: {out_path}")
