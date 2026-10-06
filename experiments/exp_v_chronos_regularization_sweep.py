#!/usr/bin/env python3
"""
Deep Mechanistic Probe: Chronos Layer 1 / Order 0 Regularization vs. Artifact
=============================================================================
Directly tests whether Layer 1 Order 0 outperformance (MAE 0.1424 vs Softmax 0.2438)
is a reproducible regularizing effect or a narrow parameter artifact.

Sweeps:
  - 4 Damping Ratios: zeta in [0.04, 0.08, 0.15, 0.25]
  - 3 Frequencies: omega in [0.5, 1.0, 2.0]
  - 2 Phase Offsets: phi in [0, pi/4]
  - 2 Noise Regimes: clean vs. noisy (sigma = 0.05)
  - 1 Multi-Harmonic Compound Oscillator

Conditions per test:
  1. Softmax Baseline
  2. Layer 1 / Order 0 (Uniform Mean Pooling Context)
  3. Layer 1 / Order 1 (Linear Moment Kernel)
  4. Layer 1 / Order 2 (Quadratic Moment Kernel)
  5. Layer 1 / Random Contextual Control (Random Dirichlet Causal Weights)
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
print(f"[Device] Using device: {device}")

model_id = "amazon/chronos-t5-mini"
print(f"[Chronos] Loading {model_id}...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()

orig_encoder_forwards = [b.layer[0].SelfAttention.forward for b in model.encoder.block]

# Quantization and Dequantization utilities
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
    values = full_centers[token_indices] * scale
    return values

def make_test_attn(mode="prime", order=0, eps=1e-5):
    """
    mode: 'prime' (with order 0, 1, 2) or 'random_causal'
    """
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_values=None,
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

        if mode == "random_causal":
            # Generate random attention weights over valid tokens
            rand_weights = torch.rand(query_states.shape[0], query_states.shape[1], input_shape[1], key_states.shape[-2], device=query_states.device)
            rand_weights = rand_weights * valid_mask
            attn_weights = rand_weights / (rand_weights.sum(dim=-1, keepdim=True) + eps)
        else:
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

def setup_layer1(mode="native", order=0):
    for i, b in enumerate(model.encoder.block):
        b.layer[0].SelfAttention.forward = orig_encoder_forwards[i]
    if mode != "native":
        fwd = make_test_attn(mode=mode, order=order)
        sa = model.encoder.block[1].layer[0].SelfAttention
        sa.forward = types.MethodType(fwd, sa)

def forecast(input_ids, forecast_len, scale):
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=forecast_len,
            min_new_tokens=forecast_len,
            do_sample=False
        )
    gen_tokens = out_ids[0, 1:forecast_len+1]
    return decode_tokens(gen_tokens, scale).cpu().numpy()

# Synthesize diverse physical benchmark suites
t_vals = torch.linspace(0, 16, 128)
history_len = 96
forecast_len = 32

test_cases = [
    # 1. Damping sweeps
    {"name": "Oscillator (zeta=0.04, light)", "s": torch.exp(-0.04 * t_vals) * torch.cos(2.0 * t_vals)},
    {"name": "Oscillator (zeta=0.08, original)", "s": torch.exp(-0.08 * t_vals) * torch.cos(2.0 * t_vals)},
    {"name": "Oscillator (zeta=0.15, heavy)", "s": torch.exp(-0.15 * t_vals) * torch.cos(2.0 * t_vals)},
    {"name": "Oscillator (zeta=0.25, overdamped)", "s": torch.exp(-0.25 * t_vals) * torch.cos(2.0 * t_vals)},
    # 2. Frequency sweeps
    {"name": "Oscillator (omega=0.5, slow)", "s": torch.exp(-0.05 * t_vals) * torch.cos(0.5 * t_vals)},
    {"name": "Oscillator (omega=1.0, mid)", "s": torch.exp(-0.05 * t_vals) * torch.cos(1.0 * t_vals)},
    {"name": "Oscillator (omega=3.0, fast)", "s": torch.exp(-0.05 * t_vals) * torch.cos(3.0 * t_vals)},
    # 3. Phase shift
    {"name": "Oscillator (phi=pi/4 phase)", "s": torch.exp(-0.05 * t_vals) * torch.cos(2.0 * t_vals + math.pi/4)},
    # 4. Noisy observations
    {"name": "Oscillator (noisy sigma=0.05)", "s": torch.exp(-0.05 * t_vals) * torch.cos(2.0 * t_vals) + torch.randn(128)*0.05},
    # 5. Multi-harmonic compound
    {"name": "Compound Dual Oscillator", "s": torch.exp(-0.05 * t_vals) * torch.cos(1.0 * t_vals) + 0.5 * torch.exp(-0.1 * t_vals) * torch.cos(2.5 * t_vals)}
]

operators = [
    ("Native Softmax", "native", 0),
    ("Layer 1: PRIME Order 0 ($S_0$)", "prime", 0),
    ("Layer 1: PRIME Order 1 ($S_0+S_1$)", "prime", 1),
    ("Layer 1: PRIME Order 2 ($S_0+S_1+S_2$)", "prime", 2),
    ("Layer 1: Random Causal Weights", "random_causal", 0)
]

print("=" * 100)
print(" RUNNING CHRONOS LAYER 1 REPRODUCIBILITY & REGULARIZATION SWEEP (10 PHYSICAL CONFIGURATIONS)")
print("=" * 100)

results = {}

for tc in test_cases:
    case_name = tc["name"]
    print(f"\n--- Physical Regime: {case_name} ---")
    results[case_name] = {}

    series = tc["s"]
    history = series[:history_len]
    gt = series[history_len:].numpy()
    in_ids, scale = encode_series(history)

    for op_name, mode, order in operators:
        setup_layer1(mode=mode, order=order)
        pred = forecast(in_ids, forecast_len, scale)

        mae = float(np.mean(np.abs(gt - pred)))
        mse = float(np.mean((gt - pred) ** 2))
        r = float(np.corrcoef(gt, pred)[0, 1]) if np.std(pred) > 1e-6 else 0.0

        results[case_name][op_name] = {"mae": mae, "mse": mse, "r": r}
        print(f"  {op_name:38s}: MAE = {mae:.4f} | MSE = {mse:.4f} | r = {r:.4f}")

out_path = "/home/phil/.gemini/antigravity/scratch/chronos_regularization_sweep_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[Saved] Chronos regularization sweep results saved to: {out_path}")
