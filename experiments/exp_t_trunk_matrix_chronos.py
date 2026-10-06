#!/usr/bin/env python3
"""
Systematic Trunk Substitution Matrix: Amazon Chronos (Physical Dynamics)
========================================================================
Isolates:
  1. Layer Position: Layer 0 (Input), Layer 1 (Early Trunk), Layer 2 (Late Trunk), Layer 3 (Output),
                     Interior Trunk [1, 2], and 100% Replacement [0, 1, 2, 3].
  2. Moment Order Ladder:
       - Order 0: Uniform / Mean Context ($S_0$)
       - Order 1: Linear Dot-Product Kernel ($S_0 + S_1$)
       - Order 2: Quadratic Polynomial Moment Kernel ($S_0 + S_1 + S_2$)
  3. Physical Dynamics Regimes:
       - Regime A: Damped Harmonic Oscillator (Spring-Mass exponential decay)
       - Regime B: Nonlinear Trend + Seasonal Multi-Harmonic Wave
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

num_encoder_layers = len(model.encoder.block)
print(f"[Config] Encoder block count: {num_encoder_layers}")

orig_encoder_forwards = [b.layer[0].SelfAttention.forward for b in model.encoder.block]

# Quantization and Dequantization utilities for Chronos
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

# Benchmarks
t_vals = torch.linspace(0, 16, 128)
benchmarks = [
    {
        "name": "Damped Harmonic Oscillator",
        "series": torch.exp(-0.05 * t_vals) * torch.cos(2.0 * t_vals),
        "history_len": 96,
        "forecast_len": 32
    },
    {
        "name": "Nonlinear Trend + Seasonal Wave",
        "series": 0.05 * t_vals + torch.sin(2.0 * 3.14159 * t_vals / 4.0),
        "history_len": 96,
        "forecast_len": 32
    }
]

def make_prime_t5_attn(order=2, eps=1e-5):
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
    fwd = make_prime_t5_attn(order=order)
    for idx in layers_to_replace:
        sa = model.encoder.block[idx].layer[0].SelfAttention
        sa.forward = types.MethodType(fwd, sa)

def run_forecast(input_ids, forecast_len, scale):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=forecast_len,
            min_new_tokens=forecast_len,
            do_sample=False
        )
    torch.cuda.synchronize()
    lat = (time.time() - t0) * 1000.0
    gen_tokens = out_ids[0, 1:forecast_len+1]
    pred = decode_tokens(gen_tokens, scale).cpu().numpy()
    return pred, lat

# Prepare Benchmark Inputs
bench_data = []
for b in benchmarks:
    series = b["series"]
    history = series[:b["history_len"]]
    gt = series[b["history_len"]:].numpy()
    in_ids, scale = encode_series(history)
    bench_data.append({
        "name": b["name"],
        "gt": gt,
        "input_ids": in_ids,
        "forecast_len": b["forecast_len"],
        "scale": scale
    })

# Define Matrix Grid
positions = [
    ("Layer 0 (Input Boundary)", [0]),
    ("Layer 1 (Early-Mid Trunk)", [1]),
    ("Layer 2 (Late-Mid Trunk)", [2]),
    ("Layer 3 (Encoder Readout)", [3]),
    ("Interior Trunk (Layers 1, 2)", [1, 2]),
    ("Trunk + Readout (1, 2, 3)", [1, 2, 3]),
    ("100% Replacement (0, 1, 2, 3)", [0, 1, 2, 3])
]

moment_orders = [
    ("Order 0 ($S_0$, Mean)", 0),
    ("Order 1 ($S_0+S_1$, Linear)", 1),
    ("Order 2 ($S_0+S_1+S_2$, Quadratic)", 2)
]

print("\n" + "=" * 80)
print(" RUNNING BASELINE SOFTMAX REFERENCE FORECASTS")
print("=" * 80)

setup_chronos([])
baseline_results = {}
for b in bench_data:
    pred, lat = run_forecast(b["input_ids"], b["forecast_len"], b["scale"])
    mae = float(np.mean(np.abs(b["gt"] - pred)))
    mse = float(np.mean((b["gt"] - pred) ** 2))
    r = float(np.corrcoef(b["gt"], pred)[0, 1])
    baseline_results[b["name"]] = {"mae": mae, "mse": mse, "r": r, "latency_ms": lat, "pred": pred}
    print(f"Softmax Baseline [{b['name']}]: MAE={mae:.4f}, MSE={mse:.4f}, r={r:.4f}, Latency={lat:.2f}ms")

matrix_results = {
    "baseline": {
        k: {m: v[m] for m in ["mae", "mse", "r", "latency_ms"]} for k, v in baseline_results.items()
    },
    "positions": {}
}

print("\n" + "=" * 80)
print(" EXECUTING CHRONOS TRUNK SUBSTITUTION MATRIX (7 POSITIONS x 3 ORDERS)")
print("=" * 80)

for pos_label, layers in positions:
    matrix_results["positions"][pos_label] = {"layers": layers, "orders": {}}
    print(f"\n>>> Position: {pos_label} (Indices: {layers})")

    for order_label, order in moment_orders:
        setup_chronos(layers, order=order)

        order_data = {}
        for b in bench_data:
            pred, lat = run_forecast(b["input_ids"], b["forecast_len"], b["scale"])
            mae = float(np.mean(np.abs(b["gt"] - pred)))
            mse = float(np.mean((b["gt"] - pred) ** 2))
            r = float(np.corrcoef(b["gt"], pred)[0, 1]) if np.std(pred) > 1e-6 else 0.0

            # Cosine similarity against Softmax prediction trajectory
            base_pred = baseline_results[b["name"]]["pred"]
            pred_cos = float(np.dot(base_pred, pred) / (np.linalg.norm(base_pred) * np.linalg.norm(pred) + 1e-8))

            order_data[b["name"]] = {
                "mae": mae,
                "mse": mse,
                "r": r,
                "pred_cosine": pred_cos,
                "latency_ms": lat
            }

        matrix_results["positions"][pos_label]["orders"][order_label] = {
            "order": order,
            "benchmarks": order_data
        }

        osc = order_data["Damped Harmonic Oscillator"]
        ts = order_data["Nonlinear Trend + Seasonal Wave"]
        print(f"  [{order_label:28s}] Osc: MAE={osc['mae']:.4f} (r={osc['r']:.4f}, cos={osc['pred_cosine']:.4f}) | "
              f"Trend: MAE={ts['mae']:.4f} (r={ts['r']:.4f}, cos={ts['pred_cosine']:.4f}) | Latency={osc['latency_ms']:.1f}ms")

# Save results
out_path = "/home/phil/.gemini/antigravity/scratch/trunk_matrix_chronos_results.json"
with open(out_path, "w") as f:
    json.dump(matrix_results, f, indent=2)
print(f"\n[Saved] Chronos Trunk Substitution Matrix saved to: {out_path}")
