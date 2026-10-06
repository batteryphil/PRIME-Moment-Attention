#!/usr/bin/env python3
"""
Domain 3 Benchmark: Time Series & Physical Dynamics (Amazon Chronos-T5)
======================================================================
Evaluates PRIME 2nd-order polynomial moment recurrence on continuous
physical time series forecasting.

Benchmarks:
  1. Damped Harmonic Oscillator (Spring-Mass Physics): y = exp(-0.05 t) * cos(t)
  2. Nonlinear Trend + Seasonal Dynamics: y = 0.05 t + sin(2 pi t / 16)

Conditions:
  - Baseline Softmax Chronos
  - PRIME Mid-Trunk (Encoder layers 1, 2 -- 2 of 4)
  - 100% Zero-Shot PRIME (All 4 encoder layers)
"""

import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["MIOPEN_USER_DB_PATH"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["TRITON_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/.triton"

import math
import time
import json
import types
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print(" DOMAIN 3: AMAZON CHRONOS TIME SERIES & PHYSICAL DYNAMICS")
print("=" * 80)

model = AutoModelForSeq2SeqLM.from_pretrained("amazon/chronos-t5-mini").to(device).eval()

num_encoder_layers = len(model.encoder.block)
orig_encoder_forwards = [b.layer[0].SelfAttention.forward for b in model.encoder.block]

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

        dot = torch.matmul(query_states * self.scaling, key_states.transpose(-1, -2))
        if position_bias is not None:
            dot = dot + position_bias
        if mask is not None:
            dot = dot + mask

        if order == 1:
            kernel = 1.0 + dot
        elif order == 2:
            kernel = 1.0 + dot + 0.5 * (dot ** 2)
        else:
            kernel = 1.0 + dot + 0.5 * (dot ** 2) + (1.0 / 6.0) * (dot ** 3)

        kernel = F.relu(kernel)
        attn_weights = kernel / (kernel.sum(dim=-1, keepdim=True) + eps)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o(attn_output)

        return attn_output, position_bias, attn_weights
    return forward

def inject_prime_chronos(layer_indices, order=2):
    for i, b in enumerate(model.encoder.block):
        b.layer[0].SelfAttention.forward = orig_encoder_forwards[i]
    fwd = make_prime_t5_attn(order=order)
    for idx in layer_indices:
        sa = model.encoder.block[idx].layer[0].SelfAttention
        sa.forward = types.MethodType(fwd, sa)

# Quantization and Dequantization utilities for Chronos
bin_edges = torch.linspace(-15.0, 15.0, 4094).to(device)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
# Pad centers for indices 0, 1 (special tokens) and out-of-bounds
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

architectures = [
    {"name": "Standard Softmax", "layers": []},
    {"name": "PRIME Mid-Trunk", "layers": [1, 2]},
    {"name": "100% Zero-Shot PRIME", "layers": list(range(num_encoder_layers))}
]

results = {}

for bench in benchmarks:
    bench_name = bench["name"]
    print(f"\n--- Benchmark: {bench_name} ---")
    results[bench_name] = {}

    series = bench["series"]
    history = series[:bench["history_len"]]
    ground_truth = series[bench["history_len"]:]

    input_ids, scale = encode_series(history)

    for arch in architectures:
        arch_name = arch["name"]
        inject_prime_chronos(arch["layers"], order=2)

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=bench["forecast_len"],
                min_new_tokens=bench["forecast_len"],
                do_sample=False
            )
        torch.cuda.synchronize()
        latency = (time.time() - t0) * 1000.0

        # Extract generated forecast tokens
        gen_tokens = out_ids[0, 1:bench["forecast_len"]+1]
        forecast_vals = decode_tokens(gen_tokens, scale).cpu()

        gt = ground_truth[:len(forecast_vals)]
        mae = F.l1_loss(forecast_vals, gt).item()
        mse = F.mse_loss(forecast_vals, gt).item()

        # Pearson correlation
        vx = forecast_vals - forecast_vals.mean()
        vy = gt - gt.mean()
        corr = (vx * vy).sum() / (torch.sqrt((vx**2).sum() * (vy**2).sum()) + 1e-6)
        r_val = corr.item()

        print(f"  {arch_name:<22} | MAE: {mae:0.4f} | MSE: {mse:0.4f} | Pearson r: {r_val:0.4f} | Latency: {latency:5.2f}ms")

        results[bench_name][arch_name] = {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "pearson_r": round(r_val, 4),
            "latency_ms": round(latency, 2)
        }

out_json = "/home/phil/.gemini/antigravity/scratch/domain3_chronos_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print(f"[+] DOMAIN 3 (TIME SERIES) COMPLETE! Saved to {out_json}")
print("=" * 80)
