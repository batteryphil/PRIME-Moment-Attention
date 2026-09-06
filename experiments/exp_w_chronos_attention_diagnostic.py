#!/usr/bin/env python3
"""
Experiment W: Direct Causal Attention Diagnostic on Chronos
==========================================================
Mechanistically investigates why Softmax attention degrades under noise while
PRIME polynomial recurrence acts as an invariant filter / regularizer.

Measures across Clean (sigma=0), Low Noise (sigma=0.02), High Noise (sigma=0.10),
and Impulsive Spike conditions:
  1. Attention Entropy: H(A_i) = - sum_j A_ij log(A_ij + eps)
  2. Temporal Locality: E[|i - j|] = sum_j A_ij |i - j|
  3. Noise-Attention Weight Correlation: Pearson r(A_ij, |eps_j|)
  4. Output Perturbation Sensitivity: ||y_hat(x + delta * e_j) - y_hat(x)||_2 / delta
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
print(f"[Chronos] Loading {model_id} with eager attention...")
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

# Set up physical benchmark parameters
np.random.seed(42)
torch.manual_seed(42)
t_vals = torch.linspace(0, 16, 128)
clean_series = torch.exp(-0.05 * t_vals) * torch.cos(2.0 * t_vals)
history_len = 96
forecast_len = 32
clean_history = clean_series[:history_len]
ground_truth_forecast = clean_series[history_len:].cpu().numpy()

# Noise conditions
noise_02 = 0.02 * torch.randn_like(clean_history)
noise_10 = 0.10 * torch.randn_like(clean_history)

# Impulsive spike at index 50
spike_noise = torch.zeros_like(clean_history)
spike_noise[50] = 0.50  # Large impulsive shock

test_conditions = [
    {
        "name": "Clean Oscillator (sigma=0.0)",
        "history": clean_history,
        "noise": torch.zeros_like(clean_history),
    },
    {
        "name": "Low Noise (sigma=0.02)",
        "history": clean_history + noise_02,
        "noise": noise_02,
    },
    {
        "name": "High Noise (sigma=0.10)",
        "history": clean_history + noise_10,
        "noise": noise_10,
    },
    {
        "name": "Impulsive Spike (+0.5 at idx 50)",
        "history": clean_history + spike_noise,
        "noise": spike_noise,
    }
]

operators = [
    {"name": "Softmax Baseline", "layers": [], "order": None},
    {"name": "PRIME Order 0 (Mean Context)", "layers": [1], "order": 0},
    {"name": "PRIME Order 1 (Linear Kernel)", "layers": [1], "order": 1},
    {"name": "PRIME Order 2 (Quadratic Moment)", "layers": [1], "order": 2},
]

diagnostic_results = []

print("\n" + "="*80)
print("EXPERIMENT W: CHRONOS CAUSAL ATTENTION DIAGNOSTIC")
print("="*80)

for cond in test_conditions:
    cond_name = cond["name"]
    hist = cond["history"]
    injected_noise = cond["noise"].abs().cpu().numpy()
    
    input_ids, scale = encode_series(hist)
    
    print(f"\n--- Condition: {cond_name} ---")
    
    for op in operators:
        op_name = op["name"]
        setup_chronos(op["layers"], order=op["order"] if op["order"] is not None else 2)
        
        # 1. Forward encoder with output_attentions=True
        with torch.no_grad():
            enc_out = model.encoder(input_ids, output_attentions=True)
            # Layer 1 attention matrix: shape (1, heads, seq_len, seq_len)
            l1_attn = enc_out.attentions[1][0].cpu().numpy() # shape (8, 96, 96)
        
        # Compute metrics across all 8 heads
        num_heads, L_q, L_k = l1_attn.shape
        
        # A. Attention Entropy: H = - sum_j A_ij * log(A_ij + 1e-12)
        eps = 1e-12
        head_entropies = -np.sum(l1_attn * np.log(l1_attn + eps), axis=-1) # (heads, L_q)
        mean_entropy = float(np.mean(head_entropies))
        max_possible_entropy = float(np.log(L_k)) # ln(96) ~= 4.564
        
        # B. Temporal Locality: E[|i - j|]
        i_indices = np.arange(L_q)[:, None]
        j_indices = np.arange(L_k)[None, :]
        dist_matrix = np.abs(i_indices - j_indices) # (L_q, L_k)
        head_distances = np.sum(l1_attn * dist_matrix[None, :, :], axis=-1) # (heads, L_q)
        mean_distance = float(np.mean(head_distances))
        
        # C. Attention-Weight Correlation with Injected Noise:
        noise_std = np.std(injected_noise)
        if noise_std > 1e-6:
            corrs = []
            for h in range(num_heads):
                for i in range(L_q):
                    a_vec = l1_attn[h, i, :]
                    if np.std(a_vec) > 1e-8:
                        r = np.corrcoef(a_vec, injected_noise)[0, 1]
                        if not np.isnan(r):
                            corrs.append(r)
            mean_noise_corr = float(np.mean(corrs)) if corrs else 0.0
        else:
            mean_noise_corr = 0.0
            
        # Attention on Spike Index 50 (if impulsive spike condition)
        attn_at_spike = float(np.mean(l1_attn[:, :, 50]))
        
        # D. Forecast Accuracy & Cosine
        forecast = run_forecast(input_ids, forecast_len, scale)
        mae = float(np.mean(np.abs(forecast - ground_truth_forecast)))
        r_val = float(np.corrcoef(forecast, ground_truth_forecast)[0, 1]) if np.std(forecast) > 1e-6 else 0.0
        
        # E. Perturbation Sensitivity: finite-difference Jacobian norm
        sensitivities = []
        delta = 0.05
        sample_indices = [10, 30, 50, 70, 90]
        for s_idx in sample_indices:
            perturbed_hist = hist.clone()
            perturbed_hist[s_idx] += delta
            p_ids, p_scale = encode_series(perturbed_hist)
            p_forecast = run_forecast(p_ids, forecast_len, p_scale)
            diff_norm = np.linalg.norm(p_forecast - forecast) / delta
            sensitivities.append(diff_norm)
        mean_sensitivity = float(np.mean(sensitivities))
        
        result_entry = {
            "condition": cond_name,
            "operator": op_name,
            "layer_substituted": "Layer 1 (Trunk)",
            "mean_entropy": mean_entropy,
            "max_possible_entropy": max_possible_entropy,
            "entropy_ratio": mean_entropy / max_possible_entropy,
            "mean_temporal_distance": mean_distance,
            "mean_noise_correlation": mean_noise_corr,
            "attn_at_spike_idx50": attn_at_spike,
            "forecast_mae": mae,
            "forecast_pearson_r": r_val,
            "output_perturbation_sensitivity": mean_sensitivity
        }
        diagnostic_results.append(result_entry)
        
        print(f"[{op_name}]")
        print(f"  Entropy: {mean_entropy:.3f} / {max_possible_entropy:.3f} ({mean_entropy/max_possible_entropy*100:.1f}%) | Locality Dist: {mean_distance:.2f}")
        print(f"  Noise Corr: {mean_noise_corr:+.4f} | Spike Attn (idx 50): {attn_at_spike:.4f}")
        print(f"  Forecast MAE: {mae:.4f} | Pearson r: {r_val:.4f} | Perturbation Sensitivity: {mean_sensitivity:.4f}")

# Reset model forwards
setup_chronos([])

# Save results
out_dir = "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/dossier/telemetry"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chronos_attention_diagnostic_results.json")
with open(out_path, "w") as f:
    json.dump(diagnostic_results, f, indent=2)

print("\n" + "="*80)
print(f"Telemetry saved to: {out_path}")
print("="*80)
