#!/usr/bin/env python3
"""
Domain 2 Benchmark: Audio & Speech AI (OpenAI Whisper-tiny)
===========================================================
Evaluates PRIME 2nd-order polynomial moment recurrence on the 
acoustic encoder across dense 1500-frame audio spectrograms.

Conditions:
  - Baseline Softmax Encoder
  - PRIME Mid-Trunk Anchor (Layers 1, 2 -- 2 of 4 layers)
  - 100% Zero-Shot PRIME (All 4 layers)
"""

import os
os.environ["MIOPEN_USER_DB_PATH"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_LOCK_FILE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_DEBUG_DISABLE_LOCK"] = "1"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["TRITON_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/.triton"

import wave
import time
import json
import types
import numpy as np
import torch
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print(" DOMAIN 2: OPENAI WHISPER ACOUSTIC ENCODER BENCHMARK")
print("=" * 80)

proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device).eval()

# Load 10 seconds of speech audio from eng.wav
with wave.open("/home/phil/.gemini/antigravity/scratch/MuseTalk/data/audio/eng.wav", "rb") as wf:
    data = wf.readframes(16000 * 10)
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

inputs = proc(audio, sampling_rate=16000, return_tensors="pt").to(device)
input_features = inputs.input_features
print(f"Acoustic Mel-Spectrogram input shape: {input_features.shape} (3000 frames -> 1500 token latent)")

num_layers = len(model.model.encoder.layers)
orig_forwards = [l.self_attn.forward for l in model.model.encoder.layers]

def make_prime_whisper_attn(order=2, eps=1e-5):
    def forward(self, hidden_states, key_value_states=None, past_key_values=None, attention_mask=None, output_attentions=False, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2).contiguous()
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2).contiguous()

        dot = torch.matmul(query_states, key_states.transpose(-1, -2))
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
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights
    return forward

def inject_prime_whisper(layer_indices, order=2):
    for i, l in enumerate(model.model.encoder.layers):
        l.self_attn.forward = orig_forwards[i]
    fwd = make_prime_whisper_attn(order=order)
    for idx in layer_indices:
        l = model.model.encoder.layers[idx]
        l.self_attn.forward = types.MethodType(fwd, l.self_attn)

architectures = [
    {"name": "Standard Softmax", "layers": []},
    {"name": "PRIME Mid-Trunk", "layers": [1, 2]},
    {"name": "100% Zero-Shot PRIME", "layers": list(range(num_layers))}
]

# Get baseline encoder representations
inject_prime_whisper([])
with torch.no_grad():
    base_enc = model.model.encoder(input_features).last_hidden_state
    base_pred_ids = model.generate(input_features, language="en")
base_text = proc.batch_decode(base_pred_ids, skip_special_tokens=True)[0]
print(f"\n[Baseline Softmax Transcription]:\n>>> {repr(base_text)}\n")

results = {}

for arch in architectures:
    inject_prime_whisper(arch["layers"], order=2)

    # Measure encoder latency
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        enc_out = model.model.encoder(input_features).last_hidden_state
    torch.cuda.synchronize()
    enc_latency = (time.time() - t0) * 1000.0

    # Measure full end-to-end transcription
    t0 = time.time()
    with torch.no_grad():
        pred_ids = model.generate(input_features, language="en")
    total_time = (time.time() - t0)
    transcription = proc.batch_decode(pred_ids, skip_special_tokens=True)[0]

    # Encoder cosine similarity to baseline
    cos_sim = F.cosine_similarity(enc_out, base_enc, dim=-1).mean().item()

    print(f"  {arch['name']:<22} | Encoder Latency: {enc_latency:5.2f}ms | Encoder Cosine: {cos_sim:0.4f} | Total: {total_time:.2f}s")
    print(f"  Transcription: {repr(transcription)}\n")

    results[arch["name"]] = {
        "encoder_latency_ms": round(enc_latency, 2),
        "encoder_cosine_fidelity": round(cos_sim, 4),
        "total_transcription_time_s": round(total_time, 2),
        "transcription": transcription
    }

out_json = "/home/phil/.gemini/antigravity/scratch/domain2_audio_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print("=" * 80)
print(f"[+] DOMAIN 2 (AUDIO) COMPLETE! Saved to {out_json}")
print("=" * 80)
