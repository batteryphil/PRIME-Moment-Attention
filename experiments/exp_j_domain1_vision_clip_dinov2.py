#!/usr/bin/env python3
"""
Domain 1 Benchmark: Multimodal Vision & ViT (CLIP & DINOv2)
===========================================================
Evaluates PRIME 2nd-order polynomial moment recurrence on:
  1. OpenAI CLIP (ViT-B/32): Zero-Shot Image-Text Classification & Cross-Modal Retrieval
  2. Facebook DINOv2 (ViT-S/14): Self-Supervised Visual Representation & Patch Alignment

Conditions:
  - Baseline Softmax ViT
  - PRIME Mid-Trunk Anchor (Layers 4, 5, 6, 7 -- 4 of 12)
  - PRIME Deep Half (Layers 3 to 8 -- 6 of 12)
  - 100% Zero-Shot PRIME (All 12 layers)
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
image_path = "/home/phil/.gemini/antigravity/scratch/baseline_image.png"
raw_image = Image.open(image_path).convert("RGB")

results = {"clip": {}, "dinov2": {}}

# ==============================================================================
# 1. OPENAI CLIP (ViT-B/32) ZERO-SHOT BENCHMARK
# ==============================================================================
print("=" * 80)
print(" 1. OPENAI CLIP (ViT-B/32) ZERO-SHOT MULTIMODAL BENCHMARK")
print("=" * 80)

clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

candidate_labels = [
    "a photo of a bald eagle soaring over mountains",
    "a photo of a sports car driving on a highway",
    "a photo of a plate of sushi",
    "a photo of a golden retriever playing fetch",
    "a photo of a city skyline at night"
]

clip_inputs = clip_proc(
    text=candidate_labels,
    images=raw_image,
    return_tensors="pt",
    padding=True
).to(device)

num_clip_layers = len(clip_model.vision_model.encoder.layers)
orig_clip_forwards = [l.self_attn.forward for l in clip_model.vision_model.encoder.layers]

def make_prime_clip_attn(order=2, eps=1e-5):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        queries = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        values = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        dot = torch.matmul(queries * self.scale, keys.transpose(-1, -2))
        if order == 1:
            kernel = 1.0 + dot
        elif order == 2:
            kernel = 1.0 + dot + 0.5 * (dot ** 2)
        else:
            kernel = 1.0 + dot + 0.5 * (dot ** 2) + (1.0 / 6.0) * (dot ** 3)

        kernel = F.relu(kernel)
        attn_weights = kernel / (kernel.sum(dim=-1, keepdim=True) + eps)
        attn_output = torch.matmul(attn_weights, values)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights
    return forward

def inject_prime_clip(layer_indices, order=2):
    for i, l in enumerate(clip_model.vision_model.encoder.layers):
        l.self_attn.forward = orig_clip_forwards[i]
    fwd = make_prime_clip_attn(order=order)
    for idx in layer_indices:
        l = clip_model.vision_model.encoder.layers[idx]
        l.self_attn.forward = types.MethodType(fwd, l.self_attn)

clip_architectures = [
    {"name": "Standard Softmax", "layers": []},
    {"name": "PRIME Mid-Trunk", "layers": [4, 5, 6, 7]},
    {"name": "PRIME Deep Half", "layers": list(range(3, 9))},
    {"name": "100% Zero-Shot PRIME", "layers": list(range(num_clip_layers))}
]

# Get baseline image features
inject_prime_clip([])
with torch.no_grad():
    base_out = clip_model(**clip_inputs)
    base_img_emb = clip_model.get_image_features(pixel_values=clip_inputs.pixel_values)
    if hasattr(base_img_emb, "pooler_output"):
        base_img_emb = base_img_emb.pooler_output
    base_probs = base_out.logits_per_image.softmax(dim=-1)[0]

print(f"Ground Truth Label: '{candidate_labels[0]}'\n")

for arch in clip_architectures:
    inject_prime_clip(arch["layers"], order=2)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = clip_model(**clip_inputs)
        img_emb = clip_model.get_image_features(pixel_values=clip_inputs.pixel_values)
        if hasattr(img_emb, "pooler_output"):
            img_emb = img_emb.pooler_output
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) * 1000.0

    probs = out.logits_per_image.softmax(dim=-1)[0]
    top1_idx = probs.argmax().item()
    top1_label = candidate_labels[top1_idx]
    correct_prob = probs[0].item()

    cos_to_base = F.cosine_similarity(img_emb, base_img_emb, dim=-1).item()

    print(f"  {arch['name']:<22} | Top-1: '{top1_label[:25]}...' | Correct Prob: {correct_prob*100:5.1f}% | Cosine Fidelity: {cos_to_base:0.4f} | Latency: {elapsed:5.2f}ms")

    results["clip"][arch["name"]] = {
        "top1_label": top1_label,
        "correct_prob": round(correct_prob, 4),
        "cosine_fidelity": round(cos_to_base, 4),
        "latency_ms": round(elapsed, 2)
    }

# ==============================================================================
# 2. FACEBOOK DINOv2 (ViT-S/14) SELF-SUPERVISED BENCHMARK
# ==============================================================================
print("\n" + "=" * 80)
print(" 2. FACEBOOK DINOv2 (ViT-S/14) SPATIAL PATCH REPRESENTATION BENCHMARK")
print("=" * 80)

dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()

dino_inputs = dino_proc(images=raw_image, return_tensors="pt").to(device)

num_dino_layers = len(dino_model.encoder.layer)
orig_dino_forwards = [l.attention.attention.forward for l in dino_model.encoder.layer]

def make_prime_dino_attn(order=2, eps=1e-5):
    def forward(self, hidden_states, **kwargs):
        batch_size = hidden_states.shape[0]
        new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size

        key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)
        query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)

        dot = torch.matmul(query_layer * self.scaling, key_layer.transpose(-1, -2))
        if order == 1:
            kernel = 1.0 + dot
        elif order == 2:
            kernel = 1.0 + dot + 0.5 * (dot ** 2)
        else:
            kernel = 1.0 + dot + 0.5 * (dot ** 2) + (1.0 / 6.0) * (dot ** 3)

        kernel = F.relu(kernel)
        attn_weights = kernel / (kernel.sum(dim=-1, keepdim=True) + eps)
        context_layer = torch.matmul(attn_weights, value_layer)

        context_layer = context_layer.transpose(1, 2)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        return context_layer, attn_weights
    return forward

def inject_prime_dino(layer_indices, order=2):
    for i, l in enumerate(dino_model.encoder.layer):
        l.attention.attention.forward = orig_dino_forwards[i]
    fwd = make_prime_dino_attn(order=order)
    for idx in layer_indices:
        l = dino_model.encoder.layer[idx]
        l.attention.attention.forward = types.MethodType(fwd, l.attention.attention)

dino_architectures = [
    {"name": "Standard Softmax", "layers": []},
    {"name": "PRIME Mid-Trunk", "layers": [4, 5, 6, 7]},
    {"name": "PRIME Deep Half", "layers": list(range(3, 9))},
    {"name": "100% Zero-Shot PRIME", "layers": list(range(num_dino_layers))}
]

# Get baseline DINOv2 embeddings
inject_prime_dino([])
with torch.no_grad():
    base_dino_out = dino_model(**dino_inputs)
    base_cls = base_dino_out.last_hidden_state[:, 0, :] # CLS token
    base_patches = base_dino_out.last_hidden_state[:, 1:, :] # 196 spatial tokens

for arch in dino_architectures:
    inject_prime_dino(arch["layers"], order=2)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        dino_out = dino_model(**dino_inputs)
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) * 1000.0

    cls_emb = dino_out.last_hidden_state[:, 0, :]
    patches_emb = dino_out.last_hidden_state[:, 1:, :]

    cls_cosine = F.cosine_similarity(cls_emb, base_cls, dim=-1).item()
    patch_cosine = F.cosine_similarity(patches_emb, base_patches, dim=-1).mean().item()

    print(f"  {arch['name']:<22} | CLS Cosine: {cls_cosine:0.4f} | Patch Mean Cosine: {patch_cosine:0.4f} | Latency: {elapsed:5.2f}ms")

    results["dinov2"][arch["name"]] = {
        "cls_cosine": round(cls_cosine, 4),
        "patch_cosine": round(patch_cosine, 4),
        "latency_ms": round(elapsed, 2)
    }

out_json = "/home/phil/.gemini/antigravity/scratch/domain1_vision_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print(f"[+] DOMAIN 1 BENCHMARK COMPLETE! Saved to {out_json}")
print("=" * 80)
