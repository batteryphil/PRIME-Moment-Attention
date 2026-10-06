#!/usr/bin/env python3
"""
Protein Structure & Contact Folding Benchmark: PRIME vs Softmax in ESM-2
========================================================================
Evaluates polynomial moment recurrence in protein language models (ESM-2) 
for predicting 3D residue-residue contact folding maps.

Benchmarks:
  1. Ubiquitin (1UBQ, 76 residues) - canonical alpha/beta roll
  2. BPTI (5PTI, 58 residues) - pancreatic trypsin inhibitor
  3. Protein G (GB1, 56 residues) - immunoglobulin-binding domain

Conditions:
  - Baseline Softmax (Standard ESM-2 8M)
  - Mid-Trunk PRIME (Layer 3 Taylor order-2 recurrence)
  - Dual-Trunk PRIME (Layers 2 & 4 Taylor order-2 recurrence)
  - 100% Zero-Shot PRIME (All 6 layers Taylor order-2 recurrence)
"""

import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["MIOPEN_USER_DB_PATH"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["TRITON_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/.triton"
import sys
sys.path.append("/home/phil/.gemini/antigravity/scratch/MuseTalk/venv/lib/python3.12/site-packages")
sys.path.append("/home/phil/.gemini/antigravity/scratch")

import time
import types
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.models.esm.modeling_esm import apply_rotary_pos_emb

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "facebook/esm2_t6_8M_UR50D"

print("[1] Loading ESM-2 tokenizer and model...")
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id, attn_implementation="eager").to(device)
model.eval()

# Store original forward methods so we can restore cleanly
orig_forwards = [layer.attention.self.forward for layer in model.esm.encoder.layer]

def make_prime_forward(order=2, eps=1e-6):
    def prime_self_attention_forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        **kwargs
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        query_layer = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        query_layer = query_layer * (self.attention_head_size ** -0.5)

        if self.position_embedding_type == "rotary":
            cos, sin = position_embeddings
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, unsqueeze_dim=1)

        dot = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if attention_mask is not None:
            dot = dot + attention_mask

        # PRIME Taylor polynomial recurrence
        if order == 1:
            kernel = 1.0 + dot
        elif order == 2:
            kernel = 1.0 + dot + 0.5 * (dot ** 2)
        else:
            kernel = 1.0 + dot + 0.5 * (dot ** 2) + (1.0 / 6.0) * (dot ** 3)

        kernel = F.relu(kernel)
        denom = kernel.sum(dim=-1, keepdim=True) + eps
        attn_weights = kernel / denom

        attn_output = torch.matmul(attn_weights, value_layer)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attn_output, attn_weights
    return prime_self_attention_forward

def restore_model_attention(model):
    for i, layer in enumerate(model.esm.encoder.layer):
        layer.attention.self.forward = orig_forwards[i]

def inject_prime_layers(model, layer_indices, order=2):
    restore_model_attention(model)
    prime_fwd = make_prime_forward(order=order)
    for idx in layer_indices:
        target = model.esm.encoder.layer[idx].attention.self
        target.forward = types.MethodType(prime_fwd, target)

def render_contact_map_image(contact_matrix, out_path, size=512):
    """
    Renders a 2D contact matrix [L, L] into a viridis/fire-style heatmap image.
    """
    mat = contact_matrix.copy()
    mat = np.clip(mat, 0.0, 1.0)
    # Custom color gradient: Dark Blue (0) -> Cyan (0.33) -> Orange/Yellow (0.66) -> Fiery Red/White (1.0)
    h, w = mat.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Vectorized color mapping
    c1 = (mat < 0.25)
    c2 = (mat >= 0.25) & (mat < 0.50)
    c3 = (mat >= 0.50) & (mat < 0.75)
    c4 = (mat >= 0.75)

    # c1: (15, 10, 45) -> (30, 80, 160)
    f1 = mat / 0.25
    rgb[c1, 0] = (15 + f1[c1] * 15).astype(np.uint8)
    rgb[c1, 1] = (10 + f1[c1] * 70).astype(np.uint8)
    rgb[c1, 2] = (45 + f1[c1] * 115).astype(np.uint8)

    # c2: (30, 80, 160) -> (40, 180, 120)
    f2 = (mat - 0.25) / 0.25
    rgb[c2, 0] = (30 + f2[c2] * 10).astype(np.uint8)
    rgb[c2, 1] = (80 + f2[c2] * 100).astype(np.uint8)
    rgb[c2, 2] = (160 - f2[c2] * 40).astype(np.uint8)

    # c3: (40, 180, 120) -> (240, 200, 30)
    f3 = (mat - 0.50) / 0.25
    rgb[c3, 0] = (40 + f3[c3] * 200).astype(np.uint8)
    rgb[c3, 1] = (180 + f3[c3] * 20).astype(np.uint8)
    rgb[c3, 2] = (120 - f3[c3] * 90).astype(np.uint8)

    # c4: (240, 200, 30) -> (255, 60, 40)
    f4 = (mat - 0.75) / 0.25
    rgb[c4, 0] = (240 + f4[c4] * 15).astype(np.uint8)
    rgb[c4, 1] = (200 - f4[c4] * 140).astype(np.uint8)
    rgb[c4, 2] = (30 + f4[c4] * 10).astype(np.uint8)

    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((size, size), Image.Resampling.NEAREST)
    img.save(out_path)
    return img

def evaluate_top_contacts(pred_mat, base_mat, top_k=50, min_separation=6):
    """
    Computes top-k contact overlap for non-local contacts (|i - j| >= min_separation).
    """
    L = pred_mat.shape[0]
    mask = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) >= min_separation
    
    pred_vals = pred_mat.copy()
    pred_vals[~mask] = -1.0
    base_vals = base_mat.copy()
    base_vals[~mask] = -1.0
    
    top_pred_indices = set(np.argpartition(pred_vals.flatten(), -top_k)[-top_k:])
    top_base_indices = set(np.argpartition(base_vals.flatten(), -top_k)[-top_k:])
    
    overlap = len(top_pred_indices.intersection(top_base_indices))
    return overlap / float(top_k)

# Test Proteins
proteins = [
    {
        "name": "ubiquitin_1ubq",
        "description": "Ubiquitin (1UBQ) - Canonical alpha/beta roll fold",
        "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    },
    {
        "name": "bpti_5pti",
        "description": "Bovine Pancreatic Trypsin Inhibitor (5PTI) - Disulfide-rich fold",
        "sequence": "RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA"
    },
    {
        "name": "gb1",
        "description": "Protein G B1 Domain (GB1) - 4-strand beta-sheet + alpha-helix",
        "sequence": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    }
]

architectures = [
    {
        "name": "softmax_baseline",
        "description": "Standard Softmax ESM-2 8M",
        "layers": []
    },
    {
        "name": "prime_mid_trunk",
        "description": "Mid-Trunk PRIME (Layer 3 Taylor order-2)",
        "layers": [3]
    },
    {
        "name": "prime_dual_trunk",
        "description": "Dual-Trunk PRIME (Layers 2 & 4 Taylor order-2)",
        "layers": [2, 4]
    },
    {
        "name": "prime_all_zeroshot",
        "description": "100% Zero-Shot PRIME (All 6 Layers Taylor order-2)",
        "layers": [0, 1, 2, 3, 4, 5]
    }
]

benchmark_results = {}

for prot in proteins:
    p_name = prot["name"]
    seq = prot["sequence"]
    L = len(seq)
    print("\n" + "=" * 80)
    print(f" EVALUATING PROTEIN: {p_name.upper()} ({L} residues)")
    print(f" {prot['description']}")
    print("=" * 80)

    inputs = tok(seq, return_tensors="pt").to(device)
    benchmark_results[p_name] = {"length": L, "description": prot["description"], "conditions": {}}

    # 1. Softmax Reference
    restore_model_attention(model)
    t0 = time.time()
    with torch.no_grad():
        out_soft = model(**inputs, output_attentions=True)
        attn_soft = torch.stack(out_soft.attentions, dim=1)
        base_contacts = model.esm.contact_head(inputs["input_ids"], attn_soft)[0].detach().cpu().numpy()
    t_soft = (time.time() - t0) * 1000.0 # ms

    soft_img_path = f"/home/phil/.gemini/antigravity/scratch/contact_map_{p_name}_softmax.png"
    render_contact_map_image(base_contacts, soft_img_path)

    benchmark_results[p_name]["conditions"]["softmax_baseline"] = {
        "latency_ms": round(t_soft, 2),
        "cosine_fidelity": 1.0,
        "pearson_corr": 1.0,
        "top_L_overlap": 1.0,
        "image_path": soft_img_path
    }

    # 2. Evaluate PRIME Conditions
    for arch in architectures[1:]:
        arch_name = arch["name"]
        inject_prime_layers(model, arch["layers"], order=2)

        t0 = time.time()
        with torch.no_grad():
            out_prime = model(**inputs, output_attentions=True)
            attn_prime = torch.stack(out_prime.attentions, dim=1)
            pred_contacts = model.esm.contact_head(inputs["input_ids"], attn_prime)[0].detach().cpu().numpy()
        t_prime = (time.time() - t0) * 1000.0

        img_path = f"/home/phil/.gemini/antigravity/scratch/contact_map_{p_name}_{arch_name}.png"
        render_contact_map_image(pred_contacts, img_path)

        # Metrics
        t_pred = torch.tensor(pred_contacts).flatten()
        t_base = torch.tensor(base_contacts).flatten()
        cos_fid = F.cosine_similarity(t_pred.unsqueeze(0), t_base.unsqueeze(0)).item()
        
        # Pearson correlation
        p_mean = pred_contacts.mean()
        b_mean = base_contacts.mean()
        pearson = np.sum((pred_contacts - p_mean) * (base_contacts - b_mean)) / (
            np.sqrt(np.sum((pred_contacts - p_mean)**2)) * np.sqrt(np.sum((base_contacts - b_mean)**2)) + 1e-8
        )

        top_L_acc = evaluate_top_contacts(pred_contacts, base_contacts, top_k=L, min_separation=6)

        print(f"[+] {arch_name:<20}: Latency: {t_prime:.2f}ms | Cosine: {cos_fid:.4f} | Pearson: {pearson:.4f} | Top-L Overlap: {top_L_acc*100:.1f}%")

        benchmark_results[p_name]["conditions"][arch_name] = {
            "injected_layers": arch["layers"],
            "latency_ms": round(t_prime, 2),
            "cosine_fidelity": round(cos_fid, 4),
            "pearson_corr": round(float(pearson), 4),
            "top_L_overlap": round(top_L_acc, 4),
            "image_path": img_path
        }

# Generate a visual 3-panel comparison strip for Ubiquitin
img_soft = Image.open(f"/home/phil/.gemini/antigravity/scratch/contact_map_ubiquitin_1ubq_softmax.png")
img_mid = Image.open(f"/home/phil/.gemini/antigravity/scratch/contact_map_ubiquitin_1ubq_prime_mid_trunk.png")
img_all = Image.open(f"/home/phil/.gemini/antigravity/scratch/contact_map_ubiquitin_1ubq_prime_all_zeroshot.png")

strip = Image.new("RGB", (512 * 3, 512))
strip.paste(img_soft, (0, 0))
strip.paste(img_mid, (512, 0))
strip.paste(img_all, (1024, 0))
strip_path = "/home/phil/.gemini/antigravity/scratch/contact_map_ubiquitin_comparison_strip.png"
strip.save(strip_path)
print(f"\n[+] Saved Ubiquitin 3-panel comparison strip to: {strip_path}")

out_json = "/home/phil/.gemini/antigravity/scratch/protein_prime_benchmark_results.json"
with open(out_json, "w") as f:
    json.dump(benchmark_results, f, indent=2)

print("=" * 80)
print(f"[+] PROTEIN BENCHMARK COMPLETE! Results saved to: {out_json}")
print("=" * 80)
