#!/usr/bin/env python3
"""
Domain 4 Benchmark: Cheminformatics & Molecular AI (ChemBERTa)
==============================================================
Evaluates PRIME 2nd-order polynomial moment recurrence on:
  - Model: DeepChem/ChemBERTa-77M-MTR (3 RoBERTa transformer layers)
  - Modality: SMILES molecular string representations

Evaluates:
  1. [CLS] Token Cosine Fidelity vs Standard Softmax
  2. Mean-Pooled Molecular Embedding Cosine Fidelity
  3. Chemical Similarity Manifold Preservation (Pairwise Cosine Distance Matrix Pearson r)
  4. Latency (ms per forward pass)

Configurations:
  - Baseline Softmax
  - PRIME Mid-Trunk (Layer 1 of 3)
  - PRIME Deep Trunk (Layers 1, 2 of 3)
  - 100% Zero-Shot PRIME (All 3 layers)
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
from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] Using device: {device}")

model_id = "DeepChem/ChemBERTa-77M-MTR"
print(f"[ChemBERTa] Loading tokenizer and model: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device).eval()

# Test molecules
molecules = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Penicillin_G": "CC1(C(N2C(S1)C(C2=O)NC(=O)Cc3ccccc3)C(=O)O)C",
    "Atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
    "Remdesivir": "CCC(CC)COC(=O)C(C)NP(=O)(OCC1C(C(C(O1)(C#N)c2ccc3c(n2)ncn3N)O)O)Oc4ccccc4"
}

mol_names = list(molecules.keys())
mol_smiles = [molecules[k] for k in mol_names]

# Tokenize inputs
encoded = tokenizer(mol_smiles, padding=True, return_tensors="pt").to(device)
print(f"[Batch] Encoded {len(mol_names)} molecules into tensor shape: {encoded.input_ids.shape}")

# Backup original forward methods
num_layers = len(model.encoder.layer)
orig_forwards = [layer.attention.self.forward for layer in model.encoder.layer]

def make_prime_roberta_attn(order=2, eps=1e-5):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor = None,
        past_key_values = None,
        **kwargs
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        # Projections
        query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)

        # Scale dot product
        scale = getattr(self, "scaling", 1.0 / math.sqrt(self.attention_head_size))
        dot = torch.matmul(query_layer * scale, key_layer.transpose(-1, -2))

        # PRIME 2nd-order moment kernel
        if order == 1:
            kernel = 1.0 + dot
        elif order == 2:
            kernel = 1.0 + dot + 0.5 * (dot ** 2)
        else:
            kernel = 1.0 + dot + 0.5 * (dot ** 2) + (1.0 / 6.0) * (dot ** 3)

        kernel = F.relu(kernel)

        if attention_mask is not None:
            # Additive mask is ~ -10000 for padded positions
            valid_mask = (attention_mask > -1e4).float()
            kernel = kernel * valid_mask

        attn_weights = kernel / (kernel.sum(dim=-1, keepdim=True) + eps)
        attn_output = torch.matmul(attn_weights, value_layer)
        attn_output = attn_output.transpose(1, 2).contiguous().view(*input_shape, -1)

        return attn_output, attn_weights

    return forward

def inject_prime(layers_to_replace, order=2):
    # Restore all originals first
    for i, layer in enumerate(model.encoder.layer):
        layer.attention.self.forward = orig_forwards[i]

    # Replace target layers
    fwd = make_prime_roberta_attn(order=order)
    for idx in layers_to_replace:
        layer = model.encoder.layer[idx]
        layer.attention.self.forward = types.MethodType(fwd, layer.attention.self)

def run_evaluation(config_name):
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(**encoded)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    repeats = 30
    with torch.no_grad():
        for _ in range(repeats):
            outputs = model(**encoded)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency_ms = ((time.time() - start_time) / repeats) * 1000.0

    last_hidden_state = outputs.last_hidden_state # [B, N, D]
    cls_embeddings = last_hidden_state[:, 0, :] # [B, D]

    # Mean pooling (taking attention mask into account)
    input_mask_expanded = encoded.attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask # [B, D]

    # Compute pairwise cosine distance matrix between molecules (B x B)
    norm_mean = F.normalize(mean_pooled, p=2, dim=1)
    pairwise_cos_sim = torch.mm(norm_mean, norm_mean.t()).cpu().numpy()

    return {
        "cls": cls_embeddings.cpu(),
        "mean_pooled": mean_pooled.cpu(),
        "pairwise_sim": pairwise_cos_sim,
        "latency_ms": latency_ms
    }

print("\n" + "=" * 80)
print(" RUNNING EVALUATIONS ACROSS 4 ARCHITECTURAL CONFIGURATIONS")
print("=" * 80)

# 1. Baseline Softmax
inject_prime([])
print("\n[Config 1] Standard Softmax Baseline...")
base_eval = run_evaluation("Softmax")
print(f"  Latency: {base_eval['latency_ms']:.2f} ms")

configs = [
    ("PRIME Mid-Trunk (Layer 1)", [1]),
    ("PRIME Deep Trunk (Layers 1, 2)", [1, 2]),
    ("100% Zero-Shot PRIME (Layers 0, 1, 2)", [0, 1, 2])
]

results = {
    "baseline": {
        "latency_ms": base_eval["latency_ms"]
    },
    "molecules": mol_names,
    "variants": {}
}

base_cls = base_eval["cls"]
base_mean = base_eval["mean_pooled"]
base_sim = base_eval["pairwise_sim"]

for cfg_name, layers in configs:
    inject_prime(layers)
    print(f"\n[{cfg_name}] (Replaced layers: {layers})...")
    eval_res = run_evaluation(cfg_name)

    # Compute cosine similarity per molecule
    cls_cos = F.cosine_similarity(base_cls, eval_res["cls"], dim=1).tolist()
    mean_cos = F.cosine_similarity(base_mean, eval_res["mean_pooled"], dim=1).tolist()

    # Correlation of pairwise similarity matrix (excluding diagonal)
    mask = ~np.eye(len(mol_names), dtype=bool)
    p_base = base_sim[mask]
    p_variant = eval_res["pairwise_sim"][mask]
    corr = float(np.corrcoef(p_base, p_variant)[0, 1])

    avg_cls_cos = float(np.mean(cls_cos))
    avg_mean_cos = float(np.mean(mean_cos))

    print(f"  Latency: {eval_res['latency_ms']:.2f} ms")
    print(f"  Mean [CLS] Cosine Fidelity: {avg_cls_cos:.4f}")
    print(f"  Mean Pooled Cosine Fidelity: {avg_mean_cos:.4f}")
    print(f"  Chemical Manifold Pearson r: {corr:.4f}")
    for idx, name in enumerate(mol_names):
        print(f"    - {name:15s}: CLS cos = {cls_cos[idx]:.4f}, Mean cos = {mean_cos[idx]:.4f}")

    results["variants"][cfg_name] = {
        "replaced_layers": layers,
        "latency_ms": eval_res["latency_ms"],
        "avg_cls_cosine": avg_cls_cos,
        "avg_mean_cosine": avg_mean_cos,
        "manifold_pearson_r": corr,
        "per_molecule": {
            mol_names[i]: {
                "cls_cosine": cls_cos[i],
                "mean_cosine": mean_cos[i]
            } for i in range(len(mol_names))
        }
    }

# Save results to JSON
out_path = "/home/phil/.gemini/antigravity/scratch/domain4_chemberta_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[Saved] Domain 4 results saved to: {out_path}")
