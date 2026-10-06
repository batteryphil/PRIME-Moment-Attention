#!/usr/bin/env python3
"""
Comparative Benchmark of PRIME Spatial Attention in Image Diffusion (SD 1.5)
=============================================================================
Evaluates:
  1. Baseline Softmax (Standard SD 1.5 UNet)
  2. Mid-Block Spatial PRIME Trunk Anchor (Taylor order-2 in mid_block attn1)
  3. Deep-Trunk Spatial PRIME (Mid-Block 8x8 + 16x16 attention layers)
  4. 100% Zero-Shot Spatial Self-Attention PRIME (All 16 attn1 layers in UNet)

Metrics:
  - Generation time (total and per-step)
  - Peak VRAM footprint
  - Cosine similarity and MSE fidelity to baseline Softmax image
  - Image pixel standard deviation (contrast/dynamic range)
"""

import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["MIOPEN_USER_DB_PATH"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
import sys
sys.path.append("/home/phil/.gemini/antigravity/scratch/MuseTalk/venv/lib/python3.12/site-packages")
sys.path.append("/home/phil/.gemini/antigravity/scratch")

import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from vision_prime_adapter import PrimeTemporalAttentionProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
sd_path = "/home/phil/.cache/huggingface/hub/models--SG161222--Realistic_Vision_V5.1_noVAE/snapshots/1e9f017a7b1eaefb63a1900ea6c5953d2739fd21"

prompt = "a majestic bald eagle soaring over a snow-capped mountain range at sunrise, cinematic lighting, 8k resolution, photorealistic"
negative_prompt = "blurry, low quality, distorted, deformed, artifacts"
seed = 42

def find_spatial_self_attention_modules(unet):
    modules = {"all": {}, "mid": {}, "deep": {}, "high_res": {}}
    for name, mod in unet.named_modules():
        if hasattr(mod, "processor") and name.endswith("attn1"):
            modules["all"][name] = mod
            if "mid_block" in name:
                modules["mid"][name] = mod
                modules["deep"][name] = mod
            elif "down_blocks.2" in name or "up_blocks.1" in name: # 16x16 resolution
                modules["deep"][name] = mod
            else: # 32x32 resolution
                modules["high_res"][name] = mod
    return modules

def restore_default_spatial_attention(unet):
    modules = find_spatial_self_attention_modules(unet)
    default_proc = AttnProcessor2_0()
    for name, mod in modules["all"].items():
        mod.set_processor(default_proc)

def inject_spatial_prime(unet, target_modules, order=2):
    proc = PrimeTemporalAttentionProcessor(mode="prime", order=order)
    for name, mod in target_modules.items():
        mod.set_processor(proc)
    return len(target_modules)

def compute_image_metrics(test_img, base_img):
    t_arr = np.array(test_img, dtype=np.float32)
    b_arr = np.array(base_img, dtype=np.float32)
    
    t_flat = torch.tensor(t_arr).flatten()
    b_flat = torch.tensor(b_arr).flatten()
    
    cos_sim = F.cosine_similarity(t_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
    mse = float(np.mean((t_arr - b_arr) ** 2))
    std = float(np.std(t_arr))
    
    return float(cos_sim), float(mse), float(std)

print("[1] Initializing Stable Diffusion Pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(sd_path, torch_dtype=torch.float16, local_files_only=True).to(device)

base_img_path = "/home/phil/.gemini/antigravity/scratch/baseline_image.png"
base_img = Image.open(base_img_path).convert("RGB")
_, _, base_std = compute_image_metrics(base_img, base_img)

experiments = [
    {
        "name": "prime_spatial_mid_trunk",
        "description": "Mid-Block Spatial PRIME Trunk Anchor (Taylor order-2 in mid_block 8x8 latent trunk)",
        "scope": "mid",
        "order": 2,
        "path": "/home/phil/.gemini/antigravity/scratch/prime_spatial_mid_trunk.png"
    },
    {
        "name": "prime_spatial_deep_trunk",
        "description": "Deep-Trunk Spatial PRIME (Mid-Block 8x8 + Low-Res 16x16 attention layers)",
        "scope": "deep",
        "order": 2,
        "path": "/home/phil/.gemini/antigravity/scratch/prime_spatial_deep_trunk.png"
    },
    {
        "name": "prime_spatial_all_zeroshot",
        "description": "100% Zero-Shot Spatial Self-Attention PRIME (All 16 spatial attn1 layers)",
        "scope": "all",
        "order": 2,
        "path": "/home/phil/.gemini/antigravity/scratch/prime_spatial_all_zeroshot.png"
    }
]

results = {
    "baseline_softmax": {
        "description": "Standard Softmax SD 1.5 (Realistic Vision V5.1)",
        "injected_layers": 0,
        "time_sec": 44.00,
        "sec_per_step": 2.20,
        "peak_vram_gb": 4.43,
        "fidelity_cosine": 1.0,
        "mse": 0.0,
        "pixel_std": round(base_std, 2),
        "path": base_img_path
    }
}

spatial_modules = find_spatial_self_attention_modules(pipe.unet)
print(f"[+] Discovered spatial self-attention modules: all={len(spatial_modules['all'])}, mid={len(spatial_modules['mid'])}, deep={len(spatial_modules['deep'])}, high_res={len(spatial_modules['high_res'])}")

for exp in experiments:
    print("\n" + "=" * 80)
    print(f" RUNNING IMAGE EXPERIMENT: {exp['name'].upper()}")
    print(f" {exp['description']}")
    print("=" * 80)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    restore_default_spatial_attention(pipe.unet)
    targets = spatial_modules[exp["scope"]]
    injected_count = inject_spatial_prime(pipe.unet, targets, order=exp["order"])
    print(f"[+] Injected PRIME into {injected_count} spatial self-attention layers.")

    generator = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    img_result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
        width=512,
        height=512
    )
    t_total = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / (1024**3)

    img = img_result.images[0]
    img.save(exp["path"])
    print(f"[+] Saved image to: {exp['path']}")

    cos_sim, mse, std = compute_image_metrics(img, base_img)

    print(f"[+] Results for {exp['name']}:")
    print(f"    Total Time: {t_total:.2f}s ({t_total/20:.3f}s/step) | Peak VRAM: {vram:.2f} GB")
    print(f"    Fidelity (Cosine Sim to Baseline): {cos_sim:.4f} | MSE: {mse:.2f}")
    print(f"    Pixel Std (Contrast): {std:.2f} (Baseline: {base_std:.2f})")

    results[exp["name"]] = {
        "description": exp["description"],
        "injected_layers": injected_count,
        "time_sec": round(t_total, 2),
        "sec_per_step": round(t_total / 20, 3),
        "peak_vram_gb": round(vram, 2),
        "fidelity_cosine": round(cos_sim, 4),
        "mse": round(mse, 2),
        "pixel_std": round(std, 2),
        "path": exp["path"]
    }

out_json = "/home/phil/.gemini/antigravity/scratch/image_prime_benchmark_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print(f"[+] IMAGE BENCHMARK COMPLETE! Results saved to: {out_json}")
print("=" * 80)
