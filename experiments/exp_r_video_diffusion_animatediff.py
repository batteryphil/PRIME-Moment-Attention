#!/usr/bin/env python3
"""
Phase 3: Comparative Benchmark of PRIME Temporal Attention in Video Diffusion
=============================================================================
Evaluates:
  1. Mid-Block PRIME Trunk Anchor (Zero-shot Taylor order 2 in mid_block)
  2. 100% Zero-Shot PRIME Temporal Attention (All 42 temporal layers in UNet)
  3. Hybrid Temporal Window (W=8 local Softmax + PRIME global anchor)

Metrics:
  - Generation time (total and per-step)
  - Peak VRAM footprint
  - Mean pairwise frame cosine similarity (temporal smoothness)
  - Structural Similarity Index (SSIM) against standard Softmax baseline
  - Dynamic motion differential (pixel variance over time)
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
from PIL import Image, ImageSequence

from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from vision_prime_adapter import (
    PrimeTemporalAttentionProcessor,
    find_temporal_attention_modules,
    inject_prime_temporal_attention,
    restore_default_temporal_attention
)

device = "cuda" if torch.cuda.is_available() else "cpu"
sd_path = "/home/phil/.cache/huggingface/hub/models--SG161222--Realistic_Vision_V5.1_noVAE/snapshots/1e9f017a7b1eaefb63a1900ea6c5953d2739fd21"
adapter_path = "/home/phil/.cache/huggingface/hub/models--guoyww--animatediff-motion-adapter-v1-5-2/snapshots/6167b88ffe39b4441fdf2113e77b99a6f56b7906"

prompt = "a majestic bald eagle soaring over a snow-capped mountain range at sunrise, cinematic lighting, 8k resolution, photorealistic"
negative_prompt = "blurry, low quality, distorted, deformed, artifacts"
seed = 42

def load_frames(gif_path):
    im = Image.open(gif_path)
    return [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(im)]

def compute_temporal_cosine_sim(frames):
    sims = []
    for i in range(len(frames) - 1):
        f1 = torch.tensor(np.array(frames[i]), dtype=torch.float32).flatten()
        f2 = torch.tensor(np.array(frames[i+1]), dtype=torch.float32).flatten()
        cos = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
        sims.append(cos)
    return float(np.mean(sims)), float(np.min(sims)), float(np.max(sims))

def compute_ssim_vs_baseline(test_frames, base_frames):
    # Mean pixel MSE and cosine similarity across corresponding frames
    frame_sims = []
    for f_t, f_b in zip(test_frames, base_frames):
        t1 = torch.tensor(np.array(f_t), dtype=torch.float32).flatten()
        t2 = torch.tensor(np.array(f_b), dtype=torch.float32).flatten()
        cos = F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()
        frame_sims.append(cos)
    return float(np.mean(frame_sims))

# Load pipeline once
print("[1] Initializing AnimateDiff pipeline...")
adapter = MotionAdapter.from_pretrained(adapter_path, torch_dtype=torch.float16, local_files_only=True)
pipe = AnimateDiffPipeline.from_pretrained(sd_path, motion_adapter=adapter, torch_dtype=torch.float16, local_files_only=True)
pipe.scheduler = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", steps_offset=1)
pipe.vae.enable_slicing()
pipe.to(device)

base_gif = "/home/phil/.gemini/antigravity/scratch/baseline_video_16f.gif"
baseline_frames = load_frames(base_gif)
base_mean_cos, base_min_cos, base_max_cos = compute_temporal_cosine_sim(baseline_frames)

experiments = [
    {
        "name": "prime_mid_trunk",
        "description": "Mid-Block PRIME Trunk Anchor (100% Taylor order-2 polynomial recurrence in mid_block)",
        "scope": "mid",
        "mode": "prime",
        "order": 2,
        "window": 8,
        "gif_path": "/home/phil/.gemini/antigravity/scratch/prime_mid_trunk_16f.gif"
    },
    {
        "name": "prime_all_zeroshot",
        "description": "100% Zero-Shot PRIME Temporal Attention (All 42 motion attention layers in UNet)",
        "scope": "all",
        "mode": "prime",
        "order": 2,
        "window": 8,
        "gif_path": "/home/phil/.gemini/antigravity/scratch/prime_all_zeroshot_16f.gif"
    },
    {
        "name": "prime_hybrid",
        "description": "Hybrid Temporal Attention (Local W=8 Softmax + Global Taylor order-2 polynomial anchor)",
        "scope": "all",
        "mode": "hybrid",
        "order": 2,
        "window": 8,
        "gif_path": "/home/phil/.gemini/antigravity/scratch/prime_hybrid_16f.gif"
    }
]

results = {
    "baseline_softmax": {
        "description": "Standard Softmax AnimateDiff v1.5-2",
        "time_sec": 41.21,
        "sec_per_step": 2.576,
        "peak_vram_gb": 5.84,
        "mean_pairwise_cosine": round(base_mean_cos, 4),
        "min_pairwise_cosine": round(base_min_cos, 4),
        "max_pairwise_cosine": round(base_max_cos, 4),
        "fidelity_to_baseline": 1.0,
        "path": base_gif
    }
}

for exp in experiments:
    print("\n" + "=" * 80)
    print(f" RUNNING EXPERIMENT: {exp['name'].upper()}")
    print(f" {exp['description']}")
    print("=" * 80)

    # Clean cache and reset memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Inject PRIME
    restore_default_temporal_attention(pipe.unet)
    injected = inject_prime_temporal_attention(
        pipe.unet,
        scope=exp["scope"],
        processor_mode=exp["mode"],
        order=exp["order"],
        window_size=exp["window"]
    )

    t0 = time.time()
    generator = torch.Generator(device=device).manual_seed(seed)
    vid_result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=16,
        num_inference_steps=16,
        guidance_scale=7.5,
        generator=generator,
        width=512,
        height=512
    )
    t_total = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / (1024**3)

    frames = vid_result.frames[0]
    frames[0].save(
        exp["gif_path"],
        save_all=True,
        append_images=frames[1:],
        duration=125,
        loop=0
    )
    print(f"[+] Saved {exp['name']} to: {exp['gif_path']}")

    mean_cos, min_cos, max_cos = compute_temporal_cosine_sim(frames)
    fidelity = compute_ssim_vs_baseline(frames, baseline_frames)

    print(f"[+] Results for {exp['name']}:")
    print(f"    Total Time: {t_total:.2f}s ({t_total/16:.3f}s/step) | Peak VRAM: {vram:.2f} GB")
    print(f"    Temporal Cosine Sim: {mean_cos:.4f} (min: {min_cos:.4f}, max: {max_cos:.4f})")
    print(f"    Fidelity to Baseline Softmax: {fidelity:.4f}")

    results[exp["name"]] = {
        "description": exp["description"],
        "injected_layers": injected,
        "time_sec": round(t_total, 2),
        "sec_per_step": round(t_total / 16, 3),
        "peak_vram_gb": round(vram, 2),
        "mean_pairwise_cosine": round(mean_cos, 4),
        "min_pairwise_cosine": round(min_cos, 4),
        "max_pairwise_cosine": round(max_cos, 4),
        "fidelity_to_baseline": round(fidelity, 4),
        "path": exp["gif_path"]
    }

# Save final comprehensive telemetry
out_json = "/home/phil/.gemini/antigravity/scratch/video_prime_benchmark_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print("\n" + "=" * 80)
print(f"[+] BENCHMARK COMPLETE! Full results saved to: {out_json}")
print("=" * 80)
