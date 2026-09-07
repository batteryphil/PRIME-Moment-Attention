#!/usr/bin/env python3
"""
AnimateDiff Temporal Attention Horizon Benchmark
================================================
Benchmarks:
  1. Baseline Softmax Temporal Attention (16 frames)
  2. PRIME Mid-Block Trunk Anchor (16 frames)
  3. Hybrid Temporal Attention (Local W=8 Softmax + Global PRIME) (16 frames)

Evaluates:
  - Latency (total and per-step)
  - Peak VRAM footprint
  - Consecutive frame cosine similarity
  - Horizon drift: similarity between frame t and frame 0
  - Visual output: animated GIFs and multi-frame progression strips
"""

import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["MIOPEN_USER_DB_PATH"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_LOCK_FILE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_DEBUG_DISABLE_LOCK"] = "1"
os.environ["TRITON_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/.triton"
os.environ["PYTHONUNBUFFERED"] = "1"

import sys
sys.path.append("/home/phil/.gemini/antigravity/scratch/MuseTalk/venv/lib/python3.12/site-packages")
sys.path.append("/home/phil/.gemini/antigravity/scratch")

import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from vision_prime_adapter import (
    inject_prime_temporal_attention,
    restore_default_temporal_attention
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] Using {device}")

sd_path = "/home/phil/.cache/huggingface/hub/models--SG161222--Realistic_Vision_V5.1_noVAE/snapshots/1e9f017a7b1eaefb63a1900ea6c5953d2739fd21"
adapter_path = "/home/phil/.cache/huggingface/hub/models--guoyww--animatediff-motion-adapter-v1-5-2/snapshots/6167b88ffe39b4441fdf2113e77b99a6f56b7906"

prompt = "a majestic bald eagle soaring over a snow-capped mountain range at sunrise, cinematic lighting, 8k resolution, photorealistic, continuous smooth flight"
negative_prompt = "blurry, low quality, distorted, deformed, artifacts, static, flickering"
seed = 42
TARGET_FRAMES = 16
INFERENCE_STEPS = 16

def compute_video_metrics(frames):
    sims = []
    drift_from_f0 = []
    f0 = torch.tensor(np.array(frames[0]), dtype=torch.float32).flatten()
    f0_norm = f0 / (f0.norm() + 1e-6)

    for i in range(len(frames)):
        fi = torch.tensor(np.array(frames[i]), dtype=torch.float32).flatten()
        fi_norm = fi / (fi.norm() + 1e-6)
        drift_from_f0.append(torch.dot(f0_norm, fi_norm).item())
        if i > 0:
            f_prev = torch.tensor(np.array(frames[i-1]), dtype=torch.float32).flatten()
            cos = F.cosine_similarity(fi.unsqueeze(0), f_prev.unsqueeze(0)).item()
            sims.append(cos)

    return {
        "mean_consecutive_cosine": float(np.mean(sims)),
        "min_consecutive_cosine": float(np.min(sims)),
        "f0_to_end_cosine": float(drift_from_f0[-1]),
        "mean_drift_from_f0": float(np.mean(drift_from_f0)),
    }

def create_progression_strip(frames, save_path, step_indices=[0, 3, 7, 11, 15]):
    selected = [frames[i] for i in step_indices if i < len(frames)]
    w, h = selected[0].size
    strip = Image.new("RGB", (w * len(selected), h))
    for idx, img in enumerate(selected):
        strip.paste(img, (idx * w, 0))
    strip.save(save_path)
    print(f"[+] Saved progression strip to: {save_path}")

print("[1] Loading AnimateDiff Pipeline...")
adapter = MotionAdapter.from_pretrained(adapter_path, torch_dtype=torch.float16, local_files_only=True)
pipe = AnimateDiffPipeline.from_pretrained(sd_path, motion_adapter=adapter, torch_dtype=torch.float16, local_files_only=True)
pipe.scheduler = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", steps_offset=1)
pipe.vae.enable_slicing()
pipe.to(device)

out_dir = "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/dossier/telemetry"
os.makedirs(out_dir, exist_ok=True)
media_dir = "/home/phil/.gemini/antigravity/scratch/media"
os.makedirs(media_dir, exist_ok=True)

conditions = [
    {
        "name": "baseline_softmax",
        "description": f"Standard Softmax Temporal Attention ({TARGET_FRAMES} frames)",
        "scope": "none",
        "mode": "softmax",
        "gif_path": os.path.join(media_dir, f"animatediff_baseline_softmax_{TARGET_FRAMES}f.gif"),
        "strip_path": os.path.join(media_dir, f"animatediff_baseline_softmax_strip_{TARGET_FRAMES}f.png")
    },
    {
        "name": "prime_mid_trunk",
        "description": f"PRIME Mid-Block Trunk Anchor ({TARGET_FRAMES} frames)",
        "scope": "mid",
        "mode": "prime",
        "order": 2,
        "window": 8,
        "gif_path": os.path.join(media_dir, f"animatediff_prime_mid_trunk_{TARGET_FRAMES}f.gif"),
        "strip_path": os.path.join(media_dir, f"animatediff_prime_mid_trunk_strip_{TARGET_FRAMES}f.png")
    },
    {
        "name": "prime_hybrid",
        "description": f"Hybrid Temporal Attention (Local W=8 Softmax + Global PRIME) ({TARGET_FRAMES} frames)",
        "scope": "all",
        "mode": "hybrid",
        "order": 2,
        "window": 8,
        "gif_path": os.path.join(media_dir, f"animatediff_prime_hybrid_{TARGET_FRAMES}f.gif"),
        "strip_path": os.path.join(media_dir, f"animatediff_prime_hybrid_strip_{TARGET_FRAMES}f.png")
    }
]

benchmark_results = {}

for cond in conditions:
    cname = cond["name"]
    print("\n" + "="*85)
    print(f"RUNNING CONDITION: {cname.upper()} ({TARGET_FRAMES} FRAMES)")
    print(f"{cond['description']}")
    print("="*85)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Reset or inject attention
    restore_default_temporal_attention(pipe.unet)
    if cond["scope"] != "none":
        injected = inject_prime_temporal_attention(
            pipe.unet,
            scope=cond["scope"],
            processor_mode=cond["mode"],
            order=cond.get("order", 2),
            window_size=cond.get("window", 8)
        )
        print(f"[+] Injected {injected} PRIME temporal processors (scope={cond['scope']})")
    else:
        print("[+] Standard Softmax temporal attention active.")

    generator = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    
    with torch.no_grad():
        vid_result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=TARGET_FRAMES,
            num_inference_steps=INFERENCE_STEPS,
            guidance_scale=7.5,
            generator=generator,
            width=512,
            height=512
        )
    t_elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    frames = vid_result.frames[0]
    
    # Save GIF
    frames[0].save(
        cond["gif_path"],
        save_all=True,
        append_images=frames[1:],
        duration=62, # ~16 fps
        loop=0
    )
    print(f"[+] Saved GIF to: {cond['gif_path']}")

    # Save 5-frame progression strip
    create_progression_strip(frames, cond["strip_path"])

    # Compute quantitative consistency metrics
    metrics = compute_video_metrics(frames)
    
    print(f"\n[+] Results for {cname}:")
    print(f"    Total Time: {t_elapsed:.2f}s ({t_elapsed/INFERENCE_STEPS:.2f}s/step) | Peak VRAM: {peak_vram:.2f} GB")
    print(f"    Consecutive Frame Cosine: {metrics['mean_consecutive_cosine']:.4f} (min: {metrics['min_consecutive_cosine']:.4f})")
    print(f"    Frame 0 -> End Retention Cosine: {metrics['f0_to_end_cosine']:.4f}")
    print(f"    Mean Trajectory Retention: {metrics['mean_drift_from_f0']:.4f}")

    benchmark_results[cname] = {
        "description": cond["description"],
        "num_frames": TARGET_FRAMES,
        "inference_steps": INFERENCE_STEPS,
        "time_sec": round(t_elapsed, 2),
        "sec_per_step": round(t_elapsed / INFERENCE_STEPS, 3),
        "peak_vram_gb": round(peak_vram, 2),
        "mean_consecutive_cosine": round(metrics["mean_consecutive_cosine"], 4),
        "min_consecutive_cosine": round(metrics["min_consecutive_cosine"], 4),
        "f0_to_end_cosine": round(metrics["f0_to_end_cosine"], 4),
        "mean_drift_from_f0": round(metrics["mean_drift_from_f0"], 4),
        "gif_path": cond["gif_path"],
        "strip_path": cond["strip_path"]
    }

# Save Telemetry
out_json = os.path.join(out_dir, "animatediff_temporal_benchmark_results.json")
with open(out_json, "w") as f:
    json.dump(benchmark_results, f, indent=2)

print("\n" + "="*85)
print(f"[+] ANIMATEDIFF TEMPORAL BENCHMARK COMPLETE! Telemetry saved to: {out_json}")
print("="*85)
