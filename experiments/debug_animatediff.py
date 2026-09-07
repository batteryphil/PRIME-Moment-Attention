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
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter
from diffusers.models.embeddings import SinusoidalPositionalEmbedding

device = "cuda" if torch.cuda.is_available() else "cpu"
sd_path = "/home/phil/.cache/huggingface/hub/models--SG161222--Realistic_Vision_V5.1_noVAE/snapshots/1e9f017a7b1eaefb63a1900ea6c5953d2739fd21"
adapter_path = "/home/phil/.cache/huggingface/hub/models--guoyww--animatediff-motion-adapter-v1-5-2/snapshots/6167b88ffe39b4441fdf2113e77b99a6f56b7906"

print("Loading adapter and pipeline...", flush=True)
adapter = MotionAdapter.from_pretrained(adapter_path, torch_dtype=torch.float16, local_files_only=True)
pipe = AnimateDiffPipeline.from_pretrained(sd_path, motion_adapter=adapter, torch_dtype=torch.float16, local_files_only=True)
pipe.vae.enable_slicing()
pipe.to(device)

def extend_pe(pipe, target_length):
    for mod in pipe.unet.modules():
        if isinstance(mod, SinusoidalPositionalEmbedding):
            channels = mod.pe.shape[-1]
            new_pe = SinusoidalPositionalEmbedding(channels, max_seq_length=target_length)
            mod.pe = new_pe.pe.to(device=mod.pe.device, dtype=mod.pe.dtype)

print("Starting Frame Sweep...", flush=True)
for nf in [16, 24, 32, 40, 48, 64]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    extend_pe(pipe, nf)
    try:
        with torch.no_grad():
            out = pipe("a photo of an eagle", num_frames=nf, num_inference_steps=2, height=512, width=512)
        vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"num_frames = {nf:2d} -> SUCCESS! Peak VRAM: {vram:.2f} GB / 15.92 GB", flush=True)
    except torch.OutOfMemoryError as e:
        print(f"num_frames = {nf:2d} -> OOM! Hit hardware ceiling.", flush=True)
        break
    except Exception as e:
        print(f"num_frames = {nf:2d} -> Exception: {type(e).__name__}: {e}", flush=True)
        break
