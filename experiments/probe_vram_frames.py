import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["MIOPEN_USER_DB_PATH"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_LOCK_FILE_DIR"] = "/home/phil/.gemini/antigravity/scratch/miopen_db"
os.environ["MIOPEN_DEBUG_DISABLE_LOCK"] = "1"
os.environ["TRITON_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/.triton"
os.environ["TMPDIR"] = "/home/phil/.gemini/antigravity/scratch"

import sys
sys.path.append("/home/phil/.gemini/antigravity/scratch/MuseTalk/venv/lib/python3.12/site-packages")
sys.path.append("/home/phil/.gemini/antigravity/scratch")

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter
from diffusers.models.embeddings import SinusoidalPositionalEmbedding

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] Using {device}")

sd_path = "/home/phil/.cache/huggingface/hub/models--SG161222--Realistic_Vision_V5.1_noVAE/snapshots/1e9f017a7b1eaefb63a1900ea6c5953d2739fd21"
adapter_path = "/home/phil/.cache/huggingface/hub/models--guoyww--animatediff-motion-adapter-v1-5-2/snapshots/6167b88ffe39b4441fdf2113e77b99a6f56b7906"

print("Loading adapter and pipeline...")
adapter = MotionAdapter.from_pretrained(adapter_path, torch_dtype=torch.float16, local_files_only=True)
pipe = AnimateDiffPipeline.from_pretrained(sd_path, motion_adapter=adapter, torch_dtype=torch.float16, local_files_only=True)
pipe.vae.enable_slicing()
pipe.to(device)

def extend_pe(target_length):
    for mod in pipe.unet.modules():
        if isinstance(mod, SinusoidalPositionalEmbedding):
            channels = mod.pe.shape[-1]
            new_pe = SinusoidalPositionalEmbedding(channels, max_seq_length=target_length)
            mod.pe = new_pe.pe.to(device=mod.pe.device, dtype=mod.pe.dtype)

timestep = torch.tensor([500], device=device)
encoder_hidden_states = torch.randn(2, 77, 768, dtype=torch.float16, device=device)

results = {}
for nf in [16, 24, 32, 48, 64, 96, 128]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    extend_pe(nf)
    try:
        latents = torch.randn(2, 4, nf, 64, 64, dtype=torch.float16, device=device)
        with torch.no_grad():
            out = pipe.unet(latents, timestep, encoder_hidden_states=encoder_hidden_states)
        vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"num_frames = {nf:3d} -> SUCCEEDED! Peak VRAM = {vram:.2f} GB")
        results[nf] = vram
    except torch.OutOfMemoryError as e:
        print(f"num_frames = {nf:3d} -> OOM! Exceeded VRAM limit.")
        break
    except Exception as e:
        print(f"num_frames = {nf:3d} -> Exception: {e}")
        break

print("Probe complete. Results:", results)
