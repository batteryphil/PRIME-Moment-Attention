#!/usr/bin/env python3
"""
Download DeepSeek-R1-Distill-Qwen-7B
====================================
Downloads model weights and tokenizer directly into /data/models/DeepSeek-R1-Distill-Qwen-7B.
"""

import os
import sys
import time
from huggingface_hub import snapshot_download

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TARGET_DIR = "/data/models/DeepSeek-R1-Distill-Qwen-7B"

os.makedirs(TARGET_DIR, exist_ok=True)

print(f"[*] Initiating download of {MODEL_ID}...")
print(f"[*] Destination: {TARGET_DIR}")

t0 = time.time()
path = snapshot_download(
    repo_id=MODEL_ID,
    local_dir=TARGET_DIR,
    local_dir_use_symlinks=False,
    resume_download=True,
)
elapsed = time.time() - t0

print(f"[SUCCESS] Download complete in {elapsed:.1f}s. Target: {path}")
