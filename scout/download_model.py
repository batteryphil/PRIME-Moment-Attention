"""
Model Downloader for Stheno-v3.2
================================
Downloads Sao10K/L3-8B-Stheno-v3.2 into the local HuggingFace cache.
"""

import sys
import time
from huggingface_hub import snapshot_download

MODEL_ID = "Sao10K/L3-8B-Stheno-v3.2"

def main():
    print(f"[*] Starting download of {MODEL_ID} (safetensors & config)...")
    t0 = time.time()
    try:
        path = snapshot_download(
            repo_id=MODEL_ID,
            repo_type="model",
            allow_patterns=["*.safetensors", "*.json"],
            max_workers=4
        )
        elapsed = time.time() - t0
        print(f"\n[✓] Successfully downloaded {MODEL_ID} to {path} in {elapsed:.1f}s")
    except Exception as e:
        print(f"\n[✗] Download failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
