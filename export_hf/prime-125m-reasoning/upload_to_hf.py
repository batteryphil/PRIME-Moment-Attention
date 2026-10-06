"""
One-Command Hugging Face Hub Uploader for PRIME-125M-Reasoning
=============================================================
Usage:
  python upload_to_hf.py --repo-id <YOUR_USERNAME>/prime-125m-reasoning [--token <HF_TOKEN>]
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo

EXPORT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(description="Upload PRIME-125M to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo ID (e.g., username/prime-125m-reasoning)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token (or login with `huggingface-cli login`)")
    parser.add_argument("--private", action="store_true", help="Create as a private repository")
    args = parser.parse_args()

    api = HfApi(token=args.token)
    
    print(f"[*] Creating/verifying repository: {args.repo_id}...")
    create_repo(repo_id=args.repo_id, token=args.token, private=args.private, exist_ok=True)

    print(f"[*] Uploading folder '{EXPORT_DIR}' to '{args.repo_id}'...")
    api.upload_folder(
        folder_path=EXPORT_DIR,
        repo_id=args.repo_id,
        token=args.token,
        ignore_patterns=["upload_to_hf.py", "__pycache__/*", "*.pyc"]
    )
    print(f"\n[+] SUCCESS! Model is live on Hugging Face:")
    print(f"    https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
