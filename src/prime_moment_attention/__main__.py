"""
PRIME Moment Attention CLI Entrypoint
======================================
Allows running and serving PRIME-enabled models directly from the terminal.

Usage:
  python -m prime_moment_attention --server --port 8080
  python -m prime_moment_attention Qwen/Qwen2.5-0.5B --prompt "Hello, world!"
"""

import argparse
import sys
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(
        description="PRIME Moment Attention: One-Line Model Runner and Server"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Hugging Face model ID or path (e.g. 'Qwen/Qwen2.5-0.5B', 'meta-llama/Llama-3.2-1B')",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain constant-state recurrent attention in simple terms.",
        help="Prompt text for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "pure"],
        default="hybrid",
        help="Attention mode: 'hybrid' (sliding window + PRIME) or 'pure' (100%% recurrence)",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Launch the high-performance zero-dependency C REST server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the REST server (default: 8080)",
    )

    args = parser.parse_args()

    # Mode 1: Launch C Native REST server
    if args.server:
        bin_path = os.path.join(os.path.dirname(__file__), "..", "..", "c", "bin", "prime")
        if not os.path.exists(bin_path):
            bin_path = os.path.join(os.path.dirname(__file__), "..", "..", "c", "bin", "prime.com")
        if os.path.exists(bin_path):
            print(f"Launching PRIME C Engine Server on port {args.port}...")
            sys.exit(subprocess.call([bin_path, "--server", "--port", str(args.port)]))
        else:
            print("Native C binary not found in c/bin/. Building with 'make native'...")
            c_dir = os.path.join(os.path.dirname(__file__), "..", "..", "c")
            subprocess.check_call(["make", "-C", c_dir, "native"])
            sys.exit(subprocess.call([os.path.join(c_dir, "bin", "prime"), "--server", "--port", str(args.port)]))

    if not args.model:
        parser.print_help()
        print("\nExamples:")
        print("  python -m prime_moment_attention Qwen/Qwen2.5-0.5B")
        print("  python -m prime_moment_attention --server --port 8080")
        sys.exit(0)

    # Mode 2: Run HuggingFace Model in 1 Line
    print("=" * 65)
    print(f"  PRIME Moment Attention: Running {args.model}")
    print("=" * 65)
    
    import prime_moment_attention as prime
    from transformers import AutoTokenizer

    print(f"[*] Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"[*] Loading and converting model with mode='{args.mode}'...")
    model = prime.load(args.model, mode=args.mode)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    print(f"\n[Prompt]: {args.prompt}\n")
    print("[Generating tokens with O(1) constant attention memory]...")

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        do_sample=True,
        temperature=0.7,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "-" * 65)
    print(generated_text)
    print("-" * 65)
    print("[✓] Generation complete. Attention state memory remained strictly O(1)!")

if __name__ == "__main__":
    main()
