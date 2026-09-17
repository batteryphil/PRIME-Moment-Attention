"""
Standalone CLI utility to load any trained PrimeForCausalLM checkpoint
and run autoregressive text generation with O(1) constant memory.

Usage:
    python eval_checkpoint.py checkpoints/prime_125m_step_500.pt --prompt "Once upon a time, Lily"
"""

import argparse
import torch
from transformers import AutoTokenizer
from prime_moment_attention import PrimeForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Autoregressive generation from trained PRIME checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint file")
    parser.add_argument("--prompt", type=str, default="Once upon a time, Lily found a dog and", help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K sampling cutoff")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {args.checkpoint} on {device}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config")
    step = ckpt.get("step", "unknown")
    loss = ckpt.get("loss", "unknown")

    model = PrimeForCausalLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).to(torch.bfloat16 if device == "cuda" else torch.float32)
    model.eval()

    print(f"Checkpoint loaded: Step {step} | Recorded Loss: {loss}")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    print(f"\nPrompt: \"{args.prompt}\"")
    print("Generating with O(1) constant recurrent state...")
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
        )
    output_text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("\n" + "=" * 60)
    print(output_text)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
