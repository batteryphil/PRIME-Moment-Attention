#!/usr/bin/env python3
"""
Identity Verification Test: Base PRIME vs Warm-Started PRIME-Selective
======================================================================
Confirms that loading the 125M Base checkpoint into PRIME-Selective produces
identical output logits and cross-entropy loss on Step 0.
"""

import sys
import torch
from prime_moment_attention import PrimeConfig, PrimeForCausalLM

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "checkpoints/prime_125m_step_10000.pt"

    print("=" * 65)
    print("  IDENTITY WARM-START VERIFICATION TEST")
    print(f"  Device: {device} | Checkpoint: {ckpt_path}")
    print("=" * 65)

    print("\n[1] Loading Base PRIME model...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    base_config = ckpt["config"]
    base_model = PrimeForCausalLM(base_config).to(device)
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_model.eval()

    print("\n[2] Loading PRIME-Selective model via Warm-Start...")
    sel_model = PrimeForCausalLM.from_pretrained_base_to_selective(ckpt_path, device=device)
    sel_model.eval()

    print("\n[3] Forward pass on test batch (L=64, B=2)...")
    torch.manual_seed(42)
    dummy_input = torch.randint(0, 1000, (2, 64), device=device)
    dummy_labels = torch.randint(0, 1000, (2, 64), device=device)

    with torch.no_grad():
        out_base = base_model(dummy_input, labels=dummy_labels)
        out_sel = sel_model(dummy_input, labels=dummy_labels)

    loss_base = out_base["loss"].item()
    loss_sel = out_sel["loss"].item()
    diff_logits = torch.max(torch.abs(out_base["logits"] - out_sel["logits"])).item()
    diff_loss = abs(loss_base - loss_sel)

    print(f"\n  Base PRIME Loss:      {loss_base:.6f}")
    print(f"  PRIME-Selective Loss: {loss_sel:.6f}")
    print(f"  Max Logit Diff:       {diff_logits:.2e}")
    print(f"  Loss Diff:            {diff_loss:.2e}")

    assert diff_logits < 1e-3, f"Logit diff too large: {diff_logits}"
    assert diff_loss < 1e-4, f"Loss diff too large: {diff_loss}"

    print("\n[4] Autoregressive Generation Probe Check...")
    prompt = dummy_input[:, :10]
    gen_base = base_model.generate(prompt, max_new_tokens=10, temperature=0.0)
    gen_sel = sel_model.generate(prompt, max_new_tokens=10, temperature=0.0)

    token_matches = (gen_base == gen_sel).all().item()
    print(f"  Generated tokens identical: {token_matches}")
    assert token_matches, "Generated tokens diverged!"

    print("\n" + "=" * 65)
    print(">>> VERIFICATION PASSED: Step 0 mathematical identity confirmed! <<<")
    print("=" * 65)

if __name__ == "__main__":
    main()
