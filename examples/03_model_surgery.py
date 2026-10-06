#!/usr/bin/env python3
"""
03_model_surgery.py
===================
Demonstrates zero-shot surgical transplantation:
  - Takes an attention block from a pretrained Transformer (or standard HF architecture).
  - Replaces its standard Softmax attention mechanism with PRIME recurrence.
  - Verifies that forward pass runs without shape or dimension mismatch.
"""

import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Config
from prime_moment_attention import PrimeTransplantedAttention, convert_transformer_to_prime

def main():
    print("=" * 70)
    print("PRIME Model Surgery: Transplanting Recurrent Attention into Transformers")
    print("=" * 70)

    # 1. Instantiate standard Transformer self-attention block
    cfg = Qwen2Config(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512
    )
    orig_attn = Qwen2Attention(cfg, layer_idx=5)
    orig_attn.eval()
    print("[1] Created standard Transformer self-attention block (Qwen2 architecture).")

    # 2. Perform surgery: Replace with PrimeTransplantedAttention
    prime_layer = PrimeTransplantedAttention(orig_attn, layer_idx=5, decay=0.9995)
    prime_layer.eval()
    print("[2] Successfully wrapped layer with PrimeTransplantedAttention.")

    # 3. Test forward pass with input sequence
    batch_size = 2
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, cfg.hidden_size)

    with torch.no_grad():
        out, _ = prime_layer(hidden_states)

    print(f"[3] Forward pass through transplanted layer: {list(hidden_states.shape)} -> {list(out.shape)}")
    assert out.shape == hidden_states.shape, "Output shape must match input hidden dimension!"

    # 4. Demonstrate convert_transformer_to_prime helper function
    class DummyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([
                nn.Module() for _ in range(4)
            ])
            for i, layer in enumerate(self.model.layers):
                layer.self_attn = Qwen2Attention(cfg, layer_idx=i)

    dummy_model = DummyTransformer()
    print(f"\n[4] Dummy 4-layer model instantiated with Softmax attention.")
    # Convert layer 1 and 2 (trunk layers) to PRIME
    converted_model, converted_layers = convert_transformer_to_prime(dummy_model, target_layers=[1, 2], decay=0.999)
    print(f"[5] Converted layers [1, 2] to PRIME recurrence (Trunk surgery).")
    print(f"    Layer 0 type: {type(converted_model.model.layers[0].self_attn).__name__} (Preserved Softmax)")
    print(f"    Layer 1 type: {type(converted_model.model.layers[1].self_attn).__name__} (PRIME Recurrent)")
    print(f"    Layer 2 type: {type(converted_model.model.layers[2].self_attn).__name__} (PRIME Recurrent)")
    print(f"    Layer 3 type: {type(converted_model.model.layers[3].self_attn).__name__} (Preserved Softmax)")

    print("\n[SUCCESS] Model surgery demonstration complete.")

if __name__ == "__main__":
    main()
