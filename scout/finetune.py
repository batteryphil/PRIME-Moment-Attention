"""
Local Romantasy LoRA Fine-Tuner for AMD ROCm
============================================
Provides self-contained LoRA (Low-Rank Adaptation) fine-tuning for 8B models
directly on Phil's local AMD GPU (ROCm 7.2) without external dependency conflicts.
"""

import os
import sys
import json
import time
import pathlib
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer


class LoRALinear(nn.Module):
    """
    Parameter-Efficient Low-Rank Adapter layer.
    W_new = W_base + (B @ A) * (alpha / r)
    """
    def __init__(self, base_layer: nn.Linear, r: int = 16, lora_alpha: float = 32.0, lora_dropout: float = 0.05):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.scaling = lora_alpha / r

        in_dim = base_layer.in_features
        out_dim = base_layer.out_features
        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        self.lora_A = nn.Linear(in_dim, r, bias=False, dtype=dtype, device=device)
        self.lora_B = nn.Linear(r, out_dim, bias=False, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Initialize weights: A with kaiming, B with zeros (so initial LoRA contribution is 0)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze base layer weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
        return base_out + lora_out


def apply_lora_to_model(model: nn.Module, target_modules: List[str] = None, r: int = 16, alpha: float = 32.0) -> Dict[str, LoRALinear]:
    """
    Recursively injects LoRALinear into specified attention projection modules.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    lora_layers = {}
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
                # Find parent module
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                layer_name = parts[-1]

                lora_layer = LoRALinear(module, r=r, lora_alpha=alpha)
                setattr(parent, layer_name, lora_layer)
                lora_layers[name] = lora_layer

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = (trainable_params / total_params) * 100.0
    print(f"[✓] [LoRA] Injected {len(lora_layers)} LoRA adapters | Trainable: {trainable_params:,} / {total_params:,} ({pct:.3f}%)")
    return lora_layers


def save_lora_weights(lora_layers: Dict[str, LoRALinear], save_path: str):
    """
    Extracts and saves only the trained LoRA adapter weights (compact file, <50MB).
    """
    state_dict = {}
    for name, layer in lora_layers.items():
        state_dict[f"{name}.lora_A.weight"] = layer.lora_A.weight.data.cpu()
        state_dict[f"{name}.lora_B.weight"] = layer.lora_B.weight.data.cpu()

    out_file = pathlib.Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, str(out_file))
    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"[✓] [LoRA] Adapter weights saved to {out_file} ({size_mb:.2f} MB)")


def load_lora_weights(model: nn.Module, weights_path: str, target_modules: List[str] = None, r: int = 16, alpha: float = 32.0):
    """
    Injects LoRA layers and loads pre-trained adapter weights.
    """
    lora_layers = apply_lora_to_model(model, target_modules=target_modules, r=r, alpha=alpha)
    state_dict = torch.load(weights_path, map_location="cpu")
    for name, layer in lora_layers.items():
        k_a = f"{name}.lora_A.weight"
        k_b = f"{name}.lora_B.weight"
        if k_a in state_dict and k_b in state_dict:
            layer.lora_A.weight.data.copy_(state_dict[k_a].to(layer.lora_A.weight.device))
            layer.lora_B.weight.data.copy_(state_dict[k_b].to(layer.lora_B.weight.device))
    print(f"[✓] [LoRA] Pre-trained weights loaded from {weights_path}")
    return lora_layers


def train_lora_on_novel(
    model_id: str,
    dataset_path: str,
    output_path: str = "models/adapters/dark_romantasy_v1.pt",
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4
):
    """
    Executes fine-tuning loop on AMD GPU with ROCm.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[*] [LoRA-Train] Loading base model: {model_id} onto {device} (bfloat16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    lora_layers = apply_lora_to_model(model, r=16, alpha=32.0)

    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"[*] Loaded {len(samples)} training samples from {dataset_path}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)

    model.train()
    total_steps = len(samples) * epochs
    step = 0
    t0 = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, item in enumerate(samples):
            text = item.get("text", "")
            inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
            labels = inputs["input_ids"].clone()

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item() * gradient_accumulation_steps

            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(samples):
                optimizer.step()
                optimizer.zero_grad()
                step += 1

            if (i + 1) % 5 == 0:
                print(f"    [Epoch {epoch+1}/{epochs} | Step {i+1}/{len(samples)}] Loss: {loss.item() * gradient_accumulation_steps:.4f}")

        avg_loss = epoch_loss / len(samples)
        print(f"[✓] Epoch {epoch+1}/{epochs} Complete | Avg Loss: {avg_loss:.4f}")

    elapsed = time.time() - t0
    print(f"\n[✓] Fine-tuning finished in {elapsed:.1f}s")
    save_lora_weights(lora_layers, output_path)
    return output_path


if __name__ == "__main__":
    print("PRIME LoRA Module initialized.")
