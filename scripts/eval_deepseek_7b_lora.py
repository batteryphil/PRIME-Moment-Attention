#!/usr/bin/env python3
"""
Test Inference with Fine-Tuned DeepSeek-R1-Distill-Qwen-7B LoRA
===============================================================
Loads base 7B model in 4-bit, attaches the fine-tuned LoRA adapter,
and generates self-correcting reasoning traces on AMD ROCm GPU.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_PATH = "/data/models/DeepSeek-R1-Distill-Qwen-7B"
LORA_PATH = "/data/prime_checkpoints/deepseek_7b_reasoning_lora"

def test_lora_inference():
    print("=" * 80)
    print("  EVALUATING FINE-TUNED DEEPSEEK-R1-DISTILL-QWEN-7B (LoRA)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target Device: {device} ({torch.cuda.get_device_name(0)})")

    print("[*] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[*] Loading base model in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": device},
    )

    print("[*] Attaching fine-tuned LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    test_prompt = "Weng earns $12 an hour for babysitting. Yesterday, she did 50 minutes of babysitting. How much did she earn?"
    formatted = f"<｜User｜>{test_prompt}<｜Assistant｜><think>\n"

    print(f"\n[Prompt]: {test_prompt}")
    print("[*] Generating reasoning trace...")

    input_ids = tokenizer.encode(formatted, return_tensors="pt").to(device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=300,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    print("\n[Generated Output]:")
    print(response)
    print("=" * 80)

if __name__ == "__main__":
    test_lora_inference()
