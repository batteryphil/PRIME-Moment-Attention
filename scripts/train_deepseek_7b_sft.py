#!/usr/bin/env python3
"""
DeepSeek-R1-Distill-Qwen-7B LoRA SFT Pipeline
=============================================
Fine-tunes DeepSeek-R1-Distill-Qwen-7B with LoRA on AMD ROCm (17.1 GB VRAM).
Focuses on self-correction, reasoning verification, and step critique in <think>.
"""

import os
import sys
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

MODEL_PATH = "/data/models/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_DIR = "/data/prime_checkpoints/deepseek_7b_reasoning_lora"

# Grounded Metacognitive & Self-Verification SFT Samples
VERIFICATION_SFT_DATA = [
    {
        "prompt": "Natalia sold clips to 48 friends in April, and then half as many in May. How many clips in total?",
        "completion": "<think>\nLet's calculate the clips sold in each month.\nIn April, Natalia sold 48 clips.\nIn May, she sold half as many clips as in April.\nInitial thought: maybe she sold 48 / 4 = 12 clips?\nWait, let me verify: 'half as many' means dividing by 2, not 4.\nLet's recompute: 48 / 2 = 24 clips in May.\nNow let's compute the total for both months:\nTotal = April + May = 48 + 24.\nCheck: 48 + 20 = 68, 68 + 4 = 72.\nDouble checking: 72 - 24 = 48. This is verified.\n</think>\nNatalia sold 72 clips altogether in April and May."
    },
    {
        "prompt": "Solve for x: 3 * x + 7 = 22",
        "completion": "<think>\nWe are given the linear algebraic equation: 3 * x + 7 = 22.\nFirst step: subtract 7 from both sides.\n3 * x = 22 - 7.\nCalculate 22 - 7: 22 - 7 = 15.\nNext step: divide both sides by 3.\nx = 15 / 3.\nCheck: 15 / 3 = 5.\nVerification by substitution:\nLeft-hand side: 3 * (5) + 7 = 15 + 7 = 22.\nRight-hand side: 22.\nThe equality holds exactly.\n</think>\nThe value of x is 5."
    },
    {
        "prompt": "A bakery had 80 cakes. They sold 35 in the morning and 25 in the afternoon. How many cakes are left?",
        "completion": "<think>\nInitial cakes available = 80.\nMorning sales = 35 cakes.\nAfternoon sales = 25 cakes.\nLet's sum the total cakes sold:\nTotal sold = 35 + 25.\nCalculate: 35 + 25 = 60.\nWait, let me make sure: 30 + 20 = 50, 5 + 5 = 10, 50 + 10 = 60. Correct.\nNow calculate remaining cakes:\nRemaining = Initial - Total sold = 80 - 60 = 20 cakes.\nAlternative check:\n80 - 35 = 45 cakes left after morning.\n45 - 25 = 20 cakes left after afternoon.\nBoth methods yield 20.\n</think>\nThere are 20 cakes left."
    },
    {
        "prompt": "What is 20 percent of 150?",
        "completion": "<think>\nWe need to find 20% of 150.\nDefinition: 20% is 20 / 100 = 0.20 = 1 / 5.\nMethod 1: Fraction multiplication\n150 * (1 / 5) = 150 / 5 = 30.\nMethod 2: Base 10 multiplication\n10% of 150 is 15.\n20% is twice 10%: 2 * 15 = 30.\nBoth methods give 30.\n</think>\n20 percent of 150 is 30."
    },
    {
        "prompt": "A customer bought 3 books for $15 each and 2 pens for $4 each. What is the total cost?",
        "completion": "<think>\nLet's break down the purchase items and prices.\nItem 1: 3 books at $15 each.\nCost of books = 3 * 15 = 45 dollars.\nItem 2: 2 pens at $4 each.\nCost of pens = 2 * 4 = 8 dollars.\nTotal cost = Cost of books + Cost of pens = 45 + 8.\nCheck: 45 + 5 = 50, 50 + 3 = 53 dollars.\nLet's verify: 53 - 8 = 45, 45 / 3 = 15. The numbers are consistent.\n</think>\nThe total cost is $53."
    },
    {
        "prompt": "Find the rule governing the sequence: 1, 4, 9, 16, 25.",
        "completion": "<think>\nLet's examine the sequence terms index by index (n = 1, 2, 3, 4, 5):\nTerm 1: 1\nTerm 2: 4\nTerm 3: 9\nTerm 4: 16\nTerm 5: 25\nLet's test differences:\n4 - 1 = 3\n9 - 4 = 5\n16 - 9 = 7\n25 - 16 = 9\nThe first differences are 3, 5, 7, 9 (increasing by 2 each step).\nConstant second difference of 2 implies a quadratic polynomial: a*n^2 + b*n + c.\nNotice directly:\n1 = 1^2\n4 = 2^2\n9 = 3^2\n16 = 4^2\n25 = 5^2\nTherefore, the closed-form invariant rule is f(n) = n^2.\n</think>\nThe sequence follows the rule f(n) = n^2 (the squares of consecutive positive integers)."
    },
]


def prepare_dataset(tokenizer, data_list, max_len=512):
    formatted = []
    for item in data_list:
        text = f"<｜User｜>{item['prompt']}<｜Assistant｜>{item['completion']}"
        tokens = tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        tokens["labels"] = list(tokens["input_ids"])
        formatted.append(tokens)
    return Dataset.from_list(formatted)


def train_lora(model_path: str = MODEL_PATH, output_dir: str = OUTPUT_DIR, epochs: int = 3, lr: float = 2e-4):
    print("=" * 80)
    print("  DEEPSEEK-R1-DISTILL-QWEN-7B: LoRA REASONING SFT")
    print(f"  Target Device: AMD ROCm (17.1 GB VRAM)")
    print(f"  Base Model: {model_path}")
    print(f"  Output Checkpoint: {output_dir}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    print("[*] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[*] Loading base model in 4-bit NF4 with gradient checkpointing...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": "cuda"},
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    print("[*] Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    print("[*] Preparing training dataset...")
    train_dataset = prepare_dataset(tokenizer, VERIFICATION_SFT_DATA)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        dataloader_num_workers=0,
        warmup_steps=1,
        report_to="none",
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )

    print("\n[*] Starting LoRA fine-tuning...")
    trainer.train()

    print(f"\n[+] Saving LoRA adapter checkpoint to {output_dir}...")
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[SUCCESS] LoRA fine-tuning complete and saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    train_lora(epochs=args.epochs, lr=args.lr)
