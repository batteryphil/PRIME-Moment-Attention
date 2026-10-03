import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

text = "Here is a mathematical proof. We know that 2 + 2 = 4. Therefore, if we take the integral of x, we get x^2 / 2. " * 5
tokens = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
batch = tokens.repeat(2, 1)

with torch.no_grad():
    outputs = model(input_ids=batch[:, :-1], labels=batch[:, 1:])
    print(f"[*] Base DeepSeek-1.5B Loss (Unmodified): {outputs.loss.item():.4f}")
