import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import sys
sys.path.append("/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/src")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from gumbel_qwen_distillation import convert_qwen_to_stage7_hybrid

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/home/phil/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

student = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, local_files_only=True).to(device)
student, _, trunk = convert_qwen_to_stage7_hybrid(student)

input_ids = tokenizer("def add(a, b):\n    return a + b", return_tensors="pt").input_ids.to(device)

print("Running student forward...")
out = student(input_ids)
print("Student logits:", out.logits.shape, "Has NaN:", out.logits.isnan().any().item(), "Min/Max:", out.logits.min().item(), out.logits.max().item())

labels = input_ids[:, 1:]
shift_logits = out.logits[:, :-1, :]
loss_lm = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), labels.view(-1))
print("Loss LM:", loss_lm.item(), "Has NaN:", loss_lm.isnan().item())
