import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import sys
sys.path.append("/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/src")
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from gumbel_qwen_distillation import convert_qwen_to_stage7_hybrid
import pyarrow.parquet as pq

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/home/phil/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-1.5B-Instruct/snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

student = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, local_files_only=True).to(device)
student, _, trunk = convert_qwen_to_stage7_hybrid(student)

code_parquet = "/home/phil/.cache/huggingface/hub/datasets--iamtarun--python_code_instructions_18k_alpaca/snapshots/7cae181e29701a8663a07a3ea43c8e105b663ba1/data/train-00000-of-00001-8b6e212f3e1ece96.parquet"
code_table = pq.read_table(code_parquet)
batch_texts = [f"{code_table['instruction'][0]}\n{code_table['output'][0]}", f"{code_table['instruction'][1]}\n{code_table['output'][1]}"]

enc = tokenizer(batch_texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt").to(device)
input_ids = enc.input_ids
attention_mask = enc.attention_mask

trainable_params = []
for idx in trunk:
    # Train only router_proj to learn the hard routing
    trainable_params.extend(list(student.model.layers[idx].self_attn.router.parameters()))

optimizer = AdamW(trainable_params, lr=1e-3)

for s in range(1, 6):
    student.train()
    optimizer.zero_grad()
    s_out = student(input_ids, attention_mask=attention_mask)
    
    labels = input_ids[:, 1:].contiguous()
    shift_logits = s_out.logits[:, :-1, :].contiguous().float()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), labels.view(-1), ignore_index=tokenizer.pad_token_id)
    
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()
    
    for idx in trunk:
        student.model.layers[idx].self_attn.router.scheduler.step()
        
    print(f"Step {s}: Loss = {loss.item():.4f}, GradNorm = {grad_norm.item():.4f}")
