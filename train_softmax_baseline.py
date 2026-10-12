import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_associative_recall_batch(batch_size, num_pairs, vocab_size, device):
    keys = torch.randint(100, 1000, (batch_size, num_pairs), device=device)
    vals = torch.randint(100, 1000, (batch_size, num_pairs), device=device)
    seq = torch.zeros(batch_size, num_pairs * 2 + 2, dtype=torch.long, device=device)
    seq[:, 0::2][:, :-1] = keys
    seq[:, 1::2][:, :-1] = vals
    target_idx = torch.randint(0, num_pairs, (batch_size,), device=device)
    query_keys = keys[torch.arange(batch_size), target_idx]
    target_vals = vals[torch.arange(batch_size), target_idx]
    seq[:, -2] = query_keys
    seq[:, -1] = target_vals
    x = seq[:, :-1]
    y = torch.full_like(seq[:, 1:], -100)
    y[:, -1] = target_vals 
    return x, y

def main():
    model_name = "HuggingFaceTB/SmolLM-135M"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cuda:0")
    
    # Randomize the attention projections so it starts from scratch just like the PRIME test did 
    # (since PRIME's projection math is different, we want a fair comparison)
    # Actually, let's leave them pre-trained! If pre-trained softmax can't learn it in 300 steps after we freeze MLP, 
    # then obviously an untrained PRIME wouldn't either.
    # Wait, if we leave them pre-trained, it already knows associative recall!
    # Let's test if it already knows it zero-shot.
    
    model.eval()
    correct = 0
    total = 200
    with torch.no_grad():
        x, y = generate_associative_recall_batch(total, 10, model.config.vocab_size, "cuda:0")
        outputs = model(input_ids=x)
        logits = outputs.logits[:, -1, :] 
        preds = logits.argmax(dim=-1)
        targets = y[:, -1]
        correct = (preds == targets).sum().item()
        
    print(f"[*] Softmax Zero-Shot Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    
    # Now let's randomize attention weights and see if it can relearn in 300 steps
    for layer in model.model.layers:
        layer.self_attn.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        layer.self_attn.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        layer.self_attn.v_proj.weight.data.normal_(mean=0.0, std=0.02)
        layer.self_attn.o_proj.weight.data.normal_(mean=0.0, std=0.02)
        
    for param in model.parameters():
        param.requires_grad = False
    for layer in model.model.layers:
        layer.self_attn.q_proj.weight.requires_grad = True
        layer.self_attn.k_proj.weight.requires_grad = True
        layer.self_attn.v_proj.weight.requires_grad = True
        layer.self_attn.o_proj.weight.requires_grad = True
        
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    model.train()
    for step in range(300):
        optimizer.zero_grad()
        x, y = generate_associative_recall_batch(32, 10, model.config.vocab_size, "cuda:0")
        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    model.eval()
    correct = 0
    with torch.no_grad():
        x, y = generate_associative_recall_batch(total, 10, model.config.vocab_size, "cuda:0")
        outputs = model(input_ids=x)
        logits = outputs.logits[:, -1, :] 
        preds = logits.argmax(dim=-1)
        targets = y[:, -1]
        correct = (preds == targets).sum().item()
        
    print(f"[*] Softmax Re-learn Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    main()
