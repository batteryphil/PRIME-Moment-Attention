import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
import sys
sys.path.insert(0, 'src')
from prime_moment_attention.thought_reconstruction_map import GenerativeThoughtReconstructionLayer

class HybridQwen2GistLayer(nn.Module):
    def __init__(self, orig_layer: Qwen2DecoderLayer, d_map=32, decay=0.995):
        super().__init__()
        self.orig_layer = orig_layer
        self.gist = GenerativeThoughtReconstructionLayer(
            d_model=orig_layer.hidden_size,
            d_map=d_map,
            decay=decay
        )
        
        device = orig_layer.self_attn.q_proj.weight.device
        dtype = orig_layer.self_attn.q_proj.weight.dtype
        self.gist.to(device=device, dtype=dtype)
        
    def forward(self, hidden_states, *args, **kwargs):
        layer_outputs = self.orig_layer(
            hidden_states,
            *args, **kwargs
        )
        
        if isinstance(layer_outputs, tuple):
            orig_hidden_states = layer_outputs[0]
        else:
            orig_hidden_states = layer_outputs
            
        gist_out, _ = self.gist(orig_hidden_states, return_state=False)
        
        if isinstance(layer_outputs, tuple):
            return (gist_out,) + layer_outputs[1:]
        else:
            return gist_out

def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"[*] Loading {model_name} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    
    print("[*] Transplanting Cognitive Gist Maps onto every 4th layer...")
    target_layers = [3, 7, 11, 15, 19, 23]
    
    for idx in target_layers:
        if idx < len(model.model.layers):
            model.model.layers[idx] = HybridQwen2GistLayer(model.model.layers[idx])
            
    print("[*] Freezing trunk, setting Gist layers to trainable...")
    for p in model.parameters():
        p.requires_grad = False
        
    trainable_params = []
    for idx in target_layers:
        if idx < len(model.model.layers):
            for p in model.model.layers[idx].gist.parameters():
                p.requires_grad = True
                trainable_params.append(p)
                
    print(f"[*] Trainable parameters in Gist: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    
    text = "The code to the vault is 81729. I must remember this sequence. " * 3
    tokens = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    batch = tokens
    
    print(f"[*] Training Gist to memorize context (sequence length: {batch.shape[1]})...")
    
    model.train()
    for step in range(5):
        optimizer.zero_grad()
        outputs = model(input_ids=batch[:, :-1], labels=batch[:, 1:])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Step {step+1} | Loss: {loss.item():.4f}")

    print("[*] Script complete. Gist layers successfully backpropagated!")

if __name__ == "__main__":
    main()
