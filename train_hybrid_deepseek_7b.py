#!/usr/bin/env python3
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
import sys
import argparse

sys.path.insert(0, 'src')
from prime_moment_attention.hybrid import HybridWindowPrimeAttention
from prime_moment_attention.thought_reconstruction_map import GenerativeThoughtReconstructionLayer

class ZeroInitGistWrapper(nn.Module):
    def __init__(self, orig_layer: Qwen2DecoderLayer, d_map=32, decay=0.995):
        super().__init__()
        self.orig_layer = orig_layer
        self.gist = GenerativeThoughtReconstructionLayer(
            d_model=orig_layer.hidden_size,
            d_map=d_map,
            decay=decay
        )
        
        nn.init.zeros_(self.gist.recon_proj.weight)
        if self.gist.recon_proj.bias is not None:
            nn.init.zeros_(self.gist.recon_proj.bias)
            
        device = orig_layer.self_attn.q_proj.weight.device
        dtype = torch.bfloat16
        self.gist.to(device=device, dtype=dtype)
        
    def forward(self, hidden_states, *args, **kwargs):
        layer_outputs = self.orig_layer(hidden_states, *args, **kwargs)
        
        if isinstance(layer_outputs, tuple):
            orig_hidden_states = layer_outputs[0]
        else:
            orig_hidden_states = layer_outputs
            
        gist_out, _ = self.gist(orig_hidden_states, return_state=False)
        
        if isinstance(layer_outputs, tuple):
            return (gist_out,) + layer_outputs[1:]
        else:
            return gist_out

def get_perplexity(model, tokenizer, text):
    model.eval()
    tokens = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=tokens[:, :-1], labels=tokens[:, 1:])
    return torch.exp(outputs.loss).item()

def main():
    parser = argparse.ArgumentParser(description="Train Hybrid DeepSeek-7B using 4-bit quantization")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    print(f"\n[1] Loading {model_name} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    layer_prime = 14
    layer_gist = 20
    
    print("\n[2] Transplanting PRIME and Gist layers into 7B...")
    
    orig_attn = model.model.layers[layer_prime].self_attn
    hybrid_layer = HybridWindowPrimeAttention(
        original_attn=orig_attn,
        layer_idx=layer_prime,
        window_size=512,
        decay=0.999,
        alpha=1.0 
    )
    model.model.layers[layer_prime].self_attn = hybrid_layer
    model.model.layers[layer_gist] = ZeroInitGistWrapper(model.model.layers[layer_gist])
    
    print(f"  - Layer {layer_prime} converted to HybridWindowPrimeAttention (alpha=1.0)")
    print(f"  - Layer {layer_gist} augmented with Zero-Init Gist Memory")

    print("\n[3] Freezing trunk and preparing injected layers for training...")
    for p in model.parameters():
        p.requires_grad = False
        
    for p in model.model.layers[layer_prime].self_attn.parameters():
        if p.dtype in [torch.float32, torch.bfloat16, torch.float16]:
            p.requires_grad = True
    for p in model.model.layers[layer_gist].gist.parameters():
        if p.dtype in [torch.float32, torch.bfloat16, torch.float16]:
            p.requires_grad = True
        
    model.gradient_checkpointing_enable()
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    # Simple training task
    train_text = "The secret launch code is 998877. " * 50  
    tokens = tokenizer(train_text, return_tensors="pt").input_ids.to(model.device)
    
    model.train()
    print(f"Starting training for {args.steps} steps...")
    for step in range(args.steps):
        optimizer.zero_grad()
        outputs = model(input_ids=tokens[:, :-1], labels=tokens[:, 1:])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"  Step {step+1}/{args.steps} | Loss: {loss.item():.4f}")

    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
