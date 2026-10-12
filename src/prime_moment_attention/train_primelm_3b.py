#!/usr/bin/env python3
"""
PrimeLM-50M: Production 3-Billion Token Training Campaign
=========================================================
Trains the 50.92M parameter PrimeLM model across 3,000,000,000 tokens.
Features:
  - 6-Corpus Balanced Multi-Stream Data Pipeline (SmolTalk, TinyStories, GSM8K, ARC, Invariants)
  - In-Layer Invariant Registers (Z_inv) and Invariant-Gated SwiGLU
  - Automated Checkpointing (Latest, Best, and Milestones every ~164M tokens)
  - Periodic Capability Probing & Repetition Degeneracy Tracking
  - Real-time JSON Telemetry & Atomic Resumption
"""

import os
import sys
import time
import json
import math
import signal
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# GPU / ROCm Performance Configuration
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["TRITON_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/.triton"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/home/phil/.gemini/antigravity/scratch/torchinductor_phil"
os.environ["TORCH_EXTENSIONS_DIR"] = "/home/phil/.gemini/antigravity/scratch/tmp"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

sys.path.insert(0, "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/src/prime_moment_attention")
sys.path.insert(0, "/home/phil/.gemini/antigravity/brain/488931f7-4ed7-40c4-bc4d-bcbc1f64de95/scratch")
from primelm_50m import PrimeLM50M
from streaming_billion_corpus import MultiCorpusBillionStreamer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--target_tokens", type=int, default=3_000_000_000, help="Total target tokens (default: 3 Billion)")
parser.add_argument("--micro_batch", type=int, default=8, help="Micro batch size per accumulation step")
parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps (8*1024*4 = 32,768 tokens/step)")
parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length (default: 1024)")
parser.add_argument("--lr", type=float, default=5.0e-4, help="Peak learning rate")
parser.add_argument("--min_lr", type=float, default=3.0e-5, help="Minimum decayed learning rate")
parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
parser.add_argument("--save_interval", type=int, default=1000, help="Save interval for latest checkpoint (steps)")
parser.add_argument("--milestone_interval", type=int, default=5000, help="Milestone checkpoint interval (steps, ~164M tokens)")
parser.add_argument("--eval_interval", type=int, default=2500, help="Evaluation probe interval (steps, ~82M tokens)")
parser.add_argument("--telemetry_interval", type=int, default=50, help="Telemetry logging interval (steps)")
parser.add_argument("--resume", action="store_true", help="Resume from latest 3B checkpoint if available")
args = parser.parse_args()

SEQ_LEN = args.seq_len
MICRO_BATCH = args.micro_batch
GRAD_ACCUM = args.grad_accum
TOKENS_PER_STEP = MICRO_BATCH * SEQ_LEN * GRAD_ACCUM  # 32,768 tokens/step
TOTAL_STEPS = math.ceil(args.target_tokens / TOKENS_PER_STEP)

CHECKPOINT_DIR = "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/checkpoints"
EXPERIMENTS_DIR = "/home/phil/.gemini/antigravity/scratch/PRIME-Moment-Attention/experiments"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

LATEST_CKPT = os.path.join(CHECKPOINT_DIR, "primelm_50m_3b_latest.pt")
BEST_CKPT = os.path.join(CHECKPOINT_DIR, "primelm_50m_3b_best.pt")
BASE_CKPT = os.path.join(CHECKPOINT_DIR, "primelm_50m_best.pt")

TELEMETRY_PATH = os.path.join(EXPERIMENTS_DIR, "primelm_50m_3b_telemetry.json")
EVAL_HISTORY_PATH = os.path.join(EXPERIMENTS_DIR, "primelm_50m_3b_eval_history.json")

print("=" * 90)
print("     PRIMELM-50M: 3.0 BILLION TOKEN CONTINUOUS PRODUCTION CAMPAIGN")
print(f"     Device: {DEVICE} | Precision: torch.bfloat16 | Target: {args.target_tokens:,} Tokens")
print(f"     Sequence Length: {SEQ_LEN} | Batch Size: {TOKENS_PER_STEP:,} Tokens/Step | Total Steps: {TOTAL_STEPS:,}")
print("=" * 90)

# Initialize Tokenizer & Model
tok = AutoTokenizer.from_pretrained("gpt2")
tok.model_max_length = 1000000

model = PrimeLM50M(
    vocab_size=50257,
    d_model=512,
    n_layers=8,
    n_heads=8,
    head_dim=64,
    d_ff=1024,
    window_size=256,
    num_registers=4,
    use_registers=True,
    use_gating=True,
    use_probe=True
).to(DEVICE).to(torch.bfloat16)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

start_step = 0
tokens_trained = 0
best_loss = float("inf")

# Checkpoint Loading / Resumption
if args.resume and os.path.exists(LATEST_CKPT):
    print(f"[*] Resuming from latest 3B checkpoint: {LATEST_CKPT}")
    checkpoint = torch.load(LATEST_CKPT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_step = checkpoint.get("step", 0)
    tokens_trained = checkpoint.get("tokens_trained", start_step * TOKENS_PER_STEP)
    best_loss = checkpoint.get("best_loss", float("inf"))
    print(f"[*] Resumed successfully at Step {start_step:,} ({tokens_trained / 1e6:.1f}M tokens pushed).")
elif os.path.exists(BASE_CKPT):
    print(f"[*] Initializing from 100M foundation checkpoint: {BASE_CKPT}")
    state = torch.load(BASE_CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    tokens_trained = 98_304_000
    print(f"[*] Successfully loaded foundation weights ({tokens_trained / 1e6:.1f}M tokens previously seen).")
else:
    print("[*] Starting from scratch with initialized weights.")

def get_lr(step):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, TOTAL_STEPS - args.warmup_steps)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))

# Graceful Exit Handler
stop_requested = False
def sigint_handler(signum, frame):
    global stop_requested
    print("\n[!] Signal received! Saving checkpoint and gracefully terminating...")
    stop_requested = True

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)

# Streaming Data Loader
print("[*] Initializing circular multi-corpus streaming data loader...")
streamer = MultiCorpusBillionStreamer(seq_len=SEQ_LEN, buffer_target=100000)

# Quick Capability Probe Suite
PROBE_PROMPTS = [
    ("Solve for x: 5 * x + 15 = 40.\nStep 1:", [5.0, 15.0, 40.0, 5.0]),
    ("Problem: Calculate heat required for 4 kg water heated by 20 K.\nFormula:", [4.0, 4.0, 20.0, 320.0]),
    ("Problem: Calculate kinetic energy of mass 6 kg moving at speed 4 m/s.\nFormula:", [6.0, 4.0, 0.5, 48.0]),
    ("Once upon a time, a little girl named Lily found a shiny key in the garden.", [0.0, 0.0, 0.0, 0.0])
]

def run_periodic_probe(step):
    model.eval()
    results = []
    with torch.no_grad():
        for prompt, inv in PROBE_PROMPTS:
            inputs = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
            inv_t = torch.tensor([inv], dtype=torch.bfloat16, device=DEVICE)
            out_ids = model.generate(inputs, max_new_tokens=35, temperature=0.2, inv_vec=inv_t)
            gen_tokens = out_ids[0, inputs.shape[1]:].tolist()
            gen_text = tok.decode(gen_tokens, skip_special_tokens=True).strip()
            
            # Compute 4-gram repetition rate
            if len(gen_tokens) >= 4:
                ngrams = [tuple(gen_tokens[i:i+4]) for i in range(len(gen_tokens)-3)]
                rep_4 = (len(ngrams) - len(set(ngrams))) / len(ngrams)
            else:
                rep_4 = 0.0
                
            results.append({"prompt": prompt.replace('\n', ' '), "generation": gen_text.replace('\n', ' ↵ '), "rep_4": round(rep_4, 4)})
    model.train()
    return results

print("[*] Starting 3-Billion Token Training Loop...")
t_campaign_start = time.time()
step_times = []
rolling_loss = 0.0
rolling_ce = 0.0
rolling_inv = 0.0

for step in range(start_step, TOTAL_STEPS):
    if stop_requested:
        break
        
    t_step_start = time.time()
    
    # Update Learning Rate
    current_lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
        
    optimizer.zero_grad()
    step_loss = 0.0
    step_ce = 0.0
    step_inv = 0.0
    
    # Gradient Accumulation
    for _ in range(GRAD_ACCUM):
        batch_ids, batch_invs = streamer.get_batch(batch_size=MICRO_BATCH)
        batch_ids = batch_ids.to(DEVICE)
        batch_invs = batch_invs.to(DEVICE)
        
        logits, pred_inv = model(batch_ids, inv_vec=batch_invs)
        
        # Shift tokens for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous().view(-1, 50257)
        shift_labels = batch_ids[:, 1:].contiguous().view(-1)
        
        loss_ce = F.cross_entropy(shift_logits, shift_labels)
        
        # Auxiliary Invariant MSE Loss (bounded via tanh matching model encoder)
        target_inv = torch.tanh(batch_invs / 50.0)
        has_inv = (batch_invs.abs().sum(dim=-1, keepdim=True) > 1e-3).float()
        loss_inv = (F.mse_loss(pred_inv, target_inv, reduction="none") * has_inv).mean()
            
        total_loss = (loss_ce + 0.1 * loss_inv) / GRAD_ACCUM
        total_loss.backward()
        
        step_loss += total_loss.item() * GRAD_ACCUM
        step_ce += (loss_ce.item() / GRAD_ACCUM)
        step_inv += (loss_inv.item() / GRAD_ACCUM)
        
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    step_duration = time.time() - t_step_start
    step_times.append(step_duration)
    if len(step_times) > 100:
        step_times.pop(0)
        
    tokens_trained += TOKENS_PER_STEP
    rolling_loss = 0.9 * rolling_loss + 0.1 * step_loss if rolling_loss > 0 else step_loss
    rolling_ce = 0.9 * rolling_ce + 0.1 * step_ce if rolling_ce > 0 else step_ce
    rolling_inv = 0.9 * rolling_inv + 0.1 * step_inv if rolling_inv > 0 else step_inv
    
    # Live Logging every telemetry_interval
    if step % args.telemetry_interval == 0 or step == start_step:
        avg_step_time = sum(step_times) / len(step_times)
        tok_per_sec = TOKENS_PER_STEP / max(1e-4, avg_step_time)
        remaining_steps = TOTAL_STEPS - step
        eta_hours = (remaining_steps * avg_step_time) / 3600.0
        vram_gb = torch.cuda.memory_allocated(DEVICE) / (1024 ** 3)
        
        print(f"Step {step:6d}/{TOTAL_STEPS:,} | Tokens: {tokens_trained/1e6:7.1f}M ({tokens_trained/args.target_tokens*100:5.2f}%) | "
              f"Loss: {rolling_loss:6.4f} (CE: {rolling_ce:6.4f}, Inv: {rolling_inv:6.4f}) | "
              f"Speed: {tok_per_sec:6.0f} tok/s | LR: {current_lr:.2e} | VRAM: {vram_gb:4.2f}GB | ETA: {eta_hours:5.1f}h")
              
        # Write JSON Telemetry
        telemetry_entry = {
            "step": step,
            "total_steps": TOTAL_STEPS,
            "tokens_trained": tokens_trained,
            "target_tokens": args.target_tokens,
            "percent_complete": round(tokens_trained / args.target_tokens * 100.0, 3),
            "loss": round(rolling_loss, 4),
            "ce_loss": round(rolling_ce, 4),
            "inv_loss": round(rolling_inv, 4),
            "tokens_per_sec": round(tok_per_sec, 1),
            "lr": round(current_lr, 8),
            "vram_gb": round(vram_gb, 2),
            "eta_hours": round(eta_hours, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(TELEMETRY_PATH, "w") as f:
            json.dump(telemetry_entry, f, indent=2)

    # Periodic Capability Probe
    if step > 0 and step % args.eval_interval == 0:
        probe_outputs = run_periodic_probe(step)
        print(f"\n--- [MILESTONE EVALUATION PROBE AT STEP {step:,} ({tokens_trained/1e6:.1f}M TOKENS)] ---")
        for p in probe_outputs:
            print(f"  Prompt: {p['prompt'][:45]}... | Rep-4: {p['rep_4']:.3f} | Out: {p['generation'][:60]}...")
            
        # Append to eval history
        eval_history = []
        if os.path.exists(EVAL_HISTORY_PATH):
            try:
                with open(EVAL_HISTORY_PATH, "r") as f:
                    eval_history = json.load(f)
            except Exception: pass
            
        eval_history.append({
            "step": step,
            "tokens_trained": tokens_trained,
            "loss": round(rolling_loss, 4),
            "probe_results": probe_outputs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        with open(EVAL_HISTORY_PATH, "w") as f:
            json.dump(eval_history, f, indent=2)
        print("--------------------------------------------------------------------------------------\n")

    # Regular Save
    if step > 0 and step % args.save_interval == 0:
        torch.save({
            "step": step,
            "tokens_trained": tokens_trained,
            "best_loss": best_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, LATEST_CKPT)
        
        if rolling_loss < best_loss:
            best_loss = rolling_loss
            torch.save(model.state_dict(), BEST_CKPT)

    # Milestone Checkpoint (~every 164M tokens)
    if step > 0 and step % args.milestone_interval == 0:
        milestone_path = os.path.join(CHECKPOINT_DIR, f"primelm_50m_3b_step_{step:06d}.pt")
        print(f"[*] Saving Milestone Checkpoint: {os.path.basename(milestone_path)}")
        torch.save({
            "step": step,
            "tokens_trained": tokens_trained,
            "loss": rolling_loss,
            "model_state": model.state_dict()
        }, milestone_path)

# Final Save
torch.save({
    "step": step,
    "tokens_trained": tokens_trained,
    "best_loss": best_loss,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict()
}, LATEST_CKPT)
if rolling_loss < best_loss:
    torch.save(model.state_dict(), BEST_CKPT)

print("\n" + "=" * 90)
print(f"  CAMPAIGN COMPLETE OR PAUSED AT STEP {step:,}")
print(f"  Total Tokens Trained: {tokens_trained:,} ({tokens_trained / 1e6:.2f}M)")
print(f"  Final Rolling Loss:   {rolling_loss:.4f}")
print(f"  Latest Checkpoint:    {LATEST_CKPT}")
print("=" * 90)
