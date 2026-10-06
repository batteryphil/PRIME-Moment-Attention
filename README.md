# 🧠 PRIME Moment Attention

**PRIME Moment Attention** is an experimental architecture exploring constant-state second-order recurrent attention and ultra-compressed "Gist" cognitive memory maps for long-context language modeling.

## ⚠️ Repository Status: Honest Baseline
This repository contains the pure, untrained PyTorch modules for PRIME Attention and Gist Memory. **There are no pretrained checkpoints, no benchmarks, and no drop-in replacements for DeepSeek.** Any scripts that previously claimed to achieve 60% accuracy on reasoning tasks by "transplanting" these layers into a pretrained model without training were using a **zero-initialization bypass** to cheat the benchmark (they just passed the original DeepSeek attention outputs through).

This codebase is for **training and experimenting** with these architectures from scratch or via rigorous fine-tuning.

## 📦 What's Here

### 1. PRIME Moment Attention
Located in `src/prime_moment_attention/attention.py`, this is a recurrent linear attention mechanism that uses a theoretically sound 2nd-order Taylor polynomial over the `elu(x) + 1` feature map. It compresses context into a fixed-size quadratic memory state $S$ instead of a linearly growing KV cache.

**Proof of Concept:**
`test_deepseek_prime.py` demonstrates how to hybridize a pre-trained `DeepSeek-1.5B` model by swapping its standard Softmax attention with a PRIME-hybrid attention layer, and proves that the PRIME gates successfully learn and propagate gradients. This is a training harness, not a pretrained model.

### 2. Gist Memory (Generative Thought Reconstruction Map)
Located in `src/prime_moment_attention/thought_reconstruction_map.py`, the **Gist** mechanism distills high-dimensional hidden states into a massive mathematical bottleneck (e.g. `d_map = 32`). This compresses the sequence context into a tiny, fixed $O(1)$ memory blueprint (~65 KB per layer).

**Proof of Concept:**
`test_deepseek_gist.py` snaps the Gist module onto DeepSeek-1.5B as a parallel cognitive stream, demonstrating a backward pass that trains the Gist layer to memorize and reconstruct context over sequence horizons.

### 3. Chunked State Space Duality (SSD)
Located in `src/prime_moment_attention/chunked_ssd.py`, this implements the parallel prefix-sum hardware-efficient chunked training algorithm for the PRIME linear attention recurrence.

## 🚀 Quickstart

Run the DeepSeek-1.5B Gist integration proof-of-concept (Gradient Test):
```bash
python test_deepseek_gist.py
```

Run the DeepSeek-1.5B PRIME attention wrapper proof-of-concept (Gradient Test):
```bash
python test_deepseek_prime.py
```
