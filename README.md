<div align="center">

# PRIME Moment Attention
### Constant-State Second-Order Recurrent Attention for Long-Context Transformer Inference

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg?style=flat&logo=huggingface)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Context Scaling](https://img.shields.io/badge/Context-1%2C048%2C576%20Tokens-success.svg)](#1-million-token-scaling-benchmark)
[![Attention Memory](https://img.shields.io/badge/Attention%20State-O(1)%20Flat-9cf.svg)](#key-empirical-results)
[![Speedup](https://img.shields.io/badge/Decode%20Speedup-55.9x%20%40%201M-brightgreen.svg)](#1-million-token-scaling-benchmark)

*Empirical evaluation of context scaling, information retention, and pretrained-model surgical replacement.*

</div>

---

## 🔬 Executive Summary

Standard autoregressive Softmax Attention requires storing every historical key-value pair $\mathcal{H}_t = \{(k_1, v_1), \dots, (k_t, v_t)\}$, creating an attention state footprint and per-token decoding cost that scale monotonically with sequence length $L$.

**PRIME Moment Attention** converts the context-dependent historical storage and repeated scanning of conventional autoregressive attention into a **bounded recurrent moment state** whose measured memory footprint and decoding cost remain independent of sequence length $L$ (scaling as $\mathcal{O}(D^2)$ with respect to head dimension $D$), while retaining substantially more injected signal than tested first-order linear attention baselines.

```
Conventional Softmax Attention:
Token t ───► Scan all t past keys/values ───► Compute QK^T [t tokens] ───► Latency scales O(t) ───► Memory O(t)

PRIME Moment Attention:
Token t ───► Recurrent update (S0, S1, S2, K0, K1, K2) ───► Tensor readout ───► Latency O(1) ───► Memory O(1)
```

---

## 📊 Key Empirical Results

### 1. 1-Million Token Scaling Benchmark (Experiment B)
Evaluated on `Qwen2.5-0.5B` architecture (24 layers, 14 query heads, 2 KV heads, $D=64$) on GPU using `float32`/`bfloat16` state accumulators:

| Context ($L$) | Softmax KV Cache | PRIME Recurrent State | Softmax Step | PRIME Step | Measured Speedup | Attention State Savings |
|--------------:|----------------:|---------------------:|-------------:|-----------:|-----------------:|------------------------:|
| 128 | 1.50 MB | **1.54 MB** | 0.88 ms | 8.31 ms | 0.1× | Baseline |
| 1,024 | 12.00 MB | **1.54 MB** | 1.28 ms | 8.44 ms | 0.2× | 87.2% |
| 4,096 | 48.00 MB | **1.54 MB** | 2.65 ms | 8.69 ms | 0.3× | 96.8% |
| 16,384 | 192.00 MB | **1.54 MB** | 8.13 ms | 8.57 ms | 0.9× | 99.2% |
| 32,768 | 384.00 MB | **1.54 MB** | 15.45 ms | 8.61 ms | 1.8× | 99.6% |
| 65,536 | 768.00 MB | **1.54 MB** | 30.08 ms | 8.59 ms | 3.5× | 99.8% |
| 131,072 | 1.50 GB | **1.54 MB** | 59.22 ms | 8.57 ms | 6.9× | 99.90% |
| 262,144 | 3.00 GB | **1.54 MB** | 117.62 ms | 8.86 ms | 13.3× | 99.95% |
| 524,288 | 6.00 GB | **1.54 MB** | 234.42 ms | 8.33 ms | 28.1× | 99.97% |
| **1,048,576 (1M)** | **12.00 GB** | **1.54 MB** | **468.02 ms** | **8.37 ms** | **55.9×** | **99.987%** |

* **Flat Decoding Latency**: PRIME decode step time is flat (**8.31 ms to 8.37 ms**), exhibiting zero context degradation from 128 to 1,048,576 tokens.
* **Attention State Footprint**: Reduced by **99.987%** at 1M tokens (1.54 MB constant vs 12.0 GB).
* **Precision Boundary**: Discovered that IEEE 754 `float16` overflows at 65,504 tokens; accumulators must use `bfloat16` or `float32`.

---

### 2. Mathematical Taylor Convergence (Experiment A)
Evaluated exact Softmax vs Full 2nd-Order Taylor vs PRIME Diagonal Moment Attention across varying query-key interaction scale $s = \frac{\max |q^T k|}{\sqrt{D}}$ with $N = 100$ tokens:

| Head Dimension ($D$) | Theoretical Expansion Order | Empirical Convergence Slope | Direct vs Recurrent Contraction Error |
|:---:|:---:|:---:|:---:|
| **$D = 2$** | $\mathcal{O}(s^3)$ | **3.98** | $< 10^{-12}$ |
| **$D = 4$** | $\mathcal{O}(s^3)$ | **3.09** | $< 10^{-12}$ |
| **$D = 8$** | $\mathcal{O}(s^3)$ | **3.16** | $< 10^{-12}$ |

* Confirms theoretical $\mathcal{O}(s^3)$ residual behavior.
* When $|s| \le 0.5$, diagonal approximation error is negligible ($\sim 10^{-3}$ to $10^{-4}$).

---

### 3. Information Survival vs Linear Baseline (Experiment C)
Signal retention cosine similarity $\cos(h_{\text{stored}}, h_{\text{target}})$ across noise distractor tokens:

| Distractor Gap (tokens) | Tested ELU+1 Linear Attention | PRIME Moment Attention |
|------------------------:|-----------------------------:|-----------------------:|
| 50 | 0.4995 | **0.9968** |
| 100 | 0.3925 | **0.9928** |
| 250 | 0.4445 | **0.9901** |
| 500 | 0.3273 | **0.9451** |
| 1,000 | -0.1039 | **0.6973** |
| 2,000 | 0.0761 | **0.5503** |

* The tested unweighted ELU+1 baseline suffers severe representation dilution beyond ~500 tokens, while PRIME's second-order state preserves signal substantially longer.

---

### 4. Memory Plasticity & Belief Overwrite Dynamics (Experiment E)
Testing belief revision on identical query keys across 500 noise tokens:

| Fact B Exposures | Cosine to Fact A | Cosine to Fact B | Margin $\Delta (B - A)$ | Active Belief State |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.8086 | 0.7578 | -0.0508 | Fact A (Historical Retention) |
| **2** | 0.6445 | **0.9023** | **+0.2578** | **Fact B (Belief Flipped)** |
| 3 | 0.5078 | 0.9570 | +0.4492 | Fact B (Reinforced) |
| 10 | 0.1807 | 0.9922 | +0.8115 | Fact B (99.2% Overwrite) |

---

### 5. Differentiable Multiscale Timescales (Experiment F)
Parameterized decay rates as $\lambda_h = \sigma(\theta_h)$ across a logarithmic temporal filter bank:
* Initial filter bank spans half-lives from $t_{1/2} = 1.0$ token (local syntax) to $t_{1/2} = 692.8$ tokens (document memory).
* Gradient backpropagation $\frac{\partial \mathcal{L}}{\partial \theta_h}$ flows cleanly without exploding or vanishing states (gradient norm: $14.6 \to 0.03$), demonstrating end-to-end optimization of the temporal basis.

---

## 📐 Mathematical Formulation

### 1. Second-Order Taylor Expansion
Causal softmax attention expands around $s = \frac{q^T k}{\sqrt{d}} = 0$:

$$\exp(s) = 1 + s + \frac{1}{2} s^2 + \mathcal{O}(s^3)$$

The second-order quadratic term expands as:
$$s^2 = \left( \frac{q^T k}{\sqrt{d}} \right)^2 = \frac{1}{d} q^T (k k^T) q$$

Yielding the normalized attention output:
$$y(q) \approx \frac{S_0 + \frac{1}{\sqrt{d}} q^T S_1 + \frac{1}{2d} q^T S_2 q}{K_0 + \frac{1}{\sqrt{d}} q^T K_1 + \frac{1}{2d} q^T K_2 q}$$

### 2. The Diagonal Moment Approximation
The full outer product $S_2 = \sum_j (k_j \otimes k_j) \otimes v_j$ forms a rank-3 tensor requiring $\mathcal{O}(D^3)$ state size ($262,144$ elements per head for $D=64$).

To maintain $\mathcal{O}(D^2)$ parameter compactness matching $S_1$, PRIME employs the **diagonal second-order approximation**:
$$k_j k_j^T \approx \operatorname{diag}(k_j^2)$$

$$\boxed{S_2 = \sum_{j=1}^t (k_j^2) v_j^T \in \mathbb{R}^{D \times D}, \quad K_2 = \sum_{j=1}^t k_j^2 \in \mathbb{R}^D}$$

This reduces 2nd-order parameter storage by **64×** while capturing quadratic magnitude curvature.

---

## 🚀 Quickstart

### Installation
```bash
git clone https://github.com/batteryphil/PRIME-Moment-Attention.git
cd PRIME-Moment-Attention
pip install -r requirements.txt
```

### Basic Usage
```python
import torch
from src.prime_moment_attention import PrimeMomentAttention

# Initialize module (hidden_size=896, 14 heads, dim=64)
attn = PrimeMomentAttention(
    hidden_size=896,
    num_heads=14,
    head_dim=64,
    decay=0.9995,
    use_qk_norm=True
)

# Autoregressive generation with O(1) state
x_token = torch.randn(1, 1, 896)
state = None

for step in range(100):
    out, state = attn(x_token, state=state, return_state=True)
    # state footprint remains 100% constant!
```

### Pretrained Model Surgery (Qwen2.5 / LLaMA)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.prime_moment_attention import convert_transformer_to_prime, PrimeMomentCache

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16)

# Transplant PRIME into 50% or 100% of attention layers
model, converted_layers = convert_transformer_to_prime(model, hybrid_ratio=1.0, decay=0.9995)

# Deploy with O(1) PrimeMomentCache
cache = PrimeMomentCache()
```

---

## 🔬 Reproducibility

Every experiment presented in the paper can be executed via the unified reproduction CLI:

```bash
# Run the complete test suite
python reproduce_all.py --all

# Or run individual experiments:
python reproduce_all.py --exp a   # Taylor convergence order verification (D in {2, 4, 8})
python reproduce_all.py --exp b   # 1-Million token context latency & memory benchmark
python reproduce_all.py --exp c   # Needle survival dynamics vs ELU+1 linear attention
python reproduce_all.py --exp d   # Multi-needle associative recall across 2,048 tokens
python reproduce_all.py --exp e   # Belief overwrite and interference dynamics
python reproduce_all.py --exp f   # Learnable timescales via backpropagation
python reproduce_all.py --exp g   # 2,048-token generation rollout stability
```

---

## 📁 Repository Structure

```
PRIME-Moment-Attention/
├── dossier/
│   └── PRIME_ATTENTION_AI_REVIEW_EVIDENCE.txt   # Complete scientific dossier
├── experiments/
│   ├── exp_a_taylor_convergence.py             # Exp A: Taylor convergence
│   ├── exp_b_1m_context_scaling.py             # Exp B: 1M token benchmark
│   ├── exp_c_needle_retention.py               # Exp C: Needle survival vs ELU+1
│   ├── exp_d_multi_needle_recall.py            # Exp D: Multi-needle recall
│   ├── exp_e_state_overwrite.py                # Exp E: State overwrite dynamics
│   ├── exp_f_learnable_timescales.py           # Exp F: Learnable filter bank
│   └── exp_g_long_rollout_stability.py         # Exp G: Rollout norm tracking
├── src/
│   └── prime_moment_attention/
│       ├── __init__.py
│       ├── attention.py                        # Core PrimeMomentAttention layer
│       ├── cache.py                            # O(1) PrimeMomentCache for HF
│       ├── surgery.py                          # In-place model surgery
│       └── timescales.py                       # Differentiable filter bank
├── reproduce_all.py                            # Unified reproduction runner
├── requirements.txt
├── LICENSE                                     # Apache 2.0
└── README.md
```

---

## 📄 Citation

```bibtex
@article{prime_moment_attention_2026,
  title   = {PRIME Moment Attention: Constant-State Second-Order Recurrent Attention for Long-Context Transformer Inference},
  author  = {PRIME-Net Research Team},
  journal = {arXiv preprint},
  year    = {2026}
}
```
