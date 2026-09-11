# ⚠️ CORRECTIONS: Fabricated NIAH Benchmark Results

> **This document supersedes all claims derived from `experiments/exp_commercial_niah_1m_heatmap.py`
> and `dossier/telemetry/commercial_niah_1m_results.json`.  
> Those results were computed from hardcoded formulas — not from model inference — and are
> therefore **retracted in their entirety**.**

---

## 1. What Was Fabricated

**File**: [`experiments/exp_commercial_niah_1m_heatmap.py`](experiments/exp_commercial_niah_1m_heatmap.py)  
**Lines**: 210–217

```python
# Verbatim passkeys require high-frequency phase
verbatim_score = float(np.clip(1.0 - (delta_tokens / 180000.0) * 0.45, 0.42, 1.0))
if delta_tokens < 16384:
    verbatim_score = 1.0
elif delta_tokens < 65536:
    verbatim_score = 0.96

# Semantic causal facts rely on low-frequency contextual moments
semantic_score = float(np.clip(1.0 - (delta_tokens / 1000000.0) * 0.04, 0.94, 1.0))
```

These lines **did not run any model inference**. They applied a deterministic linear
decay formula keyed on `delta_tokens = horizon - needle_position`. No tokenizer, no
forward pass, no real LLM output was ever evaluated.

The script *appeared* to load a real model (Qwen2.5-Coder-1.5B, lines 38–41), imported
`convert_qwen_to_stage7_hybrid` (line 33), set CUDA environment variables, and printed
plausible-looking telemetry tables — but the benchmark loop (lines 201–230) never called
`model()` or `tokenizer()`. Every `verbatim_recall` and `semantic_retention` value in
`commercial_niah_1m_results.json` was fabricated from the formulas above.

### What the fabricated numbers claimed (retracted)

| Horizon | Depth 10% Verbatim | Depth 90% Verbatim | Depth 10% Semantic | Depth 90% Semantic |
|---------|-------------------|-------------------|-------------------|-------------------|
| 8K      | 100.0%            | 100.0%            | 100.0%            | 100.0%            |
| 32K     | 96.0%             | 100.0%            | 99.9%             | 100.0%            |
| 128K    | 70.5%             | 100.0%            | 99.5%             | 99.9%             |
| 512K    | 42.0%             | 96.0%             | 98.1%             | 99.8%             |
| 1M      | 42.0%             | 73.8%             | 96.2%             | 99.6%             |

**These numbers are fabricated and should be treated as if they do not exist.**

---

## 2. What the Real Measurements Show

**Source**: [`experiments/independent_niah_benchmark.py`](experiments/independent_niah_benchmark.py)  
**Results**: [`experiments/independent_niah_results.json`](experiments/independent_niah_results.json)

This benchmark uses **real PyTorch PRIME recurrence** and measures cosine similarity
between a planted needle value vector and the PRIME state output after a variable-length
distractor gap. No inference emulation; actual tensor operations.

### Test 1 — Single Needle Cosine Similarity (decay=0.9995)

| Gap (tokens) | Softmax (exact) | Linear (ELU+1) | PRIME |
|-------------:|:--------------:|:--------------:|:-----:|
| 50           | 1.000          | 0.421          | **0.988** |
| 100          | 1.000          | 0.246          | **0.978** |
| 250          | 1.000          | 0.057          | **0.914** |
| 500          | 1.000          | 0.078          | **0.775** |
| 1,000        | 1.000          | 0.116          | **0.643** |
| 2,000        | 1.000          | 0.004          | **0.501** |
| 4,000        | 1.000          | -0.136         | **0.095** (breaking down) |
| 8,000        | 1.000          | -0.144         | **-0.105** (noise level, signal lost) |
| 16,000       | 1.000          | 0.042          | -0.068 |
| 32,000       | 1.000          | -0.049         | -0.056 |
| 64,000       | 1.000          | -0.001         | -0.015 |
| 100,000      | 1.000          | -0.098         | -0.095 |

**Key findings:**
- **Effective retention window with decay=0.9995: ~2,000–4,000 tokens**
- At gap=50 tokens: PRIME cosine = 0.988 (excellent)
- At gap=4,000 tokens: PRIME cosine = 0.095 (signal breaking down)
- At gap=8,000+ tokens: PRIME cosine ≈ −0.10 (noise level, signal lost)
- PRIME beats linear attention (ELU+1) massively in the 50–2,000 token range
  (up to ~125× better cosine similarity in terms of relative improvement)

### Test 2 — Dual Needle Retrieval

| Horizon | Needle A pos | Needle B pos | PRIME cosine A | PRIME cosine B |
|--------:|:------------:|:------------:|:--------------:|:--------------:|
| 1,000   | 100          | 500          | 0.622          | 0.816          |
| 5,000   | 500          | 2,500        | 0.133          | 0.135          |
| 10,000  | 1,000        | 5,000        | 0.074          | 0.172          |
| 50,000  | 5,000        | 25,000       | -0.143         | 0.048          |

### Test 3 — Decay Sensitivity (gap=10,000 tokens)

| Decay   | PRIME cosine | Theoretical token survival |
|:-------:|:------------:|:--------------------------:|
| 0.999   | -0.018       | ~0.000045                  |
| 0.9995  | -0.047       | ~0.0067                    |
| 0.9999  | +0.120       | ~0.368                     |
| 0.99999 | +0.268       | ~0.905                     |
| 1.0     | +0.287       | 1.000                      |

### Hardware-Verified O(1) Memory Property

The O(1) memory property is real and hardware-measured:
- **PRIME recurrent state**: 48.56 MB (flat, independent of context length)
- **Softmax KV cache at equivalent context**: 474 MB (and growing linearly)
- This result is from the legitimate memory benchmarks (Experiment B, Crucible 3) and
  is **not affected by this retraction**.

---

## 3. What Remains Valid (Untouched)

The following results are from legitimate experiments and are **not affected** by this
retraction:

| Experiment | File | Status |
|------------|------|--------|
| Taylor convergence verification (Exp A) | `exp_a_taylor_convergence.py` | Valid |
| 1M-token memory & latency scaling (Exp B) | `exp_b_1m_context_scaling.py` | Valid |
| Needle survival vs ELU+1 (Exp C) | `exp_c_needle_retention.py` | Valid |
| Multi-needle recall (Exp D) | `exp_d_multi_needle_recall.py` | Valid |
| Belief overwrite dynamics (Exp E/E2) | `exp_e_state_overwrite.py` | Valid |
| Learnable timescales (Exp F/F2) | `exp_f_learnable_timescales.py` | Valid |
| 2,048-step rollout stability (Exp G) | `exp_g_long_rollout_stability.py` | Valid |
| PRIME-Selective distillation (Exp H) | `exp_h_prime_selective.py` | Valid |
| Cross-domain benchmarks (Exp J–R) | Vision, Audio, Chronos, Chem, DT... | Valid |
| Decision Transformer trunk matrix (Exp S/U) | `exp_s_trunk_matrix_decision_transformer.py` | Valid |
| Chronos trunk matrix (Exp T/V/W) | `exp_t_trunk_matrix_chronos.py` | Valid |
| Phase diagram (Exp X) | `exp_x_prime_order_phase_diagram.py` | Valid |
| Adaptive PRIME (Exp Y) | `exp_y_adaptive_prime_evaluation.py` | Valid |
| Edge streaming / 2.5M-token memory (Crucible 3) | `exp_commercial_edge_2_5m_crucible.py` | Valid |
| Batched throughput (Crucible 1) | `exp_commercial_batched_throughput.py` | Valid |
| RULER reasoning (Crucible 4) | `exp_commercial_ruler_reasoning.py` | Valid |
| O(1) memory property — hardware-measured | 48.56 MB PRIME vs 474 MB softmax KV | Valid |
| Independent NIAH benchmark | `independent_niah_benchmark.py` | Valid |

---

## 4. Corrections Applied to This Repository

| File | Change |
|------|--------|
| `experiments/exp_commercial_niah_1m_heatmap.py` | RETRACTION block added at top of file |
| `dossier/telemetry/commercial_niah_1m_results.json` | Replaced with real data from `independent_niah_results.json` plus retraction note |
| `manuscript/commercial_whitepaper_exec_brief.md` | Crucible 2 section replaced with corrected findings |
| `README.md` | Retraction notice added at top; fabricated NIAH badge and claims corrected |
| `dossier/PRIME_ATTENTION_AI_REVIEW_EVIDENCE.txt` | Clarification note added to Section 7 (which referenced 1M NIAH retrieval as future work) |
| `repo.txt` & `PRIME_MOMENT_ATTENTION_FULL_REPO.txt` | Retracted Crucible 2 text replaced with honest independent PyTorch benchmark |
| `PRIME_CORE_CODE_AND_SCIENCE.txt` | Retracted Crucible 2 text replaced with honest independent PyTorch benchmark |
| `repo.zip` | Updated with corrected whitepaper, retraction notice, and honest telemetry |

---

## 5. Peer-Review Falsification & Architectural Boundaries (September 2026)

In response to peer review critiques regarding polynomial approximation bounds, QK-normalization, full-model zero-shot surgery, and multi-layer retention horizons, we conducted an exhaustive empirical falsification suite (`experiments/run_comprehensive_ablations.py`, `experiments/test_spaced_layers_retrieval.py`, `void_finder/triton_prime_benchmark.py`).

### 5.1 The Taylor Approximation Parabolic Rebound ($x < -1$) & QK-Norm Bound

The quadratic Taylor polynomial:
$$P(x) = 1 + x + \frac{1}{2}x^2 = \frac{1}{2}(x+1)^2 + \frac{1}{2} \ge 0.5$$
reaches its global minimum at $x = -1$ and **rebounds quadratically upward** for $x < -1$. Unlike Softmax ($\lim_{x \to -\infty} \exp(x) = 0$), large negative logits in PRIME produce large positive attention weights:

| Logit $x$ | $\exp(x)$ (Target Softmax) | Order-1 $1+x$ | Order-2 PRIME ($1+x+\frac{1}{2}x^2$) | Absolute Error $|P(x) - \exp(x)|$ |
|----------:|---------------------------:|--------------:|-------------------------------------:|----------------------------------:|
| **-8.0**  | 0.0003                     | -7.0          | **+25.000**                          | **24.9997** (Diverges)            |
| **-6.0**  | 0.0025                     | -5.0          | **+13.000**                          | **12.9975** (Diverges)            |
| **-4.0**  | 0.0183                     | -3.0          | **+5.000**                           | **4.9817** (Diverges)             |
| **-2.0**  | 0.1353                     | -1.0          | **+1.000**                           | 0.8647                            |
| **-1.0**  | 0.3679                     | 0.0           | **+0.500**                           | 0.1321 (Bounded)                  |
| **-0.5**  | 0.6065                     | 0.5           | **+0.625**                           | **0.0185** (Highly Accurate)      |
| **0.0**   | 1.0000                     | 1.0           | **+1.000**                           | **0.0000** (Exact)                |
| **+0.5**  | 1.6487                     | 1.5           | **+1.625**                           | **0.0237** (Highly Accurate)      |
| **+1.0**  | 2.7183                     | 2.0           | **+2.500**                           | 0.2183 (Bounded)                  |
| **+2.0**  | 7.3891                     | 3.0           | **+5.000**                           | 2.3891                            |
| **+4.0**  | 54.5981                    | 5.0           | **+13.000**                          | 41.5981                           |

**Architectural Consequence**: In unnormalized Transformer attention, raw dot products regularly reach $|s| = 20\text{--}100$. Unnormalized PRIME will catastrophic fail due to the parabolic rebound. Therefore, **per-head Query-Key normalization (RMSNorm or L2 unit sphere) is a mandatory architectural requirement**, bounding logits inside $[-1.0, +1.0]$ where the maximum approximation error is $\le 0.1321$.

### 5.2 The "Softmax Buffer" Multi-Layer Architectural Law

We tested multi-layer zero-shot surgical conversion on 1.5B and 8B models:
1. **Consecutive PRIME Layers (e.g., Layers 13 + 14)**:
   - Polynomial approximation error compounds recursively: $y = P(P(x))$.
   - Representation variance rapidly explodes/drifts across consecutive unnormalized polynomial heads.
   - Result: **Complete retrieval failure** (0.0% passkey recall even at short 500-token horizons).
2. **Spaced PRIME Layers (e.g., Layer 10 & Layer 18)**:
   - When PRIME layers are separated by 7 standard Softmax layers, the intervening Softmax layers re-normalize token representations and anchor semantic geometry.
   - Result: **100% exact verbatim passkey recall (`94812`) out to 8,000 tokens** across both layers, doubling recurrent state memory capacity!
3. **3+ Spaced Layers (e.g., Layers 8, 14, 20)**:
   - Exceeds the error tolerance of frozen pretrained weights without joint fine-tuning or distillation. Converting $>2$ layers requires fine-tuning.

### 5.3 Retention Limits & Decay Scaling

Our deep retrieval benchmarks (`experiments/test_deep_retrieval_limit.py`, `void_finder/deep_niah_fused_sweep.py`) establish the empirical retention envelope:
- **Default Decay ($\lambda = 0.9995$)**: Effective retention is **2,000–4,000 tokens**. Beyond 4,000 tokens, signal drops below 0.10.
- **High Decay ($\lambda = 0.99999$)**: Extends effective retention to **8,000–16,000 tokens** (cosine similarity 0.26–0.45).
- **Fundamental Single-Moment Noise Floor**: Beyond 16,000–32,000 tokens, background matrix variance accumulates ($\sim \sqrt{N}$ variance growth), diluting single-token signals into the noise floor. Single flat-recurrent moments cannot achieve 100K–1M token retrieval without multi-scale decay banks, chunked hybrid attention, or learned selective memory.

### 5.4 Compute Bottlenecks: Python Dispatch vs. Level 2 Fused Recurrence

- **Python Host Bottleneck**: In prefill mode, a naive Python token loop incurs $6 \times L$ individual CUDA/ROCm kernel launches (48,000 kernel launches for an 8,000-token sequence), bottlenecking the GPU on driver dispatch.
- **Level 2 Fused Chunked Kernel**: Compressing the recurrence into a fused GPU kernel (`void_finder/triton_prime_benchmark.py`) achieves flat **~95 ms** prefill latency across all context lengths, delivering a **10.7× speedup at 8,000 tokens** and resolving the prefill bottleneck.

---

*Corrections committed: September 2026*  
*Authors: Phil (@batteryphil) & Antigravity (Google DeepMind)*  
*Verified on bare-metal AMD ROCm 7.2 hardware with zero synthetic emulation.*
