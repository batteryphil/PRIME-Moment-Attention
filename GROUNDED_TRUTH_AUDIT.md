# 🔬 GROUNDED TRUTH AUDIT: PRIME-Moment-Attention Retrospective

**Date**: September 28, 2026  
**Status**: Autonomous Experiment Stopped & VRAM Reclaimed (15.58 GB → 158 MB)  
**Final Cycle Logged**: Cycle 955 (954 synthesized theories, 209 Socratic inquiries)  
**Objective**: Catalog all verified technical assets ("The Good"), diagnose all compromised components and misleading metrics ("The Bad"), and establish an uncompromising blueprint for rebuilding the repository with pure empirical truth.

---

## 🛑 Experiment Termination Snapshot

Both background processes were safely terminated on September 28, 2026:
* **Teacher Daemon (`falcon_autonomous_researcher.py`, PID 1811575)**: Stopped at **Cycle 955**.
  * Final topic: *"DeepSeek-R1 Cognitive Mechanics: Reasoning-Time Compute & Latent Policy Search"*.
  * Database state: 954 entries in `research_vault.sqlite3`, 90,529 lines in `THEORY_DOSSIER.md`.
* **Student Co-Learner (`prime_co_learner_daemon.py`, PID 1805757)**: Stopped at **Cycle 954**.
  * Model: `PRIME-125M-Reasoning` (`export_hf/prime-125m-reasoning`).
  * Final Socratic inquiries: 203 answered, 6 pending.
  * Final evaluation: Loss = **3.9573**, PPL = **52.32**, Anchor Drift = **0.00049**.
* **Hardware Recovery**:
  * GPU VRAM dropped from **15.58 GB (97.9%) → 158 MB (Idle)**.
  * System disk: **798 GB used / 916 GB total (92% full, 72 GB headroom remaining)**.

---

## 🟢 PART 1: THE GOOD (What is Proven, Working, and Worth Keeping)

These are the genuine scientific and systems achievements of the project. They should form the foundation of the clean rebuild.

### 1. The Mathematical 2nd-Order Recurrent Formulation
* **The Formulation**: Truncating the exponential kernel to 2nd-order Taylor:
  $$\exp\left(\frac{q^\top k}{\sqrt{D}}\right) \approx 1 + \frac{q^\top k}{\sqrt{D}} + \frac{1}{2}\left(\frac{q^\top k}{\sqrt{D}}\right)^2$$
  factorizes attention into query-independent moment accumulators:
  * Zeroth moment: $S_t^{(0)} = \lambda S_{t-1}^{(0)} + v_t \in \mathbb{R}^D$
  * First moment: $S_t^{(1)} = \lambda S_{t-1}^{(1)} + k_t v_t^\top \in \mathbb{R}^{D \times D}$
  * Second moment: $S_t^{(2)} = \lambda S_{t-1}^{(2)} + (k_t \odot k_t) v_t^\top \in \mathbb{R}^{D \times D}$
  * Partition normalizers: $K_t^{(0)} \in \mathbb{R}^1, K_t^{(1)} \in \mathbb{R}^D, K_t^{(2)} \in \mathbb{R}^D$
* **Why it matters**: It provides a mathematically legitimate polynomial bridge between 1st-order linear attention (Katharopoulos et al.) and full softmax attention.
* **Status**: Fully verified; gradient checks pass in `tests/test_attention.py`.

### 2. Strict $\mathcal{O}(1)$ Attention State Memory & Constant Decode Latency
* **Memory Invariance**: For $H=8, D=64$, the state memory is strictly:
  $$\text{Bytes} = H \times (2D^2 + 3D + 1) \times 4\text{ bytes} = 268,320\text{ bytes} \approx \mathbf{262.03\text{ KB}}$$
* **Latency Invariance**: Measured empirically on CPU:
  * Step 100: **155.1 µs / token**
  * Step 1,000: **155.4 µs / token**
  * Step 10,000: **155.8 µs / token**
* Unlike softmax attention where decoding latency grows monotonically with context length $L$, PRIME's step latency is completely flat.
* **Status**: Hardware-verified on both CPU and GPU.

### 3. The Pure C99 Native Engine & Cosmopolitan APE (`c/`)
* **Zero External Dependencies**: Standard C library (`libc` + `-lm`) only. No LibTorch, no CUDA, no Python runtime.
* **Universal Binary**: Builds via `cosmocc` into `bin/prime.com`, an Actually Portable Executable (APE) that runs natively on Linux (x86_64/aarch64), macOS, and Windows.
* **Zero-Copy Memory-Mapped Weights**: Page-aligned 4096-byte ZIP payload bundling (`zipalign.c`) allowing weights to be `mmap()`-ed in 0 ms without heap allocation.
* **CPU Throughput**: Sustained **~6,430 tokens/sec** on standard single-threaded host CPU during decoding benchmarks.
* **Embedded POSIX HTTP Server**: Minimal zero-dependency socket server handling inference requests without Python/Uvicorn.
* **Status**: Production-grade systems engineering. Keep 100%.

### 4. Hybrid Window + Recurrence Architecture (`hybrid.py`)
* **Mechanism**: Maintains a bounded local sliding-window softmax cache ($W=512$) for sharp recent syntax, while transferring evicted tokens into the 2nd-order recurrent moment state.
* **Empirical Advantage**:
  * Pure sliding-window suffers catastrophic amnesia (drops needles the moment they exit the window).
  * Pure recurrence has softer resolution for immediate local grammatical parsing.
  * **Hybrid Window-PRIME eliminates sliding-window amnesia** while preserving exact local syntax, delivering a 1.39× speedup over pure recurrence.
* **Status**: Scientifically grounded and aligns with modern production hybrid architectures (e.g., Jamba, RecurrentGemma, Griffin).

### 5. Native Pretraining From Scratch (`PrimeForCausalLM` 125M)
* **Pretraining Trajectory**: Trained on streaming `roneneldan/TinyStories` on ROCm GPU:
  * Step 1: Loss 10.91 (PPL ~54,000)
  * Step 500: Loss 5.74 (PPL ~311)
  * Step 5,000: Loss 4.35 (PPL ~77)
  * Step 10,000: Loss 3.64 (PPL ~38)
* **Real Verification**: Generates grammatically coherent children's stories with full sentence syntax using strictly constant memory.
* **Status**: Real, empirical training run. Checkpoints exist and verify end-to-end convergence.

### 6. Chunked SSD Parallel Prefill Scan (`chunked_ssd.py`)
* Eliminates the Python loop prefill bottleneck by chunking the sequence into blocks of size $C=64$ and computing intra-chunk attention via batch GEMMs, with inter-chunk state passed via scan operations.
* **Status**: Mathematically verified parity with recurrent decoding (`tests/test_chunked_ssd_parity.py`).

---

## 🔴 PART 2: THE BAD (What is Flawed, Misleading, or Compromised)

These items must be purged, corrected, or fundamentally redesigned.

### 1. Fabricated Benchmarks in Earlier Whitepapers (`CORRECTIONS.md`)
* **What Happened**: 
  * `experiments/exp_commercial_niah_1m_heatmap.py` used `np.clip(1.0 - (delta / 180000.0) * 0.45, 0.42, 1.0)` to invent retrieval scores without running inference.
  * `experiments/exp_commercial_ruler_reasoning.py` returned mock dictionary floats.
  * Crucible 3 (2.5M streaming) was an analytical extrapolation presented as live streaming.
* **Impact**: Destroys credibility if cited. While documented in `CORRECTIONS.md`, all remnants of these scripts and any unverified tables must be purged from the new codebase.

### 2. Gamified Evaluation Metrics (`student_intelligence_metrics.json`)
* **The Flaw**: The "Imitation Intelligence Index" formula in `prime_co_learner_daemon.py`:
  $$\text{Score} = \min(100, (\text{think\_tag\_rate} \times 0.4) + (\text{keyword\_matches} \times 5.0) + ((10 - \text{loss}) \times 6.0))$$
* **The Reality**: The 125M student scored **96.3 / 100**, but its actual generation on `Solve for x: 4 * x - 8 = 24` was:
  `" Sol for x - 9 - x - 10 - - 8 - - 12 - - 8 - - 7 - - 7 - 1. Solve by dozen. The ..."`
* **Verdict**: The metric rewarded the model for outputting `<think>` tags and words like *"first"*, *"step"*, and *"since"*, creating the illusion of reasoning while generating token gibberish.

### 3. Hardcoded "Reasoning" & "Deduction" in C (`c/prime_net.c`)
* **The Flaw**: `prime_net.c` claimed a "15-Domain Math Benchmark" and "7 Cognitive & Commonsense Deductions" running in native C.
* **The Reality**: Lines 720–845 were hardcoded `if (strcasestr(...))` string checks:
  * `bowling ball + cake` → hardcoded return `"crushed and flattened"`.
  * `ice cream + summer` → hardcoded return `"melted into liquid soup"`.
  * `cookie jar + drawer` → hardcoded return `"red cookie jar"`.
  * `glork + flurb` → hardcoded return `"Yes, Bob can bounce"`.
* **Verdict**: This is classical ELIZA-style string matching masquerading as cognitive invariant discovery. Must be completely removed.

### 4. Conceptual & Jargon Inflation ("Lore Overdrive")
* Throughout the repository, standard linear algebraic operations were rebranded with extreme physics jargon:
  * Simple state subtraction $S_{t-1} = (S_t - k_t v_t^\top)/\lambda$ became *"Unitary Information Conservation & Adjoint Spin-Echo Memory Recovery via the No-Hiding Theorem"*.
  * Gating on distance from mean key vector became *"Bernoulli Dynamic Pressure Drop & Venturi Entrainment"*.
  * Matrix multiplication with unitary matrices was branded as *"Non-Abelian SU(2) Lie Group Quaternionic Memory"*.
* **Verdict**: Conflating neural network tensor updates with quantum mechanics and fluid dynamics invites immediate academic dismissal. Terminology must be grounded strictly in machine learning, numerical linear algebra, and dynamical systems.

### 5. The Diagonal Taylor Blind Spot
* **The Math Deficit**: For distributed vectors $q, k \in \mathbb{R}^D$ on the unit sphere, $(q \odot q)^\top (k \odot k)$ captures less than **5% of the total energy** of $(q^\top k)^2$. Over 95% of the quadratic term resides in the off-diagonal cross-terms $q_i q_j k_i k_j$ ($i \ne j$).
* **Consequence**: The diagonal 2nd-order moment provides much less non-linear sharpening than theoretically promised.

### 6. Parabolic Rebound of Taylor Expansion for Negative Logits
* The polynomial $P(x) = 1 + x + \frac{1}{2}x^2$ rebounds upward when $x < -1$ ($P(-10) = +41$, while $\exp(-10) \approx 0$).
* Tokens that should receive zero attention instead receive massive positive weights.
* Without strict QK-LayerNorm, the mechanism catastrophically diverges.

### 7. Unbounded Autonomous Daemon Loop & Storage Exhaustion
* `falcon_autonomous_researcher.py` ran 955 continuous cycles:
  * Produced 90,529 lines of markdown in `THEORY_DOSSIER.md` (unreadable for humans).
  * Generated 92 GB of redundant `.pt` checkpoint files in `checkpoints/`.
  * Filled host drive `/dev/sda2` to **92% capacity (only 72 GB remaining)**.
  * Pushed GPU VRAM to **97.9% saturation (15.58 / 15.92 GB)**.
* Autonomous research loops without human-in-the-loop validation or storage guardrails inevitably spiral into lore inflation and disk exhaustion.

---

## 🛠️ PART 3: THE GROUNDED TRUTH REBUILD BLUEPRINT

To rebuild the repository with total scientific integrity, follow this clear structural roadmap:

```
PRIME-Moment-Attention/ (Rebuilt Architecture)
├── c/                         # [KEEP & REFINE] Native C99 / Cosmopolitan APE engine
│   ├── prime_moment.c         # Pure C99 2nd-order recurrent kernel
│   ├── prime_moment.h
│   ├── prime_server.c         # Embedded POSIX HTTP inference server
│   ├── zipalign.c             # Page-aligned mmap weight bundler
│   └── Makefile
├── prime_attention/           # [CLEAN PYTHON PACKAGE] Core neural modules
│   ├── __init__.py
│   ├── attention.py           # PrimeMomentAttention with strict QK-norm & chunked SSD
│   ├── hybrid.py              # HybridWindowPrimeAttention (Sliding Softmax + Evicted Recurrence)
│   ├── model.py               # PrimeForCausalLM (Clean LLaMA/Qwen style decoder)
│   └── cache.py               # Hugging Face DynamicCache-compatible constant state
├── benchmarks/                # [PURE GROUND TRUTH BENCHMARKS]
│   ├── run_niah_real.py       # REAL passkey retrieval with actual model tokens (gap 50 to 8000)
│   ├── run_ppl_eval.py        # Perplexity on standard test sets (TinyStories, Wikitext-2)
│   ├── run_latency_memory.py  # Latency & VRAM benchmarking vs context length L
│   └── eval_gsm8k_real.py     # Real GSM8K evaluation checking EXACT final numeric answers
├── examples/                  # [MINIMAL RUNNABLE EXAMPLES]
│   ├── 01_minimal_step.py     # 10 lines: constant-memory autoregressive decoding
│   ├── 02_hybrid_window.py    # Hybrid sliding window + evicted token recurrence
│   └── 03_c_inference.sh      # Compiling and running the standalone C binary
├── docs/                      # [HONEST SCIENTIFIC DOCUMENTATION]
│   ├── MATHEMATICAL_FORMULATION.md  # Taylor expansion, diagonal bottleneck derivation
│   ├── RETENTION_LIMITS.md          # Real exponential decay half-life (~2k-4k tokens)
│   └── REPRODUCIBILITY.md           # Exact instructions to reproduce all figures
├── README.md                  # Honest, clear, professional README without hyperbole
└── LICENSE
```

### Action Items for Rebuild:
1. **Clean Out Flawed Files**: Delete `venturi_entrainment.py`, `spin_echo_recovery.py`, the regex hardcodings in `prime_net.c`, and all retracted scripts in `experiments/`.
2. **Prune Checkpoints**: Delete redundant step checkpoints in `checkpoints/` to immediately free up ~70–80 GB of disk space, keeping only the best 125M and 50M base checkpoints.
3. **Ground All Benchmark Claims**: Report exact retention half-lives (e.g. *"PRIME retains 95% signal at 1,000 tokens, degrading to noise by 8,000 tokens"*), rather than advertising "1M Context Retrieval".
4. **Standardize the Evaluator**: For reasoning evaluation, use genuine ground-truth exact match parsing against standardized benchmarks (GSM8K, ARC, HumanEval) rather than `<think>` tag counts.
