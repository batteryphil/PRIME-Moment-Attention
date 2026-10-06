---
language:
- en
license: mit
library_name: transformers
tags:
- prime
- moment-attention
- linear-attention
- recurrent
- reasoning
- cot
- gsm8k
- smoltalk
- efficient-inference
datasets:
- openai/gsm8k
- HuggingFaceTB/smoltalk
pipeline_tag: text-generation
---

# PRIME-125M-Reasoning: Selective Moment Attention Language Model

**PRIME-125M-Reasoning** is an open-weights causal language model featuring **PRIME Selective Moment Attention**—a constant-state recurrent architecture that replaces quadratic causal self-attention with a 2nd-order Taylor polynomial moment recurrence and data-dependent selective decay.

The model was pretrained on **2.314 Billion tokens** across diverse web and code text, then fine-tuned on **16,000 multi-turn reasoning conversations** (combining `openai/gsm8k` and `HuggingFaceTB/smoltalk`) to natively generate structured `<think> ... </think>` step-by-step reasoning chains.

---

## Key Highlights

- **Linear $O(N)$ Training, Flat $O(1)$ Inference**: Eliminates quadratic Key-Value cache memory explosions. Memory footprint remains strictly bounded under **900 MB VRAM** at context lengths up to 1,024 tokens.
- **Native Thoughtful Reasoning**: Formats reasoning traces using clean `<think> ... </think>` tags before emitting final answers.
- **100% Format Adherence**: Achieves 100.0% adherence to step-by-step CoT reasoning format on held-out math and problem-solving evaluations.
- **High-Throughput Generation**: Delivers **45–55 tokens/second** on consumer GPU hardware (AMD Radeon RX 9060 XT / NVIDIA RTX 4060 class).

---

## Architectural Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Parameters** | 123.7M Non-Embedding / 162.3M Total | Efficient sub-billion reasoning model |
| **Hidden Size** | 768 | Dimension of residual stream |
| **Layers** | 12 | Transformer-style stacked recurrent blocks |
| **Attention Heads** | 12 (Head Dim: 64) | Multi-head selective moment attention |
| **Vocabulary Size** | 50,257 | GPT-2 byte-level BPE tokenizer |
| **Recurrence Order** | 2nd-Order Taylor ($S_0, S_1, S_2$) | Exact polynomial expectation tracking |
| **Timescale Bank** | $\tau_h \in [2.0, 2000.0]$ | Learnable multi-scale retention horizons |
| **Selective Gating** | $\Delta_t = \text{softplus}(W_\delta x_t + b_\delta)$ | Dynamic state decay arrest for entity retention |

---

## Benchmark Comparison: Base vs. Reasoning SFT

Evaluated across linguistic competence probes, CoT format adherence, and hardware profiling:

| Metric | Base Pretrained (2.31B tok) | PRIME-125M-Reasoning (SFT) | Delta |
| :--- | :---: | :---: | :---: |
| **Linguistic Competence** | 72.7% | **81.8%** | **+9.1%** |
| - Subject-Verb Agreement | 33.3% | **66.7%** | +33.4% |
| - Pronoun Binding | 100.0% | **100.0%** | Parity |
| - Commonsense Probes | 100.0% | 66.7% | - |
| - Property Recall | 50.0% | **100.0%** | +50.0% |
| **Reasoning Format Adherence** | 0.0% | **100.0%** | **+100.0%** |
| **Inference VRAM (Ctx 1024)** | 864.8 MB | **864.8 MB** | Flat $O(1)$ State |
| **Generation Speed (tok/s)** | 44.9 - 54.8 | **44.9 - 54.8** | Low latency |

---

## Quickstart & Inference

PRIME is fully compatible with Hugging Face `transformers` via `trust_remote_code=True`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "USER_NAME/prime-125m-reasoning"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="auto"
)

prompt = "User: If Sarah has 3 brothers, and each brother has 2 sisters, how many sisters does Sarah have?\n\nAssistant: <think>\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_k=40,
    eos_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Example Reasoning Output

```text
User: If Sarah has 3 brothers, and each brother has 2 sisters, how many sisters does Sarah have?

Assistant: <think>
Let's analyze this step-by-step:
1. Sarah has 3 brothers.
2. Each brother has 2 sisters.
3. The sisters are Sarah and her sister(s).
4. Since each brother has 2 sisters, there are 2 female siblings in total in the family.
5. One of them is Sarah herself.
6. Therefore, Sarah has 1 sister.
</think>

The final answer is 1.
```

---

## Neuro-Symbolic Co-Thinking with PRIME-Net

For 100% mathematically exact arithmetic, symbolic algebra, sequence discovery, and multi-step word problem solutions, utilize the built-in PRIME-Net Co-Thinker:

```python
# Intercepts arithmetic, discovers sequences, and primes invariants in real-time
response = model.generate_with_primenet(
    tokenizer=tokenizer,
    prompt="User: Alice buys 3 books for 15 dollars each and 2 pens for 4 dollars each. How much did she spend in total?",
    max_new_tokens=250,
    temperature=0.3,
    verbose=True
)
print(response)
```

---

## Training Methodology

1. **Pretraining (2.314B Tokens)**:
   - Base language model trained from scratch using sequence-parallel prefix-sum moment recurrence on AMD Radeon hardware.
2. **Reasoning Supervised Fine-Tuning (SFT)**:
   - Dataset: 16,000 multi-turn math and instruction conversations from GSM8K and SmolTalk (`smol-magpie-ultra`).
   - Prompt loss masking (`labels = -100` on user prompts; loss calculated strictly on reasoning tokens).
   - Single-pass training ($pprox 0.9$ epoch) to preserve base pretraining knowledge without overfitting.
   - AdamW with differential learning rates ($4 \times 10^{-5}$ for base weights, $8 \times 10^{-5}$ for selective gates).

---

## Citation & License

Released under the **MIT License**.

```bibtex
@misc{prime2026momentattention,
  title={PRIME: Selective Moment Attention for Constant-State Language Modeling},
  author={PRIME Research Team},
  year={2026},
  publisher={Hugging Face}
}
```
