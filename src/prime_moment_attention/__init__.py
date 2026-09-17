"""
PRIME Moment Attention: Constant-State Second-Order Recurrent Attention for Long-Context Transformer Inference
"""

from .attention import PrimeMomentAttention
from .cache import PrimeMomentCache
from .selective import PrimeSelectiveAttention, convert_transformer_to_prime_selective
from .surgery import PrimeTransplantedAttention, convert_transformer_to_prime
from .timescales import LearnableTimescales
from .hybrid import HybridWindowPrimeAttention, convert_transformer_to_hybrid_prime
from .adaptive import AdaptivePrimeRouter, EmpiricalBandwidthRouter
from .gumbel import gumbel_softmax_ste, GumbelHeadRouter, GumbelAnnealingScheduler
from .distillation import GumbelPrimeQwen2Attention, convert_qwen_to_stage7_hybrid
from .model import PrimeConfig, PrimeBlock, PrimeForCausalLM

__version__ = "0.4.0"

def load(
    model_name_or_path: str,
    mode: str = "hybrid",
    target_layers=None,
    window_size: int = 512,
    **kwargs
):
    """
    One-line loader: loads any Hugging Face model and transplants PRIME attention in-place.
    
    Example:
        import prime_moment_attention as prime
        model = prime.load("Qwen/Qwen2.5-0.5B")
    """
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    if mode == "hybrid":
        model, _ = convert_transformer_to_hybrid_prime(
            model, target_layers=target_layers, window_size=window_size
        )
    else:
        model, _ = convert_transformer_to_prime(model, target_layers=target_layers)
    return model

from_pretrained = load

__all__ = [
    "load",
    "from_pretrained",
    "PrimeMomentAttention",
    "PrimeMomentCache",
    "PrimeSelectiveAttention",
    "convert_transformer_to_prime_selective",
    "PrimeTransplantedAttention",
    "convert_transformer_to_prime",
    "LearnableTimescales",
    "HybridWindowPrimeAttention",
    "convert_transformer_to_hybrid_prime",
    "AdaptivePrimeRouter",
    "EmpiricalBandwidthRouter",
    "gumbel_softmax_ste",
    "GumbelHeadRouter",
    "GumbelAnnealingScheduler",
    "GumbelPrimeQwen2Attention",
    "convert_qwen_to_stage7_hybrid",
    "PrimeConfig",
    "PrimeBlock",
    "PrimeForCausalLM",
]
