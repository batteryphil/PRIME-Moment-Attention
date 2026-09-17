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

__version__ = "0.4.0"

__all__ = [
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
]
