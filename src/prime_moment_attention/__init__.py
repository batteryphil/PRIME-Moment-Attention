"""
PRIME Moment Attention: Constant-State Second-Order Recurrent Attention for Long-Context Transformer Inference
"""

from .attention import PrimeMomentAttention
from .cache import PrimeMomentCache
from .selective import PrimeSelectiveAttention, convert_transformer_to_prime_selective
from .surgery import PrimeTransplantedAttention, convert_transformer_to_prime
from .timescales import LearnableTimescales
from .hybrid import HybridWindowPrimeAttention, convert_transformer_to_hybrid_prime

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
]
