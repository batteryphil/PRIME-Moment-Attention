"""
PRIME Moment Attention: Constant-State Second-Order Recurrent Attention for Long-Context Transformer Inference
"""

from .attention import PrimeMomentAttention
from .cache import PrimeMomentCache
from .selective import PrimeSelectiveAttention, convert_transformer_to_prime_selective
from .surgery import PrimeTransplantedAttention, convert_transformer_to_prime
from .timescales import LearnableTimescales

__version__ = "0.3.0"
__all__ = [
    "PrimeMomentAttention",
    "PrimeMomentCache",
    "PrimeSelectiveAttention",
    "convert_transformer_to_prime_selective",
    "PrimeTransplantedAttention",
    "convert_transformer_to_prime",
    "LearnableTimescales",
]
