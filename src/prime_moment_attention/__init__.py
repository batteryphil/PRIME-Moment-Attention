"""
PRIME Moment Attention: Constant-State Second-Order Recurrent Attention for Long-Context Transformer Inference
"""

from .attention import PrimeMomentAttention
from .cache import PrimeMomentCache
from .surgery import PrimeTransplantedAttention, convert_transformer_to_prime
from .timescales import LearnableTimescales

__version__ = "0.2.0"
__all__ = [
    "PrimeMomentAttention",
    "PrimeMomentCache",
    "PrimeTransplantedAttention",
    "convert_transformer_to_prime",
    "LearnableTimescales",
]
