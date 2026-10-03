"""
PRIME Moment Attention: Constant-State Second-Order Recurrent Attention for Long-Context Transformer Inference
"""

from .attention import PrimeMomentAttention
from .thought_reconstruction_map import GenerativeThoughtReconstructionLayer

__version__ = "0.6.0"

__all__ = [
    "PrimeMomentAttention",
    "GenerativeThoughtReconstructionLayer",
]
