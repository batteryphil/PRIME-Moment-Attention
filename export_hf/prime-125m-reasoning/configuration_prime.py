"""
Configuration class for PRIME (Selective Moment Attention) model.
"""

from transformers.configuration_utils import PretrainedConfig


class PrimeConfig(PretrainedConfig):
    model_type = "prime"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        head_dim: int = 64,
        num_kv_heads: int = 12,
        intermediate_size: int = 2048,
        decay: float = 0.995,
        use_qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
        eps: float = 1.0,
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        use_selective: bool = True,
        min_tau: float = 2.0,
        max_tau: float = 2000.0,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: int = 50256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.intermediate_size = intermediate_size
        self.decay = decay
        self.use_qk_norm = use_qk_norm
        self.rms_norm_eps = rms_norm_eps
        self.eps = eps
        self.tie_word_embeddings = tie_word_embeddings
        self.initializer_range = initializer_range
        self.use_selective = use_selective
        self.min_tau = min_tau
        self.max_tau = max_tau
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
