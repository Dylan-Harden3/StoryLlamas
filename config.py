from dataclasses import dataclass
from typing import Optional

@dataclass
class DyLLMArgs:
    vocab_size: int
    model_dim: int
    num_heads: int
    context_length: int
    rope_theta: float
    num_layers: int
    mlp_hidden_dim: Optional[int] = None
    mlp_multiple_of: int = 256
    norm_epsilon: float = 1e-5
    dropout: float = 0.0
