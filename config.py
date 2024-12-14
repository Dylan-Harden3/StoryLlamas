from dataclasses import dataclass


@dataclass
class DyLLMArgs:
    vocab_size: int
    model_dim: int
    context_length: int
    rope_theta: float
    num_layers: int
    mlp_hidden_dim: int
    norm_epsilon: float = 1e-5
    dropout: float = 0.0
