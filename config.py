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
    num_kv_heads: Optional[int] = None


DYLLM_CONFIG_2M = DyLLMArgs(
    vocab_size=4096,
    model_dim=128,
    context_length=256,
    rope_theta=10000,
    num_layers=4,
    num_heads=4,
    dropout=0.05
)

DYLLM_CONFIG_7M = DyLLMArgs(
    vocab_size=4096,
    model_dim=256,
    context_length=256,
    rope_theta=10000,
    num_layers=6,
    num_heads=6,
    dropout=0.05
)

DYLLM_CONFIG_15M = DyLLMArgs(
    vocab_size=4096,
    model_dim=384,
    context_length=256,
    rope_theta=10000,
    num_layers=7,
    num_heads=7,
    dropout=0.05
)

CONFIGS = {
    "2m": DYLLM_CONFIG_2M,
    "7m": DYLLM_CONFIG_7M,
    "15m": DYLLM_CONFIG_15M
}
