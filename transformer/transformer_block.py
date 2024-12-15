import torch.nn as nn
from config import DyLLMArgs
import torch
from transformer.multihead_attention import MultiHeadAttention
from transformer.swiglu_mlp import SwiGLUMLP
from transformer.normalization import RMSNorm


class TransformerDecoderBlock(nn.Module):
    def __init__(self, layer_id: int, args: DyLLMArgs):
        super().__init__()
        self.layer_id = layer_id
        self.args = args

        self.attention = MultiHeadAttention(args)
        self.mlp = SwiGLUMLP(args.model_dim, args.mlp_hidden_dim, args.mlp_multiple_of, args.dropout)

        self.attention_norm = RMSNorm(args.model_dim, args.norm_epsilon)
        self.mlp_norm = RMSNorm(args.model_dim, args.norm_epsilon)

    def forward(
        self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
    ):
        attn_out = x + self.attention.forward(
            self.attention_norm(x), freqs_cos, freqs_sin
        )
        return attn_out + self.mlp.forward(self.mlp_norm(attn_out))
