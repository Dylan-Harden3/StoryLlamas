import torch.nn as nn
from config import DyLLMArgs
from transformer.transformer_block import TransformerDecoderBlock
from transformer.rope import precompute_freqs_cis
from transformer.normalization import RMSNorm
import torch


class DyLLM(nn.Module):
    def __init__(self, args: DyLLMArgs):
        super().__init__()
        self.args = args
        self.token_embeddings = nn.Embedding(args.vocab_size, args.model_dim)

        self.transformer_blocks = nn.ModuleList()
        for i in range(args.num_layers):
            self.transformer_blocks.append(TransformerDecoderBlock(i, args))

        self.norm = RMSNorm(args.model_dim, args.norm_epsilon)
        self.classifier = nn.Linear(args.model_dim, args.vocab_size, bias=False)

        freqs_cos, freqs_sin = precompute_freqs_cis(args.model_dim // args.num_heads, args.context_length)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, x: torch.Tensor):
        _, context_length = x.shape
        h = self.token_embeddings(x)

        freqs_cos, freqs_sin = (
            self.freqs_cos[:context_length],
            self.freqs_sin[:context_length],
        )

        for transformer in self.transformer_blocks:
            h = transformer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        return self.classifier(h)
