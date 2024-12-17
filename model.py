import torch.nn as nn
from config import DyLLMArgs
from transformer.transformer_block import TransformerDecoderBlock
from transformer.rope import precompute_freqs_cis
from transformer.normalization import RMSNorm
import torch
from typing import Optional
import torch.nn.functional as F
import math


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
        self.classifier.SKIP_INIT = 1
        self.token_embeddings.weight = self.classifier.weight
        self.apply(self._init_weights)

        for param_name, param in self.named_parameters():
            if param_name.endswith("w3.weight") or param_name.endswith("head_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * args.num_layers))

        freqs_cos, freqs_sin = precompute_freqs_cis(args.model_dim // args.num_heads, args.context_length)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if not hasattr(module, "SKIP_INIT"):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        _, context_length = x.shape
        h = self.token_embeddings(x)

        freqs_cos, freqs_sin = (
            self.freqs_cos[:context_length],
            self.freqs_sin[:context_length],
        )

        for transformer in self.transformer_blocks:
            h = transformer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        logits = self.classifier(h)
        
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        return logits, loss

    def configure_optimizers(self, learning_rate, weight_decay):
        params = {name: param for name, param in self.named_parameters() if param.requires_grad}

        decay_params = [param for _, param in params.items() if param.dim() >= 2]
        nodecay_params = [param for _, param in params.items() if param.dim() < 2]

        optimizer_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {num_params}")

        optimizer = torch.optim.AdamW(optimizer_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=True)
        return optimizer