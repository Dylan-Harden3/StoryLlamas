import torch.nn as nn
from config import LlamaArgs
import torch
import math
import torch.nn.functional as F
from transformer.rope import apply_rotary_emb


class MultiHeadAttention(nn.Module):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        self.dim = args.model_dim
        self.num_heads = args.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_kv_heads = args.num_heads if args.num_kv_heads is None else args.num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0
        
        self.kv_repeats = self.num_heads // self.num_kv_heads

        self.w_q = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(self.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(self.dim, self.num_kv_heads * self.head_dim, bias=False)

        self.head_proj = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)

        self.attention_dropout = nn.Dropout(args.dropout)
        self.residual_dropout = nn.Dropout(args.dropout)

        mask = torch.full((1, self.num_heads, args.context_length, args.context_length), float("-inf"))
        self.register_buffer("mask", torch.triu(mask, diagonal=1))

    def forward(
        self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
    ):
        batch_size, context_length, _ = x.shape

        # QKV
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(batch_size, context_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, context_length, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, context_length, self.num_kv_heads, self.head_dim)

        # RoPE positional encoding
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        # stack kv heads if < q heads
        if self.num_heads > self.num_kv_heads:
            k = (k[:, :, :, None, :]
                .expand(batch_size, context_length, self.num_kv_heads, self.kv_repeats, self.head_dim)
                .reshape(batch_size, context_length, self.num_kv_heads * self.kv_repeats, self.head_dim))
            v = (v[:, :, :, None, :]
                .expand(batch_size, context_length, self.num_kv_heads, self.kv_repeats, self.head_dim)
                .reshape(batch_size, context_length, self.num_kv_heads * self.kv_repeats, self.head_dim))

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # attention
        q_kt = (
            torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            + self.mask[:, :, :context_length, :context_length]
        )
        attention_scores = F.softmax(q_kt, dim=-1).type_as(q)
        attention_scores = self.attention_dropout(attention_scores)

        attention_out = torch.matmul(attention_scores, v)

        # stack all heads
        attention_out = attention_out.transpose(1, 2).contiguous().view(batch_size, context_length, -1)

        # project back to model_dim
        out = self.head_proj(attention_out)

        return self.residual_dropout(out)
