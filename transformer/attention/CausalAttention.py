import torch.nn as nn
from config import DyLLMArgs
import torch
import math
import torch.nn.functional as F
from transformer.rope import apply_rotary_emb


class CausalAttention(nn.Module):
    def __init__(self, args: DyLLMArgs):
        super().__init__()
        self.dim = args.dim
        self.head_dim = args.dim

        self.w_q = nn.Linear(self.dim, self.head_dim, bias=False)
        self.w_k = nn.Linear(self.dim, self.head_dim, bias=False)
        self.w_v = nn.Linear(self.dim, self.head_dim, bias=False)

        self.attention_dropout = nn.Dropout(args.dropout)
        self.residual_dropout = nn.Dropout(args.dropout)

        mask = torch.full((1, args.max_seq_len, args.max_seq_len), float("-inf"))
        self.register_buffer("mask", torch.triu(mask, diagonal=1))

    def forward(
        self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
    ):
        batch_size, sequence_length = x.shape
        # QKV
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(batch_size, sequence_length, 1, self.head_dim)
        k = k.view(batch_size, sequence_length, 1, self.head_dim)
        v = v.view(batch_size, sequence_length, 1, self.head_dim)

        # RoPE positional encoding
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # attention
        q_kt = (
            torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            + self.mask[:, :, :sequence_length, :sequence_length]
        )
        attention_scores = F.softmax(q_kt, dim=-1).type_as(q)
        attention_scores = self.attention_dropout(attention_scores)

        out = torch.matmul(attention_scores, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)

        return self.residual_dropout(out)
