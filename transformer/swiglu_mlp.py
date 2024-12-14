import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        up = F.silu(self.w1(x))
        gate = self.w3(x)

        out = self.w2(up * gate)
        return self.dropout(out)
