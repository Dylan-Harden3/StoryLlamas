import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, epsilon: float):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)

    def forward(self, x: torch.Tensor):
        norm = self.norm(x.float()).type_as(x)
        return norm * self.gamma


class LayerNorm(nn.Module):
    def __init__(self, dim: int, epsilon: float):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = x - mean / torch.sqrt(var + self.epsilon)
        return norm * self.gamma + self.beta
