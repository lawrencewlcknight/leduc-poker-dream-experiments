
"""Neural-network helpers for DREAM experiments."""

from __future__ import annotations

from typing import List, Sequence
import torch
from torch import nn


class SonnetLinear(nn.Module):
    """Linear layer using Xavier weight initialisation and zero bias."""

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.layer = nn.Linear(in_size, out_size)
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class MLP(nn.Module):
    """Small MLP used for advantage, baseline, and average-policy networks."""

    def __init__(self, input_size: int, hidden_sizes: Sequence[int], output_size: int):
        super().__init__()
        sizes = [int(input_size)] + list(map(int, hidden_sizes)) + [int(output_size)]
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(SonnetLinear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
