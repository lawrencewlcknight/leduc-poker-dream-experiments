
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


class ResidualHiddenLayer(nn.Module):
    """Hidden layer with a same-width residual connection when dimensions match."""

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.linear = SonnetLinear(in_size, out_size)
        self.activation = nn.ReLU()
        self.use_residual = int(in_size) == int(out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        if self.use_residual:
            out = out + x
        return out


class ResidualMLP(nn.Module):
    """MLP with residual connections between same-width hidden layers."""

    def __init__(self, input_size: int, hidden_sizes: Sequence[int], output_size: int):
        super().__init__()
        hidden_sizes = list(map(int, hidden_sizes))
        layers: List[nn.Module] = []
        in_size = int(input_size)
        for hidden_size in hidden_sizes:
            layers.append(ResidualHiddenLayer(in_size, hidden_size))
            in_size = hidden_size
        layers.append(SonnetLinear(in_size, int(output_size)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CenteredAdvantageMLP(nn.Module):
    """MLP head whose action outputs are centred to zero mean per state."""

    def __init__(self, input_size: int, hidden_sizes: Sequence[int], output_size: int):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        in_size = int(input_size)
        for hidden_size in hidden_sizes:
            hidden_size = int(hidden_size)
            self.hidden_layers.append(SonnetLinear(in_size, hidden_size))
            in_size = hidden_size
        self.action_head = SonnetLinear(in_size, int(output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        raw_advantages = self.action_head(x)
        return raw_advantages - raw_advantages.mean(dim=-1, keepdim=True)


class DuelingMLP(nn.Module):
    """MLP with a scalar state head plus centred action-advantage head."""

    def __init__(self, input_size: int, hidden_sizes: Sequence[int], output_size: int):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        in_size = int(input_size)
        for hidden_size in hidden_sizes:
            hidden_size = int(hidden_size)
            self.hidden_layers.append(SonnetLinear(in_size, hidden_size))
            in_size = hidden_size
        self.value_head = SonnetLinear(in_size, 1)
        self.action_head = SonnetLinear(in_size, int(output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        state_value = self.value_head(x)
        raw_advantages = self.action_head(x)
        centred_advantages = raw_advantages - raw_advantages.mean(dim=-1, keepdim=True)
        return state_value + centred_advantages


def build_network(
    network_type: str,
    input_size: int,
    hidden_sizes: Sequence[int],
    output_size: int,
) -> nn.Module:
    """Build a DREAM network implementation by type."""
    network_type = str(network_type).lower()
    if network_type == "mlp":
        return MLP(input_size, hidden_sizes, output_size)
    if network_type == "residual_mlp":
        return ResidualMLP(input_size, hidden_sizes, output_size)
    if network_type == "centered_advantage_mlp":
        return CenteredAdvantageMLP(input_size, hidden_sizes, output_size)
    if network_type == "dueling_mlp":
        return DuelingMLP(input_size, hidden_sizes, output_size)
    valid = ["mlp", "residual_mlp", "centered_advantage_mlp", "dueling_mlp"]
    raise ValueError(f"network_type must be one of {valid}")
