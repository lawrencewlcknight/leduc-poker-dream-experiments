import pytest
import torch

from dream_poker.networks import MLP, ResidualMLP, build_network


def test_build_network_selects_mlp_variants():
    mlp = build_network("mlp", 4, [8, 8], 3)
    residual = build_network("residual_mlp", 4, [8, 8], 3)

    assert isinstance(mlp, MLP)
    assert isinstance(residual, ResidualMLP)

    x = torch.zeros((2, 4), dtype=torch.float32)
    assert mlp(x).shape == (2, 3)
    assert residual(x).shape == (2, 3)


def test_build_network_rejects_unknown_type():
    with pytest.raises(ValueError, match="network_type"):
        build_network("unknown", 4, [8], 3)
