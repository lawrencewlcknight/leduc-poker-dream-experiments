import pytest
import torch

from dream_poker.networks import (
    CenteredAdvantageMLP,
    DuelingMLP,
    LayerNormMLP,
    MLP,
    ResidualMLP,
    build_network,
)


def test_build_network_selects_mlp_variants():
    mlp = build_network("mlp", 4, [8, 8], 3)
    layer_norm = build_network("layer_norm_mlp", 4, [8, 8], 3)
    residual = build_network("residual_mlp", 4, [8, 8], 3)
    residual_layer_norm = build_network("residual_layer_norm_mlp", 4, [8, 8], 3)
    centered = build_network("centered_advantage_mlp", 4, [8, 8], 3)
    dueling = build_network("dueling_mlp", 4, [8, 8], 3)

    assert isinstance(mlp, MLP)
    assert isinstance(layer_norm, LayerNormMLP)
    assert isinstance(residual, ResidualMLP)
    assert isinstance(residual_layer_norm, ResidualMLP)
    assert isinstance(centered, CenteredAdvantageMLP)
    assert isinstance(dueling, DuelingMLP)

    x = torch.zeros((2, 4), dtype=torch.float32)
    assert mlp(x).shape == (2, 3)
    assert layer_norm(x).shape == (2, 3)
    assert residual(x).shape == (2, 3)
    assert residual_layer_norm(x).shape == (2, 3)
    assert centered(x).shape == (2, 3)
    assert dueling(x).shape == (2, 3)


def test_layer_norm_network_variants_include_layer_norm_modules():
    layer_norm = build_network("layer_norm_mlp", 4, [8, 8], 3)
    residual_layer_norm = build_network("residual_layer_norm_mlp", 4, [8, 8], 3)

    assert sum(isinstance(module, torch.nn.LayerNorm) for module in layer_norm.modules()) == 2
    assert sum(isinstance(module, torch.nn.LayerNorm) for module in residual_layer_norm.modules()) == 2


def test_centered_advantage_head_outputs_zero_mean_actions():
    net = build_network("centered_advantage_mlp", 4, [8], 3)
    x = torch.randn((5, 4), dtype=torch.float32)

    out = net(x)

    torch.testing.assert_close(
        out.mean(dim=-1),
        torch.zeros(5, dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )


def test_dueling_head_separates_scalar_value_and_centered_action_terms():
    net = build_network("dueling_mlp", 4, [8], 3)
    x = torch.randn((5, 4), dtype=torch.float32)

    out = net(x)

    assert out.shape == (5, 3)
    assert isinstance(net, DuelingMLP)
    assert net.value_head.layer.out_features == 1
    assert net.action_head.layer.out_features == 3


def test_build_network_rejects_unknown_type():
    with pytest.raises(ValueError, match="network_type"):
        build_network("unknown", 4, [8], 3)
