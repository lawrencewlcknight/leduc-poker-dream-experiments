"""Configuration for the DREAM layer-normalisation network ablation."""

import copy
from pathlib import Path

from experiments.leduc_poker.dream_network_size_ablation.config import (
    EXPERIMENT_CONFIG as NETWORK_EXPERIMENT_CONFIG,
    SMOKE_TEST_CONFIG_OVERRIDES as NETWORK_SMOKE_TEST_CONFIG_OVERRIDES,
)


WIDTH = 32
DEPTHS = (2, 4, 8)
BASELINE_VARIANT = "plain_layers2_width32"
NETWORK_TYPES = (
    ("mlp", "plain", "Plain"),
    ("layer_norm_mlp", "layer_norm", "LayerNorm"),
    ("residual_layer_norm_mlp", "residual_layer_norm", "Residual+LayerNorm"),
)


def layer_norm_variant(
    depth: int,
    network_type: str,
    variant_prefix: str,
    label_prefix: str,
) -> dict:
    layers = [WIDTH] * int(depth)
    return {
        "variant_id": f"{variant_prefix}_layers{depth}_width{WIDTH}",
        "label": f"{label_prefix} {depth}x{WIDTH}",
        "network_treatment": network_type,
        "policy_network_type": network_type,
        "advantage_network_type": network_type,
        "baseline_network_type": network_type,
        "policy_network_layers": list(layers),
        "advantage_network_layers": list(layers),
        "baseline_network_layers": list(layers),
        "network_architecture": "x".join(str(width) for width in layers),
        "network_depth": int(depth),
        "network_max_width": WIDTH,
        "network_hidden_units": sum(layers),
        "description": (
            f"{label_prefix} DREAM networks at depth {depth} and width {WIDTH}"
        ),
    }


LAYER_NORM_NETWORK_VARIANTS = [
    layer_norm_variant(depth, network_type, variant_prefix, label_prefix)
    for depth in DEPTHS
    for network_type, variant_prefix, label_prefix in NETWORK_TYPES
]


EXPERIMENT_CONFIG = copy.deepcopy(NETWORK_EXPERIMENT_CONFIG)
EXPERIMENT_CONFIG.update(
    {
        "experiment_name": "leduc_poker_dream_layer_norm_network_ablation",
        "algorithm": "DREAM-style OpenSpiel layer-normalisation network ablation",
        "plot_prefix": "dream_layer_norm_network",
        "plot_title": "DREAM Layer-Normalisation Network Ablation",
        "baseline_variant": BASELINE_VARIANT,
        "output_root": Path("outputs") / "dream_layer_norm_network_ablation",
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = copy.deepcopy(NETWORK_SMOKE_TEST_CONFIG_OVERRIDES)
SMOKE_TEST_CONFIG_OVERRIDES["output_root"] = (
    Path("outputs") / "smoke_tests" / "dream_layer_norm_network_ablation"
)
