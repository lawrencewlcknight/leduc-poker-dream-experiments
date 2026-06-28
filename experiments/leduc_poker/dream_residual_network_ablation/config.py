"""Configuration for the DREAM residual-network ablation."""

import copy
from pathlib import Path

from experiments.leduc_poker.dream_network_size_ablation.config import (
    EXPERIMENT_CONFIG as NETWORK_EXPERIMENT_CONFIG,
    SMOKE_TEST_CONFIG_OVERRIDES as NETWORK_SMOKE_TEST_CONFIG_OVERRIDES,
)


WIDTH = 32
DEPTHS = (2, 4, 8)
BASELINE_VARIANT = "plain_layers2_width32"


def residual_variant(depth: int, network_type: str) -> dict:
    label_prefix = "Residual" if network_type == "residual_mlp" else "Plain"
    variant_prefix = "residual" if network_type == "residual_mlp" else "plain"
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
        "description": f"{label_prefix} DREAM networks at depth {depth} and width {WIDTH}",
    }


RESIDUAL_NETWORK_VARIANTS = [
    residual_variant(depth, network_type)
    for depth in DEPTHS
    for network_type in ("mlp", "residual_mlp")
]


EXPERIMENT_CONFIG = copy.deepcopy(NETWORK_EXPERIMENT_CONFIG)
EXPERIMENT_CONFIG.update(
    {
        "experiment_name": "leduc_poker_dream_residual_network_ablation",
        "algorithm": "DREAM-style OpenSpiel residual-network ablation",
        "plot_prefix": "dream_residual_network",
        "plot_title": "DREAM Residual-Network Ablation",
        "baseline_variant": BASELINE_VARIANT,
        "output_root": Path("outputs") / "dream_residual_network_ablation",
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = copy.deepcopy(NETWORK_SMOKE_TEST_CONFIG_OVERRIDES)
SMOKE_TEST_CONFIG_OVERRIDES["output_root"] = (
    Path("outputs") / "smoke_tests" / "dream_residual_network_ablation"
)
