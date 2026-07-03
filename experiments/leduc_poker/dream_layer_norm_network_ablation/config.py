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


LAYER_NORM_NETWORK_VARIANT_SETS = {
    "all": [variant["variant_id"] for variant in LAYER_NORM_NETWORK_VARIANTS],
    "plain": [
        "plain_layers2_width32",
        "plain_layers4_width32",
        "plain_layers8_width32",
    ],
    "layer_norm": [
        BASELINE_VARIANT,
        "layer_norm_layers2_width32",
        "layer_norm_layers4_width32",
        "layer_norm_layers8_width32",
    ],
    "residual_layer_norm": [
        BASELINE_VARIANT,
        "residual_layer_norm_layers2_width32",
        "residual_layer_norm_layers4_width32",
        "residual_layer_norm_layers8_width32",
    ],
}


def select_layer_norm_network_variants(variant_ids: list[str]) -> list[dict]:
    """Return configured variants in the requested order."""
    variants_by_id = {
        variant["variant_id"]: variant for variant in LAYER_NORM_NETWORK_VARIANTS
    }
    missing = [variant_id for variant_id in variant_ids if variant_id not in variants_by_id]
    if missing:
        raise ValueError(f"Unknown layer-normalisation variant ids: {missing}")
    return [variants_by_id[variant_id] for variant_id in variant_ids]


LAYER_NORM_EXPERIMENT_VARIANTS = select_layer_norm_network_variants(
    LAYER_NORM_NETWORK_VARIANT_SETS["layer_norm"]
)


EXPERIMENT_CONFIG = copy.deepcopy(NETWORK_EXPERIMENT_CONFIG)
EXPERIMENT_CONFIG.update(
    {
        "experiment_name": "leduc_poker_dream_layer_norm_network_ablation",
        "algorithm": "DREAM-style OpenSpiel LayerNorm-network ablation",
        "plot_prefix": "dream_layer_norm_network",
        "plot_title": "DREAM LayerNorm-Network Ablation",
        "baseline_variant": BASELINE_VARIANT,
        "output_root": Path("outputs") / "dream_layer_norm_network_ablation",
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = copy.deepcopy(NETWORK_SMOKE_TEST_CONFIG_OVERRIDES)
SMOKE_TEST_CONFIG_OVERRIDES["output_root"] = (
    Path("outputs") / "smoke_tests" / "dream_layer_norm_network_ablation"
)
