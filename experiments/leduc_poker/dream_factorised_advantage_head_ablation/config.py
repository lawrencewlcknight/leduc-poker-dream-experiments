"""Configuration for the DREAM factorised advantage-head ablation."""

import copy
from pathlib import Path

from experiments.leduc_poker.dream_network_size_ablation.config import (
    EXPERIMENT_CONFIG as NETWORK_EXPERIMENT_CONFIG,
    SMOKE_TEST_CONFIG_OVERRIDES as NETWORK_SMOKE_TEST_CONFIG_OVERRIDES,
)


WIDTH = 32
DEPTHS = (2, 4, 8)
POLICY_LAYERS = [32, 32]
BASELINE_LAYERS = [32, 32]
BASELINE_VARIANT = "direct_advantage_layers2_width32"
HEAD_TYPES = (
    ("mlp", "direct", "Direct"),
    ("centered_advantage_mlp", "centered", "Centred"),
    ("dueling_mlp", "dueling", "Dueling"),
)


def factorised_head_variant(
    depth: int,
    advantage_network_type: str,
    treatment: str,
    label_prefix: str,
) -> dict:
    layers = [WIDTH] * int(depth)
    return {
        "variant_id": f"{treatment}_advantage_layers{depth}_width{WIDTH}",
        "label": f"{label_prefix} advantage head {depth}x{WIDTH}",
        "network_treatment": treatment,
        "varied_network": "advantage",
        "advantage_head_type": treatment,
        "policy_network_type": "mlp",
        "advantage_network_type": advantage_network_type,
        "baseline_network_type": "mlp",
        "policy_network_layers": list(POLICY_LAYERS),
        "advantage_network_layers": list(layers),
        "baseline_network_layers": list(BASELINE_LAYERS),
        "network_architecture": "advantage_" + "x".join(str(width) for width in layers),
        "network_depth": int(depth),
        "network_max_width": WIDTH,
        "network_hidden_units": sum(layers),
        "description": (
            f"{label_prefix} DREAM advantage-network output head at depth "
            f"{depth} and width {WIDTH}; policy and baseline networks stay 2x32 MLPs."
        ),
    }


FACTORISED_ADVANTAGE_HEAD_VARIANTS = [
    factorised_head_variant(depth, network_type, treatment, label_prefix)
    for depth in DEPTHS
    for network_type, treatment, label_prefix in HEAD_TYPES
]


EXPERIMENT_CONFIG = copy.deepcopy(NETWORK_EXPERIMENT_CONFIG)
EXPERIMENT_CONFIG.update(
    {
        "experiment_name": "leduc_poker_dream_factorised_advantage_head_ablation",
        "algorithm": "DREAM-style OpenSpiel factorised advantage-head ablation",
        "policy_network_layers": list(POLICY_LAYERS),
        "advantage_network_layers": [32, 32],
        "baseline_network_layers": list(BASELINE_LAYERS),
        "policy_network_type": "mlp",
        "advantage_network_type": "mlp",
        "baseline_network_type": "mlp",
        "network_treatment": "direct",
        "plot_prefix": "dream_factorised_advantage_head",
        "plot_title": "DREAM Factorised Advantage-Head Ablation",
        "baseline_variant": BASELINE_VARIANT,
        "output_root": Path("outputs") / "dream_factorised_advantage_head_ablation",
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = copy.deepcopy(NETWORK_SMOKE_TEST_CONFIG_OVERRIDES)
SMOKE_TEST_CONFIG_OVERRIDES["output_root"] = (
    Path("outputs") / "smoke_tests" / "dream_factorised_advantage_head_ablation"
)
