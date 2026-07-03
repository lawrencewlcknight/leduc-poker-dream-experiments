"""Configuration for the DREAM residual-LayerNorm network ablation."""

import copy
from pathlib import Path

from experiments.leduc_poker.dream_layer_norm_network_ablation.config import (
    BASELINE_VARIANT,
    LAYER_NORM_NETWORK_VARIANT_SETS,
    NETWORK_EXPERIMENT_CONFIG,
    NETWORK_SMOKE_TEST_CONFIG_OVERRIDES,
    select_layer_norm_network_variants,
)


RESIDUAL_LAYER_NORM_NETWORK_VARIANTS = select_layer_norm_network_variants(
    LAYER_NORM_NETWORK_VARIANT_SETS["residual_layer_norm"]
)


EXPERIMENT_CONFIG = copy.deepcopy(NETWORK_EXPERIMENT_CONFIG)
EXPERIMENT_CONFIG.update(
    {
        "experiment_name": "leduc_poker_dream_residual_layer_norm_network_ablation",
        "algorithm": "DREAM-style OpenSpiel residual-LayerNorm network ablation",
        "plot_prefix": "dream_residual_layer_norm_network",
        "plot_title": "DREAM Residual-LayerNorm Network Ablation",
        "baseline_variant": BASELINE_VARIANT,
        "output_root": Path("outputs") / "dream_residual_layer_norm_network_ablation",
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = copy.deepcopy(NETWORK_SMOKE_TEST_CONFIG_OVERRIDES)
SMOKE_TEST_CONFIG_OVERRIDES["output_root"] = (
    Path("outputs") / "smoke_tests" / "dream_residual_layer_norm_network_ablation"
)
