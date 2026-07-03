"""Configuration for the DREAM plain-network depth reference ablation."""

import copy
from pathlib import Path

from experiments.leduc_poker.dream_layer_norm_network_ablation.config import (
    BASELINE_VARIANT,
    LAYER_NORM_NETWORK_VARIANT_SETS,
    NETWORK_EXPERIMENT_CONFIG,
    NETWORK_SMOKE_TEST_CONFIG_OVERRIDES,
    select_layer_norm_network_variants,
)


PLAIN_NETWORK_DEPTH_VARIANTS = select_layer_norm_network_variants(
    LAYER_NORM_NETWORK_VARIANT_SETS["plain"]
)


EXPERIMENT_CONFIG = copy.deepcopy(NETWORK_EXPERIMENT_CONFIG)
EXPERIMENT_CONFIG.update(
    {
        "experiment_name": "leduc_poker_dream_plain_network_depth_ablation",
        "algorithm": "DREAM-style OpenSpiel plain-network depth reference ablation",
        "plot_prefix": "dream_plain_network_depth",
        "plot_title": "DREAM Plain-Network Depth Reference",
        "baseline_variant": BASELINE_VARIANT,
        "output_root": Path("outputs") / "dream_plain_network_depth_ablation",
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = copy.deepcopy(NETWORK_SMOKE_TEST_CONFIG_OVERRIDES)
SMOKE_TEST_CONFIG_OVERRIDES["output_root"] = (
    Path("outputs") / "smoke_tests" / "dream_plain_network_depth_ablation"
)
