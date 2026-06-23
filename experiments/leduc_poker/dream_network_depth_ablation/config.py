"""Configuration for the DREAM network-depth ablation."""

import copy
from pathlib import Path

from experiments.leduc_poker.dream_network_size_ablation.config import (
    BASELINE_VARIANT,
    EXPERIMENT_CONFIG as NETWORK_EXPERIMENT_CONFIG,
    NETWORK_SIZE_VARIANT_SETS,
    SMOKE_TEST_CONFIG_OVERRIDES as NETWORK_SMOKE_TEST_CONFIG_OVERRIDES,
    select_network_variants,
)


EXPERIMENT_CONFIG = copy.deepcopy(NETWORK_EXPERIMENT_CONFIG)
EXPERIMENT_CONFIG.update(
    {
        "experiment_name": "leduc_poker_dream_network_depth_ablation",
        "algorithm": "DREAM-style OpenSpiel network-depth ablation",
        "plot_prefix": "dream_network_depth",
        "plot_title": "DREAM Network-Depth Ablation",
        "output_root": Path("outputs") / "dream_network_depth_ablation",
    }
)

NETWORK_DEPTH_VARIANTS = select_network_variants(NETWORK_SIZE_VARIANT_SETS["depth_sweep"])

SMOKE_TEST_CONFIG_OVERRIDES = copy.deepcopy(NETWORK_SMOKE_TEST_CONFIG_OVERRIDES)
SMOKE_TEST_CONFIG_OVERRIDES["output_root"] = (
    Path("outputs") / "smoke_tests" / "dream_network_depth_ablation"
)
