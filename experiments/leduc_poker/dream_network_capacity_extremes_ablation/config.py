"""Configuration for the DREAM network-capacity extremes ablation."""

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
        "experiment_name": "leduc_poker_dream_network_capacity_extremes_ablation",
        "algorithm": "DREAM-style OpenSpiel network-capacity extremes ablation",
        "plot_prefix": "dream_network_capacity_extremes",
        "plot_title": "DREAM Network-Capacity Extremes Ablation",
        "output_root": Path("outputs") / "dream_network_capacity_extremes_ablation",
    }
)

NETWORK_CAPACITY_EXTREMES_VARIANTS = select_network_variants(
    NETWORK_SIZE_VARIANT_SETS["capacity_extremes"]
)

SMOKE_TEST_CONFIG_OVERRIDES = copy.deepcopy(NETWORK_SMOKE_TEST_CONFIG_OVERRIDES)
SMOKE_TEST_CONFIG_OVERRIDES["output_root"] = (
    Path("outputs") / "smoke_tests" / "dream_network_capacity_extremes_ablation"
)
