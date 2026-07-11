"""Configuration for the DREAM role-specific capacity ablation."""

import copy
from pathlib import Path

from experiments.leduc_poker.dream_network_size_ablation.config import (
    EXPERIMENT_CONFIG as NETWORK_EXPERIMENT_CONFIG,
    SMOKE_TEST_CONFIG_OVERRIDES as NETWORK_SMOKE_TEST_CONFIG_OVERRIDES,
)


POLICY_BASELINE_LAYERS = [32, 32]
BASELINE_VARIANT = "all_2x32_reference"


def role_specific_capacity_variant(
    variant_id: str,
    label: str,
    *,
    policy_layers: list[int],
    advantage_layers: list[int],
    baseline_layers: list[int],
    network_treatment: str,
    description: str,
) -> dict:
    """Create a DREAM architecture variant with per-network hidden-layer sizes."""
    all_layers = [*policy_layers, *advantage_layers, *baseline_layers]
    return {
        "variant_id": variant_id,
        "label": label,
        "network_treatment": network_treatment,
        "policy_network_type": "mlp",
        "advantage_network_type": "mlp",
        "baseline_network_type": "mlp",
        "policy_network_layers": list(policy_layers),
        "advantage_network_layers": list(advantage_layers),
        "baseline_network_layers": list(baseline_layers),
        "network_architecture": (
            f"policy_{'x'.join(str(width) for width in policy_layers)}_"
            f"advantage_{'x'.join(str(width) for width in advantage_layers)}_"
            f"baseline_{'x'.join(str(width) for width in baseline_layers)}"
        ),
        "network_depth": max(len(policy_layers), len(advantage_layers), len(baseline_layers)),
        "network_max_width": max(all_layers),
        "network_hidden_units": sum(all_layers),
        "description": description,
    }


ROLE_SPECIFIC_CAPACITY_VARIANTS = [
    role_specific_capacity_variant(
        "all_2x32_reference",
        "All networks 2x32",
        policy_layers=POLICY_BASELINE_LAYERS,
        advantage_layers=[32, 32],
        baseline_layers=POLICY_BASELINE_LAYERS,
        network_treatment="matched_control",
        description="Matched control with policy, advantage, and baseline networks all at 2x32.",
    ),
    role_specific_capacity_variant(
        "advantage_3x64_policy_baseline_2x32",
        "Advantage 3x64",
        policy_layers=POLICY_BASELINE_LAYERS,
        advantage_layers=[64, 64, 64],
        baseline_layers=POLICY_BASELINE_LAYERS,
        network_treatment="advantage_only_capacity",
        description=(
            "Increase only the advantage networks to 3x64 while keeping the policy and "
            "baseline networks at 2x32."
        ),
    ),
    role_specific_capacity_variant(
        "advantage_2x128_policy_baseline_2x32",
        "Advantage 2x128",
        policy_layers=POLICY_BASELINE_LAYERS,
        advantage_layers=[128, 128],
        baseline_layers=POLICY_BASELINE_LAYERS,
        network_treatment="advantage_only_capacity",
        description=(
            "Increase only the advantage networks to 2x128 while keeping the policy and "
            "baseline networks at 2x32."
        ),
    ),
    role_specific_capacity_variant(
        "all_3x64_reference",
        "All networks 3x64",
        policy_layers=[64, 64, 64],
        advantage_layers=[64, 64, 64],
        baseline_layers=[64, 64, 64],
        network_treatment="all_network_capacity",
        description=(
            "Uniform 3x64 reference arm, matching the best endpoint architecture from the "
            "plain capacity sweep."
        ),
    ),
]


EXPERIMENT_CONFIG = copy.deepcopy(NETWORK_EXPERIMENT_CONFIG)
EXPERIMENT_CONFIG.update(
    {
        "experiment_name": "leduc_poker_dream_role_specific_capacity_ablation",
        "algorithm": "DREAM-style OpenSpiel role-specific capacity ablation",
        "policy_network_layers": list(POLICY_BASELINE_LAYERS),
        "advantage_network_layers": [32, 32],
        "baseline_network_layers": list(POLICY_BASELINE_LAYERS),
        "policy_network_type": "mlp",
        "advantage_network_type": "mlp",
        "baseline_network_type": "mlp",
        "network_treatment": "matched_control",
        "plot_prefix": "dream_role_specific_capacity",
        "plot_title": "DREAM Role-Specific Capacity Ablation",
        "baseline_variant": BASELINE_VARIANT,
        "output_root": Path("outputs") / "dream_role_specific_capacity_ablation",
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = copy.deepcopy(NETWORK_SMOKE_TEST_CONFIG_OVERRIDES)
SMOKE_TEST_CONFIG_OVERRIDES["output_root"] = (
    Path("outputs") / "smoke_tests" / "dream_role_specific_capacity_ablation"
)
