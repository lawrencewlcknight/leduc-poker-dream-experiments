"""Configuration for the DREAM architecture-candidate comparison."""

import copy
from pathlib import Path

from dream_poker.constants import DEFAULT_SEEDS_5, SMOKE_TEST_SEEDS
from experiments.leduc_poker.dream_network_size_ablation.config import (
    EXPERIMENT_CONFIG as NETWORK_EXPERIMENT_CONFIG,
    SMOKE_TEST_CONFIG_OVERRIDES as NETWORK_SMOKE_TEST_CONFIG_OVERRIDES,
)
from experiments.leduc_poker.dream_role_specific_capacity_ablation.config import (
    role_specific_capacity_variant,
)


BASELINE_VARIANT = "baseline_all_2x32"
CANDIDATE_VARIANT = "candidate_advantage_2x128_policy_baseline_2x32"
POLICY_BASELINE_LAYERS = [32, 32]
ADVANTAGE_CANDIDATE_LAYERS = [128, 128]


ARCHITECTURE_CANDIDATE_VARIANTS = [
    role_specific_capacity_variant(
        BASELINE_VARIANT,
        "Baseline all networks 2x32",
        policy_layers=POLICY_BASELINE_LAYERS,
        advantage_layers=POLICY_BASELINE_LAYERS,
        baseline_layers=POLICY_BASELINE_LAYERS,
        network_treatment="baseline_all_2x32",
        description=(
            "Original DREAM baseline architecture with 2x32 policy, advantage, "
            "and learned-baseline networks."
        ),
    ),
    role_specific_capacity_variant(
        CANDIDATE_VARIANT,
        "Candidate advantage 2x128",
        policy_layers=POLICY_BASELINE_LAYERS,
        advantage_layers=ADVANTAGE_CANDIDATE_LAYERS,
        baseline_layers=POLICY_BASELINE_LAYERS,
        network_treatment="architecture_selected_candidate",
        description=(
            "Architecture-selected DREAM candidate with 2x128 player-specific "
            "advantage networks and baseline-sized policy and learned-baseline networks."
        ),
    ),
]


EXPERIMENT_CONFIG = copy.deepcopy(NETWORK_EXPERIMENT_CONFIG)
EXPERIMENT_CONFIG.update(
    {
        "experiment_name": "leduc_poker_dream_architecture_candidate_comparison",
        "algorithm": "DREAM-style OpenSpiel architecture-candidate comparison",
        "policy_network_layers": list(POLICY_BASELINE_LAYERS),
        "advantage_network_layers": list(POLICY_BASELINE_LAYERS),
        "baseline_network_layers": list(POLICY_BASELINE_LAYERS),
        "policy_network_type": "mlp",
        "advantage_network_type": "mlp",
        "baseline_network_type": "mlp",
        "network_treatment": "baseline_all_2x32",
        "seeds": list(DEFAULT_SEEDS_5),
        "plot_prefix": "dream_architecture_candidate",
        "plot_title": "DREAM Architecture Candidate Comparison",
        "baseline_variant": BASELINE_VARIANT,
        "candidate_variant": CANDIDATE_VARIANT,
        "output_root": Path("outputs") / "dream_architecture_candidate_comparison",
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = copy.deepcopy(NETWORK_SMOKE_TEST_CONFIG_OVERRIDES)
SMOKE_TEST_CONFIG_OVERRIDES.update(
    {
        "seeds": SMOKE_TEST_SEEDS[:1],
        "num_iterations": 3,
        "num_traversals": 4,
        "policy_network_train_steps": 1,
        "advantage_network_train_steps": 1,
        "baseline_network_train_steps": 1,
        "policy_network_train_every": 1,
        "evaluation_interval": 1,
        "output_root": (
            Path("outputs") / "smoke_tests" / "dream_architecture_candidate_comparison"
        ),
    }
)
