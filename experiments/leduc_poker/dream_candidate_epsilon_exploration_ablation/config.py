"""Configuration for candidate-architecture DREAM epsilon exploration."""

import copy
from pathlib import Path

from dream_poker.constants import SMOKE_TEST_SEEDS
from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    make_exp22_candidate_fixed_baseline_config,
)
from experiments.leduc_poker.dream_architecture_candidate_comparison.config import (
    ADVANTAGE_CANDIDATE_LAYERS,
    POLICY_BASELINE_LAYERS,
)
from experiments.leduc_poker.dream_epsilon_exploration_ablation.config import (
    EXPERIMENT_CONFIG as EPSILON_EXPERIMENT_CONFIG,
    SMOKE_TEST_CONFIG_OVERRIDES as EPSILON_SMOKE_TEST_CONFIG_OVERRIDES,
)


BASELINE_VARIANT = "candidate_epsilon_006_baseline"
PREVIOUS_BEST_EPSILON_VARIANT = "candidate_epsilon_010"
DEFAULT_SEEDS = [1234, 2025, 31415]


def epsilon_variant(
    variant_id: str,
    label: str,
    epsilon: float,
    description: str,
) -> dict:
    return {
        "variant_id": variant_id,
        "label": label,
        "epsilon": float(epsilon),
        "description": description,
    }


CANDIDATE_EPSILON_VARIANTS = [
    epsilon_variant(
        BASELINE_VARIANT,
        "epsilon 0.06",
        0.06,
        "Inherited DREAM exploration rate under the architecture-selected candidate.",
    ),
    epsilon_variant(
        PREVIOUS_BEST_EPSILON_VARIANT,
        "epsilon 0.10",
        0.10,
        "Best-performing exploration rate from the original DREAM epsilon ablation.",
    ),
    epsilon_variant(
        "candidate_epsilon_015",
        "epsilon 0.15",
        0.15,
        "Moderately higher exploration rate to test whether the 0.10 gain continues.",
    ),
    epsilon_variant(
        "candidate_epsilon_020",
        "epsilon 0.20",
        0.20,
        "High exploration rate to test where excess randomisation begins to hurt.",
    ),
]


EXPERIMENT_CONFIG = copy.deepcopy(EPSILON_EXPERIMENT_CONFIG)
EXPERIMENT_CONFIG.update(
    {
        "experiment_name": "leduc_poker_dream_candidate_epsilon_exploration_ablation",
        "algorithm": "DREAM-style OpenSpiel candidate-architecture epsilon ablation",
        "policy_network_layers": list(POLICY_BASELINE_LAYERS),
        "advantage_network_layers": list(ADVANTAGE_CANDIDATE_LAYERS),
        "baseline_network_layers": list(POLICY_BASELINE_LAYERS),
        "network_treatment": "architecture_selected_candidate",
        "seeds": list(DEFAULT_SEEDS),
        "epsilon": 0.06,
        "plot_prefix": "dream_candidate_epsilon_exploration",
        "plot_title": "DREAM Candidate Epsilon-Exploration Ablation",
        "baseline_variant": BASELINE_VARIANT,
        "previous_best_epsilon_variant": PREVIOUS_BEST_EPSILON_VARIANT,
        "ablation_variants": list(CANDIDATE_EPSILON_VARIANTS),
        "treatment_keys": ["epsilon"],
        "fixed_baseline": make_exp22_candidate_fixed_baseline_config(),
        "output_root": Path("outputs") / "dream_candidate_epsilon_exploration_ablation",
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = copy.deepcopy(EPSILON_SMOKE_TEST_CONFIG_OVERRIDES)
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
        "fixed_baseline": {"enabled": False},
        "output_root": (
            Path("outputs") / "smoke_tests" / "dream_candidate_epsilon_exploration_ablation"
        ),
    }
)
