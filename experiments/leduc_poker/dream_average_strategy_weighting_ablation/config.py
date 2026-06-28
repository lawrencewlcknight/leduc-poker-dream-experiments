"""Configuration for the DREAM average-strategy weighting ablation."""

from pathlib import Path

from dream_poker.constants import (
    DEFAULT_SEEDS_5,
    EXPLOITABILITY_THRESHOLD,
    LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    LEDUC_GAME_VALUE_P0,
    SMOKE_TEST_SEEDS,
    THESIS_SEEDS_10,
)


BASELINE_VARIANT = "linear_avg_importance_dream_baseline"


AVERAGE_STRATEGY_WEIGHTING_VARIANTS = [
    {
        "variant_id": "linear_avg_importance_dream_baseline",
        "label": "Linear avg + importance",
        "average_strategy_weighting": "linear",
        "average_strategy_importance_correction": "reach_ratio",
        "description": (
            "Current DREAM baseline: CFR-style iteration weighting in the "
            "average-policy loss, combined with the sampled-traversal reach-ratio "
            "correction."
        ),
    },
    {
        "variant_id": "uniform_avg_importance",
        "label": "Uniform avg + importance",
        "average_strategy_weighting": "uniform",
        "average_strategy_importance_correction": "reach_ratio",
        "description": (
            "Removes iteration weighting from average-policy fitting while retaining "
            "the sampled-traversal reach-ratio correction."
        ),
    },
]


EXPERIMENT_CONFIG = {
    "experiment_name": "leduc_poker_dream_average_strategy_weighting_ablation",
    "game_name": "leduc_poker",
    "algorithm": "DREAM-style OpenSpiel average-strategy weighting ablation",
    "num_iterations": 175,
    "num_traversals": 160,
    "evaluation_interval": 25,
    "policy_network_train_every": 25,
    "policy_network_train_steps": 100,
    "advantage_network_train_steps": 50,
    "baseline_network_train_steps": 50,
    "policy_network_layers": [32, 32],
    "advantage_network_layers": [32, 32],
    "baseline_network_layers": [32, 32],
    "policy_network_type": "mlp",
    "advantage_network_type": "mlp",
    "baseline_network_type": "mlp",
    "learning_rate": 0.003,
    "batch_size_advantage": 1024,
    "batch_size_strategy": 1024,
    "batch_size_baseline": 1024,
    "advantage_memory_capacity": int(1e6),
    "strategy_memory_capacity": int(1e6),
    "baseline_memory_capacity": int(1e6),
    "epsilon": 0.06,
    "compute_exploitability": True,
    "isolate_policy_training_rng": True,
    "seeds": [1234, 2025, 31415],
    "optional_development_seeds_5": DEFAULT_SEEDS_5,
    "optional_thesis_seeds_10": THESIS_SEEDS_10,
    "game_value_player_0": LEDUC_GAME_VALUE_P0,
    "average_policy_value_target": LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": EXPLOITABILITY_THRESHOLD,
    "average_strategy_weighting": "linear",
    "average_strategy_importance_correction": "reach_ratio",
    "baseline_variant": BASELINE_VARIANT,
    "output_root": Path("outputs") / "dream_average_strategy_weighting_ablation",
}


SMOKE_TEST_CONFIG_OVERRIDES = {
    "seeds": SMOKE_TEST_SEEDS,
    "num_iterations": 10,
    "num_traversals": 50,
    "policy_network_train_steps": 20,
    "advantage_network_train_steps": 20,
    "baseline_network_train_steps": 20,
    "policy_network_train_every": 5,
    "evaluation_interval": 5,
    "output_root": (
        Path("outputs") / "smoke_tests" / "dream_average_strategy_weighting_ablation"
    ),
}
