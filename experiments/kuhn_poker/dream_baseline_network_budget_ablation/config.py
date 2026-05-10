"""Configuration for the DREAM baseline-network training-budget ablation."""

from pathlib import Path

from dream_poker.constants import (
    DEFAULT_SEEDS_5,
    EXPLOITABILITY_THRESHOLD,
    KUHN_GAME_VALUE_P0,
    SMOKE_TEST_SEEDS,
    THESIS_SEEDS_10,
)


EXPERIMENT_CONFIG = {
    "experiment_name": "kuhn_poker_dream_baseline_network_budget_ablation",
    "game_name": "kuhn_poker",
    "algorithm": "DREAM-style OpenSpiel baseline-network budget ablation",
    "num_iterations": 500,
    "num_traversals": 320,
    "evaluation_interval": 25,
    "policy_network_train_every": 25,
    "policy_network_train_steps": 200,
    "advantage_network_train_steps": 100,
    "baseline_network_train_steps": 100,
    "policy_network_layers": [32, 32],
    "advantage_network_layers": [32, 32],
    "baseline_network_layers": [32, 32],
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
    "kuhn_game_value_player_0": KUHN_GAME_VALUE_P0,
    "exploitability_threshold": EXPLOITABILITY_THRESHOLD,
    "output_root": Path("outputs") / "dream_baseline_network_budget_ablation",
}


BASELINE_VARIANT = "baseline_steps_100_exp_baseline"


BUDGET_VARIANTS = [
    {
        "variant_id": "baseline_steps_50",
        "label": "50 baseline steps",
        "baseline_network_train_steps": 50,
        "description": "Reduced baseline-network update budget",
    },
    {
        "variant_id": "baseline_steps_100_exp_baseline",
        "label": "100 baseline steps",
        "baseline_network_train_steps": 100,
        "description": "DREAM baseline configuration",
    },
    {
        "variant_id": "baseline_steps_200",
        "label": "200 baseline steps",
        "baseline_network_train_steps": 200,
        "description": "Increased baseline-network update budget",
    },
    {
        "variant_id": "baseline_steps_500",
        "label": "500 baseline steps",
        "baseline_network_train_steps": 500,
        "description": "High baseline-network update budget",
    },
]


SMOKE_TEST_CONFIG_OVERRIDES = {
    "seeds": SMOKE_TEST_SEEDS,
    "num_iterations": 10,
    "num_traversals": 50,
    "policy_network_train_steps": 20,
    "advantage_network_train_steps": 20,
    "policy_network_train_every": 5,
    "evaluation_interval": 5,
    "output_root": Path("outputs") / "smoke_tests" / "dream_baseline_network_budget_ablation",
}
