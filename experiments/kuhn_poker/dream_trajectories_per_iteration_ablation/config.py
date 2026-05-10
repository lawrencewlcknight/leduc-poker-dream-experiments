"""Configuration for the DREAM trajectories-per-iteration ablation."""

from pathlib import Path

from dream_poker.constants import (
    DEFAULT_SEEDS_5,
    EXPLOITABILITY_THRESHOLD,
    KUHN_GAME_VALUE_P0,
    SMOKE_TEST_SEEDS,
    THESIS_SEEDS_10,
)


EXPERIMENT_CONFIG = {
    "experiment_name": "kuhn_poker_dream_trajectories_per_iteration_ablation",
    "game_name": "kuhn_poker",
    "algorithm": "DREAM-style OpenSpiel trajectories-per-iteration ablation",
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
    "output_root": Path("outputs") / "dream_trajectories_per_iteration_ablation",
}


BASELINE_VARIANT = "traversals_320_exp_baseline"


TRAJECTORY_VARIANTS = [
    {
        "variant_id": "traversals_160",
        "label": "160 traversals",
        "num_traversals": 160,
        "description": "Half the baseline trajectory count",
    },
    {
        "variant_id": "traversals_320_exp_baseline",
        "label": "320 traversals",
        "num_traversals": 320,
        "description": "DREAM baseline trajectory count",
    },
    {
        "variant_id": "traversals_640",
        "label": "640 traversals",
        "num_traversals": 640,
        "description": "Double the baseline trajectory count",
    },
]


SMOKE_TEST_CONFIG_OVERRIDES = {
    "seeds": SMOKE_TEST_SEEDS,
    "num_iterations": 10,
    "policy_network_train_steps": 20,
    "advantage_network_train_steps": 20,
    "baseline_network_train_steps": 20,
    "policy_network_train_every": 5,
    "evaluation_interval": 5,
    "output_root": Path("outputs") / "smoke_tests" / "dream_trajectories_per_iteration_ablation",
}
