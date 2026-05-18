"""Configuration for the DREAM learning-rate schedule ablation."""

from pathlib import Path

from dream_poker.constants import (
    DEFAULT_SEEDS_5,
    EXPLOITABILITY_THRESHOLD,
    KUHN_AVERAGE_POLICY_VALUE_TARGET,
    KUHN_GAME_VALUE_P0,
    SMOKE_TEST_SEEDS,
    THESIS_SEEDS_10,
)


EXPERIMENT_CONFIG = {
    "experiment_name": "kuhn_poker_dream_lr_schedule_ablation",
    "game_name": "kuhn_poker",
    "algorithm": "DREAM-style OpenSpiel learning-rate schedule ablation",
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
    "learning_rate_end": 0.0003,
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
    "average_policy_value_target": KUHN_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": EXPLOITABILITY_THRESHOLD,
    "output_root": Path("outputs") / "dream_lr_schedule_ablation",
}


SCHEDULE_CONFIGS = [
    {
        "variant": "constant_baseline_dream",
        "description": "DREAM baseline with constant learning rate 0.003",
        "learning_rate_schedule": "constant",
        "learning_rate_end": EXPERIMENT_CONFIG["learning_rate"],
    },
    {
        "variant": "cosine_decay_to_10pct",
        "description": "Cosine decay from 0.003 to 0.0003 over the training run",
        "learning_rate_schedule": "cosine_decay",
        "learning_rate_end": EXPERIMENT_CONFIG["learning_rate_end"],
    },
]


BASELINE_VARIANT = "constant_baseline_dream"


SMOKE_TEST_CONFIG_OVERRIDES = {
    "seeds": SMOKE_TEST_SEEDS,
    "num_iterations": 10,
    "num_traversals": 50,
    "policy_network_train_steps": 20,
    "advantage_network_train_steps": 20,
    "baseline_network_train_steps": 20,
    "policy_network_train_every": 5,
    "evaluation_interval": 5,
    "output_root": Path("outputs") / "smoke_tests" / "dream_lr_schedule_ablation",
}
