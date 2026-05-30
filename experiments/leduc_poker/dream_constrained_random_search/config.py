"""Configuration for the DREAM constrained solver-parameter random search."""

from pathlib import Path

from dream_poker.constants import (
    EXPLOITABILITY_THRESHOLD,
    LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    LEDUC_GAME_VALUE_P0,
)


EXPERIMENT_NAME = "leduc_poker_dream_constrained_random_search"
MASTER_SEED = 271828


DREAM_BASELINE_CONFIG = {
    "config_label": "dream_baseline",
    "game_name": "leduc_poker",
    "policy_network_layers": (32, 32),
    "advantage_network_layers": (32, 32),
    "baseline_network_layers": (32, 32),
    "num_iterations": 175,
    "num_traversals": 160,
    "learning_rate": 0.003,
    "batch_size_advantage": 1024,
    "batch_size_strategy": 1024,
    "batch_size_baseline": 1024,
    "advantage_memory_capacity": int(1e6),
    "strategy_memory_capacity": int(1e6),
    "baseline_memory_capacity": int(1e6),
    "epsilon": 0.06,
    "advantage_network_train_steps": 50,
    "policy_network_train_steps": 100,
    "baseline_network_train_steps": 50,
    "policy_network_train_every": 25,
    "compute_exploitability": True,
    "isolate_policy_training_rng": True,
    "average_policy_value_target": LEDUC_AVERAGE_POLICY_VALUE_TARGET,
}


DREAM_SEARCH_SPACE = {
    "epsilon": [0.03, 0.06, 0.10, 0.15],
    "num_traversals": [80, 160, 320],
    "learning_rate": [0.001, 0.003, 0.006],
    "policy_network_layers": [(16, 16), (32, 32)],
    "advantage_network_layers": [(16, 16), (32, 32)],
    "baseline_network_layers": [(16, 16), (32, 32)],
    "batch_size_advantage": [512, 1024],
    "batch_size_strategy": [512, 1024],
    "batch_size_baseline": [512, 1024],
    "advantage_memory_capacity": [int(1e5), int(5e5), int(1e6)],
    "strategy_memory_capacity": [int(1e5), int(5e5), int(1e6)],
    "baseline_memory_capacity": [int(1e5), int(5e5), int(1e6)],
    "advantage_network_train_steps": [25, 50, 100],
    "baseline_network_train_steps": [25, 50, 100],
    "policy_network_train_steps": [50, 100, 200],
}


FIXED_PARAMETERS = {
    "policy_network_train_every": DREAM_BASELINE_CONFIG["policy_network_train_every"],
    "compute_exploitability": True,
    "game_name": DREAM_BASELINE_CONFIG["game_name"],
    "isolate_policy_training_rng": True,
    "average_policy_value_target": LEDUC_AVERAGE_POLICY_VALUE_TARGET,
}


EXPERIMENT_CONFIG = {
    "experiment_name": EXPERIMENT_NAME,
    "algorithm": "DREAM-style OpenSpiel constrained solver-parameter random search",
    "baseline_config": DREAM_BASELINE_CONFIG,
    "search_space": DREAM_SEARCH_SPACE,
    "fixed_parameters": FIXED_PARAMETERS,
    "master_seed": MASTER_SEED,
    "screening_num_iterations": 75,
    "confirmation_num_iterations": 175,
    "screening_seeds": [1234],
    "confirmation_seeds": [1234, 2025],
    "optional_confirmation_seeds_5": [1234, 2025, 31415, 27182, 16180],
    "n_random_candidates": 4,
    "n_confirmation_candidates": 2,
    "game_value_player_0": LEDUC_GAME_VALUE_P0,
    "average_policy_value_target": LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": EXPLOITABILITY_THRESHOLD,
    "output_root": Path("outputs") / "dream_constrained_random_search",
}


SMOKE_TEST_CONFIG_OVERRIDES = {
    "screening_num_iterations": 5,
    "confirmation_num_iterations": 5,
    "screening_seeds": [1234],
    "confirmation_seeds": [1234],
    "n_random_candidates": 1,
    "n_confirmation_candidates": 1,
    "output_root": Path("outputs") / "smoke_tests" / "dream_constrained_random_search",
}
