"""Configuration for the DREAM checkpoint-stability experiment."""

from pathlib import Path

from dream_poker.constants import (
    DEFAULT_SEEDS_5,
    EXPLOITABILITY_THRESHOLD,
    KUHN_AVERAGE_POLICY_VALUE_TARGET,
    KUHN_GAME_VALUE_P0,
    SMOKE_TEST_SEEDS,
    THESIS_SEEDS_10,
)


CHECKPOINT_SCHEDULE = [25, 50, 100, 200, 300, 400, 500]


EXPERIMENT_CONFIG = {
    "experiment_name": "kuhn_poker_dream_checkpoint_stability",
    "game_name": "kuhn_poker",
    "algorithm": "DREAM-style OpenSpiel checkpoint-stability experiment",
    "num_iterations": 500,
    "num_traversals": 320,
    "checkpoint_schedule": CHECKPOINT_SCHEDULE,
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
    "seeds": DEFAULT_SEEDS_5,
    "optional_thesis_seeds_10": THESIS_SEEDS_10,
    "kuhn_game_value_player_0": KUHN_GAME_VALUE_P0,
    "average_policy_value_target": KUHN_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": EXPLOITABILITY_THRESHOLD,
    "head_to_head_equivalence_epsilon": 1e-4,
    "output_root": Path("outputs") / "dream_checkpoint_stability",
    "save_full_solver_checkpoints": False,
}


SMOKE_TEST_CONFIG_OVERRIDES = {
    "seeds": SMOKE_TEST_SEEDS,
    "num_iterations": 10,
    "num_traversals": 50,
    "checkpoint_schedule": [5, 10],
    "policy_network_train_steps": 20,
    "advantage_network_train_steps": 20,
    "baseline_network_train_steps": 20,
    "policy_network_train_every": 5,
    "evaluation_interval": 5,
    "output_root": Path("outputs") / "smoke_tests" / "dream_checkpoint_stability",
}
