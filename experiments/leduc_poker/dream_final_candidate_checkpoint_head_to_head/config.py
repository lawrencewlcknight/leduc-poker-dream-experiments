"""Configuration for Experiment 43: DREAM final-candidate checkpoint head-to-head."""

from pathlib import Path

from dream_poker.constants import (
    DEFAULT_SEEDS_5,
    EXPLOITABILITY_THRESHOLD,
    LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    LEDUC_GAME_VALUE_P0,
    SMOKE_TEST_SEEDS,
    THESIS_SEEDS_10,
)
from experiments.leduc_poker.dream_architecture_candidate_comparison.config import (
    ADVANTAGE_CANDIDATE_LAYERS,
    POLICY_BASELINE_LAYERS,
)


DEFAULT_SEEDS = DEFAULT_SEEDS_5
TARGET_NODES_TOUCHED = 15_000_000
TARGET_NUM_ITERATIONS = 7_500
TARGET_NUM_TRAVERSALS = 160
BASELINE_NETWORK_TRAIN_EVERY = 50

# Five snapshots at 20%, 40%, 60%, 80%, and 100% of the long-node budget.
CHECKPOINT_SCHEDULE = (1_500, 3_000, 4_500, 6_000, 7_500)


DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_dream_final_candidate_checkpoint_head_to_head",
    "game_name": "leduc_poker",
    "algorithm": "DREAM-style OpenSpiel final-candidate temporal checkpoint head-to-head",
    "source_configuration": "Experiment 38 high-exploration sparse-baseline arm",
    "num_iterations": TARGET_NUM_ITERATIONS,
    "num_traversals": TARGET_NUM_TRAVERSALS,
    "target_nodes_touched": TARGET_NODES_TOUCHED,
    "checkpoint_schedule": CHECKPOINT_SCHEDULE,
    "evaluation_interval": 25,
    "policy_network_train_every": 25,
    "policy_network_train_steps": 100,
    "advantage_network_train_steps": 50,
    "baseline_network_train_steps": 50,
    "baseline_network_train_every": BASELINE_NETWORK_TRAIN_EVERY,
    "policy_network_layers": list(POLICY_BASELINE_LAYERS),
    "advantage_network_layers": list(ADVANTAGE_CANDIDATE_LAYERS),
    "baseline_network_layers": list(POLICY_BASELINE_LAYERS),
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
    "epsilon": 0.20,
    "compute_exploitability": True,
    "compute_baseline_grad_norm_diagnostics": False,
    "isolate_policy_training_rng": True,
    "average_strategy_weighting": "linear",
    "seeds": list(DEFAULT_SEEDS),
    "optional_thesis_seeds_10": THESIS_SEEDS_10,
    "game_value_player_0": LEDUC_GAME_VALUE_P0,
    "average_policy_value_target": LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": EXPLOITABILITY_THRESHOLD,
    "head_to_head_equivalence_epsilon": 1e-3,
    "temporal_x_axis": "nodes_touched",
    "require_complete_checkpoint_schedule": True,
    "output_root": Path("outputs") / "dream_final_candidate_checkpoint_head_to_head",
}


SMOKE_TEST_CONFIG_OVERRIDES = {
    "seeds": SMOKE_TEST_SEEDS[:1],
    "num_iterations": 10,
    "num_traversals": 4,
    "checkpoint_schedule": (2, 4, 6, 8, 10),
    "policy_network_train_every": 2,
    "evaluation_interval": 2,
    "policy_network_train_steps": 1,
    "advantage_network_train_steps": 1,
    "baseline_network_train_steps": 1,
    "baseline_network_train_every": 2,
    "batch_size_advantage": 2,
    "batch_size_strategy": 2,
    "batch_size_baseline": 2,
    "advantage_memory_capacity": 256,
    "strategy_memory_capacity": 256,
    "baseline_memory_capacity": 256,
    "output_root": Path("outputs") / "smoke_tests" / "dream_final_candidate_checkpoint_head_to_head",
}


__all__ = [
    "BASELINE_NETWORK_TRAIN_EVERY",
    "CHECKPOINT_SCHEDULE",
    "DEFAULT_CONFIG",
    "DEFAULT_SEEDS",
    "SMOKE_TEST_CONFIG_OVERRIDES",
    "TARGET_NODES_TOUCHED",
    "TARGET_NUM_ITERATIONS",
    "TARGET_NUM_TRAVERSALS",
]
