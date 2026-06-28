"""Configuration for the DREAM advantage-target processing ablation."""

from pathlib import Path

from dream_poker.constants import (
    DEFAULT_SEEDS_5,
    EXPLOITABILITY_THRESHOLD,
    LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    LEDUC_GAME_VALUE_P0,
    SMOKE_TEST_SEEDS,
    THESIS_SEEDS_10,
)


EXPERIMENT_CONFIG = {
    "experiment_name": "leduc_poker_dream_target_processing_ablation",
    "game_name": "leduc_poker",
    "algorithm": "DREAM-style OpenSpiel target-processing ablation",
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
    "target_processing": "none",
    "target_clip_value": 1.0,
    "target_standardize_epsilon": 1e-6,
    "output_root": Path("outputs") / "dream_target_processing_ablation",
}


BASELINE_VARIANT = "raw_targets_dream_baseline"


TARGET_PROCESSING_VARIANTS = [
    {
        "variant_id": "raw_targets_dream_baseline",
        "label": "Raw targets",
        "target_processing": "none",
        "target_clip_value": 1.0,
        "description": "Raw DREAM advantage targets",
    },
    {
        "variant_id": "standardized_targets",
        "label": "Standardized targets",
        "target_processing": "standardize",
        "target_clip_value": 1.0,
        "description": "Batch-standardized DREAM advantage targets",
    },
    {
        "variant_id": "clipped_targets",
        "label": "Clipped targets",
        "target_processing": "clip",
        "target_clip_value": 1.0,
        "description": "DREAM advantage targets clipped to [-1, 1]",
    },
    {
        "variant_id": "standardized_clipped_targets",
        "label": "Standardized + clipped targets",
        "target_processing": "standardize_clip",
        "target_clip_value": 1.0,
        "description": "Batch-standardized DREAM advantage targets clipped to [-1, 1]",
    },
]


SMOKE_TEST_CONFIG_OVERRIDES = {
    "seeds": SMOKE_TEST_SEEDS,
    "num_iterations": 10,
    "num_traversals": 50,
    "policy_network_train_steps": 20,
    "advantage_network_train_steps": 20,
    "baseline_network_train_steps": 20,
    "policy_network_train_every": 5,
    "evaluation_interval": 5,
    "output_root": Path("outputs") / "smoke_tests" / "dream_target_processing_ablation",
}
