"""Configuration for the DREAM final-only average-policy training ablation."""

from pathlib import Path

from dream_poker.constants import (
    DEFAULT_SEEDS_5,
    EXPLOITABILITY_THRESHOLD,
    KUHN_AVERAGE_POLICY_VALUE_TARGET,
    KUHN_GAME_VALUE_P0,
    SMOKE_TEST_SEEDS,
    THESIS_SEEDS_10,
)


POLICY_TRAINING_VARIANTS = [
    {
        "variant_id": "intermittent_every_25_baseline",
        "label": "Intermittent every 25",
        "policy_training_mode": "intermittent",
        "final_policy_network_train_steps": None,
        "description": (
            "Average-policy network trained every 25 iterations and at final evaluation, "
            "matching the DREAM baseline."
        ),
    },
    {
        "variant_id": "final_only_100_steps",
        "label": "Final only, 100 steps",
        "policy_training_mode": "final_only",
        "final_policy_network_train_steps": "match_single_event",
        "description": (
            "Average-policy network trained only once at the end for the baseline "
            "per-event update budget."
        ),
    },
    {
        "variant_id": "final_only_matched_steps",
        "label": "Final only, matched steps",
        "policy_training_mode": "final_only",
        "final_policy_network_train_steps": "match_intermediate_total",
        "description": (
            "Average-policy network trained only once at the end with the same total "
            "policy-gradient steps as the intermittent baseline."
        ),
    },
]


EXPERIMENT_CONFIG = {
    "experiment_name": "kuhn_poker_dream_final_only_policy_training_ablation",
    "game_name": "kuhn_poker",
    "algorithm": "DREAM-style OpenSpiel final-only average-policy training ablation",
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
    "seeds": DEFAULT_SEEDS_5,
    "optional_thesis_seeds_10": THESIS_SEEDS_10,
    "kuhn_game_value_player_0": KUHN_GAME_VALUE_P0,
    "average_policy_value_target": KUHN_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": EXPLOITABILITY_THRESHOLD,
    "isolate_policy_training_rng": True,
    "output_root": Path("outputs") / "dream_final_only_policy_training_ablation",
    "variants": POLICY_TRAINING_VARIANTS,
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
    "output_root": Path("outputs") / "smoke_tests" / "dream_final_only_policy_training_ablation",
}
