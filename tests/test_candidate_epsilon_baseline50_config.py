"""Configuration checks for Experiment 34 epsilon comparison with baseline cadence 50."""

import argparse

from dream_poker.variant_ablation import apply_variant_overrides
from experiments.leduc_poker.dream_architecture_candidate_comparison.config import (
    ADVANTAGE_CANDIDATE_LAYERS,
    POLICY_BASELINE_LAYERS,
)
from experiments.leduc_poker.dream_candidate_epsilon_baseline50_comparison.config import (
    BASELINE_VARIANT,
    EPSILON_020_VARIANT,
    EPSILON_BASELINE50_VARIANTS,
    EXPERIMENT_CONFIG,
)
from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    DEFAULT_SEEDS,
    add_common_arguments,
    config_from_args,
)


def test_epsilon_baseline50_comparison_trains_two_three_seed_arms():
    assert EXPERIMENT_CONFIG["fixed_baseline"]["enabled"] is False
    assert EXPERIMENT_CONFIG["seeds"] == DEFAULT_SEEDS
    assert EXPERIMENT_CONFIG["num_iterations"] == 175
    assert EXPERIMENT_CONFIG["num_traversals"] == 160
    assert EXPERIMENT_CONFIG["baseline_variant"] == BASELINE_VARIANT
    assert EXPERIMENT_CONFIG["treatment_keys"] == [
        "epsilon",
        "baseline_network_train_every",
    ]
    assert [variant["variant_id"] for variant in EPSILON_BASELINE50_VARIANTS] == [
        BASELINE_VARIANT,
        EPSILON_020_VARIANT,
    ]
    assert [variant["epsilon"] for variant in EPSILON_BASELINE50_VARIANTS] == [0.06, 0.20]
    assert [variant["baseline_network_train_every"] for variant in EPSILON_BASELINE50_VARIANTS] == [
        50,
        50,
    ]


def test_epsilon_baseline50_comparison_preserves_experiment_22_candidate_config():
    for variant in EPSILON_BASELINE50_VARIANTS:
        config = apply_variant_overrides(EXPERIMENT_CONFIG, variant)
        assert config["policy_network_layers"] == POLICY_BASELINE_LAYERS
        assert config["advantage_network_layers"] == ADVANTAGE_CANDIDATE_LAYERS
        assert config["baseline_network_layers"] == POLICY_BASELINE_LAYERS
        assert config["policy_network_train_steps"] == 100
        assert config["policy_network_train_every"] == 25
        assert config["advantage_network_train_steps"] == 50
        assert config["baseline_network_train_steps"] == 50
        assert config["baseline_network_train_every"] == 50
        assert config["batch_size_advantage"] == 1024
        assert config["learning_rate"] == 0.003
        assert config["average_strategy_weighting"] == "linear"


def test_epsilon_baseline50_cli_does_not_reuse_fixed_baseline_by_default():
    parser = add_common_arguments(argparse.ArgumentParser())
    config = config_from_args(parser.parse_args([]), EXPERIMENT_CONFIG)
    assert config["fixed_baseline"]["enabled"] is False
