"""Configuration checks for Experiment 32 long-node DREAM epsilon comparison."""

import argparse

from dream_poker.constants import DEFAULT_SEEDS_5
from dream_poker.variant_ablation import apply_variant_overrides
from experiments.leduc_poker.dream_architecture_candidate_comparison.config import (
    ADVANTAGE_CANDIDATE_LAYERS,
    POLICY_BASELINE_LAYERS,
)
from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    add_common_arguments,
    config_from_args,
)
from experiments.leduc_poker.dream_candidate_long_node_epsilon_comparison.config import (
    BASELINE_VARIANT,
    EPSILON_020_VARIANT,
    EXPERIMENT_CONFIG,
    LONG_NODE_EPSILON_VARIANTS,
    LONG_NODE_NUM_ITERATIONS,
    TARGET_NODES_TOUCHED,
)


def test_long_node_epsilon_comparison_trains_two_five_seed_arms():
    assert EXPERIMENT_CONFIG["fixed_baseline"]["enabled"] is False
    assert EXPERIMENT_CONFIG["seeds"] == DEFAULT_SEEDS_5
    assert EXPERIMENT_CONFIG["num_iterations"] == LONG_NODE_NUM_ITERATIONS == 7_300
    assert EXPERIMENT_CONFIG["num_traversals"] == 160
    assert EXPERIMENT_CONFIG["target_nodes_touched"] == TARGET_NODES_TOUCHED == 15_000_000
    assert EXPERIMENT_CONFIG["baseline_variant"] == BASELINE_VARIANT
    assert EXPERIMENT_CONFIG["treatment_keys"] == ["epsilon"]
    assert [variant["variant_id"] for variant in LONG_NODE_EPSILON_VARIANTS] == [
        BASELINE_VARIANT,
        EPSILON_020_VARIANT,
    ]
    assert [variant["epsilon"] for variant in LONG_NODE_EPSILON_VARIANTS] == [0.06, 0.20]


def test_long_node_epsilon_comparison_preserves_candidate_architecture():
    for variant in LONG_NODE_EPSILON_VARIANTS:
        config = apply_variant_overrides(EXPERIMENT_CONFIG, variant)
        assert config["policy_network_layers"] == POLICY_BASELINE_LAYERS
        assert config["advantage_network_layers"] == ADVANTAGE_CANDIDATE_LAYERS
        assert config["baseline_network_layers"] == POLICY_BASELINE_LAYERS
        assert config["policy_network_train_steps"] == 100
        assert config["policy_network_train_every"] == 25
        assert config["batch_size_advantage"] == 1024
        assert config["learning_rate"] == 0.003


def test_long_node_epsilon_cli_does_not_enable_fixed_baseline_by_default():
    parser = add_common_arguments(argparse.ArgumentParser())
    config = config_from_args(parser.parse_args([]), EXPERIMENT_CONFIG)
    assert config["fixed_baseline"]["enabled"] is False
