"""Configuration checks for Experiment 35 900-traversal long-node DREAM run."""

import argparse

from dream_poker.constants import DEFAULT_SEEDS_5
from dream_poker.variant_ablation import apply_variant_overrides
from experiments.leduc_poker.dream_architecture_candidate_comparison.config import (
    ADVANTAGE_CANDIDATE_LAYERS,
    POLICY_BASELINE_LAYERS,
)
from experiments.leduc_poker.dream_candidate_900_traversal_long_node.config import (
    CANDIDATE_900_VARIANT,
    CANDIDATE_900_VARIANTS,
    EXPERIMENT_CONFIG,
    LONG_NODE_NUM_ITERATIONS,
    LONG_NODE_NUM_TRAVERSALS,
    TARGET_NODES_TOUCHED,
)
from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    add_common_arguments,
    config_from_args,
)


def test_900_traversal_long_node_run_uses_exp22_candidate_with_five_seeds():
    assert EXPERIMENT_CONFIG["fixed_baseline"]["enabled"] is False
    assert EXPERIMENT_CONFIG["seeds"] == DEFAULT_SEEDS_5
    assert EXPERIMENT_CONFIG["num_iterations"] == LONG_NODE_NUM_ITERATIONS == 1_300
    assert EXPERIMENT_CONFIG["num_traversals"] == LONG_NODE_NUM_TRAVERSALS == 900
    assert EXPERIMENT_CONFIG["target_nodes_touched"] == TARGET_NODES_TOUCHED == 15_000_000
    assert EXPERIMENT_CONFIG["baseline_variant"] == CANDIDATE_900_VARIANT
    assert EXPERIMENT_CONFIG["treatment_keys"] == ["num_traversals"]
    assert [variant["variant_id"] for variant in CANDIDATE_900_VARIANTS] == [
        CANDIDATE_900_VARIANT,
    ]


def test_900_traversal_long_node_run_preserves_candidate_training_spec():
    config = apply_variant_overrides(EXPERIMENT_CONFIG, CANDIDATE_900_VARIANTS[0])
    assert config["policy_network_layers"] == POLICY_BASELINE_LAYERS
    assert config["advantage_network_layers"] == ADVANTAGE_CANDIDATE_LAYERS
    assert config["baseline_network_layers"] == POLICY_BASELINE_LAYERS
    assert config["policy_network_train_steps"] == 100
    assert config["policy_network_train_every"] == 25
    assert config["advantage_network_train_steps"] == 50
    assert config["baseline_network_train_steps"] == 50
    assert config["baseline_network_train_every"] == 1
    assert config["batch_size_advantage"] == 1024
    assert config["learning_rate"] == 0.003
    assert config["epsilon"] == 0.06
    assert config["average_strategy_weighting"] == "linear"


def test_900_traversal_long_node_cli_does_not_reuse_fixed_baseline_by_default():
    parser = add_common_arguments(argparse.ArgumentParser())
    config = config_from_args(parser.parse_args([]), EXPERIMENT_CONFIG)
    assert config["fixed_baseline"]["enabled"] is False
