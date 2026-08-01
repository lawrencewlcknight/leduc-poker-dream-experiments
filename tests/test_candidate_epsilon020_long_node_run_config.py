"""Configuration checks for Experiment 42 epsilon-0.20 candidate long-node run."""

import argparse

from dream_poker.constants import DEFAULT_SEEDS_5
from dream_poker.variant_ablation import apply_variant_overrides
from experiments.leduc_poker.dream_architecture_candidate_comparison.config import (
    ADVANTAGE_CANDIDATE_LAYERS,
    POLICY_BASELINE_LAYERS,
)
from experiments.leduc_poker.dream_candidate_epsilon020_long_node_run.config import (
    CANDIDATE_EPSILON020_LONG_NODE_VARIANT,
    CANDIDATE_EPSILON020_LONG_NODE_VARIANTS,
    EXPERIMENT_CONFIG,
    LONG_NODE_EPSILON,
    LONG_NODE_NUM_ITERATIONS,
    LONG_NODE_NUM_TRAVERSALS,
    TARGET_NODES_TOUCHED,
)
from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    add_common_arguments,
    config_from_args,
)


def test_epsilon020_candidate_long_node_run_is_single_five_seed_arm():
    assert EXPERIMENT_CONFIG["fixed_baseline"]["enabled"] is False
    assert EXPERIMENT_CONFIG["seeds"] == DEFAULT_SEEDS_5
    assert EXPERIMENT_CONFIG["num_iterations"] == LONG_NODE_NUM_ITERATIONS == 7_300
    assert EXPERIMENT_CONFIG["num_traversals"] == LONG_NODE_NUM_TRAVERSALS == 160
    assert EXPERIMENT_CONFIG["epsilon"] == LONG_NODE_EPSILON == 0.20
    assert EXPERIMENT_CONFIG["compute_baseline_grad_norm_diagnostics"] is False
    assert EXPERIMENT_CONFIG["target_nodes_touched"] == TARGET_NODES_TOUCHED == 15_000_000
    assert EXPERIMENT_CONFIG["baseline_variant"] == CANDIDATE_EPSILON020_LONG_NODE_VARIANT
    assert EXPERIMENT_CONFIG["treatment_keys"] == ["epsilon", "num_traversals"]
    assert [
        variant["variant_id"] for variant in CANDIDATE_EPSILON020_LONG_NODE_VARIANTS
    ] == [CANDIDATE_EPSILON020_LONG_NODE_VARIANT]


def test_epsilon020_candidate_long_node_run_preserves_experiment_41_spec_except_epsilon():
    config = apply_variant_overrides(
        EXPERIMENT_CONFIG, CANDIDATE_EPSILON020_LONG_NODE_VARIANTS[0]
    )

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
    assert config["average_strategy_weighting"] == "linear"
    assert config["epsilon"] == 0.20


def test_epsilon020_candidate_long_node_run_cli_does_not_enable_fixed_baseline_by_default():
    parser = add_common_arguments(argparse.ArgumentParser())
    config = config_from_args(parser.parse_args([]), EXPERIMENT_CONFIG)
    assert config["fixed_baseline"]["enabled"] is False
