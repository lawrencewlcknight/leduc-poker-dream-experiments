"""Configuration checks for Experiment 23 candidate epsilon ablation."""

import argparse

from dream_poker.variant_ablation import apply_variant_overrides
from experiments.leduc_poker.dream_candidate_epsilon_exploration_ablation.config import (
    ADVANTAGE_CANDIDATE_LAYERS,
    BASELINE_VARIANT,
    CANDIDATE_EPSILON_VARIANTS,
    DEFAULT_SEEDS,
    EXPERIMENT_CONFIG,
    POLICY_BASELINE_LAYERS,
    PREVIOUS_BEST_EPSILON_VARIANT,
)
from experiments.leduc_poker.dream_epsilon_exploration_ablation.run import (
    add_common_arguments,
    config_from_args,
)


def variants_by_id():
    return {variant["variant_id"]: variant for variant in CANDIDATE_EPSILON_VARIANTS}


def test_candidate_epsilon_ablation_uses_three_matched_screening_seeds():
    assert DEFAULT_SEEDS == [1234, 2025, 31415]
    assert EXPERIMENT_CONFIG["seeds"] == DEFAULT_SEEDS
    assert EXPERIMENT_CONFIG["fixed_baseline"]["enabled"] is True
    assert EXPERIMENT_CONFIG["fixed_baseline"]["source_variant"] == (
        "candidate_advantage_2x128_policy_baseline_2x32"
    )
    assert (
        EXPERIMENT_CONFIG["fixed_baseline"]["source_output_dir"]
        / EXPERIMENT_CONFIG["fixed_baseline"]["curves_filename"]
    ).exists()
    assert (
        EXPERIMENT_CONFIG["fixed_baseline"]["source_output_dir"]
        / EXPERIMENT_CONFIG["fixed_baseline"]["summary_filename"]
    ).exists()


def test_candidate_epsilon_grid_extends_above_previous_best():
    variants = variants_by_id()
    assert BASELINE_VARIANT == "candidate_epsilon_006_baseline"
    assert PREVIOUS_BEST_EPSILON_VARIANT == "candidate_epsilon_010"
    assert list(variants) == [
        "candidate_epsilon_006_baseline",
        "candidate_epsilon_010",
        "candidate_epsilon_015",
        "candidate_epsilon_020",
    ]
    assert [variant["epsilon"] for variant in CANDIDATE_EPSILON_VARIANTS] == [
        0.06,
        0.10,
        0.15,
        0.20,
    ]
    assert EXPERIMENT_CONFIG["baseline_variant"] == BASELINE_VARIANT


def test_candidate_epsilon_ablation_uses_architecture_selected_baseline():
    variants = variants_by_id()
    for variant in variants.values():
        config = apply_variant_overrides(EXPERIMENT_CONFIG, variant)
        assert config["policy_network_layers"] == POLICY_BASELINE_LAYERS
        assert config["advantage_network_layers"] == ADVANTAGE_CANDIDATE_LAYERS
        assert config["baseline_network_layers"] == POLICY_BASELINE_LAYERS


def test_candidate_epsilon_cli_reuses_fixed_baseline_by_default():
    parser = add_common_arguments(argparse.ArgumentParser())

    default_config = config_from_args(parser.parse_args([]), EXPERIMENT_CONFIG)
    assert default_config["fixed_baseline"]["enabled"] is True

    retrain_config = config_from_args(
        parser.parse_args(["--train-baseline"]),
        EXPERIMENT_CONFIG,
    )
    assert retrain_config["fixed_baseline"]["enabled"] is False
