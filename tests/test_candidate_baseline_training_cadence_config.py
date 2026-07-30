"""Configuration checks for Experiment 32 baseline-training cadence ablation."""

import argparse

from dream_poker.constants import DEFAULT_SEEDS_5
from dream_poker.variant_ablation import apply_variant_overrides
from experiments.leduc_poker.dream_architecture_candidate_comparison.config import (
    ADVANTAGE_CANDIDATE_LAYERS,
    POLICY_BASELINE_LAYERS,
)
from experiments.leduc_poker.dream_candidate_baseline_training_cadence_ablation.config import (
    BASELINE_TRAINING_CADENCE_VARIANTS,
    BASELINE_VARIANT,
    EXPERIMENT_CONFIG,
)
from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    add_common_arguments,
    config_from_args,
    ensure_fixed_baseline_variant,
    load_fixed_baseline_outputs,
)


def test_baseline_training_cadence_reuses_exp22_comparator_by_default():
    assert EXPERIMENT_CONFIG["fixed_baseline"]["enabled"] is True
    assert EXPERIMENT_CONFIG["fixed_baseline"]["source_variant"] == (
        "candidate_advantage_2x128_policy_baseline_2x32"
    )
    assert EXPERIMENT_CONFIG["seeds"] == DEFAULT_SEEDS_5
    assert EXPERIMENT_CONFIG["baseline_variant"] == BASELINE_VARIANT
    assert EXPERIMENT_CONFIG["treatment_keys"] == ["baseline_network_train_every"]
    assert [variant["baseline_network_train_every"] for variant in BASELINE_TRAINING_CADENCE_VARIANTS] == [
        1,
        5,
        10,
        25,
        50,
    ]


def test_baseline_training_cadence_preserves_candidate_config_except_cadence():
    for variant in BASELINE_TRAINING_CADENCE_VARIANTS:
        config = apply_variant_overrides(EXPERIMENT_CONFIG, variant)
        assert config["policy_network_layers"] == POLICY_BASELINE_LAYERS
        assert config["advantage_network_layers"] == ADVANTAGE_CANDIDATE_LAYERS
        assert config["baseline_network_layers"] == POLICY_BASELINE_LAYERS
        assert config["policy_network_train_steps"] == 100
        assert config["policy_network_train_every"] == 25
        assert config["advantage_network_train_steps"] == 50
        assert config["baseline_network_train_steps"] == 50
        assert config["batch_size_advantage"] == 1024
        assert config["learning_rate"] == 0.003
        assert config["epsilon"] == 0.06


def test_baseline_training_cadence_cli_can_disable_fixed_comparator():
    parser = add_common_arguments(argparse.ArgumentParser())

    default_config = config_from_args(parser.parse_args([]), EXPERIMENT_CONFIG)
    assert default_config["fixed_baseline"]["enabled"] is True

    retrain_config = config_from_args(
        parser.parse_args(["--train-baseline"]),
        EXPERIMENT_CONFIG,
    )
    assert retrain_config["fixed_baseline"]["enabled"] is False


def test_baseline_training_cadence_cli_exposes_cadence_override():
    parser = add_common_arguments(argparse.ArgumentParser())
    config = config_from_args(
        parser.parse_args(["--baseline-network-train-every", "7"]),
        EXPERIMENT_CONFIG,
    )
    assert config["baseline_network_train_every"] == 7


def test_baseline_training_cadence_loader_remaps_exp22_with_cadence_key(tmp_path):
    config = dict(EXPERIMENT_CONFIG)
    config["seeds"] = [1234]
    variants = ensure_fixed_baseline_variant(
        config,
        BASELINE_TRAINING_CADENCE_VARIANTS[1:],
    )

    curves, summaries = load_fixed_baseline_outputs(
        config,
        variants,
        config["treatment_keys"],
        tmp_path,
    )

    assert len(curves) == 1
    assert len(summaries) == 1
    assert set(curves[0]["variant"]) == {BASELINE_VARIANT}
    assert curves[0]["baseline_network_train_every"].nunique() == 1
    assert curves[0]["baseline_network_train_every"].iloc[0] == 1
    assert summaries[0]["baseline_network_train_every"] == 1
    assert summaries[0]["baseline_reused_from_artifact"] is True
