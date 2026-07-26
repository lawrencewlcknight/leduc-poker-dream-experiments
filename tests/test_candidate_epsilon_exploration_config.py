"""Configuration checks for Experiment 23 candidate epsilon ablation."""

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


def variants_by_id():
    return {variant["variant_id"]: variant for variant in CANDIDATE_EPSILON_VARIANTS}


def test_candidate_epsilon_ablation_uses_three_matched_screening_seeds():
    assert DEFAULT_SEEDS == [1234, 2025, 31415]
    assert EXPERIMENT_CONFIG["seeds"] == DEFAULT_SEEDS


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
