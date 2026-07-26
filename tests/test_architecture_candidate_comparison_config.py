"""Configuration checks for Experiment 22 architecture-candidate comparison."""

from dream_poker.constants import DEFAULT_SEEDS_5
from dream_poker.variant_ablation import apply_variant_overrides
from experiments.leduc_poker.dream_architecture_candidate_comparison.config import (
    ADVANTAGE_CANDIDATE_LAYERS,
    ARCHITECTURE_CANDIDATE_VARIANTS,
    BASELINE_VARIANT,
    CANDIDATE_VARIANT,
    EXPERIMENT_CONFIG,
    POLICY_BASELINE_LAYERS,
)


def variants_by_id():
    return {variant["variant_id"]: variant for variant in ARCHITECTURE_CANDIDATE_VARIANTS}


def test_architecture_candidate_comparison_uses_five_baseline_seeds():
    assert EXPERIMENT_CONFIG["seeds"] == DEFAULT_SEEDS_5


def test_architecture_candidate_variants_are_configured():
    assert BASELINE_VARIANT == "baseline_all_2x32"
    assert CANDIDATE_VARIANT == "candidate_advantage_2x128_policy_baseline_2x32"
    assert list(variants_by_id()) == [BASELINE_VARIANT, CANDIDATE_VARIANT]
    assert EXPERIMENT_CONFIG["baseline_variant"] == BASELINE_VARIANT
    assert EXPERIMENT_CONFIG["candidate_variant"] == CANDIDATE_VARIANT


def test_candidate_changes_only_advantage_network_capacity():
    variants = variants_by_id()
    baseline = apply_variant_overrides(EXPERIMENT_CONFIG, variants[BASELINE_VARIANT])
    candidate = apply_variant_overrides(EXPERIMENT_CONFIG, variants[CANDIDATE_VARIANT])

    assert baseline["policy_network_layers"] == POLICY_BASELINE_LAYERS
    assert baseline["advantage_network_layers"] == POLICY_BASELINE_LAYERS
    assert baseline["baseline_network_layers"] == POLICY_BASELINE_LAYERS

    assert candidate["policy_network_layers"] == POLICY_BASELINE_LAYERS
    assert candidate["advantage_network_layers"] == ADVANTAGE_CANDIDATE_LAYERS
    assert candidate["baseline_network_layers"] == POLICY_BASELINE_LAYERS

    unchanged_keys = {
        "num_iterations",
        "num_traversals",
        "policy_network_train_every",
        "policy_network_train_steps",
        "advantage_network_train_steps",
        "baseline_network_train_steps",
        "learning_rate",
        "batch_size_advantage",
        "batch_size_strategy",
        "batch_size_baseline",
        "epsilon",
        "advantage_memory_capacity",
        "strategy_memory_capacity",
        "baseline_memory_capacity",
    }
    for key in unchanged_keys:
        assert candidate[key] == baseline[key]
