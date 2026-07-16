"""Configuration checks for Experiment 21 parallel equivalence."""

from dream_poker.variant_ablation import apply_variant_overrides
from experiments.leduc_poker.dream_parallel_equivalence_ablation.config import (
    BASELINE_VARIANT_ID,
    BEST_DREAM_NETWORK_LAYERS,
    DEFAULT_SEEDS,
    EXPERIMENT_CONFIG,
    PARALLEL_NUM_WORKERS,
    PARALLEL_VARIANT_ID,
    VARIANTS,
)


def variants_by_id():
    return {variant["variant_id"]: variant for variant in VARIANTS}


def test_parallel_equivalence_uses_three_paired_seeds():
    assert DEFAULT_SEEDS == [1234, 2025, 31415]
    assert EXPERIMENT_CONFIG["seeds"] == DEFAULT_SEEDS


def test_parallel_equivalence_uses_best_documented_3x64_config():
    assert BEST_DREAM_NETWORK_LAYERS == [64, 64, 64]
    for key in [
        "policy_network_layers",
        "advantage_network_layers",
        "baseline_network_layers",
    ]:
        assert EXPERIMENT_CONFIG[key] == BEST_DREAM_NETWORK_LAYERS


def test_parallel_arm_changes_only_execution_metadata():
    variants = variants_by_id()
    sequential = apply_variant_overrides(EXPERIMENT_CONFIG, variants[BASELINE_VARIANT_ID])
    parallel = apply_variant_overrides(EXPERIMENT_CONFIG, variants[PARALLEL_VARIANT_ID])

    execution_keys = {
        "variant_id",
        "label",
        "description",
        "execution_backend",
        "parallel_num_workers",
    }
    for key, value in sequential.items():
        if key not in execution_keys:
            assert parallel[key] == value

    assert sequential["execution_backend"] == "sequential"
    assert sequential["parallel_num_workers"] == 1
    assert parallel["execution_backend"] == "ray_parallel"
    assert parallel["parallel_num_workers"] == PARALLEL_NUM_WORKERS == 3
