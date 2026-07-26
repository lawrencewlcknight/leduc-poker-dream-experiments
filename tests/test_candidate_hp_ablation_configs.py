"""Configuration checks for candidate-baseline DREAM HP ablations."""

from __future__ import annotations

import importlib

import pytest

from dream_poker.variant_ablation import apply_variant_overrides
from experiments.leduc_poker.dream_architecture_candidate_comparison.config import (
    ADVANTAGE_CANDIDATE_LAYERS,
    POLICY_BASELINE_LAYERS,
)
from experiments.leduc_poker.dream_candidate_hp_ablation_common import DEFAULT_EPSILON, DEFAULT_SEEDS


EXPERIMENT_SPECS = [
    (
        "experiments.leduc_poker.dream_candidate_traversal_budget_ablation.config",
        "TRAVERSAL_BUDGET_VARIANTS",
        "candidate_traversals_160_baseline",
        "num_traversals",
        [160, 320, 480],
    ),
    (
        "experiments.leduc_poker.dream_candidate_strategy_replay_capacity_ablation.config",
        "STRATEGY_REPLAY_CAPACITY_VARIANTS",
        "candidate_strategy_memory_1m_baseline",
        "strategy_memory_capacity",
        [int(1e6), int(5e5), int(1e5)],
    ),
    (
        "experiments.leduc_poker.dream_candidate_baseline_replay_capacity_ablation.config",
        "BASELINE_REPLAY_CAPACITY_VARIANTS",
        "candidate_baseline_memory_1m_baseline",
        "baseline_memory_capacity",
        [int(1e6), int(5e5), int(1e5)],
    ),
    (
        "experiments.leduc_poker.dream_candidate_policy_extraction_budget_ablation.config",
        "POLICY_EXTRACTION_BUDGET_VARIANTS",
        "candidate_policy_steps_100_baseline",
        "policy_network_train_steps",
        [100, 50, 200, 400],
    ),
    (
        "experiments.leduc_poker.dream_candidate_policy_extraction_cadence_ablation.config",
        "POLICY_EXTRACTION_CADENCE_VARIANTS",
        "candidate_policy_every_25_baseline",
        "policy_network_train_every",
        [25, 10, 50],
    ),
    (
        "experiments.leduc_poker.dream_candidate_advantage_fitting_steps_ablation.config",
        "ADVANTAGE_FITTING_STEPS_VARIANTS",
        "candidate_advantage_steps_50_baseline",
        "advantage_network_train_steps",
        [50, 25, 100, 200],
    ),
    (
        "experiments.leduc_poker.dream_candidate_advantage_batch_size_ablation.config",
        "ADVANTAGE_BATCH_SIZE_VARIANTS",
        "candidate_advantage_batch_1024_baseline",
        "batch_size_advantage",
        [1024, 512, 2048],
    ),
    (
        "experiments.leduc_poker.dream_candidate_constant_learning_rate_ablation.config",
        "CONSTANT_LEARNING_RATE_VARIANTS",
        "candidate_learning_rate_0_003_baseline",
        "learning_rate",
        [0.003, 0.001, 0.006],
    ),
]


@pytest.mark.parametrize(
    "module_name,variants_name,baseline_variant,treatment_key,expected_values",
    EXPERIMENT_SPECS,
)
def test_candidate_hp_ablation_metadata(
    module_name: str,
    variants_name: str,
    baseline_variant: str,
    treatment_key: str,
    expected_values: list,
):
    module = importlib.import_module(module_name)
    config = module.EXPERIMENT_CONFIG
    variants = getattr(module, variants_name)

    assert config["seeds"] == DEFAULT_SEEDS
    assert config["epsilon"] == DEFAULT_EPSILON
    assert config["average_strategy_weighting"] == "linear"
    assert config["policy_network_layers"] == POLICY_BASELINE_LAYERS
    assert config["advantage_network_layers"] == ADVANTAGE_CANDIDATE_LAYERS
    assert config["baseline_network_layers"] == POLICY_BASELINE_LAYERS
    assert config["baseline_variant"] == baseline_variant
    assert config["treatment_keys"] == [treatment_key]
    assert [variant["variant_id"] for variant in variants][0] == baseline_variant
    assert [variant[treatment_key] for variant in variants] == expected_values

    for variant in variants:
        variant_config = apply_variant_overrides(config, variant)
        assert variant_config[treatment_key] == variant[treatment_key]
        assert variant_config["policy_network_layers"] == POLICY_BASELINE_LAYERS
        assert variant_config["advantage_network_layers"] == ADVANTAGE_CANDIDATE_LAYERS
        assert variant_config["baseline_network_layers"] == POLICY_BASELINE_LAYERS
