"""Configuration for Experiment 21's parallel-equivalence ablation."""

from __future__ import annotations

import copy
from pathlib import Path

from experiments.leduc_poker.dream_network_size_ablation.config import (
    EXPERIMENT_CONFIG as NETWORK_EXPERIMENT_CONFIG,
)


DEFAULT_SEEDS = [1234, 2025, 31415]
BASELINE_VARIANT_ID = "dream_3x64_sequential"
PARALLEL_VARIANT_ID = "dream_3x64_ray_parallel"
PARALLEL_NUM_WORKERS = 3
BEST_DREAM_NETWORK_LAYERS = [64, 64, 64]

# Absolute final-metric margins for practical equivalence. These are declared
# before running the comparison and mirror the ESCHER Experiment 40 convention.
FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN = 0.05
FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN = 0.02


VARIANTS = [
    {
        "variant_id": BASELINE_VARIANT_ID,
        "label": "DREAM 3x64 sequential",
        "description": "Current DREAM learner with sequential traversal collection.",
        "execution_backend": "sequential",
        "parallel_num_workers": 1,
    },
    {
        "variant_id": PARALLEL_VARIANT_ID,
        "label": "DREAM 3x64 Ray parallel (3 workers)",
        "description": (
            "Current DREAM learner with traversal collection partitioned over "
            "three Ray actors and one central learner."
        ),
        "execution_backend": "ray_parallel",
        "parallel_num_workers": PARALLEL_NUM_WORKERS,
    },
]


EXPERIMENT_CONFIG = copy.deepcopy(NETWORK_EXPERIMENT_CONFIG)
EXPERIMENT_CONFIG.update(
    {
        "experiment_name": "leduc_poker_dream_parallel_equivalence_ablation",
        "algorithm": "DREAM-style OpenSpiel sequential/parallel equivalence ablation",
        "seeds": list(DEFAULT_SEEDS),
        "policy_network_layers": list(BEST_DREAM_NETWORK_LAYERS),
        "advantage_network_layers": list(BEST_DREAM_NETWORK_LAYERS),
        "baseline_network_layers": list(BEST_DREAM_NETWORK_LAYERS),
        "network_treatment": "all_3x64_best_documented",
        "network_architecture": "3x64",
        "network_depth": 3,
        "network_max_width": 64,
        "network_hidden_units": 192,
        "execution_backend": "sequential",
        "parallel_num_workers": PARALLEL_NUM_WORKERS,
        "parallel_ray_address": None,
        "parallel_log_to_driver": False,
        "baseline_variant": BASELINE_VARIANT_ID,
        "ablation_variants": tuple(VARIANTS),
        "plot_prefix": "dream_parallel_equivalence",
        "plot_title": "DREAM Sequential versus Ray-Parallel",
        "output_root": Path("outputs") / "dream_parallel_equivalence_ablation",
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = {
    "seeds": [1234],
    "num_iterations": 3,
    "num_traversals": 4,
    "policy_network_train_steps": 1,
    "advantage_network_train_steps": 1,
    "baseline_network_train_steps": 1,
    "policy_network_train_every": 1,
    "evaluation_interval": 1,
    "batch_size_advantage": 1,
    "batch_size_strategy": 1,
    "batch_size_baseline": 1,
    "output_root": Path("outputs") / "smoke_tests" / "dream_parallel_equivalence_ablation",
}
