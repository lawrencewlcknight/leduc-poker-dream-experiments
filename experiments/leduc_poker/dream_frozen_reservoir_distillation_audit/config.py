"""Frozen contract for Experiment 44's DREAM distillation audit."""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

from experiments.leduc_poker.dream_final_candidate_checkpoint_head_to_head.config import (
    DEFAULT_CONFIG as EXPERIMENT_43_CONFIG,
)


EXPERIMENT_ID = 44
EXPERIMENT_NAME = "dream_frozen_reservoir_distillation_audit"

# Reuse three of Experiment 43's five seeds so that the training configuration
# and seed labels remain directly comparable with the current DREAM candidate.
PRODUCTION_SEEDS = (1234, 2025, 31415)
SMOKE_SEEDS = (1234,)

TRAINING_HOURS = 12
TRAINING_SECONDS = TRAINING_HOURS * 60 * 60
MAX_ITERATIONS = 1_000_000_000

ROW_MSE = "row_weighted_mse"
ROW_CE = "row_weighted_soft_target_ce"
GROUPED_CE = "grouped_weighted_soft_target_ce"
GROUPED_CE_EXTENDED = "grouped_weighted_soft_target_ce_extended"
ARM_ORDER = (ROW_MSE, ROW_CE, GROUPED_CE, GROUPED_CE_EXTENDED)
ARM_LABELS = {
    ROW_MSE: "Row-wise weighted MSE",
    ROW_CE: "Row-wise weighted soft-target CE",
    GROUPED_CE: "Grouped weighted soft-target CE",
    GROUPED_CE_EXTENDED: "Grouped weighted CE (4x fitting)",
}

EXTENDED_FIT_MULTIPLIER = 4
DISTILLATION_SEED_OFFSET = 44_000_000

TRAINING_CONFIG = deepcopy(EXPERIMENT_43_CONFIG)
TRAINING_CONFIG.update(
    {
        "experiment_name": EXPERIMENT_NAME,
        "algorithm": "Experiment 43 DREAM candidate with frozen-reservoir audit",
        "source_configuration": "Experiment 43 / selected Experiment 38 configuration",
        "num_iterations": MAX_ITERATIONS,
        "training_time_budget_seconds": TRAINING_SECONDS,
        "seeds": list(PRODUCTION_SEEDS),
        # A time-budgeted run has no fixed iteration checkpoint schedule.
        "checkpoint_schedule": (),
        "require_complete_checkpoint_schedule": False,
    }
)


def task_schedule(seeds: Sequence[int] = PRODUCTION_SEEDS) -> tuple[int, ...]:
    return tuple(int(seed) for seed in seeds)


def smoke_config() -> dict:
    config = deepcopy(TRAINING_CONFIG)
    config.update(
        {
            "num_iterations": 8,
            "num_traversals": 4,
            "training_time_budget_seconds": 30,
            "policy_network_train_every": 2,
            "policy_network_train_steps": 1,
            "advantage_network_train_steps": 1,
            "baseline_network_train_steps": 1,
            "baseline_network_train_every": 2,
            "batch_size_advantage": 2,
            "batch_size_strategy": 2,
            "batch_size_baseline": 2,
            "advantage_memory_capacity": 256,
            "strategy_memory_capacity": 256,
            "baseline_memory_capacity": 256,
            "seeds": list(SMOKE_SEEDS),
            "smoke_reference_fit_steps": 2,
        }
    )
    return config


def validate_contract(config: dict, *, smoke: bool) -> None:
    expected_seeds = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    if tuple(map(int, config["seeds"])) != expected_seeds:
        raise ValueError(f"Expected seeds {expected_seeds}, got {config['seeds']}")
    if not smoke and int(config["training_time_budget_seconds"]) != TRAINING_SECONDS:
        raise ValueError("Production training must use the fixed 12-hour budget")
    required = {
        "epsilon": 0.20,
        "average_strategy_weighting": "linear",
    }
    if not smoke:
        required.update(
            {
                "baseline_network_train_every": 50,
                "strategy_memory_capacity": 1_000_000,
            }
        )
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Experiment 43 field {key} changed: {config.get(key)!r}")


__all__ = [
    "ARM_LABELS",
    "ARM_ORDER",
    "DISTILLATION_SEED_OFFSET",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "EXTENDED_FIT_MULTIPLIER",
    "GROUPED_CE",
    "GROUPED_CE_EXTENDED",
    "MAX_ITERATIONS",
    "PRODUCTION_SEEDS",
    "ROW_CE",
    "ROW_MSE",
    "SMOKE_SEEDS",
    "TRAINING_CONFIG",
    "TRAINING_HOURS",
    "TRAINING_SECONDS",
    "smoke_config",
    "task_schedule",
    "validate_contract",
]
