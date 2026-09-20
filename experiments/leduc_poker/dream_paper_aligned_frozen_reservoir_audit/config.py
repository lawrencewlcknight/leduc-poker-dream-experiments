"""Frozen contract for Experiment 45's paper-aligned DREAM run."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Sequence

from experiments.leduc_poker.dream_frozen_reservoir_distillation_audit.config import (
    ARM_LABELS,
    ARM_ORDER,
    EXTENDED_FIT_MULTIPLIER,
    GROUPED_CE,
    GROUPED_CE_EXTENDED,
    ROW_CE,
    ROW_MSE,
    TRAINING_CONFIG as EXPERIMENT_44_CONFIG,
)


EXPERIMENT_ID = 45
EXPERIMENT_NAME = "dream_paper_aligned_frozen_reservoir_audit"

# Paired directly with Experiment 44.
PRODUCTION_SEEDS = (1234, 2025, 31415)
SMOKE_SEEDS = (1234,)

TRAINING_HOURS = 12
TRAINING_SECONDS = TRAINING_HOURS * 60 * 60
MAX_ITERATIONS = 1_000_000_000
PAPER_POLICY_FIT_STEPS = 4_000
DISTILLATION_SEED_OFFSET = 45_000_000

TRAINING_CONFIG = deepcopy(EXPERIMENT_44_CONFIG)
TRAINING_CONFIG.update(
    {
        "experiment_name": EXPERIMENT_NAME,
        "algorithm": "Experiment 44 DREAM candidate with paper-aligned training budget",
        "source_configuration": "Experiment 44 with DREAM-paper numerical hyperparameters",
        "num_iterations": MAX_ITERATIONS,
        "training_time_budget_seconds": TRAINING_SECONDS,
        "seeds": list(PRODUCTION_SEEDS),
        "checkpoint_schedule": (),
        "require_complete_checkpoint_schedule": False,
        # Paper-aligned numerical training configuration. Network widths and the
        # OpenSpiel state representation remain those of Experiment 44.
        "num_traversals": 900,
        "epsilon": 0.50,
        "learning_rate": 0.001,
        "learning_rate_schedule": "constant",
        "batch_size_advantage": 2_048,
        "batch_size_baseline": 512,
        "batch_size_strategy": 2_048,
        "advantage_memory_capacity": 2_000_000,
        "baseline_memory_capacity": 200_000,
        # The existing solver has one shared policy reservoir. Four million
        # total slots match the paper's two-million-per-player allocation in
        # this balanced two-player experiment without changing the policy
        # representation being compared with Experiment 44.
        "strategy_memory_capacity": 4_000_000,
        "advantage_network_train_steps": 3_000,
        "baseline_network_train_steps": 1_000,
        "baseline_network_train_every": 1,
        "policy_network_train_steps": PAPER_POLICY_FIT_STEPS,
        "fresh_reference_fit_steps": PAPER_POLICY_FIT_STEPS,
        "gradient_clip_norm": 1.0,
        "advantage_network_reinitialize_every_iteration": True,
        # Average-policy fitting is deliberately excluded from the active
        # 12-hour collection budget. Each frozen-reservoir arm is initialised
        # afresh and fitted for the paper's 4,000-minibatch budget afterwards.
        "policy_training_mode": "final_only",
        "average_strategy_weighting": "linear",
        "output_root": Path("outputs") / EXPERIMENT_NAME,
    }
)


def task_schedule(seeds: Sequence[int] = PRODUCTION_SEEDS) -> tuple[int, ...]:
    return tuple(int(seed) for seed in seeds)


def smoke_config() -> dict:
    config = deepcopy(TRAINING_CONFIG)
    config.update(
        {
            "num_iterations": 3,
            "num_traversals": 4,
            "training_time_budget_seconds": 30,
            "policy_network_train_steps": 1,
            "fresh_reference_fit_steps": 2,
            "advantage_network_train_steps": 1,
            "baseline_network_train_steps": 1,
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
        "epsilon": 0.50,
        "learning_rate": 0.001,
        "learning_rate_schedule": "constant",
        "average_strategy_weighting": "linear",
        "gradient_clip_norm": 1.0,
        "advantage_network_reinitialize_every_iteration": True,
        "policy_training_mode": "final_only",
    }
    if not smoke:
        required.update(
            {
                "num_traversals": 900,
                "batch_size_advantage": 2_048,
                "batch_size_baseline": 512,
                "batch_size_strategy": 2_048,
                "advantage_memory_capacity": 2_000_000,
                "baseline_memory_capacity": 200_000,
                "strategy_memory_capacity": 4_000_000,
                "advantage_network_train_steps": 3_000,
                "baseline_network_train_steps": 1_000,
                "baseline_network_train_every": 1,
                "policy_network_train_steps": PAPER_POLICY_FIT_STEPS,
                "fresh_reference_fit_steps": PAPER_POLICY_FIT_STEPS,
                # Architecture is intentionally held at Experiment 44.
                "policy_network_layers": [32, 32],
                "advantage_network_layers": [128, 128],
                "baseline_network_layers": [32, 32],
            }
        )
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Paper-aligned field {key} changed: {config.get(key)!r}")


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
    "PAPER_POLICY_FIT_STEPS",
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
