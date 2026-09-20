"""Frozen scientific contract for DREAM Experiment 46."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from dream_poker.constants import DEFAULT_SEEDS_5
from experiments.leduc_poker.dream_frozen_reservoir_distillation_audit.config import GROUPED_CE
from experiments.leduc_poker.dream_paper_aligned_frozen_reservoir_audit.config import (
    PAPER_POLICY_FIT_STEPS,
    TRAINING_CONFIG as EXPERIMENT_45_CONFIG,
)

EXPERIMENT_ID = 46
EXPERIMENT_NAME = "dream_paper_aligned_36h_trajectory"
PRODUCTION_SEEDS = tuple(DEFAULT_SEEDS_5)
SMOKE_SEEDS = (PRODUCTION_SEEDS[0],)

TRAINING_HOURS = 36
TRAINING_SECONDS = TRAINING_HOURS * 60 * 60
CHECKPOINT_INTERVAL_SECONDS = 30 * 60
TARGET_NODES_TOUCHED = 15_000_000
MAX_ITERATIONS = 1_000_000_000
SELECTED_POLICY_ARM = GROUPED_CE
DISTILLATION_SEED_OFFSET = 46_000_000
RAW_RESERVOIR_TIME_ENDPOINT_HOURS = (12,)

TRAINING_CONFIG = deepcopy(EXPERIMENT_45_CONFIG)
TRAINING_CONFIG.update(
    {
        "experiment_name": EXPERIMENT_NAME,
        "algorithm": "Paper-aligned DREAM with deferred grouped-CE average-policy fitting",
        "source_configuration": "Experiment 45 selected paper-aligned configuration",
        "num_iterations": MAX_ITERATIONS,
        "training_time_budget_seconds": TRAINING_SECONDS,
        "checkpoint_interval_seconds": CHECKPOINT_INTERVAL_SECONDS,
        "target_nodes_touched": TARGET_NODES_TOUCHED,
        "seeds": list(PRODUCTION_SEEDS),
        "checkpoint_schedule": (),
        "require_complete_checkpoint_schedule": False,
        "compute_exploitability": False,
        "policy_training_mode": "final_only",
        "fresh_reference_fit_steps": PAPER_POLICY_FIT_STEPS,
        "selected_policy_arm": SELECTED_POLICY_ARM,
        "exclude_checkpoint_time_from_training_budget": True,
        "training_progress_every": 10,
        "checkpoint_directory": "grouped_checkpoints",
        "checkpoint_manifest_filename": "checkpoint_manifest.csv",
        "checkpoint_metrics_filename": "checkpoint_metrics.csv",
    }
)


def build_config(*, smoke: bool = False) -> dict:
    config = deepcopy(TRAINING_CONFIG)
    if smoke:
        config.update(
            {
                "num_iterations": MAX_ITERATIONS,
                "num_traversals": 2,
                "training_time_budget_seconds": 0.01,
                "checkpoint_interval_seconds": 0.001,
                "target_nodes_touched": 1,
                "policy_network_train_steps": 2,
                "fresh_reference_fit_steps": 2,
                "advantage_network_train_steps": 1,
                "baseline_network_train_steps": 1,
                "batch_size_advantage": 2,
                "batch_size_strategy": 2,
                "batch_size_baseline": 2,
                "advantage_memory_capacity": 256,
                "strategy_memory_capacity": 256,
                "baseline_memory_capacity": 256,
                "policy_network_layers": [8, 8],
                "advantage_network_layers": [8, 8],
                "baseline_network_layers": [8, 8],
                "training_progress_every": 1,
                "seeds": list(SMOKE_SEEDS),
                "smoke": True,
            }
        )
    else:
        config["smoke"] = False
    validate_contract(config, smoke=smoke)
    return config


def validate_contract(config: Mapping[str, object], *, smoke: bool) -> None:
    expected_seeds = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    if tuple(map(int, config["seeds"])) != expected_seeds:
        raise ValueError(f"Expected seeds {expected_seeds}, got {config['seeds']}")
    if str(config["experiment_name"]) != EXPERIMENT_NAME:
        raise ValueError("Experiment 46 has the wrong experiment_name")
    if str(config["selected_policy_arm"]) != SELECTED_POLICY_ARM:
        raise ValueError("Experiment 46 requires grouped soft-target cross-entropy")
    if bool(config["compute_exploitability"]):
        raise ValueError("Exploitability must be evaluated after timed training")
    if str(config["policy_training_mode"]) != "final_only":
        raise ValueError("Average-policy fitting must be deferred until after training")
    if not bool(config["exclude_checkpoint_time_from_training_budget"]):
        raise ValueError("Checkpoint time must be excluded from active training")
    if float(config["training_time_budget_seconds"]) <= 0:
        raise ValueError("training_time_budget_seconds must be positive")
    if float(config["checkpoint_interval_seconds"]) <= 0:
        raise ValueError("checkpoint_interval_seconds must be positive")
    if int(config["target_nodes_touched"]) <= 0:
        raise ValueError("target_nodes_touched must be positive")

    if not smoke:
        required = {
            "training_time_budget_seconds": TRAINING_SECONDS,
            "checkpoint_interval_seconds": CHECKPOINT_INTERVAL_SECONDS,
            "target_nodes_touched": TARGET_NODES_TOUCHED,
            "num_traversals": 900,
            "epsilon": 0.50,
            "learning_rate": 0.001,
            "learning_rate_schedule": "constant",
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
            "gradient_clip_norm": 1.0,
            "advantage_network_reinitialize_every_iteration": True,
            "average_strategy_weighting": "linear",
            "policy_network_layers": [32, 32],
            "advantage_network_layers": [128, 128],
            "baseline_network_layers": [32, 32],
        }
        for key, expected in required.items():
            if config.get(key) != expected:
                raise ValueError(f"Experiment 46 field {key} changed: {config.get(key)!r}")


__all__ = [
    "CHECKPOINT_INTERVAL_SECONDS",
    "DISTILLATION_SEED_OFFSET",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "MAX_ITERATIONS",
    "PAPER_POLICY_FIT_STEPS",
    "PRODUCTION_SEEDS",
    "RAW_RESERVOIR_TIME_ENDPOINT_HOURS",
    "SELECTED_POLICY_ARM",
    "SMOKE_SEEDS",
    "TARGET_NODES_TOUCHED",
    "TRAINING_CONFIG",
    "TRAINING_HOURS",
    "TRAINING_SECONDS",
    "build_config",
    "validate_contract",
]
