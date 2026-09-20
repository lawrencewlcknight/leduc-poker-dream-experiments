"""Run Experiment 45: paper-aligned time-budgeted DREAM audit."""

from __future__ import annotations

from experiments.leduc_poker.dream_frozen_reservoir_distillation_audit import run as base

from . import config


def _install_contract() -> None:
    """Apply Experiment 45's frozen contract to the shared audit engine."""
    for name in (
        "ARM_LABELS",
        "ARM_ORDER",
        "DISTILLATION_SEED_OFFSET",
        "EXPERIMENT_ID",
        "EXPERIMENT_NAME",
        "EXTENDED_FIT_MULTIPLIER",
        "GROUPED_CE_EXTENDED",
        "PRODUCTION_SEEDS",
        "ROW_MSE",
        "TRAINING_CONFIG",
        "smoke_config",
        "task_schedule",
        "validate_contract",
    ):
        setattr(base, name, getattr(config, name))


_install_contract()

task_name = base.task_name
run_worker = base.run_worker
run_aggregate = base.run_aggregate


def main() -> None:
    _install_contract()
    base.main()


if __name__ == "__main__":
    main()
