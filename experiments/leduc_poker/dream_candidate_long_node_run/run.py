"""Run the Experiment 22 DREAM candidate to the long-node budget."""

from __future__ import annotations

import argparse

from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    add_common_arguments,
    config_from_args,
    run_parameter_ablation,
    variants_from_args,
)

from .config import EXPERIMENT_CONFIG


def parse_args() -> argparse.Namespace:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_parameter_ablation(
        config_from_args(args, EXPERIMENT_CONFIG),
        variants_from_args(args, EXPERIMENT_CONFIG["ablation_variants"]),
    )


if __name__ == "__main__":
    main()
