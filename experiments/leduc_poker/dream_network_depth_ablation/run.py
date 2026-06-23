"""Run the independent DREAM network-depth ablation (Experiment 11)."""

from __future__ import annotations

import argparse

from experiments.leduc_poker.dream_network_size_ablation.run import (
    add_common_arguments,
    config_from_args,
    run_experiment,
)

from .config import EXPERIMENT_CONFIG, NETWORK_DEPTH_VARIANTS


def parse_args() -> argparse.Namespace:
    return add_common_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(config_from_args(args, EXPERIMENT_CONFIG), NETWORK_DEPTH_VARIANTS)


if __name__ == "__main__":
    main()
