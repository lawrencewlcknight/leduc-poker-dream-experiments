"""Run the DREAM residual-network ablation."""

from __future__ import annotations

import argparse
import copy
from typing import Dict

from dream_poker.variant_ablation import get_variant_id
from experiments.leduc_poker.dream_network_size_ablation.run import (
    add_common_arguments,
    config_from_args,
    run_experiment,
)

from .config import EXPERIMENT_CONFIG, RESIDUAL_NETWORK_VARIANTS


def parse_args() -> argparse.Namespace:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated variant ids to run; defaults to all configured variants.",
    )
    return parser.parse_args()


def variants_from_args(args: argparse.Namespace) -> list[Dict]:
    variants = copy.deepcopy(RESIDUAL_NETWORK_VARIANTS)
    if not args.variants:
        return variants
    requested = {value.strip() for value in args.variants.split(",") if value.strip()}
    selected = [variant for variant in variants if get_variant_id(variant) in requested]
    missing = sorted(requested - {get_variant_id(variant) for variant in selected})
    if missing:
        raise ValueError(f"Unknown variant ids: {missing}")
    return selected


def main() -> None:
    args = parse_args()
    run_experiment(config_from_args(args, EXPERIMENT_CONFIG), variants_from_args(args))


if __name__ == "__main__":
    main()
