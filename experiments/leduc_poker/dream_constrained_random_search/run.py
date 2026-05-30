"""Run the DREAM constrained solver-parameter random search."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict

from dream_poker.experiment_runner import json_ready, write_json
from dream_poker.experiment_utils import average_policy_value_target
from dream_poker.random_search import (
    make_tuning_plots,
    prepare_tuning_output_dir,
    run_tuning_stage,
    sample_candidate_configs,
    select_confirmation_configs,
)

from .config import EXPERIMENT_CONFIG


def build_screening_configs(config: Dict) -> list[Dict]:
    candidates = sample_candidate_configs(
        baseline_config=config["baseline_config"],
        search_space=config["search_space"],
        fixed_parameters=config["fixed_parameters"],
        n_configs=config["n_random_candidates"],
        master_seed=config["master_seed"],
    )
    return [dict(config["baseline_config"])] + candidates


def run_experiment(config: Dict) -> Path:
    screening_configs = build_screening_configs(config)
    metadata = {
        **config,
        "screening_configs": screening_configs,
    }
    run_dir = prepare_tuning_output_dir(config["output_root"], metadata)
    write_json(run_dir / "screening_configs.json", json_ready(screening_configs))

    print("Running DREAM solver-parameter screening stage...")
    screening_summary, screening_curves, screening_agg = run_tuning_stage(
        configs=screening_configs,
        seeds=config["screening_seeds"],
        stage="screening",
        num_iterations=config["screening_num_iterations"],
        fixed_parameters=config["fixed_parameters"],
        run_dir=run_dir,
    )

    confirmation_configs = select_confirmation_configs(
        screening_agg,
        screening_configs,
        baseline_label=config["baseline_config"]["config_label"],
        n_confirmation_candidates=config["n_confirmation_candidates"],
    )
    write_json(run_dir / "confirmation_configs.json", json_ready(confirmation_configs))

    print("\nRunning DREAM confirmation stage...")
    confirmation_summary, confirmation_curves, confirmation_agg = run_tuning_stage(
        configs=confirmation_configs,
        seeds=config["confirmation_seeds"],
        stage="confirmation",
        num_iterations=config["confirmation_num_iterations"],
        fixed_parameters=config["fixed_parameters"],
        run_dir=run_dir,
    )

    make_tuning_plots(
        run_dir=run_dir,
        baseline_label=config["baseline_config"]["config_label"],
        screening_agg=screening_agg,
        confirmation_summary=confirmation_summary,
        confirmation_curves=confirmation_curves,
        confirmation_agg=confirmation_agg,
        average_policy_value_target_value=average_policy_value_target(config),
    )
    print(f"Outputs written to {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-seed", type=int, default=None)
    parser.add_argument("--screening-seeds", type=str, default=None)
    parser.add_argument("--confirmation-seeds", type=str, default=None)
    parser.add_argument("--screening-iterations", type=int, default=None)
    parser.add_argument("--confirmation-iterations", type=int, default=None)
    parser.add_argument("--n-random-candidates", type=int, default=None)
    parser.add_argument("--n-confirmation-candidates", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--num-traversals", type=int, default=None, help="Override baseline traversal count.")
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--advantage-network-train-steps", type=int, default=None)
    parser.add_argument("--baseline-network-train-steps", type=int, default=None)
    parser.add_argument("--policy-network-train-every", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--allow-policy-training-rng-advance", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Dict:
    config = copy.deepcopy(EXPERIMENT_CONFIG)
    if args.master_seed is not None:
        config["master_seed"] = int(args.master_seed)
    if args.screening_seeds:
        config["screening_seeds"] = [int(s.strip()) for s in args.screening_seeds.split(",") if s.strip()]
    if args.confirmation_seeds:
        config["confirmation_seeds"] = [int(s.strip()) for s in args.confirmation_seeds.split(",") if s.strip()]
    if args.screening_iterations is not None:
        config["screening_num_iterations"] = int(args.screening_iterations)
    if args.confirmation_iterations is not None:
        config["confirmation_num_iterations"] = int(args.confirmation_iterations)
    if args.n_random_candidates is not None:
        config["n_random_candidates"] = int(args.n_random_candidates)
    if args.n_confirmation_candidates is not None:
        config["n_confirmation_candidates"] = int(args.n_confirmation_candidates)
    if args.output_root is not None:
        config["output_root"] = args.output_root

    _apply_parameter_override(config, "num_traversals", args.num_traversals)
    _apply_parameter_override(config, "policy_network_train_steps", args.policy_network_train_steps)
    _apply_parameter_override(config, "advantage_network_train_steps", args.advantage_network_train_steps)
    _apply_parameter_override(config, "baseline_network_train_steps", args.baseline_network_train_steps)
    interval = args.policy_network_train_every if args.policy_network_train_every is not None else args.evaluation_interval
    _apply_parameter_override(config, "policy_network_train_every", interval)
    if interval is not None:
        config["fixed_parameters"]["policy_network_train_every"] = int(interval)
    if args.allow_policy_training_rng_advance:
        config["baseline_config"]["isolate_policy_training_rng"] = False
        config["fixed_parameters"]["isolate_policy_training_rng"] = False
    return config


def _apply_parameter_override(config: Dict, key: str, value) -> None:
    if value is not None:
        config["baseline_config"][key] = int(value) if isinstance(value, int) else value
        if key in config["search_space"]:
            config["search_space"][key] = [config["baseline_config"][key]]


def main() -> None:
    run_experiment(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
