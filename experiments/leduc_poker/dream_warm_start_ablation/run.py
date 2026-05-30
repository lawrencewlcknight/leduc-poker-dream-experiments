"""Run the DREAM fair warm-start ablation."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch

from dream_poker.experiment_runner import (
    create_timestamped_output_dir,
    json_ready,
    make_dream_solver,
    write_json,
)
from dream_poker.experiment_utils import cleanup_training_memory, ensure_dir
from dream_poker.warm_start import (
    aggregate_warm_start_by_arm,
    build_warm_start_paired_differences,
    create_warm_start_plots,
    paired_difference_summary,
    summarise_warm_start_curve,
)

from .config import EXPERIMENT_CONFIG


def solver_config(config: Dict) -> Dict:
    solver_cfg = dict(config)
    solver_cfg["num_iterations"] = int(config["total_iterations"])
    return solver_cfg


def validate_config(config: Dict) -> None:
    total = int(config["total_iterations"])
    warm = int(config["warm_start_iteration"])
    if warm <= 0 or warm >= total:
        raise ValueError("warm_start_iteration must be greater than 0 and less than total_iterations.")
    if warm % int(config["policy_network_train_every"]) != 0:
        raise ValueError("warm_start_iteration should coincide with a policy-training/evaluation event.")


def run_continuous_arm(seed: int, config: Dict, output_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    solver = make_dream_solver(solver_config(config), seed)
    curves = solver.solve(
        target_iteration=config["total_iterations"],
        isolate_policy_training_rng=config.get("isolate_policy_training_rng", True),
    )
    curves.insert(0, "seed", int(seed))
    curves.insert(1, "arm", "baseline_continuous")
    seed_dir = ensure_dir(output_dir / f"seed_{seed}" / "baseline_continuous")
    curves.to_csv(seed_dir / "checkpoint_curves.csv", index=False)
    summary = summarise_warm_start_curve(curves, seed, "baseline_continuous", config)
    write_json(seed_dir / "seed_summary.json", summary)
    del solver
    cleanup_training_memory()
    return curves, summary


def run_warm_start_arm(seed: int, config: Dict, output_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    start_time = time.perf_counter()
    solver = make_dream_solver(solver_config(config), seed)

    curves_1 = solver.solve(
        target_iteration=config["warm_start_iteration"],
        start_time=start_time,
        isolate_policy_training_rng=config.get("isolate_policy_training_rng", True),
    )
    checkpoint_dir = ensure_dir(output_dir / f"seed_{seed}" / "warm_start_resume" / "checkpoint")
    checkpoint_path = checkpoint_dir / f"dream_checkpoint_iter_{config['warm_start_iteration']}.pt"
    torch.save(solver.full_checkpoint_state(), checkpoint_path)
    del solver
    cleanup_training_memory()

    resumed_solver = make_dream_solver(solver_config(config), seed)
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    resumed_solver.load_full_checkpoint_state(state)
    del state
    cleanup_training_memory()
    curves_2 = resumed_solver.solve(
        target_iteration=config["total_iterations"],
        start_time=start_time,
        isolate_policy_training_rng=config.get("isolate_policy_training_rng", True),
    )

    curves = pd.concat([curves_1, curves_2], ignore_index=True)
    curves.insert(0, "seed", int(seed))
    curves.insert(1, "arm", "warm_start_resume")
    seed_dir = ensure_dir(output_dir / f"seed_{seed}" / "warm_start_resume")
    curves.to_csv(seed_dir / "checkpoint_curves.csv", index=False)
    summary = summarise_warm_start_curve(curves, seed, "warm_start_resume", config)
    summary["warm_start_iteration"] = int(config["warm_start_iteration"])
    summary["checkpoint_path"] = str(checkpoint_path)
    write_json(seed_dir / "seed_summary.json", summary)
    del resumed_solver
    cleanup_training_memory()
    return curves, summary


def run_experiment(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    validate_config(config)
    output_dir = create_timestamped_output_dir(config["output_root"])
    write_json(output_dir / "experiment_metadata.json", json_ready(config))

    all_curves = []
    summaries = []
    for seed in config["seeds"]:
        print(f"Running paired DREAM warm-start experiment for seed {seed}...")
        continuous_curves, continuous_summary = run_continuous_arm(seed, config, output_dir)
        warm_curves, warm_summary = run_warm_start_arm(seed, config, output_dir)
        all_curves.extend([continuous_curves, warm_curves])
        summaries.extend([continuous_summary, warm_summary])

    curves_df = pd.concat(all_curves, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    paired_df = build_warm_start_paired_differences(summary_df, config["seeds"])
    aggregate_df = aggregate_warm_start_by_arm(summary_df)

    curves_df.to_csv(output_dir / "checkpoint_curves_by_arm.csv", index=False)
    summary_df.to_csv(output_dir / "seed_arm_summary.csv", index=False)
    paired_df.to_csv(output_dir / "paired_differences_warm_minus_baseline.csv", index=False)
    aggregate_df.to_csv(output_dir / "aggregate_summary_by_arm.csv", index=False)
    write_json(output_dir / "aggregate_summary_by_arm.json", aggregate_df.to_dict(orient="records"))
    write_json(output_dir / "paired_difference_summary.json", paired_difference_summary(paired_df))

    create_warm_start_plots(curves_df, summary_df, paired_df, output_dir, config)
    print(f"Outputs written to {output_dir}")
    return curves_df, summary_df, paired_df, output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list, e.g. 1234,2025")
    parser.add_argument("--iterations", type=int, default=None, help="Total training iterations.")
    parser.add_argument("--warm-start-iteration", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--advantage-network-train-steps", type=int, default=None)
    parser.add_argument("--baseline-network-train-steps", type=int, default=None)
    parser.add_argument("--policy-network-train-every", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--allow-policy-training-rng-advance", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Dict:
    config = copy.deepcopy(EXPERIMENT_CONFIG)
    if args.seeds:
        config["seeds"] = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.iterations is not None:
        config["total_iterations"] = int(args.iterations)
    if args.warm_start_iteration is not None:
        config["warm_start_iteration"] = int(args.warm_start_iteration)
    if args.traversals is not None:
        config["num_traversals"] = int(args.traversals)
    if args.policy_network_train_steps is not None:
        config["policy_network_train_steps"] = int(args.policy_network_train_steps)
    if args.advantage_network_train_steps is not None:
        config["advantage_network_train_steps"] = int(args.advantage_network_train_steps)
    if args.baseline_network_train_steps is not None:
        config["baseline_network_train_steps"] = int(args.baseline_network_train_steps)
    interval = args.policy_network_train_every if args.policy_network_train_every is not None else args.evaluation_interval
    if interval is not None:
        config["policy_network_train_every"] = int(interval)
        config["evaluation_interval"] = int(interval)
    if args.output_root is not None:
        config["output_root"] = args.output_root
    if args.allow_policy_training_rng_advance:
        config["isolate_policy_training_rng"] = False
    return config


def main() -> None:
    run_experiment(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
