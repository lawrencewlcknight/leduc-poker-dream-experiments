"""Run the DREAM learning-rate schedule ablation."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from dream_poker.experiment_runner import (
    create_timestamped_output_dir,
    json_ready,
    make_dream_solver,
    write_json,
)
from dream_poker.experiment_utils import ensure_dir
from dream_poker.lr_schedule import (
    aggregate_lr_schedule_summary,
    create_lr_schedule_plots,
    lr_schedule_paired_differences,
    paired_difference_summary,
    summarise_lr_schedule_curve,
)

from .config import BASELINE_VARIANT, EXPERIMENT_CONFIG, SCHEDULE_CONFIGS


def solver_config(config: Dict, schedule_config: Dict) -> Dict:
    solver_cfg = dict(config)
    solver_cfg["learning_rate_schedule"] = schedule_config["learning_rate_schedule"]
    solver_cfg["learning_rate_end"] = schedule_config.get("learning_rate_end", config["learning_rate"])
    return solver_cfg


def run_single_variant_seed(
    schedule_config: Dict,
    seed: int,
    config: Dict,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict]:
    solver = make_dream_solver(solver_config(config, schedule_config), seed)
    curves = solver.solve(isolate_policy_training_rng=config.get("isolate_policy_training_rng", True))
    curves.insert(0, "seed", int(seed))
    curves.insert(1, "variant", schedule_config["variant"])
    curves.insert(2, "learning_rate_schedule", schedule_config["learning_rate_schedule"])

    seed_dir = ensure_dir(output_dir / schedule_config["variant"] / f"seed_{seed}")
    curves.to_csv(seed_dir / "checkpoint_curves.csv", index=False)
    summary = summarise_lr_schedule_curve(curves, seed, schedule_config, config)
    write_json(seed_dir / "seed_summary.json", summary)
    return curves, summary


def run_experiment(
    config: Dict,
    schedule_configs: Sequence[Dict] = SCHEDULE_CONFIGS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    output_dir = create_timestamped_output_dir(config["output_root"])
    metadata = {**config, "schedule_configs": list(schedule_configs)}
    write_json(output_dir / "experiment_metadata.json", json_ready(metadata))

    all_curves = []
    summaries = []
    for schedule_config in schedule_configs:
        for seed in config["seeds"]:
            print(f"Running {schedule_config['variant']} seed {seed}...")
            curves, summary = run_single_variant_seed(schedule_config, int(seed), config, output_dir)
            all_curves.append(curves)
            summaries.append(summary)

    curves_df = pd.concat(all_curves, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    paired_df = lr_schedule_paired_differences(summary_df, BASELINE_VARIANT)
    aggregate_df = aggregate_lr_schedule_summary(summary_df)

    curves_df.to_csv(output_dir / "checkpoint_curves_by_variant.csv", index=False)
    summary_df.to_csv(output_dir / "seed_variant_summary.csv", index=False)
    paired_df.to_csv(output_dir / "paired_differences_vs_baseline.csv", index=False)
    aggregate_df.to_csv(output_dir / "aggregate_summary_by_variant.csv", index=False)
    write_json(output_dir / "aggregate_summary_by_variant.json", aggregate_df.to_dict(orient="records"))
    write_json(output_dir / "paired_difference_summary.json", paired_difference_summary(paired_df))
    np.savez_compressed(
        output_dir / "multiseed_curves_by_variant.npz",
        iteration=curves_df["iteration"].to_numpy(),
        nodes_touched=curves_df["nodes_touched"].to_numpy(),
        exploitability=curves_df["exploitability"].to_numpy(),
        policy_value_error=curves_df["policy_value_error"].to_numpy(),
        learning_rate=curves_df["learning_rate"].to_numpy(),
    )

    create_lr_schedule_plots(curves_df, summary_df, paired_df, config, schedule_configs, output_dir)
    print(f"Outputs written to {output_dir}")
    return curves_df, summary_df, paired_df, output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list, e.g. 1234,2025")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--learning-rate-end", type=float, default=None)
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
        config["num_iterations"] = int(args.iterations)
    if args.traversals is not None:
        config["num_traversals"] = int(args.traversals)
    if args.learning_rate is not None:
        config["learning_rate"] = float(args.learning_rate)
    if args.learning_rate_end is not None:
        config["learning_rate_end"] = float(args.learning_rate_end)
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


def schedule_configs_for_config(config: Dict) -> list[Dict]:
    schedules = copy.deepcopy(SCHEDULE_CONFIGS)
    for schedule in schedules:
        if schedule["learning_rate_schedule"] == "constant":
            schedule["learning_rate_end"] = config["learning_rate"]
        elif schedule["variant"] == "cosine_decay_to_10pct":
            schedule["learning_rate_end"] = config["learning_rate_end"]
    return schedules


def main() -> None:
    config = config_from_args(parse_args())
    run_experiment(config, schedule_configs_for_config(config))


if __name__ == "__main__":
    main()

