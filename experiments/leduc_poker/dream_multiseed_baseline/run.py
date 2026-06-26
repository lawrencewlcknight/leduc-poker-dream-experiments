
"""Run the Leduc poker DREAM-style baseline experiment."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import pyspiel
except Exception:  # pragma: no cover
    pyspiel = None

from dream_poker.experiment_utils import (
    average_policy_value_target,
    cleanup_training_memory,
    compute_auc,
    ensure_average_policy_value_columns,
    ensure_dir,
    safe_mean,
    standard_error,
)
from dream_poker.plotting import plot_mean_curve, plot_summary_bars
from dream_poker.seeding import set_seed
from dream_poker.solver import DREAMSolver
from .config import EXPERIMENT_CONFIG


def _json_ready(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def run_single_seed(seed: int, config: Dict, output_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    if pyspiel is None:
        raise RuntimeError("OpenSpiel is not installed. Install open_spiel before running this experiment.")
    set_seed(seed)
    game = pyspiel.load_game(config["game_name"])
    solver = DREAMSolver(
        game=game,
        policy_network_layers=config["policy_network_layers"],
        advantage_network_layers=config["advantage_network_layers"],
        baseline_network_layers=config["baseline_network_layers"],
        num_iterations=config["num_iterations"],
        num_traversals=config["num_traversals"],
        learning_rate=config["learning_rate"],
        batch_size_advantage=config["batch_size_advantage"],
        batch_size_strategy=config["batch_size_strategy"],
        batch_size_baseline=config["batch_size_baseline"],
        advantage_memory_capacity=config["advantage_memory_capacity"],
        strategy_memory_capacity=config["strategy_memory_capacity"],
        baseline_memory_capacity=config["baseline_memory_capacity"],
        epsilon=config["epsilon"],
        advantage_network_train_steps=config["advantage_network_train_steps"],
        policy_network_train_steps=config["policy_network_train_steps"],
        baseline_network_train_steps=config["baseline_network_train_steps"],
        policy_network_train_every=config["policy_network_train_every"],
        compute_exploitability=config["compute_exploitability"],
        game_value_player_0=config.get("game_value_player_0"),
        average_policy_value_target=config.get("average_policy_value_target"),
        seed=seed,
    )
    curves = solver.solve(isolate_policy_training_rng=config.get("isolate_policy_training_rng", True))
    curves = ensure_average_policy_value_columns(curves, config.get("average_policy_value_target"))
    curves.insert(0, "seed", int(seed))
    curves.insert(1, "variant", "dream_baseline")
    seed_dir = ensure_dir(output_dir / f"seed_{seed}")
    curves.to_csv(seed_dir / "checkpoint_curves.csv", index=False)

    final_row = curves.iloc[-1]
    final_window = curves.tail(min(5, len(curves)))
    reached = curves[curves["exploitability"] <= config["exploitability_threshold"]]
    summary = {
        "seed": int(seed),
        "variant": "dream_baseline",
        "final_iteration": int(final_row["iteration"]),
        "final_nodes_touched": int(final_row["nodes_touched"]),
        "final_wall_clock_seconds": float(final_row["wall_clock_seconds"]),
        "final_exploitability": float(final_row["exploitability"]),
        "best_exploitability": float(curves["exploitability"].min()),
        "final_window_mean_exploitability": float(final_window["exploitability"].mean()),
        "final_window_std_exploitability": float(final_window["exploitability"].std(ddof=1)) if len(final_window) > 1 else 0.0,
        "exploitability_auc_by_iteration": compute_auc(curves["iteration"], curves["exploitability"]),
        "exploitability_auc_by_nodes": compute_auc(curves["nodes_touched"], curves["exploitability"]),
        "final_policy_value_player_0": float(final_row["policy_value_player_0"]),
        "final_average_policy_value": float(final_row["average_policy_value"]),
        "final_window_mean_average_policy_value": float(final_window["average_policy_value"].mean()),
        "final_average_policy_value_error": float(final_row["average_policy_value_error"]),
        "final_policy_value_error": float(final_row["policy_value_error"]),
        "best_policy_value_error": float(curves["policy_value_error"].min()),
        "nodes_to_threshold": int(reached.iloc[0]["nodes_touched"]) if len(reached) else np.nan,
        "iterations_to_threshold": int(reached.iloc[0]["iteration"]) if len(reached) else np.nan,
        "seconds_to_threshold": float(reached.iloc[0]["wall_clock_seconds"]) if len(reached) else np.nan,
        "final_policy_training_events": int(final_row.get("policy_training_events", 0)),
        "final_policy_gradient_steps_total": int(final_row.get("policy_gradient_steps_total", 0)),
    }
    with open(seed_dir / "seed_summary.json", "w") as f:
        json.dump(_json_ready(summary), f, indent=2)
    del solver
    cleanup_training_memory()
    return curves, summary


def run_experiment(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    output_root = ensure_dir(config["output_root"])
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(output_root / run_id)
    with open(output_dir / "experiment_metadata.json", "w") as f:
        json.dump(_json_ready(config), f, indent=2)

    all_curves = []
    summaries = []
    for seed in config["seeds"]:
        print(f"Running DREAM baseline seed {seed}...")
        curves, summary = run_single_seed(seed, config, output_dir)
        all_curves.append(curves)
        summaries.append(summary)

    curves_df = pd.concat(all_curves, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    curves_df.to_csv(output_dir / "checkpoint_curves.csv", index=False)
    summary_df.to_csv(output_dir / "seed_summary.csv", index=False)

    aggregate = {}
    for col in [
        "final_exploitability", "best_exploitability", "final_window_mean_exploitability",
        "final_average_policy_value", "final_window_mean_average_policy_value",
        "exploitability_auc_by_iteration", "final_policy_value_error", "final_wall_clock_seconds",
        "final_nodes_touched", "final_policy_gradient_steps_total",
    ]:
        aggregate[f"{col}_mean"] = safe_mean(summary_df[col])
        aggregate[f"{col}_se"] = standard_error(summary_df[col])
    with open(output_dir / "aggregate_summary.json", "w") as f:
        json.dump(_json_ready(aggregate), f, indent=2)

    make_plots(curves_df, summary_df, output_dir, config)
    print(f"Outputs written to {output_dir}")
    return curves_df, summary_df, output_dir


def make_plots(curves_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path, config: Dict) -> None:
    value_target = average_policy_value_target(config)
    plot_mean_curve(
        curves_df, "iteration", "exploitability",
        "DREAM baseline exploitability by iteration", "Exploitability",
        output_dir / "exploitability_by_iteration_multiseed.png",
        title_config=config,
    )
    plot_mean_curve(
        curves_df, "nodes_touched", "exploitability",
        "DREAM baseline exploitability by nodes touched", "Exploitability",
        output_dir / "exploitability_by_nodes_multiseed.png",
        title_config=config,
    )
    plot_mean_curve(
        curves_df, "iteration", "average_policy_value",
        "DREAM baseline average policy value by iteration", "Average policy value",
        output_dir / "average_policy_value_by_iteration_multiseed.png",
        average_policy_value_target=value_target,
        title_config=config,
    )
    plot_mean_curve(
        curves_df, "nodes_touched", "average_policy_value",
        "DREAM baseline average policy value by nodes touched", "Average policy value",
        output_dir / "average_policy_value_by_nodes_multiseed.png",
        average_policy_value_target=value_target,
        title_config=config,
    )
    plot_mean_curve(
        curves_df, "iteration", "policy_value_error",
        "DREAM baseline policy-value error", "Absolute error from known value",
        output_dir / "policy_value_error_multiseed.png",
        title_config=config,
    )
    if "policy_loss" in curves_df.columns:
        plot_mean_curve(
            curves_df, "iteration", "policy_loss",
            "DREAM policy-network loss diagnostic", "Policy loss",
            output_dir / "policy_loss_diagnostic.png",
            title_config=config,
        )
    if "advantage_target_variance" in curves_df.columns:
        plot_mean_curve(
            curves_df, "iteration", "advantage_target_variance",
            "DREAM advantage-target variance diagnostic", "Target variance",
            output_dir / "advantage_target_variance_diagnostic.png",
            title_config=config,
        )
    if "baseline_reward_variance_sampled" in curves_df.columns:
        plot_mean_curve(
            curves_df, "iteration", "baseline_reward_variance_sampled",
            "DREAM baseline-replay reward variance diagnostic", "Reward variance",
            output_dir / "baseline_reward_variance_diagnostic.png",
            title_config=config,
        )
    plot_summary_bars(
        summary_df,
        ["final_exploitability", "best_exploitability", "final_window_mean_exploitability"],
        "DREAM baseline summary metrics",
        output_dir / "summary_metrics.png",
        title_config=config,
    )
    plot_summary_bars(
        summary_df,
        ["final_average_policy_value", "final_window_mean_average_policy_value"],
        "DREAM baseline average policy value summary",
        output_dir / "average_policy_value_summary_metrics.png",
        average_policy_value_target=value_target,
        title_config=config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list, e.g. 1234,2025")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--advantage-network-train-steps", type=int, default=None)
    parser.add_argument("--baseline-network-train-steps", type=int, default=None)
    parser.add_argument("--policy-network-train-every", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None, help="Alias for policy-network-train-every for output consistency.")
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Dict:
    config = copy.deepcopy(EXPERIMENT_CONFIG)
    if args.seeds:
        config["seeds"] = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.iterations is not None:
        config["num_iterations"] = int(args.iterations)
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
    return config


def main() -> None:
    run_experiment(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
