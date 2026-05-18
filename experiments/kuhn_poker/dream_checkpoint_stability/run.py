"""Run the DREAM checkpoint-stability experiment."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pyspiel
except Exception:  # pragma: no cover
    pyspiel = None

from dream_poker.checkpointing import (
    checkpoint_exploitability_metrics,
    checkpoint_strength_summaries,
    discover_policy_snapshots,
    exact_pairwise_head_to_head,
    load_policies_by_seed,
    plot_errorbar,
    plot_heatmap,
    save_dream_policy_snapshot,
    save_optional_full_checkpoint,
)
from dream_poker.constants import KUHN_AVERAGE_POLICY_VALUE_TARGET
from dream_poker.experiment_runner import (
    create_timestamped_output_dir,
    json_ready,
    make_dream_solver,
    write_json,
)
from dream_poker.experiment_utils import (
    average_policy_value_target,
    compute_auc,
    ensure_average_policy_value_columns,
    ensure_dir,
    safe_mean,
    standard_error,
)
from dream_poker.plotting import plot_mean_curve

from .config import EXPERIMENT_CONFIG


def validate_config(config: Dict) -> None:
    schedule = [int(c) for c in config["checkpoint_schedule"]]
    if not schedule:
        raise ValueError("checkpoint_schedule must contain at least one checkpoint.")
    if sorted(schedule) != schedule:
        raise ValueError("checkpoint_schedule must be sorted in ascending order.")
    if schedule[-1] != int(config["num_iterations"]):
        raise ValueError("The final checkpoint must equal num_iterations.")


def train_solver_to_checkpoints(solver, seed: int, config: Dict, output_dir: Path) -> pd.DataFrame:
    checkpoint_schedule = set(map(int, config["checkpoint_schedule"]))
    snapshot_dir = ensure_dir(output_dir / "policy_snapshots")
    full_checkpoint_dir = ensure_dir(output_dir / "full_solver_checkpoints")
    start_time = time.perf_counter()
    rows: List[Dict] = []

    for iteration in range(1, int(config["num_iterations"]) + 1):
        solver._iteration = int(iteration)
        for traverser in range(solver._num_players):
            for _ in range(solver._num_traversals):
                state = solver._game.new_initial_state()
                solver._traverse_outcome_sampling(state, traverser, pi_reach=1.0, sigma_reach=1.0)
            solver._last_advantage_loss[traverser] = solver._learn_advantage_network(traverser)
            solver._last_baseline_loss[traverser] = solver._learn_baseline_network(traverser)

        train_policy_now = (
            iteration % int(config["policy_network_train_every"]) == 0
        ) or (iteration in checkpoint_schedule)
        if train_policy_now:
            if config.get("isolate_policy_training_rng", True):
                rng_state = solver._capture_rng_state()
                solver._last_policy_loss = solver._learn_strategy_network()
                if iteration in checkpoint_schedule:
                    row = build_checkpoint_row(solver, seed, iteration, start_time, snapshot_dir, full_checkpoint_dir, config)
                    rows.append(row)
                    solver._restore_rng_state(rng_state)
                else:
                    solver._restore_rng_state(rng_state)
            else:
                solver._last_policy_loss = solver._learn_strategy_network()
                if iteration in checkpoint_schedule:
                    rows.append(
                        build_checkpoint_row(
                            solver,
                            seed,
                            iteration,
                            start_time,
                            snapshot_dir,
                            full_checkpoint_dir,
                            config,
                        )
                    )
        elif iteration in checkpoint_schedule:
            rows.append(
                build_checkpoint_row(
                    solver,
                    seed,
                    iteration,
                    start_time,
                    snapshot_dir,
                    full_checkpoint_dir,
                    config,
                )
            )

        if rows:
            pd.DataFrame(rows).to_csv(output_dir / f"seed_{seed}_checkpoint_metrics_partial.csv", index=False)

    return pd.DataFrame(rows)


def build_checkpoint_row(
    solver,
    seed: int,
    iteration: int,
    start_time: float,
    snapshot_dir: Path,
    full_checkpoint_dir: Path,
    config: Dict,
) -> Dict:
    row = solver._checkpoint_metrics(start_time)
    row["seed"] = int(seed)
    row["checkpoint_iteration"] = int(iteration)
    snapshot_path = save_dream_policy_snapshot(solver, seed, iteration, snapshot_dir, config)
    full_checkpoint_path = save_optional_full_checkpoint(solver, seed, iteration, full_checkpoint_dir, config)
    row["policy_snapshot_path"] = str(snapshot_path)
    row["full_checkpoint_path"] = str(full_checkpoint_path) if full_checkpoint_path is not None else ""
    print(
        f"Seed {seed} checkpoint {iteration}: "
        f"exploitability={row['exploitability']:.6f}, value_error={row['policy_value_error']:.6f}"
    )
    return row


def summarise_seed_checkpoints(curves: pd.DataFrame, seed: int, config: Dict) -> Dict:
    curves = ensure_average_policy_value_columns(curves, config.get("average_policy_value_target"))
    final = curves.sort_values("iteration").iloc[-1]
    expl = curves["exploitability"].astype(float)
    final_window = curves.tail(min(3, len(curves)))["exploitability"].astype(float)
    below = curves[curves["exploitability"] <= config["exploitability_threshold"]]
    return {
        "seed": int(seed),
        "num_checkpoints": int(len(curves)),
        "final_iteration": int(final["iteration"]),
        "final_exploitability": float(final["exploitability"]),
        "best_exploitability": float(expl.min()),
        "best_exploitability_iteration": int(curves.loc[expl.idxmin(), "iteration"]),
        "final_window_mean_exploitability": float(final_window.mean()),
        "final_window_std_exploitability": float(final_window.std(ddof=0)),
        "final_average_policy_value": float(final["average_policy_value"]),
        "final_window_mean_average_policy_value": float(
            curves.tail(min(3, len(curves)))["average_policy_value"].mean()
        ),
        "final_average_policy_value_error": float(final["average_policy_value_error"]),
        "final_policy_value_error": float(final["policy_value_error"]),
        "final_nodes_touched": int(final["nodes_touched"]),
        "final_wall_clock_seconds": float(final["wall_clock_seconds"]),
        "nodes_to_threshold": int(below.iloc[0]["nodes_touched"]) if len(below) else np.nan,
        "iteration_to_threshold": int(below.iloc[0]["iteration"]) if len(below) else np.nan,
        "exploitability_auc_by_iteration": compute_auc(curves["iteration"], curves["exploitability"]),
        "final_policy_training_events": int(final.get("policy_training_events", 0)),
        "final_policy_gradient_steps_total": int(final.get("policy_gradient_steps_total", 0)),
    }


def aggregate_seed_summary(summary_df: pd.DataFrame) -> Dict:
    metrics = [
        "final_exploitability",
        "best_exploitability",
        "final_window_mean_exploitability",
        "final_average_policy_value",
        "final_window_mean_average_policy_value",
        "final_policy_value_error",
        "final_wall_clock_seconds",
        "final_nodes_touched",
        "final_policy_gradient_steps_total",
    ]
    aggregate = {"num_seeds": int(len(summary_df))}
    for metric in metrics:
        aggregate[f"{metric}_mean"] = safe_mean(summary_df[metric])
        aggregate[f"{metric}_se"] = standard_error(summary_df[metric])
    return aggregate


def run_training_stage(config: Dict, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if pyspiel is None:
        raise RuntimeError("OpenSpiel is not installed. Install open_spiel before running this experiment.")
    validate_config(config)
    write_json(output_dir / "experiment_metadata.json", json_ready(config))

    all_curves = []
    summaries = []
    for seed in config["seeds"]:
        print(f"\n=== DREAM checkpoint run for seed {seed} ===")
        solver = make_dream_solver(config, int(seed))
        curves = train_solver_to_checkpoints(solver, int(seed), config, output_dir)
        curves["seed"] = int(seed)
        all_curves.append(curves)
        summaries.append(summarise_seed_checkpoints(curves, int(seed), config))

    curves_df = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    curves_df.to_csv(output_dir / "checkpoint_curves.csv", index=False)
    summary_df.to_csv(output_dir / "seed_summary.csv", index=False)
    write_json(output_dir / "aggregate_summary.json", aggregate_seed_summary(summary_df))
    make_training_plots(curves_df, output_dir, config)
    return curves_df, summary_df


def make_training_plots(curves_df: pd.DataFrame, output_dir: Path, config: Dict) -> None:
    plot_dir = ensure_dir(output_dir / "plots")
    value_target = average_policy_value_target(config)
    plot_mean_curve(
        curves_df,
        "iteration",
        "exploitability",
        "DREAM checkpoint exploitability over training",
        "Exploitability",
        plot_dir / "checkpoint_exploitability_by_iteration.png",
    )
    plot_mean_curve(
        curves_df,
        "nodes_touched",
        "exploitability",
        "DREAM checkpoint exploitability by nodes touched",
        "Exploitability",
        plot_dir / "checkpoint_exploitability_by_nodes.png",
    )
    plot_mean_curve(
        curves_df,
        "iteration",
        "average_policy_value",
        "DREAM checkpoint average policy value over training",
        "Average policy value",
        plot_dir / "checkpoint_average_policy_value_by_iteration.png",
        average_policy_value_target=value_target,
    )
    plot_mean_curve(
        curves_df,
        "nodes_touched",
        "average_policy_value",
        "DREAM checkpoint average policy value by nodes touched",
        "Average policy value",
        plot_dir / "checkpoint_average_policy_value_by_nodes.png",
        average_policy_value_target=value_target,
    )
    plot_mean_curve(
        curves_df,
        "iteration",
        "policy_value_error",
        "DREAM checkpoint policy-value error",
        "Absolute error from -1/18",
        plot_dir / "checkpoint_policy_value_error.png",
    )
    plot_mean_curve(
        curves_df,
        "iteration",
        "policy_loss",
        "DREAM average-policy training loss at checkpoints",
        "Policy loss",
        plot_dir / "checkpoint_policy_loss.png",
    )


def run_analysis_stage(config: Dict, checkpoint_dir: Path, output_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    if pyspiel is None:
        raise RuntimeError("OpenSpiel is not installed. Install open_spiel before running this experiment.")
    checkpoint_dir = Path(checkpoint_dir)
    metadata_path = checkpoint_dir / "experiment_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            saved_config = json.load(f)
        config = {**config, **saved_config}
    snapshot_dir = checkpoint_dir / "policy_snapshots" if (checkpoint_dir / "policy_snapshots").exists() else checkpoint_dir
    analysis_dir = ensure_dir(output_dir or checkpoint_dir / "head_to_head_analysis")
    plot_dir = ensure_dir(analysis_dir / "plots")

    checkpoint_df = discover_policy_snapshots(snapshot_dir, config.get("checkpoint_schedule"))
    if checkpoint_df.empty:
        raise FileNotFoundError(f"No DREAM policy snapshots found in {snapshot_dir}.")

    game = pyspiel.load_game(config["game_name"])
    policies_by_seed, loaded_policy_df = load_policies_by_seed(game, checkpoint_df)
    checkpoint_metrics_df = checkpoint_exploitability_metrics(game, policies_by_seed)
    exact_df, exact_matrices, mean_matrix, win_fraction_matrix = exact_pairwise_head_to_head(
        game,
        policies_by_seed,
        equivalence_epsilon=float(config["head_to_head_equivalence_epsilon"]),
    )
    monotonicity_df, strength_with_metrics_df, aggregate_strength_df, best_checkpoint_df = (
        checkpoint_strength_summaries(
            exact_matrices,
            checkpoint_metrics_df,
            equivalence_epsilon=float(config["head_to_head_equivalence_epsilon"]),
        )
    )

    checkpoint_df.to_csv(analysis_dir / "checkpoint_inventory.csv", index=False)
    loaded_policy_df.to_csv(analysis_dir / "loaded_policy_inventory.csv", index=False)
    checkpoint_metrics_df.to_csv(analysis_dir / "checkpoint_exploitability_metrics.csv", index=False)
    exact_df.to_csv(analysis_dir / "head_to_head_exact_pairwise.csv", index=False)
    mean_matrix.to_csv(analysis_dir / "head_to_head_exact_mean_matrix.csv")
    win_fraction_matrix.to_csv(analysis_dir / "head_to_head_seed_win_fraction_matrix.csv")
    monotonicity_df.to_csv(analysis_dir / "head_to_head_monotonicity_summary_by_seed.csv", index=False)
    strength_with_metrics_df.to_csv(analysis_dir / "head_to_head_strength_with_metrics.csv", index=False)
    aggregate_strength_df.to_csv(analysis_dir / "head_to_head_aggregate_strength_summary.csv", index=False)
    best_checkpoint_df.to_csv(analysis_dir / "best_checkpoint_summary.csv", index=False)

    make_head_to_head_plots(
        mean_matrix,
        win_fraction_matrix,
        aggregate_strength_df,
        plot_dir,
        average_policy_value_target(config),
    )
    write_json(
        analysis_dir / "head_to_head_analysis_metadata.json",
        {
            "checkpoint_dir": checkpoint_dir,
            "snapshot_dir": snapshot_dir,
            "analysis_dir": analysis_dir,
            "num_snapshots": int(len(checkpoint_df)),
            "num_seeds": int(checkpoint_df["seed"].nunique()),
            "checkpoint_schedule": list(map(int, sorted(checkpoint_df["iteration"].unique()))),
            "head_to_head_equivalence_epsilon": float(config["head_to_head_equivalence_epsilon"]),
        },
    )
    return {
        "checkpoint_inventory": checkpoint_df,
        "loaded_policy_inventory": loaded_policy_df,
        "checkpoint_metrics": checkpoint_metrics_df,
        "exact_pairwise": exact_df,
        "mean_matrix": mean_matrix,
        "win_fraction_matrix": win_fraction_matrix,
        "monotonicity": monotonicity_df,
        "strength_with_metrics": strength_with_metrics_df,
        "aggregate_strength": aggregate_strength_df,
        "best_checkpoint": best_checkpoint_df,
    }


def make_head_to_head_plots(
    mean_matrix: pd.DataFrame,
    win_fraction_matrix: pd.DataFrame,
    aggregate_strength_df: pd.DataFrame,
    plot_dir: Path,
    average_policy_value_target_value: float = KUHN_AVERAGE_POLICY_VALUE_TARGET,
) -> None:
    plot_heatmap(
        mean_matrix,
        "DREAM mean exact head-to-head EV across seeds",
        plot_dir / "dream_head_to_head_exact_mean_matrix.png",
    )
    later_vs_earlier = mean_matrix.copy()
    for i, row_checkpoint in enumerate(later_vs_earlier.index):
        for j, col_checkpoint in enumerate(later_vs_earlier.columns):
            if int(row_checkpoint) <= int(col_checkpoint):
                later_vs_earlier.iat[i, j] = np.nan
    plot_heatmap(
        later_vs_earlier,
        "DREAM later-vs-earlier checkpoint EV",
        plot_dir / "dream_later_vs_earlier_mean_matrix.png",
    )
    plot_heatmap(
        win_fraction_matrix,
        "Fraction of seeds where row checkpoint beats column checkpoint",
        plot_dir / "dream_head_to_head_seed_win_fraction_matrix.png",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        fmt=".2f",
        colorbar_label="Seed win fraction",
    )
    plot_errorbar(
        aggregate_strength_df,
        "mean_EV_vs_earlier_mean",
        "mean_EV_vs_earlier_sem",
        "Mean EV vs earlier checkpoints",
        "DREAM: does later training improve head-to-head performance?",
        plot_dir / "dream_head_to_head_strength_vs_earlier_aggregate.png",
    )
    plot_errorbar(
        aggregate_strength_df,
        "EV_vs_previous_mean",
        "EV_vs_previous_sem",
        "EV vs immediately previous checkpoint",
        "DREAM adjacent-checkpoint improvement",
        plot_dir / "dream_head_to_head_vs_previous_checkpoint_aggregate.png",
    )
    plot_errorbar(
        aggregate_strength_df,
        "exploitability_mean",
        "exploitability_sem",
        "Exploitability",
        "DREAM checkpoint exploitability",
        plot_dir / "dream_checkpoint_exploitability_aggregate.png",
        zero_line=False,
    )
    plot_errorbar(
        aggregate_strength_df,
        "average_policy_value_mean",
        "average_policy_value_sem",
        "Average policy value",
        "DREAM checkpoint average policy value",
        plot_dir / "dream_checkpoint_average_policy_value_aggregate.png",
        zero_line=False,
        average_policy_value_target=average_policy_value_target_value,
    )
    plot_errorbar(
        aggregate_strength_df,
        "policy_value_error_from_known_value_mean",
        "policy_value_error_from_known_value_sem",
        "Absolute error from -1/18",
        "DREAM checkpoint policy-value error",
        plot_dir / "dream_checkpoint_policy_value_error_aggregate.png",
        zero_line=False,
    )


def run_experiment(config: Dict, stage: str = "all", checkpoint_dir: Optional[Path] = None) -> Path:
    if stage not in {"all", "train", "analyze"}:
        raise ValueError("stage must be one of: all, train, analyze")
    if stage == "analyze":
        if checkpoint_dir is None:
            raise ValueError("--checkpoint-dir is required when --stage analyze is used.")
        run_analysis_stage(config, checkpoint_dir)
        return Path(checkpoint_dir)

    output_dir = create_timestamped_output_dir(config["output_root"])
    run_training_stage(config, output_dir)
    if stage == "all":
        run_analysis_stage(config, output_dir)
    print(f"Outputs written to {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "train", "analyze"], default="all")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list, e.g. 1234,2025")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--checkpoint-schedule", type=str, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--advantage-network-train-steps", type=int, default=None)
    parser.add_argument("--baseline-network-train-steps", type=int, default=None)
    parser.add_argument("--policy-network-train-every", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--save-full-solver-checkpoints", action="store_true")
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
    if args.checkpoint_schedule:
        config["checkpoint_schedule"] = [int(s.strip()) for s in args.checkpoint_schedule.split(",") if s.strip()]
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
    if args.save_full_solver_checkpoints:
        config["save_full_solver_checkpoints"] = True
    if args.allow_policy_training_rng_advance:
        config["isolate_policy_training_rng"] = False
    return config


def main() -> None:
    args = parse_args()
    run_experiment(config_from_args(args), stage=args.stage, checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
