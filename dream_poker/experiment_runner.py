"""Reusable runner helpers for DREAM experiment packages."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import pyspiel
except Exception:  # pragma: no cover
    pyspiel = None

from dream_poker.experiment_utils import (
    compute_auc,
    ensure_average_policy_value_columns,
    ensure_dir,
    safe_mean,
    standard_error,
)
from dream_poker.seeding import set_seed
from dream_poker.solver import DREAMSolver


CORE_SUMMARY_METRICS = [
    "final_exploitability",
    "best_exploitability",
    "final_window_mean_exploitability",
    "final_average_policy_value",
    "final_window_mean_average_policy_value",
    "exploitability_auc_by_iteration",
    "final_policy_value_error",
    "final_wall_clock_seconds",
    "final_nodes_touched",
]


def json_ready(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    return obj


def write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(json_ready(obj), f, indent=2)


def create_timestamped_output_dir(output_root: Union[str, Path]) -> Path:
    root = ensure_dir(output_root)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(root / run_id)


def make_dream_solver(config: Dict, seed: int) -> DREAMSolver:
    if pyspiel is None:
        raise RuntimeError("OpenSpiel is not installed. Install open_spiel before running this experiment.")
    set_seed(seed)
    game = pyspiel.load_game(config["game_name"])
    return DREAMSolver(
        game=game,
        policy_network_layers=config["policy_network_layers"],
        advantage_network_layers=config["advantage_network_layers"],
        baseline_network_layers=config["baseline_network_layers"],
        num_iterations=config["num_iterations"],
        num_traversals=config["num_traversals"],
        learning_rate=config["learning_rate"],
        learning_rate_schedule=config.get("learning_rate_schedule", "constant"),
        learning_rate_end=config.get("learning_rate_end"),
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
        seed=seed,
    )


def expected_intermediate_policy_events(num_iterations: int, train_every: int) -> int:
    events = int(num_iterations) // int(train_every)
    if int(num_iterations) % int(train_every) != 0:
        events += 1
    return int(events)


def resolve_final_policy_steps(variant: Dict, config: Dict) -> Optional[int]:
    value = variant.get("final_policy_network_train_steps")
    if value is None:
        return None
    if value == "match_single_event":
        return int(config["policy_network_train_steps"])
    if value == "match_intermediate_total":
        events = expected_intermediate_policy_events(
            config["num_iterations"],
            config["policy_network_train_every"],
        )
        return int(events * config["policy_network_train_steps"])
    return int(value)


def summarise_seed_curves(
    curves: pd.DataFrame,
    seed: int,
    variant: Dict,
    config: Dict,
    final_policy_steps: Optional[int] = None,
) -> Dict:
    curves = ensure_average_policy_value_columns(curves, config.get("average_policy_value_target"))
    final_row = curves.iloc[-1]
    final_window = curves.tail(min(5, len(curves)))
    reached = curves[curves["exploitability"] <= config["exploitability_threshold"]]
    summary = {
        "seed": int(seed),
        "variant": variant["variant_id"],
        "variant_label": variant.get("label", variant["variant_id"]),
        "policy_training_mode": variant.get("policy_training_mode", "intermittent"),
        "final_policy_network_train_steps_requested": -1
        if final_policy_steps is None
        else int(final_policy_steps),
        "final_iteration": int(final_row["iteration"]),
        "final_nodes_touched": int(final_row["nodes_touched"]),
        "final_wall_clock_seconds": float(final_row["wall_clock_seconds"]),
        "final_exploitability": float(final_row["exploitability"]),
        "best_exploitability": float(curves["exploitability"].min()),
        "final_window_mean_exploitability": float(final_window["exploitability"].mean()),
        "final_window_std_exploitability": float(final_window["exploitability"].std(ddof=1))
        if len(final_window) > 1
        else 0.0,
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
        "final_policy_loss": float(final_row.get("policy_loss", np.nan)),
        "final_policy_entropy_mean": float(final_row.get("policy_entropy_mean", np.nan)),
    }
    return summary


def run_dream_variant_seed(
    seed: int,
    variant: Dict,
    config: Dict,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict]:
    solver = make_dream_solver(config, seed)
    final_steps = resolve_final_policy_steps(variant, config)
    curves = solver.solve(
        policy_training_mode=variant.get("policy_training_mode", "intermittent"),
        final_policy_network_train_steps=final_steps,
        isolate_policy_training_rng=config.get("isolate_policy_training_rng", True),
    )
    curves = ensure_average_policy_value_columns(curves, config.get("average_policy_value_target"))
    curves.insert(0, "seed", int(seed))
    curves.insert(1, "variant", variant["variant_id"])
    curves.insert(2, "variant_label", variant.get("label", variant["variant_id"]))
    curves.insert(3, "policy_training_mode", variant.get("policy_training_mode", "intermittent"))
    curves.insert(4, "final_policy_network_train_steps_requested", -1 if final_steps is None else int(final_steps))

    seed_dir = ensure_dir(output_dir / variant["variant_id"] / f"seed_{seed}")
    curves.to_csv(seed_dir / "checkpoint_curves.csv", index=False)
    summary = summarise_seed_curves(curves, seed, variant, config, final_steps)
    write_json(seed_dir / "seed_summary.json", summary)
    return curves, summary


def build_paired_differences(
    summary_df: pd.DataFrame,
    baseline_variant: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for variant in sorted(summary_df["variant"].unique()):
        if variant == baseline_variant:
            continue
        for seed in sorted(set(summary_df["seed"])):
            base = summary_df[(summary_df["variant"] == baseline_variant) & (summary_df["seed"] == seed)]
            comp = summary_df[(summary_df["variant"] == variant) & (summary_df["seed"] == seed)]
            if base.empty or comp.empty:
                continue
            row = {"seed": int(seed), "baseline_variant": baseline_variant, "variant": variant}
            for metric in metrics:
                row[f"delta_{metric}"] = float(comp.iloc[0][metric] - base.iloc[0][metric])
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_by_variant(summary_df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    for variant, group in summary_df.groupby("variant", sort=False):
        row = {
            "variant": variant,
            "variant_label": group["variant_label"].iloc[0] if "variant_label" in group else variant,
            "n_seeds": int(group["seed"].nunique()),
        }
        for col in metrics:
            row[f"{col}_mean"] = safe_mean(group[col])
            row[f"{col}_se"] = standard_error(group[col])
        rows.append(row)
    return pd.DataFrame(rows)
