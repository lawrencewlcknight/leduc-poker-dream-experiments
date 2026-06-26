"""Shared helpers for DREAM warm-start/checkpoint-resume ablations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dream_poker.constants import LEDUC_AVERAGE_POLICY_VALUE_TARGET
from dream_poker.experiment_utils import (
    average_policy_value_target,
    compute_auc,
    ensure_average_policy_value_columns,
    ensure_dir,
    safe_mean,
    standard_error,
)
from dream_poker.plotting import (
    add_average_policy_value_target,
    add_nash_exploitability_target,
    format_plot_title,
    is_average_policy_value_metric,
    is_exploitability_metric,
)


WARM_START_ARMS = ["baseline_continuous", "warm_start_resume"]


def summarise_warm_start_curve(curves: pd.DataFrame, seed: int, arm: str, config: Dict) -> Dict:
    curves = ensure_average_policy_value_columns(curves, config.get("average_policy_value_target"))
    final_row = curves.iloc[-1]
    final_window = curves.tail(min(5, len(curves)))
    reached = curves[curves["exploitability"] <= config["exploitability_threshold"]]
    return {
        "seed": int(seed),
        "arm": arm,
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
    }


def build_warm_start_paired_differences(summary_df: pd.DataFrame, seeds: Sequence[int]) -> pd.DataFrame:
    paired_rows = []
    metrics = [
        "final_exploitability",
        "best_exploitability",
        "final_window_mean_exploitability",
        "final_average_policy_value",
        "final_window_mean_average_policy_value",
        "exploitability_auc_by_iteration",
        "exploitability_auc_by_nodes",
        "final_policy_value_error",
        "final_nodes_touched",
        "final_wall_clock_seconds",
        "final_policy_gradient_steps_total",
    ]
    for seed in seeds:
        base = summary_df[
            (summary_df["seed"] == int(seed)) & (summary_df["arm"] == "baseline_continuous")
        ].iloc[0]
        warm = summary_df[
            (summary_df["seed"] == int(seed)) & (summary_df["arm"] == "warm_start_resume")
        ].iloc[0]
        row = {"seed": int(seed)}
        for metric in metrics:
            row[f"delta_{metric}_warm_minus_baseline"] = float(warm[metric] - base[metric])
        paired_rows.append(row)
    return pd.DataFrame(paired_rows)


def aggregate_warm_start_by_arm(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
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
    for arm, group in summary_df.groupby("arm", sort=False):
        row = {"arm": arm, "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            row[f"{metric}_mean"] = safe_mean(group[metric])
            row[f"{metric}_se"] = standard_error(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference_summary(paired_df: pd.DataFrame) -> Dict:
    return {
        col: {
            "mean": safe_mean(paired_df[col]),
            "se": standard_error(paired_df[col]),
        }
        for col in paired_df.columns
        if col != "seed"
    }


def create_warm_start_plots(
    curves_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    output_dir: Path,
    config: Optional[Dict] = None,
) -> None:
    plot_dir = ensure_dir(output_dir / "plots")
    value_target = average_policy_value_target(config)
    plot_arm_curves(
        curves_df,
        "exploitability",
        "Exploitability",
        "DREAM Warm-Start Ablation: Exploitability by Iteration",
        plot_dir / "dream_warm_start_exploitability_by_iteration.png",
        title_config=config,
    )
    plot_arm_curves(
        curves_df,
        "exploitability",
        "Exploitability",
        "DREAM Warm-Start Ablation: Exploitability by Nodes Touched",
        plot_dir / "dream_warm_start_exploitability_by_nodes.png",
        x_col="nodes_touched",
        title_config=config,
    )
    plot_arm_curves(
        curves_df,
        "average_policy_value",
        "Average policy value",
        "DREAM Warm-Start Ablation: Average Policy Value by Iteration",
        plot_dir / "dream_warm_start_average_policy_value_by_iteration.png",
        average_policy_value_target=value_target,
        title_config=config,
    )
    plot_arm_curves(
        curves_df,
        "average_policy_value",
        "Average policy value",
        "DREAM Warm-Start Ablation: Average Policy Value by Nodes Touched",
        plot_dir / "dream_warm_start_average_policy_value_by_nodes.png",
        x_col="nodes_touched",
        average_policy_value_target=value_target,
        title_config=config,
    )
    plot_arm_curves(
        curves_df,
        "policy_value_error",
        "Absolute error from Leduc game value",
        "DREAM Warm-Start Ablation: Policy-Value Error",
        plot_dir / "dream_warm_start_policy_value_error.png",
        title_config=config,
    )
    if "advantage_target_variance" in curves_df.columns:
        plot_arm_curves(
            curves_df,
            "advantage_target_variance",
            "Target variance",
            "DREAM Warm-Start Ablation: Advantage-Target Variance",
            plot_dir / "dream_warm_start_advantage_target_variance.png",
            title_config=config,
        )
    if "baseline_loss_player_0" in curves_df.columns:
        plot_arm_curves(
            curves_df,
            "baseline_loss_player_0",
            "Baseline loss",
            "DREAM Warm-Start Ablation: Baseline-Network Loss Diagnostic",
            plot_dir / "dream_warm_start_baseline_loss_player_0.png",
            title_config=config,
        )
    plot_summary_by_arm(
        summary_df,
        "final_exploitability",
        "Final exploitability",
        "DREAM Warm-Start Ablation: Final Exploitability",
        plot_dir / "dream_warm_start_final_exploitability.png",
        title_config=config,
    )
    plot_summary_by_arm(
        summary_df,
        "final_average_policy_value",
        "Final average policy value",
        "DREAM Warm-Start Ablation: Final Average Policy Value",
        plot_dir / "dream_warm_start_final_average_policy_value.png",
        average_policy_value_target=value_target,
        title_config=config,
    )
    plot_summary_by_arm(
        summary_df,
        "final_policy_value_error",
        "Final policy-value error",
        "DREAM Warm-Start Ablation: Final Policy-Value Error",
        plot_dir / "dream_warm_start_final_policy_value_error.png",
        title_config=config,
    )
    plot_paired_deltas(
        paired_df,
        "final_exploitability",
        "Warm-start minus baseline",
        "DREAM Warm-Start Ablation: Paired Final Exploitability Difference",
        plot_dir / "dream_warm_start_paired_final_exploitability_delta.png",
        title_config=config,
    )
    plot_paired_deltas(
        paired_df,
        "final_average_policy_value",
        "Warm-start minus baseline",
        "DREAM Warm-Start Ablation: Paired Final Average Policy Value Difference",
        plot_dir / "dream_warm_start_paired_final_average_policy_value_delta.png",
        title_config=config,
    )
    plot_paired_deltas(
        paired_df,
        "final_window_mean_exploitability",
        "Warm-start minus baseline",
        "DREAM Warm-Start Ablation: Paired Final-Window Exploitability Difference",
        plot_dir / "dream_warm_start_paired_final_window_delta.png",
        title_config=config,
    )


def plot_arm_curves(
    curves_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    *,
    x_col: str = "iteration",
    average_policy_value_target: float = LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    title_config: Optional[Dict] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for arm in WARM_START_ARMS:
        arm_df = curves_df[curves_df["arm"] == arm]
        if arm_df.empty:
            continue
        grouped = arm_df.groupby(x_col)[metric]
        mean = grouped.mean()
        se = grouped.sem().fillna(0.0)
        x = mean.index.to_numpy(dtype=float)
        y = mean.to_numpy(dtype=float)
        se_y = se.to_numpy(dtype=float)
        ax.plot(x, y, linewidth=2.1, label=arm)
        ax.fill_between(x, y - se_y, y + se_y, alpha=0.16)
        for _, seed_df in arm_df.groupby("seed"):
            ax.plot(seed_df[x_col], seed_df[metric], linewidth=0.6, alpha=0.16)
    if is_exploitability_metric(metric):
        add_nash_exploitability_target(ax)
    elif is_average_policy_value_metric(metric):
        add_average_policy_value_target(ax, target=average_policy_value_target)
    ax.set_title(format_plot_title(title, title_config))
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_summary_by_arm(
    summary_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    average_policy_value_target: float = LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    title_config: Optional[Dict] = None,
) -> None:
    grouped = summary_df.groupby("arm")[metric]
    means = [grouped.mean().get(arm, np.nan) for arm in WARM_START_ARMS]
    ses = [grouped.sem().get(arm, 0.0) for arm in WARM_START_ARMS]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(WARM_START_ARMS, means, yerr=ses, capsize=6)
    if is_exploitability_metric(metric):
        add_nash_exploitability_target(ax)
        ax.legend()
    elif is_average_policy_value_metric(metric):
        add_average_policy_value_target(ax, target=average_policy_value_target)
        ax.legend()
    ax.set_title(format_plot_title(title, title_config))
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_paired_deltas(
    paired_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    title_config: Optional[Dict] = None,
) -> None:
    col = f"delta_{metric}_warm_minus_baseline"
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x = np.arange(len(paired_df))
    y = paired_df[col].to_numpy(dtype=float)
    ax.axhline(0.0, linestyle="--", linewidth=1.1)
    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in paired_df["seed"]], rotation=45)
    ax.set_title(format_plot_title(title, title_config))
    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
