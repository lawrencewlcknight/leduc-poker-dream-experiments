"""Learning-rate schedule ablation helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dream_poker.experiment_utils import compute_auc, ensure_dir, safe_mean, standard_error


def scheduled_learning_rate(
    iteration: int,
    num_iterations: int,
    learning_rate_start: float,
    learning_rate_end: float,
    schedule: str,
) -> float:
    if int(num_iterations) <= 1:
        progress = 1.0
    else:
        progress = (int(iteration) - 1) / float(int(num_iterations) - 1)
        progress = min(max(progress, 0.0), 1.0)
    if schedule == "constant":
        return float(learning_rate_start)
    if schedule == "cosine_decay":
        return float(
            learning_rate_end
            + 0.5 * (learning_rate_start - learning_rate_end) * (1.0 + math.cos(math.pi * progress))
        )
    if schedule == "linear_decay":
        return float(learning_rate_start + (learning_rate_end - learning_rate_start) * progress)
    raise ValueError(f"Unknown learning-rate schedule: {schedule}")


def summarise_lr_schedule_curve(
    curves: pd.DataFrame,
    seed: int,
    schedule_config: Dict,
    config: Dict,
) -> Dict:
    final_row = curves.iloc[-1]
    final_window = curves.tail(min(5, len(curves)))
    reached = curves[curves["exploitability"] <= config["exploitability_threshold"]]
    return {
        "seed": int(seed),
        "variant": schedule_config["variant"],
        "learning_rate_schedule": schedule_config["learning_rate_schedule"],
        "learning_rate_start": float(config["learning_rate"]),
        "learning_rate_end": float(schedule_config.get("learning_rate_end", config["learning_rate"])),
        "final_iteration": int(final_row["iteration"]),
        "final_learning_rate": float(final_row["learning_rate"]),
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
        "final_policy_value_error": float(final_row["policy_value_error"]),
        "best_policy_value_error": float(curves["policy_value_error"].min()),
        "nodes_to_threshold": int(reached.iloc[0]["nodes_touched"]) if len(reached) else np.nan,
        "iterations_to_threshold": int(reached.iloc[0]["iteration"]) if len(reached) else np.nan,
        "seconds_to_threshold": float(reached.iloc[0]["wall_clock_seconds"]) if len(reached) else np.nan,
        "final_policy_loss": float(final_row.get("policy_loss", np.nan)),
        "final_advantage_target_variance": float(final_row.get("advantage_target_variance", np.nan)),
        "final_baseline_reward_variance": float(final_row.get("baseline_reward_variance_sampled", np.nan)),
        "final_policy_entropy_mean": float(final_row.get("policy_entropy_mean", np.nan)),
        "final_policy_gradient_steps_total": int(final_row.get("policy_gradient_steps_total", 0)),
    }


def lr_schedule_paired_differences(summary_df: pd.DataFrame, baseline_variant: str) -> pd.DataFrame:
    rows = []
    metrics = [
        "final_exploitability",
        "best_exploitability",
        "final_window_mean_exploitability",
        "exploitability_auc_by_iteration",
        "final_policy_value_error",
        "final_wall_clock_seconds",
        "final_nodes_touched",
    ]
    for seed, seed_df in summary_df.groupby("seed"):
        base_df = seed_df[seed_df["variant"] == baseline_variant]
        if len(base_df) != 1:
            continue
        base = base_df.iloc[0]
        for _, row in seed_df.iterrows():
            if row["variant"] == baseline_variant:
                continue
            out = {"seed": int(seed), "variant": row["variant"], "baseline_variant": baseline_variant}
            for metric in metrics:
                out[f"delta_{metric}"] = float(row[metric]) - float(base[metric])
            rows.append(out)
    return pd.DataFrame(rows)


def aggregate_lr_schedule_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "final_exploitability",
        "best_exploitability",
        "final_window_mean_exploitability",
        "exploitability_auc_by_iteration",
        "final_policy_value_error",
        "final_wall_clock_seconds",
        "final_nodes_touched",
        "final_learning_rate",
    ]
    for variant, group in summary_df.groupby("variant", sort=False):
        row = {
            "variant": variant,
            "learning_rate_schedule": group["learning_rate_schedule"].iloc[0],
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = safe_mean(group[metric])
            row[f"{metric}_se"] = standard_error(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference_summary(paired_df: pd.DataFrame) -> Dict:
    return {
        f"{col}_mean": safe_mean(paired_df[col])
        for col in paired_df.columns
        if col.startswith("delta_")
    } | {
        f"{col}_se": standard_error(paired_df[col])
        for col in paired_df.columns
        if col.startswith("delta_")
    }


def create_lr_schedule_plots(
    curves_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    config: Dict,
    schedule_configs: Sequence[Dict],
    output_dir: Path,
) -> None:
    plots_dir = ensure_dir(output_dir / "plots")
    plot_variant_mean_curve(
        curves_df,
        "iteration",
        "exploitability",
        "DREAM learning-rate schedule ablation: exploitability by iteration",
        "Exploitability",
        plots_dir / "dream_lr_schedule_exploitability_by_iteration.png",
    )
    plot_variant_mean_curve(
        curves_df,
        "nodes_touched",
        "exploitability",
        "DREAM learning-rate schedule ablation: exploitability by nodes touched",
        "Exploitability",
        plots_dir / "dream_lr_schedule_exploitability_by_nodes.png",
    )
    plot_variant_mean_curve(
        curves_df,
        "iteration",
        "policy_value_error",
        "DREAM learning-rate schedule ablation: policy-value error",
        "Absolute error from -1/18",
        plots_dir / "dream_lr_schedule_policy_value_error.png",
    )
    plot_variant_mean_curve(
        curves_df,
        "iteration",
        "policy_loss",
        "DREAM learning-rate schedule ablation: policy loss diagnostic",
        "Policy loss",
        plots_dir / "dream_lr_schedule_policy_loss.png",
    )
    if "baseline_loss_player_0" in curves_df.columns:
        plot_variant_mean_curve(
            curves_df,
            "iteration",
            "baseline_loss_player_0",
            "DREAM learning-rate schedule ablation: baseline-network loss diagnostic",
            "Baseline loss, player 0",
            plots_dir / "dream_lr_schedule_baseline_loss_player_0.png",
        )
    if "advantage_target_variance" in curves_df.columns:
        plot_variant_mean_curve(
            curves_df,
            "iteration",
            "advantage_target_variance",
            "DREAM learning-rate schedule ablation: advantage-target variance",
            "Advantage-target variance",
            plots_dir / "dream_lr_schedule_advantage_target_variance.png",
        )
    plot_final_metric_bar(
        summary_df,
        "final_exploitability",
        "Final exploitability by learning-rate schedule",
        "Final exploitability",
        plots_dir / "dream_lr_schedule_final_exploitability.png",
    )
    plot_final_metric_bar(
        summary_df,
        "exploitability_auc_by_iteration",
        "Exploitability AUC by learning-rate schedule",
        "Normalised AUC",
        plots_dir / "dream_lr_schedule_exploitability_auc.png",
    )
    if not paired_df.empty:
        plot_paired_delta(
            paired_df,
            "final_exploitability",
            "Paired final-exploitability difference versus constant baseline",
            "Delta exploitability",
            plots_dir / "dream_lr_schedule_paired_final_exploitability_delta.png",
        )
        plot_paired_delta(
            paired_df,
            "exploitability_auc_by_iteration",
            "Paired exploitability-AUC difference versus constant baseline",
            "Delta normalised AUC",
            plots_dir / "dream_lr_schedule_paired_exploitability_auc_delta.png",
        )
    plot_learning_rate_schedules(
        config=config,
        schedule_configs=schedule_configs,
        output_path=plots_dir / "dream_lr_schedule_learning_rate_schedules.png",
    )


def plot_variant_mean_curve(
    curves_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for variant, variant_df in curves_df.groupby("variant"):
        for _, seed_df in variant_df.groupby("seed"):
            ax.plot(seed_df[x_col], seed_df[y_col], linewidth=0.75, alpha=0.18)
        grouped = variant_df.groupby(x_col)[y_col]
        mean = grouped.mean()
        se = grouped.sem().fillna(0.0)
        x = mean.index.to_numpy(dtype=float)
        y = mean.to_numpy(dtype=float)
        se_y = se.to_numpy(dtype=float)
        ax.plot(x, y, linewidth=2.2, label=variant)
        ax.fill_between(x, y - se_y, y + se_y, alpha=0.14)
    ax.set_title(title)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_final_metric_bar(
    summary_df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    grouped = summary_df.groupby("variant")[metric]
    means = grouped.mean()
    ses = grouped.sem().fillna(0.0)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    positions = np.arange(len(means))
    ax.bar(positions, means.to_numpy(dtype=float), yerr=ses.to_numpy(dtype=float), capsize=5)
    ax.set_xticks(positions)
    ax.set_xticklabels(means.index.tolist(), rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_paired_delta(
    paired_df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    delta_col = f"delta_{metric}"
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    grouped = paired_df.groupby("variant")[delta_col]
    means = grouped.mean()
    ses = grouped.sem().fillna(0.0)
    positions = np.arange(len(means))
    ax.bar(positions, means.to_numpy(dtype=float), yerr=ses.to_numpy(dtype=float), capsize=5)
    ax.axhline(0.0, linestyle="--", linewidth=1.1)
    ax.set_xticks(positions)
    ax.set_xticklabels(means.index.tolist(), rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_learning_rate_schedules(config: Dict, schedule_configs: Sequence[Dict], output_path: Path) -> None:
    iterations = np.arange(1, int(config["num_iterations"]) + 1)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for schedule in schedule_configs:
        values = [
            scheduled_learning_rate(
                iteration=iteration,
                num_iterations=config["num_iterations"],
                learning_rate_start=config["learning_rate"],
                learning_rate_end=schedule.get("learning_rate_end", config["learning_rate"]),
                schedule=schedule["learning_rate_schedule"],
            )
            for iteration in iterations
        ]
        ax.plot(iterations, values, linewidth=2.2, label=schedule["variant"])
    ax.set_title("Learning-rate schedules")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
