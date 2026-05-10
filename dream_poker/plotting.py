
"""Plotting helpers with the same look and feel as the sister repositories."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dream_poker.experiment_utils import ensure_dir


def plot_mean_curve(
    curves_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Optional[Path] = None,
    equilibrium_line: Optional[float] = None,
):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for seed, seed_df in curves_df.groupby("seed"):
        ax.plot(seed_df[x_col], seed_df[y_col], linewidth=0.9, alpha=0.28)
    grouped = curves_df.groupby(x_col)[y_col]
    mean = grouped.mean()
    se = grouped.sem().fillna(0.0)
    x = mean.index.to_numpy(dtype=float)
    y = mean.to_numpy(dtype=float)
    se_y = se.to_numpy(dtype=float)
    ax.plot(x, y, linewidth=2.2, label="Mean across seeds")
    ax.fill_between(x, y - se_y, y + se_y, alpha=0.18, label="±1 s.e.")
    if equilibrium_line is not None:
        ax.axhline(equilibrium_line, linestyle="--", linewidth=1.1, label="Known Kuhn value")
    ax.set_title(title)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig, ax


def plot_summary_bar(summary_df: pd.DataFrame, metric: str, title: str, ylabel: str, output_path: Optional[Path] = None):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    mean = summary_df[metric].mean()
    se = summary_df[metric].sem()
    ax.bar(["DREAM baseline"], [mean], yerr=[se], capsize=6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig, ax


def create_thesis_plots(curves_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = ensure_dir(output_dir / "plots")
    plot_mean_curve(
        curves_df,
        "iteration",
        "exploitability",
        "DREAM Baseline: Exploitability by Iteration",
        "Exploitability",
        plot_dir / "dream_exploitability_by_iteration.png",
    )
    plot_mean_curve(
        curves_df,
        "nodes_touched",
        "exploitability",
        "DREAM Baseline: Exploitability by Nodes Touched",
        "Exploitability",
        plot_dir / "dream_exploitability_by_nodes.png",
    )
    plot_mean_curve(
        curves_df,
        "iteration",
        "policy_value_error",
        "DREAM Baseline: Policy-Value Error",
        "Absolute error from -1/18",
        plot_dir / "dream_policy_value_error.png",
    )
    plot_mean_curve(
        curves_df,
        "iteration",
        "policy_loss",
        "DREAM Baseline: Average-Policy Loss Diagnostic",
        "Policy loss",
        plot_dir / "dream_policy_loss.png",
    )
    plot_mean_curve(
        curves_df,
        "iteration",
        "advantage_target_variance",
        "DREAM Baseline: Advantage-Target Variance Diagnostic",
        "Target variance",
        plot_dir / "dream_advantage_target_variance.png",
    )
    plot_mean_curve(
        curves_df,
        "iteration",
        "baseline_reward_variance_sampled",
        "DREAM Baseline: Baseline-Replay Reward Variance Diagnostic",
        "Reward variance",
        plot_dir / "dream_baseline_replay_variance.png",
    )
    plot_summary_bar(
        summary_df,
        "final_exploitability",
        "DREAM Baseline: Final Exploitability",
        "Final exploitability",
        plot_dir / "dream_final_exploitability.png",
    )

# After running the experiment, call:
# create_thesis_plots(curves_df, summary_df, output_dir)
