
"""Plotting helpers with the same look and feel as the sister repositories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dream_poker.constants import LEDUC_AVERAGE_POLICY_VALUE_TARGET
from dream_poker.experiment_utils import ensure_dir


NASH_EXPLOITABILITY_TARGET = 0.0
NASH_EXPLOITABILITY_TARGET_LABEL = "Nash equilibrium target (0)"
AVERAGE_POLICY_VALUE_TARGET_LABEL = "Average-policy value target"


def normalise_algorithm_title(raw: str, fallback: str = "DREAM") -> str:
    raw = raw.strip()
    if not raw:
        return fallback
    lower = raw.lower()
    if lower.startswith("deep cfr"):
        return "Deep CFR"
    if lower.startswith("escher"):
        return "ESCHER"
    if lower.startswith("dream"):
        return "DREAM"
    first_token = raw.split()[0]
    return first_token.split("-")[0].upper()


def algorithm_title_from_config(config: Optional[Dict] = None, fallback: str = "DREAM") -> str:
    if not config:
        return fallback
    raw = str(config.get("plot_algorithm") or config.get("algorithm") or fallback).strip()
    return normalise_algorithm_title(raw, fallback=fallback)


def poker_title_from_config(config: Optional[Dict] = None, fallback: str = "Leduc") -> str:
    if not config:
        return fallback
    raw = str(config.get("plot_game") or config.get("game_name") or fallback).strip()
    if not raw:
        return fallback
    raw = raw.removesuffix("_poker").removesuffix(" poker")
    words = re.split(r"[_\s-]+", raw)
    return " ".join(word.capitalize() for word in words if word)


def plot_title_prefix(config: Optional[Dict] = None, *, algorithm: Optional[str] = None, poker: Optional[str] = None) -> str:
    algorithm_name = normalise_algorithm_title(algorithm) if algorithm else algorithm_title_from_config(config)
    poker_name = poker.strip().title() if poker else poker_title_from_config(config)
    return f"{algorithm_name} - {poker_name}"


def format_plot_title(
    title: str,
    config: Optional[Dict] = None,
    *,
    algorithm: Optional[str] = None,
    poker: Optional[str] = None,
) -> str:
    prefix = plot_title_prefix(config, algorithm=algorithm, poker=poker)
    if title.startswith(f"{prefix} - "):
        return title
    return f"{prefix} - {title}"


def is_exploitability_metric(metric: str) -> bool:
    return "exploitability" in metric.lower()


def is_average_policy_value_metric(metric: str) -> bool:
    metric = metric.lower()
    return metric.endswith("average_policy_value") and not metric.startswith("delta_")


def add_nash_exploitability_target(
    ax,
    *,
    axis: str = "y",
    label: str = NASH_EXPLOITABILITY_TARGET_LABEL,
) -> None:
    if axis == "y":
        ax.axhline(NASH_EXPLOITABILITY_TARGET, linestyle="--", linewidth=1.1, label=label)
    elif axis == "x":
        ax.axvline(NASH_EXPLOITABILITY_TARGET, linestyle="--", linewidth=1.1, label=label)
    else:
        raise ValueError("axis must be 'x' or 'y'.")


def add_average_policy_value_target(
    ax,
    *,
    target: float = LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    axis: str = "y",
    label: str = AVERAGE_POLICY_VALUE_TARGET_LABEL,
) -> None:
    target_label = f"{label} ({target:.4f})"
    if axis == "y":
        ax.axhline(float(target), linestyle="--", linewidth=1.1, label=target_label)
    elif axis == "x":
        ax.axvline(float(target), linestyle="--", linewidth=1.1, label=target_label)
    else:
        raise ValueError("axis must be 'x' or 'y'.")


def plot_mean_curve(
    curves_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Optional[Path] = None,
    equilibrium_line: Optional[float] = None,
    average_policy_value_target: float = LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    title_config: Optional[Dict] = None,
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
    if is_exploitability_metric(y_col):
        add_nash_exploitability_target(ax)
    elif is_average_policy_value_metric(y_col):
        add_average_policy_value_target(ax, target=average_policy_value_target)
    elif equilibrium_line is not None:
        ax.axhline(equilibrium_line, linestyle="--", linewidth=1.1, label="Target")
    ax.set_title(format_plot_title(title, title_config))
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig, ax


def plot_summary_bar(
    summary_df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Optional[Path] = None,
    average_policy_value_target: float = LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    title_config: Optional[Dict] = None,
):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    mean = summary_df[metric].mean()
    se = summary_df[metric].sem()
    ax.bar(["DREAM baseline"], [mean], yerr=[se], capsize=6)
    if is_exploitability_metric(metric):
        add_nash_exploitability_target(ax)
        ax.legend()
    elif is_average_policy_value_metric(metric):
        add_average_policy_value_target(ax, target=average_policy_value_target)
        ax.legend()
    ax.set_title(format_plot_title(title, title_config))
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig, ax


def plot_summary_bars(
    summary_df: pd.DataFrame,
    metrics: Sequence[str],
    title: str,
    output_path: Optional[Path] = None,
    average_policy_value_target: float = LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    title_config: Optional[Dict] = None,
):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    labels = [metric.replace("_", " ").title() for metric in metrics]
    means = [summary_df[metric].mean() for metric in metrics]
    ses = [summary_df[metric].sem() for metric in metrics]
    ax.bar(labels, means, yerr=ses, capsize=6)
    if all(is_exploitability_metric(metric) for metric in metrics):
        add_nash_exploitability_target(ax)
        ax.legend()
    elif all(is_average_policy_value_metric(metric) for metric in metrics):
        add_average_policy_value_target(ax, target=average_policy_value_target)
        ax.legend()
    ax.set_title(format_plot_title(title, title_config))
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig, ax


def ordered_variants(df: pd.DataFrame, variant_order: Sequence[str]) -> list[str]:
    present = set(df["variant"])
    return [variant for variant in variant_order if variant in present]


def plot_metric_bar_by_variant(
    summary_df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    variant_order: Sequence[str],
    variant_labels: Dict[str, str],
    output_path: Optional[Path] = None,
    average_policy_value_target: float = LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    title_config: Optional[Dict] = None,
):
    order = ordered_variants(summary_df, variant_order)
    labels = [variant_labels.get(v, v) for v in order]
    means = [summary_df.loc[summary_df["variant"] == v, metric].mean() for v in order]
    ses = [summary_df.loc[summary_df["variant"] == v, metric].sem() for v in order]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.bar(labels, means, yerr=ses, capsize=6)
    if is_exploitability_metric(metric):
        add_nash_exploitability_target(ax)
        ax.legend()
    elif is_average_policy_value_metric(metric):
        add_average_policy_value_target(ax, target=average_policy_value_target)
        ax.legend()
    ax.set_title(format_plot_title(title, title_config))
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig, ax


def plot_curve_by_variant(
    curves_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    variant_order: Sequence[str],
    variant_labels: Dict[str, str],
    output_path: Optional[Path] = None,
    average_policy_value_target: float = LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    title_config: Optional[Dict] = None,
):
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for variant in ordered_variants(curves_df, variant_order):
        vdf = curves_df[curves_df["variant"] == variant]
        if vdf.empty:
            continue
        for _, seed_df in vdf.groupby("seed"):
            ax.plot(seed_df[x_col], seed_df[y_col], linewidth=0.8, alpha=0.18)
        grouped = vdf.groupby(x_col)[y_col]
        mean = grouped.mean()
        se = grouped.sem().fillna(0.0)
        x = mean.index.to_numpy(dtype=float)
        y = mean.to_numpy(dtype=float)
        se_y = se.to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            linewidth=2.2,
            marker="o" if len(x) <= 3 else None,
            label=variant_labels.get(variant, variant),
        )
        if len(x) > 1:
            ax.fill_between(x, y - se_y, y + se_y, alpha=0.15)
    if is_exploitability_metric(y_col):
        add_nash_exploitability_target(ax)
    elif is_average_policy_value_metric(y_col):
        add_average_policy_value_target(ax, target=average_policy_value_target)
    ax.set_title(format_plot_title(title, title_config))
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig, ax


def plot_paired_delta_bar(
    paired_df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    variant_order: Sequence[str],
    variant_labels: Dict[str, str],
    output_path: Optional[Path] = None,
    title_config: Optional[Dict] = None,
):
    delta_col = f"delta_{metric}"
    order = ordered_variants(paired_df, variant_order)
    labels = [variant_labels.get(v, v) for v in order]
    means = [paired_df.loc[paired_df["variant"] == v, delta_col].mean() for v in order]
    ses = [paired_df.loc[paired_df["variant"] == v, delta_col].sem() for v in order]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.bar(labels, means, yerr=ses, capsize=6)
    ax.set_title(format_plot_title(title, title_config))
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig, ax


def create_thesis_plots(
    curves_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    title_config: Optional[Dict] = None,
) -> None:
    plot_dir = ensure_dir(output_dir / "plots")
    plot_mean_curve(
        curves_df,
        "iteration",
        "exploitability",
        "DREAM Baseline: Exploitability by Iteration",
        "Exploitability",
        plot_dir / "dream_exploitability_by_iteration.png",
        title_config=title_config,
    )
    plot_mean_curve(
        curves_df,
        "nodes_touched",
        "exploitability",
        "DREAM Baseline: Exploitability by Nodes Touched",
        "Exploitability",
        plot_dir / "dream_exploitability_by_nodes.png",
        title_config=title_config,
    )
    plot_mean_curve(
        curves_df,
        "iteration",
        "policy_value_error",
        "DREAM Baseline: Policy-Value Error",
        "Absolute error from Leduc game value",
        plot_dir / "dream_policy_value_error.png",
        title_config=title_config,
    )
    plot_mean_curve(
        curves_df,
        "iteration",
        "policy_loss",
        "DREAM Baseline: Average-Policy Loss Diagnostic",
        "Policy loss",
        plot_dir / "dream_policy_loss.png",
        title_config=title_config,
    )
    plot_mean_curve(
        curves_df,
        "iteration",
        "advantage_target_variance",
        "DREAM Baseline: Advantage-Target Variance Diagnostic",
        "Target variance",
        plot_dir / "dream_advantage_target_variance.png",
        title_config=title_config,
    )
    plot_mean_curve(
        curves_df,
        "iteration",
        "baseline_reward_variance_sampled",
        "DREAM Baseline: Baseline-Replay Reward Variance Diagnostic",
        "Reward variance",
        plot_dir / "dream_baseline_replay_variance.png",
        title_config=title_config,
    )
    plot_summary_bar(
        summary_df,
        "final_exploitability",
        "DREAM Baseline: Final Exploitability",
        "Final exploitability",
        plot_dir / "dream_final_exploitability.png",
        title_config=title_config,
    )

# After running the experiment, call:
# create_thesis_plots(curves_df, summary_df, output_dir)
