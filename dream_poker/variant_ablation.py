"""Generic helpers for matched-seed DREAM variant ablations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from dream_poker.constants import LEDUC_AVERAGE_POLICY_VALUE_TARGET
from dream_poker.experiment_utils import (
    compute_auc,
    ensure_average_policy_value_columns,
    ensure_dir,
    safe_mean,
    standard_error,
)
from dream_poker.plotting import plot_curve_by_variant, plot_metric_bar_by_variant, plot_paired_delta_bar


VARIANT_METADATA_KEYS = {"variant", "variant_id", "label", "description"}


def get_variant_id(variant: Dict) -> str:
    return str(variant.get("variant_id", variant.get("variant")))


def get_variant_label(variant: Dict) -> str:
    return str(variant.get("label", variant.get("description", get_variant_id(variant))))


def apply_variant_overrides(base_config: Dict, variant: Dict) -> Dict:
    config = dict(base_config)
    for key, value in variant.items():
        if key not in VARIANT_METADATA_KEYS:
            config[key] = value
    return config


def add_variant_curve_columns(
    curves: pd.DataFrame,
    config: Dict,
    variant: Dict,
    treatment_keys: Sequence[str],
) -> pd.DataFrame:
    curves = ensure_average_policy_value_columns(curves, config.get("average_policy_value_target"))
    curves["variant"] = get_variant_id(variant)
    curves["variant_label"] = get_variant_label(variant)
    for key in treatment_keys:
        curves[key] = config[key]
    return curves


def summarise_variant_curve(
    curves: pd.DataFrame,
    seed: int,
    variant: Dict,
    config: Dict,
    treatment_keys: Sequence[str],
) -> Dict:
    curves = ensure_average_policy_value_columns(curves, config.get("average_policy_value_target"))
    final_row = curves.iloc[-1]
    final_window = curves.tail(min(5, len(curves)))
    reached = curves[curves["exploitability"] <= config["exploitability_threshold"]]
    summary = {
        "seed": int(seed),
        "variant": get_variant_id(variant),
        "variant_label": get_variant_label(variant),
        "description": str(variant.get("description", "")),
        "num_iterations": int(config["num_iterations"]),
        "num_traversals": int(config["num_traversals"]),
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
        "final_policy_loss": float(final_row.get("policy_loss", np.nan)),
        "final_policy_entropy_mean": float(final_row.get("policy_entropy_mean", np.nan)),
        "final_policy_gradient_steps_total": int(final_row.get("policy_gradient_steps_total", 0)),
        "final_advantage_target_variance": float(final_row.get("advantage_target_variance", np.nan)),
        "final_baseline_reward_variance": float(final_row.get("baseline_reward_variance_sampled", np.nan)),
        "final_baseline_loss_player_0": float(final_row.get("baseline_loss_player_0", np.nan)),
        "final_baseline_loss_player_1": float(final_row.get("baseline_loss_player_1", np.nan)),
        "final_advantage_buffer_size_player_0": int(final_row.get("advantage_buffer_size_player_0", 0)),
        "final_advantage_buffer_size_player_1": int(final_row.get("advantage_buffer_size_player_1", 0)),
        "final_strategy_buffer_size": int(final_row.get("strategy_buffer_size", 0)),
        "final_baseline_buffer_size_player_0": int(final_row.get("baseline_buffer_size_player_0", 0)),
        "final_baseline_buffer_size_player_1": int(final_row.get("baseline_buffer_size_player_1", 0)),
    }
    for key in treatment_keys:
        summary[key] = config[key]
    return summary


def aggregate_variant_summary(
    summary_df: pd.DataFrame,
    metrics: Iterable[str],
    treatment_keys: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for variant, group in summary_df.groupby("variant", sort=False):
        row = {
            "variant": variant,
            "variant_label": group["variant_label"].iloc[0],
            "n_seeds": int(group["seed"].nunique()),
        }
        for key in treatment_keys:
            row[key] = group[key].iloc[0]
        for metric in metrics:
            row[f"{metric}_mean"] = safe_mean(group[metric])
            row[f"{metric}_se"] = standard_error(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference_summary(paired_df: pd.DataFrame) -> Dict:
    summary = {}
    for col in paired_df.columns:
        if col.startswith("delta_"):
            summary[f"{col}_mean"] = safe_mean(paired_df[col])
            summary[f"{col}_se"] = standard_error(paired_df[col])
    return summary


def write_multiseed_npz(curves_df: pd.DataFrame, output_path: Path, extra_columns: Sequence[str]) -> None:
    arrays = {
        "iteration": curves_df["iteration"].to_numpy(),
        "nodes_touched": curves_df["nodes_touched"].to_numpy(),
        "exploitability": curves_df["exploitability"].to_numpy(),
        "average_policy_value": curves_df["average_policy_value"].to_numpy(),
        "policy_value_error": curves_df["policy_value_error"].to_numpy(),
    }
    for col in extra_columns:
        if col in curves_df.columns:
            arrays[col] = curves_df[col].to_numpy()
    np.savez_compressed(output_path, **arrays)


def create_variant_ablation_plots(
    curves_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    variants: Sequence[Dict],
    output_dir: Path,
    plot_prefix: str,
    title_prefix: str,
    diagnostic_metrics: Sequence[tuple[str, str, str]] = (),
    average_policy_value_target: float = LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    title_config: Dict | None = None,
) -> None:
    plot_dir = ensure_dir(output_dir / "plots")
    variant_order = [get_variant_id(variant) for variant in variants]
    variant_labels = {get_variant_id(variant): get_variant_label(variant) for variant in variants}

    plot_curve_by_variant(
        curves_df,
        "iteration",
        "exploitability",
        f"{title_prefix}: Exploitability by Iteration",
        "Exploitability",
        variant_order,
        variant_labels,
        plot_dir / f"{plot_prefix}_exploitability_by_iteration.png",
        title_config=title_config,
    )
    plot_curve_by_variant(
        curves_df,
        "nodes_touched",
        "exploitability",
        f"{title_prefix}: Exploitability by Nodes Touched",
        "Exploitability",
        variant_order,
        variant_labels,
        plot_dir / f"{plot_prefix}_exploitability_by_nodes.png",
        title_config=title_config,
    )
    plot_curve_by_variant(
        curves_df,
        "iteration",
        "average_policy_value",
        f"{title_prefix}: Average Policy Value by Iteration",
        "Average policy value",
        variant_order,
        variant_labels,
        plot_dir / f"{plot_prefix}_average_policy_value_by_iteration.png",
        average_policy_value_target=average_policy_value_target,
        title_config=title_config,
    )
    plot_curve_by_variant(
        curves_df,
        "nodes_touched",
        "average_policy_value",
        f"{title_prefix}: Average Policy Value by Nodes Touched",
        "Average policy value",
        variant_order,
        variant_labels,
        plot_dir / f"{plot_prefix}_average_policy_value_by_nodes.png",
        average_policy_value_target=average_policy_value_target,
        title_config=title_config,
    )
    plot_curve_by_variant(
        curves_df,
        "iteration",
        "policy_value_error",
        f"{title_prefix}: Policy-Value Error",
        "Absolute error from Leduc game value",
        variant_order,
        variant_labels,
        plot_dir / f"{plot_prefix}_policy_value_error.png",
        title_config=title_config,
    )
    for metric, title, ylabel in diagnostic_metrics:
        if metric in curves_df.columns:
            plot_curve_by_variant(
                curves_df,
                "iteration",
                metric,
                f"{title_prefix}: {title}",
                ylabel,
                variant_order,
                variant_labels,
                plot_dir / f"{plot_prefix}_{metric}.png",
                title_config=title_config,
            )
    plot_metric_bar_by_variant(
        summary_df,
        "final_exploitability",
        f"{title_prefix}: Final Exploitability",
        "Final exploitability",
        variant_order,
        variant_labels,
        plot_dir / f"{plot_prefix}_final_exploitability.png",
        title_config=title_config,
    )
    plot_metric_bar_by_variant(
        summary_df,
        "final_window_mean_exploitability",
        f"{title_prefix}: Final-Window Exploitability",
        "Final-window mean exploitability",
        variant_order,
        variant_labels,
        plot_dir / f"{plot_prefix}_final_window_exploitability.png",
        title_config=title_config,
    )
    plot_metric_bar_by_variant(
        summary_df,
        "final_average_policy_value",
        f"{title_prefix}: Final Average Policy Value",
        "Final average policy value",
        variant_order,
        variant_labels,
        plot_dir / f"{plot_prefix}_final_average_policy_value.png",
        average_policy_value_target=average_policy_value_target,
        title_config=title_config,
    )
    plot_metric_bar_by_variant(
        summary_df,
        "final_window_mean_average_policy_value",
        f"{title_prefix}: Final-Window Average Policy Value",
        "Final-window mean average policy value",
        variant_order,
        variant_labels,
        plot_dir / f"{plot_prefix}_final_window_average_policy_value.png",
        average_policy_value_target=average_policy_value_target,
        title_config=title_config,
    )
    if len(paired_df):
        plot_paired_delta_bar(
            paired_df,
            "final_exploitability",
            f"{title_prefix}: Paired Final-Exploitability Difference",
            "Variant - baseline",
            variant_order,
            variant_labels,
            plot_dir / f"{plot_prefix}_paired_final_exploitability_delta.png",
            title_config=title_config,
        )
        plot_paired_delta_bar(
            paired_df,
            "final_average_policy_value",
            f"{title_prefix}: Paired Final Average-Policy-Value Difference",
            "Variant - baseline",
            variant_order,
            variant_labels,
            plot_dir / f"{plot_prefix}_paired_final_average_policy_value_delta.png",
            title_config=title_config,
        )
        plot_paired_delta_bar(
            paired_df,
            "final_policy_value_error",
            f"{title_prefix}: Paired Policy-Value-Error Difference",
            "Variant - baseline",
            variant_order,
            variant_labels,
            plot_dir / f"{plot_prefix}_paired_policy_value_error_delta.png",
            title_config=title_config,
        )
