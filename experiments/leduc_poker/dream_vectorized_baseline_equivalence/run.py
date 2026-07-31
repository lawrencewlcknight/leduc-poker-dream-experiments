"""Run the DREAM vectorized-baseline equivalence experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence, Tuple

import pandas as pd

from dream_poker.experiment_runner import (
    CORE_SUMMARY_METRICS,
    build_paired_differences,
    create_timestamped_output_dir,
    json_ready,
    write_json,
)
from dream_poker.experiment_utils import average_policy_value_target, ensure_dir
from dream_poker.plotting import plot_metric_bar_by_variant, plot_paired_delta_bar
from dream_poker.variant_ablation import (
    aggregate_variant_summary,
    create_variant_ablation_plots,
    get_variant_id,
    get_variant_label,
    paired_difference_summary,
    write_multiseed_npz,
)
from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    DEFAULT_DIAGNOSTIC_METRICS,
    add_common_arguments,
    config_from_args,
    ensure_fixed_baseline_variant,
    fixed_baseline_enabled,
    load_fixed_baseline_outputs,
    run_single_variant_seed,
    variants_from_args,
)

from .config import EXP22_ORIGINAL_VARIANT, EXPERIMENT_CONFIG, IMPLEMENTATION_EQUIVALENCE_VARIANTS


TIMING_METRICS = [
    "final_wall_clock_seconds",
    "final_cumulative_traversal_seconds",
    "final_cumulative_advantage_training_seconds",
    "final_cumulative_baseline_training_seconds",
    "final_cumulative_policy_training_seconds",
    "final_cumulative_supervised_training_seconds",
]

PAIRED_METRICS = list(
    dict.fromkeys(
        [
            *CORE_SUMMARY_METRICS,
            "exploitability_auc_by_nodes",
            "final_advantage_target_variance",
            "final_baseline_reward_variance",
            "final_policy_loss",
            "final_policy_entropy_mean",
            *TIMING_METRICS,
        ]
    )
)

AGGREGATE_METRICS = [
    *PAIRED_METRICS,
    "best_policy_value_error",
    "nodes_to_threshold",
    "iterations_to_threshold",
    "final_policy_gradient_steps_total",
    "final_advantage_buffer_size_player_0",
    "final_advantage_buffer_size_player_1",
    "final_strategy_buffer_size",
    "final_baseline_buffer_size_player_0",
    "final_baseline_buffer_size_player_1",
]


def make_timing_plots(
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    variants: Sequence[Dict],
    output_dir: Path,
    config: Dict,
) -> None:
    plot_dir = ensure_dir(output_dir / "plots")
    plot_prefix = str(config["plot_prefix"])
    title_prefix = str(config["plot_title"])
    variant_order = [get_variant_id(variant) for variant in variants]
    variant_labels = {get_variant_id(variant): get_variant_label(variant) for variant in variants}

    for metric, suffix, ylabel in [
        ("final_wall_clock_seconds", "wall_clock_seconds", "Final wall-clock seconds"),
        (
            "final_cumulative_baseline_training_seconds",
            "baseline_training_seconds",
            "Cumulative baseline-training seconds",
        ),
        (
            "final_cumulative_supervised_training_seconds",
            "supervised_training_seconds",
            "Cumulative supervised-training seconds",
        ),
    ]:
        if metric not in summary_df.columns:
            continue
        plot_metric_bar_by_variant(
            summary_df,
            metric,
            f"{title_prefix}: {ylabel}",
            ylabel,
            variant_order,
            variant_labels,
            plot_dir / f"{plot_prefix}_{suffix}.png",
            title_config=config,
        )
        if len(paired_df) and f"delta_{metric}" in paired_df.columns:
            plot_paired_delta_bar(
                paired_df,
                metric,
                f"{title_prefix}: Paired {ylabel} Difference",
                "Variant - original learner",
                variant_order,
                variant_labels,
                plot_dir / f"{plot_prefix}_paired_{suffix}_delta.png",
                title_config=config,
            )


def runtime_summary_by_variant(summary_df: pd.DataFrame, treatment_keys: Sequence[str]) -> pd.DataFrame:
    present_metrics = [metric for metric in TIMING_METRICS if metric in summary_df.columns]
    return aggregate_variant_summary(summary_df, present_metrics, treatment_keys)


def run_experiment(
    config: Dict,
    variants: Sequence[Dict] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    variants = list(config["ablation_variants"] if variants is None else variants)
    variants = ensure_fixed_baseline_variant(config, variants)
    treatment_keys = list(config["treatment_keys"])
    output_dir = create_timestamped_output_dir(config["output_root"])
    baseline_variant = str(config.get("baseline_variant", EXP22_ORIGINAL_VARIANT))
    metadata = {**config, "variants": list(variants), "baseline_variant": baseline_variant}
    write_json(output_dir / "experiment_metadata.json", json_ready(metadata))

    all_curves, summaries = load_fixed_baseline_outputs(config, variants, treatment_keys, output_dir)
    for variant in variants:
        if fixed_baseline_enabled(config) and get_variant_id(variant) == baseline_variant:
            continue
        for seed in config["seeds"]:
            print(f"Running {get_variant_id(variant)} seed {seed}...", flush=True)
            curves, summary = run_single_variant_seed(variant, int(seed), config, output_dir)
            all_curves.append(curves)
            summaries.append(summary)

    curves_df = pd.concat(all_curves, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    paired_df = build_paired_differences(summary_df, baseline_variant, PAIRED_METRICS)
    aggregate_df = aggregate_variant_summary(summary_df, AGGREGATE_METRICS, treatment_keys)
    runtime_df = runtime_summary_by_variant(summary_df, treatment_keys)

    curves_df.to_csv(output_dir / "checkpoint_curves_by_variant.csv", index=False)
    summary_df.to_csv(output_dir / "seed_variant_summary.csv", index=False)
    paired_df.to_csv(output_dir / "paired_differences_vs_original_learner.csv", index=False)
    aggregate_df.to_csv(output_dir / "aggregate_summary_by_variant.csv", index=False)
    runtime_df.to_csv(output_dir / "runtime_summary_by_variant.csv", index=False)
    write_json(output_dir / "aggregate_summary_by_variant.json", aggregate_df.to_dict(orient="records"))
    write_json(output_dir / "runtime_summary_by_variant.json", runtime_df.to_dict(orient="records"))
    write_json(output_dir / "paired_difference_summary.json", paired_difference_summary(paired_df))
    write_multiseed_npz(
        curves_df,
        output_dir / "multiseed_curves_by_variant.npz",
        extra_columns=[
            *treatment_keys,
            "advantage_target_variance",
            "baseline_reward_variance_sampled",
            "policy_entropy_mean",
            "strategy_buffer_size",
            "cumulative_baseline_training_seconds",
            "cumulative_supervised_training_seconds",
        ],
    )

    create_variant_ablation_plots(
        curves_df,
        summary_df,
        paired_df,
        variants,
        output_dir,
        plot_prefix=str(config["plot_prefix"]),
        title_prefix=str(config["plot_title"]),
        diagnostic_metrics=DEFAULT_DIAGNOSTIC_METRICS,
        average_policy_value_target=average_policy_value_target(config),
        title_config=config,
    )
    make_timing_plots(summary_df, paired_df, variants, output_dir, config)
    print(f"Outputs written to {output_dir}", flush=True)
    return curves_df, summary_df, paired_df, output_dir


def parse_args() -> argparse.Namespace:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        config_from_args(args, EXPERIMENT_CONFIG),
        variants_from_args(args, IMPLEMENTATION_EQUIVALENCE_VARIANTS),
    )


if __name__ == "__main__":
    main()
