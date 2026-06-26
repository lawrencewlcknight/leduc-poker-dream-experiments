"""Run the DREAM final-only average-policy training ablation."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from dream_poker.experiment_runner import (
    CORE_SUMMARY_METRICS,
    aggregate_by_variant,
    build_paired_differences,
    create_timestamped_output_dir,
    json_ready,
    run_dream_variant_seed,
    write_json,
)
from dream_poker.experiment_utils import average_policy_value_target, ensure_dir
from dream_poker.plotting import (
    plot_curve_by_variant,
    plot_metric_bar_by_variant,
    plot_paired_delta_bar,
)

from .config import EXPERIMENT_CONFIG, POLICY_TRAINING_VARIANTS


BASELINE_VARIANT = "intermittent_every_25_baseline"
PAIRED_METRICS = [
    "final_exploitability",
    "best_exploitability",
    "final_window_mean_exploitability",
    "final_average_policy_value",
    "final_window_mean_average_policy_value",
    "final_policy_value_error",
    "final_wall_clock_seconds",
    "final_policy_gradient_steps_total",
]
AGGREGATE_METRICS = CORE_SUMMARY_METRICS + [
    "final_policy_gradient_steps_total",
    "final_policy_loss",
    "final_policy_entropy_mean",
    "final_average_policy_value_error",
]


def run_experiment(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    output_dir = create_timestamped_output_dir(config["output_root"])
    variants = list(config.get("variants", POLICY_TRAINING_VARIANTS))
    write_json(output_dir / "experiment_metadata.json", json_ready({**config, "variants": variants}))

    all_curves = []
    summaries = []
    for variant in variants:
        for seed in config["seeds"]:
            print(f"Running {variant['variant_id']} seed {seed}...")
            curves, summary = run_dream_variant_seed(seed, variant, config, output_dir)
            all_curves.append(curves)
            summaries.append(summary)

    curves_df = pd.concat(all_curves, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    paired_df = build_paired_differences(summary_df, BASELINE_VARIANT, PAIRED_METRICS)
    aggregate_df = aggregate_by_variant(summary_df, AGGREGATE_METRICS)

    curves_df.to_csv(output_dir / "checkpoint_curves_by_variant.csv", index=False)
    summary_df.to_csv(output_dir / "seed_variant_summary.csv", index=False)
    paired_df.to_csv(output_dir / "paired_differences_vs_intermediate_baseline.csv", index=False)
    aggregate_df.to_csv(output_dir / "aggregate_summary_by_variant.csv", index=False)
    write_json(output_dir / "aggregate_summary_by_variant.json", aggregate_df.to_dict(orient="records"))

    make_plots(curves_df, summary_df, paired_df, output_dir, variants, config)
    print(f"Outputs written to {output_dir}")
    return curves_df, summary_df, paired_df, output_dir


def make_plots(
    curves_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    output_dir: Path,
    variants,
    config: Dict,
) -> None:
    plot_dir = ensure_dir(output_dir / "plots")
    value_target = average_policy_value_target(config)
    variant_order = [variant["variant_id"] for variant in variants]
    variant_labels = {variant["variant_id"]: variant["label"] for variant in variants}

    plot_metric_bar_by_variant(
        summary_df,
        "final_exploitability",
        "DREAM Policy-Training Ablation: Final Exploitability",
        "Final exploitability",
        variant_order,
        variant_labels,
        plot_dir / "dream_policy_training_final_exploitability.png",
        title_config=config,
    )
    plot_metric_bar_by_variant(
        summary_df,
        "final_average_policy_value",
        "DREAM Policy-Training Ablation: Final Average Policy Value",
        "Final average policy value",
        variant_order,
        variant_labels,
        plot_dir / "dream_policy_training_final_average_policy_value.png",
        average_policy_value_target=value_target,
        title_config=config,
    )
    plot_metric_bar_by_variant(
        summary_df,
        "final_policy_value_error",
        "DREAM Policy-Training Ablation: Final Policy-Value Error",
        "Absolute error from Leduc game value",
        variant_order,
        variant_labels,
        plot_dir / "dream_policy_training_final_value_error.png",
        title_config=config,
    )
    plot_metric_bar_by_variant(
        summary_df,
        "final_policy_gradient_steps_total",
        "DREAM Policy-Training Ablation: Policy-Gradient Budget",
        "Policy-gradient steps",
        variant_order,
        variant_labels,
        plot_dir / "dream_policy_training_gradient_budget.png",
        title_config=config,
    )
    plot_metric_bar_by_variant(
        summary_df,
        "final_wall_clock_seconds",
        "DREAM Policy-Training Ablation: Runtime",
        "Wall-clock seconds",
        variant_order,
        variant_labels,
        plot_dir / "dream_policy_training_runtime.png",
        title_config=config,
    )
    if len(paired_df):
        plot_paired_delta_bar(
            paired_df,
            "final_exploitability",
            "DREAM Policy-Training Ablation: Paired Final-Exploitability Difference",
            "Variant - intermittent baseline",
            variant_order,
            variant_labels,
            plot_dir / "dream_policy_training_paired_delta_exploitability.png",
            title_config=config,
        )
        plot_paired_delta_bar(
            paired_df,
            "final_average_policy_value",
            "DREAM Policy-Training Ablation: Paired Final Average-Policy-Value Difference",
            "Variant - intermittent baseline",
            variant_order,
            variant_labels,
            plot_dir / "dream_policy_training_paired_delta_average_policy_value.png",
            title_config=config,
        )
        plot_paired_delta_bar(
            paired_df,
            "final_policy_value_error",
            "DREAM Policy-Training Ablation: Paired Policy-Value-Error Difference",
            "Variant - intermittent baseline",
            variant_order,
            variant_labels,
            plot_dir / "dream_policy_training_paired_delta_value_error.png",
            title_config=config,
        )
    plot_curve_by_variant(
        curves_df,
        "iteration",
        "exploitability",
        "DREAM Policy-Training Ablation: Exploitability by Iteration",
        "Exploitability",
        variant_order,
        variant_labels,
        plot_dir / "dream_policy_training_exploitability_by_iteration.png",
        title_config=config,
    )
    plot_curve_by_variant(
        curves_df,
        "nodes_touched",
        "exploitability",
        "DREAM Policy-Training Ablation: Exploitability by Nodes Touched",
        "Exploitability",
        variant_order,
        variant_labels,
        plot_dir / "dream_policy_training_exploitability_by_nodes.png",
        title_config=config,
    )
    plot_curve_by_variant(
        curves_df,
        "iteration",
        "average_policy_value",
        "DREAM Policy-Training Ablation: Average Policy Value by Iteration",
        "Average policy value",
        variant_order,
        variant_labels,
        plot_dir / "dream_policy_training_average_policy_value_by_iteration.png",
        average_policy_value_target=value_target,
        title_config=config,
    )
    plot_curve_by_variant(
        curves_df,
        "nodes_touched",
        "average_policy_value",
        "DREAM Policy-Training Ablation: Average Policy Value by Nodes Touched",
        "Average policy value",
        variant_order,
        variant_labels,
        plot_dir / "dream_policy_training_average_policy_value_by_nodes.png",
        average_policy_value_target=value_target,
        title_config=config,
    )
    plot_curve_by_variant(
        curves_df,
        "iteration",
        "policy_loss",
        "DREAM Policy-Training Ablation: Policy-Loss Diagnostic",
        "Policy loss",
        variant_order,
        variant_labels,
        plot_dir / "dream_policy_training_policy_loss.png",
        title_config=config,
    )
    plot_curve_by_variant(
        curves_df,
        "iteration",
        "policy_entropy_mean",
        "DREAM Policy-Training Ablation: Policy-Entropy Diagnostic",
        "Policy entropy",
        variant_order,
        variant_labels,
        plot_dir / "dream_policy_training_entropy.png",
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
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=None,
        help="Alias for policy-network-train-every for output consistency.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated variant ids to run; defaults to all configured variants.",
    )
    parser.add_argument(
        "--allow-policy-training-rng-advance",
        action="store_true",
        help="Do not restore RNG state after intermittent average-policy training.",
    )
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
    if args.variants:
        requested = {v.strip() for v in args.variants.split(",") if v.strip()}
        variants = [v for v in config["variants"] if v["variant_id"] in requested]
        missing = sorted(requested - {v["variant_id"] for v in variants})
        if missing:
            raise ValueError(f"Unknown variant id(s): {', '.join(missing)}")
        config["variants"] = variants
    if args.allow_policy_training_rng_advance:
        config["isolate_policy_training_rng"] = False
    return config


def main() -> None:
    run_experiment(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
