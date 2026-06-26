"""Run the DREAM network-size ablation."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from dream_poker.experiment_runner import (
    CORE_SUMMARY_METRICS,
    build_paired_differences,
    create_timestamped_output_dir,
    json_ready,
    make_dream_solver,
    write_json,
)
from dream_poker.experiment_utils import average_policy_value_target, cleanup_training_memory, ensure_dir
from dream_poker.plotting import format_plot_title, plot_curve_by_variant
from dream_poker.variant_ablation import (
    add_variant_curve_columns,
    aggregate_variant_summary,
    apply_variant_overrides,
    create_variant_ablation_plots,
    get_variant_id,
    get_variant_label,
    paired_difference_summary,
    summarise_variant_curve,
    write_multiseed_npz,
)

from .config import BASELINE_VARIANT, EXPERIMENT_CONFIG, NETWORK_SIZE_VARIANT_SETS, NETWORK_SIZE_VARIANTS


TREATMENT_KEYS = [
    "network_architecture",
    "network_depth",
    "network_max_width",
    "network_hidden_units",
]
PARAMETER_COUNT_COLUMNS = [
    "policy_network_parameters",
    "advantage_network_parameters_per_player",
    "baseline_network_parameters_per_player",
    "total_network_parameters",
]
PAIRED_METRICS = CORE_SUMMARY_METRICS + [
    "exploitability_auc_by_nodes",
    "final_advantage_target_variance",
    "final_baseline_reward_variance",
    "final_policy_loss",
]
AGGREGATE_METRICS = PAIRED_METRICS + [
    "best_policy_value_error",
    "nodes_to_threshold",
    "iterations_to_threshold",
    "final_policy_entropy_mean",
    "final_policy_gradient_steps_total",
    *PARAMETER_COUNT_COLUMNS,
]


def count_parameters(module) -> int:
    return int(sum(param.numel() for param in module.parameters()))


def solver_parameter_counts(solver) -> Dict[str, int]:
    policy_parameters = count_parameters(solver._policy_network)
    advantage_per_player = count_parameters(solver._advantage_networks[0])
    baseline_per_player = count_parameters(solver._baseline_networks[0])
    num_players = int(solver._num_players)
    return {
        "policy_network_parameters": policy_parameters,
        "advantage_network_parameters_per_player": advantage_per_player,
        "baseline_network_parameters_per_player": baseline_per_player,
        "total_network_parameters": policy_parameters
        + num_players * advantage_per_player
        + num_players * baseline_per_player,
    }


def add_parameter_count_columns(curves: pd.DataFrame, parameter_counts: Dict[str, int]) -> pd.DataFrame:
    curves = curves.copy()
    for key, value in parameter_counts.items():
        curves[key] = int(value)
    return curves


def summarise_network_size_curve(
    curves: pd.DataFrame,
    seed: int,
    variant: Dict,
    config: Dict,
    parameter_counts: Dict[str, int],
) -> Dict:
    summary = summarise_variant_curve(curves, seed, variant, config, TREATMENT_KEYS)
    for key, value in parameter_counts.items():
        summary[key] = int(value)
    summary["policy_network_layers"] = list(config["policy_network_layers"])
    summary["advantage_network_layers"] = list(config["advantage_network_layers"])
    summary["baseline_network_layers"] = list(config["baseline_network_layers"])
    return summary


def run_single_variant_seed(
    variant: Dict,
    seed: int,
    base_config: Dict,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict]:
    config = apply_variant_overrides(base_config, variant)
    solver = make_dream_solver(config, seed)
    parameter_counts = solver_parameter_counts(solver)
    curves = solver.solve(isolate_policy_training_rng=config.get("isolate_policy_training_rng", True))
    curves = add_variant_curve_columns(curves, config, variant, TREATMENT_KEYS)
    curves = add_parameter_count_columns(curves, parameter_counts)
    curves.insert(0, "seed", int(seed))
    front_cols = [
        "seed",
        "variant",
        "variant_label",
        *TREATMENT_KEYS,
        *PARAMETER_COUNT_COLUMNS,
    ]
    curves = curves[front_cols + [col for col in curves.columns if col not in front_cols]]

    seed_dir = ensure_dir(output_dir / get_variant_id(variant) / f"seed_{seed}")
    curves.to_csv(seed_dir / "checkpoint_curves.csv", index=False)
    summary = summarise_network_size_curve(curves, seed, variant, config, parameter_counts)
    write_json(seed_dir / "seed_summary.json", summary)
    del solver
    cleanup_training_memory()
    return curves, summary


def run_experiment(
    config: Dict,
    variants: Sequence[Dict] = NETWORK_SIZE_VARIANTS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    output_dir = create_timestamped_output_dir(config["output_root"])
    metadata = {**config, "variants": list(variants), "baseline_variant": BASELINE_VARIANT}
    write_json(output_dir / "experiment_metadata.json", json_ready(metadata))

    all_curves = []
    summaries = []
    for variant in variants:
        for seed in config["seeds"]:
            print(f"Running {get_variant_id(variant)} seed {seed}...")
            curves, summary = run_single_variant_seed(variant, int(seed), config, output_dir)
            all_curves.append(curves)
            summaries.append(summary)

    curves_df = pd.concat(all_curves, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    paired_df = build_paired_differences(summary_df, BASELINE_VARIANT, PAIRED_METRICS)
    aggregate_df = aggregate_variant_summary(summary_df, AGGREGATE_METRICS, TREATMENT_KEYS)

    curves_df.to_csv(output_dir / "checkpoint_curves_by_variant.csv", index=False)
    summary_df.to_csv(output_dir / "seed_variant_summary.csv", index=False)
    paired_df.to_csv(output_dir / "paired_differences_vs_baseline.csv", index=False)
    aggregate_df.to_csv(output_dir / "aggregate_summary_by_variant.csv", index=False)
    write_json(output_dir / "aggregate_summary_by_variant.json", aggregate_df.to_dict(orient="records"))
    write_json(output_dir / "paired_difference_summary.json", paired_difference_summary(paired_df))
    write_multiseed_npz(
        curves_df,
        output_dir / "multiseed_curves_by_variant.npz",
        extra_columns=[
            "network_depth",
            "network_max_width",
            "network_hidden_units",
            *PARAMETER_COUNT_COLUMNS,
            "advantage_target_variance",
            "baseline_reward_variance_sampled",
        ],
    )

    make_plots(curves_df, summary_df, paired_df, variants, output_dir, config)
    print(f"Outputs written to {output_dir}")
    return curves_df, summary_df, paired_df, output_dir


def plot_capacity_metric(
    summary_df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    title_config: Dict | None = None,
) -> None:
    grouped = (
        summary_df.groupby(["variant", "variant_label", "total_network_parameters"], sort=False)[metric]
        .agg(["mean", "sem"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.errorbar(
        grouped["total_network_parameters"],
        grouped["mean"],
        yerr=grouped["sem"].fillna(0.0),
        fmt="o",
        capsize=5,
    )
    for _, row in grouped.iterrows():
        ax.annotate(
            row["variant_label"],
            (row["total_network_parameters"], row["mean"]),
            textcoords="offset points",
            xytext=(4, 5),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_title(format_plot_title(title, title_config))
    ax.set_xlabel("Total trainable parameters across DREAM networks")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    curves_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    variants: Sequence[Dict],
    output_dir: Path,
    config: Dict,
) -> None:
    plot_prefix = str(config.get("plot_prefix", "dream_network_size"))
    title_prefix = str(config.get("plot_title", "DREAM Network-Size Ablation"))
    value_target = average_policy_value_target(config)
    create_variant_ablation_plots(
        curves_df,
        summary_df,
        paired_df,
        variants,
        output_dir,
        plot_prefix=plot_prefix,
        title_prefix=title_prefix,
        diagnostic_metrics=[
            ("advantage_target_variance", "Advantage-Target Variance", "Advantage-target variance"),
            ("baseline_reward_variance_sampled", "Baseline-Replay Reward Variance", "Sampled reward variance"),
            ("policy_entropy_mean", "Policy Entropy", "Mean policy entropy"),
        ],
        average_policy_value_target=value_target,
        title_config=config,
    )

    plot_dir = ensure_dir(output_dir / "plots")
    variant_order = [get_variant_id(variant) for variant in variants]
    variant_labels = {get_variant_id(variant): get_variant_label(variant) for variant in variants}
    plot_curve_by_variant(
        curves_df,
        "nodes_touched",
        "policy_loss",
        f"{title_prefix}: Policy Loss by Nodes Touched",
        "Policy loss",
        variant_order,
        variant_labels,
        plot_dir / f"{plot_prefix}_policy_loss_by_nodes.png",
        title_config=config,
    )
    plot_capacity_metric(
        summary_df,
        "final_exploitability",
        f"{title_prefix}: Final Exploitability by Parameter Count",
        "Final exploitability",
        plot_dir / f"{plot_prefix}_final_exploitability_by_parameters.png",
        title_config=config,
    )
    plot_capacity_metric(
        summary_df,
        "exploitability_auc_by_nodes",
        f"{title_prefix}: Node-Efficiency AUC by Parameter Count",
        "Exploitability AUC by nodes",
        plot_dir / f"{plot_prefix}_node_auc_by_parameters.png",
        title_config=config,
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list, e.g. 1234,2025")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
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
        "--allow-policy-training-rng-advance",
        action="store_true",
        help="Do not restore RNG state after intermittent average-policy training.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated variant ids to run; overrides --variant-set when provided.",
    )
    parser.add_argument(
        "--variant-set",
        type=str,
        default="width_sweep",
        choices=sorted(NETWORK_SIZE_VARIANT_SETS),
        help=(
            "Named variant subset to run. Cloud runs should prefer width_sweep, "
            "depth_sweep, or capacity_extremes instead of the full all set."
        ),
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace, base_config: Dict = EXPERIMENT_CONFIG) -> Dict:
    config = copy.deepcopy(base_config)
    if args.seeds:
        config["seeds"] = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.iterations is not None:
        config["num_iterations"] = int(args.iterations)
    if args.traversals is not None:
        config["num_traversals"] = int(args.traversals)
    if args.learning_rate is not None:
        config["learning_rate"] = float(args.learning_rate)
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
    if args.allow_policy_training_rng_advance:
        config["isolate_policy_training_rng"] = False
    return config


def variants_from_args(args: argparse.Namespace) -> list[Dict]:
    variants = copy.deepcopy(NETWORK_SIZE_VARIANTS)
    if args.variants:
        requested = {value.strip() for value in args.variants.split(",") if value.strip()}
    else:
        requested = set(NETWORK_SIZE_VARIANT_SETS[args.variant_set])
    selected = [variant for variant in variants if get_variant_id(variant) in requested]
    missing = sorted(requested - {get_variant_id(variant) for variant in selected})
    if missing:
        raise ValueError(f"Unknown variant ids: {missing}")
    return selected


def main() -> None:
    args = parse_args()
    run_experiment(config_from_args(args), variants_from_args(args))


if __name__ == "__main__":
    main()
