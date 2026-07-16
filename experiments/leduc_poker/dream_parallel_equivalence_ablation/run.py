"""Run Experiment 21's sequential/parallel DREAM comparison."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from dream_poker.experiment_runner import (
    build_paired_differences,
    create_timestamped_output_dir,
    json_ready,
    make_dream_solver,
    write_json,
)
from dream_poker.experiment_utils import (
    average_policy_value_target,
    cleanup_training_memory,
    ensure_average_policy_value_columns,
    ensure_dir,
    safe_mean,
    standard_error,
)
from dream_poker.parallel_utils import equivalence_summary
from dream_poker.plotting import plot_curve_by_variant, plot_metric_bar_by_variant
from dream_poker.variant_ablation import (
    add_variant_curve_columns,
    aggregate_variant_summary,
    apply_variant_overrides,
    get_variant_id,
    get_variant_label,
    paired_difference_summary,
    summarise_variant_curve,
    write_multiseed_npz,
)

from .config import (
    BASELINE_VARIANT_ID,
    DEFAULT_SEEDS,
    EXPERIMENT_CONFIG,
    FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN,
    FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN,
    PARALLEL_NUM_WORKERS,
    VARIANTS,
)


TREATMENT_KEYS = [
    "execution_backend",
    "parallel_num_workers",
    "network_treatment",
    "network_architecture",
    "policy_network_layers",
    "advantage_network_layers",
    "baseline_network_layers",
]
PAIRED_METRICS = [
    "final_exploitability",
    "final_average_policy_value",
    "final_policy_value_error",
    "final_nodes_touched",
    "final_wall_clock_seconds",
    "solver_initialization_seconds",
    "training_loop_seconds",
    "end_to_end_seconds",
    "final_cumulative_traversal_seconds",
    "final_cumulative_supervised_training_seconds",
    "final_cumulative_parallel_sync_seconds",
    "final_cumulative_replay_refresh_seconds",
]
RATIO_FIELDS = {
    "training_loop_speedup": "training_loop_seconds",
    "end_to_end_speedup": "end_to_end_seconds",
    "traversal_speedup": "final_cumulative_traversal_seconds",
    "supervised_training_speedup": "final_cumulative_supervised_training_seconds",
}
AGGREGATE_METRICS = PAIRED_METRICS + [
    "best_exploitability",
    "exploitability_auc_by_nodes",
    "final_policy_loss",
]
EQUIVALENCE_TOLERANCES = {
    "final_exploitability": FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN,
    "final_average_policy_value": FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--variant-ids", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--advantage-network-train-steps", type=int, default=None)
    parser.add_argument("--baseline-network-train-steps", type=int, default=None)
    parser.add_argument("--policy-network-train-every", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--batch-size-advantage", type=int, default=None)
    parser.add_argument("--batch-size-strategy", type=int, default=None)
    parser.add_argument("--batch-size-baseline", type=int, default=None)
    parser.add_argument("--parallel-num-workers", type=int, default=None)
    parser.add_argument("--parallel-ray-address", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--allow-policy-training-rng-advance",
        action="store_true",
        help="Do not restore RNG state after intermittent average-policy training.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Dict:
    config = copy.deepcopy(EXPERIMENT_CONFIG)
    if args.seeds:
        config["seeds"] = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
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
    if args.batch_size_advantage is not None:
        config["batch_size_advantage"] = int(args.batch_size_advantage)
    if args.batch_size_strategy is not None:
        config["batch_size_strategy"] = int(args.batch_size_strategy)
    if args.batch_size_baseline is not None:
        config["batch_size_baseline"] = int(args.batch_size_baseline)
    if args.parallel_num_workers is not None:
        config["parallel_num_workers"] = int(args.parallel_num_workers)
    if args.parallel_ray_address is not None:
        config["parallel_ray_address"] = args.parallel_ray_address
    if args.output_root is not None:
        config["output_root"] = args.output_root
    if args.allow_policy_training_rng_advance:
        config["isolate_policy_training_rng"] = False
    return config


def variants_from_args(args: argparse.Namespace, config: Dict) -> list[Dict]:
    variants = copy.deepcopy(VARIANTS)
    for variant in variants:
        if get_variant_id(variant) == BASELINE_VARIANT_ID:
            variant["parallel_num_workers"] = 1
        else:
            variant["parallel_num_workers"] = int(config.get("parallel_num_workers", PARALLEL_NUM_WORKERS))
    if not args.variant_ids:
        return variants
    requested = {value.strip() for value in args.variant_ids.split(",") if value.strip()}
    selected = [variant for variant in variants if get_variant_id(variant) in requested]
    missing = sorted(requested - {get_variant_id(variant) for variant in selected})
    if missing:
        raise ValueError(f"Unknown variant ids: {missing}")
    return selected


def run_single_variant_seed(
    variant: Dict,
    seed: int,
    base_config: Dict,
    output_dir: Path,
) -> tuple[pd.DataFrame, Dict]:
    config = apply_variant_overrides(base_config, variant)
    init_start = time.perf_counter()
    solver = make_dream_solver(config, seed)
    solver_initialization_seconds = time.perf_counter() - init_start
    try:
        curves = solver.solve(
            isolate_policy_training_rng=config.get("isolate_policy_training_rng", True)
        )
    finally:
        if hasattr(solver, "close"):
            solver.close()
    curves = ensure_average_policy_value_columns(curves, config.get("average_policy_value_target"))
    curves = add_variant_curve_columns(curves, config, variant, TREATMENT_KEYS)
    curves.insert(0, "seed", int(seed))
    curves["solver_initialization_seconds"] = float(solver_initialization_seconds)
    curves["training_loop_seconds"] = curves["wall_clock_seconds"].astype(float)
    curves["end_to_end_seconds"] = (
        curves["solver_initialization_seconds"] + curves["training_loop_seconds"]
    )
    front_cols = [
        "seed",
        "variant",
        "variant_label",
        *TREATMENT_KEYS,
        "solver_initialization_seconds",
        "training_loop_seconds",
        "end_to_end_seconds",
    ]
    curves = curves[front_cols + [col for col in curves.columns if col not in front_cols]]

    seed_dir = ensure_dir(output_dir / get_variant_id(variant) / f"seed_{seed}")
    curves.to_csv(seed_dir / "checkpoint_curves.csv", index=False)
    summary = summarise_variant_curve(curves, seed, variant, config, TREATMENT_KEYS)
    final_row = curves.iloc[-1]
    summary["solver_initialization_seconds"] = float(solver_initialization_seconds)
    summary["training_loop_seconds"] = float(final_row["training_loop_seconds"])
    summary["end_to_end_seconds"] = float(final_row["end_to_end_seconds"])
    write_json(seed_dir / "seed_summary.json", summary)
    del solver
    cleanup_training_memory()
    return curves, summary


def paired_speedup_rows(
    paired_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    baseline_variant: str,
    ratio_fields: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    for _, paired in paired_df.iterrows():
        seed = int(paired["seed"])
        variant = str(paired["variant"])
        base = summary_df[(summary_df["variant"] == baseline_variant) & (summary_df["seed"] == seed)]
        comp = summary_df[(summary_df["variant"] == variant) & (summary_df["seed"] == seed)]
        if base.empty or comp.empty:
            continue
        row = {"seed": seed, "baseline_variant": baseline_variant, "variant": variant}
        for ratio_name, metric in ratio_fields.items():
            baseline_value = float(base.iloc[0].get(metric, np.nan))
            variant_value = float(comp.iloc[0].get(metric, np.nan))
            row[ratio_name] = (
                baseline_value / variant_value
                if np.isfinite(baseline_value)
                and np.isfinite(variant_value)
                and variant_value > 0.0
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_equivalence_summary(paired_df: pd.DataFrame) -> Dict:
    summary = {}
    for metric, margin in EQUIVALENCE_TOLERANCES.items():
        col = f"delta_{metric}"
        summary[metric] = equivalence_summary(paired_df[col], margin) if col in paired_df else {}
    return summary


def speedup_summary(speedup_df: pd.DataFrame) -> Dict:
    return {
        col: {
            "mean": safe_mean(speedup_df[col]),
            "se": standard_error(speedup_df[col]),
        }
        for col in speedup_df.columns
        if col.endswith("_speedup")
    }


def make_plots(
    curves_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    variants: Sequence[Dict],
    output_dir: Path,
    config: Dict,
) -> None:
    plot_dir = ensure_dir(output_dir / "plots")
    variant_order = [get_variant_id(variant) for variant in variants]
    variant_labels = {get_variant_id(variant): get_variant_label(variant) for variant in variants}
    title_prefix = str(config.get("plot_title", "DREAM Sequential versus Ray-Parallel"))
    target = average_policy_value_target(config)
    for y_col, title, ylabel, filename in [
        ("exploitability", "Exploitability by Nodes Touched", "Exploitability", "exploitability_by_nodes.png"),
        (
            "average_policy_value",
            "Average-Policy Value by Nodes Touched",
            "Average-policy value",
            "average_policy_value_by_nodes.png",
        ),
        (
            "policy_value_error",
            "Policy-Value Error by Nodes Touched",
            "Absolute error from Leduc game value",
            "policy_value_error_by_nodes.png",
        ),
        (
            "wall_clock_seconds",
            "Wall-Clock Time by Nodes Touched",
            "Wall-clock seconds",
            "wall_clock_seconds_by_nodes.png",
        ),
        (
            "cumulative_traversal_seconds",
            "Traversal Collection Time by Nodes Touched",
            "Cumulative traversal seconds",
            "traversal_seconds_by_nodes.png",
        ),
    ]:
        if y_col in curves_df.columns:
            plot_curve_by_variant(
                curves_df,
                "nodes_touched",
                y_col,
                f"{title_prefix}: {title}",
                ylabel,
                variant_order,
                variant_labels,
                plot_dir / filename,
                average_policy_value_target=target,
                title_config=config,
            )

    for metric, title, ylabel, filename in [
        (
            "final_exploitability",
            "Final Exploitability",
            "Final exploitability",
            "final_exploitability_by_variant.png",
        ),
        (
            "end_to_end_seconds",
            "End-to-End Runtime",
            "Seconds",
            "end_to_end_seconds_by_variant.png",
        ),
        (
            "training_loop_seconds",
            "Training-Loop Runtime",
            "Seconds",
            "training_loop_seconds_by_variant.png",
        ),
    ]:
        if metric in summary_df.columns:
            plot_metric_bar_by_variant(
                summary_df,
                metric,
                f"{title_prefix}: {title}",
                ylabel,
                variant_order,
                variant_labels,
                plot_dir / filename,
                average_policy_value_target=target,
                title_config=config,
            )


def run_experiment(
    config: Dict,
    variants: Sequence[Dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    output_dir = create_timestamped_output_dir(config["output_root"])
    baseline_variant = str(config.get("baseline_variant", BASELINE_VARIANT_ID))
    metadata = {**config, "variants": list(variants), "baseline_variant": baseline_variant}
    write_json(output_dir / "experiment_metadata.json", json_ready(metadata))

    all_curves = []
    summaries = []
    for variant in variants:
        for seed in config.get("seeds", DEFAULT_SEEDS):
            print(f"Running {get_variant_id(variant)} seed {seed}...", flush=True)
            curves, summary = run_single_variant_seed(variant, int(seed), config, output_dir)
            all_curves.append(curves)
            summaries.append(summary)

    curves_df = pd.concat(all_curves, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    paired_df = build_paired_differences(summary_df, baseline_variant, PAIRED_METRICS)
    speedup_df = paired_speedup_rows(paired_df, summary_df, baseline_variant, RATIO_FIELDS)
    aggregate_df = aggregate_variant_summary(summary_df, AGGREGATE_METRICS, TREATMENT_KEYS)

    curves_df.to_csv(output_dir / "checkpoint_curves_by_variant.csv", index=False)
    summary_df.to_csv(output_dir / "seed_variant_summary.csv", index=False)
    paired_df.to_csv(output_dir / "paired_differences_vs_baseline.csv", index=False)
    speedup_df.to_csv(output_dir / "paired_speedups_vs_baseline.csv", index=False)
    aggregate_df.to_csv(output_dir / "aggregate_summary_by_variant.csv", index=False)
    write_json(output_dir / "aggregate_summary_by_variant.json", aggregate_df.to_dict(orient="records"))
    write_json(output_dir / "paired_difference_summary.json", paired_difference_summary(paired_df))
    write_json(output_dir / "paired_equivalence_summary.json", build_equivalence_summary(paired_df))
    write_json(output_dir / "paired_speedup_summary.json", speedup_summary(speedup_df))
    write_multiseed_npz(
        curves_df,
        output_dir / "multiseed_curves_by_variant.npz",
        extra_columns=[
            "parallel_num_workers",
            "wall_clock_seconds",
            "cumulative_traversal_seconds",
            "cumulative_supervised_training_seconds",
            "cumulative_parallel_sync_seconds",
            "cumulative_replay_refresh_seconds",
        ],
    )
    make_plots(curves_df, summary_df, variants, output_dir, config)
    print(f"Outputs written to {output_dir}", flush=True)
    return curves_df, summary_df, paired_df, output_dir


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    run_experiment(config, variants_from_args(args, config))


if __name__ == "__main__":
    main()
