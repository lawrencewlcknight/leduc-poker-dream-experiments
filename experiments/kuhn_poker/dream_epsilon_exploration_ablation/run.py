"""Run the DREAM epsilon-exploration ablation."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Sequence, Tuple

import pandas as pd

from dream_poker.experiment_runner import (
    CORE_SUMMARY_METRICS,
    build_paired_differences,
    create_timestamped_output_dir,
    json_ready,
    make_dream_solver,
    write_json,
)
from dream_poker.experiment_utils import average_policy_value_target, ensure_dir
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

from .config import BASELINE_VARIANT, EPSILON_VARIANTS, EXPERIMENT_CONFIG


TREATMENT_KEYS = ["epsilon"]
PAIRED_METRICS = CORE_SUMMARY_METRICS + [
    "exploitability_auc_by_nodes",
    "final_advantage_target_variance",
    "final_baseline_reward_variance",
    "final_policy_loss",
    "final_policy_entropy_mean",
]
AGGREGATE_METRICS = PAIRED_METRICS + [
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


def run_single_variant_seed(
    variant: Dict,
    seed: int,
    base_config: Dict,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict]:
    config = apply_variant_overrides(base_config, variant)
    solver = make_dream_solver(config, seed)
    curves = solver.solve(isolate_policy_training_rng=config.get("isolate_policy_training_rng", True))
    curves = add_variant_curve_columns(curves, config, variant, TREATMENT_KEYS)
    curves.insert(0, "seed", int(seed))
    front_cols = ["seed", "variant", "variant_label", *TREATMENT_KEYS]
    curves = curves[front_cols + [col for col in curves.columns if col not in front_cols]]

    seed_dir = ensure_dir(output_dir / get_variant_id(variant) / f"seed_{seed}")
    curves.to_csv(seed_dir / "checkpoint_curves.csv", index=False)
    summary = summarise_variant_curve(curves, seed, variant, config, TREATMENT_KEYS)
    write_json(seed_dir / "seed_summary.json", summary)
    return curves, summary


def run_experiment(
    config: Dict,
    variants: Sequence[Dict] = EPSILON_VARIANTS,
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
            "epsilon",
            "advantage_target_variance",
            "baseline_reward_variance_sampled",
            "policy_entropy_mean",
            "strategy_buffer_size",
        ],
    )

    create_variant_ablation_plots(
        curves_df,
        summary_df,
        paired_df,
        variants,
        output_dir,
        plot_prefix="dream_epsilon_exploration",
        title_prefix="DREAM Epsilon-Exploration Ablation",
        diagnostic_metrics=[
            ("advantage_target_variance", "Advantage-Target Variance", "Advantage-target variance"),
            ("baseline_reward_variance_sampled", "Baseline-Replay Reward Variance", "Sampled reward variance"),
            ("policy_entropy_mean", "Average-Policy Entropy", "Policy entropy"),
            ("strategy_buffer_size", "Strategy-Memory Size", "Strategy-memory entries"),
        ],
        average_policy_value_target=average_policy_value_target(config),
    )
    print(f"Outputs written to {output_dir}")
    return curves_df, summary_df, paired_df, output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    variants = copy.deepcopy(EPSILON_VARIANTS)
    if not args.variants:
        return variants
    requested = {value.strip() for value in args.variants.split(",") if value.strip()}
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
