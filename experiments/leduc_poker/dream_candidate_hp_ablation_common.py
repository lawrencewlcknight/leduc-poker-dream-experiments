"""Shared helpers for DREAM candidate-baseline hyperparameter ablations."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Sequence, Tuple

import pandas as pd

from dream_poker.constants import (
    DEFAULT_SEEDS_5,
    EXPLOITABILITY_THRESHOLD,
    LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    LEDUC_GAME_VALUE_P0,
    SMOKE_TEST_SEEDS,
    THESIS_SEEDS_10,
)
from dream_poker.experiment_runner import (
    CORE_SUMMARY_METRICS,
    build_paired_differences,
    create_timestamped_output_dir,
    json_ready,
    make_dream_solver,
    write_json,
)
from dream_poker.experiment_utils import average_policy_value_target, cleanup_training_memory, ensure_dir
from dream_poker.variant_ablation import (
    add_variant_curve_columns,
    aggregate_variant_summary,
    apply_variant_overrides,
    create_variant_ablation_plots,
    get_variant_id,
    paired_difference_summary,
    summarise_variant_curve,
    write_multiseed_npz,
)
from experiments.leduc_poker.dream_architecture_candidate_comparison.config import (
    ADVANTAGE_CANDIDATE_LAYERS,
    POLICY_BASELINE_LAYERS,
)


DEFAULT_SEEDS = [1234, 2025, 31415]
DEFAULT_EPSILON = 0.10


BASE_CANDIDATE_HP_CONFIG = {
    "game_name": "leduc_poker",
    "algorithm": "DREAM-style OpenSpiel candidate-baseline HP ablation",
    "num_iterations": 175,
    "num_traversals": 160,
    "evaluation_interval": 25,
    "policy_network_train_every": 25,
    "policy_network_train_steps": 100,
    "advantage_network_train_steps": 50,
    "baseline_network_train_steps": 50,
    "policy_network_layers": list(POLICY_BASELINE_LAYERS),
    "advantage_network_layers": list(ADVANTAGE_CANDIDATE_LAYERS),
    "baseline_network_layers": list(POLICY_BASELINE_LAYERS),
    "policy_network_type": "mlp",
    "advantage_network_type": "mlp",
    "baseline_network_type": "mlp",
    "network_treatment": "architecture_selected_candidate",
    "learning_rate": 0.003,
    "batch_size_advantage": 1024,
    "batch_size_strategy": 1024,
    "batch_size_baseline": 1024,
    "advantage_memory_capacity": int(1e6),
    "strategy_memory_capacity": int(1e6),
    "baseline_memory_capacity": int(1e6),
    "epsilon": DEFAULT_EPSILON,
    "compute_exploitability": True,
    "isolate_policy_training_rng": True,
    "average_strategy_weighting": "linear",
    "seeds": list(DEFAULT_SEEDS),
    "optional_development_seeds_5": DEFAULT_SEEDS_5,
    "optional_thesis_seeds_10": THESIS_SEEDS_10,
    "game_value_player_0": LEDUC_GAME_VALUE_P0,
    "average_policy_value_target": LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": EXPLOITABILITY_THRESHOLD,
}


DEFAULT_DIAGNOSTIC_METRICS = [
    ("advantage_target_variance", "Advantage-Target Variance", "Advantage-target variance"),
    ("baseline_reward_variance_sampled", "Baseline-Replay Reward Variance", "Sampled reward variance"),
    ("policy_entropy_mean", "Average-Policy Entropy", "Policy entropy"),
    ("strategy_buffer_size", "Strategy-Memory Size", "Strategy-memory entries"),
]


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


def make_variant(
    variant_id: str,
    label: str,
    *,
    hp_family: str,
    hp_value: str,
    description: str,
    **overrides,
) -> dict:
    """Create a labelled candidate-baseline hyperparameter variant."""
    return {
        "variant_id": variant_id,
        "label": label,
        "hp_family": hp_family,
        "hp_value": hp_value,
        "description": description,
        **overrides,
    }


def make_candidate_hp_experiment_config(
    *,
    experiment_name: str,
    algorithm: str,
    plot_prefix: str,
    plot_title: str,
    output_subdir: str,
    baseline_variant: str,
    variants: Sequence[Dict],
    treatment_keys: Sequence[str],
) -> Dict:
    config = copy.deepcopy(BASE_CANDIDATE_HP_CONFIG)
    config.update(
        {
            "experiment_name": experiment_name,
            "algorithm": algorithm,
            "plot_prefix": plot_prefix,
            "plot_title": plot_title,
            "baseline_variant": baseline_variant,
            "ablation_variants": list(variants),
            "treatment_keys": list(treatment_keys),
            "output_root": Path("outputs") / output_subdir,
        }
    )
    return config


def make_smoke_test_config_overrides(output_subdir: str) -> Dict:
    return {
        "seeds": SMOKE_TEST_SEEDS[:1],
        "num_iterations": 3,
        "num_traversals": 4,
        "policy_network_train_steps": 1,
        "advantage_network_train_steps": 1,
        "baseline_network_train_steps": 1,
        "policy_network_train_every": 1,
        "evaluation_interval": 1,
        "output_root": Path("outputs") / "smoke_tests" / output_subdir,
    }


def run_single_variant_seed(
    variant: Dict,
    seed: int,
    base_config: Dict,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict]:
    treatment_keys = list(base_config["treatment_keys"])
    config = apply_variant_overrides(base_config, variant)
    solver = make_dream_solver(config, seed)
    curves = solver.solve(isolate_policy_training_rng=config.get("isolate_policy_training_rng", True))
    curves = add_variant_curve_columns(curves, config, variant, treatment_keys)
    curves.insert(0, "seed", int(seed))
    front_cols = ["seed", "variant", "variant_label", *treatment_keys]
    curves = curves[front_cols + [col for col in curves.columns if col not in front_cols]]

    seed_dir = ensure_dir(output_dir / get_variant_id(variant) / f"seed_{seed}")
    curves.to_csv(seed_dir / "checkpoint_curves.csv", index=False)
    summary = summarise_variant_curve(curves, seed, variant, config, treatment_keys)
    write_json(seed_dir / "seed_summary.json", summary)
    if hasattr(solver, "close"):
        solver.close()
    del solver
    cleanup_training_memory()
    return curves, summary


def run_parameter_ablation(
    config: Dict,
    variants: Sequence[Dict] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    variants = list(config["ablation_variants"] if variants is None else variants)
    treatment_keys = list(config["treatment_keys"])
    output_dir = create_timestamped_output_dir(config["output_root"])
    baseline_variant = str(config["baseline_variant"])
    metadata = {**config, "variants": list(variants), "baseline_variant": baseline_variant}
    write_json(output_dir / "experiment_metadata.json", json_ready(metadata))

    all_curves = []
    summaries = []
    for variant in variants:
        for seed in config["seeds"]:
            print(f"Running {get_variant_id(variant)} seed {seed}...", flush=True)
            curves, summary = run_single_variant_seed(variant, int(seed), config, output_dir)
            all_curves.append(curves)
            summaries.append(summary)

    curves_df = pd.concat(all_curves, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    paired_df = build_paired_differences(summary_df, baseline_variant, PAIRED_METRICS)
    aggregate_df = aggregate_variant_summary(summary_df, AGGREGATE_METRICS, treatment_keys)

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
            *treatment_keys,
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
        plot_prefix=str(config["plot_prefix"]),
        title_prefix=str(config["plot_title"]),
        diagnostic_metrics=DEFAULT_DIAGNOSTIC_METRICS,
        average_policy_value_target=average_policy_value_target(config),
        title_config=config,
    )
    print(f"Outputs written to {output_dir}", flush=True)
    return curves_df, summary_df, paired_df, output_dir


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list, e.g. 1234,2025")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--advantage-network-train-steps", type=int, default=None)
    parser.add_argument("--baseline-network-train-steps", type=int, default=None)
    parser.add_argument("--policy-network-train-every", type=int, default=None)
    parser.add_argument("--batch-size-advantage", type=int, default=None)
    parser.add_argument("--batch-size-strategy", type=int, default=None)
    parser.add_argument("--batch-size-baseline", type=int, default=None)
    parser.add_argument("--advantage-memory-capacity", type=int, default=None)
    parser.add_argument("--strategy-memory-capacity", type=int, default=None)
    parser.add_argument("--baseline-memory-capacity", type=int, default=None)
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
    return parser


def config_from_args(args: argparse.Namespace, base_config: Dict) -> Dict:
    config = copy.deepcopy(base_config)
    if args.seeds:
        config["seeds"] = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.iterations is not None:
        config["num_iterations"] = int(args.iterations)
    if args.traversals is not None:
        config["num_traversals"] = int(args.traversals)
    if args.learning_rate is not None:
        config["learning_rate"] = float(args.learning_rate)
    if args.epsilon is not None:
        config["epsilon"] = float(args.epsilon)
    if args.policy_network_train_steps is not None:
        config["policy_network_train_steps"] = int(args.policy_network_train_steps)
    if args.advantage_network_train_steps is not None:
        config["advantage_network_train_steps"] = int(args.advantage_network_train_steps)
    if args.baseline_network_train_steps is not None:
        config["baseline_network_train_steps"] = int(args.baseline_network_train_steps)
    if args.batch_size_advantage is not None:
        config["batch_size_advantage"] = int(args.batch_size_advantage)
    if args.batch_size_strategy is not None:
        config["batch_size_strategy"] = int(args.batch_size_strategy)
    if args.batch_size_baseline is not None:
        config["batch_size_baseline"] = int(args.batch_size_baseline)
    if args.advantage_memory_capacity is not None:
        config["advantage_memory_capacity"] = int(args.advantage_memory_capacity)
    if args.strategy_memory_capacity is not None:
        config["strategy_memory_capacity"] = int(args.strategy_memory_capacity)
    if args.baseline_memory_capacity is not None:
        config["baseline_memory_capacity"] = int(args.baseline_memory_capacity)
    interval = args.policy_network_train_every if args.policy_network_train_every is not None else args.evaluation_interval
    if interval is not None:
        config["policy_network_train_every"] = int(interval)
        config["evaluation_interval"] = int(interval)
    if args.output_root is not None:
        config["output_root"] = args.output_root
    if args.allow_policy_training_rng_advance:
        config["isolate_policy_training_rng"] = False
    return config


def variants_from_args(args: argparse.Namespace, variants: Sequence[Dict]) -> list[Dict]:
    variants = copy.deepcopy(list(variants))
    if not args.variants:
        return variants
    requested = {value.strip() for value in args.variants.split(",") if value.strip()}
    selected = [variant for variant in variants if get_variant_id(variant) in requested]
    missing = sorted(requested - {get_variant_id(variant) for variant in selected})
    if missing:
        raise ValueError(f"Unknown variant ids: {missing}")
    return selected
