"""Run Experiment 44: time-budgeted DREAM frozen-reservoir audit."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dream_exp44_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dream_exp44_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

try:
    import pyspiel  # noqa: E402
except Exception:  # pragma: no cover
    pyspiel = None

from dream_poker.experiment_runner import make_dream_solver, write_json  # noqa: E402
from dream_poker.experiment_utils import cleanup_training_memory, standard_error  # noqa: E402

from .config import (  # noqa: E402
    ARM_LABELS,
    ARM_ORDER,
    DISTILLATION_SEED_OFFSET,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXTENDED_FIT_MULTIPLIER,
    GROUPED_CE_EXTENDED,
    PRODUCTION_SEEDS,
    TRAINING_CONFIG,
    ROW_MSE,
    smoke_config,
    task_schedule,
    validate_contract,
)
from .distillation import (  # noqa: E402
    EmpiricalReservoirPolicy,
    NetworkPolicy,
    exact_policy_metrics,
    fit_frozen_policy,
    freeze_strategy_reservoir,
    group_reservoir,
    initial_policy_state,
    load_frozen_reservoir,
    save_frozen_reservoir,
    save_grouped_reservoir,
    save_policy_model,
)


def task_name(task_index: int, seed: int) -> str:
    return f"task_{int(task_index):03d}_seed_{int(seed)}"


def _json_config(config: Mapping) -> Dict:
    return {
        str(key): str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }


def _fit_steps(solver, config: Mapping, *, smoke: bool) -> int:
    if smoke:
        return int(config.get("smoke_reference_fit_steps", 2))
    if config.get("fresh_reference_fit_steps") is not None:
        return int(config["fresh_reference_fit_steps"])
    # Match the number of policy-gradient steps consumed by the warm-started
    # Experiment 43 extractor over this seed's realised 12-hour trajectory.
    return max(int(config["policy_network_train_steps"]), int(solver._policy_gradient_steps_total))


def run_worker(
    *, task_index: int, output_root: Path, smoke: bool = False, config_override: Optional[Mapping] = None
) -> Path:
    config = smoke_config() if smoke else copy.deepcopy(TRAINING_CONFIG)
    if config_override:
        config.update(dict(config_override))
    validate_contract(config, smoke=smoke)
    seeds = task_schedule(config["seeds"])
    if task_index < 0 or task_index >= len(seeds):
        raise IndexError(f"Task index {task_index} is outside [0, {len(seeds)})")
    seed = int(seeds[task_index])
    worker_dir = Path(output_root) / "workers" / task_name(task_index, seed)
    worker_dir.mkdir(parents=True, exist_ok=True)
    success_path = worker_dir / "SUCCESS.json"
    if success_path.exists():
        return worker_dir
    if pyspiel is None:
        raise RuntimeError(f"OpenSpiel is required for Experiment {EXPERIMENT_ID}")

    write_json(
        worker_dir / "worker_manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "task_index": int(task_index),
            "seed": seed,
            "smoke": bool(smoke),
            "config": _json_config(config),
            "distillation_arms": list(ARM_ORDER),
        },
    )

    game = pyspiel.load_game(str(config["game_name"]))
    reservoir_path = worker_dir / "frozen_strategy_reservoir.npz"
    training_summary_path = worker_dir / "training_summary.json"
    if reservoir_path.exists() and training_summary_path.exists():
        # Resume after the expensive boundary: a failed distillation or upload
        # must not cause the completed 12-hour DREAM trajectory to be repeated.
        reservoir = load_frozen_reservoir(reservoir_path)
        training_summary = json.loads(training_summary_path.read_text())
        reference_steps = int(training_summary["fresh_reference_fit_steps"])
    else:
        solver = make_dream_solver(config, seed)
        training_started = time.perf_counter()
        curves = solver.solve(
            policy_training_mode=str(config.get("policy_training_mode", "intermittent")),
            isolate_policy_training_rng=bool(config.get("isolate_policy_training_rng", True)),
            start_time=training_started,
            target_iteration=int(config["num_iterations"]),
            max_training_seconds=float(config["training_time_budget_seconds"]),
        )
        training_seconds = float(time.perf_counter() - training_started)
        curves.insert(0, "seed", seed)
        curves.to_csv(worker_dir / "training_curves.csv", index=False)

        policy_was_fitted = int(solver._policy_gradient_steps_total) > 0
        deployed_metrics = (
            exact_policy_metrics(game, NetworkPolicy(solver._policy_network))
            if policy_was_fitted
            else {
                "exploitability": float("nan"),
                "policy_value_player_0": float("nan"),
            }
        )
        reference_steps = _fit_steps(solver, config, smoke=smoke)
        reservoir = freeze_strategy_reservoir(solver, game)
        save_frozen_reservoir(reservoir_path, reservoir)
        training_summary = {
            "seed": seed,
            "completed_iterations": int(solver._iteration),
            "nodes_touched": int(solver._nodes_touched),
            "training_seconds": training_seconds,
            "requested_training_seconds": float(config["training_time_budget_seconds"]),
            "strategy_rows": reservoir.size,
            "strategy_observations_seen": reservoir.observations_seen,
            "strategy_capacity": reservoir.capacity,
            "policy_training_events": int(solver._policy_training_events),
            "policy_gradient_steps_total": int(solver._policy_gradient_steps_total),
            "deployed_policy_was_fitted": policy_was_fitted,
            "fresh_reference_fit_steps": reference_steps,
            "deployed_warm_started_exploitability": deployed_metrics["exploitability"],
            "deployed_warm_started_policy_value_player_0": deployed_metrics["policy_value_player_0"],
        }
        write_json(training_summary_path, training_summary)

        # The remaining audit needs only the frozen policy data; release the much
        # larger advantage and learned-baseline memories before fitting four arms.
        if hasattr(solver, "close"):
            solver.close()
        del solver
        cleanup_training_memory()

    grouped = group_reservoir(reservoir)
    save_grouped_reservoir(worker_dir / "grouped_empirical_policy.npz", grouped)
    empirical_metrics = exact_policy_metrics(game, EmpiricalReservoirPolicy(grouped))
    write_json(
        worker_dir / "empirical_policy_metrics.json",
        {
            **empirical_metrics,
            "source_rows": reservoir.size,
            "num_information_sets": grouped.size,
            "row_to_information_set_ratio": float(reservoir.size / grouped.size),
        },
    )

    init_seed = DISTILLATION_SEED_OFFSET + seed
    base_state = initial_policy_state(
        network_type=str(config.get("policy_network_type", "mlp")),
        input_size=int(reservoir.info_states.shape[1]),
        hidden_layers=config["policy_network_layers"],
        output_size=int(reservoir.strategies.shape[1]),
        seed=init_seed,
    )

    results: List[Dict] = []
    sampling_seed = init_seed + 1
    for arm in ARM_ORDER:
        steps = reference_steps * (EXTENDED_FIT_MULTIPLIER if arm == GROUPED_CE_EXTENDED else 1)
        model, fit_metrics = fit_frozen_policy(
            arm=arm,
            reservoir=reservoir,
            grouped=grouped,
            network_type=str(config.get("policy_network_type", "mlp")),
            hidden_layers=config["policy_network_layers"],
            initial_state=base_state,
            learning_rate=float(config["learning_rate"]),
            batch_size=int(config["batch_size_strategy"]),
            train_steps=int(steps),
            sampling_seed=sampling_seed,
            gradient_clip_norm=config.get("gradient_clip_norm"),
        )
        exact = exact_policy_metrics(game, NetworkPolicy(model))
        result = {
            "seed": seed,
            "arm": arm,
            "arm_label": ARM_LABELS[arm],
            **fit_metrics,
            **exact,
            "empirical_exploitability": empirical_metrics["exploitability"],
            "empirical_policy_value_player_0": empirical_metrics["policy_value_player_0"],
            "neural_minus_empirical_gap": (
                exact["exploitability"] - empirical_metrics["exploitability"]
            ),
            "strategy_rows": reservoir.size,
            "num_information_sets": grouped.size,
            "initialization_seed": init_seed,
            "sampling_seed": sampling_seed,
        }
        results.append(result)
        save_policy_model(
            worker_dir / "policies" / f"{arm}.pt",
            model,
            {
                "seed": seed,
                "arm": arm,
                "arm_label": ARM_LABELS[arm],
                "network_type": str(config.get("policy_network_type", "mlp")),
                "policy_network_layers": list(config["policy_network_layers"]),
                "info_state_size": int(reservoir.info_states.shape[1]),
                "num_actions": int(reservoir.strategies.shape[1]),
                "metrics": result,
            },
        )
        del model
        gc.collect()

    result_df = pd.DataFrame(results)
    result_df.to_csv(worker_dir / "distillation_results.csv", index=False)
    write_json(worker_dir / "distillation_results.json", results)
    write_json(
        success_path,
        {
            "status": "complete",
            "seed": seed,
            "task_index": int(task_index),
            "num_arms": len(results),
            "training_seconds": float(training_summary["training_seconds"]),
            "strategy_rows": reservoir.size,
            "num_information_sets": grouped.size,
        },
    )
    return worker_dir


def _aggregate_table(results: pd.DataFrame) -> pd.DataFrame:
    empirical = results.sort_values("arm").groupby("seed", as_index=False).first()
    empirical_values = empirical["empirical_exploitability"].to_numpy(dtype=float)
    empirical_policy_values = empirical["empirical_policy_value_player_0"].to_numpy(dtype=float)
    rows = [
        {
            "arm": "empirical_reservoir_policy",
            "arm_label": "Grouped empirical reservoir policy",
            "n_seeds": int(len(empirical)),
            "exploitability_mean": float(np.mean(empirical_values)),
            "exploitability_se": standard_error(empirical_values),
            "neural_minus_empirical_gap_mean": 0.0,
            "neural_minus_empirical_gap_se": 0.0,
            "policy_value_player_0_mean": float(np.mean(empirical_policy_values)),
            "policy_value_player_0_se": standard_error(empirical_policy_values),
            "distillation_seconds_mean": 0.0,
            "distillation_seconds_se": 0.0,
            "train_steps_mean": 0.0,
            "train_steps_se": 0.0,
            "examples_processed_mean": 0.0,
            "examples_processed_se": 0.0,
        }
    ]
    for arm in ARM_ORDER:
        subset = results[results["arm"] == arm]
        if subset.empty:
            continue
        row = {"arm": arm, "arm_label": ARM_LABELS[arm], "n_seeds": int(len(subset))}
        for metric in (
            "exploitability",
            "neural_minus_empirical_gap",
            "policy_value_player_0",
            "distillation_seconds",
            "train_steps",
            "examples_processed",
        ):
            values = subset[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_se"] = standard_error(values)
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_table(results: pd.DataFrame) -> pd.DataFrame:
    pivot = results.pivot(index="seed", columns="arm", values="exploitability")
    rows = []
    for arm in ARM_ORDER:
        if arm == ROW_MSE or arm not in pivot or ROW_MSE not in pivot:
            continue
        for seed, value in pivot[arm].items():
            rows.append(
                {
                    "seed": int(seed),
                    "reference_arm": ROW_MSE,
                    "arm": arm,
                    "delta_exploitability_vs_row_mse": float(value - pivot.loc[seed, ROW_MSE]),
                }
            )
    return pd.DataFrame(rows)


def _plot_metric(results: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    plot_results = results.copy()
    labels = dict(ARM_LABELS)
    order = [arm for arm in ARM_ORDER if arm in set(plot_results["arm"])]
    if metric == "exploitability":
        empirical = plot_results.sort_values("arm").groupby("seed", as_index=False).first()
        empirical_rows = pd.DataFrame(
            {
                "seed": empirical["seed"],
                "arm": "empirical_reservoir_policy",
                "exploitability": empirical["empirical_exploitability"],
            }
        )
        plot_results = pd.concat([empirical_rows, plot_results], ignore_index=True)
        order = ["empirical_reservoir_policy"] + order
        labels["empirical_reservoir_policy"] = "Grouped empirical policy"
    means = [float(plot_results.loc[plot_results["arm"] == arm, metric].mean()) for arm in order]
    errors = [standard_error(plot_results.loc[plot_results["arm"] == arm, metric]) for arm in order]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(order))
    ax.bar(x, means, yerr=errors, capsize=5, color="#2878B5", alpha=0.85)
    for index, arm in enumerate(order):
        values = plot_results.loc[plot_results["arm"] == arm, metric].to_numpy(dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(index + jitter, values, color="black", s=24, zorder=3)
    ax.set_xticks(x, [labels[arm] for arm in order], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_aggregate(output_root: Path, *, expected_seeds=PRODUCTION_SEEDS) -> Path:
    output_root = Path(output_root)
    result_paths = sorted((output_root / "workers").glob("task_*/distillation_results.csv"))
    if not result_paths:
        raise FileNotFoundError(f"No worker results found under {output_root / 'workers'}")
    results = pd.concat([pd.read_csv(path) for path in result_paths], ignore_index=True)
    observed_seeds = tuple(sorted(map(int, results["seed"].unique())))
    expected = tuple(sorted(map(int, expected_seeds)))
    if observed_seeds != expected:
        raise RuntimeError(f"Expected completed seeds {expected}, got {observed_seeds}")
    counts = results.groupby("seed")["arm"].nunique()
    if not bool((counts == len(ARM_ORDER)).all()):
        raise RuntimeError(f"Incomplete arm results: {counts.to_dict()}")

    analysis = output_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    results.to_csv(analysis / "seed_arm_results.csv", index=False)
    aggregate = _aggregate_table(results)
    aggregate.to_csv(analysis / "aggregate_summary_by_arm.csv", index=False)
    paired = _paired_table(results)
    paired.to_csv(analysis / "paired_differences_vs_row_mse.csv", index=False)

    training_paths = sorted((output_root / "workers").glob("task_*/training_summary.json"))
    training = [json.loads(path.read_text()) for path in training_paths]
    pd.DataFrame(training).to_csv(analysis / "training_summaries.csv", index=False)
    write_json(analysis / "training_summaries.json", training)
    write_json(
        analysis / "analysis_summary.json",
        {
            "status": "complete",
            "experiment_id": EXPERIMENT_ID,
            "num_seeds": len(observed_seeds),
            "seeds": list(observed_seeds),
            "num_arms": len(ARM_ORDER),
            "primary_reference": ROW_MSE,
            "empirical_policy": "iteration- and reach-weighted grouped reservoir policy",
            "aggregate_rows": aggregate.to_dict(orient="records"),
        },
    )
    _plot_metric(
        results,
        "exploitability",
        "Exploitability (NashConv / 2)",
        analysis / "frozen_reservoir_policy_exploitability.png",
    )
    _plot_metric(
        results,
        "neural_minus_empirical_gap",
        "Neural minus empirical exploitability",
        analysis / "frozen_reservoir_distillation_gap.png",
    )
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "aggregate", "smoke"))
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "worker":
        run_worker(task_index=args.task_index, output_root=args.output_root, smoke=False)
    elif args.mode == "aggregate":
        run_aggregate(args.output_root)
    else:
        run_worker(task_index=0, output_root=args.output_root, smoke=True)
        run_aggregate(args.output_root, expected_seeds=smoke_config()["seeds"])


if __name__ == "__main__":
    main()
