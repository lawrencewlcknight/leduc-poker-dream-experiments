"""Run Experiment 43: DREAM final-candidate checkpoint head-to-head."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dream_poker_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dream_poker_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    import pyspiel  # noqa: E402
except Exception:  # pragma: no cover
    pyspiel = None

from dream_poker.checkpointing import (  # noqa: E402
    checkpoint_exploitability_metrics,
    checkpoint_strength_summaries,
    discover_policy_snapshots,
    exact_pairwise_head_to_head,
    load_policies_by_seed,
    plot_heatmap,
    save_dream_policy_snapshot,
)
from dream_poker.experiment_runner import (  # noqa: E402
    create_timestamped_output_dir,
    make_dream_solver,
    write_json,
)
from dream_poker.experiment_utils import (  # noqa: E402
    average_policy_value_target,
    cleanup_training_memory,
    ensure_average_policy_value_columns,
    ensure_dir,
    safe_mean,
    standard_error,
)
from dream_poker.plotting import (  # noqa: E402
    add_average_policy_value_target,
    add_nash_exploitability_target,
    plot_mean_curve,
)

from .config import DEFAULT_CONFIG, DEFAULT_SEEDS  # noqa: E402
from .statistics import build_inference_tables  # noqa: E402


def _str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_seeds(value: Optional[str]) -> List[int]:
    parsed = _parse_int_list(value)
    return list(DEFAULT_SEEDS) if parsed is None else parsed


def _parse_layers(value: Optional[str]) -> Optional[List[int]]:
    return _parse_int_list(value)


def validate_config(config: Mapping[str, object]) -> None:
    """Check that requested snapshots are ordered and freshly policy-fitted."""
    schedule = tuple(int(value) for value in config["checkpoint_schedule"])
    num_iterations = int(config["num_iterations"])
    policy_train_every = int(config["policy_network_train_every"])
    if not schedule or any(a >= b for a, b in zip(schedule, schedule[1:])):
        raise ValueError("checkpoint_schedule must be non-empty and strictly increasing")
    if schedule[-1] != num_iterations:
        raise ValueError("The final checkpoint must equal num_iterations.")
    stale = [
        checkpoint
        for checkpoint in schedule[:-1]
        if checkpoint % policy_train_every != 0
    ]
    if stale:
        raise ValueError(
            "Every intermediate checkpoint must coincide with average-policy "
            f"training; incompatible checkpoints: {stale}"
        )


def _write_training_metadata(output_dir: Path, config: Mapping[str, object], phase: str) -> None:
    write_json(
        output_dir / "experiment_metadata.json",
        {
            "experiment_config": dict(config),
            "seeds": list(config["seeds"]),
            "phase": phase,
            "head_to_head_evaluation": "exact, seat-averaged OpenSpiel expected value",
            "statistical_unit": "independent training seed",
        },
    )


def _summarise_snapshot_rows(rows: pd.DataFrame, config: Mapping[str, object]) -> Dict:
    rows = ensure_average_policy_value_columns(rows, config.get("average_policy_value_target"))
    final = rows.sort_values("checkpoint_iteration").iloc[-1]
    final_window = rows.tail(min(5, len(rows)))
    reached = rows[rows["exploitability"] <= float(config["exploitability_threshold"])]
    return {
        "seed": int(final["seed"]),
        "num_checkpoints": int(len(rows)),
        "final_checkpoint_iteration": int(final["checkpoint_iteration"]),
        "final_nodes_touched": int(final["nodes_touched"]),
        "final_wall_clock_seconds": float(final["wall_clock_seconds"]),
        "final_exploitability": float(final["exploitability"]),
        "best_exploitability": float(rows["exploitability"].min()),
        "final_window_mean_exploitability": float(final_window["exploitability"].mean()),
        "final_average_policy_value": float(final["average_policy_value"]),
        "final_policy_value_error": float(final["policy_value_error"]),
        "checkpoint_nodes_to_threshold": int(reached.iloc[0]["nodes_touched"])
        if len(reached)
        else np.nan,
        "checkpoint_iteration_to_threshold": int(reached.iloc[0]["checkpoint_iteration"])
        if len(reached)
        else np.nan,
        "final_policy_training_events": int(final.get("policy_training_events", 0)),
        "final_policy_gradient_steps_total": int(
            final.get("policy_gradient_steps_total", 0)
        ),
    }


def _aggregate_seed_summaries(summary_df: pd.DataFrame) -> Dict:
    metrics = [
        "final_exploitability",
        "best_exploitability",
        "final_window_mean_exploitability",
        "final_average_policy_value",
        "final_policy_value_error",
        "final_wall_clock_seconds",
        "final_nodes_touched",
        "final_policy_gradient_steps_total",
    ]
    aggregate = {"num_seeds": int(len(summary_df))}
    if summary_df.empty:
        return aggregate
    for metric in metrics:
        aggregate[f"{metric}_mean"] = safe_mean(summary_df[metric])
        aggregate[f"{metric}_se"] = standard_error(summary_df[metric])
    return aggregate


def run_training_stage(config: Dict, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pyspiel is None:
        raise RuntimeError("OpenSpiel is not installed. Install open_spiel before running this experiment.")
    validate_config(config)
    _write_training_metadata(output_dir, config, phase="train")

    schedule = set(map(int, config["checkpoint_schedule"]))
    snapshots_dir = ensure_dir(output_dir / "snapshots")
    stage_metrics_path = output_dir / "training_stage_metrics.csv"
    full_curves_path = output_dir / "policy_training_curves.csv"
    failed_path = output_dir / "failed_seeds.json"
    stage_rows: List[Dict] = []
    full_curves: List[pd.DataFrame] = []
    seed_summaries: List[Dict] = []
    failed: List[Dict] = []

    for seed_value in config["seeds"]:
        seed = int(seed_value)
        print(f"\n=== DREAM final-candidate checkpoint run for seed {seed} ===", flush=True)
        solver = None
        seed_rows_start = len(stage_rows)
        start_time = time.perf_counter()

        def save_checkpoint(active_solver, completed_iteration: int) -> None:
            if int(completed_iteration) not in schedule:
                return
            row = active_solver._checkpoint_metrics(start_time)
            row["seed"] = seed
            row["checkpoint_iteration"] = int(completed_iteration)
            row["checkpoint_fraction"] = float(
                completed_iteration / int(config["num_iterations"])
            )
            snapshot_path = save_dream_policy_snapshot(
                active_solver,
                seed,
                int(completed_iteration),
                snapshots_dir,
                config,
            )
            row["policy_snapshot_path"] = str(snapshot_path)
            stage_rows.append(row)
            pd.DataFrame(stage_rows).to_csv(stage_metrics_path, index=False)
            print(
                f"Seed {seed} checkpoint {completed_iteration}: "
                f"nodes={row['nodes_touched']}, exploitability={row['exploitability']:.6f}",
                flush=True,
            )

        try:
            solver = make_dream_solver(config, seed)
            curves = solver.solve(
                isolate_policy_training_rng=config.get("isolate_policy_training_rng", True),
                post_iteration_callback=save_checkpoint,
            )
            curves.insert(0, "seed", seed)
            full_curves.append(curves)
            captured = {
                int(row["checkpoint_iteration"])
                for row in stage_rows[seed_rows_start:]
            }
            missing = sorted(schedule - captured)
            if missing:
                raise RuntimeError(f"Training completed without snapshots at {missing}")
            seed_summaries.append(
                _summarise_snapshot_rows(
                    pd.DataFrame(stage_rows[seed_rows_start:]),
                    config,
                )
            )
        except Exception as exc:  # pragma: no cover - intended for cloud failures
            failed.append(
                {
                    "seed": seed,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"Seed {seed} failed: {exc}", flush=True)
        finally:
            if solver is not None and hasattr(solver, "close"):
                solver.close()
            del solver
            cleanup_training_memory()

    if full_curves:
        pd.concat(full_curves, ignore_index=True).to_csv(full_curves_path, index=False)
    stage_df = pd.DataFrame(stage_rows)
    summary_df = pd.DataFrame(seed_summaries)
    stage_df.to_csv(stage_metrics_path, index=False)
    summary_df.to_csv(output_dir / "seed_summary.csv", index=False)
    write_json(output_dir / "aggregate_training_summary.json", _aggregate_seed_summaries(summary_df))
    write_json(failed_path, failed)
    if failed and not seed_summaries:
        raise RuntimeError("No seed completed the full checkpoint schedule.")
    make_training_plots(stage_df, output_dir, config)
    _write_training_metadata(output_dir, config, phase="train_complete")
    return stage_df, summary_df


def make_training_plots(stage_df: pd.DataFrame, output_dir: Path, config: Mapping[str, object]) -> None:
    if stage_df.empty:
        return
    plot_dir = ensure_dir(output_dir / "plots")
    value_target = average_policy_value_target(config)
    plot_mean_curve(
        stage_df,
        "nodes_touched",
        "exploitability",
        "DREAM final-candidate checkpoint exploitability by nodes touched",
        "Exploitability",
        plot_dir / "checkpoint_exploitability_by_nodes.png",
        title_config=config,
    )
    plot_mean_curve(
        stage_df,
        "nodes_touched",
        "average_policy_value",
        "DREAM final-candidate checkpoint average policy value by nodes touched",
        "Average policy value",
        plot_dir / "checkpoint_average_policy_value_by_nodes.png",
        average_policy_value_target=value_target,
        title_config=config,
    )
    plot_mean_curve(
        stage_df,
        "nodes_touched",
        "policy_value_error",
        "DREAM final-candidate checkpoint policy-value error by nodes touched",
        "Absolute error from Leduc game value",
        plot_dir / "checkpoint_policy_value_error_by_nodes.png",
        title_config=config,
    )


def _mean_nodes_by_checkpoint(stage_df: pd.DataFrame) -> pd.DataFrame:
    if stage_df.empty:
        return pd.DataFrame(columns=["checkpoint", "nodes_touched_mean", "nodes_touched_sem"])
    grouped = (
        stage_df.groupby("checkpoint_iteration")["nodes_touched"]
        .agg(["mean", "sem"])
        .reset_index()
        .rename(
            columns={
                "checkpoint_iteration": "checkpoint",
                "mean": "nodes_touched_mean",
                "sem": "nodes_touched_sem",
            }
        )
    )
    grouped["nodes_touched_sem"] = grouped["nodes_touched_sem"].fillna(0.0)
    return grouped


def _node_label(checkpoint: int, node_lookup: Mapping[int, float]) -> str:
    nodes = node_lookup.get(int(checkpoint))
    if nodes is None or not np.isfinite(nodes):
        return str(checkpoint)
    return f"{nodes / 1_000_000:.1f}M\n({checkpoint})"


def _plot_errorbar_by_nodes(
    df: pd.DataFrame,
    y_mean_col: str,
    y_sem_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    *,
    zero_line: bool = True,
    average_policy_value_target_value: Optional[float] = None,
) -> None:
    if df.empty or y_mean_col not in df.columns:
        return
    x = df["nodes_touched_mean"].to_numpy(dtype=float)
    y = df[y_mean_col].to_numpy(dtype=float)
    yerr = df[y_sem_col].fillna(0.0).to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return
    x = x[finite]
    y = y[finite]
    yerr = yerr[finite]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
    if "exploitability" in y_mean_col:
        add_nash_exploitability_target(ax)
        ax.legend()
    elif average_policy_value_target_value is not None:
        add_average_policy_value_target(ax, target=average_policy_value_target_value)
        ax.legend()
    elif zero_line:
        ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_seed_effects(seed_rows: Sequence[Mapping[str, object]], output_path: Path) -> None:
    values = np.asarray(
        [float(row["mean_later_vs_earlier_ev"]) for row in seed_rows],
        dtype=np.float64,
    )
    labels = [str(row["seed"]) for row in seed_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(values.size)
    ax.scatter(x, values, s=42, label="Per-seed mean")
    if values.size:
        ax.axhline(float(np.mean(values)), linewidth=2, label="Mean across seeds")
    ax.axhline(0.0, linestyle="--", linewidth=1, label="No head-to-head difference")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Mean exact EV of later vs earlier checkpoints")
    ax.set_title("DREAM Final-Candidate Checkpoint Improvement Across Seeds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_analysis_stage(config: Dict, run_dir: Path) -> Dict[str, Path]:
    if pyspiel is None:
        raise RuntimeError("OpenSpiel is not installed. Install open_spiel before running this experiment.")
    run_dir = Path(run_dir)
    snapshots_dir = run_dir / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"No snapshots directory found at {snapshots_dir}.")

    analysis_dir = ensure_dir(run_dir / "head_to_head_analysis")
    plot_dir = ensure_dir(analysis_dir / "plots")
    checkpoint_df = discover_policy_snapshots(snapshots_dir, config.get("checkpoint_schedule"))
    if checkpoint_df.empty:
        raise FileNotFoundError(f"No DREAM policy snapshots found in {snapshots_dir}.")

    game = pyspiel.load_game(str(config["game_name"]))
    policies_by_seed, loaded_policy_df = load_policies_by_seed(game, checkpoint_df)
    checkpoint_metrics_df = checkpoint_exploitability_metrics(game, policies_by_seed)
    exact_df, exact_matrices, mean_matrix, win_fraction_matrix = exact_pairwise_head_to_head(
        game,
        policies_by_seed,
        equivalence_epsilon=float(config["head_to_head_equivalence_epsilon"]),
    )
    monotonicity_df, strength_with_metrics_df, aggregate_strength_df, best_checkpoint_df = (
        checkpoint_strength_summaries(
            exact_matrices,
            checkpoint_metrics_df,
            equivalence_epsilon=float(config["head_to_head_equivalence_epsilon"]),
        )
    )

    pairwise_df = exact_df.rename(
        columns={"checkpoint_A": "checkpoint_a", "checkpoint_B": "checkpoint_b"}
    )
    stage_metrics_path = run_dir / "training_stage_metrics.csv"
    stage_df = pd.read_csv(stage_metrics_path) if stage_metrics_path.exists() else pd.DataFrame()
    node_summary_df = _mean_nodes_by_checkpoint(stage_df)
    aggregate_strength_df = aggregate_strength_df.merge(
        node_summary_df,
        on="checkpoint",
        how="left",
    )

    seed_rows, summary_rows, pair_rows = build_inference_tables(
        pairwise_df.to_dict(orient="records"),
        config["checkpoint_schedule"],
    )
    node_lookup = {
        int(row.checkpoint): float(row.nodes_touched_mean)
        for row in node_summary_df.itertuples(index=False)
    }
    for row in pair_rows:
        row["later_nodes_touched_mean"] = node_lookup.get(
            int(row["later_checkpoint"]),
            float("nan"),
        )
        row["earlier_nodes_touched_mean"] = node_lookup.get(
            int(row["earlier_checkpoint"]),
            float("nan"),
        )

    checkpoint_df.to_csv(analysis_dir / "checkpoint_inventory.csv", index=False)
    loaded_policy_df.to_csv(analysis_dir / "loaded_policy_inventory.csv", index=False)
    checkpoint_metrics_df.to_csv(analysis_dir / "checkpoint_exploitability_metrics.csv", index=False)
    pairwise_df.to_csv(analysis_dir / "head_to_head_pairwise.csv", index=False)
    mean_matrix.to_csv(analysis_dir / "head_to_head_mean_matrix.csv")
    win_fraction_matrix.to_csv(analysis_dir / "head_to_head_seed_win_fraction_matrix.csv")
    monotonicity_df.to_csv(analysis_dir / "head_to_head_monotonicity_by_seed.csv", index=False)
    strength_with_metrics_df.to_csv(analysis_dir / "head_to_head_strength_by_checkpoint.csv", index=False)
    aggregate_strength_df.to_csv(analysis_dir / "head_to_head_strength_aggregate.csv", index=False)
    best_checkpoint_df.to_csv(analysis_dir / "best_checkpoint_summary.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(analysis_dir / "head_to_head_primary_effect_by_seed.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(analysis_dir / "head_to_head_inference_summary.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(analysis_dir / "head_to_head_pairwise_inference.csv", index=False)

    plot_heatmap(
        mean_matrix,
        "DREAM final-candidate mean exact head-to-head EV",
        plot_dir / "head_to_head_mean_matrix.png",
        title_config=config,
    )
    later_vs_earlier = mean_matrix.copy()
    for i, row_checkpoint in enumerate(later_vs_earlier.index):
        for j, col_checkpoint in enumerate(later_vs_earlier.columns):
            if int(row_checkpoint) <= int(col_checkpoint):
                later_vs_earlier.iat[i, j] = np.nan
    plot_heatmap(
        later_vs_earlier,
        "DREAM final-candidate later-vs-earlier checkpoint EV",
        plot_dir / "head_to_head_later_vs_earlier.png",
        title_config=config,
    )
    labels = {
        checkpoint: _node_label(int(checkpoint), node_lookup)
        for checkpoint in later_vs_earlier.index
    }
    plot_heatmap(
        later_vs_earlier.rename(index=labels, columns=labels),
        "DREAM final-candidate later-vs-earlier checkpoint EV by nodes",
        plot_dir / "head_to_head_later_vs_earlier_by_nodes.png",
        title_config=config,
    )
    plot_heatmap(
        win_fraction_matrix,
        "Fraction of seeds where row checkpoint beats column checkpoint",
        plot_dir / "head_to_head_seed_win_fraction_matrix.png",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        fmt=".2f",
        colorbar_label="Seed win fraction",
        title_config=config,
    )
    _plot_errorbar_by_nodes(
        aggregate_strength_df,
        "mean_EV_vs_earlier_mean",
        "mean_EV_vs_earlier_sem",
        "Mean EV vs earlier checkpoints",
        "DREAM: later checkpoints against all earlier checkpoints",
        plot_dir / "head_to_head_strength_vs_earlier_by_nodes.png",
    )
    _plot_errorbar_by_nodes(
        aggregate_strength_df,
        "EV_vs_previous_mean",
        "EV_vs_previous_sem",
        "EV vs immediately previous checkpoint",
        "DREAM adjacent-checkpoint improvement",
        plot_dir / "head_to_head_strength_vs_previous_by_nodes.png",
    )
    _plot_errorbar_by_nodes(
        aggregate_strength_df,
        "exploitability_mean",
        "exploitability_sem",
        "Exploitability",
        "DREAM checkpoint exploitability",
        plot_dir / "exploitability_by_nodes.png",
        zero_line=False,
    )
    _plot_errorbar_by_nodes(
        aggregate_strength_df,
        "average_policy_value_mean",
        "average_policy_value_sem",
        "Average policy value",
        "DREAM checkpoint average policy value",
        plot_dir / "average_policy_value_by_nodes.png",
        zero_line=False,
        average_policy_value_target_value=average_policy_value_target(config),
    )
    _plot_seed_effects(seed_rows, plot_dir / "head_to_head_primary_effect_by_seed.png")

    aggregate_path = analysis_dir / "aggregate_summary.json"
    summaries_by_estimand = {row["estimand"]: row for row in summary_rows}
    write_json(
        aggregate_path,
        {
            "analysis_unit": "independent_training_seed",
            "evaluation": "exact OpenSpiel expected value, averaged over seats",
            "primary_estimand": summaries_by_estimand.get(
                "seed_mean_ev_later_vs_all_earlier_checkpoints"
            ),
            "adjacent_checkpoint_estimand": summaries_by_estimand.get(
                "seed_mean_ev_vs_immediately_previous_checkpoint"
            ),
            "final_vs_first_estimand": summaries_by_estimand.get(
                "final_checkpoint_ev_vs_first_checkpoint"
            ),
            "multiple_testing": (
                "Secondary checkpoint-pair sign-flip p-values use Holm "
                "family-wise error correction."
            ),
            "checkpoint_schedule": list(map(int, config["checkpoint_schedule"])),
            "mean_nodes_by_checkpoint": node_lookup,
        },
    )
    write_json(
        analysis_dir / "head_to_head_analysis_metadata.json",
        {
            "run_dir": run_dir,
            "snapshots_dir": snapshots_dir,
            "analysis_dir": analysis_dir,
            "num_snapshots": int(len(checkpoint_df)),
            "num_seeds": int(checkpoint_df["seed"].nunique()),
            "head_to_head_equivalence_epsilon": float(
                config["head_to_head_equivalence_epsilon"]
            ),
        },
    )
    del policies_by_seed, exact_matrices, game
    cleanup_training_memory()
    return {
        "analysis_dir": analysis_dir,
        "aggregate_summary": aggregate_path,
        "head_to_head_pairwise": analysis_dir / "head_to_head_pairwise.csv",
        "head_to_head_inference_summary": analysis_dir / "head_to_head_inference_summary.csv",
    }


def _load_config_from_run_dir(run_dir: Path, base_config: Dict) -> Dict:
    metadata_path = run_dir / "experiment_metadata.json"
    if not metadata_path.exists():
        return copy.deepcopy(base_config)
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    stored = metadata.get("experiment_config", metadata)
    config = copy.deepcopy(base_config)
    config.update(stored)
    return config


def run_experiment(config: Dict, phase: str, run_dir: Optional[Path] = None) -> Path:
    phase = "analyse" if phase == "analyze" else phase
    if phase not in {"all", "train", "analyse"}:
        raise ValueError("phase must be one of: all, train, analyse")
    if phase == "analyse":
        if run_dir is None:
            raise ValueError("--run-dir is required for analyse phase.")
        run_dir = Path(run_dir).resolve()
        config = _load_config_from_run_dir(run_dir, config)
        validate_config(config)
        run_analysis_stage(config, run_dir)
        print(f"Analysis outputs written under {run_dir / 'head_to_head_analysis'}", flush=True)
        return run_dir

    output_dir = Path(run_dir).resolve() if run_dir is not None else create_timestamped_output_dir(config["output_root"])
    ensure_dir(output_dir)
    validate_config(config)
    run_training_stage(config, output_dir)
    if phase == "all":
        run_analysis_stage(config, output_dir)
    print(f"Outputs written to {output_dir}", flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", nargs="?", choices=["all", "train", "analyse", "analyze"], default=None)
    parser.add_argument("--stage", choices=["all", "train", "analyse", "analyze"], default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--checkpoint-schedule", type=str, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--policy-network-train-every", type=int, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--advantage-network-train-steps", type=int, default=None)
    parser.add_argument("--baseline-network-train-steps", type=int, default=None)
    parser.add_argument("--baseline-network-train-every", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--policy-network-layers", type=str, default=None)
    parser.add_argument("--advantage-network-layers", type=str, default=None)
    parser.add_argument("--baseline-network-layers", type=str, default=None)
    parser.add_argument("--batch-size-advantage", type=int, default=None)
    parser.add_argument("--batch-size-strategy", type=int, default=None)
    parser.add_argument("--batch-size-baseline", type=int, default=None)
    parser.add_argument("--memory-capacity", type=int, default=None)
    parser.add_argument("--advantage-memory-capacity", type=int, default=None)
    parser.add_argument("--strategy-memory-capacity", type=int, default=None)
    parser.add_argument("--baseline-memory-capacity", type=int, default=None)
    parser.add_argument("--compute-exploitability", type=_str2bool, default=None)
    parser.add_argument("--equivalence-epsilon", type=float, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--allow-policy-training-rng-advance", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> Dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.seeds:
        config["seeds"] = _parse_seeds(args.seeds)
    if args.iterations is not None:
        config["num_iterations"] = int(args.iterations)
    if args.traversals is not None:
        config["num_traversals"] = int(args.traversals)
    if args.checkpoint_schedule is not None:
        config["checkpoint_schedule"] = tuple(_parse_int_list(args.checkpoint_schedule) or [])
    interval = args.policy_network_train_every if args.policy_network_train_every is not None else args.evaluation_interval
    if interval is not None:
        config["policy_network_train_every"] = int(interval)
        config["evaluation_interval"] = int(interval)
    if args.policy_network_train_steps is not None:
        config["policy_network_train_steps"] = int(args.policy_network_train_steps)
    if args.advantage_network_train_steps is not None:
        config["advantage_network_train_steps"] = int(args.advantage_network_train_steps)
    if args.baseline_network_train_steps is not None:
        config["baseline_network_train_steps"] = int(args.baseline_network_train_steps)
    if args.baseline_network_train_every is not None:
        config["baseline_network_train_every"] = int(args.baseline_network_train_every)
    if args.learning_rate is not None:
        config["learning_rate"] = float(args.learning_rate)
    if args.epsilon is not None:
        config["epsilon"] = float(args.epsilon)
    if args.policy_network_layers is not None:
        config["policy_network_layers"] = _parse_layers(args.policy_network_layers)
    if args.advantage_network_layers is not None:
        config["advantage_network_layers"] = _parse_layers(args.advantage_network_layers)
    if args.baseline_network_layers is not None:
        config["baseline_network_layers"] = _parse_layers(args.baseline_network_layers)
    if args.memory_capacity is not None:
        config["advantage_memory_capacity"] = int(args.memory_capacity)
        config["strategy_memory_capacity"] = int(args.memory_capacity)
        config["baseline_memory_capacity"] = int(args.memory_capacity)
    if args.advantage_memory_capacity is not None:
        config["advantage_memory_capacity"] = int(args.advantage_memory_capacity)
    if args.strategy_memory_capacity is not None:
        config["strategy_memory_capacity"] = int(args.strategy_memory_capacity)
    if args.baseline_memory_capacity is not None:
        config["baseline_memory_capacity"] = int(args.baseline_memory_capacity)
    if args.batch_size_advantage is not None:
        config["batch_size_advantage"] = int(args.batch_size_advantage)
    if args.batch_size_strategy is not None:
        config["batch_size_strategy"] = int(args.batch_size_strategy)
    if args.batch_size_baseline is not None:
        config["batch_size_baseline"] = int(args.batch_size_baseline)
    if args.compute_exploitability is not None:
        config["compute_exploitability"] = bool(args.compute_exploitability)
    if args.equivalence_epsilon is not None:
        config["head_to_head_equivalence_epsilon"] = float(args.equivalence_epsilon)
    if args.output_root is not None:
        config["output_root"] = args.output_root
    if args.allow_policy_training_rng_advance:
        config["isolate_policy_training_rng"] = False
    return config


def main() -> int:
    args = build_parser().parse_args()
    phase = args.stage or args.phase or "all"
    run_experiment(config_from_args(args), phase=phase, run_dir=args.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
