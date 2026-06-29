"""Reusable staged random-search helpers for DREAM experiments."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dream_poker.constants import EXPLOITABILITY_THRESHOLD, LEDUC_AVERAGE_POLICY_VALUE_TARGET

try:
    import pyspiel
except Exception:  # pragma: no cover
    pyspiel = None

from dream_poker.experiment_runner import (
    create_timestamped_output_dir,
    json_ready,
    make_dream_solver,
    write_json,
)
from dream_poker.experiment_utils import cleanup_training_memory, compute_auc, ensure_dir, safe_mean, standard_error
from dream_poker.experiment_utils import ensure_average_policy_value_columns
from dream_poker.plotting import (
    _mean_curve,
    add_average_policy_value_target,
    add_nash_exploitability_target,
    format_plot_title,
    is_average_policy_value_metric,
    is_exploitability_metric,
)
from dream_poker.seeding import set_seed


def config_signature(config: Dict) -> Tuple:
    return tuple((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in sorted(config.items()))


def sample_candidate_configs(
    baseline_config: Dict,
    search_space: Dict[str, Sequence],
    fixed_parameters: Dict,
    n_configs: int,
    master_seed: int,
) -> List[Dict]:
    """Sample unique DREAM candidate configurations from a constrained search space."""
    rng = random.Random(int(master_seed))
    candidates: List[Dict] = []
    seen = {config_signature(baseline_config)}

    while len(candidates) < int(n_configs):
        config = dict(baseline_config)
        for key, values in search_space.items():
            config[key] = rng.choice(values)
        config.update(fixed_parameters)
        _clamp_batch_sizes_to_capacity(config)

        sig = config_signature(config)
        if sig in seen:
            continue
        seen.add(sig)
        config["config_label"] = f"random_candidate_{len(candidates) + 1:02d}"
        candidates.append(config)

    return candidates


def _clamp_batch_sizes_to_capacity(config: Dict) -> None:
    config["batch_size_advantage"] = min(
        int(config["batch_size_advantage"]),
        int(config["advantage_memory_capacity"]),
    )
    config["batch_size_strategy"] = min(
        int(config["batch_size_strategy"]),
        int(config["strategy_memory_capacity"]),
    )
    config["batch_size_baseline"] = min(
        int(config["batch_size_baseline"]),
        int(config["baseline_memory_capacity"]),
    )


def summarise_tuning_curve(
    curves: pd.DataFrame,
    config: Dict,
    seed: int,
    stage: str,
    train_seconds: float,
) -> Dict:
    if curves.empty:
        raise ValueError("No checkpoint metrics were returned by the solver.")
    curves = ensure_average_policy_value_columns(curves, config.get("average_policy_value_target"))
    final_row = curves.iloc[-1]
    final_window = curves.tail(min(5, len(curves)))
    reached = curves[curves["exploitability"] <= EXPLOITABILITY_THRESHOLD]
    return {
        "stage": stage,
        "config_label": config["config_label"],
        "seed": int(seed),
        "status": "ok",
        "num_iterations": int(config["num_iterations"]),
        "num_traversals": int(config["num_traversals"]),
        "learning_rate": float(config["learning_rate"]),
        "epsilon": float(config["epsilon"]),
        "policy_network_layers": str(tuple(config["policy_network_layers"])),
        "advantage_network_layers": str(tuple(config["advantage_network_layers"])),
        "baseline_network_layers": str(tuple(config["baseline_network_layers"])),
        "batch_size_advantage": int(config["batch_size_advantage"]),
        "batch_size_strategy": int(config["batch_size_strategy"]),
        "batch_size_baseline": int(config["batch_size_baseline"]),
        "advantage_memory_capacity": int(config["advantage_memory_capacity"]),
        "strategy_memory_capacity": int(config["strategy_memory_capacity"]),
        "baseline_memory_capacity": int(config["baseline_memory_capacity"]),
        "advantage_network_train_steps": int(config["advantage_network_train_steps"]),
        "baseline_network_train_steps": int(config["baseline_network_train_steps"]),
        "policy_network_train_steps": int(config["policy_network_train_steps"]),
        "policy_network_train_every": int(config["policy_network_train_every"]),
        "final_iteration": int(final_row["iteration"]),
        "final_nodes_touched": int(final_row["nodes_touched"]),
        "final_wall_clock_seconds": float(final_row["wall_clock_seconds"]),
        "train_seconds": float(train_seconds),
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
        "final_policy_training_events": int(final_row.get("policy_training_events", 0)),
        "final_policy_gradient_steps_total": int(final_row.get("policy_gradient_steps_total", 0)),
    }


def run_config_seed(config: Dict, seed: int, stage: str, run_dir: Path) -> Tuple[Dict, List[Dict]]:
    if pyspiel is None:
        raise RuntimeError("OpenSpiel is not installed. Install open_spiel before running this experiment.")
    set_seed(seed)
    solver = make_dream_solver(config, seed)

    t0 = time.perf_counter()
    curves = solver.solve(isolate_policy_training_rng=config.get("isolate_policy_training_rng", True))
    curves = ensure_average_policy_value_columns(curves, config.get("average_policy_value_target"))
    train_seconds = time.perf_counter() - t0

    curves.insert(0, "stage", stage)
    curves.insert(1, "config_label", config["config_label"])
    curves.insert(2, "seed", int(seed))

    trace_dir = ensure_dir(run_dir / "traces" / stage / config["config_label"])
    curves.to_csv(trace_dir / f"seed_{seed}_curves.csv", index=False)

    summary = summarise_tuning_curve(curves, config, seed, stage, train_seconds)
    write_json(trace_dir / f"seed_{seed}_summary.json", summary)
    curve_rows = curves.to_dict(orient="records")
    del solver
    cleanup_training_memory()
    return summary, curve_rows


def aggregate_config_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    ok = summary_df[summary_df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    numeric_cols = [
        "final_exploitability",
        "best_exploitability",
        "final_window_mean_exploitability",
        "final_average_policy_value",
        "final_window_mean_average_policy_value",
        "final_window_std_exploitability",
        "exploitability_auc_by_iteration",
        "exploitability_auc_by_nodes",
        "final_policy_value_error",
        "best_policy_value_error",
        "final_nodes_touched",
        "train_seconds",
        "final_wall_clock_seconds",
        "nodes_to_threshold",
        "iterations_to_threshold",
        "seconds_to_threshold",
        "final_policy_gradient_steps_total",
    ]
    rows = []
    for label, group in ok.groupby("config_label"):
        row = {"config_label": label, "n_seeds": int(group["seed"].nunique())}
        first = group.iloc[0]
        for col in [
            "learning_rate",
            "epsilon",
            "num_traversals",
            "policy_network_layers",
            "advantage_network_layers",
            "baseline_network_layers",
            "batch_size_advantage",
            "batch_size_strategy",
            "batch_size_baseline",
            "advantage_memory_capacity",
            "strategy_memory_capacity",
            "baseline_memory_capacity",
            "advantage_network_train_steps",
            "baseline_network_train_steps",
            "policy_network_train_steps",
        ]:
            row[col] = first[col] if col in first else np.nan
        for col in numeric_cols:
            if col in group:
                row[f"{col}_mean"] = safe_mean(group[col])
                row[f"{col}_se"] = standard_error(group[col])
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["final_window_mean_exploitability_mean", "final_exploitability_mean"],
        ascending=True,
    )


def run_tuning_stage(
    configs: List[Dict],
    seeds: Sequence[int],
    stage: str,
    num_iterations: int,
    fixed_parameters: Dict,
    run_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: List[Dict] = []
    curves: List[Dict] = []
    table_dir = ensure_dir(run_dir / "tables")

    for cfg in configs:
        stage_cfg = dict(cfg)
        stage_cfg["num_iterations"] = int(num_iterations)
        stage_cfg.update(fixed_parameters)

        for seed in seeds:
            print(f"\n[{stage}] config={stage_cfg['config_label']} seed={seed} iterations={num_iterations}")
            try:
                summary, curve_rows = run_config_seed(stage_cfg, int(seed), stage, run_dir)
            except Exception as exc:
                print(f"Run failed: {exc}")
                summary = {
                    "stage": stage,
                    "config_label": stage_cfg["config_label"],
                    "seed": int(seed),
                    "status": "failed",
                    "error": repr(exc),
                }
                curve_rows = []
            summaries.append(summary)
            curves.extend(curve_rows)

            pd.DataFrame(summaries).to_csv(table_dir / f"{stage}_run_summaries_partial.csv", index=False)
            if curves:
                pd.DataFrame(curves).to_csv(table_dir / f"{stage}_curves_partial.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    curves_df = pd.DataFrame(curves)
    agg_df = aggregate_config_summary(summary_df)

    summary_df.to_csv(table_dir / f"{stage}_run_summaries.csv", index=False)
    curves_df.to_csv(table_dir / f"{stage}_curves.csv", index=False)
    agg_df.to_csv(table_dir / f"{stage}_config_summary.csv", index=False)
    return summary_df, curves_df, agg_df


def select_confirmation_configs(
    screening_agg: pd.DataFrame,
    all_configs: List[Dict],
    baseline_label: str,
    n_confirmation_candidates: int,
) -> List[Dict]:
    cfg_by_label = {cfg["config_label"]: cfg for cfg in all_configs}
    selected = [dict(cfg_by_label[baseline_label])]
    if screening_agg.empty:
        return selected
    ranked = screening_agg[screening_agg["config_label"] != baseline_label].copy()
    ranked = ranked.sort_values(
        [
            "final_window_mean_exploitability_mean",
            "final_exploitability_mean",
            "exploitability_auc_by_iteration_mean",
        ]
    )
    selected_labels = ranked["config_label"].head(int(n_confirmation_candidates)).tolist()
    for label in selected_labels:
        selected.append(dict(cfg_by_label[label]))
    return selected


def paired_differences_vs_baseline(
    summary_df: pd.DataFrame,
    metric: str = "final_exploitability",
    baseline_label: str = "dream_baseline",
) -> pd.DataFrame:
    ok = summary_df[summary_df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    baseline = ok[ok["config_label"] == baseline_label][["seed", metric]].rename(
        columns={metric: "baseline_value"}
    )
    rows = []
    for label, group in ok.groupby("config_label"):
        if label == baseline_label:
            continue
        comp = group[["seed", metric]].rename(columns={metric: "candidate_value"})
        merged = comp.merge(baseline, on="seed", how="inner")
        if merged.empty:
            continue
        merged["delta_candidate_minus_baseline"] = merged["candidate_value"] - merged["baseline_value"]
        merged["config_label"] = label
        merged["metric"] = metric
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def prepare_tuning_output_dir(output_root: Path, metadata: Dict) -> Path:
    run_dir = create_timestamped_output_dir(output_root)
    for subdir in ["traces", "tables", "plots"]:
        ensure_dir(run_dir / subdir)
    write_json(run_dir / "experiment_metadata.json", json_ready(metadata))
    return run_dir


def make_tuning_plots(
    run_dir: Path,
    baseline_label: str,
    screening_agg: pd.DataFrame,
    confirmation_summary: pd.DataFrame,
    confirmation_curves: pd.DataFrame,
    confirmation_agg: pd.DataFrame,
    average_policy_value_target_value: float = LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    title_config: Dict | None = None,
) -> List[Path]:
    plot_dir = ensure_dir(run_dir / "plots")
    table_dir = ensure_dir(run_dir / "tables")
    paths: List[Path] = []

    if not screening_agg.empty:
        paths.append(
            _barh_metric(
                screening_agg.sort_values("final_window_mean_exploitability_mean"),
                "final_window_mean_exploitability_mean",
                "final_window_mean_exploitability_se",
                "DREAM random search screening: final-window exploitability",
                "Mean final-window exploitability",
                plot_dir / "screening_ranked_final_window_exploitability.png",
                title_config=title_config,
            )
        )
        if "final_window_mean_average_policy_value_mean" in screening_agg.columns:
            paths.append(
                _barh_metric(
                    screening_agg.sort_values("final_window_mean_average_policy_value_mean"),
                    "final_window_mean_average_policy_value_mean",
                    "final_window_mean_average_policy_value_se",
                    "DREAM random search screening: final-window average policy value",
                    "Mean final-window average policy value",
                    plot_dir / "screening_ranked_final_window_average_policy_value.png",
                    average_policy_value_target=average_policy_value_target_value,
                    title_config=title_config,
                )
            )

    if not confirmation_agg.empty:
        paths.append(
            _barh_metric(
                confirmation_agg.sort_values("final_exploitability_mean"),
                "final_exploitability_mean",
                "final_exploitability_se",
                "DREAM random search confirmation: final exploitability",
                "Mean final exploitability",
                plot_dir / "confirmation_final_exploitability.png",
                title_config=title_config,
            )
        )
        paths.append(
            _barh_metric(
                confirmation_agg.sort_values("final_window_mean_exploitability_mean"),
                "final_window_mean_exploitability_mean",
                "final_window_mean_exploitability_se",
                "DREAM random search confirmation: final-window exploitability",
                "Mean final-window exploitability",
                plot_dir / "confirmation_final_window_exploitability.png",
                title_config=title_config,
            )
        )
        if "final_average_policy_value_mean" in confirmation_agg.columns:
            paths.append(
                _barh_metric(
                    confirmation_agg.sort_values("final_average_policy_value_mean"),
                    "final_average_policy_value_mean",
                    "final_average_policy_value_se",
                    "DREAM random search confirmation: final average policy value",
                    "Mean final average policy value",
                    plot_dir / "confirmation_final_average_policy_value.png",
                    average_policy_value_target=average_policy_value_target_value,
                    title_config=title_config,
                )
            )
        if "final_window_mean_average_policy_value_mean" in confirmation_agg.columns:
            paths.append(
                _barh_metric(
                    confirmation_agg.sort_values("final_window_mean_average_policy_value_mean"),
                    "final_window_mean_average_policy_value_mean",
                    "final_window_mean_average_policy_value_se",
                    "DREAM random search confirmation: final-window average policy value",
                    "Mean final-window average policy value",
                    plot_dir / "confirmation_final_window_average_policy_value.png",
                    average_policy_value_target=average_policy_value_target_value,
                    title_config=title_config,
                )
            )

    if not confirmation_curves.empty:
        paths.append(
            _plot_grouped_curve(
                confirmation_curves,
                "iteration",
                "exploitability",
                "DREAM random search confirmation: exploitability by iteration",
                "Exploitability",
                plot_dir / "confirmation_exploitability_by_iteration.png",
                title_config=title_config,
            )
        )
        paths.append(
            _plot_grouped_curve(
                confirmation_curves,
                "nodes_touched",
                "exploitability",
                "DREAM random search confirmation: exploitability by nodes touched",
                "Exploitability",
                plot_dir / "confirmation_exploitability_by_nodes.png",
                title_config=title_config,
            )
        )
        if "average_policy_value" in confirmation_curves.columns:
            paths.append(
                _plot_grouped_curve(
                    confirmation_curves,
                    "iteration",
                    "average_policy_value",
                    "DREAM random search confirmation: average policy value by iteration",
                    "Average policy value",
                    plot_dir / "confirmation_average_policy_value_by_iteration.png",
                    average_policy_value_target=average_policy_value_target_value,
                    title_config=title_config,
                )
            )
            paths.append(
                _plot_grouped_curve(
                    confirmation_curves,
                    "nodes_touched",
                    "average_policy_value",
                    "DREAM random search confirmation: average policy value by nodes touched",
                    "Average policy value",
                    plot_dir / "confirmation_average_policy_value_by_nodes.png",
                    average_policy_value_target=average_policy_value_target_value,
                    title_config=title_config,
                )
            )
        paths.append(
            _plot_grouped_curve(
                confirmation_curves,
                "iteration",
                "policy_value_error",
                "DREAM random search confirmation: policy-value error",
                "Absolute error from Leduc game value",
                plot_dir / "confirmation_policy_value_error.png",
                title_config=title_config,
            )
        )

    paired = paired_differences_vs_baseline(
        confirmation_summary,
        metric="final_exploitability",
        baseline_label=baseline_label,
    )
    paired.to_csv(table_dir / "confirmation_paired_differences_final_exploitability.csv", index=False)
    if not paired.empty:
        delta_summary = paired.groupby("config_label")["delta_candidate_minus_baseline"].agg(
            ["mean", "sem"]
        ).reset_index()
        paths.append(
            _paired_delta_plot(
                delta_summary,
                plot_dir / "confirmation_paired_delta_final_exploitability.png",
                title_config=title_config,
            )
        )
    if "final_average_policy_value" in confirmation_summary.columns:
        paired_value = paired_differences_vs_baseline(
            confirmation_summary,
            metric="final_average_policy_value",
            baseline_label=baseline_label,
        )
        paired_value.to_csv(
            table_dir / "confirmation_paired_differences_final_average_policy_value.csv",
            index=False,
        )
        if not paired_value.empty:
            delta_summary = paired_value.groupby("config_label")["delta_candidate_minus_baseline"].agg(
                ["mean", "sem"]
            ).reset_index()
            paths.append(
                _paired_delta_plot(
                    delta_summary,
                    plot_dir / "confirmation_paired_delta_final_average_policy_value.png",
                    title="DREAM random search: paired final average policy value delta",
                    xlabel="Candidate minus baseline average policy value",
                    title_config=title_config,
                )
            )
    return paths


def _barh_metric(
    df: pd.DataFrame,
    metric_col: str,
    se_col: str,
    title: str,
    xlabel: str,
    output_path: Path,
    average_policy_value_target: float = LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    title_config: Dict | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df["config_label"], df[metric_col], xerr=df.get(se_col), capsize=3)
    if is_exploitability_metric(metric_col):
        add_nash_exploitability_target(ax, axis="x")
        ax.legend()
    elif is_average_policy_value_metric(metric_col.replace("_mean", "")):
        add_average_policy_value_target(ax, target=average_policy_value_target, axis="x")
        ax.legend()
    ax.set_title(format_plot_title(title, title_config))
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_grouped_curve(
    curves_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    average_policy_value_target: float = LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    title_config: Dict | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for label, group in curves_df.groupby("config_label"):
        x, y, e = _mean_curve(group, x_col, y_col)
        ax.plot(x, y, linewidth=2.0, label=label)
        ax.fill_between(x, y - e, y + e, alpha=0.12)
    if is_exploitability_metric(y_col):
        add_nash_exploitability_target(ax)
    elif is_average_policy_value_metric(y_col):
        add_average_policy_value_target(ax, target=average_policy_value_target)
    ax.set_title(format_plot_title(title, title_config))
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _paired_delta_plot(
    delta_summary: pd.DataFrame,
    output_path: Path,
    title: str = "DREAM random search: paired final exploitability delta",
    xlabel: str = "Candidate minus baseline exploitability",
    title_config: Dict | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.barh(
        delta_summary["config_label"],
        delta_summary["mean"],
        xerr=delta_summary["sem"].fillna(0.0),
        capsize=4,
    )
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_title(format_plot_title(title, title_config))
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
