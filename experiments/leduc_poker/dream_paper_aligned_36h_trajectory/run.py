"""Training, deferred policy fitting, and analysis for DREAM Experiment 46."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dream_exp46_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dream_exp46_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyspiel  # noqa: E402
import torch  # noqa: E402
from open_spiel.python import policy as osp_policy  # noqa: E402
from open_spiel.python.algorithms import expected_game_score  # noqa: E402

from dream_poker.experiment_runner import make_dream_solver  # noqa: E402
from dream_poker.experiment_utils import cleanup_training_memory  # noqa: E402
from dream_poker.networks import build_network  # noqa: E402
from dream_poker.seeding import set_seed  # noqa: E402
from experiments.leduc_poker.dream_frozen_reservoir_distillation_audit.distillation import (  # noqa: E402,E501
    EmpiricalReservoirPolicy,
    GroupedReservoir,
    NetworkPolicy,
    exact_policy_metrics,
    freeze_strategy_reservoir,
    group_reservoir,
    initial_policy_state,
    save_frozen_reservoir,
    save_grouped_reservoir,
    save_policy_model,
)

from .config import (  # noqa: E402
    DISTILLATION_SEED_OFFSET,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    PRODUCTION_SEEDS,
    RAW_RESERVOIR_TIME_ENDPOINT_HOURS,
    SELECTED_POLICY_ARM,
    SMOKE_SEEDS,
    TARGET_NODES_TOUCHED,
    build_config,
)

_LOGGER = logging.getLogger("dream_poker.experiment.paper_aligned_36h_trajectory")


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2)
        handle.write("\n")


def read_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in fields})


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_grouped_reservoir(path: Path) -> GroupedReservoir:
    with np.load(path, allow_pickle=False) as data:
        return GroupedReservoir(
            info_states=np.asarray(data["info_states"], dtype=np.float32),
            strategies=np.asarray(data["strategies"], dtype=np.float32),
            legal_masks=np.asarray(data["legal_masks"], dtype=np.bool_),
            objective_masses=np.asarray(data["objective_masses"], dtype=np.float64),
            objective_weights=np.asarray(data["objective_weights"], dtype=np.float32),
            source_rows=int(data["source_rows"]),
        )


def _progress_row(solver, seed: int, active_seconds: float) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "seed": int(seed),
        "iteration": int(solver._iteration),  # pylint: disable=protected-access
        "nodes_touched": int(solver._nodes_touched),  # pylint: disable=protected-access
        "active_training_seconds": float(active_seconds),
        "training_hours": float(active_seconds) / 3600.0,
        "learning_rate": float(solver._current_learning_rate),  # pylint: disable=protected-access
        "advantage_loss_player_0": float(solver._last_advantage_loss[0]),
        "advantage_loss_player_1": float(solver._last_advantage_loss[1]),
        "baseline_loss_player_0": float(solver._last_baseline_loss[0]),
        "baseline_loss_player_1": float(solver._last_baseline_loss[1]),
        "strategy_buffer_size": int(len(solver.strategy_memory)),
        "strategy_observations_seen": int(solver.strategy_memory.add_calls),
        "advantage_buffer_size_player_0": int(len(solver._advantage_memories[0])),
        "advantage_buffer_size_player_1": int(len(solver._advantage_memories[1])),
        "baseline_buffer_size_player_0": int(len(solver._baseline_replays[0])),
        "baseline_buffer_size_player_1": int(len(solver._baseline_replays[1])),
        "cumulative_traversal_seconds": float(solver._cumulative_traversal_seconds),
        "cumulative_advantage_training_seconds": float(
            solver._cumulative_advantage_training_seconds
        ),
        "cumulative_baseline_training_seconds": float(
            solver._cumulative_baseline_training_seconds
        ),
        "cumulative_policy_training_seconds": float(
            solver._cumulative_policy_training_seconds
        ),
    }


def _save_snapshot(
    *,
    solver,
    game,
    seed_dir: Path,
    run_dir: Path,
    checkpoint_index: int,
    progress: Mapping,
    time_checkpoint_index: int | None,
    scheduled_active_seconds: float | None,
    crossed_scheduled_count: int,
    is_node_15m_endpoint: bool,
    is_final_endpoint: bool,
    save_raw_reservoir: bool,
) -> dict:
    checkpoint_id = f"checkpoint_{int(checkpoint_index):03d}"
    frozen = freeze_strategy_reservoir(solver, game)
    grouped = group_reservoir(frozen)
    grouped_path = seed_dir / "grouped_checkpoints" / f"{checkpoint_id}.npz"
    save_grouped_reservoir(grouped_path, grouped)
    raw_path = None
    if save_raw_reservoir:
        raw_path = seed_dir / "raw_endpoint_reservoirs" / f"{checkpoint_id}.npz"
        save_frozen_reservoir(raw_path, frozen)
    row = {
        **dict(progress),
        "checkpoint_id": checkpoint_id,
        "checkpoint_index": int(checkpoint_index),
        "time_checkpoint_index": time_checkpoint_index,
        "scheduled_active_seconds": scheduled_active_seconds,
        "crossed_scheduled_count": int(crossed_scheduled_count),
        "is_node_15m_endpoint": bool(is_node_15m_endpoint),
        "is_final_endpoint": bool(is_final_endpoint),
        "grouped_reservoir_path": str(grouped_path.relative_to(run_dir)),
        "grouped_reservoir_sha256": sha256(grouped_path),
        "grouped_reservoir_size_bytes": int(grouped_path.stat().st_size),
        "raw_reservoir_path": "" if raw_path is None else str(raw_path.relative_to(run_dir)),
        "raw_reservoir_sha256": "" if raw_path is None else sha256(raw_path),
        "reservoir_rows": int(frozen.size),
        "reservoir_observations_seen": int(frozen.observations_seen),
        "unique_information_sets": int(grouped.size),
        "info_state_width": int(grouped.info_states.shape[1]),
        "num_actions": int(grouped.strategies.shape[1]),
    }
    del grouped, frozen
    gc.collect()
    return row


def run_training_seed(*, seed: int, output_dir: Path, smoke: bool = False) -> dict:
    """Train one seed for a fixed active-time budget and freeze diagnostics."""
    config = build_config(smoke=smoke)
    output_dir = Path(output_dir)
    seed_dir = output_dir / f"seed_{int(seed)}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_json(seed_dir / "config.json", config)
    set_seed(int(seed))
    game = pyspiel.load_game(str(config["game_name"]))
    solver = make_dream_solver(config, int(seed))

    checkpoint_rows: list[dict] = []
    progress_rows: list[dict] = []
    latest_progress: dict = {}
    interval = float(config["checkpoint_interval_seconds"])
    next_scheduled = interval
    next_time_index = 1
    node_endpoint_saved = False
    excluded_checkpoint_seconds = 0.0
    training_started = time.perf_counter()

    manifest_path = seed_dir / str(config["checkpoint_manifest_filename"])

    def persist_manifest() -> None:
        write_csv(manifest_path, checkpoint_rows)

    def _raw_needed(
        *, scheduled_seconds: float | None, node_endpoint: bool, final_endpoint: bool
    ) -> bool:
        if node_endpoint or final_endpoint:
            return True
        if scheduled_seconds is None:
            return False
        hour = float(scheduled_seconds) / 3600.0
        return any(abs(hour - endpoint) < 1e-9 for endpoint in RAW_RESERVOIR_TIME_ENDPOINT_HOURS)

    def record_checkpoint(
        *,
        progress: Mapping,
        time_index: int | None,
        scheduled_seconds: float | None,
        crossed_count: int,
        node_endpoint: bool,
        final_endpoint: bool,
    ) -> dict:
        raw_needed = _raw_needed(
            scheduled_seconds=scheduled_seconds,
            node_endpoint=node_endpoint,
            final_endpoint=final_endpoint,
        )
        if checkpoint_rows and int(checkpoint_rows[-1]["iteration"]) == int(
            progress["iteration"]
        ):
            row = checkpoint_rows[-1]
            if time_index is not None:
                row["time_checkpoint_index"] = int(time_index)
                row["scheduled_active_seconds"] = float(scheduled_seconds)
                row["crossed_scheduled_count"] = int(crossed_count)
            row["is_node_15m_endpoint"] = bool(row["is_node_15m_endpoint"] or node_endpoint)
            row["is_final_endpoint"] = bool(row["is_final_endpoint"] or final_endpoint)
            if raw_needed and not row.get("raw_reservoir_path"):
                frozen = freeze_strategy_reservoir(solver, game)
                raw_path = (
                    seed_dir / "raw_endpoint_reservoirs" / f"{row['checkpoint_id']}.npz"
                )
                save_frozen_reservoir(raw_path, frozen)
                row["raw_reservoir_path"] = str(raw_path.relative_to(output_dir))
                row["raw_reservoir_sha256"] = sha256(raw_path)
                del frozen
                gc.collect()
            persist_manifest()
            return row
        row = _save_snapshot(
            solver=solver,
            game=game,
            seed_dir=seed_dir,
            run_dir=output_dir,
            checkpoint_index=len(checkpoint_rows),
            progress=progress,
            time_checkpoint_index=time_index,
            scheduled_active_seconds=scheduled_seconds,
            crossed_scheduled_count=crossed_count,
            is_node_15m_endpoint=node_endpoint,
            is_final_endpoint=final_endpoint,
            save_raw_reservoir=raw_needed,
        )
        checkpoint_rows.append(row)
        persist_manifest()
        _LOGGER.info(
            "Seed %s froze %s at %.3f active hours and %s nodes",
            seed,
            row["checkpoint_id"],
            row["training_hours"],
            row["nodes_touched"],
        )
        return row

    def post_iteration(current_solver, completed_iteration: int) -> None:
        nonlocal excluded_checkpoint_seconds, next_scheduled, next_time_index
        nonlocal node_endpoint_saved
        callback_started = time.perf_counter()
        active_seconds = callback_started - training_started - excluded_checkpoint_seconds
        progress = _progress_row(current_solver, int(seed), active_seconds)
        if int(completed_iteration) != int(progress["iteration"]):
            raise RuntimeError("Solver callback iteration metadata is inconsistent")
        latest_progress.clear()
        latest_progress.update(progress)
        if completed_iteration == 1 or completed_iteration % int(
            config["training_progress_every"]
        ) == 0:
            progress_rows.append(dict(progress))

        time_index = None
        scheduled_seconds = None
        crossed_count = 0
        if active_seconds >= next_scheduled:
            time_index = next_time_index
            scheduled_seconds = next_scheduled
            while active_seconds >= next_scheduled:
                crossed_count += 1
                next_scheduled += interval
                next_time_index += 1
        node_endpoint = not node_endpoint_saved and int(progress["nodes_touched"]) >= int(
            config["target_nodes_touched"]
        )
        if node_endpoint:
            node_endpoint_saved = True
        if time_index is not None or node_endpoint:
            record_checkpoint(
                progress=progress,
                time_index=time_index,
                scheduled_seconds=scheduled_seconds,
                crossed_count=crossed_count,
                node_endpoint=node_endpoint,
                final_endpoint=False,
            )
        excluded_checkpoint_seconds += time.perf_counter() - callback_started

    curves = solver.solve(
        policy_training_mode="final_only",
        isolate_policy_training_rng=True,
        target_iteration=int(config["num_iterations"]),
        start_time=training_started,
        max_training_seconds=float(config["training_time_budget_seconds"]),
        post_iteration_callback=post_iteration,
        exclude_post_iteration_callback_time=True,
    )
    outer_seconds = time.perf_counter() - training_started
    if not curves.empty:
        raise RuntimeError("Timed learner unexpectedly fitted or evaluated an average policy")
    if not latest_progress:
        raise RuntimeError("DREAM did not complete a training iteration")

    final_checkpoint_started = time.perf_counter()
    record_checkpoint(
        progress=latest_progress,
        time_index=None,
        scheduled_seconds=None,
        crossed_count=0,
        node_endpoint=(
            not node_endpoint_saved
            and int(latest_progress["nodes_touched"]) >= int(config["target_nodes_touched"])
        ),
        final_endpoint=True,
    )
    excluded_checkpoint_seconds += time.perf_counter() - final_checkpoint_started
    if not progress_rows or int(progress_rows[-1]["iteration"]) != int(
        latest_progress["iteration"]
    ):
        progress_rows.append(dict(latest_progress))
    progress_path = seed_dir / "source_training_progress.csv"
    write_csv(progress_path, progress_rows)

    active_seconds = float(latest_progress["active_training_seconds"])
    requested_seconds = float(config["training_time_budget_seconds"])
    if not smoke and active_seconds < requested_seconds:
        raise RuntimeError("Training stopped before the 36-hour active-time boundary")
    result = {
        "status": "complete",
        "stage": "training",
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "seed": int(seed),
        "smoke": bool(smoke),
        "active_training_seconds": active_seconds,
        "training_budget_seconds": requested_seconds,
        "budget_overshoot_seconds": active_seconds - requested_seconds,
        "checkpoint_time_excluded_seconds": float(excluded_checkpoint_seconds),
        "outer_training_stage_seconds": float(outer_seconds),
        "final_iteration": int(latest_progress["iteration"]),
        "final_nodes_touched": int(latest_progress["nodes_touched"]),
        "hit_wall_clock_limit": bool(active_seconds >= requested_seconds),
        "num_checkpoints": len(checkpoint_rows),
        "num_time_checkpoints": sum(
            row.get("time_checkpoint_index") not in (None, "") for row in checkpoint_rows
        ),
        "node_15m_endpoint_saved": any(
            bool(row["is_node_15m_endpoint"]) for row in checkpoint_rows
        ),
        "manifest_path": str(manifest_path.relative_to(output_dir)),
        "progress_path": str(progress_path.relative_to(output_dir)),
    }
    write_json(seed_dir / "training_result.json", result)
    write_json(seed_dir / "TRAINING_SUCCESS.json", {"status": "complete", "seed": seed})
    if hasattr(solver, "close"):
        solver.close()
    del solver
    cleanup_training_memory()
    gc.collect()
    return result


def _masked_log_softmax(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.masked_fill(~masks, -1e20), dim=-1)


def fit_grouped_policy(
    *,
    grouped: GroupedReservoir,
    hidden_layers: Sequence[int],
    initial_state: Mapping[str, torch.Tensor],
    learning_rate: float,
    batch_size: int,
    train_steps: int,
    sampling_seed: int,
    gradient_clip_norm: float | None,
) -> tuple[torch.nn.Module, dict]:
    """Fit the selected grouped soft-target CE objective from sufficient statistics."""
    model = build_network(
        "mlp",
        int(grouped.info_states.shape[1]),
        hidden_layers,
        int(grouped.strategies.shape[1]),
    )
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    rng = np.random.default_rng(int(sampling_seed))
    size = int(grouped.size)
    effective_batch = min(int(batch_size), size)
    if effective_batch <= 0:
        raise ValueError("Cannot fit an empty grouped reservoir")
    last_loss = float("nan")
    started = time.perf_counter()
    for step in range(int(train_steps)):
        if effective_batch == size:
            indices = np.arange(size, dtype=np.int64)
        else:
            indices = rng.choice(size, size=effective_batch, replace=False)
        x = torch.from_numpy(grouped.info_states[indices])
        targets = torch.from_numpy(grouped.strategies[indices])
        masks = torch.from_numpy(grouped.legal_masks[indices])
        weights = torch.from_numpy(grouped.objective_weights[indices])
        logits = model(x)
        per_example = -(targets * _masked_log_softmax(logits, masks)).sum(dim=1)
        loss = torch.mean(per_example * weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite grouped CE loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
    elapsed = time.perf_counter() - started
    return model, {
        "optimizer_steps": int(train_steps),
        "batch_size": int(effective_batch),
        "network_examples_processed": int(train_steps) * int(effective_batch),
        "final_training_loss": last_loss,
        "fit_seconds": float(elapsed),
    }


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _optional_int(value) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(value)


def _optional_float(value) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _tabular_policy(game, model):
    candidate = NetworkPolicy(model)
    return osp_policy.tabular_policy_from_callable(game, candidate.action_probabilities)


def _seat_averaged_payoff(game, current, previous) -> float:
    as_player_zero = float(
        expected_game_score.policy_value(game.new_initial_state(), [current, previous])[0]
    )
    previous_as_player_zero = float(
        expected_game_score.policy_value(game.new_initial_state(), [previous, current])[0]
    )
    return 0.5 * (as_player_zero - previous_as_player_zero)


def run_evaluation_seed(
    *, seed: int, training_dir: Path, output_dir: Path, smoke: bool = False, resume: bool = True
) -> dict:
    """Fit and evaluate a fresh grouped-CE policy at every frozen checkpoint."""
    config = build_config(smoke=smoke)
    training_dir = Path(training_dir)
    output_dir = Path(output_dir)
    source_seed_dir = training_dir / f"seed_{int(seed)}"
    training_result = read_json(source_seed_dir / "training_result.json")
    manifest = read_csv(training_dir / str(training_result["manifest_path"]))
    if not manifest:
        raise RuntimeError(f"No checkpoints found for seed {seed}")
    evaluation_seed_dir = output_dir / f"seed_{int(seed)}"
    metrics_dir = evaluation_seed_dir / "checkpoint_metrics"
    policies_dir = evaluation_seed_dir / "policies"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    policies_dir.mkdir(parents=True, exist_ok=True)
    game = pyspiel.load_game(str(config["game_name"]))
    fit_seed = DISTILLATION_SEED_OFFSET + int(seed)
    first_grouped = load_grouped_reservoir(training_dir / manifest[0]["grouped_reservoir_path"])
    base_state = initial_policy_state(
        network_type="mlp",
        input_size=int(first_grouped.info_states.shape[1]),
        hidden_layers=config["policy_network_layers"],
        output_size=int(first_grouped.strategies.shape[1]),
        seed=fit_seed,
    )
    del first_grouped

    metric_rows: list[dict] = []
    head_to_head_rows: list[dict] = []
    previous_tabular = None
    previous_id = None
    for position, source in enumerate(manifest, start=1):
        checkpoint_id = str(source["checkpoint_id"])
        metric_path = metrics_dir / f"{checkpoint_id}.json"
        grouped_path = training_dir / str(source["grouped_reservoir_path"])
        if sha256(grouped_path) != str(source["grouped_reservoir_sha256"]):
            raise RuntimeError(f"Checkpoint checksum mismatch: {grouped_path}")
        # Resume is deliberately checkpoint-granular. Models are refitted when
        # needed for adjacent head-to-head comparisons, but completed metric
        # files remain authoritative and are not overwritten unnecessarily.
        existing = read_json(metric_path) if resume and metric_path.is_file() else None
        grouped = load_grouped_reservoir(grouped_path)
        model, fit_diagnostics = fit_grouped_policy(
            grouped=grouped,
            hidden_layers=config["policy_network_layers"],
            initial_state=base_state,
            learning_rate=float(config["learning_rate"]),
            batch_size=int(config["batch_size_strategy"]),
            train_steps=int(config["fresh_reference_fit_steps"]),
            sampling_seed=fit_seed + 1,
            gradient_clip_norm=config.get("gradient_clip_norm"),
        )
        empirical = exact_policy_metrics(game, EmpiricalReservoirPolicy(grouped))
        neural = exact_policy_metrics(game, NetworkPolicy(model))
        weights_path = policies_dir / f"{checkpoint_id}.pt"
        row = {
            "status": "complete",
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "seed": int(seed),
            "checkpoint_id": checkpoint_id,
            "checkpoint_index": int(source["checkpoint_index"]),
            "time_checkpoint_index": _optional_int(source["time_checkpoint_index"]),
            "scheduled_active_seconds": _optional_float(source["scheduled_active_seconds"]),
            "iteration": int(source["iteration"]),
            "nodes_touched": int(source["nodes_touched"]),
            "active_training_seconds": float(source["active_training_seconds"]),
            "training_hours": float(source["training_hours"]),
            "is_node_15m_endpoint": _bool(source["is_node_15m_endpoint"]),
            "is_final_endpoint": _bool(source["is_final_endpoint"]),
            "policy_arm": SELECTED_POLICY_ARM,
            "fit_seed": int(fit_seed),
            "neural_exploitability": neural["exploitability"],
            "neural_nash_conv": neural["nash_conv"],
            "neural_policy_value_player_0": neural["policy_value_player_0"],
            "empirical_reservoir_exploitability": empirical["exploitability"],
            "empirical_reservoir_policy_value_player_0": empirical["policy_value_player_0"],
            "distillation_gap": neural["exploitability"] - empirical["exploitability"],
            "reservoir_rows": int(source["reservoir_rows"]),
            "unique_information_sets": int(grouped.size),
            "grouped_reservoir_sha256": str(source["grouped_reservoir_sha256"]),
            "weights_path": str(weights_path.relative_to(output_dir)),
            **fit_diagnostics,
        }
        if existing is None:
            write_json(metric_path, row)
        else:
            # Refitting is deterministic; reject stale/incompatible artifacts.
            if (
                int(existing.get("experiment_id", -1)) != EXPERIMENT_ID
                or existing.get("grouped_reservoir_sha256")
                != source["grouped_reservoir_sha256"]
            ):
                raise RuntimeError(f"Invalid resumed metric file: {metric_path}")
            row = existing
        save_policy_model(
            weights_path,
            model,
            {
                "experiment_id": EXPERIMENT_ID,
                "experiment_name": EXPERIMENT_NAME,
                "seed": int(seed),
                "checkpoint_id": checkpoint_id,
                "policy_arm": SELECTED_POLICY_ARM,
                "network_type": "mlp",
                "policy_network_layers": list(config["policy_network_layers"]),
                "info_state_size": int(grouped.info_states.shape[1]),
                "num_actions": int(grouped.strategies.shape[1]),
                "metrics": row,
            },
        )
        current_tabular = _tabular_policy(game, model)
        if previous_tabular is not None:
            head_to_head_rows.append(
                {
                    "seed": int(seed),
                    "previous_checkpoint_id": previous_id,
                    "current_checkpoint_id": checkpoint_id,
                    "current_time_checkpoint_index": row.get("time_checkpoint_index"),
                    "current_training_hours": float(row["training_hours"]),
                    "current_nodes_touched": int(row["nodes_touched"]),
                    "current_seat_averaged_payoff": _seat_averaged_payoff(
                        game, current_tabular, previous_tabular
                    ),
                }
            )
        previous_tabular = current_tabular
        previous_id = checkpoint_id
        metric_rows.append(row)
        write_csv(evaluation_seed_dir / str(config["checkpoint_metrics_filename"]), metric_rows)
        write_csv(evaluation_seed_dir / "adjacent_head_to_head.csv", head_to_head_rows)
        _LOGGER.info(
            "Seed %s evaluated %s (%s/%s): exploitability %.6f",
            seed,
            checkpoint_id,
            position,
            len(manifest),
            float(row["neural_exploitability"]),
        )
        del model, grouped
        gc.collect()

    metrics_path = evaluation_seed_dir / str(config["checkpoint_metrics_filename"])
    h2h_path = evaluation_seed_dir / "adjacent_head_to_head.csv"
    result = {
        "status": "complete",
        "stage": "evaluation",
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "seed": int(seed),
        "smoke": bool(smoke),
        "policy_arm": SELECTED_POLICY_ARM,
        "num_source_checkpoints": len(manifest),
        "num_metric_rows": len(metric_rows),
        "num_adjacent_head_to_head_rows": len(head_to_head_rows),
        "metrics_path": str(metrics_path.relative_to(output_dir)),
        "head_to_head_path": str(h2h_path.relative_to(output_dir)),
    }
    write_json(evaluation_seed_dir / "evaluation_result.json", result)
    write_json(
        evaluation_seed_dir / "EVALUATION_SUCCESS.json",
        {"status": "complete", "seed": int(seed)},
    )
    return result


def _stats(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": float("nan"), "std": float("nan"), "se": float("nan"), "n": 0}
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return {
        "mean": float(np.mean(array)),
        "std": std,
        "se": float(std / math.sqrt(len(array))),
        "n": int(len(array)),
    }


def _time_summary(rows: Sequence[Mapping]) -> list[dict]:
    summary = []
    indices = sorted(
        {
            int(row["time_checkpoint_index"])
            for row in rows
            if row.get("time_checkpoint_index") not in (None, "", "None")
        }
    )
    for index in indices:
        selected = [
            row
            for row in rows
            if row.get("time_checkpoint_index") not in (None, "", "None")
            and int(row["time_checkpoint_index"]) == index
        ]
        record = {"time_checkpoint_index": index, "n_seeds": len(selected)}
        for source, target in (
            ("training_hours", "training_hours"),
            ("nodes_touched", "nodes_touched"),
            ("neural_exploitability", "neural_exploitability"),
            ("empirical_reservoir_exploitability", "empirical_exploitability"),
            ("distillation_gap", "distillation_gap"),
            ("neural_policy_value_player_0", "policy_value_player_0"),
        ):
            stats = _stats([float(row[source]) for row in selected])
            for statistic, value in stats.items():
                record[f"{target}_{statistic}"] = value
        summary.append(record)
    return summary


def _node_summary(rows: Sequence[Mapping], grid_points: int = 73) -> list[dict]:
    seeds = sorted({int(row["seed"]) for row in rows})
    paths = {
        seed: sorted(
            (row for row in rows if int(row["seed"]) == seed),
            key=lambda row: float(row["nodes_touched"]),
        )
        for seed in seeds
    }
    common_min = max(min(float(row["nodes_touched"]) for row in path) for path in paths.values())
    common_max = min(max(float(row["nodes_touched"]) for row in path) for path in paths.values())
    grid = np.linspace(common_min, common_max, int(grid_points))
    if common_min <= TARGET_NODES_TOUCHED <= common_max:
        grid = np.unique(np.concatenate((grid, [float(TARGET_NODES_TOUCHED)])))
    summary = []
    for node in grid:
        values = {"neural": [], "empirical": [], "gap": []}
        for path in paths.values():
            x = np.asarray([float(row["nodes_touched"]) for row in path])
            for key, field in (
                ("neural", "neural_exploitability"),
                ("empirical", "empirical_reservoir_exploitability"),
                ("gap", "distillation_gap"),
            ):
                y = np.asarray([float(row[field]) for row in path])
                values[key].append(float(np.interp(node, x, y)))
        record = {"nodes_touched": float(node)}
        for key, collected in values.items():
            stats = _stats(collected)
            for statistic, value in stats.items():
                record[f"{key}_{statistic}"] = value
        summary.append(record)
    return summary


def _plot_trajectory(
    rows: Sequence[Mapping], summary: Sequence[Mapping], *, by_nodes: bool, output: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    x_field = "nodes_touched" if by_nodes else "training_hours"
    for seed in sorted({int(row["seed"]) for row in rows}):
        selected = sorted(
            (row for row in rows if int(row["seed"]) == seed),
            key=lambda row: float(row[x_field]),
        )
        ax.plot(
            [float(row[x_field]) for row in selected],
            [float(row["neural_exploitability"]) for row in selected],
            color="#2f6f9f",
            alpha=0.22,
            linewidth=1.0,
        )
    if by_nodes:
        x = np.asarray([float(row["nodes_touched"]) for row in summary])
        neural = np.asarray([float(row["neural_mean"]) for row in summary])
        neural_se = np.asarray([float(row["neural_se"]) for row in summary])
        empirical = np.asarray([float(row["empirical_mean"]) for row in summary])
        empirical_se = np.asarray([float(row["empirical_se"]) for row in summary])
    else:
        x = np.asarray([float(row["training_hours_mean"]) for row in summary])
        neural = np.asarray([float(row["neural_exploitability_mean"]) for row in summary])
        neural_se = np.asarray([float(row["neural_exploitability_se"]) for row in summary])
        empirical = np.asarray([float(row["empirical_exploitability_mean"]) for row in summary])
        empirical_se = np.asarray([float(row["empirical_exploitability_se"]) for row in summary])
    ax.plot(x, neural, color="#1f5a85", linewidth=2.4, label="Neural average policy")
    ax.fill_between(x, neural - neural_se, neural + neural_se, color="#1f5a85", alpha=0.18)
    ax.plot(x, empirical, color="#d17a22", linewidth=2.2, label="Grouped empirical policy")
    ax.fill_between(
        x, empirical - empirical_se, empirical + empirical_se, color="#d17a22", alpha=0.16
    )
    ax.set_xlabel("Nodes touched" if by_nodes else "Active training time (hours)")
    ax.set_ylabel("Exploitability (NashConv / 2)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_time_metric(
    summary: Sequence[Mapping], mean_field: str, se_field: str, ylabel: str, output: Path
) -> None:
    x = np.asarray([float(row["training_hours_mean"]) for row in summary])
    mean = np.asarray([float(row[mean_field]) for row in summary])
    se = np.asarray([float(row[se_field]) for row in summary])
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.plot(x, mean, color="#1f5a85", linewidth=2.4)
    ax.fill_between(x, mean - se, mean + se, color="#1f5a85", alpha=0.18)
    ax.set_xlabel("Active training time (hours)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _endpoint_rows(rows: Sequence[Mapping], endpoint: str) -> list[dict]:
    if endpoint == "15m":
        selected = [row for row in rows if _bool(row.get("is_node_15m_endpoint"))]
    elif endpoint == "36h":
        selected = [row for row in rows if _bool(row.get("is_final_endpoint"))]
    else:
        target_index = int(float(endpoint.rstrip("h")) * 2)
        selected = [
            row
            for row in rows
            if row.get("time_checkpoint_index") not in (None, "", "None")
            and int(row["time_checkpoint_index"]) == target_index
        ]
    return selected


def _endpoint_summary(rows: Sequence[Mapping], endpoint: str) -> dict:
    selected = _endpoint_rows(rows, endpoint)
    record = {"endpoint": endpoint, "n_seeds": len(selected)}
    for field in (
        "neural_exploitability",
        "empirical_reservoir_exploitability",
        "distillation_gap",
        "neural_policy_value_player_0",
        "nodes_touched",
        "training_hours",
    ):
        stats = _stats([float(row[field]) for row in selected])
        for statistic, value in stats.items():
            record[f"{field}_{statistic}"] = value
    return record


def aggregate_results(
    *, metric_rows: Sequence[Mapping], head_to_head_rows: Sequence[Mapping], output_dir: Path,
    expected_seed_count: int
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted({int(row["seed"]) for row in metric_rows})
    if len(seeds) != int(expected_seed_count):
        raise RuntimeError(f"Expected {expected_seed_count} seeds, got {seeds}")
    time_summary = _time_summary(metric_rows)
    node_summary = _node_summary(metric_rows)
    endpoint_summaries = [
        _endpoint_summary(metric_rows, endpoint)
        for endpoint in ("12h", "24h", "36h", "15m")
    ]
    endpoint_seed_rows = []
    for endpoint in ("12h", "24h", "36h", "15m"):
        endpoint_seed_rows.extend(
            dict(row, endpoint=endpoint) for row in _endpoint_rows(metric_rows, endpoint)
        )

    auc_rows = []
    final_window_rows = []
    for seed in seeds:
        path = sorted(
            (row for row in metric_rows if int(row["seed"]) == seed),
            key=lambda row: float(row["training_hours"]),
        )
        hours = np.asarray([float(row["training_hours"]) for row in path])
        nodes = np.asarray([float(row["nodes_touched"]) for row in path])
        exploitability = np.asarray([float(row["neural_exploitability"]) for row in path])
        auc_rows.append(
            {
                "seed": seed,
                "time_auc": float(np.trapz(exploitability, hours)),
                "node_auc": float(np.trapz(exploitability, nodes)),
            }
        )
        window = path[-5:]
        final_window_rows.append(
            {
                "seed": seed,
                "num_checkpoints": len(window),
                "mean_neural_exploitability": float(
                    np.mean([float(row["neural_exploitability"]) for row in window])
                ),
                "std_neural_exploitability": float(
                    np.std([float(row["neural_exploitability"]) for row in window], ddof=1)
                ) if len(window) > 1 else 0.0,
            }
        )

    write_csv(output_dir / "checkpoint_seed_metrics.csv", metric_rows)
    write_csv(output_dir / "trajectory_by_training_time_summary.csv", time_summary)
    write_csv(output_dir / "trajectory_by_nodes_summary.csv", node_summary)
    write_csv(output_dir / "endpoint_seed_metrics.csv", endpoint_seed_rows)
    write_csv(output_dir / "endpoint_aggregate_summary.csv", endpoint_summaries)
    write_csv(output_dir / "adjacent_checkpoint_head_to_head.csv", head_to_head_rows)
    write_csv(output_dir / "seed_auc_summary.csv", auc_rows)
    write_csv(output_dir / "final_window_summary.csv", final_window_rows)
    _plot_trajectory(
        metric_rows,
        time_summary,
        by_nodes=False,
        output=output_dir / "exploitability_by_training_time.png",
    )
    _plot_trajectory(
        metric_rows,
        node_summary,
        by_nodes=True,
        output=output_dir / "exploitability_by_nodes.png",
    )
    _plot_time_metric(
        time_summary,
        "distillation_gap_mean",
        "distillation_gap_se",
        "Neural minus empirical exploitability",
        output_dir / "distillation_gap_by_training_time.png",
    )
    _plot_time_metric(
        time_summary,
        "policy_value_player_0_mean",
        "policy_value_player_0_se",
        "Policy value for player 0",
        output_dir / "policy_value_by_training_time.png",
    )
    result = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "num_seeds": len(seeds),
        "seeds": seeds,
        "num_checkpoint_rows": len(metric_rows),
        "num_head_to_head_rows": len(head_to_head_rows),
        "primary_endpoint": "36h neural exact exploitability",
        "endpoint_summaries": endpoint_summaries,
    }
    write_json(output_dir / "aggregate_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate", "aggregate", "smoke"))
    parser.add_argument("--seed", type=int, default=PRODUCTION_SEEDS[0])
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.mode == "train":
        print(json.dumps(run_training_seed(seed=args.seed, output_dir=args.output_dir), indent=2))
    elif args.mode == "evaluate":
        if args.training_dir is None:
            raise SystemExit("--training-dir is required for evaluate")
        print(
            json.dumps(
                run_evaluation_seed(
                    seed=args.seed,
                    training_dir=args.training_dir,
                    output_dir=args.output_dir,
                ),
                indent=2,
            )
        )
    elif args.mode == "smoke":
        training = args.output_dir / "training"
        evaluation = args.output_dir / "evaluation"
        run_training_seed(seed=SMOKE_SEEDS[0], output_dir=training, smoke=True)
        run_evaluation_seed(
            seed=SMOKE_SEEDS[0],
            training_dir=training,
            output_dir=evaluation,
            smoke=True,
            resume=False,
        )
        result_path = evaluation / f"seed_{SMOKE_SEEDS[0]}" / "evaluation_result.json"
        result = read_json(result_path)
        metrics = read_csv(evaluation / str(result["metrics_path"]))
        h2h = read_csv(evaluation / str(result["head_to_head_path"]))
        print(
            json.dumps(
                aggregate_results(
                    metric_rows=metrics,
                    head_to_head_rows=h2h,
                    output_dir=args.output_dir / "analysis",
                    expected_seed_count=1,
                ),
                indent=2,
            )
        )
    else:
        raise SystemExit("Use the cloud wrapper for multi-seed aggregation")


if __name__ == "__main__":
    main()
