"""Checkpoint snapshot and head-to-head helpers for DREAM experiments."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    import pyspiel
    from open_spiel.python import policy as osp_policy
    from open_spiel.python.algorithms import expected_game_score
    from open_spiel.python.algorithms import exploitability
except Exception:  # pragma: no cover
    pyspiel = None
    osp_policy = None
    expected_game_score = None
    exploitability = None

from dream_poker.constants import LEDUC_AVERAGE_POLICY_VALUE_TARGET, LEDUC_GAME_VALUE_P0
from dream_poker.experiment_utils import ensure_dir
from dream_poker.networks import build_network
from dream_poker.plotting import (
    add_average_policy_value_target,
    add_nash_exploitability_target,
    format_plot_title,
    is_average_policy_value_metric,
    is_exploitability_metric,
)


CHECKPOINT_PATTERN = re.compile(
    r"leduc_poker_dream_seed_(?P<seed>\d+)_policy_snapshot_(?P<iteration>\d+)_iters\.pt$"
)


class LoadedDREAMPolicy(osp_policy.Policy if osp_policy is not None else object):
    """OpenSpiel policy wrapper around a saved DREAM average-policy network."""

    def __init__(self, game, snapshot_path: Path, map_location: str = "cpu"):
        if osp_policy is not None:
            super().__init__(game, list(range(game.num_players())))
        self.path = Path(snapshot_path)
        try:
            snapshot = torch.load(self.path, map_location=map_location, weights_only=False)
        except TypeError:
            snapshot = torch.load(self.path, map_location=map_location)
        input_size, hidden_sizes, output_size = infer_architecture_from_snapshot(snapshot)
        network_type = str(snapshot.get("policy_network_type", "mlp"))
        self.network = build_network(network_type, input_size, hidden_sizes, output_size)
        self.network.load_state_dict(snapshot["policy_network_state_dict"])
        self.network.eval()
        self.iteration_inside_checkpoint = int(snapshot.get("checkpoint_iteration", -1))
        self.seed = str(snapshot.get("seed", "unknown"))
        self.snapshot = {
            key: value
            for key, value in snapshot.items()
            if key != "policy_network_state_dict"
        }
        del snapshot

    def action_probabilities(self, state, player_id=None):
        del player_id
        cur_player = int(state.current_player())
        legal_actions = list(state.legal_actions(cur_player))
        if not legal_actions:
            return {}
        info_state = np.asarray(state.information_state_tensor(cur_player), dtype=np.float32)[None, :]
        with torch.no_grad():
            logits = self.network(torch.from_numpy(info_state))[0].cpu().numpy()
        legal_logits = np.asarray([logits[a] for a in legal_actions], dtype=np.float64)
        legal_logits -= float(np.max(legal_logits))
        probs = np.exp(legal_logits)
        probs = probs / float(np.sum(probs))
        return {int(a): float(p) for a, p in zip(legal_actions, probs)}


def save_dream_policy_snapshot(solver, seed: int, checkpoint_iteration: int, output_dir: Path, config: Dict) -> Path:
    """Save a lightweight average-policy snapshot for head-to-head analysis."""
    output_dir = ensure_dir(output_dir)
    path = output_dir / f"leduc_poker_dream_seed_{seed}_policy_snapshot_{checkpoint_iteration}_iters.pt"
    torch.save(
        {
            "kind": "dream_policy_snapshot",
            "algorithm": "DREAM-style OpenSpiel",
            "seed": int(seed),
            "checkpoint_iteration": int(checkpoint_iteration),
            "internal_iteration": int(solver._iteration),
            "nodes_touched": int(solver._nodes_touched),
            "game_name": config["game_name"],
            "policy_network_state_dict": solver._policy_network.state_dict(),
            "policy_network_type": solver._policy_network_type,
            "policy_network_layers": list(solver._policy_network_layers),
            "info_state_size": int(solver._info_state_size),
            "num_actions": int(solver._num_actions),
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in dict(config).items()},
        },
        path,
    )
    return path


def save_optional_full_checkpoint(
    solver,
    seed: int,
    checkpoint_iteration: int,
    output_dir: Path,
    config: Dict,
) -> Optional[Path]:
    if not config.get("save_full_solver_checkpoints", False):
        return None
    output_dir = ensure_dir(output_dir)
    path = output_dir / f"leduc_poker_dream_seed_{seed}_full_ckpt_{checkpoint_iteration}_iters.pt"
    torch.save(
        {
            "kind": "dream_full_checkpoint",
            "seed": int(seed),
            "checkpoint_iteration": int(checkpoint_iteration),
            "policy_network_type": solver._policy_network_type,
            "advantage_network_type": solver._advantage_network_type,
            "baseline_network_type": solver._baseline_network_type,
            "policy_network_state_dict": solver._policy_network.state_dict(),
            "advantage_network_state_dicts": [n.state_dict() for n in solver._advantage_networks],
            "baseline_network_state_dicts": [n.state_dict() for n in solver._baseline_networks],
            "optimizer_policy_state_dict": solver._optimizer_policy.state_dict(),
            "optimizer_advantage_state_dicts": [o.state_dict() for o in solver._optimizer_advantages],
            "optimizer_baseline_state_dicts": [o.state_dict() for o in solver._optimizer_baselines],
            "nodes_touched": int(solver._nodes_touched),
            "internal_iteration": int(solver._iteration),
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in dict(config).items()},
        },
        path,
    )
    return path


def infer_architecture_from_snapshot(snapshot: Dict) -> tuple[int, list[int], int]:
    layers = snapshot.get("policy_network_layers")
    input_size = snapshot.get("info_state_size")
    output_size = snapshot.get("num_actions")
    if layers is not None and input_size is not None and output_size is not None:
        return int(input_size), list(map(int, layers)), int(output_size)
    raise ValueError("Snapshot does not contain enough architecture metadata.")


def parse_snapshot_path(path: Path) -> Optional[Dict]:
    match = CHECKPOINT_PATTERN.search(path.name)
    if not match:
        return None
    return {
        "kind": "policy_snapshot",
        "seed": str(match.group("seed")),
        "iteration": int(match.group("iteration")),
        "path": path.resolve(),
    }


def discover_policy_snapshots(
    checkpoint_dir: Path,
    allowed_checkpoint_iterations: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    allowed = None if allowed_checkpoint_iterations is None else set(map(int, allowed_checkpoint_iterations))
    rows = []
    for path in sorted(Path(checkpoint_dir).glob("*.pt")):
        parsed = parse_snapshot_path(path)
        if parsed and (allowed is None or parsed["iteration"] in allowed):
            rows.append(parsed)
    return pd.DataFrame(rows).sort_values(["seed", "iteration"]) if rows else pd.DataFrame(rows)


def load_policies_by_seed(game, checkpoint_df: pd.DataFrame) -> tuple[dict[str, dict[int, LoadedDREAMPolicy]], pd.DataFrame]:
    policies_by_seed = defaultdict(dict)
    loaded_rows = []
    for row in checkpoint_df.itertuples(index=False):
        pol = LoadedDREAMPolicy(game, row.path, map_location="cpu")
        policies_by_seed[str(row.seed)][int(row.iteration)] = pol
        loaded_rows.append(
            {
                "seed": str(row.seed),
                "checkpoint": int(row.iteration),
                "internal_checkpoint_iteration": pol.iteration_inside_checkpoint,
                "kind": row.kind,
                "path": str(row.path),
            }
        )
    loaded_policy_df = pd.DataFrame(loaded_rows).sort_values(["seed", "checkpoint"])
    return dict(policies_by_seed), loaded_policy_df


def checkpoint_exploitability_metrics(game, policies_by_seed: Dict[str, Dict[int, LoadedDREAMPolicy]]) -> pd.DataFrame:
    metric_rows = []
    for seed, policies in policies_by_seed.items():
        for checkpoint, pol in sorted(policies.items()):
            tab_policy = osp_policy.tabular_policy_from_callable(game, pol.action_probabilities)
            nash_conv = float(exploitability.nash_conv(game, tab_policy))
            value = float(
                expected_game_score.policy_value(game.new_initial_state(), [tab_policy] * game.num_players())[0]
            )
            average_policy_value = value
            metric_rows.append(
                {
                    "seed": str(seed),
                    "checkpoint": int(checkpoint),
                    "nash_conv": nash_conv,
                    "exploitability": nash_conv / 2.0,
                    "policy_value_player_0": value,
                    "average_policy_value": average_policy_value,
                    "policy_value_error_from_known_value": abs(value - LEDUC_GAME_VALUE_P0),
                    "average_policy_value_error_from_target": abs(
                        average_policy_value - LEDUC_AVERAGE_POLICY_VALUE_TARGET
                    ),
                }
            )
    return pd.DataFrame(metric_rows).sort_values(["seed", "checkpoint"])


def exact_expected_value_two_player(game, pol0, pol1) -> tuple[float, float]:
    vals = expected_game_score.policy_value(game.new_initial_state(), [pol0, pol1])
    return float(vals[0]), float(vals[1])


def exact_seat_averaged_value_for_a(game, agent_a_policy, agent_b_policy) -> Dict[str, float]:
    a_as_p0 = exact_expected_value_two_player(game, agent_a_policy, agent_b_policy)[0]
    a_as_p1 = exact_expected_value_two_player(game, agent_b_policy, agent_a_policy)[1]
    return {
        "A_EV_as_player_0": float(a_as_p0),
        "A_EV_as_player_1": float(a_as_p1),
        "A_EV_seat_averaged": float((a_as_p0 + a_as_p1) / 2.0),
    }


def exact_pairwise_head_to_head(
    game,
    policies_by_seed: Dict[str, Dict[int, LoadedDREAMPolicy]],
    equivalence_epsilon: float = 1e-4,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    exact_records = []
    exact_matrices = {}
    for seed, policies in policies_by_seed.items():
        checkpoints = sorted(policies)
        matrix = pd.DataFrame(index=checkpoints, columns=checkpoints, dtype=float)
        for i in checkpoints:
            for j in checkpoints:
                if i == j:
                    result = {
                        "A_EV_as_player_0": 0.0,
                        "A_EV_as_player_1": 0.0,
                        "A_EV_seat_averaged": 0.0,
                    }
                else:
                    result = exact_seat_averaged_value_for_a(game, policies[i], policies[j])
                matrix.loc[i, j] = result["A_EV_seat_averaged"]
                exact_records.append(
                    {
                        "seed": str(seed),
                        "checkpoint_A": int(i),
                        "checkpoint_B": int(j),
                        **result,
                    }
                )
        exact_matrices[str(seed)] = matrix

    exact_df = pd.DataFrame(exact_records)
    all_checkpoints = sorted({c for m in exact_matrices.values() for c in m.index})
    stack = np.stack(
        [
            m.reindex(index=all_checkpoints, columns=all_checkpoints).to_numpy(dtype=float)
            for m in exact_matrices.values()
        ]
    )
    mean_matrix = pd.DataFrame(np.nanmean(stack, axis=0), index=all_checkpoints, columns=all_checkpoints)
    win_fraction_matrix = pd.DataFrame(
        np.nanmean(stack > equivalence_epsilon, axis=0),
        index=all_checkpoints,
        columns=all_checkpoints,
    )
    return exact_df, exact_matrices, mean_matrix, win_fraction_matrix


def classify_ev(ev: float, eps: float = 1e-4) -> str:
    if ev > eps:
        return "clear_win"
    if ev < -eps:
        return "clear_loss"
    return "tie"


def monotonicity_summary_for_seed(seed: str, matrix: pd.DataFrame, equivalence_epsilon: float = 1e-4) -> Dict:
    checkpoints = list(matrix.index)
    later_earlier_evs = []
    adjacent_evs = []
    for a_idx, later in enumerate(checkpoints):
        for earlier in checkpoints[:a_idx]:
            later_earlier_evs.append(float(matrix.loc[later, earlier]))
        if a_idx > 0:
            adjacent_evs.append(float(matrix.loc[later, checkpoints[a_idx - 1]]))
    later_earlier_evs = np.asarray(later_earlier_evs, dtype=float)
    adjacent_evs = np.asarray(adjacent_evs, dtype=float)
    classes = [classify_ev(x, equivalence_epsilon) for x in later_earlier_evs]
    adj_classes = [classify_ev(x, equivalence_epsilon) for x in adjacent_evs]
    return {
        "seed": str(seed),
        "num_later_vs_earlier_pairs": int(len(later_earlier_evs)),
        "all_pairs_clear_improvement_rate": float(np.mean([c == "clear_win" for c in classes]))
        if classes
        else np.nan,
        "all_pairs_tie_rate": float(np.mean([c == "tie" for c in classes])) if classes else np.nan,
        "all_pairs_clear_regression_rate": float(np.mean([c == "clear_loss" for c in classes]))
        if classes
        else np.nan,
        "adjacent_clear_improvement_rate": float(np.mean([c == "clear_win" for c in adj_classes]))
        if adj_classes
        else np.nan,
        "adjacent_tie_rate": float(np.mean([c == "tie" for c in adj_classes])) if adj_classes else np.nan,
        "adjacent_clear_regression_rate": float(np.mean([c == "clear_loss" for c in adj_classes]))
        if adj_classes
        else np.nan,
        "mean_later_vs_earlier_ev": float(np.mean(later_earlier_evs)) if len(later_earlier_evs) else np.nan,
        "worst_monotonicity_violation": float(np.min(later_earlier_evs)) if len(later_earlier_evs) else np.nan,
        "best_later_vs_earlier_gain": float(np.max(later_earlier_evs)) if len(later_earlier_evs) else np.nan,
    }


def checkpoint_strength_summaries(
    exact_matrices: Dict[str, pd.DataFrame],
    checkpoint_metrics_df: pd.DataFrame,
    equivalence_epsilon: float = 1e-4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monotonicity_df = pd.DataFrame(
        [
            monotonicity_summary_for_seed(seed, matrix, equivalence_epsilon)
            for seed, matrix in exact_matrices.items()
        ]
    )

    strength_rows = []
    for seed, matrix in exact_matrices.items():
        checkpoints = list(matrix.index)
        for checkpoint in checkpoints:
            earlier = [c for c in checkpoints if int(c) < int(checkpoint)]
            previous = max(earlier) if earlier else None
            strength_rows.append(
                {
                    "seed": str(seed),
                    "checkpoint": int(checkpoint),
                    "mean_EV_vs_all_other_checkpoints": float(matrix.loc[checkpoint].drop(index=checkpoint).mean())
                    if len(checkpoints) > 1
                    else np.nan,
                    "mean_EV_vs_earlier": float(matrix.loc[checkpoint, earlier].mean()) if earlier else np.nan,
                    "EV_vs_previous": float(matrix.loc[checkpoint, previous]) if previous is not None else np.nan,
                }
            )
    strength_df = pd.DataFrame(strength_rows)
    strength_with_metrics_df = strength_df.merge(
        checkpoint_metrics_df,
        on=["seed", "checkpoint"],
        how="left",
    )

    agg_cols = [
        "mean_EV_vs_all_other_checkpoints",
        "mean_EV_vs_earlier",
        "EV_vs_previous",
        "exploitability",
        "average_policy_value",
        "policy_value_error_from_known_value",
    ]
    aggregate_strength_df = strength_with_metrics_df.groupby("checkpoint")[agg_cols].agg(["mean", "sem"]).reset_index()
    aggregate_strength_df.columns = [
        "_".join([str(x) for x in col if str(x)])
        for col in aggregate_strength_df.columns.to_flat_index()
    ]
    aggregate_strength_df = aggregate_strength_df.rename(columns={"checkpoint_": "checkpoint"})

    best_rows = []
    for seed, group in strength_with_metrics_df.groupby("seed"):
        best_head = group.loc[group["mean_EV_vs_all_other_checkpoints"].idxmax()]
        best_expl = group.loc[group["exploitability"].idxmin()]
        final = group.loc[group["checkpoint"].idxmax()]
        best_rows.append(
            {
                "seed": seed,
                "best_checkpoint_by_head_to_head_EV": int(best_head["checkpoint"]),
                "best_mean_EV_vs_all_other_checkpoints": float(best_head["mean_EV_vs_all_other_checkpoints"]),
                "best_checkpoint_by_exploitability": int(best_expl["checkpoint"]),
                "lowest_exploitability": float(best_expl["exploitability"]),
                "final_checkpoint": int(final["checkpoint"]),
                "final_exploitability": float(final["exploitability"]),
            }
        )
    best_checkpoint_df = pd.DataFrame(best_rows)
    return monotonicity_df, strength_with_metrics_df, aggregate_strength_df, best_checkpoint_df


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    *,
    cmap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fmt: str = ".3f",
    colorbar_label: str = "Seat-averaged EV for checkpoint A",
    title_config: Optional[Dict] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    data = matrix.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([str(x) for x in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([str(x) for x in matrix.index])
    ax.set_xlabel("Checkpoint B")
    ax.set_ylabel("Checkpoint A")
    ax.set_title(format_plot_title(title, title_config))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_errorbar(
    df: pd.DataFrame,
    y_mean_col: str,
    y_sem_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    zero_line: bool = True,
    average_policy_value_target: float = LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    title_config: Optional[Dict] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = df["checkpoint"].to_numpy(dtype=float)
    y = df[y_mean_col].to_numpy(dtype=float)
    yerr = df[y_sem_col].fillna(0.0).to_numpy(dtype=float)
    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
    if is_exploitability_metric(y_mean_col):
        add_nash_exploitability_target(ax)
        ax.legend()
    elif is_average_policy_value_metric(y_mean_col.replace("_mean", "")):
        add_average_policy_value_target(ax, target=average_policy_value_target)
        ax.legend()
    elif zero_line:
        ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Checkpoint iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(format_plot_title(title, title_config))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
