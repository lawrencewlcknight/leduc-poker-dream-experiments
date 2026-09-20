"""Frozen-reservoir export, grouping, policy fitting, and exact evaluation."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch

try:
    from open_spiel.python import policy
    from open_spiel.python.algorithms import expected_game_score, exploitability
except Exception:  # pragma: no cover - guarded by the experiment runner
    policy = None
    expected_game_score = None
    exploitability = None

from dream_poker.networks import build_network
from dream_poker.seeding import set_seed

from .config import GROUPED_CE, GROUPED_CE_EXTENDED, ROW_CE, ROW_MSE


@dataclass(frozen=True)
class FrozenReservoir:
    info_states: np.ndarray
    iterations: np.ndarray
    strategies: np.ndarray
    reach_weights: np.ndarray
    legal_masks: np.ndarray
    capacity: int
    observations_seen: int

    @property
    def size(self) -> int:
        return int(len(self.iterations))


@dataclass(frozen=True)
class GroupedReservoir:
    info_states: np.ndarray
    strategies: np.ndarray
    legal_masks: np.ndarray
    objective_masses: np.ndarray
    objective_weights: np.ndarray
    source_rows: int

    @property
    def size(self) -> int:
        return int(len(self.objective_masses))


def _key(info_state: np.ndarray) -> bytes:
    return np.asarray(info_state, dtype=np.float32).tobytes()


def enumerate_legal_masks(game) -> Dict[bytes, np.ndarray]:
    """Enumerate the legal-action mask associated with every Leduc information state."""
    masks: Dict[bytes, np.ndarray] = {}
    stack = [game.new_initial_state()]
    num_actions = int(game.num_distinct_actions())
    while stack:
        state = stack.pop()
        if state.is_terminal():
            continue
        if state.is_chance_node():
            stack.extend(state.child(int(action)) for action, _ in state.chance_outcomes())
            continue
        player_id = int(state.current_player())
        info_state = np.asarray(state.information_state_tensor(player_id), dtype=np.float32)
        mask = np.zeros(num_actions, dtype=np.bool_)
        legal_actions = list(map(int, state.legal_actions(player_id)))
        mask[legal_actions] = True
        key = _key(info_state)
        previous = masks.get(key)
        if previous is not None and not np.array_equal(previous, mask):
            raise ValueError("Legal actions are inconsistent within an information state")
        masks[key] = mask
        stack.extend(state.child(action) for action in legal_actions)
    return masks


def freeze_strategy_reservoir(solver, game) -> FrozenReservoir:
    data = list(solver.strategy_memory._data)
    if not data:
        raise ValueError("DREAM strategy reservoir is empty")
    legal_lookup = enumerate_legal_masks(game)
    info_states = np.asarray([row.info_state for row in data], dtype=np.float32)
    iterations = np.asarray([row.iteration for row in data], dtype=np.int32)
    strategies = np.asarray([row.strategy_action_probs for row in data], dtype=np.float32)
    reach_weights = np.asarray([row.weight for row in data], dtype=np.float32)
    legal_masks = np.asarray([legal_lookup[_key(row)] for row in info_states], dtype=np.bool_)
    return FrozenReservoir(
        info_states=info_states,
        iterations=iterations,
        strategies=strategies,
        reach_weights=reach_weights,
        legal_masks=legal_masks,
        capacity=int(solver._strategy_memory_capacity),
        observations_seen=int(solver.strategy_memory.add_calls),
    )


def save_frozen_reservoir(path: Path, reservoir: FrozenReservoir) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        info_states=reservoir.info_states,
        iterations=reservoir.iterations,
        strategies=reservoir.strategies,
        reach_weights=reservoir.reach_weights,
        legal_masks=reservoir.legal_masks,
        capacity=np.asarray(reservoir.capacity, dtype=np.int64),
        observations_seen=np.asarray(reservoir.observations_seen, dtype=np.int64),
    )


def load_frozen_reservoir(path: Path) -> FrozenReservoir:
    with np.load(path, allow_pickle=False) as data:
        return FrozenReservoir(
            info_states=np.asarray(data["info_states"], dtype=np.float32),
            iterations=np.asarray(data["iterations"], dtype=np.int32),
            strategies=np.asarray(data["strategies"], dtype=np.float32),
            reach_weights=np.asarray(data["reach_weights"], dtype=np.float32),
            legal_masks=np.asarray(data["legal_masks"], dtype=np.bool_),
            capacity=int(data["capacity"]),
            observations_seen=int(data["observations_seen"]),
        )


def row_objective_weights(reservoir: FrozenReservoir) -> np.ndarray:
    # This exactly matches DREAMSolver._average_strategy_loss_multiplier squared.
    reach = np.clip(reservoir.reach_weights.astype(np.float64), 0.0, 100.0)
    reach = np.maximum(reach, 1e-8)
    iteration = np.maximum(reservoir.iterations.astype(np.float64), 1.0)
    return iteration * reach


def group_reservoir(reservoir: FrozenReservoir) -> GroupedReservoir:
    weights = row_objective_weights(reservoir)
    unique, inverse, counts = np.unique(
        reservoir.info_states, axis=0, return_inverse=True, return_counts=True
    )
    masses = np.zeros(len(unique), dtype=np.float64)
    numerators = np.zeros((len(unique), reservoir.strategies.shape[1]), dtype=np.float64)
    np.add.at(masses, inverse, weights)
    np.add.at(numerators, inverse, reservoir.strategies.astype(np.float64) * weights[:, None])
    if np.any(masses <= 0.0):
        raise ValueError("Grouped reservoir contains non-positive objective mass")
    targets = numerators / masses[:, None]

    order = np.argsort(inverse, kind="stable")
    first = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
    grouped_masks = reservoir.legal_masks[order[first]]
    if not np.all(grouped_masks[inverse] == reservoir.legal_masks):
        raise ValueError("Legal-action masks differ within an information state")

    # mean(group_weight * loss) then equals mean(row_weight * loss).
    objective_weights = masses * (float(len(unique)) / float(reservoir.size))
    return GroupedReservoir(
        info_states=unique.astype(np.float32, copy=False),
        strategies=targets.astype(np.float32),
        legal_masks=grouped_masks.astype(np.bool_, copy=False),
        objective_masses=masses,
        objective_weights=objective_weights.astype(np.float32),
        source_rows=reservoir.size,
    )


def save_grouped_reservoir(path: Path, grouped: GroupedReservoir) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        info_states=grouped.info_states,
        strategies=grouped.strategies,
        legal_masks=grouped.legal_masks,
        objective_masses=grouped.objective_masses,
        objective_weights=grouped.objective_weights,
        source_rows=np.asarray(grouped.source_rows, dtype=np.int64),
    )


class EmpiricalReservoirPolicy:
    def __init__(self, grouped: GroupedReservoir):
        self._lookup = {
            _key(info): np.asarray(target, dtype=np.float64)
            for info, target in zip(grouped.info_states, grouped.strategies)
        }

    def action_probabilities(self, state, player_id=None):
        del player_id
        current_player = int(state.current_player())
        legal = list(map(int, state.legal_actions(current_player)))
        info = np.asarray(state.information_state_tensor(current_player), dtype=np.float32)
        probabilities = self._lookup.get(_key(info))
        if probabilities is None:
            return {action: 1.0 / len(legal) for action in legal}
        legal_values = np.asarray([max(float(probabilities[a]), 0.0) for a in legal])
        total = float(legal_values.sum())
        if total <= 0.0:
            legal_values[:] = 1.0 / len(legal)
        else:
            legal_values /= total
        return {action: float(value) for action, value in zip(legal, legal_values)}


class NetworkPolicy:
    def __init__(self, model):
        self.model = model

    def action_probabilities(self, state, player_id=None):
        del player_id
        current_player = int(state.current_player())
        legal = list(map(int, state.legal_actions(current_player)))
        info = np.asarray(state.information_state_tensor(current_player), dtype=np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(info[None, :]))[0].cpu().numpy()
        legal_logits = np.asarray([logits[action] for action in legal], dtype=np.float64)
        legal_logits -= float(legal_logits.max())
        probabilities = np.exp(legal_logits)
        probabilities /= float(probabilities.sum())
        return {action: float(value) for action, value in zip(legal, probabilities)}


def exact_policy_metrics(game, candidate) -> Dict[str, float]:
    if policy is None:
        raise RuntimeError("OpenSpiel is required for exact policy evaluation")
    tabular = policy.tabular_policy_from_callable(game, candidate.action_probabilities)
    value = float(
        expected_game_score.policy_value(game.new_initial_state(), [tabular] * 2)[0]
    )
    nash_conv = float(exploitability.nash_conv(game, tabular))
    return {
        "exploitability": nash_conv / 2.0,
        "nash_conv": nash_conv,
        "policy_value_player_0": value,
    }


def initial_policy_state(
    *, network_type: str, input_size: int, hidden_layers: Sequence[int], output_size: int, seed: int
) -> Mapping[str, torch.Tensor]:
    set_seed(int(seed))
    model = build_network(network_type, input_size, hidden_layers, output_size)
    return copy.deepcopy(model.state_dict())


def _masked_log_softmax(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.masked_fill(~masks, -1e20), dim=-1)


def fit_frozen_policy(
    *,
    arm: str,
    reservoir: FrozenReservoir,
    grouped: GroupedReservoir,
    network_type: str,
    hidden_layers: Sequence[int],
    initial_state: Mapping[str, torch.Tensor],
    learning_rate: float,
    batch_size: int,
    train_steps: int,
    sampling_seed: int,
    gradient_clip_norm: Optional[float] = None,
) -> tuple[torch.nn.Module, Dict[str, float]]:
    if arm not in {ROW_MSE, ROW_CE, GROUPED_CE, GROUPED_CE_EXTENDED}:
        raise ValueError(f"Unknown distillation arm: {arm}")
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if gradient_clip_norm is not None and float(gradient_clip_norm) <= 0.0:
        raise ValueError("gradient_clip_norm must be positive when provided")
    model = build_network(
        network_type,
        int(reservoir.info_states.shape[1]),
        hidden_layers,
        int(reservoir.strategies.shape[1]),
    )
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    rng = np.random.default_rng(int(sampling_seed))
    grouped_arm = arm in {GROUPED_CE, GROUPED_CE_EXTENDED}
    size = grouped.size if grouped_arm else reservoir.size
    effective_batch = min(int(batch_size), int(size))
    if effective_batch <= 0:
        raise ValueError("Cannot fit an empty reservoir")

    row_weights = row_objective_weights(reservoir).astype(np.float32)
    last_loss = float("nan")
    started = time.perf_counter()
    for step in range(int(train_steps)):
        if grouped_arm:
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
        else:
            indices = rng.integers(0, size, size=effective_batch, endpoint=False)
            x = torch.from_numpy(reservoir.info_states[indices])
            targets = torch.from_numpy(reservoir.strategies[indices])
            weights = torch.from_numpy(row_weights[indices])
            logits = model(x)
            if arm == ROW_MSE:
                multiplier = torch.sqrt(weights).reshape(-1, 1)
                probabilities = torch.softmax(logits, dim=-1)
                loss = torch.mean(torch.square(multiplier * probabilities - multiplier * targets))
            else:
                masks = torch.from_numpy(reservoir.legal_masks[indices])
                per_example = -(targets * _masked_log_softmax(logits, masks)).sum(dim=1)
                loss = torch.mean(per_example * weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {arm} loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())

    elapsed = time.perf_counter() - started
    return model, {
        "train_steps": int(train_steps),
        "batch_size": int(effective_batch),
        "examples_processed": int(train_steps) * int(effective_batch),
        "final_training_loss": last_loss,
        "distillation_seconds": float(elapsed),
    }


def save_policy_model(path: Path, model, metadata: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "dream_frozen_reservoir_distilled_policy",
            "policy_network_state_dict": model.state_dict(),
            **dict(metadata),
        },
        path,
    )
