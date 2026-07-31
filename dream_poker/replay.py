
"""Replay memory utilities used by the DREAM-style solver."""

from __future__ import annotations

import collections
import random
from typing import List, Optional

import numpy as np


DreamAdvantageMemory = collections.namedtuple(
    "DreamAdvantageMemory", "info_state iteration advantage weight"
)
DreamStrategyMemory = collections.namedtuple(
    "DreamStrategyMemory", "info_state iteration strategy_action_probs weight"
)
BaselineTransition = collections.namedtuple(
    "BaselineTransition",
    "info_state action reward next_info_state next_legal_actions next_player done",
)
BaselineReplayBatch = collections.namedtuple(
    "BaselineReplayBatch",
    "info_states actions rewards next_info_states next_legal_action_masks next_players dones",
)


class ReservoirBuffer:
    """Reservoir sampling buffer with bounded memory."""

    def __init__(self, capacity: int):
        self._capacity = int(capacity)
        self._data: List = []
        self._add_calls = 0

    def __len__(self) -> int:
        return len(self._data)

    @property
    def add_calls(self) -> int:
        return int(self._add_calls)

    def add(self, element) -> None:
        self._add_calls += 1
        if len(self._data) < self._capacity:
            self._data.append(element)
        else:
            idx = random.randrange(self._add_calls)
            if idx < self._capacity:
                self._data[idx] = element

    def sample(self, batch_size: int):
        if len(self._data) == 0:
            return []
        return random.choices(self._data, k=int(batch_size))

    def sample_up_to(self, max_size: int):
        if len(self._data) <= max_size:
            return list(self._data)
        return random.sample(self._data, int(max_size))

    def state_dict(self) -> dict:
        return {
            "capacity": int(self._capacity),
            "data": list(self._data),
            "add_calls": int(self._add_calls),
        }

    def load_state_dict(self, state: dict) -> None:
        self._data = list(state["data"])
        self._add_calls = int(state["add_calls"])


class CircularReplay:
    """Bounded circular replay buffer for learned baseline targets.

    Baseline transitions are stored in typed NumPy arrays and exposed through
    ``sample_baseline_batch`` so the baseline learner can operate on whole
    minibatches. The older Python-object ``data`` checkpoint format is still
    accepted by ``load_state_dict``.
    """

    def __init__(
        self,
        capacity: int,
        info_state_size: Optional[int] = None,
        num_actions: Optional[int] = None,
    ):
        self._capacity = int(capacity)
        if self._capacity <= 0:
            raise ValueError("capacity must be positive")
        self._data: List = []
        self._idx = 0
        self._size = 0
        self._info_state_size = int(info_state_size) if info_state_size is not None else None
        self._num_actions = int(num_actions) if num_actions is not None else None
        self._info_states = None
        self._actions = None
        self._rewards = None
        self._next_info_states = None
        self._next_legal_action_masks = None
        self._next_players = None
        self._dones = None
        if self._info_state_size is not None and self._num_actions is not None:
            self._init_baseline_arrays(self._info_state_size, self._num_actions)

    def __len__(self) -> int:
        if self._has_baseline_arrays:
            return int(self._size)
        return len(self._data)

    def add(self, element) -> None:
        if isinstance(element, BaselineTransition):
            self._ensure_baseline_arrays(element)
            if self._size < self._capacity:
                idx = self._size
                self._size += 1
            else:
                idx = self._idx
                self._idx = (self._idx + 1) % self._capacity
            self._write_baseline_transition(idx, element)
            return

        if len(self._data) < self._capacity:
            self._data.append(element)
        else:
            self._data[self._idx] = element
            self._idx = (self._idx + 1) % self._capacity

    def sample(self, batch_size: int):
        if len(self) == 0:
            return []
        if self._has_baseline_arrays:
            indices = random.choices(range(self._size), k=int(batch_size))
            return [self._baseline_transition_at(int(idx)) for idx in indices]
        return random.choices(self._data, k=int(batch_size))

    def sample_up_to(self, max_size: int):
        if len(self) == 0:
            return []
        if self._has_baseline_arrays:
            if len(self) <= int(max_size):
                indices = range(self._size)
            else:
                indices = random.sample(range(self._size), int(max_size))
            return [self._baseline_transition_at(int(idx)) for idx in indices]
        if len(self._data) <= max_size:
            return list(self._data)
        return random.sample(self._data, int(max_size))

    def sample_baseline_batch(self, batch_size: int):
        """Sample a tensor-ready baseline minibatch using the Python RNG."""
        if len(self) == 0:
            return None
        if not self._has_baseline_arrays:
            self._tensorize_existing_baseline_data()
        indices = np.asarray(
            random.choices(range(self._size), k=int(batch_size)),
            dtype=np.int64,
        )
        return self._baseline_batch_from_indices(indices)

    def baseline_rewards_sample_up_to(self, max_size: int) -> np.ndarray:
        """Return sampled baseline rewards without reconstructing transitions."""
        if len(self) == 0:
            return np.asarray([], dtype=np.float32)
        if self._has_baseline_arrays:
            if len(self) <= int(max_size):
                indices = np.arange(self._size, dtype=np.int64)
            else:
                indices = np.asarray(
                    random.sample(range(self._size), int(max_size)),
                    dtype=np.int64,
                )
            return self._rewards[indices].astype(np.float32, copy=True)
        return np.asarray(
            [float(tr.reward) for tr in self.sample_up_to(max_size)],
            dtype=np.float32,
        )

    def state_dict(self) -> dict:
        if self._has_baseline_arrays:
            size = int(self._size)
            return {
                "capacity": int(self._capacity),
                "idx": int(self._idx),
                "tensorized_baseline": True,
                "size": size,
                "info_state_size": int(self._info_state_size),
                "num_actions": int(self._num_actions),
                "info_states": self._info_states[:size].copy(),
                "actions": self._actions[:size].copy(),
                "rewards": self._rewards[:size].copy(),
                "next_info_states": self._next_info_states[:size].copy(),
                "next_legal_action_masks": self._next_legal_action_masks[:size].copy(),
                "next_players": self._next_players[:size].copy(),
                "dones": self._dones[:size].copy(),
            }
        return {
            "capacity": int(self._capacity),
            "data": list(self._data),
            "idx": int(self._idx),
        }

    def load_state_dict(self, state: dict) -> None:
        self._capacity = int(state.get("capacity", self._capacity))
        if state.get("tensorized_baseline", False) or "info_states" in state:
            info_states = np.asarray(state["info_states"], dtype=np.float32)
            next_info_states = np.asarray(state["next_info_states"], dtype=np.float32)
            next_legal_action_masks = np.asarray(state["next_legal_action_masks"], dtype=bool)
            size = int(state.get("size", len(info_states)))
            self._data = []
            self._idx = int(state.get("idx", 0))
            self._size = size
            self._info_state_size = int(
                state.get(
                    "info_state_size",
                    info_states.shape[1] if info_states.ndim == 2 else self._info_state_size,
                )
            )
            self._num_actions = int(
                state.get(
                    "num_actions",
                    next_legal_action_masks.shape[1]
                    if next_legal_action_masks.ndim == 2
                    else self._num_actions,
                )
            )
            self._init_baseline_arrays(self._info_state_size, self._num_actions)
            self._info_states[:size] = info_states[:size]
            self._actions[:size] = np.asarray(state["actions"], dtype=np.int64)[:size]
            self._rewards[:size] = np.asarray(state["rewards"], dtype=np.float32)[:size]
            self._next_info_states[:size] = next_info_states[:size]
            self._next_legal_action_masks[:size] = next_legal_action_masks[:size]
            self._next_players[:size] = np.asarray(state["next_players"], dtype=np.int64)[:size]
            self._dones[:size] = np.asarray(state["dones"], dtype=bool)[:size]
            return

        self._data = list(state["data"])
        self._idx = int(state.get("idx", 0))
        self._size = len(self._data)
        self._clear_baseline_arrays()
        if self._data and all(isinstance(element, BaselineTransition) for element in self._data):
            self._tensorize_existing_baseline_data()

    @property
    def _has_baseline_arrays(self) -> bool:
        return self._info_states is not None

    def _clear_baseline_arrays(self) -> None:
        self._info_states = None
        self._actions = None
        self._rewards = None
        self._next_info_states = None
        self._next_legal_action_masks = None
        self._next_players = None
        self._dones = None

    def _init_baseline_arrays(self, info_state_size: int, num_actions: int) -> None:
        self._info_state_size = int(info_state_size)
        self._num_actions = int(num_actions)
        self._info_states = np.zeros((self._capacity, self._info_state_size), dtype=np.float32)
        self._actions = np.zeros(self._capacity, dtype=np.int64)
        self._rewards = np.zeros(self._capacity, dtype=np.float32)
        self._next_info_states = np.zeros((self._capacity, self._info_state_size), dtype=np.float32)
        self._next_legal_action_masks = np.zeros((self._capacity, self._num_actions), dtype=bool)
        self._next_players = np.full(self._capacity, -1, dtype=np.int64)
        self._dones = np.ones(self._capacity, dtype=bool)

    def _ensure_baseline_arrays(self, element: BaselineTransition) -> None:
        if self._has_baseline_arrays:
            return
        info_state = np.asarray(element.info_state, dtype=np.float32)
        if self._info_state_size is None:
            self._info_state_size = int(info_state.shape[0])
        if self._num_actions is None:
            legal_actions = list(element.next_legal_actions)
            max_action = max([int(element.action), *map(int, legal_actions)], default=-1)
            self._num_actions = max_action + 1
        self._init_baseline_arrays(self._info_state_size, self._num_actions)
        if self._data:
            self._tensorize_existing_baseline_data()

    def _write_baseline_transition(self, idx: int, element: BaselineTransition) -> None:
        info_state = np.asarray(element.info_state, dtype=np.float32)
        next_info_state = np.asarray(element.next_info_state, dtype=np.float32)
        if info_state.shape[0] != self._info_state_size:
            raise ValueError("Baseline info_state has incompatible shape.")
        if next_info_state.shape[0] != self._info_state_size:
            raise ValueError("Baseline next_info_state has incompatible shape.")
        action = int(element.action)
        if action < 0 or action >= self._num_actions:
            raise ValueError("Baseline action is outside the configured action range.")
        self._info_states[idx] = info_state
        self._actions[idx] = action
        self._rewards[idx] = float(element.reward)
        self._next_info_states[idx] = next_info_state
        self._next_legal_action_masks[idx] = False
        for legal_action in element.next_legal_actions:
            legal_action = int(legal_action)
            if legal_action < 0 or legal_action >= self._num_actions:
                raise ValueError(
                    "Baseline next legal action is outside the configured action range."
                )
            self._next_legal_action_masks[idx, legal_action] = True
        self._next_players[idx] = int(element.next_player)
        self._dones[idx] = bool(element.done)

    def _baseline_transition_at(self, idx: int) -> BaselineTransition:
        legal_actions = np.flatnonzero(self._next_legal_action_masks[idx]).astype(int).tolist()
        return BaselineTransition(
            self._info_states[idx].astype(np.float32, copy=True),
            int(self._actions[idx]),
            float(self._rewards[idx]),
            self._next_info_states[idx].astype(np.float32, copy=True),
            legal_actions,
            int(self._next_players[idx]),
            bool(self._dones[idx]),
        )

    def _baseline_batch_from_indices(self, indices: np.ndarray) -> BaselineReplayBatch:
        return BaselineReplayBatch(
            self._info_states[indices].astype(np.float32, copy=True),
            self._actions[indices].astype(np.int64, copy=True),
            self._rewards[indices].astype(np.float32, copy=True),
            self._next_info_states[indices].astype(np.float32, copy=True),
            self._next_legal_action_masks[indices].astype(bool, copy=True),
            self._next_players[indices].astype(np.int64, copy=True),
            self._dones[indices].astype(bool, copy=True),
        )

    def _tensorize_existing_baseline_data(self) -> None:
        data = list(self._data)
        if not data:
            return
        if not all(isinstance(element, BaselineTransition) for element in data):
            raise TypeError("CircularReplay contains non-baseline data and cannot be tensorized.")
        if self._info_state_size is None:
            self._info_state_size = int(np.asarray(data[0].info_state, dtype=np.float32).shape[0])
        if self._num_actions is None:
            max_action = -1
            for element in data:
                legal_actions = list(element.next_legal_actions)
                max_action = max(max_action, int(element.action), *map(int, legal_actions))
            self._num_actions = max_action + 1
        self._init_baseline_arrays(self._info_state_size, self._num_actions)
        self._size = len(data)
        for idx, element in enumerate(data):
            self._write_baseline_transition(idx, element)
        self._data = []
