
"""Replay memory utilities used by the DREAM-style solver."""

from __future__ import annotations

import collections
import random
from typing import List


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
    """Bounded circular replay buffer for learned baseline targets."""

    def __init__(self, capacity: int):
        self._capacity = int(capacity)
        self._data: List = []
        self._idx = 0

    def __len__(self) -> int:
        return len(self._data)

    def add(self, element) -> None:
        if len(self._data) < self._capacity:
            self._data.append(element)
        else:
            self._data[self._idx] = element
            self._idx = (self._idx + 1) % self._capacity

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
            "idx": int(self._idx),
        }

    def load_state_dict(self, state: dict) -> None:
        self._data = list(state["data"])
        self._idx = int(state.get("idx", 0))
