"""Ray-based DREAM traversal collection with a single central learner.

This mirrors the ESCHER parallelization used elsewhere in this workspace: Ray
actors collect traversal experience with synchronized inference networks, while
one central learner performs all supervised updates. Traversal and replay
budgets are partitioned across workers rather than multiplied by worker count.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import numpy as np
import torch

import pyspiel

from dream_poker.parallel_utils import partition_total, worker_seed
from dream_poker.replay import CircularReplay, ReservoirBuffer
from dream_poker.seeding import set_seed
from dream_poker.solver import DREAMSolver


class DREAMExperienceWorker:
    """Ray actor payload for traversal-only DREAM experience generation."""

    def __init__(
        self,
        game_name: str,
        solver_kwargs: Dict[str, Any],
        worker_seed_value: int,
    ):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        set_seed(int(worker_seed_value))
        game = pyspiel.load_game(str(game_name))
        kwargs = dict(solver_kwargs)
        kwargs["seed"] = int(worker_seed_value)
        self._solver = DREAMSolver(game, **kwargs)

    def ping(self) -> bool:
        return True

    def collect(
        self,
        n: int,
        traverser: int,
        advantage_weights,
        baseline_weights,
        iteration: int,
    ) -> Dict[str, float]:
        self._solver.set_traversal_weights(advantage_weights, baseline_weights)
        self._solver.set_iteration(int(iteration))
        before = int(self._solver.nodes_touched)
        start = time.perf_counter()
        for _ in range(int(n)):
            state = self._solver.new_initial_state()
            self._solver.traverse_outcome_sampling(
                state,
                int(traverser),
                pi_reach=1.0,
                sigma_reach=1.0,
            )
        elapsed = time.perf_counter() - start
        return {
            "nodes_touched": int(self._solver.nodes_touched - before),
            "worker_collection_seconds": float(elapsed),
        }

    def replay_state(self) -> Dict:
        return {
            "advantage_memories": [
                buffer.state_dict() for buffer in self._solver.advantage_memories
            ],
            "strategy_memory": self._solver.strategy_memory.state_dict(),
            "baseline_replays": [
                buffer.state_dict() for buffer in self._solver.baseline_replays
            ],
        }


class ParallelDREAMSolver(DREAMSolver):
    """Single-learner DREAM with Ray-parallel traversal collection."""

    def __init__(
        self,
        game,
        *,
        game_name: str,
        parallel_num_workers: int = 3,
        parallel_run_seed: int = 0,
        parallel_ray_address: str | None = None,
        parallel_log_to_driver: bool = False,
        **solver_kwargs,
    ):
        self._parallel_num_workers = int(parallel_num_workers)
        if self._parallel_num_workers < 2:
            raise ValueError("parallel_num_workers must be at least 2.")

        for key in (
            "advantage_memory_capacity",
            "strategy_memory_capacity",
            "baseline_memory_capacity",
        ):
            if int(solver_kwargs[key]) < self._parallel_num_workers:
                raise ValueError(f"{key} must provide at least one slot per worker.")

        super().__init__(game, **solver_kwargs)
        self._game_name = str(game_name)
        self._parallel_run_seed = int(parallel_run_seed)
        self._workers = []
        self._ray = None
        self._owns_ray_runtime = False
        self._cumulative_worker_collection_seconds = 0.0

        try:
            import ray

            self._ray = ray
            self._owns_ray_runtime = not ray.is_initialized()
            if self._owns_ray_runtime:
                init_kwargs = {
                    "include_dashboard": False,
                    "log_to_driver": bool(parallel_log_to_driver),
                    "ignore_reinit_error": True,
                }
                if parallel_ray_address:
                    init_kwargs["address"] = str(parallel_ray_address)
                else:
                    init_kwargs["num_cpus"] = self._parallel_num_workers
                ray.init(**init_kwargs)

            worker_class = ray.remote(num_cpus=1)(DREAMExperienceWorker)
            advantage_capacities = partition_total(
                int(solver_kwargs["advantage_memory_capacity"]),
                self._parallel_num_workers,
            )
            strategy_capacities = partition_total(
                int(solver_kwargs["strategy_memory_capacity"]),
                self._parallel_num_workers,
            )
            baseline_capacities = partition_total(
                int(solver_kwargs["baseline_memory_capacity"]),
                self._parallel_num_workers,
            )
            for worker_index in range(self._parallel_num_workers):
                worker_kwargs = dict(solver_kwargs)
                worker_kwargs.update(
                    {
                        "advantage_memory_capacity": int(advantage_capacities[worker_index]),
                        "strategy_memory_capacity": int(strategy_capacities[worker_index]),
                        "baseline_memory_capacity": int(baseline_capacities[worker_index]),
                        "compute_exploitability": False,
                    }
                )
                seed = worker_seed(self._parallel_run_seed, worker_index)
                self._workers.append(
                    worker_class.remote(
                        self._game_name,
                        worker_kwargs,
                        seed,
                    )
                )
            ray.get([worker.ping.remote() for worker in self._workers])
            self._refresh_replay_from_workers()
        except Exception:
            self.close()
            raise

    @property
    def parallel_num_workers(self) -> int:
        return int(self._parallel_num_workers)

    def close(self) -> None:
        ray = getattr(self, "_ray", None)
        if ray is None:
            return
        for worker in getattr(self, "_workers", []):
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                pass
        self._workers = []
        if self._owns_ray_runtime and ray.is_initialized():
            ray.shutdown()
        self._ray = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _worker_results(self, method_name: str, *args):
        refs = [getattr(worker, method_name).remote(*args) for worker in self._workers]
        return self._ray.get(refs)

    def _traversal_weights(self) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        return {
            "advantage": [
                {name: value.detach().cpu() for name, value in net.state_dict().items()}
                for net in self._advantage_networks
            ],
            "baseline": [
                {name: value.detach().cpu() for name, value in net.state_dict().items()}
                for net in self._baseline_networks
            ],
        }

    def _collect_traversals_for_player(self, traverser: int) -> None:
        traversal_start = time.perf_counter()
        sync_start = time.perf_counter()
        weights = self._traversal_weights()
        advantage_weights = self._ray.put(weights["advantage"])
        baseline_weights = self._ray.put(weights["baseline"])
        self._cumulative_parallel_sync_seconds += time.perf_counter() - sync_start

        counts = partition_total(int(self._num_traversals), self._parallel_num_workers)
        refs = [
            worker.collect.remote(
                int(count),
                int(traverser),
                advantage_weights,
                baseline_weights,
                int(self._iteration),
            )
            for worker, count in zip(self._workers, counts)
            if count > 0
        ]
        results = self._ray.get(refs) if refs else []
        self._nodes_touched += sum(int(row["nodes_touched"]) for row in results)
        self._cumulative_worker_collection_seconds += sum(
            float(row.get("worker_collection_seconds", 0.0)) for row in results
        )
        self._cumulative_traversal_seconds += time.perf_counter() - traversal_start
        self._refresh_replay_from_workers()

    def _refresh_replay_from_workers(self) -> None:
        refresh_start = time.perf_counter()
        states = self._worker_results("replay_state")
        self._advantage_memories = [
            self._combine_reservoir_states(
                [state["advantage_memories"][player] for state in states],
                self._advantage_memory_capacity,
            )
            for player in range(self._num_players)
        ]
        self._strategy_memories = self._combine_reservoir_states(
            [state["strategy_memory"] for state in states],
            self._strategy_memory_capacity,
        )
        self._baseline_replays = [
            self._combine_circular_states(
                [state["baseline_replays"][player] for state in states],
                self._baseline_memory_capacity,
            )
            for player in range(self._num_players)
        ]
        self._cumulative_replay_refresh_seconds += time.perf_counter() - refresh_start

    @staticmethod
    def _combine_reservoir_states(states: List[Dict], capacity: int) -> ReservoirBuffer:
        combined = ReservoirBuffer(int(capacity))
        data = []
        add_calls = 0
        for state in states:
            data.extend(list(state.get("data", [])))
            add_calls += int(state.get("add_calls", len(state.get("data", []))))
        if len(data) > int(capacity):
            data = data[: int(capacity)]
        combined._data = data  # pylint: disable=protected-access
        combined._add_calls = int(add_calls)  # pylint: disable=protected-access
        return combined

    @staticmethod
    def _combine_circular_states(states: List[Dict], capacity: int) -> CircularReplay:
        combined = CircularReplay(int(capacity))
        tensor_chunks = []
        legacy_data = []
        for state in states:
            if state.get("tensorized_baseline", False) or "info_states" in state:
                size = int(state.get("size", len(state.get("actions", []))))
                if size <= 0:
                    continue
                tensor_chunks.append(
                    {
                        "info_states": np.asarray(state["info_states"], dtype=np.float32)[:size],
                        "actions": np.asarray(state["actions"], dtype=np.int64)[:size],
                        "rewards": np.asarray(state["rewards"], dtype=np.float32)[:size],
                        "next_info_states": np.asarray(
                            state["next_info_states"],
                            dtype=np.float32,
                        )[:size],
                        "next_legal_action_masks": np.asarray(
                            state["next_legal_action_masks"],
                            dtype=bool,
                        )[:size],
                        "next_players": np.asarray(state["next_players"], dtype=np.int64)[:size],
                        "dones": np.asarray(state["dones"], dtype=bool)[:size],
                        "info_state_size": int(state["info_state_size"]),
                        "num_actions": int(state["num_actions"]),
                    }
                )
            else:
                legacy_data.extend(list(state.get("data", [])))

        if tensor_chunks:
            merged = {
                key: np.concatenate([chunk[key] for chunk in tensor_chunks], axis=0)
                for key in (
                    "info_states",
                    "actions",
                    "rewards",
                    "next_info_states",
                    "next_legal_action_masks",
                    "next_players",
                    "dones",
                )
            }
            if len(merged["actions"]) > int(capacity):
                keep = slice(len(merged["actions"]) - int(capacity), len(merged["actions"]))
                merged = {key: value[keep] for key, value in merged.items()}
            combined.load_state_dict(
                {
                    "capacity": int(capacity),
                    "idx": 0,
                    "tensorized_baseline": True,
                    "size": int(len(merged["actions"])),
                    "info_state_size": int(tensor_chunks[0]["info_state_size"]),
                    "num_actions": int(tensor_chunks[0]["num_actions"]),
                    **merged,
                }
            )
            for element in legacy_data:
                combined.add(element)
            return combined

        if len(legacy_data) > int(capacity):
            legacy_data = legacy_data[-int(capacity) :]
        combined.load_state_dict(
            {
                "capacity": int(capacity),
                "data": list(legacy_data),
                "idx": 0,
            }
        )
        return combined

    def _checkpoint_metrics(self, start_time: float) -> Dict:
        self._refresh_replay_from_workers()
        row = super()._checkpoint_metrics(start_time)
        row["parallel_num_workers"] = int(self._parallel_num_workers)
        row["cumulative_worker_collection_seconds"] = float(
            self._cumulative_worker_collection_seconds
        )
        return row

    def full_checkpoint_state(self) -> Dict:
        raise NotImplementedError(
            "Parallel replay shards are not yet supported by full DREAM checkpoints."
        )
