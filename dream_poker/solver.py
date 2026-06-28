
"""DREAM-style OpenSpiel solver for Leduc poker experiments.

This is an OpenSpiel/PyTorch adaptation intended for thesis experiments. It is not a
bit-for-bit port of the original PokerRL DREAM repository.
"""

from __future__ import annotations

import time
import random
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    import pyspiel
    from open_spiel.python import policy
    from open_spiel.python.algorithms import expected_game_score
    from open_spiel.python.algorithms import exploitability
except Exception:  # pragma: no cover - OpenSpiel may not be installed in all environments.
    pyspiel = None
    policy = None
    expected_game_score = None
    exploitability = None

from dream_poker.constants import LEDUC_AVERAGE_POLICY_VALUE_TARGET, LEDUC_GAME_VALUE_P0
from dream_poker.experiment_utils import grad_norm, safe_mean, safe_std
from dream_poker.networks import MLP
from dream_poker.replay import (
    BaselineTransition,
    CircularReplay,
    DreamAdvantageMemory,
    DreamStrategyMemory,
    ReservoirBuffer,
)
from dream_poker.seeding import set_seed


# -----------------------------
# DREAM-style OpenSpiel solver
# -----------------------------

class DREAMSolver(policy.Policy if policy is not None else object):
    """OpenSpiel/PyTorch DREAM-style solver for sequential imperfect-information games.

    This is a thesis-oriented implementation designed to make DREAM comparable with the
    Deep CFR and ESCHER notebooks used in this project. It implements outcome-sampling
    regret updates with learned advantage baselines and a learned average-policy network.

    Important caveat: the official DREAM codebase is implemented in PokerRL and supports
    Single-style averaging. This OpenSpiel adaptation instead exposes an average-policy
    network so that exact exploitability can be computed with the same OpenSpiel tooling
    used elsewhere in the thesis.
    """

    def __init__(
        self,
        game,
        policy_network_layers: Sequence[int] = (32, 32),
        advantage_network_layers: Sequence[int] = (32, 32),
        baseline_network_layers: Optional[Sequence[int]] = None,
        num_iterations: int = 500,
        num_traversals: int = 320,
        learning_rate: float = 3e-3,
        learning_rate_schedule: str = "constant",
        learning_rate_end: Optional[float] = None,
        batch_size_advantage: int = 1024,
        batch_size_strategy: int = 1024,
        batch_size_baseline: Optional[int] = None,
        advantage_memory_capacity: int = int(1e6),
        strategy_memory_capacity: int = int(1e6),
        baseline_memory_capacity: int = int(1e6),
        epsilon: float = 0.06,
        advantage_network_train_steps: int = 100,
        policy_network_train_steps: int = 200,
        baseline_network_train_steps: int = 100,
        policy_network_train_every: int = 25,
        compute_exploitability: bool = True,
        game_value_player_0: Optional[float] = None,
        average_policy_value_target: Optional[float] = None,
        target_processing: str = "none",
        target_clip_value: float = 1.0,
        target_standardize_epsilon: float = 1e-6,
        seed: Optional[int] = None,
    ):
        if policy is not None:
            super().__init__(game, list(range(game.num_players())))
        if pyspiel is not None and game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
            raise ValueError("DREAMSolver only supports sequential games in this notebook.")

        self._game = game
        self._num_players = int(game.num_players())
        self._num_actions = int(game.num_distinct_actions())
        self._info_state_size = int(game.information_state_tensor_shape()[0])

        self._policy_network_layers = tuple(policy_network_layers)
        self._advantage_network_layers = tuple(advantage_network_layers)
        self._baseline_network_layers = tuple(baseline_network_layers or advantage_network_layers)

        self._num_iterations = int(num_iterations)
        self._num_traversals = int(num_traversals)
        self._learning_rate = float(learning_rate)
        self._learning_rate_schedule = str(learning_rate_schedule)
        self._learning_rate_end = (
            float(learning_rate_end) if learning_rate_end is not None else float(learning_rate)
        )
        self._current_learning_rate = float(learning_rate)
        self._epsilon = float(epsilon)
        self._batch_size_advantage = int(batch_size_advantage)
        self._batch_size_strategy = int(batch_size_strategy)
        self._batch_size_baseline = int(batch_size_baseline or batch_size_advantage)
        self._advantage_memory_capacity = int(advantage_memory_capacity)
        self._strategy_memory_capacity = int(strategy_memory_capacity)
        self._baseline_memory_capacity = int(baseline_memory_capacity)
        self._advantage_network_train_steps = int(advantage_network_train_steps)
        self._policy_network_train_steps = int(policy_network_train_steps)
        self._baseline_network_train_steps = int(baseline_network_train_steps)
        self._policy_network_train_every = int(policy_network_train_every)
        self._compute_exploitability = bool(compute_exploitability)
        self._game_value_player_0 = (
            float(game_value_player_0) if game_value_player_0 is not None else LEDUC_GAME_VALUE_P0
        )
        self._average_policy_value_target = (
            float(average_policy_value_target)
            if average_policy_value_target is not None
            else LEDUC_AVERAGE_POLICY_VALUE_TARGET
        )
        valid_target_processing = {"none", "standardize", "clip", "standardize_clip"}
        self._target_processing = str(target_processing).lower()
        if self._target_processing not in valid_target_processing:
            raise ValueError(
                f"target_processing must be one of {sorted(valid_target_processing)}, "
                f"got {target_processing!r}"
            )
        self._target_clip_value = float(target_clip_value)
        if self._target_clip_value <= 0.0:
            raise ValueError("target_clip_value must be positive")
        self._target_standardize_epsilon = float(target_standardize_epsilon)
        if self._target_standardize_epsilon <= 0.0:
            raise ValueError("target_standardize_epsilon must be positive")

        if seed is not None:
            set_seed(seed)

        self._policy_network = MLP(self._info_state_size, self._policy_network_layers, self._num_actions)
        self._policy_softmax = nn.Softmax(dim=-1)
        self._optimizer_policy = torch.optim.Adam(self._policy_network.parameters(), lr=self._learning_rate)

        self._advantage_networks = [
            MLP(self._info_state_size, self._advantage_network_layers, self._num_actions)
            for _ in range(self._num_players)
        ]
        self._baseline_networks = [
            MLP(self._info_state_size, self._baseline_network_layers, self._num_actions)
            for _ in range(self._num_players)
        ]
        self._optimizer_advantages = [
            torch.optim.Adam(net.parameters(), lr=self._learning_rate) for net in self._advantage_networks
        ]
        self._optimizer_baselines = [
            torch.optim.Adam(net.parameters(), lr=self._learning_rate) for net in self._baseline_networks
        ]

        self._advantage_memories = [ReservoirBuffer(self._advantage_memory_capacity) for _ in range(self._num_players)]
        self._strategy_memories = ReservoirBuffer(self._strategy_memory_capacity)
        self._baseline_replays = [CircularReplay(self._baseline_memory_capacity) for _ in range(self._num_players)]

        self._loss_mse = nn.MSELoss(reduction="mean")
        self._iteration = 0
        self._nodes_touched = 0
        self._last_advantage_loss = [np.nan] * self._num_players
        self._last_baseline_loss = [np.nan] * self._num_players
        self._last_policy_loss = np.nan
        self._last_advantage_grad_norm = [np.nan] * self._num_players
        self._last_baseline_grad_norm = [np.nan] * self._num_players
        self._last_policy_grad_norm = np.nan
        self._last_processed_advantage_target_mean = [np.nan] * self._num_players
        self._last_processed_advantage_target_variance = [np.nan] * self._num_players
        self._last_processed_advantage_target_abs_mean = [np.nan] * self._num_players
        self._last_target_standardization_mean = [np.nan] * self._num_players
        self._last_target_standardization_scale = [np.nan] * self._num_players
        self._last_target_clip_fraction = [np.nan] * self._num_players
        self._policy_training_events = 0
        self._policy_gradient_steps_total = 0

    def _apply_learning_rate_schedule(self, iteration: int) -> None:
        iteration = int(iteration)
        if self._num_iterations <= 1:
            progress = 1.0
        else:
            progress = (iteration - 1) / float(self._num_iterations - 1)
            progress = min(max(progress, 0.0), 1.0)

        if self._learning_rate_schedule == "constant":
            lr = self._learning_rate
        elif self._learning_rate_schedule == "cosine_decay":
            start = self._learning_rate
            end = self._learning_rate_end
            lr = end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))
        elif self._learning_rate_schedule == "linear_decay":
            start = self._learning_rate
            end = self._learning_rate_end
            lr = start + (end - start) * progress
        else:
            raise ValueError(f"Unknown learning-rate schedule: {self._learning_rate_schedule}")

        self._current_learning_rate = float(lr)
        optimizers = [self._optimizer_policy] + self._optimizer_advantages + self._optimizer_baselines
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = float(lr)

    # ---- Public OpenSpiel policy interface ----
    def action_probabilities(self, state, player_id=None) -> Dict[int, float]:
        del player_id
        cur_player = int(state.current_player())
        legal_actions = list(state.legal_actions(cur_player))
        if not legal_actions:
            return {}
        info_state = np.asarray(state.information_state_tensor(cur_player), dtype=np.float32)[None, :]
        with torch.no_grad():
            logits = self._policy_network(torch.from_numpy(info_state))[0].cpu().numpy()
        legal_logits = np.asarray([logits[a] for a in legal_actions], dtype=np.float64)
        legal_logits -= float(np.max(legal_logits))
        exp_logits = np.exp(legal_logits)
        probs = exp_logits / float(np.sum(exp_logits))
        return {int(a): float(p) for a, p in zip(legal_actions, probs)}

    # ---- Main training loop ----
    def solve(
        self,
        policy_training_mode: str = "intermittent",
        final_policy_network_train_steps: Optional[int] = None,
        isolate_policy_training_rng: bool = True,
        target_iteration: Optional[int] = None,
        start_time: Optional[float] = None,
    ):
        """Train DREAM and return checkpoint metrics.

        ``intermittent`` trains the average-policy network periodically, matching
        the baseline protocol. ``final_only`` delays average-policy fitting until
        after traversal data collection. The average-policy network is not used
        for traversal, so RNG isolation keeps intermediate policy fitting from
        perturbing later traversal and baseline/advantage samples.

        ``target_iteration`` allows a caller to train incrementally from the
        current solver iteration, which is used by checkpoint/resume ablations.
        """
        mode = str(policy_training_mode)
        if mode not in {"intermittent", "final_only"}:
            raise ValueError("policy_training_mode must be 'intermittent' or 'final_only'.")

        if start_time is None:
            start_time = time.perf_counter()
        if target_iteration is None:
            target_iteration = self._num_iterations
        target_iteration = int(target_iteration)
        if target_iteration < self._iteration:
            raise ValueError("target_iteration must be at least the current solver iteration.")

        curves: List[Dict] = []

        # In the official DREAM repository an iteration denotes a sequential update for both players.
        for iteration in range(self._iteration + 1, target_iteration + 1):
            self._apply_learning_rate_schedule(iteration)
            self._iteration = int(iteration)
            for traverser in range(self._num_players):
                for _ in range(self._num_traversals):
                    state = self._game.new_initial_state()
                    self._traverse_outcome_sampling(state, traverser, pi_reach=1.0, sigma_reach=1.0)

                self._last_advantage_loss[traverser] = self._learn_advantage_network(traverser)
                self._last_baseline_loss[traverser] = self._learn_baseline_network(traverser)

            if mode == "intermittent":
                train_policy_now = (
                    iteration % self._policy_network_train_every == 0
                ) or (iteration == target_iteration)
                if train_policy_now:
                    if isolate_policy_training_rng:
                        rng_state = self._capture_rng_state()
                        self._last_policy_loss = self._learn_strategy_network()
                        curves.append(self._checkpoint_metrics(start_time))
                        self._restore_rng_state(rng_state)
                    else:
                        self._last_policy_loss = self._learn_strategy_network()
                        curves.append(self._checkpoint_metrics(start_time))

            elif mode == "final_only" and iteration == target_iteration:
                original_steps = self._policy_network_train_steps
                if final_policy_network_train_steps is not None:
                    self._policy_network_train_steps = int(final_policy_network_train_steps)
                self._last_policy_loss = self._learn_strategy_network_controlled(isolate_rng=False)
                self._policy_network_train_steps = original_steps
                curves.append(self._checkpoint_metrics(start_time))

        return pd.DataFrame(curves)

    def full_checkpoint_state(self) -> Dict:
        """Return full training state for checkpoint/resume experiments."""
        return {
            "iteration": int(self._iteration),
            "nodes_touched": int(self._nodes_touched),
            "last_advantage_loss": list(self._last_advantage_loss),
            "last_baseline_loss": list(self._last_baseline_loss),
            "last_policy_loss": float(self._last_policy_loss),
            "last_advantage_grad_norm": list(self._last_advantage_grad_norm),
            "last_baseline_grad_norm": list(self._last_baseline_grad_norm),
            "last_policy_grad_norm": float(self._last_policy_grad_norm),
            "last_processed_advantage_target_mean": list(
                self._last_processed_advantage_target_mean
            ),
            "last_processed_advantage_target_variance": list(
                self._last_processed_advantage_target_variance
            ),
            "last_processed_advantage_target_abs_mean": list(
                self._last_processed_advantage_target_abs_mean
            ),
            "last_target_standardization_mean": list(
                self._last_target_standardization_mean
            ),
            "last_target_standardization_scale": list(
                self._last_target_standardization_scale
            ),
            "last_target_clip_fraction": list(self._last_target_clip_fraction),
            "policy_training_events": int(self._policy_training_events),
            "policy_gradient_steps_total": int(self._policy_gradient_steps_total),
            "learning_rate_schedule": self._learning_rate_schedule,
            "learning_rate_end": float(self._learning_rate_end),
            "current_learning_rate": float(self._current_learning_rate),
            "target_processing": self._target_processing,
            "target_clip_value": float(self._target_clip_value),
            "target_standardize_epsilon": float(self._target_standardize_epsilon),
            "policy_network": self._policy_network.state_dict(),
            "optimizer_policy": self._optimizer_policy.state_dict(),
            "advantage_networks": [net.state_dict() for net in self._advantage_networks],
            "optimizer_advantages": [opt.state_dict() for opt in self._optimizer_advantages],
            "baseline_networks": [net.state_dict() for net in self._baseline_networks],
            "optimizer_baselines": [opt.state_dict() for opt in self._optimizer_baselines],
            "advantage_memories": [buf.state_dict() for buf in self._advantage_memories],
            "strategy_memory": self._strategy_memories.state_dict(),
            "baseline_replays": [buf.state_dict() for buf in self._baseline_replays],
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        }

    def load_full_checkpoint_state(self, state: Dict) -> None:
        """Restore a checkpoint created by ``full_checkpoint_state``."""
        self._iteration = int(state["iteration"])
        self._nodes_touched = int(state["nodes_touched"])
        self._last_advantage_loss = list(state.get("last_advantage_loss", [np.nan] * self._num_players))
        self._last_baseline_loss = list(state.get("last_baseline_loss", [np.nan] * self._num_players))
        self._last_policy_loss = float(state.get("last_policy_loss", np.nan))
        self._last_advantage_grad_norm = list(
            state.get("last_advantage_grad_norm", [np.nan] * self._num_players)
        )
        self._last_baseline_grad_norm = list(
            state.get("last_baseline_grad_norm", [np.nan] * self._num_players)
        )
        self._last_policy_grad_norm = float(state.get("last_policy_grad_norm", np.nan))
        self._last_processed_advantage_target_mean = list(
            state.get(
                "last_processed_advantage_target_mean",
                [np.nan] * self._num_players,
            )
        )
        self._last_processed_advantage_target_variance = list(
            state.get(
                "last_processed_advantage_target_variance",
                [np.nan] * self._num_players,
            )
        )
        self._last_processed_advantage_target_abs_mean = list(
            state.get(
                "last_processed_advantage_target_abs_mean",
                [np.nan] * self._num_players,
            )
        )
        self._last_target_standardization_mean = list(
            state.get("last_target_standardization_mean", [np.nan] * self._num_players)
        )
        self._last_target_standardization_scale = list(
            state.get("last_target_standardization_scale", [np.nan] * self._num_players)
        )
        self._last_target_clip_fraction = list(
            state.get("last_target_clip_fraction", [np.nan] * self._num_players)
        )
        self._policy_training_events = int(state.get("policy_training_events", 0))
        self._policy_gradient_steps_total = int(state.get("policy_gradient_steps_total", 0))
        self._learning_rate_schedule = str(state.get("learning_rate_schedule", self._learning_rate_schedule))
        self._learning_rate_end = float(state.get("learning_rate_end", self._learning_rate_end))
        self._current_learning_rate = float(state.get("current_learning_rate", self._current_learning_rate))
        self._target_processing = str(state.get("target_processing", self._target_processing))
        self._target_clip_value = float(state.get("target_clip_value", self._target_clip_value))
        self._target_standardize_epsilon = float(
            state.get(
                "target_standardize_epsilon",
                self._target_standardize_epsilon,
            )
        )

        self._policy_network.load_state_dict(state["policy_network"])
        self._optimizer_policy.load_state_dict(state["optimizer_policy"])
        for net, net_state in zip(self._advantage_networks, state["advantage_networks"]):
            net.load_state_dict(net_state)
        for opt, opt_state in zip(self._optimizer_advantages, state["optimizer_advantages"]):
            opt.load_state_dict(opt_state)
        for net, net_state in zip(self._baseline_networks, state["baseline_networks"]):
            net.load_state_dict(net_state)
        for opt, opt_state in zip(self._optimizer_baselines, state["optimizer_baselines"]):
            opt.load_state_dict(opt_state)

        for buf, buf_state in zip(self._advantage_memories, state["advantage_memories"]):
            buf.load_state_dict(buf_state)
        self._strategy_memories.load_state_dict(state["strategy_memory"])
        for buf, buf_state in zip(self._baseline_replays, state["baseline_replays"]):
            buf.load_state_dict(buf_state)

        if "python_random_state" in state:
            random.setstate(state["python_random_state"])
        if "numpy_random_state" in state:
            np.random.set_state(state["numpy_random_state"])
        if "torch_random_state" in state:
            torch.set_rng_state(state["torch_random_state"])

    # ---- Traversal ----
    def _advance_through_chance(self, state):
        while state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = int(np.random.choice(outcomes, p=probs))
            state = state.child(action)
        return state

    def _traverse_outcome_sampling(self, state, traverser: int, pi_reach: float, sigma_reach: float) -> float:
        self._nodes_touched += 1

        if state.is_terminal():
            return float(state.returns()[traverser])

        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = int(np.random.choice(outcomes, p=probs))
            return self._traverse_outcome_sampling(state.child(action), traverser, pi_reach, sigma_reach)

        current_player = int(state.current_player())
        legal_actions = list(state.legal_actions(current_player))
        info_state = np.asarray(state.information_state_tensor(current_player), dtype=np.float32)

        pi = self._regret_matching_policy(info_state, legal_actions, current_player)
        if current_player == traverser:
            sigma = self._epsilon_mixed_sampling_policy(pi, legal_actions)
        else:
            sigma = pi.copy()

        # Numerical cleanup for sampling.
        sample_probs = np.zeros(self._num_actions, dtype=np.float64)
        for a in legal_actions:
            sample_probs[a] = max(float(sigma[a]), 0.0)
        total = float(np.sum(sample_probs))
        if total <= 0:
            for a in legal_actions:
                sample_probs[a] = 1.0 / len(legal_actions)
        else:
            sample_probs /= total

        action = int(np.random.choice(self._num_actions, p=sample_probs))
        reach_ratio = float(pi_reach / max(sigma_reach, 1e-12))

        # Store average-policy target. The weight carries the correction caused by traverser exploration.
        self._strategy_memories.add(DreamStrategyMemory(info_state, self._iteration, pi.tolist(), reach_ratio))

        next_state = self._advance_through_chance(state.child(action))
        done = next_state.is_terminal()
        reward = float(next_state.returns()[traverser]) if done else 0.0

        if done:
            next_info = np.zeros_like(info_state, dtype=np.float32)
            next_legal_actions: List[int] = []
            next_player = -1
        else:
            next_player = int(next_state.current_player())
            next_info = np.asarray(next_state.information_state_tensor(next_player), dtype=np.float32)
            next_legal_actions = list(next_state.legal_actions(next_player))

        self._baseline_replays[traverser].add(
            BaselineTransition(info_state, action, reward, next_info, next_legal_actions, next_player, done)
        )

        if done:
            child_value = reward
        else:
            if current_player == traverser:
                pi_reach_next = pi_reach * float(pi[action])
                sigma_reach_next = sigma_reach * float(sigma[action])
            else:
                pi_reach_next = pi_reach
                sigma_reach_next = sigma_reach
            child_value = self._traverse_outcome_sampling(next_state, traverser, pi_reach_next, sigma_reach_next)

        baseline_values = self._baseline_values(traverser, info_state)
        sampled_prob = float(sigma[action] if current_player == traverser else pi[action])
        sampled_prob = max(sampled_prob, 1e-12)

        action_values = baseline_values.copy()
        action_values[action] = baseline_values[action] + (child_value - baseline_values[action]) / sampled_prob

        state_value = 0.0
        for a in legal_actions:
            state_value += float(pi[a]) * float(action_values[a])

        if current_player == traverser:
            advantage = np.zeros(self._num_actions, dtype=np.float32)
            for a in legal_actions:
                advantage[a] = float(action_values[a] - state_value)
            self._advantage_memories[traverser].add(
                DreamAdvantageMemory(info_state, self._iteration, advantage.tolist(), reach_ratio)
            )

        return float(state_value)

    # ---- Policy and value helpers ----
    def _regret_matching_policy(self, info_state: np.ndarray, legal_actions: Sequence[int], player: int) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(info_state.astype(np.float32)[None, :])
            raw_advantages = self._advantage_networks[player](x)[0].cpu().numpy()
        positive = np.maximum(raw_advantages, 0.0)
        denom = float(np.sum(positive[list(legal_actions)]))
        pi = np.zeros(self._num_actions, dtype=np.float32)
        if denom > 0:
            for a in legal_actions:
                pi[a] = positive[a] / denom
        else:
            # Standard regret-matching fallback: uniform over legal actions.
            for a in legal_actions:
                pi[a] = 1.0 / float(len(legal_actions))
        return pi

    def _epsilon_mixed_sampling_policy(self, pi: np.ndarray, legal_actions: Sequence[int]) -> np.ndarray:
        sigma = np.zeros(self._num_actions, dtype=np.float32)
        if not legal_actions:
            return sigma
        uniform = 1.0 / float(len(legal_actions))
        for a in legal_actions:
            sigma[a] = (1.0 - self._epsilon) * float(pi[a]) + self._epsilon * uniform
        sigma_sum = float(np.sum(sigma))
        if sigma_sum > 0:
            sigma /= sigma_sum
        return sigma

    def _baseline_values(self, traverser: int, info_state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(info_state.astype(np.float32)[None, :])
            values = self._baseline_networks[traverser](x)[0].cpu().numpy()
        return values.astype(np.float32)

    def _process_advantage_targets(self, targets: np.ndarray, player: int) -> np.ndarray:
        """Transform sampled advantage targets for the supervised fitting loss."""
        raw = np.asarray(targets, dtype=np.float32)
        processed = raw.copy()
        raw_mean = float(np.mean(raw)) if raw.size else float("nan")

        standardization_mean = 0.0
        standardization_scale = 1.0
        if self._target_processing in {"standardize", "standardize_clip"}:
            standardization_mean = raw_mean
            standardization_scale = max(
                float(np.std(processed)), self._target_standardize_epsilon
            )
            processed = (processed - standardization_mean) / standardization_scale

        clip_fraction = 0.0
        if self._target_processing in {"clip", "standardize_clip"}:
            before_clip = processed.copy()
            processed = np.clip(
                processed,
                -self._target_clip_value,
                self._target_clip_value,
            )
            clip_fraction = (
                float(np.mean(before_clip != processed)) if processed.size else 0.0
            )

        self._last_processed_advantage_target_mean[player] = (
            float(np.mean(processed)) if processed.size else float("nan")
        )
        self._last_processed_advantage_target_variance[player] = (
            float(np.var(processed)) if processed.size else float("nan")
        )
        self._last_processed_advantage_target_abs_mean[player] = (
            float(np.mean(np.abs(processed))) if processed.size else float("nan")
        )
        self._last_target_standardization_mean[player] = float(standardization_mean)
        self._last_target_standardization_scale[player] = float(standardization_scale)
        self._last_target_clip_fraction[player] = float(clip_fraction)
        return processed.astype(np.float32)

    # ---- Learning updates ----
    def _learn_advantage_network(self, player: int) -> float:
        if len(self._advantage_memories[player]) < self._batch_size_advantage:
            return float("nan")
        net = self._advantage_networks[player]
        opt = self._optimizer_advantages[player]
        last_loss = float("nan")
        for _ in range(self._advantage_network_train_steps):
            samples = self._advantage_memories[player].sample(self._batch_size_advantage)
            info_states = np.asarray([s.info_state for s in samples], dtype=np.float32)
            targets = np.asarray([s.advantage for s in samples], dtype=np.float32)
            targets = self._process_advantage_targets(targets, player)
            iterations = np.asarray([s.iteration for s in samples], dtype=np.float32).reshape(-1, 1)
            imp_weights = np.asarray([s.weight for s in samples], dtype=np.float32).reshape(-1, 1)
            imp_weights = np.clip(imp_weights, 0.0, 100.0)
            mult = np.sqrt(np.maximum(iterations, 1.0) * np.maximum(imp_weights, 1e-8)).astype(np.float32)
            x = torch.from_numpy(info_states)
            y = torch.from_numpy(targets)
            m = torch.from_numpy(mult)
            opt.zero_grad()
            out = net(x)
            loss = self._loss_mse(m * out, m * y)
            loss.backward()
            self._last_advantage_grad_norm[player] = grad_norm(net.parameters())
            opt.step()
            last_loss = float(loss.detach().cpu().item())
        return last_loss

    def _learn_baseline_network(self, traverser: int) -> float:
        if len(self._baseline_replays[traverser]) < self._batch_size_baseline:
            return float("nan")
        net = self._baseline_networks[traverser]
        opt = self._optimizer_baselines[traverser]
        last_loss = float("nan")
        for _ in range(self._baseline_network_train_steps):
            samples = self._baseline_replays[traverser].sample(self._batch_size_baseline)
            preds: List[torch.Tensor] = []
            targets: List[torch.Tensor] = []
            for tr in samples:
                s = torch.from_numpy(np.asarray(tr.info_state, dtype=np.float32)[None, :])
                q_values = net(s)[0]
                preds.append(q_values[int(tr.action)])
                if bool(tr.done):
                    targets.append(torch.tensor(float(tr.reward), dtype=torch.float32))
                    continue
                next_info = np.asarray(tr.next_info_state, dtype=np.float32)
                next_legal = list(tr.next_legal_actions)
                next_player = int(tr.next_player)
                pi_next = self._regret_matching_policy(next_info, next_legal, next_player)
                with torch.no_grad():
                    q_next = net(torch.from_numpy(next_info[None, :]))[0].cpu().numpy()
                exp_next = sum(float(pi_next[a]) * float(q_next[a]) for a in next_legal)
                targets.append(torch.tensor(float(tr.reward) + exp_next, dtype=torch.float32))
            pred_t = torch.stack(preds)
            target_t = torch.stack(targets)
            opt.zero_grad()
            loss = self._loss_mse(pred_t, target_t)
            loss.backward()
            self._last_baseline_grad_norm[traverser] = grad_norm(net.parameters())
            opt.step()
            last_loss = float(loss.detach().cpu().item())
        return last_loss

    def _capture_rng_state(self):
        return random.getstate(), np.random.get_state(), torch.get_rng_state()

    def _restore_rng_state(self, rng_state) -> None:
        py_state, np_state, torch_state = rng_state
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)

    def _learn_strategy_network_controlled(self, isolate_rng: bool = True) -> float:
        if not isolate_rng:
            return self._learn_strategy_network()
        rng_state = self._capture_rng_state()
        loss = self._learn_strategy_network()
        self._restore_rng_state(rng_state)
        return loss

    def _learn_strategy_network(self) -> float:
        if len(self._strategy_memories) < self._batch_size_strategy:
            return float("nan")
        self._policy_training_events += 1
        last_loss = float("nan")
        for _ in range(self._policy_network_train_steps):
            samples = self._strategy_memories.sample(self._batch_size_strategy)
            info_states = np.asarray([s.info_state for s in samples], dtype=np.float32)
            strategies = np.asarray([s.strategy_action_probs for s in samples], dtype=np.float32)
            iterations = np.asarray([s.iteration for s in samples], dtype=np.float32).reshape(-1, 1)
            imp_weights = np.asarray([s.weight for s in samples], dtype=np.float32).reshape(-1, 1)
            imp_weights = np.clip(imp_weights, 0.0, 100.0)
            mult = np.sqrt(np.maximum(iterations, 1.0) * np.maximum(imp_weights, 1e-8)).astype(np.float32)
            x = torch.from_numpy(info_states)
            y = torch.from_numpy(strategies)
            m = torch.from_numpy(mult)
            self._optimizer_policy.zero_grad()
            logits = self._policy_network(x)
            probs = self._policy_softmax(logits)
            loss = self._loss_mse(m * probs, m * y)
            loss.backward()
            self._last_policy_grad_norm = grad_norm(self._policy_network.parameters())
            self._optimizer_policy.step()
            self._policy_gradient_steps_total += 1
            last_loss = float(loss.detach().cpu().item())
        return last_loss

    # ---- Diagnostics ----
    def _checkpoint_metrics(self, start_time: float) -> Dict:
        tab_policy = policy.tabular_policy_from_callable(self._game, self.action_probabilities)
        policy_value = float(
            expected_game_score.policy_value(self._game.new_initial_state(), [tab_policy] * self._num_players)[0]
        )
        average_policy_value = policy_value
        if self._compute_exploitability:
            nash_conv = float(exploitability.nash_conv(self._game, tab_policy))
            expl = nash_conv / 2.0
        else:
            nash_conv = float("nan")
            expl = float("nan")
        policy_diag = self._policy_network_diagnostics()
        advantage_diag = self._advantage_target_summary()
        processed_advantage_diag = self._processed_advantage_target_diagnostics()
        baseline_diag = self._baseline_replay_summary()
        row = {
            "iteration": int(self._iteration),
            "nodes_touched": int(self._nodes_touched),
            "wall_clock_seconds": float(time.perf_counter() - start_time),
            "learning_rate": float(self._current_learning_rate),
            "nash_conv": nash_conv,
            "exploitability": expl,
            "policy_value_player_0": policy_value,
            "average_policy_value": average_policy_value,
            "policy_value_error": abs(policy_value - self._game_value_player_0),
            "average_policy_value_error": abs(
                average_policy_value - self._average_policy_value_target
            ),
            "policy_loss": float(self._last_policy_loss),
            "advantage_loss_player_0": float(self._last_advantage_loss[0]),
            "advantage_loss_player_1": float(self._last_advantage_loss[1]),
            "baseline_loss_player_0": float(self._last_baseline_loss[0]),
            "baseline_loss_player_1": float(self._last_baseline_loss[1]),
            "advantage_grad_norm_player_0": float(self._last_advantage_grad_norm[0]),
            "advantage_grad_norm_player_1": float(self._last_advantage_grad_norm[1]),
            "baseline_grad_norm_player_0": float(self._last_baseline_grad_norm[0]),
            "baseline_grad_norm_player_1": float(self._last_baseline_grad_norm[1]),
            "policy_grad_norm": float(self._last_policy_grad_norm),
            "policy_training_events": int(self._policy_training_events),
            "policy_gradient_steps_total": int(self._policy_gradient_steps_total),
            "strategy_buffer_size": int(len(self._strategy_memories)),
            "advantage_buffer_size_player_0": int(len(self._advantage_memories[0])),
            "advantage_buffer_size_player_1": int(len(self._advantage_memories[1])),
            "baseline_buffer_size_player_0": int(len(self._baseline_replays[0])),
            "baseline_buffer_size_player_1": int(len(self._baseline_replays[1])),
        }
        row.update(policy_diag)
        row.update(advantage_diag)
        row.update(processed_advantage_diag)
        row.update(baseline_diag)
        return row

    def _policy_network_diagnostics(self, max_states: int = 64) -> Dict[str, float]:
        if len(self._strategy_memories) == 0:
            return {"legal_action_mass_mean": np.nan, "legal_action_mass_min": np.nan, "policy_entropy_mean": np.nan}
        samples = self._strategy_memories.sample_up_to(max_states)
        masses = []
        entropies = []
        for s in samples:
            info = torch.from_numpy(np.asarray(s.info_state, dtype=np.float32)[None, :])
            with torch.no_grad():
                probs = self._policy_softmax(self._policy_network(info))[0].cpu().numpy()
            legal = np.where(np.asarray(s.strategy_action_probs) > 0)[0].tolist()
            if not legal:
                continue
            p_legal = np.asarray([probs[a] for a in legal], dtype=np.float64)
            mass = float(np.sum(p_legal))
            masses.append(mass)
            if mass > 0:
                p_legal = p_legal / mass
                entropies.append(float(-np.sum(p_legal * np.log(np.maximum(p_legal, 1e-12)))))
        return {
            "legal_action_mass_mean": safe_mean(masses),
            "legal_action_mass_min": float(np.min(masses)) if masses else float("nan"),
            "policy_entropy_mean": safe_mean(entropies),
        }

    def _advantage_target_summary(self, max_samples_per_player: int = 5000) -> Dict[str, float]:
        vals = []
        weights = []
        for p in range(self._num_players):
            for s in self._advantage_memories[p].sample_up_to(max_samples_per_player):
                vals.extend(list(s.advantage))
                weights.append(float(s.weight))
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return {
            "advantage_target_count_sampled": int(len(arr)),
            "advantage_target_mean": float(arr.mean()) if len(arr) else np.nan,
            "advantage_target_variance": float(arr.var()) if len(arr) else np.nan,
            "advantage_target_abs_mean": float(np.mean(np.abs(arr))) if len(arr) else np.nan,
            "importance_weight_mean": safe_mean(weights),
            "importance_weight_std": safe_std(weights),
        }

    def _processed_advantage_target_diagnostics(self) -> Dict[str, float]:
        diagnostics = {}
        for player in range(self._num_players):
            diagnostics[f"processed_advantage_target_mean_player_{player}"] = float(
                self._last_processed_advantage_target_mean[player]
            )
            diagnostics[f"processed_advantage_target_variance_player_{player}"] = float(
                self._last_processed_advantage_target_variance[player]
            )
            diagnostics[f"processed_advantage_target_abs_mean_player_{player}"] = float(
                self._last_processed_advantage_target_abs_mean[player]
            )
            diagnostics[f"target_standardization_mean_player_{player}"] = float(
                self._last_target_standardization_mean[player]
            )
            diagnostics[f"target_standardization_scale_player_{player}"] = float(
                self._last_target_standardization_scale[player]
            )
            diagnostics[f"target_clip_fraction_player_{player}"] = float(
                self._last_target_clip_fraction[player]
            )
        return diagnostics

    def _baseline_replay_summary(self, max_samples_per_player: int = 5000) -> Dict[str, float]:
        rewards = []
        for p in range(self._num_players):
            for tr in self._baseline_replays[p].sample_up_to(max_samples_per_player):
                rewards.append(float(tr.reward))
        return {
            "baseline_reward_mean_sampled": safe_mean(rewards),
            "baseline_reward_variance_sampled": float(np.var(rewards)) if len(rewards) else np.nan,
        }
