import math

import numpy as np
import pytest

from dream_poker.replay import BaselineTransition, CircularReplay


def _transition(value: float, *, done: bool = False) -> BaselineTransition:
    info_state = np.asarray([value, value + 1.0, value + 2.0], dtype=np.float32)
    next_info_state = np.asarray([value + 3.0, value + 4.0, value + 5.0], dtype=np.float32)
    next_legal_actions = [] if done else [0, 2]
    return BaselineTransition(
        info_state,
        1,
        float(value),
        next_info_state,
        next_legal_actions,
        0 if not done else -1,
        done,
    )


def test_circular_replay_samples_tensorized_baseline_batches():
    replay = CircularReplay(capacity=4, info_state_size=3, num_actions=3)
    for value in range(4):
        replay.add(_transition(float(value)))

    batch = replay.sample_baseline_batch(6)

    assert batch.info_states.shape == (6, 3)
    assert batch.next_info_states.shape == (6, 3)
    assert batch.next_legal_action_masks.shape == (6, 3)
    assert batch.actions.dtype == np.int64
    assert batch.rewards.dtype == np.float32
    assert batch.dones.dtype == bool
    assert np.all(batch.next_legal_action_masks[:, [0, 2]])
    assert not np.any(batch.next_legal_action_masks[:, 1])


def test_circular_replay_state_round_trip_keeps_tensorized_storage():
    replay = CircularReplay(capacity=3, info_state_size=3, num_actions=3)
    for value in range(5):
        replay.add(_transition(float(value), done=value == 4))

    state = replay.state_dict()
    restored = CircularReplay(capacity=3)
    restored.load_state_dict(state)

    assert len(restored) == 3
    assert restored.state_dict()["tensorized_baseline"] is True
    rewards = sorted(restored.baseline_rewards_sample_up_to(3).tolist())
    assert rewards == [2.0, 3.0, 4.0]
    sample = restored.sample(1)[0]
    assert isinstance(sample, BaselineTransition)


def test_baseline_grad_norm_diagnostic_is_optional():
    pyspiel = pytest.importorskip("pyspiel")

    from dream_poker.solver import DREAMSolver

    game = pyspiel.load_game("kuhn_poker")
    solver = DREAMSolver(
        game,
        policy_network_layers=(4,),
        advantage_network_layers=(4,),
        baseline_network_layers=(4,),
        batch_size_advantage=1,
        batch_size_strategy=1,
        batch_size_baseline=1,
        baseline_network_train_steps=1,
        compute_baseline_grad_norm_diagnostics=False,
        seed=1234,
    )

    info_state = np.zeros(solver._info_state_size, dtype=np.float32)
    solver.baseline_replays[0].add(
        BaselineTransition(info_state, 0, 1.0, info_state, [], -1, True)
    )

    loss = solver._learn_baseline_network(0)

    assert math.isfinite(loss)
    assert math.isnan(solver._last_baseline_grad_norm[0])


def test_baseline_grad_norm_diagnostic_can_be_enabled():
    pyspiel = pytest.importorskip("pyspiel")

    from dream_poker.solver import DREAMSolver

    game = pyspiel.load_game("kuhn_poker")
    solver = DREAMSolver(
        game,
        policy_network_layers=(4,),
        advantage_network_layers=(4,),
        baseline_network_layers=(4,),
        batch_size_advantage=1,
        batch_size_strategy=1,
        batch_size_baseline=1,
        baseline_network_train_steps=1,
        compute_baseline_grad_norm_diagnostics=True,
        seed=1234,
    )

    info_state = np.zeros(solver._info_state_size, dtype=np.float32)
    solver.baseline_replays[0].add(
        BaselineTransition(info_state, 0, 1.0, info_state, [], -1, True)
    )

    loss = solver._learn_baseline_network(0)

    assert math.isfinite(loss)
    assert math.isfinite(solver._last_baseline_grad_norm[0])


def test_vectorized_baseline_learner_handles_nonterminal_targets():
    pyspiel = pytest.importorskip("pyspiel")

    from dream_poker.solver import DREAMSolver

    game = pyspiel.load_game("kuhn_poker")
    solver = DREAMSolver(
        game,
        policy_network_layers=(4,),
        advantage_network_layers=(4,),
        baseline_network_layers=(4,),
        batch_size_advantage=1,
        batch_size_strategy=1,
        batch_size_baseline=1,
        baseline_network_train_steps=1,
        seed=1234,
    )

    info_state = np.zeros(solver._info_state_size, dtype=np.float32)
    next_info_state = np.ones(solver._info_state_size, dtype=np.float32)
    solver.baseline_replays[0].add(
        BaselineTransition(
            info_state,
            0,
            0.5,
            next_info_state,
            [0, 1],
            0,
            False,
        )
    )

    loss = solver._learn_baseline_network(0)

    assert math.isfinite(loss)
