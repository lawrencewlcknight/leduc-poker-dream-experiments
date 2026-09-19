import numpy as np

from experiments.leduc_poker.dream_frozen_reservoir_distillation_audit.distillation import (
    FrozenReservoir,
    group_reservoir,
    load_frozen_reservoir,
    row_objective_weights,
    save_frozen_reservoir,
)


def _reservoir() -> FrozenReservoir:
    return FrozenReservoir(
        info_states=np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32),
        iterations=np.asarray([1, 3, 2], dtype=np.int32),
        strategies=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.25, 0.75, 0.0]],
            dtype=np.float32,
        ),
        reach_weights=np.asarray([2.0, 1.0, 4.0], dtype=np.float32),
        legal_masks=np.asarray(
            [[True, True, False], [True, True, False], [True, True, False]],
            dtype=np.bool_,
        ),
        capacity=10,
        observations_seen=20,
    )


def test_grouping_uses_iteration_times_reach_as_exact_sufficient_statistic():
    reservoir = _reservoir()
    assert np.allclose(row_objective_weights(reservoir), [2.0, 3.0, 8.0])
    grouped = group_reservoir(reservoir)
    assert grouped.source_rows == 3
    assert grouped.size == 2
    lookup = {
        tuple(info.tolist()): (target, mass)
        for info, target, mass in zip(
            grouped.info_states, grouped.strategies, grouped.objective_masses
        )
    }
    target, mass = lookup[(1.0, 0.0)]
    assert mass == 5.0
    assert np.allclose(target, [0.4, 0.6, 0.0])
    target, mass = lookup[(0.0, 1.0)]
    assert mass == 8.0
    assert np.allclose(target, [0.25, 0.75, 0.0])

    predicted = {
        (1.0, 0.0): np.asarray([0.3, 0.7, 0.0]),
        (0.0, 1.0): np.asarray([0.6, 0.4, 0.0]),
    }
    row_weights = row_objective_weights(reservoir)
    row_losses = []
    for info, target, weight in zip(
        reservoir.info_states, reservoir.strategies, row_weights
    ):
        row_losses.append(
            -weight * np.sum(target[:2] * np.log(predicted[tuple(info.tolist())][:2]))
        )
    grouped_losses = []
    for info, target, objective_weight in zip(
        grouped.info_states, grouped.strategies, grouped.objective_weights
    ):
        grouped_losses.append(
            -objective_weight
            * np.sum(target[:2] * np.log(predicted[tuple(info.tolist())][:2]))
        )
    assert np.isclose(np.mean(row_losses), np.mean(grouped_losses))


def test_compact_reservoir_round_trip(tmp_path):
    source = _reservoir()
    path = tmp_path / "reservoir.npz"
    save_frozen_reservoir(path, source)
    restored = load_frozen_reservoir(path)
    assert restored.capacity == source.capacity
    assert restored.observations_seen == source.observations_seen
    assert np.array_equal(restored.info_states, source.info_states)
    assert np.array_equal(restored.iterations, source.iterations)
    assert np.array_equal(restored.strategies, source.strategies)
    assert np.array_equal(restored.reach_weights, source.reach_weights)
    assert np.array_equal(restored.legal_masks, source.legal_masks)
