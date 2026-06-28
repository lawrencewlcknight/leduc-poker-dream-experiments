import numpy as np

from dream_poker.solver import DREAMSolver


def _solver_for_average_strategy_weighting(mode: str) -> DREAMSolver:
    solver = DREAMSolver.__new__(DREAMSolver)
    solver._average_strategy_weighting = mode
    return solver


def test_linear_average_strategy_weighting_matches_existing_multiplier():
    solver = _solver_for_average_strategy_weighting("linear")
    iterations = np.asarray([[1.0], [4.0], [9.0]], dtype=np.float32)
    imp_weights = np.asarray([[1.0], [0.25], [4.0]], dtype=np.float32)

    multiplier = solver._average_strategy_loss_multiplier(iterations, imp_weights)

    expected = np.sqrt(iterations * imp_weights).astype(np.float32)
    np.testing.assert_allclose(multiplier, expected)


def test_uniform_average_strategy_weighting_preserves_importance_correction():
    solver = _solver_for_average_strategy_weighting("uniform")
    iterations = np.asarray([[1.0], [4.0], [9.0]], dtype=np.float32)
    imp_weights = np.asarray([[1.0], [0.25], [4.0]], dtype=np.float32)

    multiplier = solver._average_strategy_loss_multiplier(iterations, imp_weights)

    expected = np.sqrt(imp_weights).astype(np.float32)
    np.testing.assert_allclose(multiplier, expected)
