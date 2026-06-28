import numpy as np

from dream_poker.solver import DREAMSolver


def _solver_for_target_processing(mode: str) -> DREAMSolver:
    solver = DREAMSolver.__new__(DREAMSolver)
    solver._target_processing = mode
    solver._target_clip_value = 1.0
    solver._target_standardize_epsilon = 1e-6
    solver._num_players = 2
    solver._last_processed_advantage_target_mean = [np.nan, np.nan]
    solver._last_processed_advantage_target_variance = [np.nan, np.nan]
    solver._last_processed_advantage_target_abs_mean = [np.nan, np.nan]
    solver._last_target_standardization_mean = [np.nan, np.nan]
    solver._last_target_standardization_scale = [np.nan, np.nan]
    solver._last_target_clip_fraction = [np.nan, np.nan]
    return solver


def test_raw_target_processing_keeps_targets_unchanged():
    solver = _solver_for_target_processing("none")
    targets = np.asarray([[0.0, 2.0, -2.0], [0.0, 4.0, -4.0]], dtype=np.float32)

    processed = solver._process_advantage_targets(targets, player=0)

    np.testing.assert_allclose(processed, targets)
    assert solver._last_processed_advantage_target_variance[0] == np.var(targets)
    assert solver._last_target_standardization_scale[0] == 1.0
    assert solver._last_target_clip_fraction[0] == 0.0


def test_standardize_clip_target_processing_records_diagnostics():
    solver = _solver_for_target_processing("standardize_clip")
    targets = np.asarray([[0.0, 2.0, -2.0], [0.0, 4.0, -4.0]], dtype=np.float32)

    processed = solver._process_advantage_targets(targets, player=1)

    assert processed.dtype == np.float32
    assert np.max(processed) <= 1.0
    assert np.min(processed) >= -1.0
    assert solver._last_target_standardization_mean[1] == np.mean(targets)
    assert solver._last_target_standardization_scale[1] == np.std(targets)
    assert solver._last_target_clip_fraction[1] > 0.0
    assert solver._last_processed_advantage_target_variance[1] == np.var(processed)
