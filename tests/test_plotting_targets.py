import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from dream_poker.plotting import (
    AVERAGE_POLICY_VALUE_TARGET_LABEL,
    NASH_EXPLOITABILITY_TARGET_LABEL,
    plot_curve_by_variant,
    plot_mean_curve,
)


def _target_lines(ax):
    return [line for line in ax.lines if line.get_label() == NASH_EXPLOITABILITY_TARGET_LABEL]


def _average_policy_value_target_lines(ax):
    return [
        line
        for line in ax.lines
        if line.get_label().startswith(AVERAGE_POLICY_VALUE_TARGET_LABEL)
    ]


def test_mean_exploitability_curve_has_single_nash_target_line():
    curves_df = pd.DataFrame(
        {
            "seed": [1, 1, 2, 2],
            "iteration": [1, 2, 1, 2],
            "exploitability": [0.2, 0.1, 0.3, 0.15],
        }
    )

    fig, ax = plot_mean_curve(
        curves_df,
        "iteration",
        "exploitability",
        "Exploitability",
        "Exploitability",
    )

    target_lines = _target_lines(ax)
    assert len(target_lines) == 1
    assert set(target_lines[0].get_ydata()) == {0.0}
    plt.close(fig)


def test_non_exploitability_curve_has_no_nash_target_line():
    curves_df = pd.DataFrame(
        {
            "seed": [1, 1, 2, 2],
            "iteration": [1, 2, 1, 2],
            "policy_loss": [0.8, 0.6, 0.9, 0.7],
        }
    )

    fig, ax = plot_mean_curve(
        curves_df,
        "iteration",
        "policy_loss",
        "Policy loss",
        "Policy loss",
    )

    assert _target_lines(ax) == []
    plt.close(fig)


def test_variant_exploitability_curve_has_single_nash_target_line():
    curves_df = pd.DataFrame(
        {
            "variant": ["baseline", "baseline", "candidate", "candidate"],
            "seed": [1, 1, 1, 1],
            "iteration": [1, 2, 1, 2],
            "exploitability": [0.2, 0.1, 0.25, 0.12],
        }
    )

    fig, ax = plot_curve_by_variant(
        curves_df,
        "iteration",
        "exploitability",
        "Exploitability",
        "Exploitability",
        ["baseline", "candidate"],
        {"baseline": "Baseline", "candidate": "Candidate"},
    )

    target_lines = _target_lines(ax)
    assert len(target_lines) == 1
    assert set(target_lines[0].get_ydata()) == {0.0}
    plt.close(fig)


def test_average_policy_value_curve_has_configurable_target_line():
    curves_df = pd.DataFrame(
        {
            "seed": [1, 1, 2, 2],
            "iteration": [1, 2, 1, 2],
            "average_policy_value": [-0.08, -0.06, -0.07, -0.055],
        }
    )

    fig, ax = plot_mean_curve(
        curves_df,
        "iteration",
        "average_policy_value",
        "Average policy value",
        "Average policy value",
        average_policy_value_target=-0.0556,
    )

    target_lines = _average_policy_value_target_lines(ax)
    assert len(target_lines) == 1
    assert set(target_lines[0].get_ydata()) == {-0.0556}
    plt.close(fig)
