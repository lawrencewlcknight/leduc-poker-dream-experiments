"""Regression tests for reusing the Experiment 22 candidate comparator."""

from __future__ import annotations

import copy

from experiments.leduc_poker.dream_candidate_advantage_batch_size_ablation.config import (
    ADVANTAGE_BATCH_SIZE_VARIANTS,
    BASELINE_VARIANT,
    EXPERIMENT_CONFIG,
)
from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    ensure_fixed_baseline_variant,
    load_fixed_baseline_outputs,
)


def test_fixed_baseline_loader_remaps_exp22_candidate_outputs(tmp_path):
    config = copy.deepcopy(EXPERIMENT_CONFIG)
    config["seeds"] = [1234]
    variants = ensure_fixed_baseline_variant(config, ADVANTAGE_BATCH_SIZE_VARIANTS[1:])

    curves, summaries = load_fixed_baseline_outputs(
        config,
        variants,
        config["treatment_keys"],
        tmp_path,
    )

    assert len(curves) == 1
    curves_df = curves[0]
    assert len(summaries) == 1
    assert set(curves_df["variant"]) == {BASELINE_VARIANT}
    assert summaries[0]["variant"] == BASELINE_VARIANT
    assert curves_df["batch_size_advantage"].nunique() == 1
    assert curves_df["batch_size_advantage"].iloc[0] == 1024
    assert summaries[0]["batch_size_advantage"] == 1024
    assert curves_df["baseline_reused_from_artifact"].all()
    assert summaries[0]["baseline_reused_from_artifact"] is True
    assert (tmp_path / BASELINE_VARIANT / "seed_1234" / "checkpoint_curves.csv").exists()
    assert (tmp_path / BASELINE_VARIANT / "seed_1234" / "seed_summary.json").exists()
