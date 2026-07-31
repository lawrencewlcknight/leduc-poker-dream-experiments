import copy

from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    ensure_fixed_baseline_variant,
    load_fixed_baseline_outputs,
)
from experiments.leduc_poker.dream_vectorized_baseline_equivalence.config import (
    EXP22_ORIGINAL_VARIANT,
    EXPERIMENT_CONFIG,
    IMPLEMENTATION_EQUIVALENCE_VARIANTS,
    VECTORIZED_BASELINE_VARIANT,
)
from experiments.leduc_poker.dream_vectorized_baseline_equivalence.run import (
    PAIRED_METRICS,
    TIMING_METRICS,
)


def test_vectorized_equivalence_uses_exp22_artifact_by_default():
    fixed = EXPERIMENT_CONFIG["fixed_baseline"]

    assert fixed["enabled"] is True
    assert EXPERIMENT_CONFIG["baseline_variant"] == EXP22_ORIGINAL_VARIANT
    assert (fixed["source_output_dir"] / fixed["curves_filename"]).exists()
    assert (fixed["source_output_dir"] / fixed["summary_filename"]).exists()
    assert "final_cumulative_baseline_training_seconds" in TIMING_METRICS
    assert "final_cumulative_baseline_training_seconds" in PAIRED_METRICS


def test_vectorized_equivalence_rerun_variant_disables_grad_norm_diagnostic():
    variants_by_id = {variant["variant_id"]: variant for variant in IMPLEMENTATION_EQUIVALENCE_VARIANTS}

    rerun = variants_by_id[VECTORIZED_BASELINE_VARIANT]

    assert rerun["baseline_learner_implementation"] == "vectorized_tensorized"
    assert rerun["baseline_replay_storage"] == "tensorized_numpy_arrays"
    assert rerun["compute_baseline_grad_norm_diagnostics"] is False


def test_fixed_exp22_comparator_is_remapped_to_original_variant(tmp_path):
    config = copy.deepcopy(EXPERIMENT_CONFIG)
    config["seeds"] = [1234]
    variants = ensure_fixed_baseline_variant(config, IMPLEMENTATION_EQUIVALENCE_VARIANTS)

    curves, summaries = load_fixed_baseline_outputs(
        config,
        variants,
        config["treatment_keys"],
        tmp_path,
    )

    assert len(curves) == 1
    assert len(summaries) == 1
    assert summaries[0]["variant"] == EXP22_ORIGINAL_VARIANT
    assert summaries[0]["baseline_learner_implementation"] == "original_loop"
    assert summaries[0]["compute_baseline_grad_norm_diagnostics"] is True
    assert (
        tmp_path
        / EXP22_ORIGINAL_VARIANT
        / "seed_1234"
        / "checkpoint_curves.csv"
    ).exists()
