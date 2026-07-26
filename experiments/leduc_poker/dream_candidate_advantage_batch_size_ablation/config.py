"""Configuration for the candidate-baseline DREAM advantage minibatch-size ablation."""

from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    make_candidate_hp_experiment_config,
    make_smoke_test_config_overrides,
    make_variant,
)


BASELINE_VARIANT = "candidate_advantage_batch_1024_baseline"


ADVANTAGE_BATCH_SIZE_VARIANTS = [
    make_variant(
        BASELINE_VARIANT,
        "Advantage batch 1024",
        hp_family="advantage_batch_size",
        hp_value="batch_size_advantage=1024",
        description="Architecture-selected candidate with the inherited advantage minibatch size.",
        batch_size_advantage=1024,
    ),
    make_variant(
        "candidate_advantage_batch_512",
        "Advantage batch 512",
        hp_family="advantage_batch_size",
        hp_value="batch_size_advantage=512",
        description=(
            "Smaller advantage minibatches, testing whether additional gradient noise "
            "acts as useful regularisation for sampled regret targets."
        ),
        batch_size_advantage=512,
    ),
    make_variant(
        "candidate_advantage_batch_2048",
        "Advantage batch 2048",
        hp_family="advantage_batch_size",
        hp_value="batch_size_advantage=2048",
        description=(
            "Larger advantage minibatches, testing whether lower supervised-gradient "
            "variance improves regret fitting under the 2x128 advantage networks."
        ),
        batch_size_advantage=2048,
    ),
]


EXPERIMENT_CONFIG = make_candidate_hp_experiment_config(
    experiment_name="leduc_poker_dream_candidate_advantage_batch_size_ablation",
    algorithm="DREAM-style OpenSpiel candidate advantage minibatch-size ablation",
    plot_prefix="dream_candidate_advantage_batch_size",
    plot_title="DREAM Candidate Advantage Batch-Size Ablation",
    output_subdir="dream_candidate_advantage_batch_size_ablation",
    baseline_variant=BASELINE_VARIANT,
    variants=ADVANTAGE_BATCH_SIZE_VARIANTS,
    treatment_keys=["batch_size_advantage"],
)


SMOKE_TEST_CONFIG_OVERRIDES = make_smoke_test_config_overrides(
    "dream_candidate_advantage_batch_size_ablation"
)
