"""Configuration for the candidate-baseline DREAM constant learning-rate ablation."""

from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    make_candidate_hp_experiment_config,
    make_smoke_test_config_overrides,
    make_variant,
)


BASELINE_VARIANT = "candidate_learning_rate_0_003_baseline"


CONSTANT_LEARNING_RATE_VARIANTS = [
    make_variant(
        BASELINE_VARIANT,
        "Learning rate 0.003",
        hp_family="constant_learning_rate",
        hp_value="learning_rate=0.003",
        description="Architecture-selected candidate with the inherited constant learning rate.",
        learning_rate=0.003,
    ),
    make_variant(
        "candidate_learning_rate_0_001",
        "Learning rate 0.001",
        hp_family="constant_learning_rate",
        hp_value="learning_rate=0.001",
        description=(
            "Lower constant optimiser step size, testing whether late-training "
            "instability is caused by overly aggressive supervised updates."
        ),
        learning_rate=0.001,
    ),
    make_variant(
        "candidate_learning_rate_0_006",
        "Learning rate 0.006",
        hp_family="constant_learning_rate",
        hp_value="learning_rate=0.006",
        description=(
            "Higher constant optimiser step size, motivated by the strong random-search "
            "candidate that used faster supervised updates."
        ),
        learning_rate=0.006,
    ),
]


EXPERIMENT_CONFIG = make_candidate_hp_experiment_config(
    experiment_name="leduc_poker_dream_candidate_constant_learning_rate_ablation",
    algorithm="DREAM-style OpenSpiel candidate constant-learning-rate ablation",
    plot_prefix="dream_candidate_constant_learning_rate",
    plot_title="DREAM Candidate Constant Learning-Rate Ablation",
    output_subdir="dream_candidate_constant_learning_rate_ablation",
    baseline_variant=BASELINE_VARIANT,
    variants=CONSTANT_LEARNING_RATE_VARIANTS,
    treatment_keys=["learning_rate"],
)


SMOKE_TEST_CONFIG_OVERRIDES = make_smoke_test_config_overrides(
    "dream_candidate_constant_learning_rate_ablation"
)
