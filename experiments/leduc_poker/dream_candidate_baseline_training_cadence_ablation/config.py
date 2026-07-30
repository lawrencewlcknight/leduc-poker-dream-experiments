"""Configuration for the candidate DREAM baseline-training cadence ablation."""

from dream_poker.constants import DEFAULT_SEEDS_5
from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    make_candidate_hp_experiment_config,
    make_smoke_test_config_overrides,
    make_variant,
)


BASELINE_VARIANT = "candidate_baseline_every_1_baseline"


BASELINE_TRAINING_CADENCE_VARIANTS = [
    make_variant(
        BASELINE_VARIANT,
        "Baseline every 1",
        hp_family="baseline_training_cadence",
        hp_value="baseline_network_train_every=1",
        description=(
            "Experiment 22 architecture-selected candidate with learned-baseline "
            "training on every DREAM iteration."
        ),
        baseline_network_train_every=1,
    ),
    make_variant(
        "candidate_baseline_every_5",
        "Baseline every 5",
        hp_family="baseline_training_cadence",
        hp_value="baseline_network_train_every=5",
        description=(
            "Trains the learned baseline on the first DREAM iteration and then "
            "on iterations divisible by 5, testing whether Q-baseline targets "
            "change slowly enough to permit substantial compute savings."
        ),
        baseline_network_train_every=5,
    ),
    make_variant(
        "candidate_baseline_every_10",
        "Baseline every 10",
        hp_family="baseline_training_cadence",
        hp_value="baseline_network_train_every=10",
        description=(
            "Trains the learned baseline on the first DREAM iteration and then "
            "on iterations divisible by 10, probing a more aggressive reduction "
            "in baseline fitting cost."
        ),
        baseline_network_train_every=10,
    ),
    make_variant(
        "candidate_baseline_every_25",
        "Baseline every 25",
        hp_family="baseline_training_cadence",
        hp_value="baseline_network_train_every=25",
        description=(
            "Aligns learned-baseline training with the policy-evaluation cadence, "
            "testing whether stale control variates still provide useful variance "
            "reduction."
        ),
        baseline_network_train_every=25,
    ),
    make_variant(
        "candidate_baseline_every_50",
        "Baseline every 50",
        hp_family="baseline_training_cadence",
        hp_value="baseline_network_train_every=50",
        description=(
            "Sparse learned-baseline fitting schedule designed to expose the point "
            "at which stale Q-baselines begin to harm DREAM learning."
        ),
        baseline_network_train_every=50,
    ),
]


EXPERIMENT_CONFIG = make_candidate_hp_experiment_config(
    experiment_name="leduc_poker_dream_candidate_baseline_training_cadence_ablation",
    algorithm="DREAM-style OpenSpiel candidate baseline-training cadence ablation",
    plot_prefix="dream_candidate_baseline_training_cadence",
    plot_title="DREAM Candidate Baseline-Training Cadence Ablation",
    output_subdir="dream_candidate_baseline_training_cadence_ablation",
    baseline_variant=BASELINE_VARIANT,
    variants=BASELINE_TRAINING_CADENCE_VARIANTS,
    treatment_keys=["baseline_network_train_every"],
)
EXPERIMENT_CONFIG.update({"seeds": list(DEFAULT_SEEDS_5)})


SMOKE_TEST_CONFIG_OVERRIDES = make_smoke_test_config_overrides(
    "dream_candidate_baseline_training_cadence_ablation"
)
