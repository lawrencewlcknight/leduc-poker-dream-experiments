"""Configuration for the candidate-baseline DREAM advantage-fitting steps ablation."""

from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    make_candidate_hp_experiment_config,
    make_smoke_test_config_overrides,
    make_variant,
)


BASELINE_VARIANT = "candidate_advantage_steps_50_baseline"


ADVANTAGE_FITTING_STEPS_VARIANTS = [
    make_variant(
        BASELINE_VARIANT,
        "Advantage steps 50",
        hp_family="advantage_fitting_steps",
        hp_value="advantage_network_train_steps=50",
        description="Architecture-selected candidate with the inherited advantage fitting budget.",
        advantage_network_train_steps=50,
    ),
    make_variant(
        "candidate_advantage_steps_25",
        "Advantage steps 25",
        hp_family="advantage_fitting_steps",
        hp_value="advantage_network_train_steps=25",
        description=(
            "Reduced regret-approximator fitting effort, testing whether fewer updates "
            "regularise noisy outcome-sampled advantage targets."
        ),
        advantage_network_train_steps=25,
    ),
    make_variant(
        "candidate_advantage_steps_100",
        "Advantage steps 100",
        hp_family="advantage_fitting_steps",
        hp_value="advantage_network_train_steps=100",
        description=(
            "Doubles regret-approximator fitting effort to test whether the larger "
            "2x128 advantage networks are under-optimised."
        ),
        advantage_network_train_steps=100,
    ),
    make_variant(
        "candidate_advantage_steps_200",
        "Advantage steps 200",
        hp_family="advantage_fitting_steps",
        hp_value="advantage_network_train_steps=200",
        description=(
            "Strong regret-approximator fitting test for identifying whether extra "
            "supervised optimisation improves DREAM's average policy."
        ),
        advantage_network_train_steps=200,
    ),
]


EXPERIMENT_CONFIG = make_candidate_hp_experiment_config(
    experiment_name="leduc_poker_dream_candidate_advantage_fitting_steps_ablation",
    algorithm="DREAM-style OpenSpiel candidate advantage-fitting step-count ablation",
    plot_prefix="dream_candidate_advantage_fitting_steps",
    plot_title="DREAM Candidate Advantage-Fitting Steps Ablation",
    output_subdir="dream_candidate_advantage_fitting_steps_ablation",
    baseline_variant=BASELINE_VARIANT,
    variants=ADVANTAGE_FITTING_STEPS_VARIANTS,
    treatment_keys=["advantage_network_train_steps"],
)


SMOKE_TEST_CONFIG_OVERRIDES = make_smoke_test_config_overrides(
    "dream_candidate_advantage_fitting_steps_ablation"
)
