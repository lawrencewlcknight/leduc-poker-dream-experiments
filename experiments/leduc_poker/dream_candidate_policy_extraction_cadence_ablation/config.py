"""Configuration for the candidate-baseline DREAM policy-extraction cadence ablation."""

from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    make_candidate_hp_experiment_config,
    make_smoke_test_config_overrides,
    make_variant,
)


BASELINE_VARIANT = "candidate_policy_every_25_baseline"


POLICY_EXTRACTION_CADENCE_VARIANTS = [
    make_variant(
        BASELINE_VARIANT,
        "Policy train every 25",
        hp_family="policy_extraction_cadence",
        hp_value="policy_network_train_every=25",
        description="Architecture-selected candidate with the inherited intermittent policy-training cadence.",
        policy_network_train_every=25,
        evaluation_interval=25,
    ),
    make_variant(
        "candidate_policy_every_10",
        "Policy train every 10",
        hp_family="policy_extraction_cadence",
        hp_value="policy_network_train_every=10",
        description=(
            "More frequent average-policy fitting, testing whether reduced lag behind "
            "the evolving regret-induced strategy improves the evaluated policy."
        ),
        policy_network_train_every=10,
        evaluation_interval=10,
    ),
    make_variant(
        "candidate_policy_every_50",
        "Policy train every 50",
        hp_family="policy_extraction_cadence",
        hp_value="policy_network_train_every=50",
        description=(
            "Less frequent average-policy fitting, testing whether larger refresh "
            "intervals reduce optimisation churn without harming final extraction."
        ),
        policy_network_train_every=50,
        evaluation_interval=50,
    ),
]


EXPERIMENT_CONFIG = make_candidate_hp_experiment_config(
    experiment_name="leduc_poker_dream_candidate_policy_extraction_cadence_ablation",
    algorithm="DREAM-style OpenSpiel candidate average-policy extraction-cadence ablation",
    plot_prefix="dream_candidate_policy_extraction_cadence",
    plot_title="DREAM Candidate Policy-Extraction Cadence Ablation",
    output_subdir="dream_candidate_policy_extraction_cadence_ablation",
    baseline_variant=BASELINE_VARIANT,
    variants=POLICY_EXTRACTION_CADENCE_VARIANTS,
    treatment_keys=["policy_network_train_every"],
)


SMOKE_TEST_CONFIG_OVERRIDES = make_smoke_test_config_overrides(
    "dream_candidate_policy_extraction_cadence_ablation"
)
