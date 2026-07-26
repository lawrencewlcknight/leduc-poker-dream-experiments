"""Configuration for the candidate-baseline DREAM policy-extraction budget ablation."""

from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    make_candidate_hp_experiment_config,
    make_smoke_test_config_overrides,
    make_variant,
)


BASELINE_VARIANT = "candidate_policy_steps_100_baseline"


POLICY_EXTRACTION_BUDGET_VARIANTS = [
    make_variant(
        BASELINE_VARIANT,
        "Policy steps 100",
        hp_family="policy_extraction_budget",
        hp_value="policy_network_train_steps=100",
        description="Architecture-selected candidate with the inherited average-policy update budget.",
        policy_network_train_steps=100,
    ),
    make_variant(
        "candidate_policy_steps_50",
        "Policy steps 50",
        hp_family="policy_extraction_budget",
        hp_value="policy_network_train_steps=50",
        description=(
            "Reduces the average-policy supervised budget, testing whether the policy "
            "extractor is over-optimising noisy strategy replay."
        ),
        policy_network_train_steps=50,
    ),
    make_variant(
        "candidate_policy_steps_200",
        "Policy steps 200",
        hp_family="policy_extraction_budget",
        hp_value="policy_network_train_steps=200",
        description=(
            "Doubles the average-policy supervised budget, testing whether exploitability "
            "is limited by underfitting the evaluated policy network."
        ),
        policy_network_train_steps=200,
    ),
    make_variant(
        "candidate_policy_steps_400",
        "Policy steps 400",
        hp_family="policy_extraction_budget",
        hp_value="policy_network_train_steps=400",
        description=(
            "Strong policy-extraction fit test for identifying whether the average-policy "
            "approximator remains the bottleneck under the candidate architecture."
        ),
        policy_network_train_steps=400,
    ),
]


EXPERIMENT_CONFIG = make_candidate_hp_experiment_config(
    experiment_name="leduc_poker_dream_candidate_policy_extraction_budget_ablation",
    algorithm="DREAM-style OpenSpiel candidate average-policy extraction-budget ablation",
    plot_prefix="dream_candidate_policy_extraction_budget",
    plot_title="DREAM Candidate Policy-Extraction Budget Ablation",
    output_subdir="dream_candidate_policy_extraction_budget_ablation",
    baseline_variant=BASELINE_VARIANT,
    variants=POLICY_EXTRACTION_BUDGET_VARIANTS,
    treatment_keys=["policy_network_train_steps"],
)


SMOKE_TEST_CONFIG_OVERRIDES = make_smoke_test_config_overrides(
    "dream_candidate_policy_extraction_budget_ablation"
)
