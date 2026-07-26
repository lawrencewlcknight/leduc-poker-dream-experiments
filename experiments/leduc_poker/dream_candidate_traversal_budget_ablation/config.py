"""Configuration for the candidate-baseline DREAM traversal-budget ablation."""

from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    make_candidate_hp_experiment_config,
    make_smoke_test_config_overrides,
    make_variant,
)


BASELINE_VARIANT = "candidate_traversals_160_baseline"


TRAVERSAL_BUDGET_VARIANTS = [
    make_variant(
        BASELINE_VARIANT,
        "160 traversals",
        hp_family="traversal_budget",
        hp_value="num_traversals=160",
        description="Architecture-selected candidate with the original traversal budget.",
        num_traversals=160,
    ),
    make_variant(
        "candidate_traversals_320",
        "320 traversals",
        hp_family="traversal_budget",
        hp_value="num_traversals=320",
        description=(
            "Doubles the outcome-sampling traversals per player per DREAM iteration, "
            "matching the strongest traversal signal from the constrained search."
        ),
        num_traversals=320,
    ),
    make_variant(
        "candidate_traversals_480",
        "480 traversals",
        hp_family="traversal_budget",
        hp_value="num_traversals=480",
        description=(
            "Tests whether additional sampled traversals continue to reduce target "
            "noise after the 320-traversal setting."
        ),
        num_traversals=480,
    ),
]


EXPERIMENT_CONFIG = make_candidate_hp_experiment_config(
    experiment_name="leduc_poker_dream_candidate_traversal_budget_ablation",
    algorithm="DREAM-style OpenSpiel candidate traversal-budget ablation",
    plot_prefix="dream_candidate_traversal_budget",
    plot_title="DREAM Candidate Traversal-Budget Ablation",
    output_subdir="dream_candidate_traversal_budget_ablation",
    baseline_variant=BASELINE_VARIANT,
    variants=TRAVERSAL_BUDGET_VARIANTS,
    treatment_keys=["num_traversals"],
)


SMOKE_TEST_CONFIG_OVERRIDES = make_smoke_test_config_overrides(
    "dream_candidate_traversal_budget_ablation"
)
