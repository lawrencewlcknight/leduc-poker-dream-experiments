"""Configuration for the candidate DREAM epsilon comparison with baseline cadence 50."""

from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    DEFAULT_SEEDS,
    make_candidate_hp_experiment_config,
    make_smoke_test_config_overrides,
    make_variant,
)


BASELINE_VARIANT = "candidate_epsilon_006_baseline_every_50"
EPSILON_020_VARIANT = "candidate_epsilon_020_baseline_every_50"


EPSILON_BASELINE50_VARIANTS = [
    make_variant(
        BASELINE_VARIANT,
        "epsilon 0.06, baseline every 50",
        hp_family="epsilon_baseline50",
        hp_value="epsilon=0.06; baseline_network_train_every=50",
        description=(
            "Experiment 22 architecture-selected candidate with the inherited "
            "epsilon value and sparse learned-baseline training every 50 DREAM "
            "iterations."
        ),
        epsilon=0.06,
        baseline_network_train_every=50,
    ),
    make_variant(
        EPSILON_020_VARIANT,
        "epsilon 0.20, baseline every 50",
        hp_family="epsilon_baseline50",
        hp_value="epsilon=0.20; baseline_network_train_every=50",
        description=(
            "Sparse-baseline treatment that changes only the traversal exploration "
            "rate to epsilon 0.20 while keeping the Experiment 22 candidate "
            "architecture and optimisation settings."
        ),
        epsilon=0.20,
        baseline_network_train_every=50,
    ),
]


EXPERIMENT_CONFIG = make_candidate_hp_experiment_config(
    experiment_name="leduc_poker_dream_candidate_epsilon_baseline50_comparison",
    algorithm="DREAM-style OpenSpiel candidate epsilon comparison with baseline cadence 50",
    plot_prefix="dream_candidate_epsilon_baseline50",
    plot_title="DREAM Candidate Epsilon Comparison with Baseline Cadence 50",
    output_subdir="dream_candidate_epsilon_baseline50_comparison",
    baseline_variant=BASELINE_VARIANT,
    variants=EPSILON_BASELINE50_VARIANTS,
    treatment_keys=["epsilon", "baseline_network_train_every"],
)
EXPERIMENT_CONFIG.update(
    {
        "seeds": list(DEFAULT_SEEDS),
        "baseline_network_train_every": 50,
        "fixed_baseline": {"enabled": False},
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = make_smoke_test_config_overrides(
    "dream_candidate_epsilon_baseline50_comparison"
)
