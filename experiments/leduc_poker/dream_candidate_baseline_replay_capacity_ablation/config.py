"""Configuration for the candidate-baseline DREAM learned-baseline replay ablation."""

from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    make_candidate_hp_experiment_config,
    make_smoke_test_config_overrides,
    make_variant,
)


BASELINE_VARIANT = "candidate_baseline_memory_1m_baseline"


BASELINE_REPLAY_CAPACITY_VARIANTS = [
    make_variant(
        BASELINE_VARIANT,
        "Baseline memory 1M",
        hp_family="baseline_replay_capacity",
        hp_value="baseline_memory_capacity=1000000",
        description="Architecture-selected candidate with the inherited 1M learned-baseline reservoir.",
        baseline_memory_capacity=int(1e6),
    ),
    make_variant(
        "candidate_baseline_memory_500k",
        "Baseline memory 500k",
        hp_family="baseline_replay_capacity",
        hp_value="baseline_memory_capacity=500000",
        description=(
            "Moderately fresher learned-baseline replay, testing whether the control "
            "variate benefits from tracking the current sampling distribution."
        ),
        baseline_memory_capacity=int(5e5),
    ),
    make_variant(
        "candidate_baseline_memory_100k",
        "Baseline memory 100k",
        hp_family="baseline_replay_capacity",
        hp_value="baseline_memory_capacity=100000",
        description=(
            "Aggressive learned-baseline replay freshness test, motivated by the "
            "strong random-search candidate with a 100k baseline reservoir."
        ),
        baseline_memory_capacity=int(1e5),
    ),
]


EXPERIMENT_CONFIG = make_candidate_hp_experiment_config(
    experiment_name="leduc_poker_dream_candidate_baseline_replay_capacity_ablation",
    algorithm="DREAM-style OpenSpiel candidate learned-baseline replay-capacity ablation",
    plot_prefix="dream_candidate_baseline_replay_capacity",
    plot_title="DREAM Candidate Learned-Baseline Replay-Capacity Ablation",
    output_subdir="dream_candidate_baseline_replay_capacity_ablation",
    baseline_variant=BASELINE_VARIANT,
    variants=BASELINE_REPLAY_CAPACITY_VARIANTS,
    treatment_keys=["baseline_memory_capacity"],
)


SMOKE_TEST_CONFIG_OVERRIDES = make_smoke_test_config_overrides(
    "dream_candidate_baseline_replay_capacity_ablation"
)
