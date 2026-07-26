"""Configuration for the candidate-baseline DREAM strategy replay-capacity ablation."""

from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    make_candidate_hp_experiment_config,
    make_smoke_test_config_overrides,
    make_variant,
)


BASELINE_VARIANT = "candidate_strategy_memory_1m_baseline"


STRATEGY_REPLAY_CAPACITY_VARIANTS = [
    make_variant(
        BASELINE_VARIANT,
        "Strategy memory 1M",
        hp_family="strategy_replay_capacity",
        hp_value="strategy_memory_capacity=1000000",
        description="Architecture-selected candidate with the inherited 1M strategy reservoir.",
        strategy_memory_capacity=int(1e6),
    ),
    make_variant(
        "candidate_strategy_memory_500k",
        "Strategy memory 500k",
        hp_family="strategy_replay_capacity",
        hp_value="strategy_memory_capacity=500000",
        description=(
            "Moderately fresher average-policy replay, reducing stale strategy samples "
            "while retaining broad historical coverage."
        ),
        strategy_memory_capacity=int(5e5),
    ),
    make_variant(
        "candidate_strategy_memory_100k",
        "Strategy memory 100k",
        hp_family="strategy_replay_capacity",
        hp_value="strategy_memory_capacity=100000",
        description=(
            "Aggressive strategy-replay freshness test, motivated by the smaller "
            "strategy reservoir used by strong random-search candidates."
        ),
        strategy_memory_capacity=int(1e5),
    ),
]


EXPERIMENT_CONFIG = make_candidate_hp_experiment_config(
    experiment_name="leduc_poker_dream_candidate_strategy_replay_capacity_ablation",
    algorithm="DREAM-style OpenSpiel candidate strategy-replay capacity ablation",
    plot_prefix="dream_candidate_strategy_replay_capacity",
    plot_title="DREAM Candidate Strategy-Replay Capacity Ablation",
    output_subdir="dream_candidate_strategy_replay_capacity_ablation",
    baseline_variant=BASELINE_VARIANT,
    variants=STRATEGY_REPLAY_CAPACITY_VARIANTS,
    treatment_keys=["strategy_memory_capacity"],
)


SMOKE_TEST_CONFIG_OVERRIDES = make_smoke_test_config_overrides(
    "dream_candidate_strategy_replay_capacity_ablation"
)
