"""Configuration for the long-node epsilon comparison with baseline cadence 50."""

from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    DEFAULT_SEEDS,
    make_candidate_hp_experiment_config,
    make_smoke_test_config_overrides,
    make_variant,
)


BASELINE_VARIANT = "long_nodes_candidate_epsilon_006_baseline_every_50"
EPSILON_020_VARIANT = "long_nodes_candidate_epsilon_020_baseline_every_50"
TARGET_NODES_TOUCHED = 15_000_000
LONG_NODE_NUM_ITERATIONS = 7_500
LONG_NODE_NUM_TRAVERSALS = 160
BASELINE_NETWORK_TRAIN_EVERY = 50


EPSILON_BASELINE50_LONG_NODE_VARIANTS = [
    make_variant(
        BASELINE_VARIANT,
        "epsilon 0.06, baseline every 50",
        hp_family="long_node_epsilon_baseline50",
        hp_value="epsilon=0.06; baseline_network_train_every=50; target_nodes_touched=15m",
        description=(
            "Experiment 34 inherited-exploration arm extended to the long-node "
            "horizon with sparse learned-baseline training every 50 DREAM "
            "iterations."
        ),
        epsilon=0.06,
        baseline_network_train_every=BASELINE_NETWORK_TRAIN_EVERY,
    ),
    make_variant(
        EPSILON_020_VARIANT,
        "epsilon 0.20, baseline every 50",
        hp_family="long_node_epsilon_baseline50",
        hp_value="epsilon=0.20; baseline_network_train_every=50; target_nodes_touched=15m",
        description=(
            "Experiment 34 high-exploration arm extended to the long-node "
            "horizon with sparse learned-baseline training every 50 DREAM "
            "iterations."
        ),
        epsilon=0.20,
        baseline_network_train_every=BASELINE_NETWORK_TRAIN_EVERY,
    ),
]


EXPERIMENT_CONFIG = make_candidate_hp_experiment_config(
    experiment_name="leduc_poker_dream_candidate_epsilon_baseline50_long_node_comparison",
    algorithm="DREAM-style OpenSpiel long-node epsilon comparison with baseline cadence 50",
    plot_prefix="dream_candidate_epsilon_baseline50_long_node",
    plot_title="DREAM Candidate Epsilon Comparison with Baseline Cadence 50 at Long-Node Horizon",
    output_subdir="dream_candidate_epsilon_baseline50_long_node_comparison",
    baseline_variant=BASELINE_VARIANT,
    variants=EPSILON_BASELINE50_LONG_NODE_VARIANTS,
    treatment_keys=["epsilon", "baseline_network_train_every"],
)
EXPERIMENT_CONFIG.update(
    {
        "num_iterations": LONG_NODE_NUM_ITERATIONS,
        "num_traversals": LONG_NODE_NUM_TRAVERSALS,
        "seeds": list(DEFAULT_SEEDS),
        "baseline_network_train_every": BASELINE_NETWORK_TRAIN_EVERY,
        "target_nodes_touched": TARGET_NODES_TOUCHED,
        "fixed_baseline": {"enabled": False},
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = make_smoke_test_config_overrides(
    "dream_candidate_epsilon_baseline50_long_node_comparison"
)
