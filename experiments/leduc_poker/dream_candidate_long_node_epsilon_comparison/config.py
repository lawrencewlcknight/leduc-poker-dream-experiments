"""Configuration for the long-node DREAM candidate epsilon comparison."""

from dream_poker.constants import DEFAULT_SEEDS_5
from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    make_candidate_hp_experiment_config,
    make_smoke_test_config_overrides,
    make_variant,
)


BASELINE_VARIANT = "long_nodes_candidate_epsilon_006"
EPSILON_020_VARIANT = "long_nodes_candidate_epsilon_020"
TARGET_NODES_TOUCHED = 15_000_000
LONG_NODE_NUM_ITERATIONS = 7_300


LONG_NODE_EPSILON_VARIANTS = [
    make_variant(
        BASELINE_VARIANT,
        "epsilon 0.06",
        hp_family="long_node_epsilon",
        hp_value="epsilon=0.06",
        description=(
            "Architecture-selected DREAM candidate before the candidate-parameter "
            "ablations: 160 traversals, epsilon 0.06, and Experiment 22 network "
            "capacity."
        ),
        epsilon=0.06,
    ),
    make_variant(
        EPSILON_020_VARIANT,
        "epsilon 0.20",
        hp_family="long_node_epsilon",
        hp_value="epsilon=0.20",
        description=(
            "Long-node treatment that changes only the traversal exploration rate "
            "to the strongest value from the candidate epsilon ablation."
        ),
        epsilon=0.20,
    ),
]


EXPERIMENT_CONFIG = make_candidate_hp_experiment_config(
    experiment_name="leduc_poker_dream_candidate_long_node_epsilon_comparison",
    algorithm="DREAM-style OpenSpiel long-node candidate epsilon comparison",
    plot_prefix="dream_candidate_long_node_epsilon",
    plot_title="DREAM Candidate Long-Node Epsilon Comparison",
    output_subdir="dream_candidate_long_node_epsilon_comparison",
    baseline_variant=BASELINE_VARIANT,
    variants=LONG_NODE_EPSILON_VARIANTS,
    treatment_keys=["epsilon"],
)
EXPERIMENT_CONFIG.update(
    {
        "num_iterations": LONG_NODE_NUM_ITERATIONS,
        "num_traversals": 160,
        "seeds": list(DEFAULT_SEEDS_5),
        "target_nodes_touched": TARGET_NODES_TOUCHED,
        "fixed_baseline": {"enabled": False},
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = make_smoke_test_config_overrides(
    "dream_candidate_long_node_epsilon_comparison"
)
