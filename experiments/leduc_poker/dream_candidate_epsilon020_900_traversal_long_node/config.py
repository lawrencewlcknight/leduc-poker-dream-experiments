"""Configuration for the epsilon-0.20 900-traversal long-node DREAM run."""

from dream_poker.constants import DEFAULT_SEEDS_5
from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    make_candidate_hp_experiment_config,
    make_smoke_test_config_overrides,
    make_variant,
)


CANDIDATE_EPSILON020_900_VARIANT = "candidate_epsilon_020_traversals_900_long_nodes"
CANDIDATE_EPSILON020_900_BASELINE50_VARIANT = (
    "candidate_epsilon_020_traversals_900_baseline_every_50_long_nodes"
)
TARGET_NODES_TOUCHED = 15_000_000
LONG_NODE_NUM_TRAVERSALS = 900
LONG_NODE_NUM_ITERATIONS = 1_300
LONG_NODE_EPSILON = 0.20


CANDIDATE_EPSILON020_900_VARIANTS = [
    make_variant(
        CANDIDATE_EPSILON020_900_VARIANT,
        "epsilon 0.20, 900 traversals, baseline every 1",
        hp_family="long_node_epsilon_traversal_budget",
        hp_value=(
            "epsilon=0.20; num_traversals=900; "
            "baseline_network_train_every=1; target_nodes_touched=15m"
        ),
        description=(
            "Experiment 22 architecture-selected DREAM candidate trained with "
            "epsilon 0.20 and 900 outcome-sampling traversals per player per "
            "iteration, with learned-baseline training on every DREAM iteration."
        ),
        epsilon=LONG_NODE_EPSILON,
        num_traversals=LONG_NODE_NUM_TRAVERSALS,
        baseline_network_train_every=1,
    ),
    make_variant(
        CANDIDATE_EPSILON020_900_BASELINE50_VARIANT,
        "epsilon 0.20, 900 traversals, baseline every 50",
        hp_family="long_node_epsilon_traversal_budget",
        hp_value=(
            "epsilon=0.20; num_traversals=900; "
            "baseline_network_train_every=50; target_nodes_touched=15m"
        ),
        description=(
            "Experiment 22 architecture-selected DREAM candidate trained with "
            "epsilon 0.20 and 900 outcome-sampling traversals per player per "
            "iteration, with sparse learned-baseline training every 50 DREAM "
            "iterations."
        ),
        epsilon=LONG_NODE_EPSILON,
        num_traversals=LONG_NODE_NUM_TRAVERSALS,
        baseline_network_train_every=50,
    ),
]


EXPERIMENT_CONFIG = make_candidate_hp_experiment_config(
    experiment_name="leduc_poker_dream_candidate_epsilon020_900_traversal_long_node",
    algorithm="DREAM-style OpenSpiel epsilon-0.20 900-traversal long-node run",
    plot_prefix="dream_candidate_epsilon020_900_traversal_long_node",
    plot_title="DREAM Candidate Epsilon-0.20 900-Traversal Long-Node Run",
    output_subdir="dream_candidate_epsilon020_900_traversal_long_node",
    baseline_variant=CANDIDATE_EPSILON020_900_VARIANT,
    variants=CANDIDATE_EPSILON020_900_VARIANTS,
    treatment_keys=["epsilon", "num_traversals", "baseline_network_train_every"],
)
EXPERIMENT_CONFIG.update(
    {
        "num_iterations": LONG_NODE_NUM_ITERATIONS,
        "num_traversals": LONG_NODE_NUM_TRAVERSALS,
        "epsilon": LONG_NODE_EPSILON,
        "baseline_network_train_every": 1,
        "compute_baseline_grad_norm_diagnostics": False,
        "seeds": list(DEFAULT_SEEDS_5),
        "target_nodes_touched": TARGET_NODES_TOUCHED,
        "fixed_baseline": {"enabled": False},
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = make_smoke_test_config_overrides(
    "dream_candidate_epsilon020_900_traversal_long_node"
)
