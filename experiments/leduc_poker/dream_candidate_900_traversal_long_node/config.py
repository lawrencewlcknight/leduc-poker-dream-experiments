"""Configuration for the 900-traversal long-node DREAM candidate run."""

from dream_poker.constants import DEFAULT_SEEDS_5
from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    make_candidate_hp_experiment_config,
    make_smoke_test_config_overrides,
    make_variant,
)


CANDIDATE_900_VARIANT = "candidate_traversals_900_long_nodes"
TARGET_NODES_TOUCHED = 15_000_000
LONG_NODE_NUM_TRAVERSALS = 900
LONG_NODE_NUM_ITERATIONS = 1_300


CANDIDATE_900_VARIANTS = [
    make_variant(
        CANDIDATE_900_VARIANT,
        "900 traversals",
        hp_family="long_node_traversal_budget",
        hp_value="num_traversals=900; target_nodes_touched=15m",
        description=(
            "Experiment 22 architecture-selected DREAM candidate trained with "
            "900 outcome-sampling traversals per player per iteration. The "
            "iteration count is set to keep the total node budget close to the "
            "15m-node long-run target."
        ),
        num_traversals=LONG_NODE_NUM_TRAVERSALS,
    ),
]


EXPERIMENT_CONFIG = make_candidate_hp_experiment_config(
    experiment_name="leduc_poker_dream_candidate_900_traversal_long_node",
    algorithm="DREAM-style OpenSpiel 900-traversal long-node candidate run",
    plot_prefix="dream_candidate_900_traversal_long_node",
    plot_title="DREAM Candidate 900-Traversal Long-Node Run",
    output_subdir="dream_candidate_900_traversal_long_node",
    baseline_variant=CANDIDATE_900_VARIANT,
    variants=CANDIDATE_900_VARIANTS,
    treatment_keys=["num_traversals"],
)
EXPERIMENT_CONFIG.update(
    {
        "num_iterations": LONG_NODE_NUM_ITERATIONS,
        "num_traversals": LONG_NODE_NUM_TRAVERSALS,
        "seeds": list(DEFAULT_SEEDS_5),
        "target_nodes_touched": TARGET_NODES_TOUCHED,
        "fixed_baseline": {"enabled": False},
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = make_smoke_test_config_overrides(
    "dream_candidate_900_traversal_long_node"
)
