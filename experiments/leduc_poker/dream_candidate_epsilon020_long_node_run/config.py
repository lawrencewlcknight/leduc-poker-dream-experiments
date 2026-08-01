"""Configuration for the epsilon-0.20 DREAM candidate long-node run."""

from dream_poker.constants import DEFAULT_SEEDS_5
from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    make_candidate_hp_experiment_config,
    make_smoke_test_config_overrides,
    make_variant,
)


CANDIDATE_EPSILON020_LONG_NODE_VARIANT = "candidate_epsilon_020_long_nodes"
TARGET_NODES_TOUCHED = 15_000_000
LONG_NODE_NUM_ITERATIONS = 7_300
LONG_NODE_NUM_TRAVERSALS = 160
LONG_NODE_EPSILON = 0.20


CANDIDATE_EPSILON020_LONG_NODE_VARIANTS = [
    make_variant(
        CANDIDATE_EPSILON020_LONG_NODE_VARIANT,
        "Experiment 22 candidate, epsilon 0.20 long-node run",
        hp_family="long_node_candidate_control",
        hp_value="epsilon=0.20; num_traversals=160; target_nodes_touched=15m",
        description=(
            "Experiment 41 long-node DREAM candidate with only the exploration "
            "rate changed to epsilon 0.20: 160 outcome-sampling traversals per "
            "player per iteration and the Experiment 22 network capacity and "
            "optimisation settings."
        ),
        epsilon=LONG_NODE_EPSILON,
        num_traversals=LONG_NODE_NUM_TRAVERSALS,
    ),
]


EXPERIMENT_CONFIG = make_candidate_hp_experiment_config(
    experiment_name="leduc_poker_dream_candidate_epsilon020_long_node_run",
    algorithm="DREAM-style OpenSpiel epsilon-0.20 candidate long-node run",
    plot_prefix="dream_candidate_epsilon020_long_node",
    plot_title="DREAM Candidate Epsilon-0.20 Long-Node Run",
    output_subdir="dream_candidate_epsilon020_long_node_run",
    baseline_variant=CANDIDATE_EPSILON020_LONG_NODE_VARIANT,
    variants=CANDIDATE_EPSILON020_LONG_NODE_VARIANTS,
    treatment_keys=["epsilon", "num_traversals"],
)
EXPERIMENT_CONFIG.update(
    {
        "num_iterations": LONG_NODE_NUM_ITERATIONS,
        "num_traversals": LONG_NODE_NUM_TRAVERSALS,
        "epsilon": LONG_NODE_EPSILON,
        "compute_baseline_grad_norm_diagnostics": False,
        "seeds": list(DEFAULT_SEEDS_5),
        "target_nodes_touched": TARGET_NODES_TOUCHED,
        "fixed_baseline": {"enabled": False},
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = make_smoke_test_config_overrides(
    "dream_candidate_epsilon020_long_node_run"
)
