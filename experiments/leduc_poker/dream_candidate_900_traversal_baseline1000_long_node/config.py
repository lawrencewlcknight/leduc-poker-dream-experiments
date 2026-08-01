"""Configuration for the 900-traversal, baseline-1000 long-node DREAM run."""

from experiments.leduc_poker.dream_candidate_hp_ablation_common import (
    DEFAULT_SEEDS,
    make_candidate_hp_experiment_config,
    make_smoke_test_config_overrides,
    make_variant,
)


CANDIDATE_900_BASELINE1000_VARIANT = "candidate_traversals_900_baseline_steps_1000_long_nodes"
TARGET_NODES_TOUCHED = 15_000_000
LONG_NODE_NUM_TRAVERSALS = 900
LONG_NODE_NUM_ITERATIONS = 1_300
BASELINE_NETWORK_TRAIN_STEPS = 1_000
BASELINE_NETWORK_TRAIN_EVERY = 1


CANDIDATE_900_BASELINE1000_VARIANTS = [
    make_variant(
        CANDIDATE_900_BASELINE1000_VARIANT,
        "900 traversals, 1000 baseline minibatches",
        hp_family="paper_style_baseline_budget",
        hp_value=(
            "num_traversals=900; baseline_network_train_steps=1000; "
            "baseline_network_train_every=1; target_nodes_touched=15m"
        ),
        description=(
            "Experiment 22 architecture-selected DREAM candidate trained with "
            "900 outcome-sampling traversals per player per iteration and 1000 "
            "learned-baseline minibatches per player per iteration, matching the "
            "large baseline-fitting budget used in the DREAM paper."
        ),
        num_traversals=LONG_NODE_NUM_TRAVERSALS,
        baseline_network_train_steps=BASELINE_NETWORK_TRAIN_STEPS,
        baseline_network_train_every=BASELINE_NETWORK_TRAIN_EVERY,
    ),
]


EXPERIMENT_CONFIG = make_candidate_hp_experiment_config(
    experiment_name="leduc_poker_dream_candidate_900_traversal_baseline1000_long_node",
    algorithm="DREAM-style OpenSpiel 900-traversal baseline-1000 long-node run",
    plot_prefix="dream_candidate_900_traversal_baseline1000_long_node",
    plot_title="DREAM Candidate 900-Traversal Baseline-1000 Long-Node Run",
    output_subdir="dream_candidate_900_traversal_baseline1000_long_node",
    baseline_variant=CANDIDATE_900_BASELINE1000_VARIANT,
    variants=CANDIDATE_900_BASELINE1000_VARIANTS,
    treatment_keys=["num_traversals", "baseline_network_train_steps", "baseline_network_train_every"],
)
EXPERIMENT_CONFIG.update(
    {
        "num_iterations": LONG_NODE_NUM_ITERATIONS,
        "num_traversals": LONG_NODE_NUM_TRAVERSALS,
        "baseline_network_train_steps": BASELINE_NETWORK_TRAIN_STEPS,
        "baseline_network_train_every": BASELINE_NETWORK_TRAIN_EVERY,
        "seeds": list(DEFAULT_SEEDS),
        "target_nodes_touched": TARGET_NODES_TOUCHED,
        "fixed_baseline": {"enabled": False},
    }
)


SMOKE_TEST_CONFIG_OVERRIDES = make_smoke_test_config_overrides(
    "dream_candidate_900_traversal_baseline1000_long_node"
)
