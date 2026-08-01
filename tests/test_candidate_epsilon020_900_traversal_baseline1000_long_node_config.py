from experiments.leduc_poker.dream_candidate_epsilon020_900_traversal_baseline1000_long_node.config import (
    BASELINE_NETWORK_TRAIN_EVERY,
    BASELINE_NETWORK_TRAIN_STEPS,
    CANDIDATE_EPSILON020_900_BASELINE1000_VARIANT,
    EXPERIMENT_CONFIG,
    LONG_NODE_EPSILON,
    LONG_NODE_NUM_ITERATIONS,
    LONG_NODE_NUM_TRAVERSALS,
    TARGET_NODES_TOUCHED,
)


def test_epsilon020_900_traversal_baseline1000_long_node_config_matches_request():
    assert EXPERIMENT_CONFIG["num_iterations"] == LONG_NODE_NUM_ITERATIONS == 1_300
    assert EXPERIMENT_CONFIG["num_traversals"] == LONG_NODE_NUM_TRAVERSALS == 900
    assert EXPERIMENT_CONFIG["epsilon"] == LONG_NODE_EPSILON == 0.20
    assert EXPERIMENT_CONFIG["baseline_network_train_steps"] == BASELINE_NETWORK_TRAIN_STEPS == 1_000
    assert EXPERIMENT_CONFIG["baseline_network_train_every"] == BASELINE_NETWORK_TRAIN_EVERY == 1
    assert EXPERIMENT_CONFIG["target_nodes_touched"] == TARGET_NODES_TOUCHED == 15_000_000
    assert EXPERIMENT_CONFIG["seeds"] == [1234, 2025, 31415]
    assert EXPERIMENT_CONFIG["fixed_baseline"] == {"enabled": False}


def test_epsilon020_900_traversal_baseline1000_long_node_variant_is_single_arm():
    variants = EXPERIMENT_CONFIG["ablation_variants"]

    assert len(variants) == 1
    assert variants[0]["variant_id"] == CANDIDATE_EPSILON020_900_BASELINE1000_VARIANT
    assert variants[0]["epsilon"] == 0.20
    assert variants[0]["num_traversals"] == 900
    assert variants[0]["baseline_network_train_steps"] == 1_000
    assert variants[0]["baseline_network_train_every"] == 1
    assert EXPERIMENT_CONFIG["baseline_variant"] == CANDIDATE_EPSILON020_900_BASELINE1000_VARIANT
