from experiments.leduc_poker.dream_candidate_epsilon_baseline50_long_node_comparison.config import (
    BASELINE_NETWORK_TRAIN_EVERY,
    BASELINE_VARIANT,
    EPSILON_020_VARIANT,
    EXPERIMENT_CONFIG,
    LONG_NODE_NUM_ITERATIONS,
    LONG_NODE_NUM_TRAVERSALS,
    TARGET_NODES_TOUCHED,
)


def test_epsilon_baseline50_long_node_config_extends_experiment_34_horizon():
    assert EXPERIMENT_CONFIG["num_iterations"] == LONG_NODE_NUM_ITERATIONS == 7_500
    assert EXPERIMENT_CONFIG["num_traversals"] == LONG_NODE_NUM_TRAVERSALS == 160
    assert EXPERIMENT_CONFIG["target_nodes_touched"] == TARGET_NODES_TOUCHED == 15_000_000
    assert EXPERIMENT_CONFIG["baseline_network_train_every"] == BASELINE_NETWORK_TRAIN_EVERY == 50
    assert EXPERIMENT_CONFIG["fixed_baseline"] == {"enabled": False}
    assert EXPERIMENT_CONFIG["seeds"] == [1234, 2025, 31415]


def test_epsilon_baseline50_long_node_variants_match_experiment_34_treatments():
    variants = {variant["variant_id"]: variant for variant in EXPERIMENT_CONFIG["ablation_variants"]}

    assert set(variants) == {BASELINE_VARIANT, EPSILON_020_VARIANT}
    assert variants[BASELINE_VARIANT]["epsilon"] == 0.06
    assert variants[EPSILON_020_VARIANT]["epsilon"] == 0.20
    assert variants[BASELINE_VARIANT]["baseline_network_train_every"] == 50
    assert variants[EPSILON_020_VARIANT]["baseline_network_train_every"] == 50
    assert EXPERIMENT_CONFIG["treatment_keys"] == ["epsilon", "baseline_network_train_every"]
