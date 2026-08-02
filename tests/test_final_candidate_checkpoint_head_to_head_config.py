"""Configuration checks for Experiment 43 DREAM checkpoint head-to-head."""

from experiments.leduc_poker.dream_final_candidate_checkpoint_head_to_head.config import (
    BASELINE_NETWORK_TRAIN_EVERY,
    CHECKPOINT_SCHEDULE,
    DEFAULT_CONFIG,
    TARGET_NODES_TOUCHED,
    TARGET_NUM_ITERATIONS,
    TARGET_NUM_TRAVERSALS,
)


def test_final_candidate_checkpoint_schedule_matches_long_node_horizon():
    assert DEFAULT_CONFIG["num_iterations"] == TARGET_NUM_ITERATIONS == 7_500
    assert DEFAULT_CONFIG["num_traversals"] == TARGET_NUM_TRAVERSALS == 160
    assert DEFAULT_CONFIG["target_nodes_touched"] == TARGET_NODES_TOUCHED == 15_000_000
    assert DEFAULT_CONFIG["checkpoint_schedule"] == CHECKPOINT_SCHEDULE
    assert CHECKPOINT_SCHEDULE == (1_500, 3_000, 4_500, 6_000, 7_500)
    assert CHECKPOINT_SCHEDULE[-1] == DEFAULT_CONFIG["num_iterations"]


def test_final_candidate_config_matches_experiment_38_selected_arm():
    assert DEFAULT_CONFIG["epsilon"] == 0.20
    assert DEFAULT_CONFIG["baseline_network_train_every"] == BASELINE_NETWORK_TRAIN_EVERY == 50
    assert DEFAULT_CONFIG["policy_network_layers"] == [32, 32]
    assert DEFAULT_CONFIG["baseline_network_layers"] == [32, 32]
    assert DEFAULT_CONFIG["advantage_network_layers"] == [128, 128]
    assert DEFAULT_CONFIG["policy_network_train_every"] == 25
    assert DEFAULT_CONFIG["policy_network_train_steps"] == 100
    assert DEFAULT_CONFIG["advantage_network_train_steps"] == 50
    assert DEFAULT_CONFIG["baseline_network_train_steps"] == 50
    assert DEFAULT_CONFIG["average_strategy_weighting"] == "linear"


def test_checkpoints_are_fresh_average_policy_fits():
    train_every = DEFAULT_CONFIG["policy_network_train_every"]
    assert all(checkpoint % train_every == 0 for checkpoint in CHECKPOINT_SCHEDULE)
