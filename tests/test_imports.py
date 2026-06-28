
def test_import_shared_modules():
    import dream_poker.checkpointing  # noqa: F401
    import dream_poker.constants  # noqa: F401
    import dream_poker.experiment_runner  # noqa: F401
    import dream_poker.experiment_utils  # noqa: F401
    import dream_poker.lr_schedule  # noqa: F401
    import dream_poker.network_budget  # noqa: F401
    import dream_poker.networks  # noqa: F401
    import dream_poker.random_search  # noqa: F401
    import dream_poker.replay  # noqa: F401
    import dream_poker.seeding  # noqa: F401
    import dream_poker.variant_ablation  # noqa: F401
    import dream_poker.warm_start  # noqa: F401


def test_import_experiment_configs():
    import experiments.leduc_poker.dream_baseline_network_budget_ablation.config  # noqa: F401
    import experiments.leduc_poker.dream_checkpoint_stability.config  # noqa: F401
    import experiments.leduc_poker.dream_constrained_random_search.config  # noqa: F401
    import experiments.leduc_poker.dream_average_strategy_weighting_ablation.config  # noqa: F401
    import experiments.leduc_poker.dream_epsilon_exploration_ablation.config  # noqa: F401
    import experiments.leduc_poker.dream_factorised_advantage_head_ablation.config  # noqa: F401
    import experiments.leduc_poker.dream_final_only_policy_training_ablation.config  # noqa: F401
    import experiments.leduc_poker.dream_lr_schedule_ablation.config  # noqa: F401
    import experiments.leduc_poker.dream_multiseed_baseline.config  # noqa: F401
    import experiments.leduc_poker.dream_network_capacity_extremes_ablation.config  # noqa: F401
    import experiments.leduc_poker.dream_network_depth_ablation.config  # noqa: F401
    import experiments.leduc_poker.dream_network_size_ablation.config  # noqa: F401
    import experiments.leduc_poker.dream_residual_network_ablation.config  # noqa: F401
    import experiments.leduc_poker.dream_target_processing_ablation.config  # noqa: F401
    import experiments.leduc_poker.dream_trajectories_per_iteration_ablation.config  # noqa: F401
    import experiments.leduc_poker.dream_warm_start_ablation.config  # noqa: F401


def test_replay_buffer_adds_items():
    from dream_poker.replay import ReservoirBuffer

    buf = ReservoirBuffer(10)
    for i in range(20):
        buf.add(i)
    assert len(buf) == 10
    assert buf.add_calls == 20
