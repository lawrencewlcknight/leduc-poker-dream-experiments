
# Testing

This file records the basic checks used for the DREAM Leduc poker experiments repository.

## Environment setup

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

OpenSpiel installation can vary by platform. Follow the official OpenSpiel installation instructions if the wheel is unavailable for your system.

## Syntax/import checks

```bash
python -m compileall dream_poker experiments tests
pytest -q
```

## Local smoke tests

The following commands run deliberately tiny two-seed DREAM experiments. They are intended to check that the runners, outputs, and plotting pipelines work; they are not scientifically meaningful.

Run them from the activated environment created above. If `python` reports
`ModuleNotFoundError: No module named 'matplotlib'`, the selected interpreter
does not have `requirements.txt` installed.

For smoke tests that run on GCP instead of the local Python environment, use the
Batch smoke-test commands in `README.md` or `docs/GCP_BATCH_EXPERIMENTS.md`.
The recent DREAM ablations can be submitted together with:

```bash
./gcp/submit_recent_ablation_smoke_tests.sh
```

To submit only the latest two DREAM ablations, use:

```bash
./gcp/submit_latest_ablation_smoke_tests.sh
```

```bash
python -m experiments.leduc_poker.dream_multiseed_baseline.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_multiseed_baseline

python -m experiments.leduc_poker.dream_final_only_policy_training_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_final_only_policy_training_ablation

python -m experiments.leduc_poker.dream_checkpoint_stability.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --checkpoint-schedule 5,10 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_checkpoint_stability

python -m experiments.leduc_poker.dream_constrained_random_search.run \
  --screening-seeds 1234 \
  --confirmation-seeds 1234 \
  --screening-iterations 5 \
  --confirmation-iterations 5 \
  --n-random-candidates 1 \
  --n-confirmation-candidates 1 \
  --num-traversals 20 \
  --policy-network-train-steps 5 \
  --advantage-network-train-steps 5 \
  --baseline-network-train-steps 5 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_constrained_random_search

python -m experiments.leduc_poker.dream_warm_start_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --warm-start-iteration 5 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_warm_start_ablation

python -m experiments.leduc_poker.dream_lr_schedule_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_lr_schedule_ablation

python -m experiments.leduc_poker.dream_baseline_network_budget_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --evaluation-interval 5 \
  --variants baseline_steps_50,baseline_steps_100_exp_baseline \
  --output-root outputs/smoke_tests/dream_baseline_network_budget_ablation

python -m experiments.leduc_poker.dream_epsilon_exploration_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_epsilon_exploration_ablation

python -m experiments.leduc_poker.dream_trajectories_per_iteration_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --variants traversals_160,traversals_320_exp_baseline \
  --output-root outputs/smoke_tests/dream_trajectories_per_iteration_ablation

python -m experiments.leduc_poker.dream_target_processing_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --batch-size-advantage 1 \
  --batch-size-strategy 1 \
  --batch-size-baseline 1 \
  --evaluation-interval 5 \
  --variants raw_targets_dream_baseline,standardized_clipped_targets \
  --output-root outputs/smoke_tests/dream_target_processing_ablation

python -m experiments.leduc_poker.dream_network_size_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_network_width_ablation

python -m experiments.leduc_poker.dream_network_depth_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_network_depth_ablation

python -m experiments.leduc_poker.dream_network_capacity_extremes_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_network_capacity_extremes_ablation

python -m experiments.leduc_poker.dream_residual_network_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants plain_layers2_width32,residual_layers2_width32 \
  --output-root outputs/smoke_tests/dream_residual_network_ablation

python -m experiments.leduc_poker.dream_average_strategy_weighting_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --batch-size-advantage 1 \
  --batch-size-strategy 1 \
  --batch-size-baseline 1 \
  --evaluation-interval 1 \
  --output-root outputs/smoke_tests/dream_average_strategy_weighting_ablation

python -m experiments.leduc_poker.dream_factorised_advantage_head_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants direct_advantage_layers2_width32,centered_advantage_layers2_width32,dueling_advantage_layers2_width32 \
  --output-root outputs/smoke_tests/dream_factorised_advantage_head_ablation

python -m experiments.leduc_poker.dream_layer_norm_network_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants plain_layers2_width32,layer_norm_layers2_width32,residual_layer_norm_layers2_width32 \
  --output-root outputs/smoke_tests/dream_layer_norm_network_ablation

# Leduc Experiment 20 — role-specific capacity ablation smoke test
python -m experiments.leduc_poker.dream_role_specific_capacity_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants all_2x32_reference,advantage_3x64_policy_baseline_2x32,advantage_2x128_policy_baseline_2x32,all_3x64_reference \
  --output-root outputs/smoke_tests/dream_role_specific_capacity_ablation

# Leduc Experiment 21 — sequential/parallel equivalence smoke test
python -m experiments.leduc_poker.dream_parallel_equivalence_ablation.run \
  --seeds 1234 \
  --variant-ids dream_3x64_sequential,dream_3x64_ray_parallel \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --batch-size-advantage 1 \
  --batch-size-strategy 1 \
  --batch-size-baseline 1 \
  --output-root outputs/smoke_tests/dream_parallel_equivalence_ablation

# Leduc Experiment 22 — architecture-candidate comparison smoke test
python -m experiments.leduc_poker.dream_architecture_candidate_comparison.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants baseline_all_2x32,candidate_advantage_2x128_policy_baseline_2x32 \
  --output-root outputs/smoke_tests/dream_architecture_candidate_comparison

# Leduc Experiment 23 — candidate-architecture epsilon smoke test
python -m experiments.leduc_poker.dream_candidate_epsilon_exploration_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_epsilon_006_baseline,candidate_epsilon_010 \
  --output-root outputs/smoke_tests/dream_candidate_epsilon_exploration_ablation

# Leduc Experiment 24 — candidate traversal-budget smoke test
python -m experiments.leduc_poker.dream_candidate_traversal_budget_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_traversals_160_baseline,candidate_traversals_320 \
  --output-root outputs/smoke_tests/dream_candidate_traversal_budget_ablation

# Leduc Experiment 25 — candidate strategy replay-capacity smoke test
python -m experiments.leduc_poker.dream_candidate_strategy_replay_capacity_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_strategy_memory_1m_baseline,candidate_strategy_memory_100k \
  --output-root outputs/smoke_tests/dream_candidate_strategy_replay_capacity_ablation

# Leduc Experiment 26 — candidate learned-baseline replay-capacity smoke test
python -m experiments.leduc_poker.dream_candidate_baseline_replay_capacity_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_baseline_memory_1m_baseline,candidate_baseline_memory_100k \
  --output-root outputs/smoke_tests/dream_candidate_baseline_replay_capacity_ablation

# Leduc Experiment 27 — candidate policy-extraction budget smoke test
python -m experiments.leduc_poker.dream_candidate_policy_extraction_budget_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_policy_steps_100_baseline,candidate_policy_steps_200 \
  --output-root outputs/smoke_tests/dream_candidate_policy_extraction_budget_ablation

# Leduc Experiment 28 — candidate policy-extraction cadence smoke test
python -m experiments.leduc_poker.dream_candidate_policy_extraction_cadence_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --variants candidate_policy_every_25_baseline,candidate_policy_every_10 \
  --output-root outputs/smoke_tests/dream_candidate_policy_extraction_cadence_ablation

# Leduc Experiment 29 — candidate advantage-fitting steps smoke test
python -m experiments.leduc_poker.dream_candidate_advantage_fitting_steps_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_advantage_steps_50_baseline,candidate_advantage_steps_25 \
  --output-root outputs/smoke_tests/dream_candidate_advantage_fitting_steps_ablation

# Leduc Experiment 30 — candidate advantage batch-size smoke test
python -m experiments.leduc_poker.dream_candidate_advantage_batch_size_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_advantage_batch_1024_baseline,candidate_advantage_batch_512 \
  --output-root outputs/smoke_tests/dream_candidate_advantage_batch_size_ablation

# Leduc Experiment 31 — candidate constant learning-rate smoke test
python -m experiments.leduc_poker.dream_candidate_constant_learning_rate_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants candidate_learning_rate_0_003_baseline,candidate_learning_rate_0_006 \
  --output-root outputs/smoke_tests/dream_candidate_constant_learning_rate_ablation
```

## Full runs

```bash
python -m experiments.leduc_poker.dream_multiseed_baseline.run
python -m experiments.leduc_poker.dream_final_only_policy_training_ablation.run
python -m experiments.leduc_poker.dream_checkpoint_stability.run
python -m experiments.leduc_poker.dream_constrained_random_search.run
python -m experiments.leduc_poker.dream_warm_start_ablation.run
python -m experiments.leduc_poker.dream_lr_schedule_ablation.run
python -m experiments.leduc_poker.dream_baseline_network_budget_ablation.run
python -m experiments.leduc_poker.dream_epsilon_exploration_ablation.run
python -m experiments.leduc_poker.dream_trajectories_per_iteration_ablation.run
python -m experiments.leduc_poker.dream_network_size_ablation.run
python -m experiments.leduc_poker.dream_network_depth_ablation.run
python -m experiments.leduc_poker.dream_network_capacity_extremes_ablation.run
python -m experiments.leduc_poker.dream_target_processing_ablation.run
python -m experiments.leduc_poker.dream_residual_network_ablation.run
python -m experiments.leduc_poker.dream_average_strategy_weighting_ablation.run
python -m experiments.leduc_poker.dream_factorised_advantage_head_ablation.run
python -m experiments.leduc_poker.dream_layer_norm_network_ablation.run
python -m experiments.leduc_poker.dream_role_specific_capacity_ablation.run
python -m experiments.leduc_poker.dream_parallel_equivalence_ablation.run
python -m experiments.leduc_poker.dream_architecture_candidate_comparison.run
python -m experiments.leduc_poker.dream_candidate_epsilon_exploration_ablation.run
python -m experiments.leduc_poker.dream_candidate_traversal_budget_ablation.run
python -m experiments.leduc_poker.dream_candidate_strategy_replay_capacity_ablation.run
python -m experiments.leduc_poker.dream_candidate_baseline_replay_capacity_ablation.run
python -m experiments.leduc_poker.dream_candidate_policy_extraction_budget_ablation.run
python -m experiments.leduc_poker.dream_candidate_policy_extraction_cadence_ablation.run
python -m experiments.leduc_poker.dream_candidate_advantage_fitting_steps_ablation.run
python -m experiments.leduc_poker.dream_candidate_advantage_batch_size_ablation.run
python -m experiments.leduc_poker.dream_candidate_constant_learning_rate_ablation.run
```

The full runs can be computationally expensive. Use the smoke tests first after making code changes.
