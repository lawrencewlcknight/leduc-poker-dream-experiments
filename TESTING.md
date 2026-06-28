
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
```

The full runs can be computationally expensive. Use the smoke tests first after making code changes.
