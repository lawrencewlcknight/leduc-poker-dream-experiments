
# Testing

This file records the basic checks used for the DREAM Kuhn poker experiments repository.

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

## Quick smoke test

The following commands run deliberately tiny two-seed DREAM experiments. They are intended to check that the runners, outputs, and plotting pipelines work; they are not scientifically meaningful.

```bash
python -m experiments.kuhn_poker.dream_multiseed_baseline.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_multiseed_baseline

python -m experiments.kuhn_poker.dream_final_only_policy_training_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_final_only_policy_training_ablation

python -m experiments.kuhn_poker.dream_checkpoint_stability.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --checkpoint-schedule 5,10 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_checkpoint_stability

python -m experiments.kuhn_poker.dream_constrained_random_search.run \
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

python -m experiments.kuhn_poker.dream_warm_start_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --warm-start-iteration 5 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_warm_start_ablation

python -m experiments.kuhn_poker.dream_lr_schedule_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_lr_schedule_ablation

python -m experiments.kuhn_poker.dream_baseline_network_budget_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --evaluation-interval 5 \
  --variants baseline_steps_50,baseline_steps_100_exp_baseline \
  --output-root outputs/smoke_tests/dream_baseline_network_budget_ablation

python -m experiments.kuhn_poker.dream_epsilon_exploration_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_epsilon_exploration_ablation

python -m experiments.kuhn_poker.dream_trajectories_per_iteration_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --variants traversals_160,traversals_320_exp_baseline \
  --output-root outputs/smoke_tests/dream_trajectories_per_iteration_ablation
```

## Full runs

```bash
python -m experiments.kuhn_poker.dream_multiseed_baseline.run
python -m experiments.kuhn_poker.dream_final_only_policy_training_ablation.run
python -m experiments.kuhn_poker.dream_checkpoint_stability.run
python -m experiments.kuhn_poker.dream_constrained_random_search.run
python -m experiments.kuhn_poker.dream_warm_start_ablation.run
python -m experiments.kuhn_poker.dream_lr_schedule_ablation.run
python -m experiments.kuhn_poker.dream_baseline_network_budget_ablation.run
python -m experiments.kuhn_poker.dream_epsilon_exploration_ablation.run
python -m experiments.kuhn_poker.dream_trajectories_per_iteration_ablation.run
```

The full runs can be computationally expensive. Use the smoke tests first after making code changes.
