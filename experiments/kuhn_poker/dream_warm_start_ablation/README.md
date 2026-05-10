# DREAM fair warm-start ablation

This experiment compares a continuous DREAM run with a checkpoint/resume warm-start arm under the same total budget. The warm-start arm trains to an intermediate iteration, saves the full solver state, constructs a new solver, restores the saved state, and continues to the same final iteration.

The purpose is checkpoint/resume fidelity, not giving the warm-start arm extra data or optimisation.

## Run

```bash
python -m experiments.kuhn_poker.dream_warm_start_ablation.run
```

## Smoke test

```bash
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
```

Key outputs include `checkpoint_curves_by_arm.csv`, `seed_arm_summary.csv`, `aggregate_summary_by_arm.csv`, `paired_differences_warm_minus_baseline.csv`, `paired_difference_summary.json`, per-seed saved checkpoints, and thesis-style plots under `plots/`.
