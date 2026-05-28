# DREAM baseline-network training-budget ablation

This experiment tests whether DREAM performance in Kuhn poker is limited by the amount of supervised fitting allocated to the learned baseline/control-variate networks.

The default variants are:

- `baseline_steps_25`: 25 baseline-network updates per player per iteration;
- `baseline_steps_50_exp_baseline`: 50 updates, matching the aligned DREAM baseline;
- `baseline_steps_100`: 100 updates.

All other core DREAM training settings are held fixed.

## Run

```bash
python -m experiments.kuhn_poker.dream_baseline_network_budget_ablation.run
```

## Smoke test

```bash
python -m experiments.kuhn_poker.dream_baseline_network_budget_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --evaluation-interval 5 \
  --variants baseline_steps_25,baseline_steps_50_exp_baseline \
  --output-root outputs/smoke_tests/dream_baseline_network_budget_ablation
```

Key outputs include `checkpoint_curves_by_variant.csv`, `seed_variant_summary.csv`, `aggregate_summary_by_variant.csv`, `paired_differences_vs_baseline.csv`, `paired_difference_summary.json`, `multiseed_curves_by_variant.npz`, and plots under `plots/`.
