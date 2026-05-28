# DREAM trajectories-per-iteration ablation

This experiment tests how the number of outcome-sampling trajectories per DREAM iteration affects exploitability, stability, and sample efficiency in Kuhn poker.

The default variants are:

- `traversals_80`: 80 traversals per player per iteration;
- `traversals_160_exp_baseline`: 160 traversals, matching the aligned DREAM baseline;
- `traversals_320`: 320 traversals.

All other core DREAM training settings are held fixed. Outputs include exploitability by iteration, nodes touched, and sampled trajectories so convergence can be compared both by update count and by environment-interaction cost.

## Run

```bash
python -m experiments.kuhn_poker.dream_trajectories_per_iteration_ablation.run
```

## Smoke test

```bash
python -m experiments.kuhn_poker.dream_trajectories_per_iteration_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --variants traversals_80,traversals_160_exp_baseline \
  --output-root outputs/smoke_tests/dream_trajectories_per_iteration_ablation
```

Key outputs include `checkpoint_curves_by_variant.csv`, `seed_variant_summary.csv`, `aggregate_summary_by_variant.csv`, `paired_differences_vs_baseline.csv`, `paired_difference_summary.json`, `multiseed_curves_by_variant.npz`, and plots under `plots/`.
