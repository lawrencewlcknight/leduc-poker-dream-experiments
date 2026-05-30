# DREAM epsilon-exploration ablation

This experiment tests the effect of DREAM's epsilon-mixed sampling policy during outcome-sampling traversals in Leduc poker.

The default variants are:

- `epsilon_003`: epsilon `0.03`, lower exploration than baseline;
- `epsilon_006_exp_baseline`: epsilon `0.06`, matching the aligned DREAM baseline;
- `epsilon_010`: epsilon `0.10`, higher exploration than baseline.

All other core DREAM training settings are held fixed.

## Run

```bash
python -m experiments.leduc_poker.dream_epsilon_exploration_ablation.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_epsilon_exploration_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_epsilon_exploration_ablation
```

Key outputs include `checkpoint_curves_by_variant.csv`, `seed_variant_summary.csv`, `aggregate_summary_by_variant.csv`, `paired_differences_vs_baseline.csv`, `paired_difference_summary.json`, `multiseed_curves_by_variant.npz`, and plots under `plots/`.
