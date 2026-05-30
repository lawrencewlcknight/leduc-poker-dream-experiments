# DREAM learning-rate schedule ablation

This experiment tests whether changing only the optimiser learning-rate schedule improves the DREAM-style OpenSpiel baseline on Leduc poker.

The default variants are:

- `constant_baseline_dream`: the aligned DREAM baseline with constant learning rate `0.003`;
- `cosine_decay_to_10pct`: cosine decay from `0.003` to `0.0003` over the training run.

All other core training settings are held fixed.

## Run

```bash
python -m experiments.leduc_poker.dream_lr_schedule_ablation.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_lr_schedule_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_lr_schedule_ablation
```

Key outputs include `checkpoint_curves_by_variant.csv`, `seed_variant_summary.csv`, `aggregate_summary_by_variant.csv`, `paired_differences_vs_baseline.csv`, `paired_difference_summary.json`, `multiseed_curves_by_variant.npz`, and plots under `plots/`.
