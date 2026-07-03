# DREAM plain-network depth reference ablation

This experiment is the plain-MLP reference split from the original DREAM
layer-normalisation grid. It compares plain DREAM networks at hidden depths
`2`, `4`, and `8`, all at width `32`, with the same architecture applied to
the average-policy, advantage, and learned-baseline networks.

The experiment establishes the depth-only control condition for the subsequent
LayerNorm and residual-LayerNorm experiments. The paired-difference baseline is
`plain_layers2_width32`.

## Run

```bash
python -m experiments.leduc_poker.dream_plain_network_depth_ablation.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_plain_network_depth_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --variants plain_layers2_width32,plain_layers4_width32 \
  --output-root outputs/smoke_tests/dream_plain_network_depth_ablation
```

Key outputs follow the DREAM architecture-ablation convention:
`checkpoint_curves_by_variant.csv`, `seed_variant_summary.csv`,
`aggregate_summary_by_variant.csv`, `paired_differences_vs_baseline.csv`,
`paired_difference_summary.json`, `multiseed_curves_by_variant.npz`, and plots
under `plots/`.
