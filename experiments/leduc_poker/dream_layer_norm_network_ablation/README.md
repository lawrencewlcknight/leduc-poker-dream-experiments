# DREAM layer-normalisation network ablation

This experiment tests whether normalising hidden activations improves DREAM
network optimisation in Leduc poker. It is the DREAM analogue of the Deep CFR
layer-normalisation network ablation.

The grid compares:

- plain MLPs
- MLPs with `LayerNorm` after each hidden activation
- residual MLPs with `LayerNorm` after each residual hidden block

Each treatment is run at hidden depths `2`, `4`, and `8`, all at width `32`.
The same treatment is applied to the average-policy, advantage, and
learned-baseline networks within each variant. All non-architecture settings
match the aligned DREAM architecture-ablation baseline.

## Run

```bash
python -m experiments.leduc_poker.dream_layer_norm_network_ablation.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_layer_norm_network_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --variants plain_layers2_width32,layer_norm_layers2_width32,residual_layer_norm_layers2_width32 \
  --output-root outputs/smoke_tests/dream_layer_norm_network_ablation
```

Key outputs follow the DREAM architecture-ablation convention:
`checkpoint_curves_by_variant.csv`, `seed_variant_summary.csv`,
`aggregate_summary_by_variant.csv`, `paired_differences_vs_baseline.csv`,
`paired_difference_summary.json`, `multiseed_curves_by_variant.npz`, and plots
under `plots/`.
