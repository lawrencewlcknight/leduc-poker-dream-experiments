# DREAM residual-network ablation

This experiment tests whether residual skip connections improve optimisation
stability for deeper DREAM function approximators in Leduc poker. It compares
plain MLPs against residual MLPs at fixed width `32` and hidden depths `2`,
`4`, and `8`.

Each variant applies the same network treatment and hidden architecture to all
three DREAM network families: the average-policy network, both player-specific
advantage networks, and both player-specific learned baseline/control-variate
networks. All non-architecture parameters match the aligned DREAM baseline.

## Run

```bash
python -m experiments.leduc_poker.dream_residual_network_ablation.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_residual_network_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --evaluation-interval 1 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --variants plain_layers2_width32,residual_layers2_width32 \
  --output-root outputs/smoke_tests/dream_residual_network_ablation
```

Key outputs follow the DREAM architecture-ablation convention:
`checkpoint_curves_by_variant.csv`, `seed_variant_summary.csv`,
`aggregate_summary_by_variant.csv`, `paired_differences_vs_baseline.csv`,
`paired_difference_summary.json`, `multiseed_curves_by_variant.npz`, and plots
under `plots/`.
