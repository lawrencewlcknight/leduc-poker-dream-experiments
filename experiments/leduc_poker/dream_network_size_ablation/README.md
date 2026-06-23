# DREAM network-width ablation

Experiment 10 isolates hidden-layer width while holding depth at two layers and keeping the aligned DREAM training protocol fixed.

Each variant applies the same hidden architecture to all three DREAM network families: the average-policy network, both player-specific advantage networks, and both player-specific learned baseline/control-variate networks.

The default variants are:

- `arch_2x16`: hidden layers `[16, 16]`;
- `arch_2x32_exp_baseline`: hidden layers `[32, 32]`, matching the aligned DREAM baseline;
- `arch_2x64`: hidden layers `[64, 64]`;
- `arch_2x128`: hidden layers `[128, 128]`.

The output tables include depth, maximum width, total hidden units, and parameter counts for the constructed PyTorch networks.

The baseline is included so matched-seed paired differences are produced. Experiments 11 and 12 cover depth and capacity extremes separately.

## Run

```bash
python -m experiments.leduc_poker.dream_network_size_ablation.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_network_size_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_network_width_ablation
```

Key outputs include `checkpoint_curves_by_variant.csv`, `seed_variant_summary.csv`, `aggregate_summary_by_variant.csv`, `paired_differences_vs_baseline.csv`, `paired_difference_summary.json`, `multiseed_curves_by_variant.npz`, and plots under `plots/`.
