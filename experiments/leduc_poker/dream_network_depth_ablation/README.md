# DREAM network-depth ablation

Experiment 11 isolates hidden-layer depth while holding width at 32 units and keeping the aligned DREAM training protocol fixed.

The variants are:

- `arch_1x32`: hidden layers `[32]`;
- `arch_2x32_exp_baseline`: hidden layers `[32, 32]`, matching Experiment 1;
- `arch_3x32`: hidden layers `[32, 32, 32]`.

The same architecture is applied to the average-policy, advantage, and baseline network families. The baseline is included so matched-seed paired differences are produced.

## Run

```bash
python -m experiments.leduc_poker.dream_network_depth_ablation.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_network_depth_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --output-root outputs/smoke_tests/dream_network_depth_ablation
```
