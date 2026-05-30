
# DREAM-style multi-seed baseline

This experiment trains a DREAM-style OpenSpiel implementation on Leduc poker using a fixed multi-seed protocol. It is intended to sit alongside the Deep CFR and ESCHER baseline experiments in the thesis.

The primary metric is exploitability, reported as NashConv divided by two. Secondary metrics include policy-value error from the known Leduc game value, nodes touched, wall-clock time, and final/best/final-window exploitability. Diagnostics include policy loss, advantage loss, learned-baseline loss, replay-buffer sizes, target variance, policy entropy, and gradient norms.

## Run

From the repository root:

```bash
python -m experiments.leduc_poker.dream_multiseed_baseline.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_multiseed_baseline.run       --seeds 1234,2025       --iterations 10       --traversals 50       --policy-network-train-steps 20       --advantage-network-train-steps 20       --baseline-network-train-steps 20       --evaluation-interval 5       --output-root outputs/smoke_tests/dream_multiseed_baseline
```
