# DREAM role-specific capacity ablation

This experiment tests whether DREAM benefits more from concentrating extra capacity in
the advantage networks than from scaling every network family together.

The variants are:

- `all_2x32_reference`: policy, advantage, and baseline networks all use `[32, 32]`;
- `advantage_3x64_policy_baseline_2x32`: only the advantage networks use `[64, 64, 64]`;
- `advantage_2x128_policy_baseline_2x32`: only the advantage networks use `[128, 128]`;
- `all_3x64_reference`: policy, advantage, and baseline networks all use `[64, 64, 64]`.

The matched control is included so paired seed-level differences are produced.  The
`all_3x64_reference` arm tests whether role-specific capacity is better than simply
carrying forward the strongest endpoint architecture from the plain capacity sweep.

## Run

```bash
python -m experiments.leduc_poker.dream_role_specific_capacity_ablation.run
```

## Smoke test

```bash
python -m experiments.leduc_poker.dream_role_specific_capacity_ablation.run \
  --seeds 1234 \
  --iterations 3 \
  --traversals 4 \
  --policy-network-train-steps 1 \
  --advantage-network-train-steps 1 \
  --baseline-network-train-steps 1 \
  --evaluation-interval 1 \
  --variants all_2x32_reference,advantage_3x64_policy_baseline_2x32,advantage_2x128_policy_baseline_2x32,all_3x64_reference \
  --output-root outputs/smoke_tests/dream_role_specific_capacity_ablation
```
